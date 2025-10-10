import type { LanguageModel, Tool } from "ai";
import { Experimental_Agent, generateText, tool } from "ai";
import { z } from "zod";
import {
  DataFlowTracker,
  Policy,
  PolicyViolationError,
  type ToolCallContext,
} from "./data-flow.ts";

export interface ToolWithPolicies {
  readonly tool: Tool;
  readonly policies?: readonly Policy[];
}

export type ToolConfiguration = Record<string, Tool | ToolWithPolicies>;

export class Agent {
  private readonly quarantinedModel: LanguageModel;
  private readonly privilegedModel: LanguageModel;
  private readonly systemPrompt: string;
  private readonly toolConfigs: Map<string, ToolWithPolicies>;
  private readonly dataFlowTracker: DataFlowTracker;

  constructor(
    quarantinedModel: LanguageModel,
    privilegedModel: LanguageModel,
    systemPrompt: string,
    tools: ToolConfiguration
  ) {
    this.quarantinedModel = quarantinedModel;
    this.privilegedModel = privilegedModel;
    this.systemPrompt = systemPrompt;
    this.toolConfigs = this.normalizeToolConfiguration(tools);
    this.dataFlowTracker = new DataFlowTracker();
  }

  private normalizeToolConfiguration(
    tools: ToolConfiguration
  ): Map<string, ToolWithPolicies> {
    const normalized = new Map<string, ToolWithPolicies>();

    for (const [name, config] of Object.entries(tools)) {
      if (this.isToolWithPolicies(config)) {
        normalized.set(name, config);
      } else {
        normalized.set(name, { tool: config, policies: [] });
      }
    }

    return normalized;
  }

  private isToolWithPolicies(
    config: Tool | ToolWithPolicies
  ): config is ToolWithPolicies {
    return "tool" in config && "policies" in config;
  }

  private wrapToolWithPolicyCheck(
    toolName: string,
    originalTool: Tool,
    policies: readonly Policy[]
  ): Tool {
    const wrappedExecute = async (params: any, options: any) => {
      const dependencies = this.extractDependencyNodes(params);
      const context = this.buildToolCallContext(toolName, params, dependencies);

      this.enforcePolices(policies, context);

      if (!originalTool.execute) {
        throw new Error(`Tool "${toolName}" has no execute function`);
      }

      const result = await originalTool.execute(params, options);
      const nodeId = this.trackToolResult(result, dependencies);

      return this.dataFlowTracker.wrapValue(result, nodeId);
    };

    return {
      ...originalTool,
      execute: wrappedExecute,
    };
  }

  private extractDependencyNodes(
    params: any
  ): ReadonlyMap<string, import("./data-flow.ts").DataFlowNode> {
    const dependencyIds = this.dataFlowTracker.extractDependencies(params);
    const dependencyNodes = new Map();

    for (const depId of dependencyIds) {
      const node = this.dataFlowTracker.getNode(depId);
      if (node) {
        dependencyNodes.set(depId, node);
      }
    }

    return dependencyNodes;
  }

  private buildToolCallContext(
    toolName: string,
    params: any,
    dependencies: ReadonlyMap<string, import("./data-flow.ts").DataFlowNode>
  ): ToolCallContext {
    return {
      toolName,
      parameters: params,
      dependencies,
    };
  }

  private enforcePolices(
    policies: readonly Policy[],
    context: ToolCallContext
  ): void {
    for (const policy of policies) {
      const result = policy.check(context);

      if (!result.allowed) {
        throw new PolicyViolationError(
          `Policy "${policy.name}" violated: ${
            result.reason || "No reason provided"
          }`,
          policy.name,
          context.toolName
        );
      }
    }
  }

  private trackToolResult(
    result: any,
    dependencies: ReadonlyMap<string, import("./data-flow.js").DataFlowNode>
  ): string {
    const dependencyIds = Array.from(dependencies.keys());
    return this.dataFlowTracker.createNode(result, false, dependencyIds);
  }

  async generate(prompt: string) {
    const sanitizedData = new Map<number, string>();
    const sanitizedNodeIds = new Map<number, string>();

    const generateSantiziedData = async ({
      extractionPrompt,
    }: {
      extractionPrompt: string;
    }) => {
      const extractedValue = await generateText({
        model: this.quarantinedModel,
        system: extractionPrompt,
        prompt,
      });
      const index = [...sanitizedData.entries()].length;
      sanitizedData.set(index, extractedValue.text);

      const nodeId = this.dataFlowTracker.createNode(
        extractedValue.text,
        true,
        []
      );
      sanitizedNodeIds.set(index, nodeId);

      const result = {
        variable: index,
        value: extractedValue.text,
      };

      return this.dataFlowTracker.wrapValue(result, nodeId);
    };

    const fetchSantiziedData = async ({ index }: { index: number }) => {
      const value = sanitizedData.get(index);
      const nodeId = sanitizedNodeIds.get(index);

      const result = {
        variable: index,
        value,
      };

      if (nodeId) {
        return this.dataFlowTracker.wrapValue(result, nodeId);
      }

      return result;
    };

    const wrappedTools: Record<string, Tool> = {};
    for (const [toolName, config] of this.toolConfigs.entries()) {
      const policies = config.policies ?? [];
      wrappedTools[toolName] = this.wrapToolWithPolicyCheck(
        toolName,
        config.tool,
        policies
      );
    }

    const agent = new Experimental_Agent({
      model: this.privilegedModel,
      system: this.systemPrompt,
      tools: {
        ...wrappedTools,
        generateSantiziedData: tool({
          description:
            "Extract a piece of information from the user's unstructed input",
          inputSchema: z.object({
            extractionPrompt: z.string().describe("The extraction prompt"),
          }),
          execute: generateSantiziedData,
        }),
        fetchSantiziedData: tool({
          description: "Fetch a sanitized value that was previously extracted",
          inputSchema: z.object({
            index: z.number().describe("The index of the value"),
          }),
          execute: fetchSantiziedData,
        }),
      },
    });

    const result = await agent.generate({
      prompt,
    });

    return result;
  }
}
