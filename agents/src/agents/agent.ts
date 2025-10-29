import type { LanguageModel } from "ai";
import { Experimental_Agent, generateText, stepCountIs, tool } from "ai";
import { z } from "zod";
import { KvNode, KvStore } from "./kv.ts";
import { type Policy, PolicyViolationError } from "./policy.ts";
import { SYSTEM_PROMPT } from "./prompts.ts";
import type { Tool, ToolContext, ToolWithPolicies } from "./tool.ts";

export class Agent {
  private readonly qLLM: LanguageModel;
  private readonly pLLM: LanguageModel;
  private readonly tools: Record<string, ToolWithPolicies>;
  private readonly kv: KvStore = new KvStore();

  constructor(
    qLLM: LanguageModel,
    pLLM: LanguageModel,
    tools: Record<string, ToolWithPolicies>
  ) {
    this.qLLM = qLLM;
    this.pLLM = pLLM;
    this.tools = tools;
  }

  private wrapToolWithPolicyCheck(
    name: string,
    tool: Tool,
    policies: readonly Policy[],
    isExternal: boolean
  ): Tool {
    const wrappedExecute: typeof tool.execute = async (params, options) => {
      let dependencies: KvNode[] = [];
      const context: ToolContext = {
        name,
        parameters: {},
      };

      for (const [key, value] of Object.entries(params)) {
        if (typeof value === "string") {
          if (this.kv.isKey(value)) {
            const node = this.kv.getNode(value);
            if (node) {
              params[key] = JSON.stringify(node.value);
              dependencies.push(node);
              context.parameters[key] = {
                value: node.value,
                isUntrusted: true,
              };
            } else {
              throw new Error("Index of " + value + " not found");
            }
          } else {
            context.parameters[key] = { value, isUntrusted: false };
          }
        }
      }

      this.enforceToolPolices(policies, context);

      if (!tool.execute) {
        throw new Error(`Tool "${name}" has no execute function`);
      }

      const result = await tool.execute(params, options);
      if (dependencies.length > 0 || isExternal) {
        const id = this.kv.createNode(result, true, dependencies);
        return { __id: id };
      } else {
        return result;
      }
    };

    return {
      ...tool,
      execute: wrappedExecute,
    };
  }

  private enforceToolPolices(
    policies: readonly Policy[],
    context: ToolContext
  ): void {
    for (const policy of policies) {
      const result = policy.check(context);

      if (!result.allowed) {
        throw new PolicyViolationError(
          `Policy "${policy.name}" violated: ${
            result.reason || "No reason provided"
          }`,
          policy.name,
          context.name
        );
      }
    }
  }

  async generate(prompt: string) {
    this.kv.reset();

    const qLLM = tool({
      description: "Invoke the quarantined LLM to interact with untrusted data",
      inputSchema: z.object({
        qPrompt: z.string().describe("Instructions for the quarantined LLM"),
        untrustedData: z
          .string()
          .describe("The untrusted data to interact with (e.g. $0 or $1)"),
      }),
      execute: async ({ qPrompt, untrustedData }) => {
        if (!this.kv.isKey(untrustedData)) {
          throw new Error("Invalid key: " + untrustedData);
        }
        const node = this.kv.getNode(untrustedData);
        if (!node) {
          throw new Error("Key not found: " + untrustedData);
        }
        const extractedValue = await generateText({
          model: this.qLLM,
          system: qPrompt,
          prompt: JSON.stringify(node.value),
        });

        const id = this.kv.createNode(extractedValue.text, true);
        return { __id: id };
      },
    });

    const wrappedTools: Record<string, Tool> = {};
    for (const [name, config] of Object.entries(this.tools)) {
      wrappedTools[name] = this.wrapToolWithPolicyCheck(
        name,
        config.tool,
        config.policies,
        config.isExternal
      );
    }

    const agent = new Experimental_Agent({
      model: this.pLLM,
      system: SYSTEM_PROMPT,
      tools: { ...wrappedTools, qLLM },
      stopWhen: stepCountIs(3),
    });

    const result = await agent.generate({ prompt });

    await Bun.write("agent_output.txt", JSON.stringify(result.steps, null, 2));
    console.log(`[💾] Saved agent output to agent_output.txt`);

    if (this.kv.isKey(result.text)) {
      const node = this.kv.getNode(result.text);
      if (node) {
        return node.value;
      }
    }

    for (const step of result.steps) {
      console.log(step.toolCalls, step.toolResults);
    }

    return result.text;
  }
}
