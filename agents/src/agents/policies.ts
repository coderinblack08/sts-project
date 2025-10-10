import {
  Policy,
  type PolicyResult,
  type ToolCallContext,
  type DataFlowNode,
} from "./data-flow.ts";

export class NoSanitizedDependenciesPolicy extends Policy {
  readonly name = "NoSanitizedDependenciesPolicy";

  check(context: ToolCallContext): PolicyResult {
    for (const [nodeId, node] of context.dependencies) {
      if (this.hasSanitizedAncestor(node, context.dependencies)) {
        return {
          allowed: false,
          reason: `Cannot use data that depends on sanitized values. Node ${nodeId} has sanitized dependencies.`,
        };
      }
    }

    return { allowed: true };
  }

  private hasSanitizedAncestor(
    node: DataFlowNode,
    allNodes: ReadonlyMap<string, DataFlowNode>
  ): boolean {
    if (node.isSanitized) {
      return true;
    }

    for (const depId of node.dependencies) {
      const depNode = allNodes.get(depId);
      if (depNode && this.hasSanitizedAncestor(depNode, allNodes)) {
        return true;
      }
    }

    return false;
  }
}

export class ConditionalPolicy extends Policy {
  readonly name: string;
  private readonly condition: (context: ToolCallContext) => boolean;
  private readonly errorMessage: string;

  constructor(
    name: string,
    condition: (context: ToolCallContext) => boolean,
    errorMessage: string
  ) {
    super();
    this.name = name;
    this.condition = condition;
    this.errorMessage = errorMessage;
  }

  check(context: ToolCallContext): PolicyResult {
    if (!this.condition(context)) {
      return {
        allowed: false,
        reason: this.errorMessage,
      };
    }

    return { allowed: true };
  }
}

export function createPolicyBuilder() {
  return {
    blockSanitizedDependencies(): Policy {
      return new NoSanitizedDependenciesPolicy();
    },

    custom(
      name: string,
      condition: (context: ToolCallContext) => boolean,
      errorMessage: string
    ): Policy {
      return new ConditionalPolicy(name, condition, errorMessage);
    },

    blockToolsWithSanitizedData(blockedTools: readonly string[]): Policy {
      const blockedSet = new Set(blockedTools);
      return new ConditionalPolicy(
        "BlockToolsWithSanitizedData",
        (context) => {
          if (!blockedSet.has(context.toolName)) {
            return true;
          }

          return !Array.from(context.dependencies.values()).some(
            (node) => node.isSanitized
          );
        },
        `This tool is blocked from using sanitized data`
      );
    },

    requireCleanData(): Policy {
      return new ConditionalPolicy(
        "RequireCleanData",
        (context) =>
          !Array.from(context.dependencies.values()).some(
            (node) => node.isSanitized
          ),
        "This tool requires clean (non-sanitized) data only"
      );
    },
  };
}
