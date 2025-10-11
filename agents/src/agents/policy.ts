import type { ToolContext } from "./tool.ts";

export abstract class Policy {
  abstract readonly name: string;
  abstract check(context: ToolContext): PolicyResult;
}

export type PolicyResult =
  | { readonly allowed: true }
  | { readonly allowed: false; readonly reason?: string };

export class PolicyViolationError extends Error {
  constructor(
    message: string,
    public readonly policy: string,
    public readonly toolName: string
  ) {
    super(message);
    this.name = "PolicyViolationError";
  }
}

export class NoUntrustedDependenciesPolicy extends Policy {
  readonly name = "NoUntrustedDependenciesPolicy";

  check(context: ToolContext): PolicyResult {
    for (const [key, { value, isUntrusted }] of Object.entries(
      context.parameters
    )) {
      if (isUntrusted) {
        return {
          allowed: false,
          reason:
            "No parameter may use values that depends on untrusted data. $" +
            key +
            " has untrusted dependencies.",
        };
      }
    }

    return { allowed: true };
  }
}

export class ConditionalPolicy<TC extends ToolContext> extends Policy {
  readonly name: string;
  private readonly condition: (context: TC) => boolean;
  private readonly errorMessage: string;

  constructor(
    name: string,
    condition: (context: TC) => boolean,
    errorMessage: string
  ) {
    super();
    this.name = name;
    this.condition = condition;
    this.errorMessage = errorMessage;
  }

  check(context: TC): PolicyResult {
    if (!this.condition(context)) {
      return {
        allowed: false,
        reason: this.errorMessage,
      };
    }

    return { allowed: true };
  }
}

class ConditionalPolicyBuilder<TC extends ToolContext = ToolContext> {
  private policyName?: string;
  private conditionFn?: (context: TC) => boolean;
  private errorMsg?: string;

  name(name: string): this {
    this.policyName = name;
    return this;
  }

  condition(condition: (context: TC) => boolean): this {
    this.conditionFn = condition;
    return this;
  }

  error(message: string): this {
    this.errorMsg = message;
    return this;
  }

  build(): ConditionalPolicy<TC> {
    if (!this.policyName) {
      throw new Error("Policy name is required");
    }
    if (!this.conditionFn) {
      throw new Error("Policy condition is required");
    }
    if (!this.errorMsg) {
      throw new Error("Policy error message is required");
    }
    return new ConditionalPolicy(
      this.policyName,
      this.conditionFn,
      this.errorMsg
    );
  }
}

export const p = {
  noUntrustedDependencies(): Policy {
    return new NoUntrustedDependenciesPolicy();
  },

  conditional<TC extends ToolContext = ToolContext>(
    nameOrCondition: string | ((context: TC) => boolean)
  ): ConditionalPolicyBuilder<TC> {
    const builder = new ConditionalPolicyBuilder<TC>();
    if (typeof nameOrCondition === "string") {
      return builder.name(nameOrCondition);
    } else {
      return builder.condition(nameOrCondition);
    }
  },

  custom<TC extends ToolContext = ToolContext>(
    name: string,
    condition: (context: TC) => boolean,
    errorMessage: string
  ): Policy {
    return new ConditionalPolicy(name, condition, errorMessage);
  },
};
