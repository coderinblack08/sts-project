import type { Tool as _T } from "ai";
import type { KvNode } from "./kv.ts";
import type { Policy } from "./policy.ts";

export type Tool = _T;

export interface ToolWithPolicies {
  readonly tool: Tool;
  readonly policies: readonly Policy[];
}

export type Tools = Record<string, Tool | ToolWithPolicies>;

export interface ToolContext<
  TParams = {
    [key: string]: {
      readonly value: string | number | boolean | null | undefined;
      readonly isUntrusted: boolean;
    };
  }
> {
  readonly name: string;
  readonly parameters: TParams;
}
