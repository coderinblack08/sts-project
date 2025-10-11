import type { Tool as _Tool } from "ai";
import type { KvNode } from "./kv.ts";
import type { Policy } from "./policy.ts";

import type { z } from "zod";

export type Tool = _Tool;

export interface ToolWithPolicies {
  readonly tool: Tool;
  readonly policies: readonly Policy[];
}

export type Tools = Record<string, Tool | ToolWithPolicies>;

export interface ToolContext<Schema> {
  readonly name: string;
  readonly parameters: z.infer<Schema>;
}
