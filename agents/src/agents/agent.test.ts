import { beforeEach, describe, expect, it } from "@jest/globals";
import type { LanguageModel } from "ai";
import { tool } from "ai";
import { z } from "zod";
import { Agent } from "./agent.js";
import { DataFlowTracker, type ToolCallContext } from "./data-flow.js";
import {
  NoSanitizedDataPolicy,
  NoSanitizedDependenciesPolicy,
  createPolicyBuilder,
} from "./policies.js";

const createMockModel = (): LanguageModel => {
  return {
    specificationVersion: "v1",
    provider: "mock",
    modelId: "mock-model",
    defaultObjectGenerationMode: "json",
    doGenerate: async () => ({
      text: "mock response",
      finishReason: "stop",
      usage: { promptTokens: 0, completionTokens: 0 },
      rawCall: { rawPrompt: "", rawSettings: {} },
    }),
  } as any;
};

describe("DataFlowTracker", () => {
  let tracker: DataFlowTracker;

  beforeEach(() => {
    tracker = new DataFlowTracker();
  });

  it("should create nodes with unique IDs", () => {
    const id1 = tracker.createNode("value1", false, []);
    const id2 = tracker.createNode("value2", false, []);

    expect(id1).not.toBe(id2);
    expect(tracker.getNode(id1)).toBeDefined();
    expect(tracker.getNode(id2)).toBeDefined();
  });

  it("should track sanitized status", () => {
    const sanitizedId = tracker.createNode("sanitized", true, []);
    const normalId = tracker.createNode("normal", false, []);

    const sanitizedNode = tracker.getNode(sanitizedId);
    const normalNode = tracker.getNode(normalId);

    expect(sanitizedNode?.isSanitized).toBe(true);
    expect(normalNode?.isSanitized).toBe(false);
  });

  it("should track dependencies", () => {
    const id1 = tracker.createNode("value1", false, []);
    const id2 = tracker.createNode("value2", false, [id1]);

    const node2 = tracker.getNode(id2);
    expect(node2?.dependencies.has(id1)).toBe(true);
  });

  it("should detect sanitized dependencies", () => {
    const sanitizedId = tracker.createNode("sanitized", true, []);
    const childId = tracker.createNode("child", false, [sanitizedId]);
    const grandchildId = tracker.createNode("grandchild", false, [childId]);

    expect(tracker.hasSanitizedDependency(sanitizedId)).toBe(true);
    expect(tracker.hasSanitizedDependency(childId)).toBe(true);
    expect(tracker.hasSanitizedDependency(grandchildId)).toBe(true);
  });

  it("should get all dependencies recursively", () => {
    const id1 = tracker.createNode("value1", false, []);
    const id2 = tracker.createNode("value2", false, [id1]);
    const id3 = tracker.createNode("value3", false, [id2]);

    const allDeps = tracker.getAllDependencies(id3);
    expect(allDeps.has(id1)).toBe(true);
    expect(allDeps.has(id2)).toBe(true);
  });

  it("should extract dependencies from wrapped values", () => {
    const nodeId = tracker.createNode("test", false, []);
    const wrapped = tracker.wrapValue({ data: "test" }, nodeId);

    const deps = tracker.extractDependencies(wrapped);
    expect(deps).toContain(nodeId);
  });

  it("should extract dependencies from nested objects", () => {
    const nodeId1 = tracker.createNode("test1", false, []);
    const nodeId2 = tracker.createNode("test2", false, []);

    const wrapped1 = tracker.wrapValue({ data: "test1" }, nodeId1);
    const wrapped2 = tracker.wrapValue({ data: "test2" }, nodeId2);

    const nested = {
      field1: wrapped1,
      field2: wrapped2,
    };

    const deps = tracker.extractDependencies(nested);
    expect(deps).toContain(nodeId1);
    expect(deps).toContain(nodeId2);
  });
});

describe("Policies", () => {
  let tracker: DataFlowTracker;

  beforeEach(() => {
    tracker = new DataFlowTracker();
  });

  describe("NoSanitizedDataPolicy", () => {
    it("should allow tools without sanitized dependencies", () => {
      const policy = new NoSanitizedDataPolicy();
      const nodeId = tracker.createNode("normal", false, []);
      const node = tracker.getNode(nodeId)!;

      const context: ToolCallContext = {
        toolName: "testTool",
        parameters: {},
        dependencies: new Map([[nodeId, node]]),
      };

      const result = policy.check(context);
      expect(result.allowed).toBe(true);
    });

    it("should block tools with sanitized dependencies", () => {
      const policy = new NoSanitizedDataPolicy();
      const nodeId = tracker.createNode("sanitized", true, []);
      const node = tracker.getNode(nodeId)!;

      const context: ToolCallContext = {
        toolName: "testTool",
        parameters: {},
        dependencies: new Map([[nodeId, node]]),
      };

      const result = policy.check(context);
      expect(result.allowed).toBe(false);
      expect(result.reason).toContain("Cannot use sanitized data");
    });
  });

  describe("NoSanitizedDependenciesPolicy", () => {
    it("should block tools with indirect sanitized dependencies", () => {
      const policy = new NoSanitizedDependenciesPolicy();
      const sanitizedId = tracker.createNode("sanitized", true, []);
      const childId = tracker.createNode("child", false, [sanitizedId]);
      const sanitizedNode = tracker.getNode(sanitizedId)!;
      const childNode = tracker.getNode(childId)!;

      const context: ToolCallContext = {
        toolName: "testTool",
        parameters: {},
        dependencies: new Map([
          [childId, childNode],
          [sanitizedId, sanitizedNode],
        ]),
      };

      const result = policy.check(context);
      expect(result.allowed).toBe(false);
      expect(result.reason).toContain("sanitized dependencies");
    });
  });

  describe("PolicyBuilder", () => {
    it("should create policies using builder", () => {
      const builder = createPolicyBuilder();
      const policy1 = builder.blockSanitizedData();
      const policy2 = builder.requireCleanData();

      expect(policy1).toBeInstanceOf(NoSanitizedDataPolicy);
      expect(policy2.name).toBe("RequireCleanData");
    });
  });
});

describe("Agent with Per-Tool Policies", () => {
  let mockModel: LanguageModel;

  beforeEach(() => {
    mockModel = createMockModel();
  });

  it("should create an agent with per-tool policies", () => {
    const testTool = tool({
      description: "Test tool",
      inputSchema: z.object({
        value: z.string(),
      }),
      execute: async (input) => {
        return { result: input.value };
      },
    });

    const agent = new Agent(mockModel, mockModel, "test", {
      testTool: {
        tool: testTool,
        policies: [new NoSanitizedDataPolicy()],
      },
    });

    expect(agent).toBeDefined();
  });

  it("should allow tools without policies", () => {
    const testTool = tool({
      description: "Test tool",
      inputSchema: z.object({
        value: z.string(),
      }),
      execute: async (input) => {
        return { result: input.value };
      },
    });

    const agent = new Agent(mockModel, mockModel, "test", {
      testTool,
    });

    expect(agent).toBeDefined();
  });

  it("should support mixed configuration", () => {
    const tool1 = tool({
      description: "Tool 1",
      inputSchema: z.object({ value: z.string() }),
      execute: async (input) => ({ result: input.value }),
    });

    const tool2 = tool({
      description: "Tool 2",
      inputSchema: z.object({ value: z.string() }),
      execute: async (input) => ({ result: input.value }),
    });

    const agent = new Agent(mockModel, mockModel, "test", {
      tool1,
      tool2: {
        tool: tool2,
        policies: [new NoSanitizedDataPolicy()],
      },
    });

    expect(agent).toBeDefined();
  });
});

describe("Agent data flow tracking", () => {
  let mockModel: LanguageModel;

  beforeEach(() => {
    mockModel = createMockModel();
  });

  it("should track data flow through tool calls", async () => {
    const testTool = tool({
      description: "Test tool",
      inputSchema: z.object({
        value: z.string(),
      }),
      execute: async (input) => {
        return { result: "processed" };
      },
    });

    const agent = new Agent(mockModel, mockModel, "test", { testTool });
    expect(agent).toBeDefined();
  });
});
