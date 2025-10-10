export interface DataFlowNode {
  readonly id: string;
  readonly value: any;
  readonly isSanitized: boolean;
  readonly dependencies: ReadonlySet<string>;
  readonly metadata?: Readonly<Record<string, any>>;
}

export interface ToolCallContext {
  readonly toolName: string;
  readonly parameters: Readonly<Record<string, any>>;
  readonly dependencies: ReadonlyMap<string, DataFlowNode>;
}

export interface PolicyResult {
  readonly allowed: boolean;
  readonly reason?: string;
}

export abstract class Policy {
  abstract readonly name: string;
  abstract check(context: ToolCallContext): PolicyResult;
}

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

export class DataFlowTracker {
  private readonly nodes = new Map<string, DataFlowNode>();
  private nextId = 0;

  createNode(
    value: any,
    isSanitized: boolean,
    dependencies: readonly string[] = []
  ): string {
    const id = `node_${this.nextId++}`;
    const node: DataFlowNode = {
      id,
      value,
      isSanitized,
      dependencies: new Set(dependencies),
    };
    this.nodes.set(id, node);
    return id;
  }

  getNode(id: string): DataFlowNode | undefined {
    return this.nodes.get(id);
  }

  hasSanitizedDependency(nodeId: string): boolean {
    const node = this.nodes.get(nodeId);
    if (!node) {
      return false;
    }

    if (node.isSanitized) {
      return true;
    }

    return Array.from(node.dependencies).some((depId) =>
      this.hasSanitizedDependency(depId)
    );
  }

  getAllDependencies(nodeId: string): ReadonlySet<string> {
    const allDeps = new Set<string>();
    const visited = new Set<string>();

    const collectDependencies = (id: string): void => {
      if (visited.has(id)) {
        return;
      }

      visited.add(id);
      const node = this.nodes.get(id);

      if (!node) {
        return;
      }

      for (const depId of node.dependencies) {
        allDeps.add(depId);
        collectDependencies(depId);
      }
    };

    collectDependencies(nodeId);
    return allDeps;
  }

  extractDependencies(value: any): readonly string[] {
    if (!this.isObject(value)) {
      return [];
    }

    const deps: string[] = [];

    if (Array.isArray(value)) {
      for (const item of value) {
        deps.push(...this.extractDependencies(item));
      }
    } else if (this.hasDataFlowId(value)) {
      deps.push(value.__dataFlowId);
    } else {
      const objValue = value as Record<string, any>;
      for (const key in objValue) {
        if (Object.prototype.hasOwnProperty.call(objValue, key)) {
          deps.push(...this.extractDependencies(objValue[key]));
        }
      }
    }

    return deps;
  }

  wrapValue(value: any, nodeId: string): any {
    if (!this.isObject(value)) {
      return value;
    }

    return {
      ...value,
      __dataFlowId: nodeId,
    };
  }

  private isObject(value: any): value is object {
    return typeof value === "object" && value !== null;
  }

  private hasDataFlowId(
    value: any
  ): value is { __dataFlowId: string } & Record<string, any> {
    return "__dataFlowId" in value && typeof value.__dataFlowId === "string";
  }
}
