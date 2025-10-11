export class KvNode {
  constructor(
    readonly id: string,
    readonly value: any,
    readonly isUntrusted: boolean,
    readonly dependencies: ReadonlySet<string>
  ) {}
}

export class KvStore {
  private readonly nodes = new Map<string, KvNode>();
  private index = 0;

  reset(): void {
    this.nodes.clear();
    this.index = 0;
  }

  createNode(
    value: any,
    isUntrusted: boolean,
    dependencies: readonly string[] = []
  ): string {
    const id = "$" + (this.index++).toString();
    const node = new KvNode(id, value, isUntrusted, new Set(dependencies));
    this.nodes.set(id, node);
    return id;
  }

  getNode(id: string): KvNode | undefined {
    const match = id.match(/^\$(\d+)$/);
    if (!match) {
      throw new Error("Invalid id: " + id);
    }
    return this.nodes.get(id);
  }

  hasUntrustedDependency(id: string): boolean {
    const node = this.nodes.get(id);
    if (!node) {
      return false;
    }

    if (node.isUntrusted) {
      return true;
    }

    return Array.from(node.dependencies).some((depId) =>
      this.hasUntrustedDependency(depId)
    );
  }

  getAllDependencies(id: string): ReadonlySet<KvNode> {
    const dependencies = new Set<KvNode>();
    const queue = [id];

    while (queue.length > 0) {
      const currentId = queue.pop()!;
      const node = this.nodes.get(currentId);

      if (!node) continue;

      for (const depId of node.dependencies) {
        const depNode = this.nodes.get(depId);
        if (depNode && !dependencies.has(depNode)) {
          dependencies.add(node);
          queue.push(depId);
        }
      }
    }

    return dependencies;
  }

  flattenDependencies(value: any): readonly string[] {
    if (!this.isObject(value)) {
      return [];
    }

    const deps: string[] = [];

    if (Array.isArray(value)) {
      for (const item of value) {
        deps.push(...this.flattenDependencies(item));
      }
    } else if (this.hasId(value)) {
      deps.push(value.__id);
    } else {
      const objValue = value as Record<string, any>;
      for (const key in objValue) {
        if (Object.prototype.hasOwnProperty.call(objValue, key)) {
          deps.push(...this.flattenDependencies(objValue[key]));
        }
      }
    }

    return deps;
  }

  wrapValue(value: any, id: string): any {
    if (!this.isObject(value)) {
      return value;
    }

    return {
      ...value,
      __id: id,
    };
  }

  private isObject(value: any): value is object {
    return typeof value === "object" && value !== null;
  }

  private hasId(value: any): value is { __id: string } & Record<string, any> {
    return "__id" in value && typeof value.__id === "string";
  }
}
