export class KvNode {
  constructor(
    readonly id: string,
    readonly value: any,
    readonly isExternal: boolean,
    readonly dependencies: ReadonlySet<KvNode>
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
    isExternal: boolean,
    dependencies: readonly KvNode[] = []
  ): string {
    const id = "$" + (this.index++).toString();
    const node = new KvNode(id, value, isExternal, new Set(dependencies));
    this.nodes.set(id, node);
    return id;
  }

  getNode(id: string): KvNode | undefined {
    if (!this.isKey(id)) {
      return undefined;
    }
    return this.nodes.get(id);
  }

  getDependencies(id: string): ReadonlySet<KvNode> {
    const dependencies = new Set<KvNode>();
    const queue = [id];

    while (queue.length > 0) {
      const currentId = queue.pop()!;
      const node = this.nodes.get(currentId);

      if (!node) continue;

      for (const dependency of node.dependencies) {
        const depNode = this.nodes.get(dependency.id);
        if (depNode && !dependencies.has(depNode)) {
          dependencies.add(node);
          queue.push(dependency.id);
        }
      }
    }

    return dependencies;
  }

  wrapValue(value: any, id: string): { value: any; __id: string } {
    return {
      value,
      __id: id,
    };
  }

  isKey(id: string): boolean {
    return /^\$(\d+)$/.test(id);
  }

  private hasId(value: any): value is { __id: string } & Record<string, any> {
    return "__id" in value && typeof value.__id === "string";
  }
}
