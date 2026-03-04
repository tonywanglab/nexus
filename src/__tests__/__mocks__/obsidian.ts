/* Minimal Obsidian API mock for unit testing. */

export interface EventRef {
  _event: string;
  _fn: (...args: any[]) => void;
}

export abstract class TAbstractFile {
  vault!: Vault;
  path: string;
  name: string;
  parent: TFolder | null = null;

  constructor(path: string) {
    this.path = path;
    this.name = path.split("/").pop() ?? path;
  }
}

export class TFile extends TAbstractFile {
  stat = { ctime: 0, mtime: 0, size: 0 };
  basename: string;
  extension: string;

  constructor(path: string) {
    super(path);
    const dot = this.name.lastIndexOf(".");
    if (dot !== -1) {
      this.basename = this.name.slice(0, dot);
      this.extension = this.name.slice(dot + 1);
    } else {
      this.basename = this.name;
      this.extension = "";
    }
  }
}

export class TFolder extends TAbstractFile {
  children: TAbstractFile[] = [];

  constructor(path: string) {
    super(path);
  }

  isRoot(): boolean {
    return this.parent === null;
  }
}

export class Vault {
  private handlers = new Map<string, Array<(...args: any[]) => void>>();

  on(event: string, callback: (...args: any[]) => void): EventRef {
    if (!this.handlers.has(event)) this.handlers.set(event, []);
    this.handlers.get(event)!.push(callback);
    return { _event: event, _fn: callback };
  }

  offref(ref: EventRef): void {
    const list = this.handlers.get(ref._event);
    if (list) {
      const idx = list.indexOf(ref._fn);
      if (idx !== -1) list.splice(idx, 1);
    }
  }

  /** Test helper — simulates a vault event. */
  trigger(event: string, ...args: any[]): void {
    for (const fn of this.handlers.get(event) ?? []) {
      fn(...args);
    }
  }
}

export class Plugin {
  app = { vault: new Vault(), workspace: { on: () => ({}) } };
  registerEvent(_ref: EventRef): void {}
  onload(): void {}
  onunload(): void {}
}

export class WorkspaceLeaf {
  view: any = null;
}
