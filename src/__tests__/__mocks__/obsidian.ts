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

  constructor(path: string, stat?: { ctime?: number; mtime?: number; size?: number }) {
    super(path);
    const dot = this.name.lastIndexOf(".");
    if (dot !== -1) {
      this.basename = this.name.slice(0, dot);
      this.extension = this.name.slice(dot + 1);
    } else {
      this.basename = this.name;
      this.extension = "";
    }
    if (stat) {
      this.stat = {
        ctime: stat.ctime ?? 0,
        mtime: stat.mtime ?? 0,
        size: stat.size ?? 0,
      };
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
  private files = new Map<string, TFile>();
  private contents = new Map<string, string>();

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

  /** Test helper — registers a file in the mock vault. */
  _addFile(file: TFile, content = ""): void {
    this.files.set(file.path, file);
    this.contents.set(file.path, content);
  }

  /** Test helper — removes a file from the mock vault. */
  _removeFile(path: string): void {
    this.files.delete(path);
    this.contents.delete(path);
  }

  getMarkdownFiles(): TFile[] {
    return [...this.files.values()].filter((f) => f.extension === "md");
  }

  getAbstractFileByPath(path: string): TAbstractFile | null {
    return this.files.get(path) ?? null;
  }

  async cachedRead(file: TFile): Promise<string> {
    return this.contents.get(file.path) ?? "";
  }

  async read(file: TFile): Promise<string> {
    return this.contents.get(file.path) ?? "";
  }

  async modify(file: TFile, content: string): Promise<void> {
    this.contents.set(file.path, content);
    const f = this.files.get(file.path);
    if (f) f.stat.mtime = Date.now();
  }
}

export class Workspace {
  private handlers = new Map<string, Array<(...args: any[]) => void>>();
  private activeFile: TFile | null = null;

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

  trigger(event: string, ...args: any[]): void {
    for (const fn of this.handlers.get(event) ?? []) {
      fn(...args);
    }
  }

  getActiveFile(): TFile | null {
    return this.activeFile;
  }

  _setActiveFile(file: TFile | null): void {
    this.activeFile = file;
  }

  getRightLeaf(_newPane: boolean): WorkspaceLeaf {
    return new WorkspaceLeaf();
  }

  revealLeaf(_leaf: WorkspaceLeaf): void {}

  getLeavesOfType(_type: string): WorkspaceLeaf[] {
    return [];
  }
}

export class Plugin {
  app = { vault: new Vault(), workspace: new Workspace() };
  private _data: any = null;

  registerEvent(_ref: EventRef): void {}
  registerView(_type: string, _factory: (leaf: WorkspaceLeaf) => any): void {}
  addCommand(_cmd: any): void {}
  addRibbonIcon(_icon: string, _title: string, _callback: () => void): HTMLElement {
    return {} as HTMLElement;
  }
  async loadData(): Promise<any> {
    return this._data;
  }
  async saveData(data: any): Promise<void> {
    this._data = data;
  }
  onload(): void {}
  onunload(): void {}
}

export class Modal {
  app: any;
  contentEl: any = createElementStub();
  constructor(app: any) {
    this.app = app;
  }
  open(): void {}
  close(): void {}
  onOpen(): void {}
  onClose(): void {}
}

export class ItemView {
  leaf: WorkspaceLeaf;
  containerEl: any;
  contentEl: any;

  constructor(leaf: WorkspaceLeaf) {
    this.leaf = leaf;
    this.contentEl = createElementStub();
    this.containerEl = { children: [createElementStub(), this.contentEl] };
  }

  getViewType(): string {
    return "";
  }
  getDisplayText(): string {
    return "";
  }
  async onOpen(): Promise<void> {}
  async onClose(): Promise<void> {}
}

export class Notice {
  noticeEl: any = createElementStub();
  constructor(public message: string, public timeout?: number) {}
  hide(): void {}
  setMessage(message: string): this {
    this.message = message;
    return this;
  }
}

/** Minimal element stub — enough to exercise Obsidian plugin UI code in Node. */
export function createElementStub(): any {
  const listeners: Record<string, Array<(...args: any[]) => void>> = {};
  const el: any = {
    children: [] as any[],
    _attrs: {} as Record<string, string>,
    _text: "",
    _classes: new Set<string>(),
    createEl(_tag: string, opts?: any): any {
      const child = createElementStub();
      if (opts?.text) child._text = opts.text;
      if (opts?.cls) {
        const classes = Array.isArray(opts.cls) ? opts.cls : [opts.cls];
        for (const c of classes) child._classes.add(c);
      }
      if (opts?.attr) {
        for (const [k, v] of Object.entries(opts.attr)) child._attrs[k] = String(v);
      }
      el.children.push(child);
      return child;
    },
    createDiv(opts?: any): any {
      return el.createEl("div", opts);
    },
    createSpan(opts?: any): any {
      return el.createEl("span", opts);
    },
    addEventListener(event: string, cb: (...args: any[]) => void): void {
      if (!listeners[event]) listeners[event] = [];
      listeners[event].push(cb);
    },
    _dispatch(event: string, ...args: any[]): void {
      for (const cb of listeners[event] ?? []) cb(...args);
    },
    setAttr(k: string, v: any): void {
      el._attrs[k] = String(v);
    },
    setText(t: string): void {
      el._text = t;
    },
    empty(): void {
      el.children = [];
      el._text = "";
    },
    remove(): void {},
    addClass(c: string): void {
      el._classes.add(c);
    },
    removeClass(c: string): void {
      el._classes.delete(c);
    },
    hasClass(c: string): boolean {
      return el._classes.has(c);
    },
  };
  return el;
}

export class WorkspaceLeaf {
  view: any = null;
}
