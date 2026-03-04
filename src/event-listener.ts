import { Vault, TAbstractFile, TFile, EventRef } from "obsidian";

export type VaultEventType = "create" | "modify" | "delete" | "rename";

export interface VaultEvent {
  type: VaultEventType;
  file: TFile;
  oldPath?: string;
}

export type VaultEventCallback = (event: VaultEvent) => void;

export class EventListener {
  private refs: EventRef[] = [];
  private subscribed = false;

  constructor(
    private vault: Vault,
    private callback: VaultEventCallback
  ) {}

  subscribe(): void {
    if (this.subscribed) return;
    this.subscribed = true;

    const handle = (type: VaultEventType) => {
      return (file: TAbstractFile, oldPath?: string) => {
        if (file instanceof TFile && file.extension === "md") {
          this.callback({ type, file, oldPath });
        }
      };
    };

    this.refs.push(this.vault.on("create", handle("create")));
    this.refs.push(this.vault.on("modify", handle("modify")));
    this.refs.push(this.vault.on("delete", handle("delete")));
    this.refs.push(this.vault.on("rename", handle("rename")));
  }

  unsubscribe(): void {
    for (const ref of this.refs) {
      this.vault.offref(ref);
    }
    this.refs = [];
    this.subscribed = false;
  }

  get isSubscribed(): boolean {
    return this.subscribed;
  }
}
