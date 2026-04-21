import { App, Modal } from "obsidian";

export class ConfirmModal extends Modal {
  private message: string;
  private resolve!: (confirmed: boolean) => void;

  constructor(app: App, message: string) {
    super(app);
    this.message = message;
  }

  static prompt(app: App, message: string): Promise<boolean> {
    return new Promise((resolve) => {
      const modal = new ConfirmModal(app, message);
      modal.resolve = resolve;
      modal.open();
    });
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("p", { text: this.message });
    const btns = contentEl.createEl("div", { cls: "nexus-confirm-buttons" });

    const cancel = btns.createEl("button", { text: "Cancel", cls: "nexus-btn" });
    cancel.addEventListener("click", () => {
      this.resolve(false);
      this.close();
    });

    const confirm = btns.createEl("button", { text: "Continue", cls: "nexus-btn mod-cta" });
    confirm.addEventListener("click", () => {
      this.resolve(true);
      this.close();
    });
  }

  onClose(): void {
    this.contentEl.empty();
  }
}
