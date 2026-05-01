import { Notice } from "obsidian";

export class ProgressToast {
  private notice: Notice;
  private total: number;
  private remaining: number;
  private label: string;
  private doneLabel: string;
  private currentFile: string | null = null;

  constructor(total: number, label: string = "re-resolving", doneLabel: string = "re-resolve complete.") {
    this.total = total;
    this.remaining = total;
    this.label = label;
    this.doneLabel = doneLabel;
    this.notice = new Notice(this.buildMessage(), 0);
  }

  //  Update the name of the file currently being processed.
  setCurrent(path: string | null): void {
    this.currentFile = path;
    if (this.remaining > 0) this.notice.setMessage(this.buildMessage());
  }

  tick(): void {
    this.remaining = Math.max(0, this.remaining - 1);
    if (this.remaining === 0) {
      this.notice.setMessage(`Nexus: ${this.doneLabel}`);
      setTimeout(() => this.notice.hide(), 2000);
    } else {
      this.notice.setMessage(this.buildMessage());
    }
  }

  dismiss(): void {
    this.notice.hide();
  }

  private buildMessage(): string {
    const done = this.total - this.remaining;
    const base = `Nexus: ${this.label} ${done} of ${this.total}`;
    if (!this.currentFile) return `${base}…`;
    const name = basename(this.currentFile);
    return `${base} — ${name}…`;
  }
}

function basename(path: string): string {
  const slash = path.lastIndexOf("/");
  const name = slash === -1 ? path : path.slice(slash + 1);
  return name.endsWith(".md") ? name.slice(0, -3) : name;
}
