import { Plugin, TFile, WorkspaceLeaf } from "obsidian";
import { EventListener, VaultEvent } from "./event-listener";
import { JobQueue } from "./job-queue";
import { YakeLite } from "./keyphrase/yake-lite";

export default class NexusPlugin extends Plugin {
  private eventListener!: EventListener;
  private jobQueue!: JobQueue;
  private yakeLite!: YakeLite;

  async onload() {
    console.log("Nexus: loading plugin");

    this.yakeLite = new YakeLite();

    this.jobQueue = new JobQueue(async (job) => {
      console.log(`Nexus: processing ${job.filePath} (${job.priority})`);
      const file = this.app.vault.getAbstractFileByPath(job.filePath);
      if (file instanceof TFile) {
        const content = await this.app.vault.cachedRead(file);
        const phrases = this.yakeLite.extract(content);
        if (phrases.length > 0) {
          console.log(`Nexus: keyphrases for "${job.filePath}":`);
          console.table(phrases.map(p => ({
            phrase: p.phrase,
            score: p.score.toFixed(3),
            offset: `${p.startOffset}:${p.endOffset}`,
          })));
        }
      }
    });

    this.eventListener = new EventListener(this.app.vault, (event: VaultEvent) => {
      console.log(`Nexus: vault event [${event.type}] ${event.file.path} +${Date.now() % 100000}ms`);
      switch (event.type) {
        case "create":
        case "modify":
          this.jobQueue.enqueue(event.file.path, "process");
          break;
        case "delete":
          this.jobQueue.cancel(event.file.path);
          break;
        case "rename":
          if (event.oldPath) this.jobQueue.cancel(event.oldPath);
          this.jobQueue.enqueue(event.file.path, "reindex");
          break;
      }
    });

    this.eventListener.subscribe();

    // Track active file for priority debouncing and enqueue it for processing
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", (leaf: WorkspaceLeaf | null) => {
        const view = leaf?.view as any;
        const file = view?.file;
        const path = file?.path ?? null;
        this.jobQueue.setActiveFile(path);
        if (path && typeof path === "string" && path.endsWith(".md")) {
          this.jobQueue.enqueue(path, "process");
        }
      })
    );
  }

  async onunload() {
    this.eventListener.unsubscribe();
    this.jobQueue.clear();
    console.log("Nexus: unloaded");
  }
}
