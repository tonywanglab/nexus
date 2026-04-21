import { Plugin, TFile, WorkspaceLeaf } from "obsidian";
import { EventListener, VaultEvent } from "./event-listener";
import { JobQueue } from "./job-queue";
import { SpanExtractor } from "./keyphrase/span-extractor";
import { AliasResolver } from "./resolver";
import { EmbeddingResolver } from "./resolver/embedding-resolver";
import { TransformersIframeProvider, EmbeddingProvider } from "./resolver/embedding-provider";
import { loadBundledSAE } from "./resolver/sae-weights";
import { VaultContext } from "./types";

export default class NexusPlugin extends Plugin {
  private eventListener!: EventListener;
  private jobQueue!: JobQueue;
  private extractor!: SpanExtractor;
  private resolver!: AliasResolver;
  private embeddingResolver!: EmbeddingResolver;
  private embeddingProvider!: EmbeddingProvider;

  async onload() {
    console.log("Nexus: loading plugin");

    this.extractor = new SpanExtractor();
    this.resolver = new AliasResolver();
    this.embeddingProvider = new TransformersIframeProvider();
    const sae = loadBundledSAE();
    console.log(
      `Nexus: loaded SAE (d_model=${sae.dModel}, d_hidden=${sae.dHidden}, k=${sae.k})`,
    );
    this.embeddingResolver = new EmbeddingResolver({
      embeddingProvider: this.embeddingProvider,
      sae,
    });

    this.jobQueue = new JobQueue(async (job) => {
      console.log(`Nexus: processing ${job.filePath} (${job.priority})`);
      const file = this.app.vault.getAbstractFileByPath(job.filePath);
      if (file instanceof TFile) {
        const content = await this.app.vault.cachedRead(file);
        const noteTitles = this.app.vault.getMarkdownFiles().map((f: TFile) => f.basename);
        const vaultContext: VaultContext = { noteTitles };
        const phrases = this.extractor.extract(content, vaultContext);
        if (phrases.length > 0) {
          console.log(`Nexus: keyphrases for "${job.filePath}":`);
          console.table(phrases.map(p => ({
            phrase: p.phrase,
            score: p.score.toFixed(3),
            offset: `${p.startOffset}:${p.endOffset}`,
          })));

          const edges = this.resolver.resolve(phrases, noteTitles, job.filePath);
          if (edges.length > 0) {
            console.log(`Nexus: deterministic edges for "${job.filePath}":`);
            console.table(edges.map(e => ({
              phrase: e.phrase.phrase,
              target: e.targetPath,
              similarity: e.similarity.toFixed(3),
              phraseScore: e.phrase.score.toFixed(3),
            })));
          } else {
            console.log(`Nexus: no deterministic edges for "${job.filePath}"`);
          }

          try {
            const embeddingEdges = await this.embeddingResolver.resolve(
              phrases, noteTitles, job.filePath,
            );
            if (embeddingEdges.length > 0) {
              console.log(`Nexus: stochastic edges for "${job.filePath}":`);
              console.table(embeddingEdges.map(e => ({
                phrase: e.phrase.phrase,
                target: e.targetPath,
                similarity: e.similarity.toFixed(3),
                phraseScore: e.phrase.score.toFixed(3),
              })));
            } else {
              console.log(`Nexus: no stochastic edges for "${job.filePath}"`);
            }
          } catch (err) {
            console.warn("Nexus: embedding resolver failed:", err);
          }
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
          this.jobQueue.enqueue(event.file.path, "process");
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
    await this.embeddingProvider.dispose();
    console.log("Nexus: unloaded");
  }
}
