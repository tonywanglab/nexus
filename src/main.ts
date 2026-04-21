import { Plugin, TFile, WorkspaceLeaf, Vault, Notice } from "obsidian";
import { EventListener, VaultEvent } from "./event-listener";
import { JobQueue } from "./job-queue";
import { SpanExtractor } from "./keyphrase/span-extractor";
import { AliasResolver } from "./resolver";
import { EmbeddingResolver } from "./resolver/embedding-resolver";
import { TransformersIframeProvider, EmbeddingProvider } from "./resolver/embedding-provider";
import { loadBundledSAE } from "./resolver/sae-weights";
import { SAEFeatureLabels } from "./resolver/sae-feature-labels";
import { VaultContext } from "./types";
import { NoteRegistry } from "./note-registry";
import { EdgeStore } from "./edge-store";
import { deserialize, serialize } from "./persistence";
import { initializeFromCache } from "./initialize";
import { mergeByTarget } from "./resolver/merge-by-target";
import { NexusApprovalView, APPROVAL_VIEW_TYPE } from "./ui/approval-view";
import { NexusSettingsModal } from "./ui/settings-modal";
import { ProgressToast } from "./ui/progress-toast";

export default class NexusPlugin extends Plugin {
  private eventListener!: EventListener;
  private jobQueue!: JobQueue;
  private extractor!: SpanExtractor;
  private resolver!: AliasResolver;
  private embeddingResolver!: EmbeddingResolver;
  private embeddingProvider!: EmbeddingProvider;
  private featureLabels!: SAEFeatureLabels;
  private noteRegistry!: NoteRegistry;
  private edgeStore!: EdgeStore;
  private indexingProgress: { toast: ProgressToast; pending: Set<string> } | null = null;

  async onload() {
    console.log("Nexus: loading plugin");

    const rawData = await this.loadData();
    const persisted = deserialize(rawData);

    this.noteRegistry = new NoteRegistry(persisted.notes);
    this.edgeStore = new EdgeStore(persisted, async (state) => {
      // Merge current path info from registry into the state before saving.
      const noteMap = this.noteRegistry.toNoteMap();
      for (const [id, meta] of Object.entries(noteMap)) {
        if (state.notes[id]) state.notes[id].path = meta.path;
      }
      await this.saveData(serialize(state));
    });

    this.extractor = new SpanExtractor();
    this.resolver = new AliasResolver();
    this.embeddingProvider = new TransformersIframeProvider();
    this.wireModelDownloadNotice();
    const sae = loadBundledSAE();
    this.featureLabels = new SAEFeatureLabels();
    console.log(
      `Nexus: loaded SAE (d_model=${sae.dModel}, d_hidden=${sae.dHidden}, k=${sae.k}), feature labels: ${this.featureLabels.liveCount}/${sae.dHidden} live`,
    );
    this.embeddingResolver = new EmbeddingResolver({
      embeddingProvider: this.embeddingProvider,
      sae,
    });

    this.jobQueue = new JobQueue(async (job) => {
      const file = this.app.vault.getAbstractFileByPath(job.filePath);
      if (!(file instanceof TFile)) return;

      const t0 = performance.now();
      console.log(`Nexus: processing ${job.filePath}`);
      if (this.indexingProgress?.pending.has(job.filePath)) {
        this.indexingProgress.toast.setCurrent(job.filePath);
      }

      const sourceId = this.noteRegistry.ensureId(job.filePath, file.stat.mtime, file.stat.size);
      const content = await this.app.vault.cachedRead(file);
      const allFiles = this.app.vault.getMarkdownFiles() as TFile[];
      const noteTitles = allFiles.map((f) => f.basename);
      // Edges carry `targetPath` as a basename (from `noteTitles`), but the
      // registry is keyed by full path — so we need a basename→id map to turn
      // them into the stable ids that denial/approval records rely on.
      const basenameToId = new Map<string, string>();
      for (const f of allFiles) {
        const id = this.noteRegistry.getId(f.path);
        if (id) basenameToId.set(f.basename, id);
      }
      const vaultContext: VaultContext = { noteTitles };
      const phrases = this.extractor.extract(content, vaultContext);
      console.log(`Nexus:   extracted ${phrases.length} phrase(s)`);

      if (phrases.length === 0) {
        this.edgeStore.setEdgesForFile(sourceId, [], file.stat.mtime);
        console.log(`Nexus: done ${job.filePath} (0 edges, ${Math.round(performance.now() - t0)}ms)`);
        this.tickIndexingProgress(job.filePath);
        return;
      }

      const detEdges = this.resolver.resolve(phrases, noteTitles, job.filePath);
      console.log(`Nexus:   ${detEdges.length} deterministic edge(s)`);

      // Annotate with ids
      const annotated = detEdges.map((e) => ({
        ...e,
        sourceId,
        targetId: basenameToId.get(e.targetPath),
      }));

      // Publish deterministic edges immediately so first-pass files don't wait
      // on embeddings. Skip this write when the file already has edges — an
      // approve/modify triggers reprocess, and wiping the emb-edge list during
      // the ~60s embedding window makes it look like approving nuked the list.
      // Don't bump mtime either way — if we're killed before embeddings finish,
      // the file should be reprocessed on next load.
      const detMerged = mergeByTarget(annotated);
      if (!this.edgeStore.hasEdgesFor(sourceId)) {
        this.edgeStore.setInterimEdgesForFile(sourceId, detMerged);
      }

      try {
        const embT0 = performance.now();
        const embEdges = await this.embeddingResolver.resolve(
          phrases,
          noteTitles,
          job.filePath,
          job.priority,
        );
        console.log(
          `Nexus:   ${embEdges.length} embedding edge(s) (${Math.round(performance.now() - embT0)}ms)`,
        );
        const annotatedEmb = embEdges.map((e) => ({
          ...e,
          sourceId,
          targetId: basenameToId.get(e.targetPath),
        }));

        let annotatedSparse: typeof annotatedEmb = [];
        try {
          const sparseT0 = performance.now();
          const sparseEdges = await this.embeddingResolver.resolveBySparseFeatures(
            phrases,
            noteTitles,
            job.filePath,
            this.featureLabels,
            { similarityThreshold: 0.75, priority: job.priority },
          );
          console.log(
            `Nexus:   ${sparseEdges.length} sparse feature edge(s) (${Math.round(performance.now() - sparseT0)}ms)`,
          );
          annotatedSparse = sparseEdges.map((e) => ({
            ...e,
            sourceId,
            targetId: basenameToId.get(e.targetPath),
          }));
        } catch (sparseErr) {
          console.warn("Nexus: sparse resolver failed:", sparseErr);
        }

        const merged = mergeByTarget([...annotated, ...annotatedEmb, ...annotatedSparse]);
        this.edgeStore.setEdgesForFile(sourceId, merged, file.stat.mtime);
        console.log(
          `Nexus: done ${job.filePath} (${merged.length} merged edge(s), ${Math.round(performance.now() - t0)}ms)`,
        );
      } catch (err) {
        console.warn("Nexus: embedding resolver failed:", err);
        // Commit det-only result with mtime so we don't infinite-retry a file
        // whose embeddings keep failing. User can force-reindex from settings.
        this.edgeStore.setEdgesForFile(sourceId, detMerged, file.stat.mtime);
        console.log(
          `Nexus: done ${job.filePath} (${detMerged.length} det-only edge(s), ${Math.round(performance.now() - t0)}ms)`,
        );
      }
      this.tickIndexingProgress(job.filePath);
    });

    this.eventListener = new EventListener(this.app.vault, (event: VaultEvent) => {
      switch (event.type) {
        case "create":
        case "modify":
          this.jobQueue.enqueue(event.file.path, "process");
          break;
        case "delete":
          this.jobQueue.cancel(event.file.path);
          const deletedId = this.noteRegistry.drop(event.file.path);
          if (deletedId) this.edgeStore.dropNote(deletedId);
          break;
        case "rename":
          if (event.oldPath) {
            this.jobQueue.cancel(event.oldPath);
            this.noteRegistry.rename(event.oldPath, event.file.path);
          }
          this.jobQueue.enqueue(event.file.path, "process");
          break;
      }
    });

    // Track active file for priority debouncing — only enqueue if mtime has changed.
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", (leaf: WorkspaceLeaf | null) => {
        const view = leaf?.view as any;
        const file = view?.file;
        const path = file?.path ?? null;
        this.jobQueue.setActiveFile(path);
        if (path && typeof path === "string" && path.endsWith(".md")) {
          const id = this.noteRegistry.getId(path);
          const cachedMtime = id ? this.edgeStore.getMtime(id) : null;
          const currentFile = this.app.vault.getAbstractFileByPath(path);
          const currentMtime = currentFile instanceof TFile ? currentFile.stat.mtime : null;
          if (cachedMtime === null || cachedMtime !== currentMtime) {
            this.jobQueue.enqueue(path, "process");
          }
        }
      })
    );

    // Register side panel view.
    this.registerView(
      APPROVAL_VIEW_TYPE,
      (leaf) => new NexusApprovalView(leaf, this.edgeStore, this),
    );

    this.addRibbonIcon("link", "Nexus: candidate links", () => this.activateApprovalView());

    this.addCommand({
      id: "nexus-open-approvals",
      name: "Open Nexus approvals",
      callback: () => this.activateApprovalView(),
    });

    this.addCommand({
      id: "nexus-settings",
      name: "Nexus settings",
      callback: () =>
        new NexusSettingsModal(
          this.app,
          this.edgeStore,
          this.jobQueue,
          this.noteRegistry,
        ).open(),
    });

    // Defer vault scan + event subscription until Obsidian finishes its own
    // initial indexing. If we ran `getMarkdownFiles()` straight from `onload`,
    // it returned `[]` before the vault was ready — `reconcile` then dropped
    // every persisted note as an orphan, and Obsidian's backfill `create`
    // events re-queued every file for a cold reprocess. `onLayoutReady` fires
    // after the vault is populated (or immediately if already ready).
    this.app.workspace.onLayoutReady(async () => {
      // One-time repair: edges persisted before the targetId fix were stored
      // with targetId=undefined because getId() was being called with a
      // basename instead of a full path. Rebuild the map from the current
      // vault and repopulate missing ids so approve/deny on legacy edges
      // works again. Also backfills missing sourceId from the map key.
      {
        const allFiles = this.app.vault.getMarkdownFiles() as TFile[];
        const basenameToId = new Map<string, string>();
        for (const f of allFiles) {
          const id = this.noteRegistry.getId(f.path);
          if (id) basenameToId.set(f.basename, id);
        }
        const repaired = this.edgeStore.repairEdgeIds(basenameToId);
        if (repaired > 0) console.log(`Nexus: repaired ids on ${repaired} legacy edge(s)`);
      }

      const { enqueuedPaths } = await initializeFromCache(
        this.app.vault,
        this.noteRegistry,
        this.edgeStore,
        this.jobQueue,
      );
      if (enqueuedPaths.length > 0) {
        this.indexingProgress = {
          toast: new ProgressToast(enqueuedPaths.length, "indexing", "indexing complete."),
          pending: new Set(enqueuedPaths),
        };
      }

      // Subscribe AFTER the cache-reconcile pass so Obsidian's startup
      // `create` events (fired as it indexed the vault) don't get mistaken
      // for genuine user-initiated creates and re-queue every file.
      this.eventListener.subscribe();
    });
  }

  private tickIndexingProgress(path: string): void {
    if (!this.indexingProgress) return;
    if (!this.indexingProgress.pending.delete(path)) return;
    this.indexingProgress.toast.tick();
    if (this.indexingProgress.pending.size === 0) this.indexingProgress = null;
  }

  private wireModelDownloadNotice(): void {
    const provider = this.embeddingProvider as TransformersIframeProvider;
    if (typeof provider.onProgress !== "function") return;

    let notice: Notice | null = null;
    let currentFile: string | null = null;
    provider.onProgress((ev) => {
      if (ev.type === "download") {
        if (!notice) notice = new Notice("Nexus: downloading embedding model…", 0);
        if (ev.file !== currentFile) currentFile = ev.file;
        notice.setMessage(`Nexus: downloading ${currentFile} — ${ev.pct}%`);
      } else if (ev.type === "ready") {
        if (notice) {
          notice.setMessage("Nexus: embedding model ready");
          const n = notice;
          setTimeout(() => n.hide(), 2000);
          notice = null;
        }
      }
    });
  }

  private async activateApprovalView(): Promise<void> {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(APPROVAL_VIEW_TYPE)[0];
    if (!leaf) {
      leaf = workspace.getRightLeaf(false) ?? workspace.getLeaf(true);
      await leaf.setViewState({ type: APPROVAL_VIEW_TYPE, active: true });
    }
    workspace.revealLeaf(leaf);
  }

  async onunload() {
    this.eventListener.unsubscribe();
    this.jobQueue.clear();
    await this.edgeStore.shutdown();
    await this.embeddingProvider.dispose();
    console.log("Nexus: unloaded");
  }
}
