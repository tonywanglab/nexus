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
import { serializeEmbeddings, deserializeEmbeddings } from "./embedding-persistence";

/**
 * Phrases whose LCS match against some title is this strong or higher can skip
 * dense/sparse embedding — the lexical match alone is decisive enough that the
 * extra compute rarely surfaces new targets worth showing.
 */
const STRONG_LCS_MATCH = 0.95;
const MAX_EMBED_PHRASES = 60;
const TITLE_EMBEDDINGS_FILENAME = "title-embeddings.bin";
const TITLE_EMBEDDINGS_SAVE_DEBOUNCE_MS = 3000;

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
  private titleEmbeddingsSaveTimer: ReturnType<typeof setTimeout> | null = null;
  private titleEmbeddingsDim: number | null = null;

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
    // Fire-and-forget warmup so the model starts downloading immediately rather
    // than on the first user-triggered embedding request.
    void this.embeddingProvider.embed("warmup").catch((err) => {
      console.warn("Nexus: embedding warmup failed", err);
    });
    const sae = loadBundledSAE();
    this.featureLabels = new SAEFeatureLabels(sae.dHidden);
    this.featureLabels.setOverrides(persisted.labelOverrides ?? {});
    console.log(
      `Nexus: loaded SAE (d_model=${sae.dModel}, d_hidden=${sae.dHidden}, k=${sae.k}), feature labels: ${this.featureLabels.liveCount}/${sae.dHidden} live`,
    );
    this.embeddingResolver = new EmbeddingResolver({
      embeddingProvider: this.embeddingProvider,
      sae,
      featureLabels: this.featureLabels,
      onTitleEmbeddingsChanged: () => this.scheduleTitleEmbeddingsSave(),
    });
    // Load persisted title embeddings; if none exist yet, first indexing pass
    // will populate them. Failures are non-fatal (we'll just re-embed).
    await this.loadTitleEmbeddings();

    this.jobQueue = new JobQueue(async (job) => {
      // Yield once before starting heavy work — the debounce timer fires
      // synchronously, so without this the full pipeline runs in the same
      // macrotask and delays the next paint the editor queued for a keystroke.
      await new Promise<void>((r) => setTimeout(r));

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
      const phrases = await this.extractor.extract(content, vaultContext);
      console.log(`Nexus:   extracted ${phrases.length} phrase(s)`);

      if (phrases.length === 0) {
        this.edgeStore.setEdgesForFile(sourceId, [], file.stat.mtime);
        console.log(`Nexus: done ${job.filePath} (0 edges, ${Math.round(performance.now() - t0)}ms)`);
        this.tickIndexingProgress(job.filePath);
        return;
      }

      const detEdges = await this.resolver.resolve(phrases, noteTitles, job.filePath);
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

      // Skip embedding for phrases that already have a near-perfect LCS match —
      // dense/sparse rarely surface anything new for those, and embedding cost
      // dominates the pipeline. Phrases with no LCS match or only weak matches
      // still go through so dense can find semantically related targets.
      const stronglyMatched = new Set<string>();
      for (const e of detEdges) {
        if (e.similarity >= STRONG_LCS_MATCH) stronglyMatched.add(e.phrase.phrase);
      }
      const phrasesFiltered = stronglyMatched.size === 0
        ? phrases
        : phrases.filter((p) => !stronglyMatched.has(p.phrase));
      if (stronglyMatched.size > 0) {
        console.log(
          `Nexus:   skipping embedding for ${stronglyMatched.size} phrase(s) with strong LCS match (>=${STRONG_LCS_MATCH})`,
        );
      }
      const cap = MAX_EMBED_PHRASES;
      const phrasesForEmbedding = phrasesFiltered.length > cap
        ? phrasesFiltered.slice().sort((a, b) => a.phrase.length - b.phrase.length).slice(0, cap)
        : phrasesFiltered;
      if (phrasesFiltered.length > cap) {
        console.log(`Nexus:   capped embedding phrases ${phrasesFiltered.length} → ${cap}`);
      }

      try {
        const embT0 = performance.now();
        const embEdges = phrasesForEmbedding.length === 0
          ? []
          : await this.embeddingResolver.resolve(
              phrasesForEmbedding,
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

        // Persist det+dense now so the UI reflects dense edges even if sparse is
        // still running or gets interrupted. mtime is not bumped — sparse will do
        // the final write with mtime, keeping the file eligible for retry if killed.
        this.edgeStore.setInterimEdgesForFile(
          sourceId,
          mergeByTarget([...annotated, ...annotatedEmb]),
        );

        let annotatedSparse: typeof annotatedEmb = [];
        try {
          const sparseT0 = performance.now();
          const sparseEdges = phrasesForEmbedding.length === 0
            ? []
            : await this.embeddingResolver.resolveBySparseFeatures(
                phrasesForEmbedding,
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
        // Drop background normal-priority pending jobs so the active file's job
        // cuts ahead rather than waiting behind stale background work.
        if (path) this.jobQueue.cancelNormalExcept(path);
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

      // Drop persisted title embeddings for titles that no longer exist
      // (renames, deletions). Triggers a debounced save if anything changed.
      {
        const activeTitles = new Set(
          (this.app.vault.getMarkdownFiles() as TFile[]).map((f) => f.basename),
        );
        const removed = this.embeddingResolver.pruneTitleCacheTo(activeTitles);
        if (removed > 0) console.log(`Nexus: pruned ${removed} stale title embedding(s)`);
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

  private titleEmbeddingsFilePath(): string {
    return `${this.manifest.dir}/${TITLE_EMBEDDINGS_FILENAME}`;
  }

  private async loadTitleEmbeddings(): Promise<void> {
    const path = this.titleEmbeddingsFilePath();
    try {
      const adapter = this.app.vault.adapter;
      if (!(await adapter.exists(path))) return;
      const buf = await adapter.readBinary(path);
      const { dim, map } = deserializeEmbeddings(buf);
      this.titleEmbeddingsDim = dim;
      this.embeddingResolver.seedTitleEmbeddings(map);
      console.log(`Nexus: loaded ${map.size} persisted title embedding(s) dim=${dim}`);
    } catch (err) {
      console.warn("Nexus: failed to load title embeddings, continuing without cache:", err);
    }
  }

  private scheduleTitleEmbeddingsSave(): void {
    if (this.titleEmbeddingsSaveTimer !== null) clearTimeout(this.titleEmbeddingsSaveTimer);
    this.titleEmbeddingsSaveTimer = setTimeout(() => {
      this.titleEmbeddingsSaveTimer = null;
      void this.flushTitleEmbeddings();
    }, TITLE_EMBEDDINGS_SAVE_DEBOUNCE_MS);
  }

  private async flushTitleEmbeddings(): Promise<void> {
    const map = this.embeddingResolver.exportTitleEmbeddings();
    if (map.size === 0) return;
    try {
      const dim = this.titleEmbeddingsDim
        ?? (map.values().next().value as Float32Array | undefined)?.length
        ?? 0;
      if (dim === 0) return;
      this.titleEmbeddingsDim = dim;
      const buf = serializeEmbeddings(map, dim);
      await this.app.vault.adapter.writeBinary(this.titleEmbeddingsFilePath(), buf);
    } catch (err) {
      console.warn("Nexus: failed to persist title embeddings:", err);
    }
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

  async saveLabelOverride(idx: number, label: string): Promise<void> {
    this.featureLabels.setOverride(idx, label);
    const raw = await this.loadData();
    const state = deserialize(raw);
    if (!state.labelOverrides) state.labelOverrides = {};
    (state.labelOverrides as any)[idx] = label;
    await this.saveData(serialize(state));
    this.edgeStore.refreshViews();
  }

  async onunload() {
    this.eventListener.unsubscribe();
    this.jobQueue.clear();
    if (this.titleEmbeddingsSaveTimer !== null) {
      clearTimeout(this.titleEmbeddingsSaveTimer);
      this.titleEmbeddingsSaveTimer = null;
    }
    await this.flushTitleEmbeddings();
    await this.edgeStore.shutdown();
    await this.embeddingProvider.dispose();
    console.log("Nexus: unloaded");
  }
}
