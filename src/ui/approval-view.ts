import { ItemView, WorkspaceLeaf, Notice } from "obsidian";
import { CandidateEdge, NoteId, Resolver } from "../types";
import { EdgeStore } from "../edge-store";

export const APPROVAL_VIEW_TYPE = "nexus-approval";

/** Max cards rendered per tab — keeps the panel snappy when a note has many matches. */

export interface CardVM {
  edge: CandidateEdge;
  phraseText: string;
  targetTitle: string;
  similarity: number;
  similarityLabel: string;
  badge: string;
  scoreDetail?: string;
  sourceId: NoteId;
  targetId: NoteId;
}

/**
 * Read contributing resolvers from an edge, falling back to legacy `matchType`
 * for edges persisted before `matchedBy` existed.
 */
export function resolversOf(edge: CandidateEdge): Resolver[] {
  if (edge.matchedBy && edge.matchedBy.length > 0) return edge.matchedBy;
  switch (edge.matchType) {
    case "deterministic": return ["lcs"];
    case "stochastic": return ["dense"];
    case "sparse-feature": return ["sparse"];
    case "both": return ["lcs", "dense"];
    default: return [];
  }
}

const RESOLVER_ORDER: Resolver[] = ["lcs", "dense", "sparse"];

function orderedResolvers(rs: Resolver[]): Resolver[] {
  const set = new Set(rs);
  return RESOLVER_ORDER.filter((r) => set.has(r));
}

export function buildCardViewModel(
  edges: CandidateEdge[],
  fallbackSourceId: NoteId = "",
): CardVM[] {
  return [...edges]
    .sort((a, b) => b.similarity - a.similarity)
    .map((e) => {
      const resolvers = orderedResolvers(resolversOf(e));
      const badge = resolvers.length > 0 ? resolvers.join("+") : "lcs";

      let scoreDetail: string | undefined;
      if (resolvers.length > 1) {
        const parts: string[] = [];
        for (const r of resolvers) {
          const score = r === "lcs" ? e.lcsSimilarity
            : r === "dense" ? e.denseSimilarity
            : e.sparseSimilarity;
          parts.push(`${r} ${score?.toFixed(2) ?? "?"}`);
        }
        scoreDetail = parts.join(" · ");
      }

      return {
        edge: e,
        phraseText: e.phrase.phrase,
        targetTitle: e.targetPath,
        similarity: e.similarity,
        similarityLabel: `${Math.round(e.similarity * 100)}%`,
        badge,
        scoreDetail,
        sourceId: e.sourceId ?? fallbackSourceId,
        targetId: e.targetId ?? "",
      };
    });
}

export type EdgeTab = Resolver;

/**
 * Predicate for the sidebar tab filter. An edge "belongs to" a tab when the
 * tab's resolver contributed to it — so multi-resolver edges show up under
 * every contributing tab rather than being hidden.
 */
export function matchesTab(tab: EdgeTab): (e: CandidateEdge) => boolean {
  return (e) => resolversOf(e).includes(tab);
}

export class NexusApprovalView extends ItemView {
  private store: EdgeStore;
  private plugin: any; // Plugin reference for vault access
  private unsub: (() => void) | null = null;
  private activeFilePath: string | null = null;
  private pendingApprovals = new Map<string, Promise<void>>();
  private activeTab: EdgeTab = "lcs";
  private renderRaf: number | null = null;

  constructor(leaf: WorkspaceLeaf, store: EdgeStore, plugin: any) {
    super(leaf);
    this.store = store;
    this.plugin = plugin;
  }

  getViewType(): string {
    return APPROVAL_VIEW_TYPE;
  }

  getDisplayText(): string {
    return "Nexus: candidate links";
  }

  getIcon(): string {
    return "link";
  }

  async onOpen(): Promise<void> {
    this.unsub = this.store.subscribe(() => {
      // Debounce via rAF: batch rapid store writes (e.g. bulk indexing) into a
      // single render, keeping the UI responsive during background processing.
      if (this.renderRaf !== null) cancelAnimationFrame(this.renderRaf);
      this.renderRaf = requestAnimationFrame(() => {
        this.renderRaf = null;
        this.render();
      });
    });

    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => {
        const active = this.app.workspace.getActiveFile();
        this.activeFilePath = active?.path ?? null;
        // Cancel any pending rAF so we render immediately for the new active file.
        if (this.renderRaf !== null) {
          cancelAnimationFrame(this.renderRaf);
          this.renderRaf = null;
        }
        this.render();
      }),
    );

    const active = this.app.workspace.getActiveFile();
    this.activeFilePath = active?.path ?? null;
    this.render();
  }

  async onClose(): Promise<void> {
    this.unsub?.();
    this.unsub = null;
    if (this.renderRaf !== null) {
      cancelAnimationFrame(this.renderRaf);
      this.renderRaf = null;
    }
  }

  private render(): void {
    const el = this.contentEl;
    el.empty();

    const registry = (this.plugin as any).noteRegistry;
    const activeId = this.activeFilePath && registry
      ? registry.getId(this.activeFilePath)
      : null;

    if (!activeId) {
      const isMarkdown = this.activeFilePath?.endsWith(".md") ?? false;
      el.createEl("p", {
        text: isMarkdown
          ? "Indexing this note…"
          : "Open a note to see candidate links.",
        cls: "nexus-empty",
      });
      return;
    }

    this.renderToolbar(el);

    const edges = this.store.getEdgesForFile(activeId);
    const lcsCount = edges.filter(matchesTab("lcs")).length;
    const denseCount = edges.filter(matchesTab("dense")).length;
    const sparseCount = edges.filter(matchesTab("sparse")).length;
    this.renderTabs(el, lcsCount, denseCount, sparseCount);

    const filtered = edges.filter(matchesTab(this.activeTab));
    const cards = buildCardViewModel(filtered, activeId);

    if (cards.length === 0) {
      const emptyText: Record<EdgeTab, string> = {
        lcs: "No LCS candidates for this note.",
        dense: "No dense embedding candidates for this note.",
        sparse: "No sparse candidates for this note.",
      };
      el.createEl("p", {
        text: emptyText[this.activeTab] ?? "No candidates for this note.",
        cls: "nexus-empty",
      });
      return;
    }

    const list = el.createEl("div", { cls: "nexus-card-list" });

    for (const vm of cards) {
      this.renderCard(list, vm);
    }
  }

  private renderTabs(container: any, lcsCount: number, denseCount: number, sparseCount: number): void {
    const tabs = container.createEl("div", { cls: "nexus-tabs" });
    const mk = (tab: EdgeTab, label: string, count: number) => {
      const active = this.activeTab === tab;
      const btn = tabs.createEl("button", {
        text: `${label} (${count})`,
        cls: `nexus-tab${active ? " nexus-tab-active" : ""}`,
      });
      btn.addEventListener("click", () => {
        if (this.activeTab === tab) return;
        this.activeTab = tab;
        this.render();
      });
    };
    mk("lcs", "LCS", lcsCount);
    mk("dense", "Dense", denseCount);
    mk("sparse", "Sparse", sparseCount);
  }

  private renderToolbar(container: any): void {
    const toolbar = container.createEl("div", { cls: "nexus-toolbar" });
    const reindexBtn = toolbar.createEl("button", {
      text: "Reindex this note",
      cls: "nexus-btn nexus-reindex",
    });
    reindexBtn.addEventListener("click", () => {
      const path = this.activeFilePath;
      if (!path) return;
      const registry = (this.plugin as any).noteRegistry;
      const id = registry?.getId(path);
      if (id) this.store.clearEdgesForFile(id);
      this.plugin.jobQueue?.forceEnqueue(path, "process");
      new Notice("Nexus: reindexing — re-extracting phrases, running LCS + embeddings");
    });
  }

  private renderCard(container: any, vm: CardVM): void {
    const card = container.createEl("div", { cls: "nexus-card" });

    const header = card.createEl("div", { cls: "nexus-card-header" });
    header.createEl("span", { text: vm.phraseText, cls: "nexus-phrase" });
    header.createEl("span", { text: " → ", cls: "nexus-arrow" });
    header.createEl("span", { text: vm.targetTitle, cls: "nexus-target" });

    const meta = card.createEl("div", { cls: "nexus-card-meta" });
    meta.createEl("span", { text: vm.badge, cls: `nexus-badge nexus-badge-${vm.badge.replace(/\+/g, "-")}` });
    meta.createEl("span", { text: vm.similarityLabel, cls: "nexus-similarity" });
    if (vm.scoreDetail) {
      meta.createEl("span", { text: vm.scoreDetail, cls: "nexus-score-detail" });
    }

    if (vm.edge.sparseFeatures) {
      this.renderSparseFeatures(card, vm.edge.sparseFeatures);
    }

    const actions = card.createEl("div", { cls: "nexus-card-actions" });

    const approveBtn = actions.createEl("button", { text: "Approve", cls: "nexus-btn nexus-approve" });
    approveBtn.addEventListener("click", () => this.handleApprove(vm));

    const denyBtn = actions.createEl("button", { text: "Deny", cls: "nexus-btn nexus-deny" });
    denyBtn.addEventListener("click", () => this.handleDeny(vm));
  }

  private renderSparseFeatures(
    card: any,
    sparseFeatures: NonNullable<CandidateEdge["sparseFeatures"]>,
  ): void {
    // Collapse the phrase/title payloads into the features that fire on both
    // sides — the interpretable "why these two match" signal. Rank by pVal*tVal
    // (the feature's literal contribution to sparseCosine), and cap at 2 chips.
    const phraseByIdx = new Map<number, { value: number; label: string }>();
    for (const f of sparseFeatures.phraseFeatures) {
      phraseByIdx.set(f.idx, { value: f.value, label: f.label });
    }
    const shared: { idx: number; label: string; score: number }[] = [];
    for (const tf of sparseFeatures.titleFeatures) {
      const pf = phraseByIdx.get(tf.idx);
      if (!pf) continue;
      shared.push({ idx: tf.idx, label: pf.label, score: pf.value * tf.value });
    }
    if (shared.length === 0) return;
    shared.sort((a, b) => b.score - a.score);
    const top = shared.slice(0, 2);

    const el = card.createEl("div", { cls: "nexus-sparse-features" });
    const chips = el.createEl("div", { cls: "nexus-sparse-chips" });
    for (const f of top) {
      chips.createEl("span", { text: f.label, cls: "nexus-sparse-chip nexus-sparse-chip-match" });
    }
  }

  private handleApprove(vm: CardVM): void {
    // Import dynamically to avoid circular deps at module load time.
    import("./wikilink-insert").then(({ buildWikilinkReplacement }) => {
      const path = this.activeFilePath;
      if (!path) return;

      const key = path;
      const prev = this.pendingApprovals.get(key) ?? Promise.resolve();
      const next = prev.then(async () => {
        const vault = this.app.vault as any;
        const file = vault.getAbstractFileByPath(path);
        if (!file) return;

        const content = await vault.read(file);
        const result = buildWikilinkReplacement(content, vm.edge.phrase, vm.targetTitle);

        if (result.reason === "already-linked") {
          this.store.approveEdge(vm.sourceId, vm.phraseText, vm.targetId);
          return;
        }

        if (result.replaced) {
          await vault.modify(file, result.content);
          this.store.approveEdge(vm.sourceId, vm.phraseText, vm.targetId);
        } else {
          // Span drifted — re-scan and surface again after reprocess.
          this.store.removeEdge(vm.sourceId, vm.phraseText, vm.targetId);
          this.plugin.jobQueue?.forceEnqueue(path, "process");
          new Notice("Phrase moved — rescanning…");
        }
      });

      this.pendingApprovals.set(key, next.catch(() => {}));
    });
  }

  private handleDeny(vm: CardVM): void {
    console.log(
      `Nexus: deny click — sourceId=${vm.sourceId || "(empty)"} target=${vm.targetTitle} targetId=${vm.targetId || "(empty)"} phrase="${vm.phraseText}"`,
    );
    this.store.denyEdge(vm.sourceId, vm.phraseText, vm.targetId);
  }
}
