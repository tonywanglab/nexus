import { ItemView, WorkspaceLeaf, Notice } from "obsidian";
import { CandidateEdge, NoteId } from "../types";
import { EdgeStore } from "../edge-store";

export const APPROVAL_VIEW_TYPE = "nexus-approval";

export interface CardVM {
  edge: CandidateEdge;
  phraseText: string;
  targetTitle: string;
  similarity: number;
  similarityLabel: string;
  badge: "lcs" | "emb" | "lcs+emb";
  scoreDetail?: string;
  sourceId: NoteId;
  targetId: NoteId;
}

export function buildCardViewModel(
  edges: CandidateEdge[],
  fallbackSourceId: NoteId = "",
): CardVM[] {
  return [...edges]
    .sort((a, b) => b.similarity - a.similarity)
    .map((e) => {
      let badge: CardVM["badge"];
      let scoreDetail: string | undefined;

      if (e.matchType === "both") {
        badge = "lcs+emb";
        const lcs = e.lcsSimilarity?.toFixed(2) ?? "?";
        const emb = e.embSimilarity?.toFixed(2) ?? "?";
        scoreDetail = `lcs ${lcs} · emb ${emb}`;
      } else if (e.matchType === "stochastic") {
        badge = "emb";
      } else {
        badge = "lcs";
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

export type EdgeTab = "lcs" | "emb";

/**
 * Predicate for the sidebar tab filter. An edge "belongs to" a tab when the
 * tab's resolver contributed to it — so `both` edges show up under both tabs
 * rather than being hidden.
 */
export function matchesTab(tab: EdgeTab): (e: CandidateEdge) => boolean {
  return (e) => {
    if (tab === "lcs") return e.matchType === "deterministic" || e.matchType === "both";
    return e.matchType === "stochastic" || e.matchType === "both";
  };
}

export class NexusApprovalView extends ItemView {
  private store: EdgeStore;
  private plugin: any; // Plugin reference for vault access
  private unsub: (() => void) | null = null;
  private activeFilePath: string | null = null;
  private pendingApprovals = new Map<string, Promise<void>>();
  private activeTab: EdgeTab = "lcs";

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
    this.unsub = this.store.subscribe(() => this.render());

    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => {
        const active = this.app.workspace.getActiveFile();
        this.activeFilePath = active?.path ?? null;
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
  }

  private render(): void {
    const el = this.contentEl;
    el.empty();

    const registry = (this.plugin as any).noteRegistry;
    const activeId = this.activeFilePath && registry
      ? registry.getId(this.activeFilePath)
      : null;

    if (!activeId) {
      el.createEl("p", { text: "Open a note to see candidate links.", cls: "nexus-empty" });
      return;
    }

    this.renderToolbar(el);

    const edges = this.store.getEdgesForFile(activeId);
    const lcsCount = edges.filter(matchesTab("lcs")).length;
    const embCount = edges.filter(matchesTab("emb")).length;
    this.renderTabs(el, lcsCount, embCount);

    const filtered = edges.filter(matchesTab(this.activeTab));
    const cards = buildCardViewModel(filtered, activeId);

    if (cards.length === 0) {
      el.createEl("p", {
        text:
          this.activeTab === "lcs"
            ? "No LCS candidates for this note."
            : "No embedding candidates for this note.",
        cls: "nexus-empty",
      });
      return;
    }

    const list = el.createEl("div", { cls: "nexus-card-list" });

    for (const vm of cards) {
      this.renderCard(list, vm);
    }
  }

  private renderTabs(container: any, lcsCount: number, embCount: number): void {
    const tabs = container.createEl("div", { cls: "nexus-tabs" });
    tabs.style.display = "flex";
    tabs.style.gap = "4px";
    tabs.style.marginBottom = "8px";
    const mk = (tab: EdgeTab, label: string, count: number) => {
      const active = this.activeTab === tab;
      const btn = tabs.createEl("button", {
        text: `${label} (${count})`,
        cls: `nexus-tab${active ? " nexus-tab-active" : ""}`,
      });
      if (active) {
        btn.style.fontWeight = "600";
        btn.style.borderBottom = "2px solid var(--interactive-accent)";
      }
      btn.addEventListener("click", () => {
        if (this.activeTab === tab) return;
        this.activeTab = tab;
        this.render();
      });
    };
    mk("lcs", "LCS", lcsCount);
    mk("emb", "Embeddings", embCount);
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
    meta.createEl("span", { text: vm.badge, cls: `nexus-badge nexus-badge-${vm.badge}` });
    meta.createEl("span", { text: vm.similarityLabel, cls: "nexus-similarity" });
    if (vm.scoreDetail) {
      meta.createEl("span", { text: vm.scoreDetail, cls: "nexus-score-detail" });
    }

    const actions = card.createEl("div", { cls: "nexus-card-actions" });

    const approveBtn = actions.createEl("button", { text: "Approve", cls: "nexus-btn nexus-approve" });
    approveBtn.addEventListener("click", () => this.handleApprove(vm));

    const denyBtn = actions.createEl("button", { text: "Deny", cls: "nexus-btn nexus-deny" });
    denyBtn.addEventListener("click", () => this.handleDeny(vm));
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
