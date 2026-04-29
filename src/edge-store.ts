import {
  CandidateEdge,
  NoteId,
  PersistedState,
  DenialRecord,
  ApprovalRecord,
  phraseKey,
} from "./types";
import { serialize } from "./persistence";

type Listener = () => void;

const PERSIST_DEBOUNCE_MS = 300;
// Hard ceiling on how long we'll coalesce writes — guarantees persistence
// progresses during long-running bulk indexing, where per-file completions
// would otherwise keep resetting the debounce timer indefinitely.
const PERSIST_MAX_WAIT_MS = 5000;

export class EdgeStore {
  private edges = new Map<NoteId, CandidateEdge[]>();
  private mtimes = new Map<NoteId, number>();
  private denials: DenialRecord[] = [];
  private approvals: ApprovalRecord[] = [];
  private threshold: number;
  private listeners = new Set<Listener>();
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private maxWaitTimer: ReturnType<typeof setTimeout> | null = null;
  private inFlightPersist: Promise<void> | null = null;
  private onPersist: (state: PersistedState) => Promise<void> | void;

  constructor(initial: PersistedState, onPersist: (state: PersistedState) => Promise<void> | void) {
    this.threshold = initial.similarityThreshold;
    this.onPersist = onPersist;
    this.denials = [...initial.denials];
    this.approvals = [...initial.approvals];
    for (const [id, edgeList] of Object.entries(initial.edges)) {
      this.edges.set(id, [...edgeList]);
    }
    for (const [id, meta] of Object.entries(initial.notes)) {
      this.mtimes.set(id, meta.mtime);
    }
  }

  // ── read ─────────────────────────────────────────────────────

  getEdgesForFile(sourceId: NoteId): CandidateEdge[] {
    const stored = this.edges.get(sourceId) ?? [];
    return stored
      .filter((e) => e.similarity >= this.threshold)
      .sort((a, b) => b.similarity - a.similarity);
  }

  getMtime(sourceId: NoteId): number | null {
    return this.mtimes.get(sourceId) ?? null;
  }

  getThreshold(): number {
    return this.threshold;
  }

  isDenied(sourceId: NoteId, phrase: string, targetId: NoteId): boolean {
    const key = phraseKey(phrase);
    return this.denials.some(
      (d) => d.sourceId === sourceId && d.phraseKey === key && d.targetId === targetId,
    );
  }

  isApproved(sourceId: NoteId, phrase: string, targetId: NoteId): boolean {
    const key = phraseKey(phrase);
    return this.approvals.some(
      (a) => a.sourceId === sourceId && a.phraseKey === key && a.targetId === targetId,
    );
  }

  allSourceIds(): NoteId[] {
    return [...this.mtimes.keys()];
  }

  hasEdgesFor(sourceId: NoteId): boolean {
    const list = this.edges.get(sourceId);
    return !!list && list.length > 0;
  }

  /**
   * One-time repair for edges persisted before sourceId/targetId annotation
   * was fixed. Walks every stored edge and backfills missing ids — sourceId
   * from the map key the edge lives under, targetId via the provided
   * basename→id map. Returns the count of edges that were updated.
   */
  repairEdgeIds(basenameToId: Map<string, NoteId>): number {
    let repaired = 0;
    for (const [sourceId, edgeList] of this.edges) {
      let changed = false;
      const next = edgeList.map((e) => {
        const needsSource = !e.sourceId;
        const needsTarget = !e.targetId;
        if (!needsSource && !needsTarget) return e;
        const patch: Partial<CandidateEdge> = {};
        if (needsSource) patch.sourceId = sourceId;
        if (needsTarget) {
          const id = basenameToId.get(e.targetPath);
          if (id) patch.targetId = id;
        }
        if (!("sourceId" in patch) && !("targetId" in patch)) return e;
        changed = true;
        repaired++;
        return { ...e, ...patch };
      });
      if (changed) this.edges.set(sourceId, next);
    }
    if (repaired > 0) {
      this.notify();
      this.schedulePersist();
    }
    return repaired;
  }

  // ── write ────────────────────────────────────────────────────

  setEdgesForFile(sourceId: NoteId, edges: CandidateEdge[], mtime: number): void {
    const filtered = edges.filter((e) => !this.isDeniedEdge(e) && !this.isApprovedEdge(e));
    this.edges.set(sourceId, filtered);
    this.mtimes.set(sourceId, mtime);
    this.notify();
    this.schedulePersist();
  }

  /**
   * Intermediate write used while a file's pipeline is still running. Stores
   * edges and notifies listeners but does NOT bump mtime — so if the process
   * is killed before the final write, the file is still eligible for retry on
   * next load.
   */
  setInterimEdgesForFile(sourceId: NoteId, edges: CandidateEdge[]): void {
    const filtered = edges.filter((e) => !this.isDeniedEdge(e) && !this.isApprovedEdge(e));
    this.edges.set(sourceId, filtered);
    this.notify();
    this.schedulePersist();
  }

  removeEdge(sourceId: NoteId, phrase: string, targetId: NoteId): void {
    const key = phraseKey(phrase);
    const current = this.edges.get(sourceId) ?? [];
    const next = current.filter(
      (e) =>
        !(
          phraseKey(e.phrase.phrase) === key &&
          (e.targetId === targetId || (!e.targetId && !targetId))
        ),
    );
    if (next.length === current.length) {
      console.warn(
        `Nexus: removeEdge found no match — sourceId=${sourceId || "(empty)"} targetId=${targetId || "(empty)"} phrase="${phrase}" (edges here: ${current.length})`,
      );
    }
    this.edges.set(sourceId, next);
    this.notify();
    this.schedulePersist();
  }

  denyEdge(sourceId: NoteId, phrase: string, targetId: NoteId): void {
    const key = phraseKey(phrase);
    if (!this.isDenied(sourceId, phrase, targetId)) {
      this.denials.push({ sourceId, phraseKey: key, targetId });
    }
    this.removeEdge(sourceId, phrase, targetId);
  }

  approveEdge(sourceId: NoteId, phrase: string, targetId: NoteId): void {
    const key = phraseKey(phrase);
    if (!this.isApproved(sourceId, phrase, targetId)) {
      this.approvals.push({ sourceId, phraseKey: key, targetId, approvedAt: Date.now() });
    }
    this.removeEdge(sourceId, phrase, targetId);
  }

  setThresholdAndPrune(value: number): void {
    this.threshold = value;
    for (const [id, edgeList] of this.edges) {
      this.edges.set(id, edgeList.filter((e) => e.similarity >= value));
    }
    this.notify();
    this.schedulePersist();
  }

  setThresholdNoPrune(value: number): void {
    this.threshold = value;
    this.schedulePersist();
  }

  /**
   * Clears the edge list for a single file without touching the registry or
   * denials/approvals. Used by the "reindex this note" UI action to force a
   * clean rebuild (so the subsequent interim det write is visible immediately).
   */
  clearEdgesForFile(sourceId: NoteId): void {
    if (!this.edges.has(sourceId)) return;
    this.edges.delete(sourceId);
    this.mtimes.delete(sourceId);
    this.notify();
    this.schedulePersist();
  }

  dropNote(id: NoteId): void {
    // Remove as source
    this.edges.delete(id);
    this.mtimes.delete(id);
    // Remove as target from all edge lists
    for (const [srcId, edgeList] of this.edges) {
      const filtered = edgeList.filter((e) => e.targetId !== id);
      if (filtered.length !== edgeList.length) this.edges.set(srcId, filtered);
    }
    this.notify();
    this.schedulePersist();
  }

  // ── subscriptions ────────────────────────────────────────────

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Triggers a view refresh without changing underlying data. */
  refreshViews(): void {
    this.notify();
  }

  // ── lifecycle ────────────────────────────────────────────────

  async shutdown(): Promise<void> {
    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    if (this.maxWaitTimer !== null) {
      clearTimeout(this.maxWaitTimer);
      this.maxWaitTimer = null;
    }
    if (this.inFlightPersist) await this.inFlightPersist;
    await this.flush();
  }

  // ── private ──────────────────────────────────────────────────

  private isDeniedEdge(e: CandidateEdge): boolean {
    if (!e.sourceId || !e.targetId) return false;
    return this.isDenied(e.sourceId, e.phrase.phrase, e.targetId);
  }

  private isApprovedEdge(e: CandidateEdge): boolean {
    if (!e.sourceId || !e.targetId) return false;
    return this.isApproved(e.sourceId, e.phrase.phrase, e.targetId);
  }

  private notify(): void {
    for (const l of this.listeners) l();
  }

  private schedulePersist(): void {
    if (this.debounceTimer !== null) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      if (this.maxWaitTimer !== null) {
        clearTimeout(this.maxWaitTimer);
        this.maxWaitTimer = null;
      }
      void this.flush();
    }, PERSIST_DEBOUNCE_MS);
    // Arm the max-wait timer only on the first write of a debounce window, so
    // continuous writes can't keep pushing persistence past this ceiling.
    if (this.maxWaitTimer === null) {
      this.maxWaitTimer = setTimeout(() => {
        this.maxWaitTimer = null;
        if (this.debounceTimer !== null) {
          clearTimeout(this.debounceTimer);
          this.debounceTimer = null;
        }
        void this.flush();
      }, PERSIST_MAX_WAIT_MS);
    }
  }

  private async flush(): Promise<void> {
    if (this.inFlightPersist) await this.inFlightPersist;
    const result = this.onPersist(this.toState());
    this.inFlightPersist = Promise.resolve(result).then(
      () => {
        this.inFlightPersist = null;
      },
      (err) => {
        this.inFlightPersist = null;
        console.warn("Nexus: persist failed:", err);
      },
    );
    await this.inFlightPersist;
  }

  private toState(): PersistedState {
    const notes: PersistedState["notes"] = {};
    for (const [id, mtime] of this.mtimes) {
      notes[id] = { path: "", mtime }; // path filled by caller via NoteRegistry
    }
    const edges: PersistedState["edges"] = {};
    for (const [id, edgeList] of this.edges) {
      edges[id] = edgeList;
    }
    return serialize({
      version: 1,
      similarityThreshold: this.threshold,
      notes,
      edges,
      denials: [...this.denials],
      approvals: [...this.approvals],
    }) as PersistedState;
  }
}
