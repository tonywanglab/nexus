export type JobType = "process";

export interface QueueJob {
  filePath: string;
  type: JobType;
  priority: "high" | "normal";
  enqueuedAt: number;
}

export interface ExtractedPhrase {
  phrase: string;
  score: number;
  startOffset: number;
  endOffset: number;
  spanId: string;
}

export type NoteId = string;

export type MatchType = "deterministic" | "stochastic" | "both";

export interface CandidateEdge {
  sourcePath: string;
  sourceId?: NoteId;
  phrase: ExtractedPhrase;
  targetPath: string;
  targetId?: NoteId;
  similarity: number;
  /** Per-resolver scores, populated when matchType === "both". */
  lcsSimilarity?: number;
  embSimilarity?: number;
  approved?: boolean;
  matchType?: MatchType;
}

export interface DenialRecord {
  sourceId: NoteId;
  phraseKey: string;
  targetId: NoteId;
}

export interface ApprovalRecord {
  sourceId: NoteId;
  phraseKey: string;
  targetId: NoteId;
  approvedAt: number;
}

export interface NoteMeta {
  path: string;
  mtime: number;
}

export interface PersistedState {
  version: 1;
  similarityThreshold: number;
  notes: Record<NoteId, NoteMeta>;
  edges: Record<NoteId, CandidateEdge[]>;
  denials: DenialRecord[];
  approvals: ApprovalRecord[];
}

export const DEFAULT_SIMILARITY_THRESHOLD = 0.7;

export function createEmptyPersistedState(): PersistedState {
  return {
    version: 1,
    similarityThreshold: DEFAULT_SIMILARITY_THRESHOLD,
    notes: {},
    edges: {},
    denials: [],
    approvals: [],
  };
}

/**
 * Canonicalizes a phrase string for denial/approval key matching.
 * lowercase → trim → collapse internal whitespace.
 */
export function phraseKey(phrase: string): string {
  return phrase.toLowerCase().trim().replace(/\s+/g, " ");
}

export interface VaultContext {
  /** Existing note titles (basenames without .md), original casing. */
  noteTitles?: string[];
  /** Per-term document frequency map (term → number of documents containing it). */
  documentFrequencies?: Map<string, number>;
  /** Total number of documents in the vault. */
  totalDocuments?: number;
}
