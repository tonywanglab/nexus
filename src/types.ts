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

export type Resolver = "lcs" | "dense" | "sparse";

// legacy single-resolver tag. Retained so edges persisted before `matchedBy`
// existed still round-trip. New code should read `matchedBy` instead.
export type MatchType = "deterministic" | "stochastic" | "sparse-feature" | "both" | "mixed";

export interface CandidateEdge {
  sourcePath: string;
  sourceId?: NoteId;
  phrase: ExtractedPhrase;
  targetPath: string;
  targetId?: NoteId;
  similarity: number;
  //  Set of resolvers that contributed to this edge.
  matchedBy?: Resolver[];
  //  Per-resolver scores. Populated when the corresponding resolver contributed.
  lcsSimilarity?: number;
  denseSimilarity?: number;
  sparseSimilarity?: number;
  approved?: boolean;
  matchType?: MatchType;
  //  Top-4 labeled SAE features per side. Populated only when sparse resolver contributed.
  sparseFeatures?: {
    phraseFeatures: { idx: number; value: number; label: string }[];
    titleFeatures:  { idx: number; value: number; label: string }[];
  };
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
  labelOverrides?: Record<number, string>;
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
    labelOverrides: {},
  };
}

// canonicalizes a phrase string for denial/approval key matching.
// lowercase → trim → collapse internal whitespace.
export function phraseKey(phrase: string): string {
  return phrase.toLowerCase().trim().replace(/\s+/g, " ");
}

export interface VaultContext {
  //  Existing note titles (basenames without .md), original casing.
  noteTitles?: string[];
  //  Per-term document frequency map (term → number of documents containing it).
  documentFrequencies?: Map<string, number>;
  //  Total number of documents in the vault.
  totalDocuments?: number;
}
