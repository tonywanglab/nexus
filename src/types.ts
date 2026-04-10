export type JobType = "process" | "reindex";

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

export interface CandidateEdge {
  sourcePath: string;
  phrase: ExtractedPhrase;
  targetPath: string;
  similarity: number;
  approved?: boolean;
}

export interface VaultContext {
  /** Existing note titles (basenames without .md), original casing. */
  noteTitles?: string[];
  /** Per-term document frequency map (term → number of documents containing it). */
  documentFrequencies?: Map<string, number>;
  /** Total number of documents in the vault. */
  totalDocuments?: number;
}
