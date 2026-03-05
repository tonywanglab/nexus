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
  /** Lowercased term → number of notes containing the term. */
  documentFrequencies?: Map<string, number>;
  /** Total number of notes in the vault. */
  totalDocuments?: number;
  /** Existing note titles (basenames without .md), original casing. */
  noteTitles?: string[];
}
