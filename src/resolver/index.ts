import { CandidateEdge, ExtractedPhrase } from "../types";

/**
 * Maps extracted keyphrases to existing note titles using
 * deterministic (LCS) and stochastic (embeddings) methods.
 */
export class AliasResolver {
  resolve(_phrases: ExtractedPhrase[], _noteTitles: string[]): CandidateEdge[] {
    // TODO: implement alias resolution pipeline
    return [];
  }
}
