import { CandidateEdge } from "../types";

/**
 * Deduplicates candidate edges by target (keeping highest similarity)
 * and ranks by combined score: similarity × (1 - phraseScore).
 */
export function dedupAndRank(candidates: CandidateEdge[]): CandidateEdge[] {
  const bestByTarget = new Map<string, CandidateEdge>();
  for (const edge of candidates) {
    const existing = bestByTarget.get(edge.targetPath);
    if (!existing || edge.similarity > existing.similarity) {
      bestByTarget.set(edge.targetPath, edge);
    }
  }
  const deduped = Array.from(bestByTarget.values());
  deduped.sort((a, b) => {
    const scoreA = a.similarity * (1 - a.phrase.score);
    const scoreB = b.similarity * (1 - b.phrase.score);
    return scoreB - scoreA;
  });
  return deduped;
}
