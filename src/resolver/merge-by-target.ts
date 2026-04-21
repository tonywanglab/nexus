import { CandidateEdge, phraseKey } from "../types";

/**
 * Merges LCS + embedding edges with the same (phraseKey, targetId).
 * On collision, keeps the higher similarity as `similarity`, widens matchType
 * to "both", and preserves both raw scores in lcsSimilarity / embSimilarity.
 * Output is sorted by similarity descending.
 */
export function mergeByTarget(edges: CandidateEdge[]): CandidateEdge[] {
  const map = new Map<string, CandidateEdge>();

  for (const edge of edges) {
    const key = `${phraseKey(edge.phrase.phrase)}|${edge.targetId ?? edge.targetPath}`;
    const existing = map.get(key);

    if (!existing) {
      const entry = { ...edge };
      if (edge.matchType === "deterministic") {
        entry.lcsSimilarity = edge.similarity;
      } else if (edge.matchType === "stochastic") {
        entry.embSimilarity = edge.similarity;
      }
      map.set(key, entry);
      continue;
    }

    // Merge — widen to "both" and keep each resolver's score.
    const lcs =
      edge.matchType === "deterministic" ? edge.similarity : existing.lcsSimilarity;
    const emb =
      edge.matchType === "stochastic" ? edge.similarity : existing.embSimilarity;

    existing.matchType = "both";
    existing.lcsSimilarity = lcs;
    existing.embSimilarity = emb;
    existing.similarity = Math.max(lcs ?? 0, emb ?? 0);
  }

  return [...map.values()].sort((a, b) => b.similarity - a.similarity);
}
