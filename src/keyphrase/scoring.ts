import { ExtractedPhrase } from "../types";

/**
 * Sort candidates by score ascending, take the top N, then min-max normalize
 * scores to [0, 1]. Lower score = more important.
 */
export function normalizeScores(
  candidates: ExtractedPhrase[],
  topN: number,
): ExtractedPhrase[] {
  if (candidates.length === 0) return [];

  const sorted = [...candidates].sort((a, b) => a.score - b.score);
  const top = sorted.slice(0, topN);

  if (top.length === 0) return [];

  const minScore = top[0].score;
  const maxScore = top[top.length - 1].score;
  const range = maxScore - minScore;

  return top.map((p) => ({
    ...p,
    score: range > 0 ? (p.score - minScore) / range : 0,
    spanId: `${p.startOffset}:${p.endOffset}`,
  }));
}
