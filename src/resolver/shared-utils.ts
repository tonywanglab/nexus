import { CandidateEdge, phraseKey } from "../types";

// time-budget yielder for long synchronous loops. Returns a function that
// awaits `setTimeout(0)` only when more than `budgetMs` has elapsed since
// the last yield — so warm-cache loops pay ~nothing while heavy CPU loops
// break into UI-paintable chunks. `setTimeout` (macrotask) is required;
// `Promise.resolve()` (microtask) does not let the browser paint.
export function makeYielder(budgetMs = 16): () => Promise<void> {
  let lastYield = performance.now();
  return async () => {
    if (performance.now() - lastYield > budgetMs) {
      await new Promise<void>((r) => setTimeout(r));
      lastYield = performance.now();
    }
  };
}

// deduplicates candidate edges by (phrase, target) keeping highest similarity,
// then ranks by combined score: similarity × (1 - phraseScore).
export function dedupAndRank(candidates: CandidateEdge[]): CandidateEdge[] {
  const best = new Map<string, CandidateEdge>();
  for (const edge of candidates) {
    const key = `${phraseKey(edge.phrase.phrase)}|${edge.targetPath}`;
    const existing = best.get(key);
    if (!existing || edge.similarity > existing.similarity) {
      best.set(key, edge);
    }
  }
  const deduped = Array.from(best.values());
  deduped.sort((a, b) => {
    const scoreA = a.similarity * (1 - a.phrase.score);
    const scoreB = b.similarity * (1 - b.phrase.score);
    return scoreB - scoreA;
  });
  return deduped;
}
