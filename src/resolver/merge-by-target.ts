import { CandidateEdge, MatchType, Resolver, phraseKey } from "../types";

const TYPE_TO_RESOLVER: Record<string, Resolver | undefined> = {
  deterministic: "lcs",
  stochastic: "dense",
  "sparse-feature": "sparse",
};

const SIMILARITY_FIELD: Record<Resolver, "lcsSimilarity" | "denseSimilarity" | "sparseSimilarity"> = {
  lcs: "lcsSimilarity",
  dense: "denseSimilarity",
  sparse: "sparseSimilarity",
};

function resolverOf(edge: CandidateEdge): Resolver | null {
  if (edge.matchedBy && edge.matchedBy.length === 1) return edge.matchedBy[0];
  const r = edge.matchType ? TYPE_TO_RESOLVER[edge.matchType] : undefined;
  return r ?? null;
}

function legacyMatchType(matchedBy: Resolver[]): MatchType {
  if (matchedBy.length === 1) {
    const r = matchedBy[0];
    return r === "lcs" ? "deterministic" : r === "dense" ? "stochastic" : "sparse-feature";
  }
  if (matchedBy.length === 2 && matchedBy.includes("lcs") && matchedBy.includes("dense")) {
    return "both";
  }
  return "mixed";
}

// merges LCS + dense + sparse edges with the same (phraseKey, targetId).
// on collision, unions `matchedBy`, keeps each resolver's score in its own
// similarity field, carries forward `sparseFeatures` when present, and sets
// aggregate `similarity` to the max across contributors.
// output is sorted by similarity descending.
export function mergeByTarget(edges: CandidateEdge[]): CandidateEdge[] {
  const map = new Map<string, CandidateEdge>();

  for (const edge of edges) {
    const key = `${phraseKey(edge.phrase.phrase)}|${edge.targetId ?? edge.targetPath}`;
    const resolver = resolverOf(edge);
    const existing = map.get(key);

    if (!existing) {
      const entry: CandidateEdge = { ...edge };
      if (resolver) {
        entry.matchedBy = [resolver];
        entry[SIMILARITY_FIELD[resolver]] = edge.similarity;
      }
      entry.matchType = entry.matchedBy ? legacyMatchType(entry.matchedBy) : edge.matchType;
      map.set(key, entry);
      continue;
    }

    if (resolver && !(existing.matchedBy ?? []).includes(resolver)) {
      existing.matchedBy = [...(existing.matchedBy ?? []), resolver];
      existing[SIMILARITY_FIELD[resolver]] = edge.similarity;
    }
    // prefer richer sparseFeatures: the sparse-feature resolver emits up to 4
    // per side; the dense resolver emits just the top 1-2 shared explanations.
    // when both fire for the same (phrase, target), keep whichever carries more.
    if (edge.sparseFeatures) {
      const incoming = edge.sparseFeatures.phraseFeatures.length
        + edge.sparseFeatures.titleFeatures.length;
      const current = existing.sparseFeatures
        ? existing.sparseFeatures.phraseFeatures.length + existing.sparseFeatures.titleFeatures.length
        : 0;
      if (incoming > current) existing.sparseFeatures = edge.sparseFeatures;
    }
    existing.matchType = existing.matchedBy ? legacyMatchType(existing.matchedBy) : existing.matchType;
    existing.similarity = Math.max(
      existing.lcsSimilarity ?? 0,
      existing.denseSimilarity ?? 0,
      existing.sparseSimilarity ?? 0,
    );
  }

  return [...map.values()].sort((a, b) => b.similarity - a.similarity);
}
