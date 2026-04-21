import { mergeByTarget } from "../resolver/merge-by-target";
import { CandidateEdge } from "../types";

function makeEdge(
  phraseText: string,
  targetId: string,
  similarity: number,
  matchType: "deterministic" | "stochastic" | "sparse-feature",
  sourceId = "id-src",
  extra: Partial<CandidateEdge> = {},
): CandidateEdge {
  return {
    sourcePath: "src.md",
    sourceId,
    phrase: { phrase: phraseText, score: 0.1, startOffset: 0, endOffset: phraseText.length, spanId: `0:${phraseText.length}` },
    targetPath: targetId,
    targetId,
    similarity,
    matchType,
    ...extra,
  };
}

describe("mergeByTarget", () => {
  it("passes through single-resolver edges with matchedBy set", () => {
    const edges = [
      makeEdge("foo", "id-a", 0.9, "deterministic"),
      makeEdge("bar", "id-b", 0.8, "stochastic"),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(2);
    const foo = out.find((e) => e.phrase.phrase === "foo")!;
    expect(foo.matchType).toBe("deterministic");
    expect(foo.matchedBy).toEqual(["lcs"]);
    expect(foo.lcsSimilarity).toBe(0.9);
    expect(foo.denseSimilarity).toBeUndefined();
  });

  it("merges same (phraseKey, targetId) from lcs + dense resolvers", () => {
    const edges = [
      makeEdge("neural nets", "id-tgt", 0.92, "deterministic"),
      makeEdge("neural nets", "id-tgt", 0.68, "stochastic"),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(1);
    expect(out[0].matchType).toBe("both");
    expect(out[0].matchedBy).toEqual(["lcs", "dense"]);
    expect(out[0].similarity).toBe(0.92); // max
    expect(out[0].lcsSimilarity).toBe(0.92);
    expect(out[0].denseSimilarity).toBe(0.68);
  });

  it("mergeByTarget is case/whitespace insensitive on phrase", () => {
    const edges = [
      makeEdge("Neural  Nets", "id-tgt", 0.9, "deterministic"),
      makeEdge("neural nets", "id-tgt", 0.7, "stochastic"),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(1);
    expect(out[0].matchType).toBe("both");
  });

  it("keeps different targets as separate edges", () => {
    const edges = [
      makeEdge("foo", "id-a", 0.9, "deterministic"),
      makeEdge("foo", "id-b", 0.8, "stochastic"),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(2);
    expect(out.every((e) => e.matchType !== "both")).toBe(true);
  });

  it("sorts output by similarity descending", () => {
    const edges = [
      makeEdge("a", "id-a", 0.7, "deterministic"),
      makeEdge("b", "id-b", 0.95, "stochastic"),
      makeEdge("c", "id-c", 0.85, "deterministic"),
    ];
    const out = mergeByTarget(edges);
    expect(out.map((e) => e.similarity)).toEqual([0.95, 0.85, 0.7]);
  });

  it("handles empty input", () => {
    expect(mergeByTarget([])).toEqual([]);
  });

  it("stochastic edge that merges keeps dense score, lcs score from deterministic", () => {
    const edges = [
      makeEdge("foo", "id-a", 0.6, "stochastic"),
      makeEdge("foo", "id-a", 0.88, "deterministic"),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(1);
    expect(out[0].lcsSimilarity).toBe(0.88);
    expect(out[0].denseSimilarity).toBe(0.6);
    expect(out[0].similarity).toBe(0.88);
  });

  it("3-way merge: lcs + dense + sparse yield matchedBy union and max similarity", () => {
    const sparseFeatures = {
      phraseFeatures: [{ idx: 42, value: 0.5, label: "foo · bar · baz" }],
      titleFeatures:  [{ idx: 42, value: 0.6, label: "foo · bar · baz" }],
    };
    const edges = [
      makeEdge("graph theory", "id-tgt", 0.82, "deterministic"),
      makeEdge("graph theory", "id-tgt", 0.77, "stochastic"),
      makeEdge("graph theory", "id-tgt", 0.91, "sparse-feature", "id-src", { sparseFeatures }),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(1);
    const e = out[0];
    expect(e.matchedBy).toEqual(["lcs", "dense", "sparse"]);
    expect(e.matchType).toBe("mixed");
    expect(e.lcsSimilarity).toBe(0.82);
    expect(e.denseSimilarity).toBe(0.77);
    expect(e.sparseSimilarity).toBe(0.91);
    expect(e.similarity).toBe(0.91); // max
    expect(e.sparseFeatures).toBe(sparseFeatures);
  });

  it("sparse-only edge carries sparseFeatures and matchedBy=['sparse']", () => {
    const sparseFeatures = {
      phraseFeatures: [{ idx: 1, value: 0.3, label: "a · b · c" }],
      titleFeatures:  [{ idx: 1, value: 0.4, label: "a · b · c" }],
    };
    const edges = [
      makeEdge("graph theory", "id-tgt", 0.85, "sparse-feature", "id-src", { sparseFeatures }),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(1);
    expect(out[0].matchedBy).toEqual(["sparse"]);
    expect(out[0].sparseSimilarity).toBe(0.85);
    expect(out[0].sparseFeatures).toBe(sparseFeatures);
  });
});
