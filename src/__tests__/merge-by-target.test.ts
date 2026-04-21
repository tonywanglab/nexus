import { mergeByTarget } from "../resolver/merge-by-target";
import { CandidateEdge } from "../types";

function makeEdge(
  phraseText: string,
  targetId: string,
  similarity: number,
  matchType: "deterministic" | "stochastic",
  sourceId = "id-src",
): CandidateEdge {
  return {
    sourcePath: "src.md",
    sourceId,
    phrase: { phrase: phraseText, score: 0.1, startOffset: 0, endOffset: phraseText.length, spanId: `0:${phraseText.length}` },
    targetPath: targetId,
    targetId,
    similarity,
    matchType,
  };
}

describe("mergeByTarget", () => {
  it("passes through single-resolver edges unchanged", () => {
    const edges = [
      makeEdge("foo", "id-a", 0.9, "deterministic"),
      makeEdge("bar", "id-b", 0.8, "stochastic"),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(2);
    const foo = out.find((e) => e.phrase.phrase === "foo")!;
    expect(foo.matchType).toBe("deterministic");
    expect(foo.lcsSimilarity).toBe(0.9);
    expect(foo.embSimilarity).toBeUndefined();
  });

  it("merges same (phraseKey, targetId) from both resolvers", () => {
    const edges = [
      makeEdge("neural nets", "id-tgt", 0.92, "deterministic"),
      makeEdge("neural nets", "id-tgt", 0.68, "stochastic"),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(1);
    expect(out[0].matchType).toBe("both");
    expect(out[0].similarity).toBe(0.92); // max
    expect(out[0].lcsSimilarity).toBe(0.92);
    expect(out[0].embSimilarity).toBe(0.68);
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

  it("stochastic edge that merges keeps emb score, lcs score from deterministic", () => {
    const edges = [
      makeEdge("foo", "id-a", 0.6, "stochastic"),
      makeEdge("foo", "id-a", 0.88, "deterministic"),
    ];
    const out = mergeByTarget(edges);
    expect(out).toHaveLength(1);
    expect(out[0].lcsSimilarity).toBe(0.88);
    expect(out[0].embSimilarity).toBe(0.6);
    expect(out[0].similarity).toBe(0.88);
  });
});
