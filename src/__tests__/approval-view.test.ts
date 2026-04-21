import { buildCardViewModel, CardVM, matchesTab } from "../ui/approval-view";
import { CandidateEdge } from "../types";

function makeEdge(
  phrase: string,
  targetPath: string,
  similarity: number,
  matchType: "deterministic" | "stochastic" | "both",
  overrides: Partial<CandidateEdge> = {},
): CandidateEdge {
  return {
    sourcePath: "src.md",
    sourceId: "id-src",
    phrase: { phrase, score: 0.1, startOffset: 0, endOffset: phrase.length, spanId: `0:${phrase.length}` },
    targetPath,
    targetId: "id-tgt",
    similarity,
    matchType,
    ...overrides,
  };
}

describe("buildCardViewModel", () => {
  it("returns empty array for empty input", () => {
    expect(buildCardViewModel([])).toEqual([]);
  });

  it("sorts by similarity descending", () => {
    const edges = [
      makeEdge("a", "A", 0.7, "deterministic"),
      makeEdge("b", "B", 0.95, "stochastic"),
      makeEdge("c", "C", 0.85, "deterministic"),
    ];
    const vms = buildCardViewModel(edges);
    expect(vms.map((v) => v.similarity)).toEqual([0.95, 0.85, 0.7]);
  });

  it("formats similarity as percentage string", () => {
    const edges = [makeEdge("foo", "Foo Note", 0.923, "deterministic")];
    const [vm] = buildCardViewModel(edges);
    expect(vm.similarityLabel).toBe("92%");
  });

  it("maps deterministic matchType to lcs badge", () => {
    const [vm] = buildCardViewModel([makeEdge("x", "X", 0.9, "deterministic")]);
    expect(vm.badge).toBe("lcs");
    expect(vm.scoreDetail).toBeUndefined();
  });

  it("maps stochastic matchType to emb badge", () => {
    const [vm] = buildCardViewModel([makeEdge("x", "X", 0.9, "stochastic")]);
    expect(vm.badge).toBe("emb");
  });

  it("maps both matchType to lcs+emb badge with dual scores", () => {
    const edge = makeEdge("x", "X", 0.92, "both", {
      lcsSimilarity: 0.92,
      embSimilarity: 0.68,
    });
    const [vm] = buildCardViewModel([edge]);
    expect(vm.badge).toBe("lcs+emb");
    expect(vm.scoreDetail).toBe("lcs 0.92 · emb 0.68");
  });

  it("exposes phrase text and target title", () => {
    const [vm] = buildCardViewModel([makeEdge("neural nets", "Neural Networks", 0.9, "deterministic")]);
    expect(vm.phraseText).toBe("neural nets");
    expect(vm.targetTitle).toBe("Neural Networks");
  });

  it("exposes sourceId, phrase, targetId for action handlers", () => {
    const edge = makeEdge("foo", "Bar", 0.9, "deterministic");
    const [vm] = buildCardViewModel([edge]);
    expect(vm.sourceId).toBe("id-src");
    expect(vm.targetId).toBe("id-tgt");
    expect(vm.edge).toBe(edge);
  });
});

describe("matchesTab", () => {
  const det = makeEdge("a", "A", 0.8, "deterministic");
  const sto = makeEdge("b", "B", 0.8, "stochastic");
  const both = makeEdge("c", "C", 0.8, "both");

  it("lcs tab includes deterministic and both", () => {
    expect([det, sto, both].filter(matchesTab("lcs"))).toEqual([det, both]);
  });

  it("emb tab includes stochastic and both", () => {
    expect([det, sto, both].filter(matchesTab("emb"))).toEqual([sto, both]);
  });
});
