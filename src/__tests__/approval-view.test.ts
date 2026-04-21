import { buildCardViewModel, CardVM, matchesTab, resolversOf } from "../ui/approval-view";
import { CandidateEdge } from "../types";

function makeEdge(
  phrase: string,
  targetPath: string,
  similarity: number,
  matchType: "deterministic" | "stochastic" | "both" | "sparse-feature" | "mixed",
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

  it("maps stochastic matchType to dense badge", () => {
    const [vm] = buildCardViewModel([makeEdge("x", "X", 0.9, "stochastic")]);
    expect(vm.badge).toBe("dense");
  });

  it("maps sparse-feature matchType to sparse badge", () => {
    const [vm] = buildCardViewModel([makeEdge("x", "X", 0.85, "sparse-feature")]);
    expect(vm.badge).toBe("sparse");
  });

  it("maps both matchType to lcs+dense badge with dual scores", () => {
    const edge = makeEdge("x", "X", 0.92, "both", {
      lcsSimilarity: 0.92,
      denseSimilarity: 0.68,
    });
    const [vm] = buildCardViewModel([edge]);
    expect(vm.badge).toBe("lcs+dense");
    expect(vm.scoreDetail).toBe("lcs 0.92 · dense 0.68");
  });

  it("uses matchedBy over matchType when present", () => {
    const edge = makeEdge("x", "X", 0.91, "mixed", {
      matchedBy: ["lcs", "dense", "sparse"],
      lcsSimilarity: 0.82,
      denseSimilarity: 0.77,
      sparseSimilarity: 0.91,
    });
    const [vm] = buildCardViewModel([edge]);
    expect(vm.badge).toBe("lcs+dense+sparse");
    expect(vm.scoreDetail).toBe("lcs 0.82 · dense 0.77 · sparse 0.91");
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
  const sparse = makeEdge("d", "D", 0.8, "sparse-feature");

  it("lcs tab includes deterministic and both (legacy matchType)", () => {
    expect([det, sto, both, sparse].filter(matchesTab("lcs"))).toEqual([det, both]);
  });

  it("dense tab includes stochastic and both (legacy matchType)", () => {
    expect([det, sto, both, sparse].filter(matchesTab("dense"))).toEqual([sto, both]);
  });

  it("sparse tab includes sparse-feature (legacy matchType)", () => {
    expect([det, sto, both, sparse].filter(matchesTab("sparse"))).toEqual([sparse]);
  });

  it("uses matchedBy over matchType when present", () => {
    const lcsAndSparse = makeEdge("x", "X", 0.9, "mixed", {
      matchedBy: ["lcs", "sparse"],
    });
    expect(matchesTab("lcs")(lcsAndSparse)).toBe(true);
    expect(matchesTab("dense")(lcsAndSparse)).toBe(false);
    expect(matchesTab("sparse")(lcsAndSparse)).toBe(true);
  });

  it("triple matchedBy edge appears in all three tabs", () => {
    const triple = makeEdge("x", "X", 0.91, "mixed", {
      matchedBy: ["lcs", "dense", "sparse"],
    });
    expect(matchesTab("lcs")(triple)).toBe(true);
    expect(matchesTab("dense")(triple)).toBe(true);
    expect(matchesTab("sparse")(triple)).toBe(true);
  });

  it("legacy edge with matchType=both matches lcs and dense but not sparse", () => {
    const legacyBoth = makeEdge("x", "X", 0.9, "both");
    expect(matchesTab("lcs")(legacyBoth)).toBe(true);
    expect(matchesTab("dense")(legacyBoth)).toBe(true);
    expect(matchesTab("sparse")(legacyBoth)).toBe(false);
  });
});

describe("resolversOf", () => {
  it("prefers matchedBy when present", () => {
    const e = makeEdge("x", "X", 0.9, "both", { matchedBy: ["lcs", "sparse"] });
    expect(resolversOf(e)).toEqual(["lcs", "sparse"]);
  });

  it("falls back to matchType=deterministic → lcs", () => {
    expect(resolversOf(makeEdge("x", "X", 0.9, "deterministic"))).toEqual(["lcs"]);
  });

  it("falls back to matchType=stochastic → dense", () => {
    expect(resolversOf(makeEdge("x", "X", 0.9, "stochastic"))).toEqual(["dense"]);
  });

  it("falls back to matchType=sparse-feature → sparse", () => {
    expect(resolversOf(makeEdge("x", "X", 0.9, "sparse-feature"))).toEqual(["sparse"]);
  });

  it("falls back to matchType=both → lcs + dense", () => {
    expect(resolversOf(makeEdge("x", "X", 0.9, "both"))).toEqual(["lcs", "dense"]);
  });
});
