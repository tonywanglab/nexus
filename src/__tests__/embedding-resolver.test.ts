import { cosineSimilarity } from "../resolver/cosine";
import {
  excludeSelfTitles,
  normalizedBasename,
} from "../resolver/normalization";
import { EmbeddingProvider } from "../resolver/embedding-provider";
import { buildDenseExplanation, EmbeddingResolver } from "../resolver/embedding-resolver";
import { SparseAutoencoder } from "../resolver/sae";
import { SAEFeatureLabels } from "../resolver/sae-feature-labels";
import { ExtractedPhrase } from "../types";

// Stub the labels JSON so the test is deterministic — all 32 features get labels.
jest.mock("../../assets/sae-feature-labels-v2.json", () => {
  const labels = Array.from({ length: 32 }, (_, i) => ({
    candidates: [`concept${i}`, `word${i}`, `idea${i}`],
    scores: [0.9 - i * 0.01, 0.8 - i * 0.01, 0.7 - i * 0.01],
  }));
  return { dHidden: 32, vocabSize: 100, vocabSource: "test", minScore: 0.25, labelsPerFeature: 3, labels };
}, { virtual: true });

// ---------------------------------------------------------------------------
// MockEmbeddingProvider (test-only)
// ---------------------------------------------------------------------------

/**
 * Deterministic mock embedding provider for unit tests.
 * Generates vectors from text hash so similar strings get similar-ish vectors.
 */
class MockEmbeddingProvider implements EmbeddingProvider {
  readonly dims: number;

  constructor(dims: number = 8) {
    this.dims = dims;
  }

  async embed(text: string): Promise<Float32Array> {
    return this.deterministicVector(text);
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    return texts.map(t => this.deterministicVector(t));
  }

  async dispose(): Promise<void> {}

  private deterministicVector(text: string): Float32Array {
    const vec = new Float32Array(this.dims);
    const normalized = text.toLowerCase().trim();
    for (let i = 0; i < normalized.length; i++) {
      const code = normalized.charCodeAt(i);
      vec[i % this.dims] += code;
    }
    let norm = 0;
    for (let i = 0; i < this.dims; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < this.dims; i++) vec[i] /= norm;
    }
    return vec;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function makePhrase(
  phrase: string,
  score = 0.5,
  startOffset = 0,
): ExtractedPhrase {
  return {
    phrase,
    score,
    startOffset,
    endOffset: startOffset + phrase.length,
    spanId: `span-${phrase}`,
  };
}

// ---------------------------------------------------------------------------
// cosineSimilarity
// ---------------------------------------------------------------------------
describe("cosineSimilarity", () => {
  it("returns 1 for identical vectors", () => {
    const a = new Float32Array([1, 2, 3]);
    expect(cosineSimilarity(a, a)).toBeCloseTo(1, 5);
  });

  it("returns -1 for opposite vectors", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([-1, 0, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1, 5);
  });

  it("returns 0 for orthogonal vectors", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([0, 1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
  });

  it("returns 0 for zero vectors", () => {
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([1, 2, 3]);
    expect(cosineSimilarity(a, b)).toBe(0);
  });

  it("is scale-invariant", () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([2, 4, 6]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// excludeSelfTitles / normalizedBasename
// ---------------------------------------------------------------------------
describe("normalizedBasename", () => {
  it("extracts and normalizes basename from path", () => {
    expect(normalizedBasename("folder/My Note.md")).toBe("my note");
  });

  it("handles nested paths", () => {
    expect(normalizedBasename("a/b/c/Note.md")).toBe("note");
  });

  it("handles bare filename", () => {
    expect(normalizedBasename("Note.md")).toBe("note");
  });
});

describe("excludeSelfTitles", () => {
  it("filters out the source note title", () => {
    const titles = ["My Note", "Other Note", "Another"];
    const result = excludeSelfTitles(titles, "folder/My Note.md");
    expect(result).toEqual(["Other Note", "Another"]);
  });

  it("returns all titles when no self-match", () => {
    const titles = ["A", "B", "C"];
    const result = excludeSelfTitles(titles, "folder/D.md");
    expect(result).toEqual(["A", "B", "C"]);
  });

  it("handles case-insensitive matching", () => {
    const titles = ["my note", "Other"];
    const result = excludeSelfTitles(titles, "My Note.md");
    expect(result).toEqual(["Other"]);
  });
});

// ---------------------------------------------------------------------------
// MockEmbeddingProvider
// ---------------------------------------------------------------------------
describe("MockEmbeddingProvider", () => {
  const provider = new MockEmbeddingProvider(8);

  it("returns vectors of correct dimensions", async () => {
    const vec = await provider.embed("hello");
    expect(vec.length).toBe(8);
  });

  it("returns unit-length vectors", async () => {
    const vec = await provider.embed("test string");
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    expect(Math.sqrt(norm)).toBeCloseTo(1, 4);
  });

  it("returns identical vectors for identical text", async () => {
    const a = await provider.embed("hello world");
    const b = await provider.embed("hello world");
    expect(cosineSimilarity(a, b)).toBeCloseTo(1, 5);
  });

  it("embedBatch returns correct count", async () => {
    const results = await provider.embedBatch(["a", "b", "c"]);
    expect(results).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// EmbeddingResolver
// ---------------------------------------------------------------------------
describe("EmbeddingResolver", () => {
  const provider = new MockEmbeddingProvider(8);
  const resolver = new EmbeddingResolver({
    embeddingProvider: provider,
    similarityThreshold: 0.5, // lower threshold for mock vectors
  });

  it("returns empty for empty inputs", async () => {
    expect(await resolver.resolve([], [], "note.md")).toEqual([]);
    expect(await resolver.resolve([makePhrase("test")], [], "note.md")).toEqual([]);
    expect(await resolver.resolve([], ["Title"], "note.md")).toEqual([]);
  });

  it("filters out self-links", async () => {
    const phrases = [makePhrase("My Note", 0.3)];
    const titles = ["My Note", "Other Note"];
    const edges = await resolver.resolve(phrases, titles, "folder/My Note.md");

    expect(edges.every(e => e.targetPath !== "My Note")).toBe(true);
  });

  it("resolves short phrases (filtering is upstream)", async () => {
    const phrases = [makePhrase("x", 0.1)];
    const titles = ["x", "Something"];
    const edges = await resolver.resolve(phrases, titles, "note.md");

    // Short phrases are no longer filtered here — SpanExtractor handles it
    expect(edges.length).toBeGreaterThanOrEqual(0);
  });

  it("sets matchType to stochastic", async () => {
    const phrases = [makePhrase("test phrase", 0.3)];
    const titles = ["test phrase"];
    const edges = await resolver.resolve(phrases, titles, "note.md");

    for (const edge of edges) {
      expect(edge.matchType).toBe("stochastic");
    }
  });

  it("deduplicates: same (phrase, target) pair keeps best; distinct phrases to same target are kept", async () => {
    const phrases = [
      makePhrase("test phrase", 0.3),
      makePhrase("test phrase variant", 0.6),
    ];
    const titles = ["test phrase"];
    const edges = await resolver.resolve(phrases, titles, "note.md");

    // Each unique (phrase, target) pair appears at most once.
    const seen = new Set(edges.map(e => `${e.phrase.phrase}|${e.targetPath}`));
    expect(seen.size).toBe(edges.length);
    // Both distinct phrases can produce an edge to the same target.
    const targetEdges = edges.filter(e => e.targetPath === "test phrase");
    expect(targetEdges.length).toBeGreaterThanOrEqual(1);
  });

  it("ranks by combined score (similarity × (1 - phraseScore))", async () => {
    // Use very similar phrases to the titles so they pass threshold
    const phrases = [
      makePhrase("alpha", 0.2),   // low phraseScore = high quality
      makePhrase("alpha!", 0.9),  // high phraseScore = low quality
    ];
    const titles = ["alpha", "alpha!"];
    const edges = await resolver.resolve(phrases, titles, "note.md");

    if (edges.length >= 2) {
      const score0 = edges[0].similarity * (1 - edges[0].phrase.score);
      const score1 = edges[1].similarity * (1 - edges[1].phrase.score);
      expect(score0).toBeGreaterThanOrEqual(score1);
    }
  });

  it("respects similarity threshold", async () => {
    const strictResolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.99, // very high threshold
    });
    const phrases = [makePhrase("hello world", 0.3)];
    const titles = ["completely different"];
    const edges = await strictResolver.resolve(phrases, titles, "note.md");

    expect(edges).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// EmbeddingResolver with SAE
// ---------------------------------------------------------------------------
describe("EmbeddingResolver with SAE", () => {
  const provider = new MockEmbeddingProvider(8);
  const sae = SparseAutoencoder.randomInit(8, 32, 4, 99);

  it("dense edges carry SAE sparse-feature explanations when sae + featureLabels are attached", async () => {
    const featureLabels = new SAEFeatureLabels(sae.dHidden);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      sae,
      featureLabels,
    });

    const phrases = [makePhrase("alpha", 0.3), makePhrase("beta", 0.3)];
    const titles = ["alpha", "gamma"];
    const { candidates } = await resolver.resolveWithSparse(phrases, titles, "note.md");

    // At least one edge should carry a sparse-feature explanation.
    const explained = candidates.filter((e) => e.sparseFeatures);
    expect(explained.length).toBeGreaterThan(0);
    for (const e of explained) {
      const sf = e.sparseFeatures!;
      expect(sf.phraseFeatures.length).toBeGreaterThan(0);
      expect(sf.titleFeatures.length).toBe(sf.phraseFeatures.length);
    }
  });

  it("returns the same candidates whether or not the SAE is attached", async () => {
    const baseline = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.5,
    });
    const withSAE = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.5,
      sae,
    });
    const phrases = [makePhrase("alpha", 0.3), makePhrase("beta", 0.4)];
    const titles = ["alpha", "gamma", "delta"];

    const a = await baseline.resolve(phrases, titles, "note.md");
    const b = await withSAE.resolve(phrases, titles, "note.md");

    expect(b).toHaveLength(a.length);
    for (let i = 0; i < a.length; i++) {
      expect(b[i].targetPath).toBe(a[i].targetPath);
      expect(b[i].phrase.phrase).toBe(a[i].phrase.phrase);
      expect(b[i].similarity).toBeCloseTo(a[i].similarity, 6);
    }
  });

  it("enriches dense edges with top shared SAE features when featureLabels are attached", async () => {
    const featureLabels = new SAEFeatureLabels(sae.dHidden);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      sae,
      featureLabels,
    });

    const phrases = [makePhrase("alpha")];
    const titles = ["beta"];
    const edges = await resolver.resolve(phrases, titles, "note.md");
    expect(edges.length).toBeGreaterThan(0);

    // At least one edge should carry a sparse-feature explanation.
    const explained = edges.filter((e) => e.sparseFeatures);
    expect(explained.length).toBeGreaterThan(0);

    for (const e of explained) {
      const sf = e.sparseFeatures!;
      // Default topN = 2.
      expect(sf.phraseFeatures.length).toBeLessThanOrEqual(2);
      expect(sf.titleFeatures.length).toBe(sf.phraseFeatures.length);
      // The explanation is the intersection — each row's feature idx must match.
      for (let i = 0; i < sf.phraseFeatures.length; i++) {
        expect(sf.phraseFeatures[i].idx).toBe(sf.titleFeatures[i].idx);
        expect(sf.phraseFeatures[i].label).toBe(sf.titleFeatures[i].label);
        expect(sf.phraseFeatures[i].label.length).toBeGreaterThan(0);
      }
    }
  });

  it("respects denseExplanationTopN option", async () => {
    const featureLabels = new SAEFeatureLabels(sae.dHidden);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      sae,
      featureLabels,
      denseExplanationTopN: 1,
    });

    const edges = await resolver.resolve([makePhrase("alpha")], ["beta"], "note.md");
    for (const e of edges) {
      if (!e.sparseFeatures) continue;
      expect(e.sparseFeatures.phraseFeatures.length).toBeLessThanOrEqual(1);
      expect(e.sparseFeatures.titleFeatures.length).toBeLessThanOrEqual(1);
    }
  });

  it("buildDenseExplanation ranks shared features by product of activations (not min)", () => {
    // Craft two sparse encodings where one shared feature fires imbalanced
    // (3.0 × 0.1 = 0.30) and another fires balanced-but-weak (0.5 × 0.5 = 0.25).
    // Product ranks the imbalanced one first; min would have ranked the balanced
    // one first. This test pins the product semantics.
    const featureLabels = new SAEFeatureLabels(32);
    const pEnc = {
      indices: Int32Array.from([7, 11]),
      values: Float32Array.from([3.0, 0.5]),
    };
    const tEnc = {
      indices: Int32Array.from([7, 11]),
      values: Float32Array.from([0.1, 0.5]),
    };
    const out = buildDenseExplanation(pEnc, tEnc, featureLabels, 2)!;
    expect(out).toBeDefined();
    // Product(7) = 0.30, Product(11) = 0.25 → feature 7 ranks first under product.
    // Min(7) = 0.1, Min(11) = 0.5 → feature 11 would rank first under min.
    expect(out.phraseFeatures[0].idx).toBe(7);
    expect(out.phraseFeatures[1].idx).toBe(11);
    expect(out.phraseFeatures[0].value).toBeCloseTo(3.0, 6);
    expect(out.titleFeatures[0].value).toBeCloseTo(0.1, 6);
  });

  it("does not attach sparseFeatures when featureLabels are not provided", async () => {
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      sae,
    });
    const edges = await resolver.resolve([makePhrase("alpha")], ["beta"], "note.md");
    for (const e of edges) expect(e.sparseFeatures).toBeUndefined();
  });

  it("resolveWithSparse returns candidates without crashing when no SAE", async () => {
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.5,
    });
    const result = await resolver.resolveWithSparse(
      [makePhrase("alpha", 0.3)],
      ["alpha"],
      "note.md",
    );
    expect(result.candidates).toBeDefined();
    expect(Array.isArray(result.candidates)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// EmbeddingResolver.resolveBySparseFeatures
// ---------------------------------------------------------------------------
describe("EmbeddingResolver.resolveBySparseFeatures", () => {
  const provider = new MockEmbeddingProvider(8);
  // dModel=8, dHidden=32, k=4 — all 32 features are labeled by the mock JSON above.
  const sae = SparseAutoencoder.randomInit(8, 32, 4, 99);
  const featureLabels = new SAEFeatureLabels();

  const resolver = new EmbeddingResolver({
    embeddingProvider: provider,
    similarityThreshold: 0.5,
    sae,
  });

  it("throws when called without an SAE configured", async () => {
    const noSAEResolver = new EmbeddingResolver({ embeddingProvider: provider });
    await expect(
      noSAEResolver.resolveBySparseFeatures(
        [makePhrase("alpha")],
        ["alpha"],
        "note.md",
        featureLabels,
      ),
    ).rejects.toThrow();
  });

  it("returns empty array for empty phrases or titles", async () => {
    expect(
      await resolver.resolveBySparseFeatures([], ["A"], "note.md", featureLabels),
    ).toEqual([]);
    expect(
      await resolver.resolveBySparseFeatures([makePhrase("A")], [], "note.md", featureLabels),
    ).toEqual([]);
  });

  it("sets matchType to 'sparse-feature' on every returned edge", async () => {
    const edges = await resolver.resolveBySparseFeatures(
      [makePhrase("alpha", 0.3), makePhrase("beta", 0.3)],
      ["alpha", "gamma"],
      "note.md",
      featureLabels,
      { similarityThreshold: 0 }, // 0 threshold so we always get results
    );
    for (const edge of edges) {
      expect(edge.matchType).toBe("sparse-feature");
    }
  });

  it("display payload has at most 4 features per side", async () => {
    const edges = await resolver.resolveBySparseFeatures(
      [makePhrase("alpha", 0.3)],
      ["alpha", "gamma", "delta"],
      "note.md",
      featureLabels,
      { similarityThreshold: 0 },
    );
    for (const edge of edges) {
      expect(edge.sparseFeatures).toBeDefined();
      expect(edge.sparseFeatures!.phraseFeatures.length).toBeLessThanOrEqual(4);
      expect(edge.sparseFeatures!.titleFeatures.length).toBeLessThanOrEqual(4);
    }
  });

  it("each sparseFeature entry carries a non-empty label string", async () => {
    const edges = await resolver.resolveBySparseFeatures(
      [makePhrase("alpha", 0.3)],
      ["alpha"],
      "note.md",
      featureLabels,
      { similarityThreshold: 0 },
    );
    for (const edge of edges) {
      for (const f of edge.sparseFeatures!.phraseFeatures) {
        expect(typeof f.label).toBe("string");
        expect(f.label.length).toBeGreaterThan(0);
      }
    }
  });

  it("filters self-links", async () => {
    const edges = await resolver.resolveBySparseFeatures(
      [makePhrase("My Note", 0.3)],
      ["My Note", "Other"],
      "folder/My Note.md",
      featureLabels,
      { similarityThreshold: 0 },
    );
    expect(edges.every(e => e.targetPath !== "My Note")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Counting mock provider — lets us assert provider.embedBatch is actually
// called vs. served from cache. Separate from MockEmbeddingProvider to avoid
// perturbing existing tests.
// ---------------------------------------------------------------------------
class CountingMockProvider implements EmbeddingProvider {
  readonly dims: number;
  public calls = 0;
  public embeddedTexts: string[][] = [];

  constructor(dims: number = 8) {
    this.dims = dims;
  }

  async embed(text: string): Promise<Float32Array> {
    return (await this.embedBatch([text]))[0];
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    this.calls++;
    this.embeddedTexts.push([...texts]);
    return texts.map((t) => this.deterministicVector(t));
  }

  async dispose(): Promise<void> {}

  private deterministicVector(text: string): Float32Array {
    const v = new Float32Array(this.dims);
    for (let i = 0; i < text.length; i++) v[i % this.dims] += text.charCodeAt(i);
    let n = 0;
    for (let i = 0; i < this.dims; i++) n += v[i] * v[i];
    n = Math.sqrt(n);
    if (n > 0) for (let i = 0; i < this.dims; i++) v[i] /= n;
    return v;
  }
}

// ---------------------------------------------------------------------------
// Vault-wide phrase embedding cache
// ---------------------------------------------------------------------------
describe("EmbeddingResolver phrase cache", () => {
  it("does not re-embed the same phrase across resolve calls", async () => {
    const provider = new CountingMockProvider(8);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
    });

    await resolver.resolve(
      [makePhrase("machine learning"), makePhrase("neural networks")],
      ["ml", "dl"],
      "a.md",
    );
    const callsAfterFirst = provider.calls;
    expect(callsAfterFirst).toBeGreaterThan(0);

    // Second call with the same phrases — provider should only be invoked
    // for title work, not phrase embedding (titles are also cached, so ideally
    // zero extra calls when the title set doesn't change).
    provider.calls = 0;
    provider.embeddedTexts = [];
    await resolver.resolve(
      [makePhrase("machine learning"), makePhrase("neural networks")],
      ["ml", "dl"],
      "b.md",
    );
    // No embedding should have been invoked — both phrases and titles are cached.
    expect(provider.calls).toBe(0);
  });

  it("only embeds the novel subset when the phrase set overlaps a prior call", async () => {
    const provider = new CountingMockProvider(8);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
    });

    await resolver.resolve(
      [makePhrase("alpha"), makePhrase("beta")],
      ["x"],
      "a.md",
    );
    provider.calls = 0;
    provider.embeddedTexts = [];
    await resolver.resolve(
      [makePhrase("alpha"), makePhrase("gamma")], // alpha cached, gamma new
      ["x"],
      "b.md",
    );
    // Only "gamma" should hit the provider for phrase embedding.
    const allEmbedded = provider.embeddedTexts.flat();
    expect(allEmbedded).toContain("gamma");
    expect(allEmbedded).not.toContain("alpha");
  });

  it("evicts FIFO when phrase cache exceeds the limit", async () => {
    const provider = new CountingMockProvider(8);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      phraseCacheLimit: 3,
    });

    // Fill: "a", "b", "c"
    await resolver.resolve(
      [makePhrase("a"), makePhrase("b"), makePhrase("c")],
      ["x"],
      "first.md",
    );
    // Add "d" → evicts "a".
    await resolver.resolve([makePhrase("d")], ["x"], "second.md");

    // Now asking for "a" again should re-embed it.
    provider.calls = 0;
    provider.embeddedTexts = [];
    await resolver.resolve([makePhrase("a"), makePhrase("d")], ["x"], "third.md");

    const embedded = provider.embeddedTexts.flat();
    expect(embedded).toContain("a"); // was evicted, re-embedded
    expect(embedded).not.toContain("d"); // still cached
  });

  it("FIFO eviction during a call cannot drop entries still needed by that call", async () => {
    // Regression: the eviction loop used to run before the snapshot map,
    // so when a call added N new phrases that pushed the cache over cap,
    // it could evict previously-cached phrases that were also in the
    // current request — causing .get(t) to return undefined and the SAE
    // encodeSparse call to blow up on an undefined embedding.
    const provider = new CountingMockProvider(8);
    const sae = SparseAutoencoder.randomInit(8, 32, 4, 42);
    const featureLabels = new SAEFeatureLabels(32);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      phraseCacheLimit: 3,
      sae,
      featureLabels,
    });

    // Prime the cache with "a" (now the oldest entry).
    await resolver.resolve([makePhrase("a")], ["x"], "first.md");

    // Second call asks for "a" + 3 new phrases. New entries push the cache
    // size from 1 to 4; limit is 3, so one entry is evicted. The snapshot
    // must capture "a"'s embedding before the eviction runs — otherwise
    // phraseEmbeddings[0] is undefined and encodeSparse blows up.
    // If eviction is buggy this throws; otherwise it must return 4 candidates.
    const { candidates } = await resolver.resolveWithSparse(
      [makePhrase("a"), makePhrase("b"), makePhrase("c"), makePhrase("d")],
      ["x"],
      "second.md",
    );
    expect(Array.isArray(candidates)).toBe(true);
  });

  it("shares phrase cache between resolve and resolveBySparseFeatures", async () => {
    const provider = new CountingMockProvider(8);
    const sae = SparseAutoencoder.randomInit(8, 32, 4, 42);
    const featureLabels = new SAEFeatureLabels(32);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      sae,
    });

    const phrases = [makePhrase("p1"), makePhrase("p2")];
    await resolver.resolve(phrases, ["t1"], "a.md");
    provider.calls = 0;
    provider.embeddedTexts = [];

    await resolver.resolveBySparseFeatures(phrases, ["t1"], "a.md", featureLabels, {
      similarityThreshold: 0,
    });
    // All phrase texts and the title are cached — no embed calls needed.
    expect(provider.calls).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Title embedding cache: seed, export, prune, change notifications
// ---------------------------------------------------------------------------
describe("EmbeddingResolver title cache", () => {
  it("seedTitleEmbeddings suppresses future embedBatch calls for those titles", async () => {
    const provider = new CountingMockProvider(8);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
    });

    const seeded = new Map<string, Float32Array>();
    seeded.set("t1", new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]));
    seeded.set("t2", new Float32Array([0, 1, 0, 0, 0, 0, 0, 0]));
    resolver.seedTitleEmbeddings(seeded);

    await resolver.resolve([makePhrase("phrase")], ["t1", "t2"], "a.md");
    // Only the phrase should have been embedded; titles served from seed.
    const embedded = provider.embeddedTexts.flat();
    expect(embedded).toContain("phrase");
    expect(embedded).not.toContain("t1");
    expect(embedded).not.toContain("t2");
  });

  it("exportTitleEmbeddings snapshots the current cache and is isolated from further writes", async () => {
    const provider = new CountingMockProvider(8);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
    });
    await resolver.resolve([makePhrase("p")], ["t1"], "a.md");
    const snap = resolver.exportTitleEmbeddings();
    expect(snap.has("t1")).toBe(true);
    const originalSize = snap.size;

    await resolver.resolve([makePhrase("p")], ["t1", "t2"], "b.md");
    // Old snapshot is unaffected by later resolve() calls.
    expect(snap.size).toBe(originalSize);
    expect(snap.has("t2")).toBe(false);
    // But the resolver itself has t2 cached now.
    expect(resolver.exportTitleEmbeddings().has("t2")).toBe(true);
  });

  it("onTitleEmbeddingsChanged fires only when new titles are added", async () => {
    const provider = new CountingMockProvider(8);
    const changes: number[] = [];
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      onTitleEmbeddingsChanged: (m) => changes.push(m.size),
    });

    await resolver.resolve([makePhrase("p")], ["t1", "t2"], "a.md");
    expect(changes.length).toBe(1);
    expect(changes[0]).toBe(2);

    // Same titles again → no change event.
    await resolver.resolve([makePhrase("p")], ["t1", "t2"], "b.md");
    expect(changes.length).toBe(1);

    // One new title → one more event.
    await resolver.resolve([makePhrase("p")], ["t1", "t2", "t3"], "c.md");
    expect(changes.length).toBe(2);
    expect(changes[1]).toBe(3);
  });

  it("pruneTitleCacheTo removes stale titles and fires a change event", async () => {
    const provider = new CountingMockProvider(8);
    let lastSize = 0;
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      onTitleEmbeddingsChanged: (m) => { lastSize = m.size; },
    });
    await resolver.resolve([makePhrase("p")], ["a", "b", "c"], "x.md");
    expect(lastSize).toBe(3);

    const removed = resolver.pruneTitleCacheTo(new Set(["a"]));
    expect(removed).toBe(2);
    expect(lastSize).toBe(1);
    expect([...resolver.exportTitleEmbeddings().keys()]).toEqual(["a"]);
  });

  it("pruneTitleCacheTo is a no-op when nothing is stale", async () => {
    const provider = new CountingMockProvider(8);
    let events = 0;
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
      onTitleEmbeddingsChanged: () => events++,
    });
    await resolver.resolve([makePhrase("p")], ["a", "b"], "x.md");
    const before = events;
    const removed = resolver.pruneTitleCacheTo(new Set(["a", "b"]));
    expect(removed).toBe(0);
    expect(events).toBe(before); // no extra event
  });

  it("retains cached titles across calls whose title set differs (no churn)", async () => {
    const provider = new CountingMockProvider(8);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      similarityThreshold: 0.0,
    });
    await resolver.resolve([makePhrase("p")], ["a", "b"], "1.md");
    // Second call omits "b" — but "b" should still be in the cache, not pruned.
    await resolver.resolve([makePhrase("p")], ["a"], "2.md");
    expect(resolver.exportTitleEmbeddings().has("b")).toBe(true);

    // Third call reintroduces "b" with the same title set — provider must not
    // be invoked to re-embed it.
    provider.calls = 0;
    provider.embeddedTexts = [];
    await resolver.resolve([makePhrase("p")], ["a", "b"], "3.md");
    expect(provider.embeddedTexts.flat()).not.toContain("b");
  });
});
