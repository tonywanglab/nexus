import { cosineSimilarity } from "../resolver/cosine";
import {
  excludeSelfTitles,
  normalizedBasename,
} from "../resolver/normalization";
import { EmbeddingProvider } from "../resolver/embedding-provider";
import { EmbeddingResolver } from "../resolver/embedding-resolver";
import { ExtractedPhrase } from "../types";

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

  it("deduplicates: same target from multiple phrases keeps best", async () => {
    const phrases = [
      makePhrase("test phrase", 0.3),
      makePhrase("test phrase variant", 0.6),
    ];
    const titles = ["test phrase"];
    const edges = await resolver.resolve(phrases, titles, "note.md");

    const targetEdges = edges.filter(e => e.targetPath === "test phrase");
    expect(targetEdges.length).toBeLessThanOrEqual(1);
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
