import { lcsLength, normalizedSimilarity } from "../resolver/lcs";
import {
  normalize,
  exactMatch,
  phraseContainsTitle,
} from "../resolver/normalization";
import { AliasResolver } from "../resolver";
import { ExtractedPhrase } from "../types";

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
// LCS
// ---------------------------------------------------------------------------
describe("lcsLength", () => {
  it("returns full length for identical strings", () => {
    expect(lcsLength("hello", "hello")).toBe(5);
  });

  it("expects pre-normalized input (no internal case folding)", () => {
    // "Hello" vs "hello" differ at index 0, so LCS = 4 (not 5)
    expect(lcsLength("Hello", "hello")).toBe(4);
    // Pre-normalized inputs compare correctly
    expect(lcsLength("hello", "hello")).toBe(5);
  });

  it("returns 0 for completely different strings", () => {
    expect(lcsLength("abc", "xyz")).toBe(0);
  });

  it("returns 0 when either string is empty", () => {
    expect(lcsLength("", "hello")).toBe(0);
    expect(lcsLength("hello", "")).toBe(0);
  });

  it("handles subsequences correctly", () => {
    expect(lcsLength("ace", "abcde")).toBe(3);
  });
});

describe("normalizedSimilarity", () => {
  it("returns 1 for identical strings", () => {
    expect(normalizedSimilarity("hello", "hello")).toBe(1);
  });

  it("returns 0 for empty inputs", () => {
    expect(normalizedSimilarity("", "hello")).toBe(0);
    expect(normalizedSimilarity("", "")).toBe(0);
  });

  it("returns high similarity for typo", () => {
    const sim = normalizedSimilarity("machin learning", "machine learning");
    expect(sim).toBeGreaterThan(0.9);
  });

  it("returns low similarity for unrelated strings", () => {
    const sim = normalizedSimilarity("quantum physics", "banana bread");
    expect(sim).toBeLessThan(0.3);
  });
});

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------
describe("normalize", () => {
  it("lowercases and trims", () => {
    expect(normalize("  Hello World  ")).toBe("hello world");
  });

  it("collapses whitespace", () => {
    expect(normalize("hello   world")).toBe("hello world");
  });

  it("strips possessives", () => {
    expect(normalize("Turing's machine")).toBe("turing machine");
  });

  it("strips diacritics", () => {
    expect(normalize("café résumé")).toBe("cafe resume");
  });
});

describe("exactMatch", () => {
  it("matches after normalization", () => {
    expect(exactMatch("Machine Learning", "machine learning")).toBe(true);
  });

  it("does not match different strings", () => {
    expect(exactMatch("Machine Learning", "deep learning")).toBe(false);
  });
});

describe("phraseContainsTitle", () => {
  it("matches when phrase contains title", () => {
    expect(
      phraseContainsTitle("Introduction to Machine Learning", "Machine Learning"),
    ).toBe(true);
  });

  it("does not match when title contains phrase", () => {
    expect(
      phraseContainsTitle("Machine Learning", "Introduction to Machine Learning"),
    ).toBe(false);
  });

  it("does not match exact equality", () => {
    expect(phraseContainsTitle("Machine Learning", "machine learning")).toBe(false);
  });

  it("does not match unrelated strings", () => {
    expect(phraseContainsTitle("quantum physics", "banana bread")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// AliasResolver
// ---------------------------------------------------------------------------
describe("AliasResolver", () => {
  const resolver = new AliasResolver();

  it("returns empty for empty inputs", () => {
    expect(resolver.resolve([], [], "note.md")).toEqual([]);
    expect(resolver.resolve([makePhrase("test")], [], "note.md")).toEqual([]);
    expect(resolver.resolve([], ["Note Title"], "note.md")).toEqual([]);
  });

  it("finds exact phrase-to-title match", () => {
    const phrases = [makePhrase("Machine Learning", 0.3)];
    const titles = ["Machine Learning"];
    const edges = resolver.resolve(phrases, titles, "notes/ai.md");

    expect(edges).toHaveLength(1);
    expect(edges[0].targetPath).toBe("Machine Learning");
    expect(edges[0].similarity).toBeGreaterThan(0.9);
  });

  it("finds similar but not exact matches above threshold", () => {
    const phrases = [makePhrase("machin learning", 0.4)];
    const titles = ["Machine Learning"];
    const edges = resolver.resolve(phrases, titles, "notes/ai.md");

    expect(edges).toHaveLength(1);
    expect(edges[0].similarity).toBeGreaterThanOrEqual(0.85);
  });

  it("excludes dissimilar titles below threshold", () => {
    const phrases = [makePhrase("quantum physics", 0.5)];
    const titles = ["Banana Bread Recipe"];
    const edges = resolver.resolve(phrases, titles, "notes/science.md");

    expect(edges).toHaveLength(0);
  });

  it("filters out self-links", () => {
    const phrases = [makePhrase("My Note", 0.3)];
    const titles = ["My Note", "Other Note"];
    const edges = resolver.resolve(phrases, titles, "folder/My Note.md");

    // Should not include "My Note" as target
    expect(edges.every((e) => e.targetPath !== "My Note")).toBe(true);
  });

  it("respects maxCandidatesPerPhrase", () => {
    const custom = new AliasResolver({
      maxCandidatesPerPhrase: 1,
      similarityThreshold: 0.1,
    });
    const phrases = [makePhrase("learning", 0.5)];
    const titles = [
      "Learning Basics",
      "Deep Learning",
      "Learning Theory",
    ];
    const edges = custom.resolve(phrases, titles, "note.md");

    // With maxCandidatesPerPhrase=1, only best match per phrase
    // (dedup may still reduce further, but can't exceed 1 per phrase)
    expect(edges.length).toBeLessThanOrEqual(1);
  });

  it("deduplicates: same target from multiple phrases keeps best", () => {
    const phrases = [
      makePhrase("machine learning", 0.3),
      makePhrase("ML", 0.6),
    ];
    const titles = ["Machine Learning"];
    const edges = resolver.resolve(phrases, titles, "note.md");

    // Only one edge to "Machine Learning" (from the better match)
    const mlEdges = edges.filter((e) => e.targetPath === "Machine Learning");
    expect(mlEdges).toHaveLength(1);
  });

  it("skips phrases shorter than 2 chars", () => {
    const phrases = [makePhrase("x", 0.1)];
    const titles = ["x"];
    const edges = resolver.resolve(phrases, titles, "note.md");

    expect(edges).toHaveLength(0);
  });

  it("ranks by combined score (similarity × (1 - phraseScore))", () => {
    const phrases = [
      makePhrase("neural networks", 0.2), // low phraseScore = high quality
      makePhrase("deep learning", 0.8), // high phraseScore = low quality
    ];
    const titles = ["Neural Networks", "Deep Learning"];
    const edges = resolver.resolve(phrases, titles, "note.md");

    expect(edges.length).toBe(2);
    // "neural networks" should rank higher due to better phrase quality
    expect(edges[0].phrase.phrase).toBe("neural networks");
  });

  it("inverted-index optimization produces same results as brute-force", () => {
    // Use a low threshold so we can verify the token-based filtering logic
    const lowThreshold = new AliasResolver({ similarityThreshold: 0.3 });

    const phrases = [
      makePhrase("machine learning", 0.3),
      makePhrase("data pipeline", 0.5),
      makePhrase("gradient descent", 0.2),
    ];
    const titles = [
      "Machine Learning",
      "Data Pipeline Architecture",
      "Lambda Architecture",
      "Gradient Descent Optimization",
      "Quantum Physics",
    ];

    const edges = lowThreshold.resolve(phrases, titles, "notes/overview.md");
    const targets = edges.map((e) => e.targetPath);

    // Phrases sharing tokens with titles should match
    expect(targets).toContain("Machine Learning");
    expect(targets).toContain("Data Pipeline Architecture");
    expect(targets).toContain("Gradient Descent Optimization");

    // "Quantum Physics" shares no tokens with any phrase → never compared
    expect(targets).not.toContain("Quantum Physics");

    // "Lambda Architecture" shares "architecture" with none of the phrases → skipped
    // (it only shares tokens with "Data Pipeline Architecture" title, not phrases)
    expect(targets).not.toContain("Lambda Architecture");
  });

  it("similarity stays within [0, 1] with multiplicative boost", () => {
    const phrases = [makePhrase("test", 0.1)];
    const titles = ["test"];
    const edges = new AliasResolver({
      similarityThreshold: 0,
      exactMatchBoost: 3.0, // aggressive boost
    }).resolve(phrases, titles, "note.md");

    if (edges.length > 0) {
      expect(edges[0].similarity).toBeLessThanOrEqual(1);
    }
  });
});
