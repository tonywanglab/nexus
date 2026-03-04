import { YakeLite } from "../keyphrase/yake-lite";
import { ExtractedPhrase } from "../types";

describe("YakeLite", () => {
  let yake: YakeLite;

  beforeEach(() => {
    yake = new YakeLite();
  });

  // ── Edge cases ─────────────────────────────────────────────

  describe("edge cases", () => {
    it("returns empty array for empty string", () => {
      expect(yake.extract("")).toEqual([]);
    });

    it("returns empty array for whitespace-only string", () => {
      expect(yake.extract("   \n\n  ")).toEqual([]);
    });

    it("returns empty array for stopwords-only content", () => {
      expect(yake.extract("the a an is are was were")).toEqual([]);
    });

    it("handles single-word content", () => {
      const result = yake.extract("TypeScript");
      expect(result.length).toBeGreaterThanOrEqual(1);
      expect(result[0].phrase.toLowerCase()).toBe("typescript");
    });
  });

  // ── Output shape ───────────────────────────────────────────

  describe("output format", () => {
    const content = "Machine learning algorithms process data efficiently. Neural networks are a type of machine learning model.";

    it("returns ExtractedPhrase objects with all required fields", () => {
      const result = yake.extract(content);
      expect(result.length).toBeGreaterThan(0);

      for (const phrase of result) {
        expect(phrase).toHaveProperty("phrase");
        expect(phrase).toHaveProperty("score");
        expect(phrase).toHaveProperty("startOffset");
        expect(phrase).toHaveProperty("endOffset");
        expect(phrase).toHaveProperty("spanId");
        expect(typeof phrase.phrase).toBe("string");
        expect(typeof phrase.score).toBe("number");
        expect(typeof phrase.startOffset).toBe("number");
        expect(typeof phrase.endOffset).toBe("number");
        expect(typeof phrase.spanId).toBe("string");
      }
    });

    it("scores are normalized to [0, 1]", () => {
      const result = yake.extract(content);
      for (const phrase of result) {
        expect(phrase.score).toBeGreaterThanOrEqual(0);
        expect(phrase.score).toBeLessThanOrEqual(1);
      }
    });

    it("spanId matches offset format", () => {
      const result = yake.extract(content);
      for (const phrase of result) {
        expect(phrase.spanId).toBe(`${phrase.startOffset}:${phrase.endOffset}`);
      }
    });

    it("results are sorted by score ascending (best first)", () => {
      const result = yake.extract(content);
      for (let i = 1; i < result.length; i++) {
        expect(result[i].score).toBeGreaterThanOrEqual(result[i - 1].score);
      }
    });
  });

  // ── Keyphrase quality ──────────────────────────────────────

  describe("keyphrase quality", () => {
    it("extracts meaningful terms, not stopwords", () => {
      const content = "The quick brown fox jumps over the lazy dog. The fox is very quick and nimble.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      // Should find content words
      expect(phrases.some((p) => p.includes("fox") || p.includes("quick") || p.includes("brown"))).toBe(true);

      // Should NOT find pure stopwords as phrases
      for (const phrase of phrases) {
        const words = phrase.split(" ");
        // At least one non-stopword in every phrase
        expect(words.some((w) => !["the", "a", "an", "is", "are", "and", "or", "over", "very"].includes(w))).toBe(true);
      }
    });

    it("favors terms appearing early in the document", () => {
      const content = "Quantum computing revolutionizes cryptography. " +
        "Many researchers study this field. " +
        "The implications are vast and wide-ranging. " +
        "Banana smoothies are also delicious.";
      const result = yake.extract(content);

      // "quantum" or "computing" or "cryptography" should rank higher than "banana" or "smoothies"
      const topPhrases = result.slice(0, 5).map((r) => r.phrase.toLowerCase());
      const earlyTermPresent = topPhrases.some((p) =>
        p.includes("quantum") || p.includes("computing") || p.includes("cryptography")
      );
      expect(earlyTermPresent).toBe(true);
    });

    it("detects capitalized terms as important", () => {
      const content = "The API uses REST endpoints. The api handles JSON responses. The API documentation is thorough.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());
      expect(phrases.some((p) => p.includes("api"))).toBe(true);
    });
  });

  // ── N-gram handling ────────────────────────────────────────

  describe("n-grams", () => {
    it("extracts multi-word phrases", () => {
      const content = "Machine learning is transforming data science. Machine learning models require large datasets. Deep learning architectures have advanced rapidly.";
      const yakeNgram = new YakeLite({ maxNgramSize: 3, topN: 30 });
      const result = yakeNgram.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      // Should find at least one multi-word phrase
      const hasMultiWord = phrases.some((p) => p.split(" ").length > 1);
      expect(hasMultiWord).toBe(true);
    });

    it("respects maxNgramSize=1 (unigrams only)", () => {
      const content = "Machine learning is transforming data science.";
      const yakeUnigram = new YakeLite({ maxNgramSize: 1 });
      const result = yakeUnigram.extract(content);

      for (const phrase of result) {
        expect(phrase.phrase.split(" ").length).toBe(1);
      }
    });
  });

  // ── Configuration ──────────────────────────────────────────

  describe("configuration", () => {
    it("respects topN limit", () => {
      const content = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon.";
      const yakeSmall = new YakeLite({ topN: 3 });
      const result = yakeSmall.extract(content);
      expect(result.length).toBeLessThanOrEqual(3);
    });

    it("accepts custom frequency dampening", () => {
      const content = "Testing frequency dampening factor. Testing repeated words. Testing is important.";
      const yake1 = new YakeLite({ frequencyDampening: 0.1 });
      const yake2 = new YakeLite({ frequencyDampening: 0.9 });
      const result1 = yake1.extract(content);
      const result2 = yake2.extract(content);

      // Both should produce results; exact scores will differ
      expect(result1.length).toBeGreaterThan(0);
      expect(result2.length).toBeGreaterThan(0);
    });
  });

  // ── Markdown handling ──────────────────────────────────────

  describe("markdown content", () => {
    it("handles frontmatter + markdown", () => {
      const content = `---
title: My Note
tags: [test]
---
# Introduction

This note discusses **artificial intelligence** and its applications.

## Key Concepts

Neural networks process [[training data]] to learn patterns.`;

      const result = yake.extract(content);
      expect(result.length).toBeGreaterThan(0);

      // Should extract meaningful terms from the content
      const phrases = result.map((r) => r.phrase.toLowerCase());
      expect(phrases.some((p) =>
        p.includes("neural") || p.includes("artificial") ||
        p.includes("intelligence") || p.includes("networks") ||
        p.includes("training") || p.includes("patterns")
      )).toBe(true);
    });

    it("offsets point to valid positions in original content", () => {
      const content = "# Title\nThe [[Concept]] is important for understanding.";
      const result = yake.extract(content);

      for (const phrase of result) {
        expect(phrase.startOffset).toBeGreaterThanOrEqual(0);
        expect(phrase.endOffset).toBeLessThanOrEqual(content.length);
        expect(phrase.endOffset).toBeGreaterThan(phrase.startOffset);
      }
    });
  });
});
