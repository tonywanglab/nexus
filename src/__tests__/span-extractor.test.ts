import { SpanExtractor } from "../keyphrase/span-extractor";
import { ExtractedPhrase, VaultContext } from "../types";

describe("SpanExtractor", () => {
  let extractor: SpanExtractor;

  beforeEach(() => {
    extractor = new SpanExtractor();
  });

  // ── Edge cases ─────────────────────────────────────────────────

  describe("edge cases", () => {
    it("returns empty array for empty string", async () => {
      expect(await extractor.extract("")).toEqual([]);
    });

    it("returns empty array for whitespace-only", async () => {
      expect(await extractor.extract("   \n\n  \t  ")).toEqual([]);
    });

    it("returns empty array for stopwords-only text", async () => {
      const result = await extractor.extract("the and or but is are");
      // All spans end with a stopword, so all should be filtered
      // except spans where the last word is not a stopword — "the" etc. are stopwords
      expect(result.length).toBe(0);
    });

    it("handles single word input", async () => {
      const result = await extractor.extract("Algorithms");
      expect(result.length).toBeGreaterThanOrEqual(1);
      expect(result[0].phrase).toBe("Algorithms");
    });
  });

  // ── Output format ──────────────────────────────────────────────

  describe("output format", () => {
    it("produces ExtractedPhrase with all required fields", async () => {
      const result = await extractor.extract("Machine learning is transformative.");
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

    it("scores are in [0, 1]", async () => {
      const result = await extractor.extract("Machine learning and deep learning are subfields of artificial intelligence.");
      for (const phrase of result) {
        expect(phrase.score).toBeGreaterThanOrEqual(0);
        expect(phrase.score).toBeLessThanOrEqual(1);
      }
    });

    it("spanId matches startOffset:endOffset format", async () => {
      const result = await extractor.extract("Neural networks process data.");
      for (const phrase of result) {
        expect(phrase.spanId).toBe(`${phrase.startOffset}:${phrase.endOffset}`);
      }
    });
  });

  // ── Span generation ────────────────────────────────────────────

  describe("span generation", () => {
    it("generates multi-word spans up to default 10 tokens", async () => {
      const tokens = Array.from({ length: 12 }, (_, i) => `word${i}`);
      const text = tokens.join(" ");
      const result = await extractor.extract(text);

      // Should have spans of various lengths
      const phraseLengths = result.map((p) => p.phrase.split(" ").length);
      expect(Math.max(...phraseLengths)).toBeLessThanOrEqual(10);
      expect(phraseLengths).toContain(1);
      expect(phraseLengths.some((l) => l > 1)).toBe(true);
    });

    it("respects maxNgramSize option", async () => {
      const custom = new SpanExtractor({ maxNgramSize: 3 });
      const text = "alpha beta gamma delta epsilon";
      const result = await custom.extract(text);

      for (const phrase of result) {
        const wordCount = phrase.phrase.split(" ").length;
        expect(wordCount).toBeLessThanOrEqual(3);
      }
    });
  });

  // ── Stopword filtering ─────────────────────────────────────────

  describe("stopword filtering", () => {
    it("does not produce spans ending with a stopword", async () => {
      const result = await extractor.extract("The theory of computation is fundamental.");
      for (const phrase of result) {
        const words = phrase.phrase.split(" ");
        const lastWord = words[words.length - 1].toLowerCase();
        // Common stopwords: "the", "of", "is"
        expect(["the", "of", "is"].includes(lastWord)).toBe(false);
      }
    });

    it("allows first token to be a stopword", async () => {
      // "the algorithm" should be valid since "algorithm" is the last token
      const result = await extractor.extract("The algorithm works well.");
      const phrases = result.map((p) => p.phrase.toLowerCase());
      expect(phrases).toContain("the algorithm");
    });

    it("allows internal stopwords", async () => {
      const result = await extractor.extract("Theory of computation matters.");
      const phrases = result.map((p) => p.phrase.toLowerCase());
      expect(phrases).toContain("theory of computation");
    });
  });

  // ── Deduplication ──────────────────────────────────────────────

  describe("deduplication", () => {
    it("deduplicates same phrase text (case-insensitive)", async () => {
      const result = await extractor.extract("Machine learning is great. machine learning is used widely.");
      const mlCount = result.filter(
        (p) => p.phrase.toLowerCase() === "machine learning"
      ).length;
      expect(mlCount).toBe(1);
    });
  });

  // ── Vault-title scan ───────────────────────────────────────────

  describe("vault-title scan", () => {
    it("includes vault titles found verbatim in text", async () => {
      const vaultContext: VaultContext = {
        noteTitles: ["Data Warehouse"],
      };
      const result = await extractor.extract(
        "A data warehouse stores historical data for analysis.",
        vaultContext,
      );
      const phrases = result.map((p) => p.phrase.toLowerCase());
      expect(phrases).toContain("data warehouse");
    });

    it("works without vaultContext", async () => {
      const result = await extractor.extract("Some text about algorithms.");
      expect(result.length).toBeGreaterThan(0);
    });
  });

  // ── Markdown handling ──────────────────────────────────────────

  describe("markdown handling", () => {
    it("strips frontmatter", async () => {
      const content = `---
title: Test Note
tags: [test]
---

Machine learning is powerful.`;
      const result = await extractor.extract(content);
      const phrases = result.map((p) => p.phrase.toLowerCase());
      expect(phrases).not.toContain("title");
      expect(phrases).not.toContain("tags");
    });

    it("produces valid offsets into original content", async () => {
      const content = "# Heading\n\nSome **bold** text about algorithms.";
      const result = await extractor.extract(content);

      for (const phrase of result) {
        expect(phrase.startOffset).toBeGreaterThanOrEqual(0);
        expect(phrase.endOffset).toBeLessThanOrEqual(content.length);
        expect(phrase.endOffset).toBeGreaterThan(phrase.startOffset);
      }
    });
  });

  // ── topN ───────────────────────────────────────────────────────

  describe("topN", () => {
    it("respects topN limit", async () => {
      const limited = new SpanExtractor({ topN: 5 });
      const text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu.";
      const result = await limited.extract(text);
      expect(result.length).toBeLessThanOrEqual(5);
    });
  });
});
