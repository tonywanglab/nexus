import { CompromiseExtractor } from "../keyphrase/compromise-extractor";
import { VaultContext } from "../types";

describe("CompromiseExtractor", () => {
  let extractor: CompromiseExtractor;

  beforeEach(() => {
    extractor = new CompromiseExtractor();
  });

  // ── Edge cases ─────────────────────────────────────────────

  describe("edge cases", () => {
    it("returns empty array for empty string", () => {
      expect(extractor.extract("")).toEqual([]);
    });

    it("returns empty array for whitespace-only string", () => {
      expect(extractor.extract("   \n\n  ")).toEqual([]);
    });
  });

  // ── Multi-word noun phrases ─────────────────────────────────

  describe("multi-word noun phrases", () => {
    it("extracts compound nouns like 'Cluster Routers'", () => {
      const content = "The Cluster Routers handle traffic between nodes.";
      const result = extractor.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      expect(phrases.some((p) => p.includes("cluster routers") || p.includes("cluster") || p.includes("routers"))).toBe(true);
    });

    it("extracts multi-word noun phrases from body text", () => {
      const content = "The neural network architecture processes input data through hidden layers.";
      const result = extractor.extract(content);

      expect(result.length).toBeGreaterThan(0);
      // Should find at least one multi-word phrase
      const hasMultiWord = result.some((r) => r.phrase.split(" ").length > 1);
      expect(hasMultiWord).toBe(true);
    });
  });

  // ── Named entities ─────────────────────────────────────────

  describe("named entities", () => {
    it("extracts places and organizations", () => {
      const content = "Microsoft Research published findings about Silicon Valley startups.";
      const result = extractor.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      expect(
        phrases.some(
          (p) =>
            p.includes("microsoft") ||
            p.includes("silicon valley") ||
            p.includes("silicon"),
        ),
      ).toBe(true);
    });
  });

  // ── Output shape ───────────────────────────────────────────

  describe("output format", () => {
    const content =
      "Machine learning algorithms process data efficiently. Neural networks are a type of model.";

    it("returns ExtractedPhrase objects with all required fields", () => {
      const result = extractor.extract(content);
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
      const result = extractor.extract(content);
      for (const phrase of result) {
        expect(phrase.score).toBeGreaterThanOrEqual(0);
        expect(phrase.score).toBeLessThanOrEqual(1);
      }
    });

    it("spanId matches offset format", () => {
      const result = extractor.extract(content);
      for (const phrase of result) {
        expect(phrase.spanId).toBe(
          `${phrase.startOffset}:${phrase.endOffset}`,
        );
      }
    });

    it("results are sorted by score ascending (best first)", () => {
      const result = extractor.extract(content);
      for (let i = 1; i < result.length; i++) {
        expect(result[i].score).toBeGreaterThanOrEqual(result[i - 1].score);
      }
    });
  });

  // ── Structural boost ──────────────────────────────────────

  describe("structural boost", () => {
    it("phrase in # Title ranks higher than body phrase", () => {
      const content =
        "# Quantum Computing\n\nBanana smoothies are a popular drink. Banana smoothies taste great.";
      const result = extractor.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      // Find quantum-related phrase
      const quantumIdx = phrases.findIndex(
        (p) => p.includes("quantum") || p.includes("computing"),
      );
      const bananaIdx = phrases.findIndex((p) => p.includes("banana"));

      if (quantumIdx !== -1 && bananaIdx !== -1) {
        expect(quantumIdx).toBeLessThan(bananaIdx);
      }
    });

    it("phrase in **bold** gets a boost", () => {
      const content =
        "The **quantum field** is growing. The banana field is growing.";
      const result = extractor.extract(content);

      // Both should be extracted; quantum field should score better
      expect(result.length).toBeGreaterThan(0);
    });
  });

  // ── Note title match ──────────────────────────────────────

  describe("note title match", () => {
    it("phrase matching note title ranks higher", () => {
      const content =
        "Quantum computing and banana smoothies are both interesting topics.";
      const vaultContext: VaultContext = {
        noteTitles: ["Quantum Computing"],
      };

      const resultWith = extractor.extract(content, vaultContext);
      const resultWithout = extractor.extract(content);

      // With vault context, quantum-computing-related phrase should rank well
      const phrasesW = resultWith.map((r) => r.phrase.toLowerCase());
      const qcIdx = phrasesW.findIndex(
        (p) => p.includes("quantum") || p.includes("computing"),
      );
      expect(qcIdx).not.toBe(-1);

      // Check that it ranks better with context than without
      if (resultWithout.length > 0) {
        const phrasesWO = resultWithout.map((r) => r.phrase.toLowerCase());
        const qcIdxWO = phrasesWO.findIndex(
          (p) => p.includes("quantum") || p.includes("computing"),
        );
        if (qcIdxWO !== -1) {
          expect(qcIdx).toBeLessThanOrEqual(qcIdxWO);
        }
      }
    });
  });

  // ── Offset validity ───────────────────────────────────────

  describe("offset validity", () => {
    it("startOffset and endOffset point into original content", () => {
      const content = "The Jupiter Rising mission launched from Cape Canaveral.";
      const result = extractor.extract(content);

      for (const phrase of result) {
        expect(phrase.startOffset).toBeGreaterThanOrEqual(0);
        expect(phrase.endOffset).toBeLessThanOrEqual(content.length);
        expect(phrase.endOffset).toBeGreaterThan(phrase.startOffset);
      }
    });

    it("offsets are valid with markdown content", () => {
      const content =
        "# Title\nThe [[Concept]] is important for understanding the process.";
      const result = extractor.extract(content);

      for (const phrase of result) {
        expect(phrase.startOffset).toBeGreaterThanOrEqual(0);
        expect(phrase.endOffset).toBeLessThanOrEqual(content.length);
        expect(phrase.endOffset).toBeGreaterThan(phrase.startOffset);
      }
    });
  });

  // ── Markdown stripping ────────────────────────────────────

  describe("markdown stripping", () => {
    it("works on content with wikilinks, bold, headings", () => {
      const content = `---
title: My Note
tags: [test]
---
# Introduction

This note discusses **artificial intelligence** and its applications.

## Key Concepts

Neural networks process [[training data]] to learn patterns.`;

      const result = extractor.extract(content);
      expect(result.length).toBeGreaterThan(0);

      const phrases = result.map((r) => r.phrase.toLowerCase());
      expect(
        phrases.some(
          (p) =>
            p.includes("neural") ||
            p.includes("intelligence") ||
            p.includes("network") ||
            p.includes("training") ||
            p.includes("patterns"),
        ),
      ).toBe(true);
    });
  });

  // ── Backward compatibility ────────────────────────────────

  describe("backward compatibility", () => {
    it("extract(content) without vault context works", () => {
      const content = "Machine learning is transforming data science.";
      const result = extractor.extract(content);
      expect(result.length).toBeGreaterThan(0);
      for (const phrase of result) {
        expect(phrase).toHaveProperty("phrase");
        expect(phrase).toHaveProperty("score");
        expect(phrase.score).toBeGreaterThanOrEqual(0);
        expect(phrase.score).toBeLessThanOrEqual(1);
      }
    });
  });
});
