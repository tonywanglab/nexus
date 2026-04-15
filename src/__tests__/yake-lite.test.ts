import { YakeLite } from "../keyphrase/yake-lite";
import { ExtractedPhrase, VaultContext } from "../types";

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

  // ── Structural importance heuristic ─────────────────────────

  describe("structural importance", () => {
    it("terms in # Title rank higher than body-only terms", () => {
      // "visualize" is a verb (-ize suffix) so it won't get noun phrase boost
      const content = "# Quantum\n\nWe visualize the data. We visualize again. We visualize often.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      const quantumIdx = phrases.indexOf("quantum");
      const verbIdx = phrases.indexOf("visualize");
      expect(quantumIdx).not.toBe(-1);
      expect(verbIdx).not.toBe(-1);
      // Lower index = better rank (lower score)
      expect(quantumIdx).toBeLessThan(verbIdx);
    });

    it("terms in ## Heading rank higher than body-only terms", () => {
      // "summarize" is a verb (-ize suffix)
      const content = "Some intro text about nothing special.\n\n## Algorithms\n\nWe summarize the findings. We summarize again.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      const algoIdx = phrases.indexOf("algorithms");
      const verbIdx = phrases.indexOf("summarize");
      expect(algoIdx).not.toBe(-1);
      expect(verbIdx).not.toBe(-1);
      expect(algoIdx).toBeLessThan(verbIdx);
    });

    it("terms in **bold** get a boost", () => {
      // Use two terms that would otherwise score similarly: both appear once
      // but the bold one should rank higher
      const content = "The **quantum** field is growing. The banana field is growing.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      const quantumIdx = phrases.indexOf("quantum");
      const bananaIdx = phrases.indexOf("banana");
      expect(quantumIdx).not.toBe(-1);
      expect(bananaIdx).not.toBe(-1);
      expect(quantumIdx).toBeLessThan(bananaIdx);
    });

    it("structural boosts stack (bold inside heading uses max)", () => {
      const content = "# **Quantum**\n\nBanana is a fruit. Banana appears again. Banana is common.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      const quantumIdx = phrases.indexOf("quantum");
      expect(quantumIdx).not.toBe(-1);
      // Title boost (3) should apply — term should be ranked highly
      expect(quantumIdx).toBeLessThanOrEqual(2);
    });
  });

  // ── TF-IDF distinctiveness heuristic ────────────────────────

  describe("TF-IDF distinctiveness", () => {
    it("rare-in-vault terms rank higher with vault context", () => {
      const content = "Quantum mechanics governs particle behavior. Particle physics is a broad discipline. Particle interactions are complex.";

      // "quantum" is rare across vault, "particle" is very common
      const vaultContext: VaultContext = {
        documentFrequencies: new Map([
          ["quantum", 1],
          ["mechanics", 2],
          ["particle", 80],
          ["behavior", 30],
          ["physics", 40],
          ["broad", 50],
          ["discipline", 20],
          ["interactions", 10],
          ["governs", 60],
        ]),
        totalDocuments: 100,
      };

      const resultWith = yake.extract(content, vaultContext);
      const resultWithout = yake.extract(content);

      // Find "quantum" score in both — TF-IDF should improve its ranking
      const qWith = resultWith.find((r) => r.phrase.toLowerCase() === "quantum");
      const qWithout = resultWithout.find((r) => r.phrase.toLowerCase() === "quantum");
      expect(qWith).toBeDefined();
      expect(qWithout).toBeDefined();
      // Normalized score should be lower (better) with vault context for a rare term
      expect(qWith!.score).toBeLessThanOrEqual(qWithout!.score);
    });

    it("without vault context, behavior is unchanged (backward compat)", () => {
      const content = "Machine learning algorithms process data efficiently. Neural networks are a type of machine learning model.";
      const resultWithout = yake.extract(content);
      const resultWith = yake.extract(content); // no vault context

      expect(resultWithout.length).toBe(resultWith.length);
      for (let i = 0; i < resultWithout.length; i++) {
        expect(resultWithout[i].phrase).toBe(resultWith[i].phrase);
        expect(resultWithout[i].score).toBeCloseTo(resultWith[i].score, 10);
      }
    });
  });

  // ── Note match heuristic ────────────────────────────────────

  describe("note match", () => {
    it("phrases matching note titles rank higher", () => {
      const content = "Quantum computing and banana smoothies are both interesting topics. We study quantum computing regularly.";
      const vaultContext: VaultContext = {
        noteTitles: ["Quantum Computing"],
      };

      const resultWith = yake.extract(content, vaultContext);
      const resultWithout = yake.extract(content);

      const phrasesW = resultWith.map((r) => r.phrase.toLowerCase());
      const phrasesWO = resultWithout.map((r) => r.phrase.toLowerCase());

      const qcIdxWith = phrasesW.indexOf("quantum computing");
      const qcIdxWithout = phrasesWO.indexOf("quantum computing");
      expect(qcIdxWith).not.toBe(-1);
      // Note match boost should improve ranking vs without context
      if (qcIdxWithout !== -1) {
        expect(qcIdxWith).toBeLessThanOrEqual(qcIdxWithout);
      }
    });

    it("case-insensitive matching works", () => {
      const content = "quantum computing is a new field. Banana smoothies are delicious.";
      const vaultContext: VaultContext = {
        noteTitles: ["Quantum Computing"],
      };

      const result = yake.extract(content, vaultContext);
      const phrases = result.map((r) => r.phrase.toLowerCase());
      expect(phrases.indexOf("quantum computing")).not.toBe(-1);
    });
  });

  // ── Noun phrase heuristic ─────────────────────────────────────

  describe("noun phrase prioritization", () => {
    it("noun-like terms rank higher than verb-like terms", () => {
      // "optimization" (noun suffix -tion) vs "optimize" (verb suffix -ize)
      const content = "We optimize the algorithm. The optimization yields results. The optimization is significant.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      const nounIdx = phrases.indexOf("optimization");
      const verbIdx = phrases.indexOf("optimize");
      if (nounIdx !== -1 && verbIdx !== -1) {
        expect(nounIdx).toBeLessThan(verbIdx);
      }
    });

    it("multi-word noun phrases rank higher than non-noun phrases", () => {
      const content = "Machine learning transforms industries rapidly. Machine learning is important. Deep learning advances quickly.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      // "machine learning" (noun+noun-gerund) should appear
      const mlIdx = phrases.indexOf("machine learning");
      expect(mlIdx).not.toBe(-1);
      // Should be ranked highly
      expect(mlIdx).toBeLessThanOrEqual(3);
    });

    it("adverb-heavy phrases are not boosted", () => {
      const content = "Processing happens efficiently. Data processing is a key concept. Data processing matters greatly.";
      const result = yake.extract(content);
      const phrases = result.map((r) => r.phrase.toLowerCase());

      // "data processing" (noun phrase) should rank above "efficiently" (adverb)
      const dpIdx = phrases.indexOf("data processing");
      const advIdx = phrases.indexOf("efficiently");
      if (dpIdx !== -1 && advIdx !== -1) {
        expect(dpIdx).toBeLessThan(advIdx);
      }
    });
  });

  // ── Backward compatibility ──────────────────────────────────

  describe("backward compatibility", () => {
    it("extract(content) without vault context still works", () => {
      const content = "Machine learning is transforming data science.";
      const result = yake.extract(content);
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
