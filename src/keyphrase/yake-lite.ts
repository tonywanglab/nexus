import { ExtractedPhrase } from "../types";
import { preprocess, Sentence, Token } from "./preprocessing";
import { STOPWORDS } from "./stopwords";

/**
 * Configuration for the YAKE-lite keyphrase extractor.
 */
export interface YakeLiteOptions {
  /** Maximum n-gram size (default 3: unigrams, bigrams, trigrams). */
  maxNgramSize?: number;
  /** Co-occurrence window size for relatedness (default 3). */
  windowSize?: number;
  /** Maximum number of phrases to return (default 20). */
  topN?: number;
  /** Frequency dampening factor α (default 0.5). */
  frequencyDampening?: number;
}

const DEFAULTS: Required<YakeLiteOptions> = {
  maxNgramSize: 3,
  windowSize: 3,
  topN: 20,
  frequencyDampening: 0.5,
};

/** Per-term statistics collected during the first pass. */
interface TermStats {
  /** Total term frequency across all sentences. */
  tf: number;
  /** Number of times the term appears with first letter uppercase. */
  tfUpper: number;
  /** Number of times the term appears in ALL CAPS. */
  tfAllCaps: number;
  /** Sentence indices where the term occurs. */
  sentenceIndices: number[];
  /** Left-context distinct words (within window). */
  leftContext: Set<string>;
  /** Right-context distinct words (within window). */
  rightContext: Set<string>;
  /** Total left co-occurrences. */
  leftTotal: number;
  /** Total right co-occurrences. */
  rightTotal: number;
}

/**
 * Lightweight YAKE keyphrase extractor tailored for Obsidian notes.
 *
 * Weights positional and contextual cues over frequency.
 * All processing is local — no external APIs.
 */
export class YakeLite {
  private opts: Required<YakeLiteOptions>;

  constructor(options?: YakeLiteOptions) {
    this.opts = { ...DEFAULTS, ...options };
  }

  /**
   * Extract keyphrases from Obsidian note content.
   * Returns phrases sorted by score (lower = more important),
   * with offsets pointing into the original content.
   */
  extract(content: string): ExtractedPhrase[] {
    if (!content || !content.trim()) return [];

    const { sentences } = preprocess(content);
    if (sentences.length === 0) return [];

    // ── 1. Collect per-term statistics ────────────────────────
    const stats = this.collectTermStats(sentences);
    if (stats.size === 0) return [];

    // ── 2. Compute per-term YAKE scores (H) ──────────────────
    const termScores = this.computeTermScores(stats);

    // ── 3. Generate and score n-grams ────────────────────────
    const candidates = this.scoreNgrams(sentences, termScores, stats);

    // ── 4. Deduplicate overlapping candidates ────────────────
    const deduped = this.deduplicateByOffset(candidates);

    // ── 5. Normalize scores to [0, 1] and return top N ───────
    return this.normalizeAndTruncate(deduped);
  }

  // ── Term statistics collection ─────────────────────────────

  private collectTermStats(sentences: Sentence[]): Map<string, TermStats> {
    const stats = new Map<string, TermStats>();

    const getOrCreate = (term: string): TermStats => {
      let s = stats.get(term);
      if (!s) {
        s = {
          tf: 0,
          tfUpper: 0,
          tfAllCaps: 0,
          sentenceIndices: [],
          leftContext: new Set(),
          rightContext: new Set(),
          leftTotal: 0,
          rightTotal: 0,
        };
        stats.set(term, s);
      }
      return s;
    };

    for (const sentence of sentences) {
      const tokens = sentence.tokens.filter((t) => !STOPWORDS.has(t.lower));

      for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        const term = token.lower;
        const s = getOrCreate(term);

        s.tf++;
        s.sentenceIndices.push(sentence.index);

        // Casing stats
        if (token.original.length > 0) {
          const firstChar = token.original[0];
          if (firstChar === firstChar.toUpperCase() && firstChar !== firstChar.toLowerCase()) {
            s.tfUpper++;
          }
          if (token.original === token.original.toUpperCase() &&
              token.original !== token.original.toLowerCase() &&
              token.original.length > 1) {
            s.tfAllCaps++;
          }
        }

        // Co-occurrence context (within window, among non-stopword tokens)
        for (let w = 1; w <= this.opts.windowSize; w++) {
          if (i - w >= 0) {
            s.leftContext.add(tokens[i - w].lower);
            s.leftTotal++;
          }
          if (i + w < tokens.length) {
            s.rightContext.add(tokens[i + w].lower);
            s.rightTotal++;
          }
        }
      }
    }

    return stats;
  }

  // ── Per-term YAKE score computation ────────────────────────

  private computeTermScores(stats: Map<string, TermStats>): Map<string, number> {
    const scores = new Map<string, number>();

    // Aggregate stats for frequency normalization
    const allTFs: number[] = [];
    let maxTF = 0;
    for (const s of stats.values()) {
      allTFs.push(s.tf);
      if (s.tf > maxTF) maxTF = s.tf;
    }
    const meanTF = allTFs.reduce((a, b) => a + b, 0) / allTFs.length;
    const stdTF = Math.sqrt(
      allTFs.reduce((sum, tf) => sum + (tf - meanTF) ** 2, 0) / allTFs.length
    );

    for (const [term, s] of stats) {
      // Feature 1: Casing
      // TF_C = count of capitalized occurrences, TF_U = count of all-caps
      const tfCasing = Math.max(s.tfUpper, s.tfAllCaps);
      const casing = tfCasing / (1 + Math.log2(1 + s.tf));

      // Feature 2: Position
      // Lower score = earlier = more important
      const median = this.median(s.sentenceIndices);
      const position = Math.log2(3 + median);

      // Feature 3: Frequency (dampened by α)
      const frequency = s.tf / (meanTF + stdTF + 1);

      // Feature 4: Relatedness
      // Words with diverse context (many distinct neighbors relative to total) are stopword-ish
      const wdl = s.leftContext.size;
      const wil = Math.max(s.leftTotal, 1);
      const wdr = s.rightContext.size;
      const wir = Math.max(s.rightTotal, 1);
      const relatedness = 1 + (wdl / wil + wdr / wir) * (s.tf / maxTF);

      // Feature 5: DifSentence
      const uniqueSentences = new Set(s.sentenceIndices).size;
      const difSentence = uniqueSentences / s.tf;

      // Combined score: H = (Relatedness × Position) /
      //   (Casing + α×Frequency/Relatedness + DifSentence/Relatedness)
      const α = this.opts.frequencyDampening;
      const denominator = casing + (α * frequency / relatedness) + (difSentence / relatedness);
      const H = (relatedness * position) / (denominator + 1e-10);

      scores.set(term, H);
    }

    return scores;
  }

  // ── N-gram generation and scoring ──────────────────────────

  private scoreNgrams(
    sentences: Sentence[],
    termScores: Map<string, number>,
    stats: Map<string, TermStats>,
  ): ExtractedPhrase[] {
    const candidates: Map<string, ExtractedPhrase> = new Map();

    for (const sentence of sentences) {
      const tokens = sentence.tokens;

      for (let n = 1; n <= this.opts.maxNgramSize; n++) {
        for (let i = 0; i <= tokens.length - n; i++) {
          const gram = tokens.slice(i, i + n);

          // Skip if any constituent is a stopword (except as internal words in n>2)
          // For unigrams: skip stopwords entirely
          // For n-grams: first and last word must not be stopwords
          if (STOPWORDS.has(gram[0].lower) || STOPWORDS.has(gram[gram.length - 1].lower)) {
            continue;
          }

          const phrase = gram.map((t) => t.original).join(" ");
          const phraseLower = phrase.toLowerCase();

          // Skip single-character tokens
          if (n === 1 && gram[0].lower.length <= 1) continue;

          const startOffset = gram[0].startOffset;
          const endOffset = gram[gram.length - 1].endOffset;

          // Compute n-gram score: S(kw) = ∏(H_i) / (TF_kw × (1 + ∑(H_i)))
          const constituentScores = gram
            .filter((t) => !STOPWORDS.has(t.lower))
            .map((t) => termScores.get(t.lower) ?? 1);

          if (constituentScores.length === 0) continue;

          const product = constituentScores.reduce((a, b) => a * b, 1);
          const sum = constituentScores.reduce((a, b) => a + b, 0);

          // Count n-gram frequency across all sentences
          const ngramTF = this.countNgramFrequency(sentences, gram);

          const score = product / (ngramTF * (1 + sum));

          // Keep best (lowest) score for each phrase
          const existing = candidates.get(phraseLower);
          if (!existing || score < existing.score) {
            candidates.set(phraseLower, {
              phrase,
              score,
              startOffset,
              endOffset,
              spanId: `${startOffset}:${endOffset}`,
            });
          }
        }
      }
    }

    return Array.from(candidates.values());
  }

  private countNgramFrequency(sentences: Sentence[], gram: Token[]): number {
    const target = gram.map((t) => t.lower).join(" ");
    let count = 0;

    for (const sentence of sentences) {
      for (let i = 0; i <= sentence.tokens.length - gram.length; i++) {
        const window = sentence.tokens.slice(i, i + gram.length);
        const candidate = window.map((t) => t.lower).join(" ");
        if (candidate === target) count++;
      }
    }

    return Math.max(count, 1);
  }

  // ── Deduplication ──────────────────────────────────────────

  /**
   * Remove candidates whose spans are exactly contained within a
   * higher-scoring longer phrase, or that exactly contain a higher-scoring
   * shorter phrase at the same position. Prefers longer phrases when
   * scores are close, to surface multi-word keyphrases.
   */
  private deduplicateByOffset(candidates: ExtractedPhrase[]): ExtractedPhrase[] {
    // Sort by score ascending (best first), then by phrase length descending
    // (prefer longer phrases at similar scores)
    const sorted = [...candidates].sort((a, b) => {
      const scoreDiff = a.score - b.score;
      if (Math.abs(scoreDiff) > 1e-6) return scoreDiff;
      // Tie-break: longer phrases first
      return b.phrase.length - a.phrase.length;
    });

    const result: ExtractedPhrase[] = [];

    for (const candidate of sorted) {
      // Only skip if an already-accepted candidate covers the EXACT same span
      const exactDuplicate = result.some(
        (r) =>
          r.startOffset === candidate.startOffset &&
          r.endOffset === candidate.endOffset
      );
      if (!exactDuplicate) {
        result.push(candidate);
      }
    }

    return result;
  }

  // ── Normalization ──────────────────────────────────────────

  private normalizeAndTruncate(candidates: ExtractedPhrase[]): ExtractedPhrase[] {
    if (candidates.length === 0) return [];

    // Sort by score ascending (lower = more important)
    const sorted = [...candidates].sort((a, b) => a.score - b.score);
    const topN = sorted.slice(0, this.opts.topN);

    if (topN.length === 0) return [];

    const minScore = topN[0].score;
    const maxScore = topN[topN.length - 1].score;
    const range = maxScore - minScore;

    return topN.map((p) => ({
      ...p,
      score: range > 0 ? (p.score - minScore) / range : 0,
      spanId: `${p.startOffset}:${p.endOffset}`,
    }));
  }

  // ── Utilities ──────────────────────────────────────────────

  private median(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }
}
