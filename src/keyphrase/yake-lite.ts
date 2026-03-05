import { ExtractedPhrase, VaultContext } from "../types";
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
  /** Boost applied when a phrase matches an existing note title (default 5). */
  noteMatchBoost?: number;
  /** Boost applied to noun-phrase-shaped n-grams (default 4). */
  nounPhraseBoost?: number;
}

const DEFAULTS: Required<YakeLiteOptions> = {
  maxNgramSize: 3,
  windowSize: 3,
  topN: Infinity,
  frequencyDampening: 0.5,
  noteMatchBoost: 5,
  nounPhraseBoost: 4,
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

// ── Lightweight POS heuristics (suffix-based) ─────────────────

const NOUN_SUFFIXES = [
  "tion", "sion", "ment", "ness", "ity", "ence", "ance", "ism", "ist",
  "ics", "ology", "phy", "dom", "ship", "ure", "age", "ery", "ium",
  "sis", "oma", "oid", "ule", "ette",
];

const ADJ_SUFFIXES = [
  "ical", "ial", "ful", "ous", "ive", "able", "ible", "less", "ular",
  "ary", "ory", "ent", "ant", "ic", "al", "ed",
];

const VERB_SUFFIXES = [
  "ize", "ise", "ify", "ate",
];

const ADVERB_SUFFIX = "ly";

type PosGuess = "noun" | "adj" | "verb" | "adverb" | "unknown";

function guessPOS(word: string): PosGuess {
  const w = word.toLowerCase();

  // Short words (<=3 chars) — don't guess by suffix
  if (w.length <= 3) return "unknown";

  // Adverb check first (before adj, since "ally" ends in "ly")
  if (w.endsWith(ADVERB_SUFFIX) && w.length > 4) return "adverb";

  // Gerunds/present participles ending in -ing: treat as noun (gerund) unless
  // the word is very short, since "computing", "learning", etc. are noun-like
  if (w.endsWith("ing") && w.length > 5) return "noun";

  // Check verb suffixes before noun/adj (since some overlap)
  for (const s of VERB_SUFFIXES) {
    if (w.endsWith(s) && w.length > s.length + 2) return "verb";
  }

  for (const s of NOUN_SUFFIXES) {
    if (w.endsWith(s)) return "noun";
  }

  for (const s of ADJ_SUFFIXES) {
    if (w.endsWith(s)) return "adj";
  }

  // Capitalized words in the middle of a sentence are likely proper nouns
  if (word[0] === word[0].toUpperCase() && word[0] !== word[0].toLowerCase()) {
    return "noun";
  }

  return "unknown";
}

/**
 * Check if an n-gram looks like a noun phrase: (adj|noun|unknown)* noun/unknown
 * where the head (last non-stopword) must be noun-like or unknown,
 * and no word is a verb or adverb.
 */
function isNounPhrase(words: string[]): boolean {
  if (words.length === 0) return false;

  for (let i = 0; i < words.length; i++) {
    const pos = guessPOS(words[i]);
    if (pos === "verb" || pos === "adverb") return false;
  }

  // Head word (last) should be noun or unknown (not adj-only)
  const headPos = guessPOS(words[words.length - 1]);
  return headPos === "noun" || headPos === "unknown";
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
   *
   * When `vaultContext` is provided, TF-IDF distinctiveness and
   * existing-note-match heuristics are applied. Structural boosts
   * (title, heading, bold) are always applied.
   */
  extract(content: string, vaultContext?: VaultContext): ExtractedPhrase[] {
    if (!content || !content.trim()) return [];

    const { sentences } = preprocess(content);
    if (sentences.length === 0) return [];

    // ── 1. Collect per-term statistics ────────────────────────
    const stats = this.collectTermStats(sentences);
    if (stats.size === 0) return [];

    // ── 2. Compute per-term structural boost (max across occurrences) ─
    const structuralBoosts = this.collectStructuralBoosts(sentences);

    // ── 3. Compute per-term YAKE scores (H) ──────────────────
    const termScores = this.computeTermScores(stats);

    // ── 4. Generate and score n-grams ────────────────────────
    const candidates = this.scoreNgrams(
      sentences, termScores, stats, structuralBoosts, vaultContext,
    );

    // ── 5. Deduplicate overlapping candidates ────────────────
    const deduped = this.deduplicateByOffset(candidates);

    // ── 6. Normalize scores to [0, 1] and return top N ───────
    return this.normalizeAndTruncate(deduped);
  }

  // ── Structural boost collection ──────────────────────────────

  private collectStructuralBoosts(sentences: Sentence[]): Map<string, number> {
    const boosts = new Map<string, number>();
    for (const sentence of sentences) {
      for (const token of sentence.tokens) {
        if (STOPWORDS.has(token.lower)) continue;
        const current = boosts.get(token.lower) ?? 0;
        if (token.structuralBoost > current) {
          boosts.set(token.lower, token.structuralBoost);
        }
      }
    }
    return boosts;
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
    structuralBoosts: Map<string, number>,
    vaultContext?: VaultContext,
  ): ExtractedPhrase[] {
    const candidates: Map<string, ExtractedPhrase> = new Map();

    // Pre-compute lowercased note titles for matching
    const noteTitlesLower = vaultContext?.noteTitles?.map((t) => t.toLowerCase()) ?? [];

    // Pre-compute max TF-IDF for normalization
    let maxTfIdf = 0;
    const tfidfCache = new Map<string, number>();
    if (vaultContext?.documentFrequencies && vaultContext.totalDocuments) {
      const { documentFrequencies, totalDocuments } = vaultContext;
      for (const [term, s] of stats) {
        const tf = s.tf;
        const docFreq = documentFrequencies.get(term) ?? 0;
        const idf = Math.log(totalDocuments / (1 + docFreq));
        const tfidf = tf * idf;
        tfidfCache.set(term, tfidf);
        if (tfidf > maxTfIdf) maxTfIdf = tfidf;
      }
    }

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

          // Compute n-gram score using geometric-mean normalization so
          // multi-word phrases compete fairly with unigrams:
          //   S(kw) = geomean(H_i) / (TF_kw × (1 + mean(H_i)))
          const constituentScores = gram
            .filter((t) => !STOPWORDS.has(t.lower))
            .map((t) => termScores.get(t.lower) ?? 1);

          if (constituentScores.length === 0) continue;

          const k = constituentScores.length;
          const product = constituentScores.reduce((a, b) => a * b, 1);
          const geomean = Math.pow(product, 1 / k);
          const mean = constituentScores.reduce((a, b) => a + b, 0) / k;

          // Count n-gram frequency across all sentences
          const ngramTF = this.countNgramFrequency(sentences, gram);

          let rawScore = geomean / (ngramTF * (1 + mean));

          // ── Heuristic boosts (applied as divisors) ───────────

          // 1. Structural boost: max across constituent terms
          let totalBoost = 0;
          const nonStopGram = gram.filter((t) => !STOPWORDS.has(t.lower));
          for (const t of nonStopGram) {
            const sb = structuralBoosts.get(t.lower) ?? 0;
            if (sb > totalBoost) totalBoost = sb;
          }

          // 2. TF-IDF boost (when vault context provided)
          if (maxTfIdf > 0) {
            let phraseMaxTfIdf = 0;
            for (const t of nonStopGram) {
              const v = tfidfCache.get(t.lower) ?? 0;
              if (v > phraseMaxTfIdf) phraseMaxTfIdf = v;
            }
            // Normalize to [0, 1] range, then scale to a reasonable boost
            totalBoost += (phraseMaxTfIdf / maxTfIdf) * 2;
          }

          // 3. Note match boost (when vault context provided)
          if (noteTitlesLower.length > 0 && noteTitlesLower.includes(phraseLower)) {
            totalBoost += this.opts.noteMatchBoost;
          }

          // 4. Noun phrase boost
          const nonStopWords = nonStopGram.map((t) => t.original);
          if (isNounPhrase(nonStopWords)) {
            totalBoost += this.opts.nounPhraseBoost;
          }

          // Apply boosts: lower score = more important
          const score = rawScore / (1 + totalBoost);

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
