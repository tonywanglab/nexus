import { ExtractedPhrase, VaultContext } from "../types";
import { preprocess } from "./preprocessing";
import { STOPWORDS } from "./stopwords";
import { normalizeScores } from "./scoring";
import { makeYielder } from "../resolver/shared-utils";

/**
 * Configuration for the span-based keyphrase extractor.
 */
export interface SpanExtractorOptions {
  /** Maximum n-gram window size in tokens/words (default 10). */
  maxNgramSize?: number;
  /** Maximum number of phrases to return (default Infinity). */
  topN?: number;
}

const DEFAULTS: Required<SpanExtractorOptions> = {
  maxNgramSize: 10,
  topN: Infinity,
};

/**
 * High-recall span-based candidate generator.
 *
 * Generates all contiguous token windows (1..maxNgramSize) per sentence,
 * applies minimal filtering (stopword tail, single-char unigrams), deduplicates
 * by phrase text, and assigns a flat score of 0.  Vault-title scan catches
 * titles that span sentence boundaries or exceed the token window.
 */
export class SpanExtractor {
  private opts: Required<SpanExtractorOptions>;

  constructor(options?: SpanExtractorOptions) {
    this.opts = { ...DEFAULTS, ...options };
  }

  async extract(content: string, vaultContext?: VaultContext): Promise<ExtractedPhrase[]> {
    if (!content || !content.trim()) return [];

    const { sentences, stripped, offsetMap } = preprocess(content);
    if (sentences.length === 0) return [];

    // Case-insensitive dedup map: lowered phrase → ExtractedPhrase (first occurrence wins)
    const seen = new Map<string, ExtractedPhrase>();

    // ── 1. Generate all contiguous token windows per sentence.
    // Time-budget yielder keeps the UI responsive on large documents. ──
    const yieldIfNeeded = makeYielder();
    for (const sentence of sentences) {
      await yieldIfNeeded();
      const tokens = sentence.tokens;

      for (let n = 1; n <= Math.min(this.opts.maxNgramSize, tokens.length); n++) {
        for (let i = 0; i <= tokens.length - n; i++) {
          const gram = tokens.slice(i, i + n);

          // Skip if last token is a stopword
          if (STOPWORDS.has(gram[gram.length - 1].lower)) continue;

          // Skip single-char unigrams
          if (n === 1 && gram[0].lower.length <= 1) continue;

          const phrase = gram.map((t) => t.original).join(" ");
          const phraseLower = phrase.toLowerCase();

          // Keep first occurrence only
          if (seen.has(phraseLower)) continue;

          const startOffset = gram[0].startOffset;
          const endOffset = gram[gram.length - 1].endOffset;

          seen.set(phraseLower, {
            phrase,
            score: 0,
            startOffset,
            endOffset,
            spanId: `${startOffset}:${endOffset}`,
          });
        }
      }
    }

    // ── 2. Vault-title scan ──
    if (vaultContext?.noteTitles) {
      for (const result of await this.scanVaultTitles(stripped, offsetMap, vaultContext)) {
        const key = result.phrase.toLowerCase();
        if (!seen.has(key)) {
          seen.set(key, result);
        }
      }
    }

    const candidates = Array.from(seen.values());

    // ── 3. Normalize scores and apply topN ──
    return normalizeScores(candidates, this.opts.topN);
  }

  /**
   * Scans stripped text for verbatim occurrences of vault note titles.
   * Returns one ExtractedPhrase per matched title (first occurrence only),
   * with score=0 so they rank highly in topN selection. Time-budget yielder
   * keeps the UI responsive for large vaults (each title is a fresh regex
   * compile + exec — hundreds of titles can otherwise pin the main thread).
   */
  private async scanVaultTitles(
    stripped: string,
    offsetMap: number[],
    vaultContext: VaultContext,
  ): Promise<ExtractedPhrase[]> {
    const { noteTitles } = vaultContext;
    if (!noteTitles || noteTitles.length === 0) return [];

    const results: ExtractedPhrase[] = [];
    const yieldIfNeeded = makeYielder();

    for (const title of noteTitles) {
      await yieldIfNeeded();
      if (!title || title.length < 2) continue;

      const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const pattern = new RegExp(`(?<![a-zA-Z0-9])${escaped}(?![a-zA-Z0-9])`, "gi");

      const m = pattern.exec(stripped);
      if (!m) continue;

      const matchStart = m.index;
      const matchEnd = m.index + m[0].length;
      const origStart = offsetMap[matchStart] ?? matchStart;
      const origEnd = offsetMap[matchEnd - 1] !== undefined
        ? offsetMap[matchEnd - 1] + 1
        : matchEnd;

      results.push({
        phrase: title,
        score: 0,
        startOffset: origStart,
        endOffset: origEnd,
        spanId: `${origStart}:${origEnd}`,
      });
    }

    return results;
  }
}
