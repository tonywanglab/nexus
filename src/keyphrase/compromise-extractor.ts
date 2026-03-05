/**
 * Keyphrase extractor using compromise.js for POS-based noun chunking.
 *
 * Complements YakeLite for short notes where statistical signals are weak.
 * Uses real POS tagging to find multi-word noun phrases and named entities.
 */

import nlp from "compromise";
import { ExtractedPhrase, VaultContext } from "../types";
import { preprocess, getBoostForRange } from "./preprocessing";

export interface CompromiseExtractorOptions {
  /** Boost applied when a phrase matches an existing note title (default 5). */
  noteMatchBoost?: number;
  /** Maximum number of phrases to return (default 20). */
  topN?: number;
}

const DEFAULTS: Required<CompromiseExtractorOptions> = {
  noteMatchBoost: 5,
  topN: 20,
};

export class CompromiseExtractor {
  private opts: Required<CompromiseExtractorOptions>;

  constructor(options?: CompromiseExtractorOptions) {
    this.opts = { ...DEFAULTS, ...options };
  }

  /**
   * Extract keyphrases from Obsidian note content using compromise.js.
   * Returns phrases sorted by score (lower = more important),
   * with offsets pointing into the original content.
   */
  extract(content: string, vaultContext?: VaultContext): ExtractedPhrase[] {
    if (!content || !content.trim()) return [];

    const { offsetMap, structuralRanges, stripped } = preprocess(content);
    if (!stripped.trim()) return [];

    // ── 1. Extract noun phrases and topics via compromise ────
    const doc = nlp(stripped);
    const nouns = doc.nouns().out("array") as string[];
    const topics = doc.topics().out("array") as string[];

    // ── 2. Deduplicate phrase texts (case-insensitive) ───────
    const seen = new Set<string>();
    const uniquePhrases: string[] = [];

    for (const phrase of [...nouns, ...topics]) {
      const trimmed = phrase.trim();
      if (!trimmed) continue;
      const key = trimmed.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        uniquePhrases.push(trimmed);
      }
    }

    if (uniquePhrases.length === 0) return [];

    // Pre-compute lowercased note titles
    const noteTitlesLower =
      vaultContext?.noteTitles?.map((t) => t.toLowerCase()) ?? [];

    // ── 3. Locate phrases in stripped text, compute scores ───
    const candidates: ExtractedPhrase[] = [];

    for (const phraseText of uniquePhrases) {
      // Find all occurrences in stripped text for frequency count
      const escapedPhrase = phraseText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const phraseRe = new RegExp(`\\b${escapedPhrase}\\b`, "gi");
      const matches = [...stripped.matchAll(phraseRe)];

      if (matches.length === 0) continue;

      const frequency = matches.length;

      // Use first occurrence for offsets
      const firstMatch = matches[0];
      const strippedStart = firstMatch.index!;
      const strippedEnd = strippedStart + firstMatch[0].length;

      // Map back to original content offsets
      const origStart = offsetMap[strippedStart] ?? strippedStart;
      const origEnd =
        strippedEnd - 1 < offsetMap.length
          ? (offsetMap[strippedEnd - 1] ?? strippedEnd - 1) + 1
          : strippedEnd;

      // ── Scoring (lower = more important) ─────────────────
      const structuralBoost = getBoostForRange(
        origStart,
        origEnd,
        structuralRanges,
      );

      const wordCount = phraseText.split(/\s+/).length;
      const phraseLower = phraseText.toLowerCase();
      const matchesNoteTitle =
        noteTitlesLower.length > 0 && noteTitlesLower.includes(phraseLower);

      let raw = 1.0;
      raw /= 1 + structuralBoost;
      raw /= 1 + (matchesNoteTitle ? this.opts.noteMatchBoost : 0);
      raw /= 1 + (wordCount - 1) * 0.5;
      raw /= 1 + Math.log1p(frequency);

      candidates.push({
        phrase: firstMatch[0], // preserve original casing from text
        score: raw,
        startOffset: origStart,
        endOffset: origEnd,
        spanId: `${origStart}:${origEnd}`,
      });
    }

    if (candidates.length === 0) return [];

    // ── 4. Normalize scores to [0, 1], sort ascending ───────
    candidates.sort((a, b) => a.score - b.score);
    const topN = candidates.slice(0, this.opts.topN);

    if (topN.length === 0) return [];

    const minScore = topN[0].score;
    const maxScore = topN[topN.length - 1].score;
    const range = maxScore - minScore;

    return topN.map((p) => ({
      ...p,
      score: range > 0 ? (p.score - minScore) / range : 0,
    }));
  }
}
