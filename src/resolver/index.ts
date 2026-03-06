import { CandidateEdge, ExtractedPhrase } from "../types";
import { normalizedSimilarity } from "./lcs";
import { normalize, exactMatch, phraseContainsTitle } from "./normalization";

export interface ResolverOptions {
  similarityThreshold?: number;
  maxCandidatesPerPhrase?: number;
  /** Multiplicative boost for exact matches (default 1.5 → sim × 1.5, clamped to 1). */
  exactMatchBoost?: number;
  /** Multiplicative boost when phrase contains the title (default 1.1). */
  phraseContainsTitleBoost?: number;
}

const DEFAULTS: Required<ResolverOptions> = {
  similarityThreshold: 0.85,
  maxCandidatesPerPhrase: 3,
  exactMatchBoost: 1.5,
  phraseContainsTitleBoost: 1.1,
};

/**
 * Maps extracted keyphrases to existing note titles using
 * deterministic (LCS) and stochastic (embeddings) methods.
 */
export class AliasResolver {
  private opts: Required<ResolverOptions>;

  constructor(options?: ResolverOptions) {
    this.opts = { ...DEFAULTS, ...options };
  }

  resolve(
    phrases: ExtractedPhrase[],
    noteTitles: string[],
    sourcePath: string,
  ): CandidateEdge[] {
    const sourceBasename = sourcePath.replace(/\.md$/, "").split("/").pop() ?? "";
    const candidates: CandidateEdge[] = [];

    // Build token inverted index: token → Set<title>
    const normalizedSource = normalize(sourceBasename);
    const tokenToTitles = new Map<string, Set<string>>();
    const normalizedTitleCache = new Map<string, string>();

    for (const title of noteTitles) {
      const nt = normalize(title);
      normalizedTitleCache.set(title, nt);
      if (nt === normalizedSource) continue; // skip self
      for (const token of nt.split(" ")) {
        if (!token) continue;
        let set = tokenToTitles.get(token);
        if (!set) {
          set = new Set();
          tokenToTitles.set(token, set);
        }
        set.add(title);
      }
    }

    for (const phrase of phrases) {
      if (phrase.phrase.length < 2) continue;

      const normalizedPhrase = normalize(phrase.phrase);
      const phraseCandidates: CandidateEdge[] = [];

      // Collect candidate titles that share at least one token with the phrase
      const candidateTitles = new Set<string>();
      for (const token of normalizedPhrase.split(" ")) {
        const titles = tokenToTitles.get(token);
        if (titles) {
          for (const t of titles) candidateTitles.add(t);
        }
      }

      for (const title of candidateTitles) {
        const normalizedTitle = normalizedTitleCache.get(title) ?? normalize(title);
        const baseSimilarity = normalizedSimilarity(normalizedPhrase, normalizedTitle);

        // Apply multiplicative boosts (strongest match wins)
        let boost = 1;
        if (exactMatch(phrase.phrase, title)) {
          boost = this.opts.exactMatchBoost;
        } else if (phraseContainsTitle(phrase.phrase, title)) {
          boost = this.opts.phraseContainsTitleBoost;
        }

        // Multiplicative boost clamped to [0, 1]
        const similarity = Math.min(1, baseSimilarity * boost);

        if (similarity >= this.opts.similarityThreshold) {
          phraseCandidates.push({
            sourcePath,
            phrase,
            targetPath: title,
            similarity,
          });
        }
      }

      // Sort by similarity descending, take top N
      phraseCandidates.sort((a, b) => b.similarity - a.similarity);
      candidates.push(...phraseCandidates.slice(0, this.opts.maxCandidatesPerPhrase));
    }

    // Deduplicate: if multiple phrases resolve to same target, keep highest similarity
    const bestByTarget = new Map<string, CandidateEdge>();
    for (const edge of candidates) {
      const existing = bestByTarget.get(edge.targetPath);
      if (!existing || edge.similarity > existing.similarity) {
        bestByTarget.set(edge.targetPath, edge);
      }
    }

    // Rank by combined score: similarity × (1 - phraseScore)
    const deduped = Array.from(bestByTarget.values());
    deduped.sort((a, b) => {
      const scoreA = a.similarity * (1 - a.phrase.score);
      const scoreB = b.similarity * (1 - b.phrase.score);
      return scoreB - scoreA;
    });

    return deduped;
  }
}
