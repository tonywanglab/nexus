import { CandidateEdge, ExtractedPhrase } from "../types";
import { normalizedSimilarity } from "./lcs";
import { normalize, exactMatch, phraseContainsTitle, excludeSelfTitles } from "./normalization";
import { dedupAndRank } from "./shared-utils";

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
 * Maps extracted keyphrases to existing note titles using deterministic LCS string similarity.
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
    const filteredTitles = excludeSelfTitles(noteTitles, sourcePath);
    const candidates: CandidateEdge[] = [];

    // Build token inverted index: token → Set<title>
    const tokenToTitles = new Map<string, Set<string>>();
    const normalizedTitleCache = new Map<string, string>();

    for (const title of filteredTitles) {
      const nt = normalize(title);
      normalizedTitleCache.set(title, nt);
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
            matchType: "deterministic",
          });
        }
      }

      // Sort by similarity descending, take top N
      phraseCandidates.sort((a, b) => b.similarity - a.similarity);
      candidates.push(...phraseCandidates.slice(0, this.opts.maxCandidatesPerPhrase));
    }

    return dedupAndRank(candidates);
  }
}
