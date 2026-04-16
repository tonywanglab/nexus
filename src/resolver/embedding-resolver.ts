import { CandidateEdge, ExtractedPhrase } from "../types";
import { EmbeddingProvider } from "./embedding-provider";
import { cosineSimilarity } from "./cosine";
import { excludeSelfTitles } from "./normalization";
import { dedupAndRank } from "./shared-utils";

export interface EmbeddingResolverOptions {
  embeddingProvider: EmbeddingProvider;
  similarityThreshold?: number;
  maxCandidatesPerPhrase?: number;
}

const DEFAULTS = {
  similarityThreshold: 0.7,
  maxCandidatesPerPhrase: 3,
};

/**
 * Resolves extracted keyphrases to note titles using dense embedding
 * cosine similarity. Same input/output shape as AliasResolver but async.
 */
export class EmbeddingResolver {
  private provider: EmbeddingProvider;
  private similarityThreshold: number;
  private maxCandidatesPerPhrase: number;

  /** Cached title embeddings, invalidated when title list changes. */
  private titleEmbeddingCache = new Map<string, Float32Array>();

  constructor(options: EmbeddingResolverOptions) {
    this.provider = options.embeddingProvider;
    this.similarityThreshold = options.similarityThreshold ?? DEFAULTS.similarityThreshold;
    this.maxCandidatesPerPhrase = options.maxCandidatesPerPhrase ?? DEFAULTS.maxCandidatesPerPhrase;
  }

  async resolve(
    phrases: ExtractedPhrase[],
    noteTitles: string[],
    sourcePath: string,
  ): Promise<CandidateEdge[]> {
    const filteredTitles = excludeSelfTitles(noteTitles, sourcePath);
    if (filteredTitles.length === 0 || phrases.length === 0) return [];

    // Embed titles (with caching)
    const titleEmbeddings = await this.embedTitles(filteredTitles);

    // Embed phrases
    const phraseTexts = phrases.map(p => p.phrase);
    const phraseEmbeddings = await this.provider.embedBatch(phraseTexts);

    // Score each phrase against all titles
    const candidates: CandidateEdge[] = [];

    for (let pi = 0; pi < phrases.length; pi++) {
      const phrase = phrases[pi];
      const phraseVec = phraseEmbeddings[pi];
      const phraseCandidates: CandidateEdge[] = [];

      for (let ti = 0; ti < filteredTitles.length; ti++) {
        const title = filteredTitles[ti];
        const titleVec = titleEmbeddings.get(title)!;
        const similarity = cosineSimilarity(phraseVec, titleVec);

        if (similarity >= this.similarityThreshold) {
          phraseCandidates.push({
            sourcePath,
            phrase,
            targetPath: title,
            similarity,
            matchType: "stochastic",
          });
        }
      }

      // Sort by similarity descending, take top N
      phraseCandidates.sort((a, b) => b.similarity - a.similarity);
      candidates.push(...phraseCandidates.slice(0, this.maxCandidatesPerPhrase));
    }

    return dedupAndRank(candidates);
  }

  private async embedTitles(titles: string[]): Promise<Map<string, Float32Array>> {
    const titleSet = new Set(titles);

    const toEmbed = titles.filter(t => !this.titleEmbeddingCache.has(t));
    if (toEmbed.length > 0) {
      const embeddings = await this.provider.embedBatch(toEmbed);
      for (let i = 0; i < toEmbed.length; i++) {
        this.titleEmbeddingCache.set(toEmbed[i], embeddings[i]);
      }
    }

    for (const key of this.titleEmbeddingCache.keys()) {
      if (!titleSet.has(key)) {
        this.titleEmbeddingCache.delete(key);
      }
    }

    return this.titleEmbeddingCache;
  }
}
