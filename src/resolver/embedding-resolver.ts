import { CandidateEdge, ExtractedPhrase } from "../types";
import { EmbeddingProvider, EmbeddingPriority } from "./embedding-provider";
import { cosineSimilarity } from "./cosine";
import { excludeSelfTitles } from "./normalization";
import { dedupAndRank } from "./shared-utils";
import { SparseAutoencoder } from "./sae";

export interface EmbeddingResolverOptions {
  embeddingProvider: EmbeddingProvider;
  similarityThreshold?: number;
  maxCandidatesPerPhrase?: number;
  /**
   * Optional k-sparse autoencoder. When provided, the resolver additionally
   * produces `titleSparseEmbeddings` and `phraseSparseEmbeddings` via
   * `sae.encode()`. Scoring still uses the dense embeddings - the sparse
   * activations are available for future downstream use.
   */
  sae?: SparseAutoencoder;
}

const DEFAULTS = {
  similarityThreshold: 0.7,
  maxCandidatesPerPhrase: 3,
};

/**
 * Resolves extracted keyphrases to note titles using dense embedding
 * cosine similarity. Same input/output shape as AliasResolver but async.
 *
 * When constructed with an SAE, per-call sparse encodings of titles and
 * phrases are computed alongside the dense scoring (see `resolveWithSparse`).
 */
export class EmbeddingResolver {
  private provider: EmbeddingProvider;
  private similarityThreshold: number;
  private maxCandidatesPerPhrase: number;
  private sae: SparseAutoencoder | undefined;

  /** Cached title embeddings, invalidated when title list changes. */
  private titleEmbeddingCache = new Map<string, Float32Array>();

  constructor(options: EmbeddingResolverOptions) {
    this.provider = options.embeddingProvider;
    this.similarityThreshold = options.similarityThreshold ?? DEFAULTS.similarityThreshold;
    this.maxCandidatesPerPhrase = options.maxCandidatesPerPhrase ?? DEFAULTS.maxCandidatesPerPhrase;
    this.sae = options.sae;
  }

  async resolve(
    phrases: ExtractedPhrase[],
    noteTitles: string[],
    sourcePath: string,
    priority: EmbeddingPriority = "normal",
  ): Promise<CandidateEdge[]> {
    const { candidates } = await this.resolveWithSparse(phrases, noteTitles, sourcePath, priority);
    return candidates;
  }

  /**
   * Same as `resolve()` but also returns the sparse title/phrase encodings
   * when an SAE is configured. Scoring is unchanged (dense cosine similarity).
   */
  async resolveWithSparse(
    phrases: ExtractedPhrase[],
    noteTitles: string[],
    sourcePath: string,
    priority: EmbeddingPriority = "normal",
  ): Promise<{
    candidates: CandidateEdge[];
    titleSparseEmbeddings?: Map<string, Float32Array>;
    phraseSparseEmbeddings?: Float32Array[];
  }> {
    const filteredTitles = excludeSelfTitles(noteTitles, sourcePath);
    if (filteredTitles.length === 0 || phrases.length === 0) {
      return { candidates: [] };
    }

    const titleEmbeddings = await this.embedTitles(filteredTitles, priority);

    const phraseTexts = phrases.map(p => p.phrase);
    const phraseEmbeddings = await this.provider.embedBatch(phraseTexts, { priority });

    let titleSparseEmbeddings: Map<string, Float32Array> | undefined;
    let phraseSparseEmbeddings: Float32Array[] | undefined;
    if (this.sae) {
      titleSparseEmbeddings = new Map();
      for (const title of filteredTitles) {
        const dense = titleEmbeddings.get(title);
        if (dense) titleSparseEmbeddings.set(title, this.sae.encode(dense));
      }
      phraseSparseEmbeddings = this.sae.encodeBatch(phraseEmbeddings);
    }

    const candidates = scorePhrasesAgainstTitles({
      phrases,
      phraseEmbeddings,
      filteredTitles,
      titleEmbeddings,
      sourcePath,
      similarityThreshold: this.similarityThreshold,
      maxCandidatesPerPhrase: this.maxCandidatesPerPhrase,
    });

    const result: {
      candidates: CandidateEdge[];
      titleSparseEmbeddings?: Map<string, Float32Array>;
      phraseSparseEmbeddings?: Float32Array[];
    } = { candidates: dedupAndRank(candidates) };
    if (titleSparseEmbeddings) result.titleSparseEmbeddings = titleSparseEmbeddings;
    if (phraseSparseEmbeddings) result.phraseSparseEmbeddings = phraseSparseEmbeddings;
    return result;
  }

  private async embedTitles(
    titles: string[],
    priority: EmbeddingPriority = "normal",
  ): Promise<Map<string, Float32Array>> {
    const titleSet = new Set(titles);

    const toEmbed = titles.filter(t => !this.titleEmbeddingCache.has(t));
    if (toEmbed.length > 0) {
      const embeddings = await this.provider.embedBatch(toEmbed, { priority });
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

/**
 * Score each phrase against all titles via dense cosine similarity and return
 * the top-N candidates per phrase. Pure function - does not hold state and
 * does not dedup cross-phrase (callers should pipe through `dedupAndRank`).
 */
export function scorePhrasesAgainstTitles(params: {
  phrases: ExtractedPhrase[];
  phraseEmbeddings: Float32Array[];
  filteredTitles: string[];
  titleEmbeddings: Map<string, Float32Array>;
  sourcePath: string;
  similarityThreshold: number;
  maxCandidatesPerPhrase: number;
}): CandidateEdge[] {
  const {
    phrases,
    phraseEmbeddings,
    filteredTitles,
    titleEmbeddings,
    sourcePath,
    similarityThreshold,
    maxCandidatesPerPhrase,
  } = params;

  const candidates: CandidateEdge[] = [];

  for (let pi = 0; pi < phrases.length; pi++) {
    const phrase = phrases[pi];
    const phraseVec = phraseEmbeddings[pi];
    const phraseCandidates: CandidateEdge[] = [];

    for (let ti = 0; ti < filteredTitles.length; ti++) {
      const title = filteredTitles[ti];
      const titleVec = titleEmbeddings.get(title);
      if (!titleVec) continue;
      const similarity = cosineSimilarity(phraseVec, titleVec);

      if (similarity >= similarityThreshold) {
        phraseCandidates.push({
          sourcePath,
          phrase,
          targetPath: title,
          similarity,
          matchType: "stochastic",
        });
      }
    }

    phraseCandidates.sort((a, b) => b.similarity - a.similarity);
    candidates.push(...phraseCandidates.slice(0, maxCandidatesPerPhrase));
  }

  return candidates;
}
