import { CandidateEdge, ExtractedPhrase } from "../types";
import { EmbeddingProvider, EmbeddingPriority } from "./embedding-provider";
import { cosineSimilarity } from "./cosine";
import { excludeSelfTitles } from "./normalization";
import { dedupAndRank, makeYielder } from "./shared-utils";
import { SparseAutoencoder, SparseEncoding } from "./sae";
import { SAEFeatureLabels } from "./sae-feature-labels";
import { sparseCosine } from "./sparse-feature-match";

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
  /**
   * Feature labels for the SAE. When provided together with `sae`, each dense
   * edge returned by `resolve()` is annotated with the top shared labeled
   * features that explain the dense match (see `denseExplanationTopN`).
   */
  featureLabels?: SAEFeatureLabels;
  /** Number of top shared labeled features to attach per dense edge. Default 2. */
  denseExplanationTopN?: number;
  /** FIFO cap on the vault-wide phrase embedding cache. */
  phraseCacheLimit?: number;
  /** Invoked when the title embedding cache gains new entries worth persisting. */
  onTitleEmbeddingsChanged?: (map: Map<string, Float32Array>) => void;
}

const DEFAULTS = {
  similarityThreshold: 0.82,
  maxCandidatesPerPhrase: 3,
  phraseCacheLimit: 5000,
  denseExplanationTopN: 2,
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
  private featureLabels: SAEFeatureLabels | undefined;
  private denseExplanationTopN: number;

  /** Cached title embeddings, invalidated when title list changes. */
  private titleEmbeddingCache = new Map<string, Float32Array>();
  /** Sparse SAE encodings per title — computed once alongside the dense embedding. */
  private titleSparseCache = new Map<string, SparseEncoding>();
  /** Vault-wide phrase embedding cache (FIFO cap). Key is raw phrase text. */
  private phraseEmbeddingCache = new Map<string, Float32Array>();
  /** Sparse SAE encodings per phrase — mirrors phraseEmbeddingCache eviction. */
  private phraseSparseCache = new Map<string, SparseEncoding>();
  private phraseCacheLimit: number;
  private onTitleEmbeddingsChanged: ((m: Map<string, Float32Array>) => void) | undefined;

  constructor(options: EmbeddingResolverOptions) {
    this.provider = options.embeddingProvider;
    this.similarityThreshold = options.similarityThreshold ?? DEFAULTS.similarityThreshold;
    this.maxCandidatesPerPhrase = options.maxCandidatesPerPhrase ?? DEFAULTS.maxCandidatesPerPhrase;
    this.sae = options.sae;
    this.featureLabels = options.featureLabels;
    this.denseExplanationTopN = options.denseExplanationTopN ?? DEFAULTS.denseExplanationTopN;
    this.phraseCacheLimit = options.phraseCacheLimit ?? DEFAULTS.phraseCacheLimit;
    this.onTitleEmbeddingsChanged = options.onTitleEmbeddingsChanged;
  }

  /** Swap in a new SAE + labels (e.g. after a version toggle). Clears the sparse cache. */
  updateSAE(sae: SparseAutoencoder, featureLabels: SAEFeatureLabels): void {
    this.sae = sae;
    this.featureLabels = featureLabels;
    this.titleSparseCache.clear();
  }

  /** Seed the title cache from persisted state (call once on plugin load). */
  seedTitleEmbeddings(map: Map<string, Float32Array>): void {
    for (const [k, v] of map) this.titleEmbeddingCache.set(k, v);
  }

  /** Snapshot of the current title cache for persistence. */
  exportTitleEmbeddings(): Map<string, Float32Array> {
    return new Map(this.titleEmbeddingCache);
  }

  /**
   * Given a batch of phrase texts, return their embeddings — using and updating
   * the vault-wide cache. Misses are batched into a single provider call.
   */
  private async embedPhrases(
    phraseTexts: string[],
    priority: EmbeddingPriority,
  ): Promise<Float32Array[]> {
    const missTexts: string[] = [];
    for (let i = 0; i < phraseTexts.length; i++) {
      if (!this.phraseEmbeddingCache.has(phraseTexts[i])) {
        missTexts.push(phraseTexts[i]);
      }
    }
    if (missTexts.length > 0) {
      const fresh = await this.provider.embedBatch(missTexts, { priority });
      for (let i = 0; i < missTexts.length; i++) {
        this.phraseEmbeddingCache.set(missTexts[i], fresh[i]);
      }
    }
    // Snapshot before eviction so the FIFO pass can't take entries needed by
    // this call out from under us.
    const out = phraseTexts.map((t) => this.phraseEmbeddingCache.get(t)!);
    while (this.phraseEmbeddingCache.size > this.phraseCacheLimit) {
      const firstKey = this.phraseEmbeddingCache.keys().next().value as string | undefined;
      if (!firstKey) break;
      this.phraseEmbeddingCache.delete(firstKey);
      this.phraseSparseCache.delete(firstKey);
    }
    return out;
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
  ): Promise<{ candidates: CandidateEdge[] }> {
    const filteredTitles = excludeSelfTitles(noteTitles, sourcePath);
    if (filteredTitles.length === 0 || phrases.length === 0) {
      return { candidates: [] };
    }

    const titleEmbeddings = await this.embedTitles(filteredTitles, priority);

    const phraseTexts = phrases.map(p => p.phrase);
    const phraseEmbeddings = await this.embedPhrases(phraseTexts, priority);

    // Compute and cache sparse encodings for titles and phrases when an SAE is
    // configured. Cache is shared with resolveBySparseFeatures so titles are
    // never encoded more than once per session. Time-budget yielder: warm
    // cache iterations are ~microseconds (no yield needed); cold encodeSparse
    // calls are ~15ms each, so we yield after the first one past the budget.
    let titleSparseEncodings: Map<string, SparseEncoding> | undefined;
    let phraseSparseEncodings: SparseEncoding[] | undefined;
    if (this.sae) {
      const yieldIfNeeded = makeYielder();
      titleSparseEncodings = new Map();
      for (const title of filteredTitles) {
        await yieldIfNeeded();
        const dense = titleEmbeddings.get(title);
        if (!dense) continue;
        let enc = this.titleSparseCache.get(title);
        if (!enc) {
          enc = this.sae.encodeSparse(dense);
          this.titleSparseCache.set(title, enc);
        }
        titleSparseEncodings.set(title, enc);
      }
      phraseSparseEncodings = [];
      for (let i = 0; i < phraseTexts.length; i++) {
        await yieldIfNeeded();
        const text = phraseTexts[i];
        let enc = this.phraseSparseCache.get(text);
        if (!enc) {
          enc = this.sae!.encodeSparse(phraseEmbeddings[i]);
          this.phraseSparseCache.set(text, enc);
        }
        phraseSparseEncodings.push(enc);
      }
    }

    const candidates = await scorePhrasesAgainstTitles({
      phrases,
      phraseEmbeddings,
      filteredTitles,
      titleEmbeddings,
      sourcePath,
      similarityThreshold: this.similarityThreshold,
      maxCandidatesPerPhrase: this.maxCandidatesPerPhrase,
    });

    // Annotate each dense edge with the top shared labeled SAE features that
    // explain the dense match — interpretability for free.
    if (this.featureLabels && phraseSparseEncodings && titleSparseEncodings) {
      const phraseIndex = new Map<ExtractedPhrase, number>();
      for (let i = 0; i < phrases.length; i++) phraseIndex.set(phrases[i], i);
      for (const cand of candidates) {
        const pi = phraseIndex.get(cand.phrase);
        if (pi === undefined) continue;
        const pEnc = phraseSparseEncodings[pi];
        const tEnc = titleSparseEncodings.get(cand.targetPath);
        if (!tEnc) continue;
        const explanation = buildDenseExplanation(
          pEnc, tEnc, this.featureLabels, this.denseExplanationTopN,
        );
        if (explanation) cand.sparseFeatures = explanation;
      }
    }

    return { candidates: dedupAndRank(candidates) };
  }

  /**
   * Third resolver path: match phrases to titles via sparse SAE feature cosine similarity.
   * Scoring uses all labeled active features; the edge payload carries the top-4 for display.
   * Requires an SAE configured at construction time (throws otherwise).
   */
  async resolveBySparseFeatures(
    phrases: ExtractedPhrase[],
    noteTitles: string[],
    sourcePath: string,
    featureLabels: SAEFeatureLabels,
    opts?: { similarityThreshold?: number; maxCandidatesPerPhrase?: number; priority?: EmbeddingPriority },
  ): Promise<CandidateEdge[]> {
    if (!this.sae) throw new Error("resolveBySparseFeatures requires an SAE");
    const similarityThreshold = opts?.similarityThreshold ?? this.similarityThreshold;
    const maxCandidatesPerPhrase = opts?.maxCandidatesPerPhrase ?? this.maxCandidatesPerPhrase;
    const priority = opts?.priority ?? "normal";

    const filteredTitles = excludeSelfTitles(noteTitles, sourcePath);
    if (filteredTitles.length === 0 || phrases.length === 0) return [];

    const titleEmbeddings = await this.embedTitles(filteredTitles, priority);
    const phraseTexts = phrases.map(p => p.phrase);
    const phraseEmbeddings = await this.embedPhrases(phraseTexts, priority);

    // Pre-compute sparse features for all titles, using the session cache.
    // Time-budget yielder: skips yields on warm cache, yields promptly when
    // encodeSparse actually runs. Shared across both phase loops so we don't
    // over-yield at the boundary between them.
    const yieldIfNeeded = makeYielder();
    const titleMatchFeatures = new Map<string, ReturnType<SAEFeatureLabels["pickAllLabeled"]>>();
    const titleDisplayFeatures = new Map<string, ReturnType<SAEFeatureLabels["pickTop4Labeled"]>>();
    for (const title of filteredTitles) {
      await yieldIfNeeded();
      const dense = titleEmbeddings.get(title);
      if (!dense) continue;
      let enc = this.titleSparseCache.get(title);
      if (!enc) {
        enc = this.sae.encodeSparse(dense);
        this.titleSparseCache.set(title, enc);
      }
      titleMatchFeatures.set(title, featureLabels.pickAllLabeled(enc));
      titleDisplayFeatures.set(title, featureLabels.pickTop4Labeled(enc));
    }

    const candidates: CandidateEdge[] = [];

    for (let pi = 0; pi < phrases.length; pi++) {
      await yieldIfNeeded();
      const phrase = phrases[pi];
      const phraseText = phraseTexts[pi];
      let phraseEnc = this.phraseSparseCache.get(phraseText);
      if (!phraseEnc) {
        phraseEnc = this.sae.encodeSparse(phraseEmbeddings[pi]);
        this.phraseSparseCache.set(phraseText, phraseEnc);
      }
      const phraseMatch = featureLabels.pickAllLabeled(phraseEnc);
      const phraseDisplay = featureLabels.pickTop4Labeled(phraseEnc);

      if (phraseMatch.indices.length === 0) continue;

      const phraseCandidates: CandidateEdge[] = [];

      for (const title of filteredTitles) {
        const titleMatch = titleMatchFeatures.get(title);
        const titleDisplay = titleDisplayFeatures.get(title);
        if (!titleMatch || titleMatch.indices.length === 0 || !titleDisplay) continue;

        const similarity = sparseCosine(phraseMatch, titleMatch);
        if (similarity < similarityThreshold) continue;

        phraseCandidates.push({
          sourcePath,
          phrase,
          targetPath: title,
          similarity,
          matchType: "sparse-feature",
          sparseFeatures: {
            phraseFeatures: phraseDisplay.indices.map((idx, i) => ({
              idx,
              value: phraseDisplay.values[i],
              label: phraseDisplay.labels[i],
            })),
            titleFeatures: titleDisplay.indices.map((idx, i) => ({
              idx,
              value: titleDisplay.values[i],
              label: titleDisplay.labels[i],
            })),
          },
        });
      }

      phraseCandidates.sort((a, b) => b.similarity - a.similarity);
      candidates.push(...phraseCandidates.slice(0, maxCandidatesPerPhrase));
    }

    return dedupAndRank(candidates);
  }

  private async embedTitles(
    titles: string[],
    priority: EmbeddingPriority = "normal",
  ): Promise<Map<string, Float32Array>> {
    const toEmbed = titles.filter(t => !this.titleEmbeddingCache.has(t));
    if (toEmbed.length > 0) {
      const embeddings = await this.provider.embedBatch(toEmbed, { priority });
      for (let i = 0; i < toEmbed.length; i++) {
        this.titleEmbeddingCache.set(toEmbed[i], embeddings[i]);
      }
      // Signal persistence — only when we actually added entries. Renames/
      // deletions are handled by a separate pruning pass on plugin startup.
      this.onTitleEmbeddingsChanged?.(this.titleEmbeddingCache);
    }
    // Return a view restricted to the requested titles so callers only see the
    // vectors relevant to this resolve call — the internal cache is a superset.
    const out = new Map<string, Float32Array>();
    for (const t of titles) {
      const v = this.titleEmbeddingCache.get(t);
      if (v) out.set(t, v);
    }
    return out;
  }

  /** Remove cached embeddings (dense + sparse) for titles that no longer exist in the vault. */
  pruneTitleCacheTo(activeTitles: Set<string>): number {
    let removed = 0;
    for (const k of this.titleEmbeddingCache.keys()) {
      if (!activeTitles.has(k)) {
        this.titleEmbeddingCache.delete(k);
        this.titleSparseCache.delete(k);
        removed++;
      }
    }
    if (removed > 0) this.onTitleEmbeddingsChanged?.(this.titleEmbeddingCache);
    return removed;
  }
}

/**
 * Build the top-N shared labeled-feature explanation for a dense match.
 * Returns the intersection of labeled active features between phrase and title,
 * ranked so features that fire strongly on both sides come first. Returns
 * undefined when no labeled features are shared (no explanation available).
 */
export function buildDenseExplanation(
  phraseEnc: SparseEncoding,
  titleEnc: SparseEncoding,
  featureLabels: SAEFeatureLabels,
  topN: number,
): NonNullable<CandidateEdge["sparseFeatures"]> | undefined {
  const phraseLabeled = featureLabels.pickAllLabeled(phraseEnc);
  const titleLabeled = featureLabels.pickAllLabeled(titleEnc);
  if (phraseLabeled.indices.length === 0 || titleLabeled.indices.length === 0) {
    return undefined;
  }

  const phraseMap = new Map<number, { value: number; label: string }>();
  for (let i = 0; i < phraseLabeled.indices.length; i++) {
    phraseMap.set(phraseLabeled.indices[i], {
      value: phraseLabeled.values[i],
      label: phraseLabeled.labels[i],
    });
  }

  const shared: { idx: number; pVal: number; tVal: number; label: string }[] = [];
  for (let i = 0; i < titleLabeled.indices.length; i++) {
    const idx = titleLabeled.indices[i];
    const p = phraseMap.get(idx);
    if (!p) continue;
    shared.push({ idx, pVal: p.value, tVal: titleLabeled.values[i], label: p.label });
  }
  if (shared.length === 0) return undefined;

  // Rank by pVal * tVal — each feature's literal contribution to
  // sparseCosine(p, t) = Σ(pVal · tVal) / (|p|·|t|), so the top entries are
  // the features that most explain the dense match.
  shared.sort((a, b) => b.pVal * b.tVal - a.pVal * a.tVal);
  const top = shared.slice(0, Math.max(1, topN));

  return {
    phraseFeatures: top.map((s) => ({ idx: s.idx, value: s.pVal, label: s.label })),
    titleFeatures: top.map((s) => ({ idx: s.idx, value: s.tVal, label: s.label })),
  };
}

/**
 * Score each phrase against all titles via dense cosine similarity and return
 * the top-N candidates per phrase. Pure function - does not hold state and
 * does not dedup cross-phrase (callers should pipe through `dedupAndRank`).
 */
export async function scorePhrasesAgainstTitles(params: {
  phrases: ExtractedPhrase[];
  phraseEmbeddings: Float32Array[];
  filteredTitles: string[];
  titleEmbeddings: Map<string, Float32Array>;
  sourcePath: string;
  similarityThreshold: number;
  maxCandidatesPerPhrase: number;
}): Promise<CandidateEdge[]> {
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
  const yieldIfNeeded = makeYielder();

  for (let pi = 0; pi < phrases.length; pi++) {
    const phrase = phrases[pi];
    const phraseVec = phraseEmbeddings[pi];
    const phraseCandidates: CandidateEdge[] = [];

    for (let ti = 0; ti < filteredTitles.length; ti++) {
      if ((ti & 31) === 0) await yieldIfNeeded();
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
