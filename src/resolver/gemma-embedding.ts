/**
 * Shared EmbeddingGemma detection and pooling math (runtime iframe, Node scripts, tests).
 * Keep the iframe Gemma embedder logic numerically aligned with these functions.
 */

/** Case-insensitive: model id contains "gemma" (matches embed-corpus / embed-vocab routing). */
export function isEmbeddingGemmaModelId(modelId: string): boolean {
  return modelId.toLowerCase().includes("gemma");
}

/**
 * Mean pool over sequence positions, then L2-normalize (last_hidden_state layout:
 * row-major [seqLen, hiddenDim]).
 */
export function meanPoolNormalizeL2FromLastHidden(
  data: Float32Array,
  seqLen: number,
  hiddenDim: number,
): Float32Array {
  const pooled = new Float32Array(hiddenDim);
  for (let s = 0; s < seqLen; s++) {
    const row = s * hiddenDim;
    for (let d = 0; d < hiddenDim; d++) {
      pooled[d] += data[row + d];
    }
  }
  for (let d = 0; d < hiddenDim; d++) pooled[d] /= seqLen;
  let norm = 0;
  for (let d = 0; d < hiddenDim; d++) norm += pooled[d] * pooled[d];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let d = 0; d < hiddenDim; d++) pooled[d] /= norm;
  }
  return pooled;
}
