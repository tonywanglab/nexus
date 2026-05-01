// shared EmbeddingGemma detection and pooling math (runtime iframe, Node scripts, tests).
// keep the iframe Gemma embedder logic numerically aligned with these functions.

//  Case-insensitive: model id contains "gemma" (matches embed-corpus / embed-vocab routing).
export function isEmbeddingGemmaModelId(modelId: string): boolean {
  return modelId.toLowerCase().includes("gemma");
}

// mean pool over sequence positions, then L2-normalize (last_hidden_state layout:
// row-major [seqLen, hiddenDim]).
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

function attentionMaskValue(mask: ArrayBufferView, index: number): number {
  if (mask instanceof BigInt64Array || mask instanceof BigUint64Array) {
    return Number(mask[index]);
  }
  return (mask as Float32Array | Int32Array | Uint8Array)[index] as number;
}

// mean-pool over non-padding positions (attention mask), then L2-normalize per row.
// `data` is row-major last_hidden_state [batch, seqLen, hiddenDim].
export function meanPoolNormalizeL2BatchedFromLastHidden(
  data: Float32Array,
  batch: number,
  seqLen: number,
  hiddenDim: number,
  attentionMask: ArrayBufferView,
): Float32Array[] {
  const results: Float32Array[] = [];
  const acc = new Float32Array(hiddenDim);
  for (let b = 0; b < batch; b++) {
    acc.fill(0);
    let count = 0;
    for (let s = 0; s < seqLen; s++) {
      if (attentionMaskValue(attentionMask, b * seqLen + s) === 0) continue;
      count++;
      const row = ((b * seqLen + s) * hiddenDim) >>> 0;
      for (let d = 0; d < hiddenDim; d++) {
        acc[d] += data[row + d];
      }
    }
    const out = new Float32Array(hiddenDim);
    if (count > 0) {
      for (let d = 0; d < hiddenDim; d++) out[d] = acc[d] / count;
    }
    let norm = 0;
    for (let d = 0; d < hiddenDim; d++) norm += out[d] * out[d];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let d = 0; d < hiddenDim; d++) out[d] /= norm;
    }
    results.push(out);
  }
  return results;
}
