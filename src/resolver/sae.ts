// k-sparse (dictionary-learning) autoencoder with explicit TopK sparsity control.
//
// architecture (matches the reference article, modern SAE formulation):
//
//   z = TopK(ReLU(W_enc · (x - b_pre) + b_enc), k)   // exactly k non-zeros
//   x̂ = W_dec · z + b_dec + b_pre                    // dictionary reconstruction
//
// no L1 / KL soft regularization - sparsity is enforced structurally. The decoder
// matrix is a learned dictionary: each of its d_hidden columns is a unit-norm
// "atom", and reconstructions are k-sparse linear combinations of atoms.
//
// this file contains the forward-only runtime (pure Float32Array math, no deps).
// training lives in sae-trainer.ts and depends on @tensorflow/tfjs-node.

export interface SAEWeights {
  dModel: number;
  dHidden: number;
  k: number;
  //  Encoder weights, shape [dHidden, dModel] row-major.
  wEnc: Float32Array;
  //  Encoder bias, length dHidden.
  bEnc: Float32Array;
  //  Pre-bias (centering vector), length dModel.
  bPre: Float32Array;
  //  Decoder weights, shape [dModel, dHidden] row-major. Columns are dictionary atoms.
  wDec: Float32Array;
  //  Decoder bias, length dModel.
  bDec: Float32Array;
}

export interface SparseEncoding {
  //  Length k, the indices of the active features (unsorted).
  indices: Int32Array;
  //  Length k, the values at those indices (in matching order).
  values: Float32Array;
}

const MAGIC = "SAE1";

export class SparseAutoencoder {
  readonly dModel: number;
  readonly dHidden: number;
  readonly k: number;
  readonly wEnc: Float32Array;
  readonly bEnc: Float32Array;
  readonly bPre: Float32Array;
  readonly wDec: Float32Array;
  readonly bDec: Float32Array;

  constructor(weights: SAEWeights) {
    const { dModel, dHidden, k, wEnc, bEnc, bPre, wDec, bDec } = weights;
    if (!Number.isInteger(dModel) || dModel <= 0) {
      throw new Error(`Invalid dModel: ${dModel}`);
    }
    if (!Number.isInteger(dHidden) || dHidden <= 0) {
      throw new Error(`Invalid dHidden: ${dHidden}`);
    }
    if (!Number.isInteger(k) || k <= 0 || k > dHidden) {
      throw new Error(`Invalid k: ${k} (must be in [1, ${dHidden}])`);
    }
    if (wEnc.length !== dHidden * dModel) {
      throw new Error(`wEnc length ${wEnc.length} != dHidden*dModel ${dHidden * dModel}`);
    }
    if (bEnc.length !== dHidden) {
      throw new Error(`bEnc length ${bEnc.length} != dHidden ${dHidden}`);
    }
    if (bPre.length !== dModel) {
      throw new Error(`bPre length ${bPre.length} != dModel ${dModel}`);
    }
    if (wDec.length !== dModel * dHidden) {
      throw new Error(`wDec length ${wDec.length} != dModel*dHidden ${dModel * dHidden}`);
    }
    if (bDec.length !== dModel) {
      throw new Error(`bDec length ${bDec.length} != dModel ${dModel}`);
    }

    this.dModel = dModel;
    this.dHidden = dHidden;
    this.k = k;
    this.wEnc = wEnc;
    this.bEnc = bEnc;
    this.bPre = bPre;
    this.wDec = wDec;
    this.bDec = bDec;
  }

  encode(x: Float32Array): Float32Array {
    if (x.length !== this.dModel) {
      throw new Error(`Input length ${x.length} != dModel ${this.dModel}`);
    }
    const { dModel, dHidden, k, wEnc, bEnc, bPre } = this;

    const centered = new Float32Array(dModel);
    for (let j = 0; j < dModel; j++) {
      centered[j] = x[j] - bPre[j];
    }

    const pre = new Float32Array(dHidden);
    for (let i = 0; i < dHidden; i++) {
      let acc = bEnc[i];
      const rowStart = i * dModel;
      for (let j = 0; j < dModel; j++) {
        acc += wEnc[rowStart + j] * centered[j];
      }
      pre[i] = acc > 0 ? acc : 0;
    }

    const { indices, values } = topK(pre, k);
    const sparse = new Float32Array(dHidden);
    for (let t = 0; t < k; t++) {
      sparse[indices[t]] = values[t];
    }
    return sparse;
  }

  // compact k-sparse encoding: returns the k active indices and their values.
  // `encode(x)` is equivalent to scattering these into a zero Float32Array.
  encodeSparse(x: Float32Array): SparseEncoding {
    if (x.length !== this.dModel) {
      throw new Error(`Input length ${x.length} != dModel ${this.dModel}`);
    }
    const { dModel, dHidden, k, wEnc, bEnc, bPre } = this;

    const centered = new Float32Array(dModel);
    for (let j = 0; j < dModel; j++) {
      centered[j] = x[j] - bPre[j];
    }

    const pre = new Float32Array(dHidden);
    for (let i = 0; i < dHidden; i++) {
      let acc = bEnc[i];
      const rowStart = i * dModel;
      for (let j = 0; j < dModel; j++) {
        acc += wEnc[rowStart + j] * centered[j];
      }
      pre[i] = acc > 0 ? acc : 0;
    }

    return topK(pre, k);
  }

  encodeBatch(xs: Float32Array[]): Float32Array[] {
    const out: Float32Array[] = new Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      out[i] = this.encode(xs[i]);
    }
    return out;
  }

  // dictionary reconstruction: x̂ = W_dec · z + b_dec + b_pre.
  // equivalent to summing the active atoms weighted by their coefficients.
  decode(z: Float32Array): Float32Array {
    if (z.length !== this.dHidden) {
      throw new Error(`z length ${z.length} != dHidden ${this.dHidden}`);
    }
    const { dModel, dHidden, wDec, bDec, bPre } = this;
    const out = new Float32Array(dModel);
    for (let i = 0; i < dModel; i++) {
      let acc = bDec[i] + bPre[i];
      const rowStart = i * dHidden;
      for (let j = 0; j < dHidden; j++) {
        acc += wDec[rowStart + j] * z[j];
      }
      out[i] = acc;
    }
    return out;
  }

  forward(x: Float32Array): { sparse: Float32Array; reconstruction: Float32Array } {
    const sparse = this.encode(x);
    const reconstruction = this.decode(sparse);
    return { sparse, reconstruction };
  }

  // project each dictionary atom (decoder column) to unit L2 norm in place.
  // prevents the encoder-decoder scale-cheating loophole and keeps feature
  // magnitudes interpretable.
  normalizeDecoder(): void {
    const { dModel, dHidden, wDec } = this;
    for (let j = 0; j < dHidden; j++) {
      let sumSq = 0;
      for (let i = 0; i < dModel; i++) {
        const v = wDec[i * dHidden + j];
        sumSq += v * v;
      }
      const norm = Math.sqrt(sumSq);
      if (norm < 1e-8) continue;
      const inv = 1 / norm;
      for (let i = 0; i < dModel; i++) {
        wDec[i * dHidden + j] *= inv;
      }
    }
  }

  // serialize to a compact binary blob:
  //   4 bytes: magic "SAE1"
  //   4 bytes: header JSON length (u32 LE)
  //   N bytes: header JSON `{dModel, dHidden, k}`
  //   then packed Float32 LE: wEnc, bEnc, bPre, wDec, bDec
  serialize(): Uint8Array {
    return SparseAutoencoder.serialize({
      dModel: this.dModel,
      dHidden: this.dHidden,
      k: this.k,
      wEnc: this.wEnc,
      bEnc: this.bEnc,
      bPre: this.bPre,
      wDec: this.wDec,
      bDec: this.bDec,
    });
  }

  static serialize(w: SAEWeights): Uint8Array {
    const header = JSON.stringify({ dModel: w.dModel, dHidden: w.dHidden, k: w.k });
    const rawHeader = new TextEncoder().encode(header);
    // pad the header so the Float32 section starts on a 4-byte boundary
    // (8 bytes magic+length + headerLen must be divisible by 4).
    const padding = (4 - ((8 + rawHeader.length) % 4)) % 4;
    const headerBytes = new Uint8Array(rawHeader.length + padding);
    headerBytes.set(rawHeader);
    // pad with spaces so the JSON remains valid if re-parsed as a string.
    for (let i = rawHeader.length; i < headerBytes.length; i++) headerBytes[i] = 0x20;

    const totalFloats =
      w.wEnc.length + w.bEnc.length + w.bPre.length + w.wDec.length + w.bDec.length;
    const totalBytes = 4 + 4 + headerBytes.length + totalFloats * 4;

    const buf = new ArrayBuffer(totalBytes);
    const u8 = new Uint8Array(buf);
    const dv = new DataView(buf);

    u8[0] = MAGIC.charCodeAt(0);
    u8[1] = MAGIC.charCodeAt(1);
    u8[2] = MAGIC.charCodeAt(2);
    u8[3] = MAGIC.charCodeAt(3);
    dv.setUint32(4, headerBytes.length, true);
    u8.set(headerBytes, 8);

    let offset = 8 + headerBytes.length;
    const writeFloats = (arr: Float32Array): void => {
      const view = new Float32Array(buf, offset, arr.length);
      view.set(arr);
      offset += arr.length * 4;
    };
    writeFloats(w.wEnc);
    writeFloats(w.bEnc);
    writeFloats(w.bPre);
    writeFloats(w.wDec);
    writeFloats(w.bDec);

    return u8;
  }

  static deserialize(buf: ArrayBuffer | Uint8Array): SparseAutoencoder {
    const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    const ab = u8.buffer;
    const byteOffset = u8.byteOffset;

    const dv = new DataView(ab, byteOffset, u8.byteLength);
    const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
    if (magic !== MAGIC) {
      throw new Error(`Invalid SAE weights magic: expected ${MAGIC}, got ${magic}`);
    }
    const headerLen = dv.getUint32(4, true);
    const header = new TextDecoder().decode(u8.subarray(8, 8 + headerLen));
    const { dModel, dHidden, k } = JSON.parse(header) as {
      dModel: number;
      dHidden: number;
      k: number;
    };

    let offset = byteOffset + 8 + headerLen;
    // float32Array requires 4-byte alignment; copy if offset misaligns.
    const readFloats = (n: number): Float32Array => {
      let out: Float32Array;
      if (offset % 4 === 0) {
        out = new Float32Array(ab, offset, n).slice();
      } else {
        const tmp = new ArrayBuffer(n * 4);
        new Uint8Array(tmp).set(new Uint8Array(ab, offset, n * 4));
        out = new Float32Array(tmp);
      }
      offset += n * 4;
      return out;
    };

    const wEnc = readFloats(dHidden * dModel);
    const bEnc = readFloats(dHidden);
    const bPre = readFloats(dModel);
    const wDec = readFloats(dModel * dHidden);
    const bDec = readFloats(dModel);

    return new SparseAutoencoder({ dModel, dHidden, k, wEnc, bEnc, bPre, wDec, bDec });
  }

  // random Gaussian init, then project decoder columns to unit norm.
  // used by the training script; runtime always deserializes trained weights.
  static randomInit(
    dModel: number,
    dHidden: number,
    k: number,
    seed: number = 0xC0FFEE,
  ): SparseAutoencoder {
    const rng = mulberry32(seed);
    const gauss = (): number => {
      // box-Muller, discard the second value for simplicity.
      let u1 = rng();
      if (u1 < 1e-12) u1 = 1e-12;
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };

    const encScale = 1 / Math.sqrt(dModel);
    const wEnc = new Float32Array(dHidden * dModel);
    for (let i = 0; i < wEnc.length; i++) wEnc[i] = gauss() * encScale;

    const decScale = 1 / Math.sqrt(dHidden);
    const wDec = new Float32Array(dModel * dHidden);
    for (let i = 0; i < wDec.length; i++) wDec[i] = gauss() * decScale;

    const sae = new SparseAutoencoder({
      dModel,
      dHidden,
      k,
      wEnc,
      bEnc: new Float32Array(dHidden),
      bPre: new Float32Array(dModel),
      wDec,
      bDec: new Float32Array(dModel),
    });
    sae.normalizeDecoder();
    return sae;
  }
}

// min-heap based top-k selection. Returns indices+values (unsorted).
// runs in O(n log k) with stable behavior for ties (first-seen wins).
export function topK(values: Float32Array, k: number): SparseEncoding {
  const n = values.length;
  if (k > n) throw new Error(`topK: k=${k} > n=${n}`);

  const heapVals = new Float32Array(k);
  const heapIdx = new Int32Array(k);

  for (let i = 0; i < k; i++) {
    heapVals[i] = values[i];
    heapIdx[i] = i;
  }
  for (let i = (k >> 1) - 1; i >= 0; i--) {
    siftDown(heapVals, heapIdx, i, k);
  }

  for (let i = k; i < n; i++) {
    if (values[i] > heapVals[0]) {
      heapVals[0] = values[i];
      heapIdx[0] = i;
      siftDown(heapVals, heapIdx, 0, k);
    }
  }

  return { indices: heapIdx, values: heapVals };
}

// returns the top-n entries of an existing SparseEncoding by activation value.
// preserves Int32Array/Float32Array types. Throws if n > enc.indices.length.
export function topNOf(enc: SparseEncoding, n: number): SparseEncoding {
  const k = enc.indices.length;
  if (n > k) throw new Error(`topNOf: n=${n} > k=${k}`);
  const order = Array.from({ length: k }, (_, i) => i)
    .sort((a, b) => enc.values[b] - enc.values[a]);
  const outIdx = new Int32Array(n);
  const outVal = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    outIdx[i] = enc.indices[order[i]];
    outVal[i] = enc.values[order[i]];
  }
  return { indices: outIdx, values: outVal };
}

function siftDown(vals: Float32Array, idx: Int32Array, start: number, n: number): void {
  let i = start;
  while (true) {
    const l = 2 * i + 1;
    const r = 2 * i + 2;
    let smallest = i;
    if (l < n && vals[l] < vals[smallest]) smallest = l;
    if (r < n && vals[r] < vals[smallest]) smallest = r;
    if (smallest === i) break;
    const tv = vals[i];
    vals[i] = vals[smallest];
    vals[smallest] = tv;
    const ti = idx[i];
    idx[i] = idx[smallest];
    idx[smallest] = ti;
    i = smallest;
  }
}

//  Deterministic 32-bit PRNG for reproducible random init.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function (): number {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
