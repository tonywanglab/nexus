/**
 * K-sparse SAE trainer built on @tensorflow/tfjs-node.
 *
 * Training only. The runtime plugin bundle does NOT import this file - keep it
 * out of any module graph reachable from src/main.ts. Adam + autograd come from
 * TF.js; TopK is natively differentiable (gradient flows only through the
 * selected positions, giving the straight-through semantics the article
 * describes).
 *
 * Matrix shape convention (matches runtime SparseAutoencoder row-major layout):
 *   wEnc: [dHidden, dModel]   (use matMul(x, wEnc, false, true) for forward)
 *   wDec: [dModel,  dHidden]  (use matMul(z, wDec, false, true) for decode)
 *   b_pre / b_dec: [dModel]
 *   b_enc:         [dHidden]
 */

import "./tf-shim";
import * as tf from "@tensorflow/tfjs-node";
import { SparseAutoencoder } from "./sae";

export interface SAETrainerOptions {
  learningRate?: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}

export interface TrainStepResult {
  /** Mean-squared reconstruction error for the batch. */
  reconLoss: number;
  /** Average L0 (non-zero count per sample). Should be exactly k after training. */
  l0: number;
}

const DEFAULTS = {
  learningRate: 3e-4,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
};

export class SAETrainer {
  readonly dModel: number;
  readonly dHidden: number;
  readonly k: number;

  private wEnc: tf.Variable<tf.Rank.R2>;
  private bEnc: tf.Variable<tf.Rank.R1>;
  private bPre: tf.Variable<tf.Rank.R1>;
  private wDec: tf.Variable<tf.Rank.R2>;
  private bDec: tf.Variable<tf.Rank.R1>;

  private optimizer: tf.AdamOptimizer;

  /** Per-feature lifetime fire count. Used to detect dead features. */
  private featureFireCount: Int32Array;
  /** Samples processed since fire-count reset. */
  private samplesSinceReset: number;

  constructor(initial: SparseAutoencoder, opts: SAETrainerOptions = {}) {
    this.dModel = initial.dModel;
    this.dHidden = initial.dHidden;
    this.k = initial.k;

    this.wEnc = tf.variable(
      tf.tensor2d(initial.wEnc, [this.dHidden, this.dModel], "float32"),
      true,
      "wEnc",
    );
    this.bEnc = tf.variable(tf.tensor1d(initial.bEnc, "float32"), true, "bEnc");
    this.bPre = tf.variable(tf.tensor1d(initial.bPre, "float32"), true, "bPre");
    this.wDec = tf.variable(
      tf.tensor2d(initial.wDec, [this.dModel, this.dHidden], "float32"),
      true,
      "wDec",
    );
    this.bDec = tf.variable(tf.tensor1d(initial.bDec, "float32"), true, "bDec");

    this.optimizer = tf.train.adam(
      opts.learningRate ?? DEFAULTS.learningRate,
      opts.beta1 ?? DEFAULTS.beta1,
      opts.beta2 ?? DEFAULTS.beta2,
      opts.epsilon ?? DEFAULTS.epsilon,
    );

    this.featureFireCount = new Int32Array(this.dHidden);
    this.samplesSinceReset = 0;
  }

  /** Set the pre-bias to a precomputed sample mean (standard SAE warm-start). */
  setPreBias(mean: Float32Array): void {
    if (mean.length !== this.dModel) {
      throw new Error(`setPreBias: length ${mean.length} != dModel ${this.dModel}`);
    }
    tf.tidy(() => {
      this.bPre.assign(tf.tensor1d(mean, "float32"));
    });
  }

  /**
   * Compute the top-k binary mask for a features tensor using CPU-side topk,
   * returning a fresh tensor detached from the gradient tape. Used to
   * implement straight-through TopK: gradient flows through `features` only
   * at the k selected positions, zero elsewhere, because `mask` is a leaf
   * tensor (no backward ancestry) and `features * mask`'s gradient w.r.t.
   * features equals `mask * dy`.
   *
   * We do the topk on CPU rather than via tf.topk because tf.topk has no
   * registered gradient function in TF.js and even customGrad-wrapping the
   * oneHot+sum path hits a kernel type error on tfjs-node.
   */
  private computeTopKMask(features: tf.Tensor2D): tf.Tensor2D {
    const [B, D] = features.shape;
    const flat = features.dataSync() as Float32Array;
    const mask = new Float32Array(B * D);
    const valBuf = new Float32Array(this.k);
    const idxBuf = new Int32Array(this.k);
    for (let b = 0; b < B; b++) {
      const rowOff = b * D;
      partialTopK(flat, rowOff, D, this.k, valBuf, idxBuf);
      for (let t = 0; t < this.k; t++) {
        mask[rowOff + idxBuf[t]] = 1;
      }
    }
    return tf.tensor2d(mask, [B, D]);
  }

  /**
   * K-sparse forward: z = TopK(ReLU(W_enc (x - b_pre) + b_enc), k),
   * x̂ = W_dec z + b_dec + b_pre.
   *
   * Straight-through TopK: `mask` is a detached leaf tensor, so backprop
   * flows through `features` exactly at the k selected indices.
   */
  private kSparseForward(batch: tf.Tensor2D): { sparse: tf.Tensor2D; recon: tf.Tensor2D } {
    const centered = tf.sub(batch, this.bPre);
    const preRelu = tf.add(tf.matMul(centered, this.wEnc, false, true), this.bEnc);
    const features = tf.relu(preRelu) as tf.Tensor2D;
    const mask = this.computeTopKMask(features);
    const sparse = tf.mul(features, mask) as tf.Tensor2D;
    const recon = tf.add(
      tf.add(tf.matMul(sparse, this.wDec, false, true), this.bDec),
      this.bPre,
    ) as tf.Tensor2D;
    mask.dispose();
    return { sparse, recon };
  }

  /** Single gradient step on one batch. Updates variables in place. */
  trainStep(batch: tf.Tensor2D): TrainStepResult {
    if (batch.shape[1] !== this.dModel) {
      throw new Error(`batch dim ${batch.shape[1]} != dModel ${this.dModel}`);
    }

    // optimizer.minimize handles variable tracking, backprop, and disposal
    // of intermediate tensors. returnCost=true hands us back the loss scalar.
    const cost = this.optimizer.minimize(() => {
      const { recon } = this.kSparseForward(batch);
      return tf.losses.meanSquaredError(batch, recon) as tf.Scalar;
    }, true);

    if (!cost) {
      throw new Error("optimizer.minimize returned null");
    }
    const reconLoss = (cost.dataSync() as Float32Array)[0];
    cost.dispose();

    // Metrics pass (no-grad). We reuse the same CPU top-k path as the forward
    // to track L0 (should equal k unless features degenerate to all-zeros for
    // some rows) and per-feature fire counts for dead-feature monitoring.
    const featuresArr = tf.tidy(() => {
      const centered = tf.sub(batch, this.bPre);
      const preRelu = tf.add(tf.matMul(centered, this.wEnc, false, true), this.bEnc);
      return (tf.relu(preRelu).dataSync() as Float32Array).slice();
    });
    const { l0, fires } = this.cpuL0AndFires(featuresArr, batch.shape[0]);
    for (let i = 0; i < this.dHidden; i++) {
      if (fires[i] > 0) this.featureFireCount[i] += fires[i];
    }
    this.samplesSinceReset += batch.shape[0];

    return { reconLoss, l0 };
  }

  /**
   * Given a flat features buffer [B, dHidden] from a training step, compute
   * mean L0 (non-zero count among top-k per row) and per-feature fire counts.
   * All CPU, avoids round-tripping through tfjs-node's topk/onehot kernels.
   */
  private cpuL0AndFires(
    featuresFlat: Float32Array,
    batchSize: number,
  ): { l0: number; fires: Float32Array } {
    const D = this.dHidden;
    const k = this.k;
    const fires = new Float32Array(D);
    const valBuf = new Float32Array(k);
    const idxBuf = new Int32Array(k);
    let activeTotal = 0;
    for (let b = 0; b < batchSize; b++) {
      const rowOff = b * D;
      partialTopK(featuresFlat, rowOff, D, k, valBuf, idxBuf);
      for (let t = 0; t < k; t++) {
        if (valBuf[t] > 0) {
          fires[idxBuf[t]]++;
          activeTotal++;
        }
      }
    }
    return { l0: activeTotal / batchSize, fires };
  }

  /** Pure forward pass; returns MSE without touching gradients or variables. */
  evaluate(batch: tf.Tensor2D): number {
    return tf.tidy(() => {
      const { recon } = this.kSparseForward(batch);
      const loss = tf.losses.meanSquaredError(batch, recon) as tf.Scalar;
      return (loss.dataSync() as Float32Array)[0];
    });
  }

  /**
   * Project each dictionary atom (decoder column) to unit L2 norm.
   * Call periodically during training and once at the end.
   */
  normalizeDecoder(): void {
    tf.tidy(() => {
      const norms = tf.norm(this.wDec, 2, 0); // [dHidden]
      const safe = tf.maximum(norms, tf.scalar(1e-8));
      const normalized = tf.div(this.wDec, safe);
      this.wDec.assign(normalized as tf.Tensor2D);
    });
  }

  /** Mean L2 norm of decoder columns (expected ~1.0 after training). */
  meanDecoderColNorm(): number {
    return tf.tidy(() => {
      const norms = tf.norm(this.wDec, 2, 0); // [dHidden]
      return (tf.mean(norms).dataSync() as Float32Array)[0];
    });
  }

  /** Number of features that have never fired since the last reset. */
  deadFeatureCount(): number {
    let dead = 0;
    for (let i = 0; i < this.dHidden; i++) {
      if (this.featureFireCount[i] === 0) dead++;
    }
    return dead;
  }

  resetFeatureFireCount(): void {
    this.featureFireCount.fill(0);
    this.samplesSinceReset = 0;
  }

  /**
   * Snapshot current TF variable state into a runtime-compatible
   * SparseAutoencoder (pure Float32 math, no TF.js dependency).
   */
  exportSAE(): SparseAutoencoder {
    return new SparseAutoencoder({
      dModel: this.dModel,
      dHidden: this.dHidden,
      k: this.k,
      wEnc: new Float32Array(this.wEnc.dataSync() as Float32Array),
      bEnc: new Float32Array(this.bEnc.dataSync() as Float32Array),
      bPre: new Float32Array(this.bPre.dataSync() as Float32Array),
      wDec: new Float32Array(this.wDec.dataSync() as Float32Array),
      bDec: new Float32Array(this.bDec.dataSync() as Float32Array),
    });
  }

  dispose(): void {
    this.wEnc.dispose();
    this.bEnc.dispose();
    this.bPre.dispose();
    this.wDec.dispose();
    this.bDec.dispose();
  }
}

/**
 * In-place min-heap top-k over a contiguous row inside a flat Float32Array.
 * Fills valBuf/idxBuf (length k, unsorted) with the k largest values and
 * their local (within-row) indices.
 */
function partialTopK(
  flat: Float32Array,
  rowOffset: number,
  rowLen: number,
  k: number,
  valBuf: Float32Array,
  idxBuf: Int32Array,
): void {
  for (let i = 0; i < k; i++) {
    valBuf[i] = flat[rowOffset + i];
    idxBuf[i] = i;
  }
  for (let i = (k >> 1) - 1; i >= 0; i--) heapSiftDown(valBuf, idxBuf, i, k);
  for (let i = k; i < rowLen; i++) {
    const v = flat[rowOffset + i];
    if (v > valBuf[0]) {
      valBuf[0] = v;
      idxBuf[0] = i;
      heapSiftDown(valBuf, idxBuf, 0, k);
    }
  }
}

function heapSiftDown(vals: Float32Array, idx: Int32Array, start: number, n: number): void {
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
