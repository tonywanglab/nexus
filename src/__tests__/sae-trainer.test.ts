// smoke test for SAETrainer: on a tiny synthetic dataset of sparse concept
// combinations (mirroring the article's `generate_structured_data`), a handful
// of gradient steps should reduce MSE and L0 should stay at `k`.
//
// this test imports @tensorflow/tfjs-node which is a heavy native dependency.
// we gate on availability so CI without tfjs-node just skips cleanly.

import { SparseAutoencoder } from "../resolver/sae";

let tf: typeof import("@tensorflow/tfjs-node") | null = null;
let SAETrainer: typeof import("../resolver/sae-trainer").SAETrainer | null = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  tf = require("@tensorflow/tfjs-node");
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  SAETrainer = require("../resolver/sae-trainer").SAETrainer;
} catch {
  // tfjs-node not available in this environment; the describe block below
  // will skip itself.
}

const suite = tf && SAETrainer ? describe : describe.skip;

suite("SAETrainer", () => {
  const dModel = 16;
  const dHidden = 64;
  const k = 4;
  const numConcepts = 8;
  const numSamples = 512;
  const seed = 0xDEADBEEF;

  //  Article-style synthetic data: each sample is a sparse mix of 2 concepts.
  function generateStructuredData(): Float32Array[] {
    const rng = seededRandom(seed);
    const concepts: Float32Array[] = [];
    for (let c = 0; c < numConcepts; c++) {
      const v = new Float32Array(dModel);
      for (let d = 0; d < dModel; d++) v[d] = rng.gauss();
      concepts.push(v);
    }
    const samples: Float32Array[] = [];
    for (let s = 0; s < numSamples; s++) {
      const a = Math.floor(rng.u() * numConcepts);
      let b = Math.floor(rng.u() * numConcepts);
      if (b === a) b = (b + 1) % numConcepts;
      const w1 = rng.u();
      const w2 = rng.u();
      const x = new Float32Array(dModel);
      for (let d = 0; d < dModel; d++) {
        x[d] = w1 * concepts[a][d] + w2 * concepts[b][d] + 0.05 * rng.gauss();
      }
      samples.push(x);
    }
    return samples;
  }

  function seededRandom(s: number): { u: () => number; gauss: () => number } {
    let a = s >>> 0;
    const next = (): number => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
    };
    return {
      u: next,
      gauss: () => {
        let u1 = next();
        if (u1 < 1e-12) u1 = 1e-12;
        const u2 = next();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      },
    };
  }

  it("MSE decreases substantially over 20 epochs and L0 stays at/below k", () => {
    const samples = generateStructuredData();

    const initial = SparseAutoencoder.randomInit(dModel, dHidden, k, seed);
    const trainer = new SAETrainer!(initial, { learningRate: 1e-2 });

    // warm start: pre-bias to sample mean
    const mean = new Float32Array(dModel);
    for (const s of samples) for (let d = 0; d < dModel; d++) mean[d] += s[d];
    for (let d = 0; d < dModel; d++) mean[d] /= samples.length;
    trainer.setPreBias(mean);

    const batchSize = 64;
    const epochs = 20;

    const flat = new Float32Array(samples.length * dModel);
    for (let i = 0; i < samples.length; i++) flat.set(samples[i], i * dModel);

    const idx: number[] = [];
    for (let i = 0; i < samples.length; i++) idx.push(i);

    const epochLosses: number[] = [];
    const epochL0s: number[] = [];
    const rng = seededRandom(seed + 99);

    for (let epoch = 0; epoch < epochs; epoch++) {
      // shuffle
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(rng.u() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }

      let lossSum = 0;
      let l0Sum = 0;
      let count = 0;
      const batchesPerEpoch = Math.floor(samples.length / batchSize);

      for (let b = 0; b < batchesPerEpoch; b++) {
        const batchBuf = new Float32Array(batchSize * dModel);
        for (let i = 0; i < batchSize; i++) {
          const src = idx[b * batchSize + i] * dModel;
          for (let d = 0; d < dModel; d++) {
            batchBuf[i * dModel + d] = flat[src + d];
          }
        }
        const batch = tf!.tensor2d(batchBuf, [batchSize, dModel]);
        const step = trainer.trainStep(batch);
        batch.dispose();
        lossSum += step.reconLoss;
        l0Sum += step.l0;
        count++;
      }
      trainer.normalizeDecoder();
      epochLosses.push(lossSum / count);
      epochL0s.push(l0Sum / count);
    }

    // l0 is bounded above by k (ReLU zeros non-positive features; we only
    // count positive selected activations). At steady state L0 is near k.
    for (const l0 of epochL0s) {
      expect(l0).toBeLessThanOrEqual(k);
      expect(l0).toBeGreaterThan(k * 0.9);
    }
    // MSE must decrease substantially (last epoch << first).
    expect(epochLosses[epochLosses.length - 1]).toBeLessThan(epochLosses[0] * 0.5);

    // exported SAE must never exceed k non-zeros per input (TopK guarantees
    // this structurally). In practice on trained weights it's exactly k.
    const exported = trainer.exportSAE();
    const probe = samples[0];
    let nonzero = 0;
    const enc = exported.encode(probe);
    for (let i = 0; i < enc.length; i++) if (enc[i] !== 0) nonzero++;
    expect(nonzero).toBeLessThanOrEqual(k);

    // decoder columns should be ~unit norm after training.
    for (let j = 0; j < dHidden; j++) {
      let sumSq = 0;
      for (let i = 0; i < dModel; i++) {
        const v = exported.wDec[i * dHidden + j];
        sumSq += v * v;
      }
      expect(Math.sqrt(sumSq)).toBeCloseTo(1, 2);
    }

    trainer.dispose();
  }, 60_000);
});
