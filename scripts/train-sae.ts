#!/usr/bin/env npx ts-node
/**
 * Pre-train the k-sparse SAE on the Wikipedia embedding cache.
 *
 * Reads:
 *   - data/wiki/embeddings.f32.bin           (raw Float32 LE, shape [N, dims])
 *   - data/wiki/embeddings-manifest.json     (must be `complete: true`)
 *   - data/wiki/corpus.txt                   (for feature interpretability)
 *
 * Writes:
 *   - assets/sae-weights.bin                 (bundled into the plugin)
 *   - assets/sae-training-report.json        (hyperparams + epoch metrics)
 *   - assets/sae-feature-preview.md          (top-activating strings per feature)
 *
 * Usage:
 *   npm run train:sae
 *   npm run train:sae -- --epochs=50 --batch-size=256 --k=32 --d-hidden=1536
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import * as tf from "@tensorflow/tfjs-node";

import { SparseAutoencoder } from "../src/resolver/sae";
import { SAETrainer } from "../src/resolver/sae-trainer";

interface CliOptions {
  embeddings: string;
  manifest: string;
  corpus: string;
  output: string;
  reportOutput: string;
  previewOutput: string;
  dHidden: number | null;
  expansion: number;
  k: number;
  epochs: number;
  batchSize: number;
  lr: number;
  decoderNormEvery: number;
  valFraction: number;
  preBiasSamples: number;
  seed: number;
  featuresToPreview: number;
}

function parseCli(argv: string[]): CliOptions {
  const opts: Record<string, string> = {};
  for (const arg of argv.slice(2)) {
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) opts[m[1]] = m[2];
  }
  const num = (k: string, d: number): number => (opts[k] !== undefined ? Number(opts[k]) : d);
  const str = (k: string, d: string): string => opts[k] ?? d;
  return {
    embeddings: str("embeddings", "data/wiki/embeddings.f32.bin"),
    manifest: str("manifest", "data/wiki/embeddings-manifest.json"),
    corpus: str("corpus", "data/wiki/corpus.txt"),
    output: str("output", "assets/sae-weights.bin"),
    reportOutput: str("report", "assets/sae-training-report.json"),
    previewOutput: str("preview", "assets/sae-feature-preview.md"),
    dHidden: opts["d-hidden"] ? Number(opts["d-hidden"]) : null,
    expansion: num("expansion", 4),
    k: num("k", 32),
    epochs: num("epochs", 50),
    batchSize: num("batch-size", 256),
    lr: num("lr", 3e-4),
    decoderNormEvery: num("decoder-norm-every", 100),
    valFraction: num("val-fraction", 0.05),
    preBiasSamples: num("pre-bias-samples", 50_000),
    seed: num("seed", 0xC0FFEE),
    featuresToPreview: num("features-to-preview", 20),
  };
}

interface EmbeddingManifest {
  dims: number;
  totalRows: number;
  rowsWritten: number;
  complete: boolean;
  model: string;
}

function loadEmbeddings(manifestPath: string, binPath: string): {
  data: Float32Array;
  manifest: EmbeddingManifest;
} {
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Embedding manifest not found: ${manifestPath}`);
  }
  if (!fs.existsSync(binPath)) {
    throw new Error(`Embedding binary not found: ${binPath}`);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as EmbeddingManifest;
  if (!manifest.complete) {
    throw new Error(`Embeddings not marked complete. Re-run wiki:embed first.`);
  }
  const N = manifest.rowsWritten;
  const D = manifest.dims;
  const expected = N * D * 4;
  const actual = fs.statSync(binPath).size;
  if (actual !== expected) {
    throw new Error(`bin size ${actual} != expected ${expected} for ${N}x${D} float32`);
  }
  const buf = fs.readFileSync(binPath);
  const data = new Float32Array(buf.buffer, buf.byteOffset, N * D);
  return { data: new Float32Array(data), manifest };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function shuffleInPlace<T extends { length: number }>(arr: T, rng: () => number): void {
  const a = arr as unknown as { [k: number]: unknown; length: number };
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }
}

/** Build a batch tensor from a flat [N, D] Float32Array + row index list. */
function gatherBatch(
  data: Float32Array,
  dims: number,
  indices: Int32Array,
  start: number,
  size: number,
): tf.Tensor2D {
  const buf = new Float32Array(size * dims);
  for (let i = 0; i < size; i++) {
    const rowStart = indices[start + i] * dims;
    for (let d = 0; d < dims; d++) {
      buf[i * dims + d] = data[rowStart + d];
    }
  }
  return tf.tensor2d(buf, [size, dims]);
}

function computeMean(data: Float32Array, dims: number, rowIndices: Int32Array): Float32Array {
  const mean = new Float32Array(dims);
  const n = rowIndices.length;
  for (let i = 0; i < n; i++) {
    const off = rowIndices[i] * dims;
    for (let d = 0; d < dims; d++) mean[d] += data[off + d];
  }
  for (let d = 0; d < dims; d++) mean[d] /= n;
  return mean;
}

interface EpochMetrics {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  avgL0: number;
  meanDecColNorm: number;
  deadFeatures: number;
  totalFeatures: number;
  elapsedSec: number;
}

async function main(): Promise<void> {
  const opts = parseCli(process.argv);

  process.stderr.write(`Training options: ${JSON.stringify(opts, null, 2)}\n`);

  // ── Load data ──────────────────────────────────────────────────
  process.stderr.write(`\nLoading embeddings from ${opts.embeddings}\n`);
  const { data, manifest } = loadEmbeddings(opts.manifest, opts.embeddings);
  const dims = manifest.dims;
  const N = manifest.rowsWritten;
  process.stderr.write(`  ${N.toLocaleString()} rows × ${dims} dims (${manifest.model})\n`);

  const dHidden = opts.dHidden ?? dims * opts.expansion;
  if (opts.k > dHidden) {
    throw new Error(`k=${opts.k} > d_hidden=${dHidden}`);
  }

  // ── Train / validation split ──────────────────────────────────
  const rng = mulberry32(opts.seed);
  const allIdx = new Int32Array(N);
  for (let i = 0; i < N; i++) allIdx[i] = i;
  shuffleInPlace(allIdx, rng);

  const valCount = Math.max(1, Math.floor(N * opts.valFraction));
  const trainCount = N - valCount;
  const trainIdx = allIdx.slice(0, trainCount);
  const valIdx = allIdx.slice(trainCount);
  process.stderr.write(`Split: ${trainCount.toLocaleString()} train / ${valCount.toLocaleString()} val\n`);

  // ── Pre-bias from sample mean ─────────────────────────────────
  const preBiasSampleSize = Math.min(opts.preBiasSamples, trainCount);
  const preBiasSampleIdx = trainIdx.slice(0, preBiasSampleSize);
  process.stderr.write(
    `Computing pre-bias from ${preBiasSampleSize.toLocaleString()} samples...\n`,
  );
  const preBiasMean = computeMean(data, dims, preBiasSampleIdx);

  // ── Instantiate SAE + trainer ─────────────────────────────────
  process.stderr.write(
    `Initializing SAE: dModel=${dims}, dHidden=${dHidden}, k=${opts.k}\n`,
  );
  const initial = SparseAutoencoder.randomInit(dims, dHidden, opts.k, opts.seed);
  const trainer = new SAETrainer(initial, { learningRate: opts.lr });
  trainer.setPreBias(preBiasMean);

  // ── Training loop ─────────────────────────────────────────────
  const epochMetrics: EpochMetrics[] = [];
  const t0Total = Date.now();

  // Graceful early stop: Ctrl+C once to finish current epoch + finalize;
  // Ctrl+C twice to abort immediately (default Node behavior).
  let stopRequested = false;
  const onSigint = (): void => {
    if (stopRequested) {
      process.stderr.write(
        `\nSecond SIGINT - aborting immediately without checkpointing.\n`,
      );
      process.exit(130);
    }
    stopRequested = true;
    process.stderr.write(
      `\nSIGINT received - finishing current epoch, then stopping. ` +
        `Ctrl+C again to abort immediately.\n`,
    );
  };
  process.on("SIGINT", onSigint);

  let completedEpochs = 0;
  for (let epoch = 1; epoch <= opts.epochs; epoch++) {
    shuffleInPlace(trainIdx, rng);
    trainer.resetFeatureFireCount();

    const batchesPerEpoch = Math.floor(trainCount / opts.batchSize);
    let stepLossSum = 0;
    let stepLossCount = 0;
    let avgL0Sum = 0;
    let avgL0Count = 0;
    const epochT0 = Date.now();
    let lastLog = Date.now();

    for (let bi = 0; bi < batchesPerEpoch; bi++) {
      const batch = gatherBatch(
        data,
        dims,
        trainIdx,
        bi * opts.batchSize,
        opts.batchSize,
      );

      const step = trainer.trainStep(batch);
      batch.dispose();
      stepLossSum += step.reconLoss;
      stepLossCount++;
      avgL0Sum += step.l0;
      avgL0Count++;

      if ((bi + 1) % opts.decoderNormEvery === 0) {
        trainer.normalizeDecoder();
      }

      if (Date.now() - lastLog > 3000) {
        lastLog = Date.now();
        process.stderr.write(
          `  epoch ${epoch} batch ${bi + 1}/${batchesPerEpoch}  ` +
            `train MSE=${(stepLossSum / stepLossCount).toExponential(3)}  ` +
            `L0=${(avgL0Sum / avgL0Count).toFixed(1)}\n`,
        );
      }
    }
    trainer.normalizeDecoder();

    // Validation
    let valLossSum = 0;
    let valBatchCount = 0;
    for (let bi = 0; bi < Math.floor(valCount / opts.batchSize); bi++) {
      const batch = gatherBatch(data, dims, valIdx, bi * opts.batchSize, opts.batchSize);
      valLossSum += trainer.evaluate(batch);
      batch.dispose();
      valBatchCount++;
    }
    const valLoss = valBatchCount > 0 ? valLossSum / valBatchCount : 0;
    const trainLoss = stepLossSum / Math.max(1, stepLossCount);
    const avgL0 = avgL0Sum / Math.max(1, avgL0Count);
    const meanDecColNorm = trainer.meanDecoderColNorm();
    const deadFeatures = trainer.deadFeatureCount();
    const elapsedSec = (Date.now() - epochT0) / 1000;

    const m: EpochMetrics = {
      epoch,
      trainLoss,
      valLoss,
      avgL0,
      meanDecColNorm,
      deadFeatures,
      totalFeatures: dHidden,
      elapsedSec,
    };
    epochMetrics.push(m);
    completedEpochs = epoch;
    process.stderr.write(
      `Epoch ${epoch}/${opts.epochs} | train=${trainLoss.toExponential(3)} ` +
        `val=${valLoss.toExponential(3)} L0=${avgL0.toFixed(1)} ` +
        `decNorm=${meanDecColNorm.toFixed(3)} dead=${deadFeatures}/${dHidden} ` +
        `(${elapsedSec.toFixed(1)}s)\n`,
    );

    // Checkpoint after every epoch so a later Ctrl+C (or crash) leaves a
    // usable assets/sae-weights.bin from the most recent completed epoch.
    const ckptSAE = trainer.exportSAE();
    const ckptBytes = ckptSAE.serialize();
    ensureDir(opts.output);
    fs.writeFileSync(opts.output, ckptBytes);

    if (stopRequested) {
      process.stderr.write(
        `Early stop after epoch ${epoch}/${opts.epochs}; weights already checkpointed.\n`,
      );
      break;
    }
  }
  process.off("SIGINT", onSigint);

  const totalElapsedSec = (Date.now() - t0Total) / 1000;
  process.stderr.write(
    `\nTraining done in ${(totalElapsedSec / 60).toFixed(1)} min ` +
      `(${completedEpochs}/${opts.epochs} epochs completed)\n`,
  );

  // ── Export + validation stats ─────────────────────────────────
  const finalSAE = trainer.exportSAE();
  const binBytes = finalSAE.serialize();

  ensureDir(opts.output);
  fs.writeFileSync(opts.output, binBytes);
  process.stderr.write(`Wrote ${opts.output} (${binBytes.length.toLocaleString()} bytes)\n`);

  // Per-sample validation MSE distribution
  process.stderr.write(`Computing per-sample val MSE distribution...\n`);
  const valMSEs: number[] = [];
  for (let i = 0; i < valIdx.length; i++) {
    const row = data.subarray(valIdx[i] * dims, (valIdx[i] + 1) * dims);
    const { reconstruction } = finalSAE.forward(new Float32Array(row));
    let sse = 0;
    for (let d = 0; d < dims; d++) {
      const diff = reconstruction[d] - row[d];
      sse += diff * diff;
    }
    valMSEs.push(sse / dims);
  }
  valMSEs.sort((a, b) => a - b);
  const percentile = (p: number): number =>
    valMSEs[Math.min(valMSEs.length - 1, Math.floor(p * valMSEs.length))];
  const valStats = {
    mean: valMSEs.reduce((a, b) => a + b, 0) / valMSEs.length,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
    max: valMSEs[valMSEs.length - 1],
  };
  process.stderr.write(
    `  val MSE: mean=${valStats.mean.toExponential(3)}, ` +
      `p50=${valStats.p50.toExponential(3)}, ` +
      `p95=${valStats.p95.toExponential(3)}, ` +
      `p99=${valStats.p99.toExponential(3)}\n`,
  );

  // Training report
  const report = {
    createdAt: new Date().toISOString(),
    hyperparameters: {
      dModel: dims,
      dHidden,
      k: opts.k,
      epochs: opts.epochs,
      epochsCompleted: completedEpochs,
      batchSize: opts.batchSize,
      learningRate: opts.lr,
      decoderNormEvery: opts.decoderNormEvery,
      valFraction: opts.valFraction,
      seed: opts.seed,
      preBiasSamples: preBiasSampleSize,
      earlyStopped: completedEpochs < opts.epochs,
    },
    model: manifest.model,
    trainSamples: trainCount,
    valSamples: valCount,
    finalMetrics: {
      trainLoss: epochMetrics[epochMetrics.length - 1].trainLoss,
      valLoss: epochMetrics[epochMetrics.length - 1].valLoss,
      avgL0: epochMetrics[epochMetrics.length - 1].avgL0,
      meanDecColNorm: epochMetrics[epochMetrics.length - 1].meanDecColNorm,
      deadFeatures: epochMetrics[epochMetrics.length - 1].deadFeatures,
      valMSEDistribution: valStats,
    },
    epochs: epochMetrics,
    totalElapsedSec,
  };
  ensureDir(opts.reportOutput);
  fs.writeFileSync(opts.reportOutput, JSON.stringify(report, null, 2));
  process.stderr.write(`Wrote ${opts.reportOutput}\n`);

  // Feature interpretability: top-activating strings for N random features
  process.stderr.write(`\nBuilding feature preview (top-activating strings)...\n`);
  await writeFeaturePreview({
    finalSAE,
    data,
    dims,
    corpusFile: opts.corpus,
    outputFile: opts.previewOutput,
    featuresToPreview: opts.featuresToPreview,
    seed: opts.seed,
  });

  trainer.dispose();
  process.stderr.write(`\nAll done.\n`);
}

async function writeFeaturePreview(args: {
  finalSAE: SparseAutoencoder;
  data: Float32Array;
  dims: number;
  corpusFile: string;
  outputFile: string;
  featuresToPreview: number;
  seed: number;
}): Promise<void> {
  const { finalSAE, data, dims, corpusFile, outputFile, featuresToPreview, seed } = args;
  const N = data.length / dims;

  // Sample up to 20k rows for feature analysis (random subset, not sequential).
  const rng = mulberry32(seed ^ 0x5eed);
  const sampleSize = Math.min(20_000, N);
  const sampleIdx = new Int32Array(sampleSize);
  const seen = new Set<number>();
  let filled = 0;
  while (filled < sampleSize) {
    const j = Math.floor(rng() * N);
    if (seen.has(j)) continue;
    seen.add(j);
    sampleIdx[filled++] = j;
  }

  // Read corresponding corpus lines (sorted by row index to scan once).
  const neededRows = new Set<number>();
  for (let i = 0; i < sampleSize; i++) neededRows.add(sampleIdx[i]);
  const rowToLine = new Map<number, string>();
  const rl = readline.createInterface({
    input: fs.createReadStream(corpusFile),
    crlfDelay: Infinity,
  });
  let lineNo = 0;
  for await (const line of rl) {
    if (neededRows.has(lineNo)) rowToLine.set(lineNo, line);
    lineNo++;
  }

  // For each of featuresToPreview random features, find top-5 activations.
  const featuresChosen = new Set<number>();
  const features: number[] = [];
  while (features.length < featuresToPreview) {
    const f = Math.floor(rng() * finalSAE.dHidden);
    if (!featuresChosen.has(f)) {
      featuresChosen.add(f);
      features.push(f);
    }
  }

  interface FeatureTop {
    featureIdx: number;
    topRows: { rowIdx: number; activation: number; text: string }[];
    fireCount: number;
  }
  const results: FeatureTop[] = features.map((f) => ({ featureIdx: f, topRows: [], fireCount: 0 }));

  // Iterate sample rows, maintain top-5 per feature via tiny heaps.
  for (let i = 0; i < sampleSize; i++) {
    const rowIdx = sampleIdx[i];
    const row = data.subarray(rowIdx * dims, (rowIdx + 1) * dims);
    const { indices, values } = finalSAE.encodeSparse(new Float32Array(row));

    // Build map activeFeature -> value
    const activeMap = new Map<number, number>();
    for (let t = 0; t < indices.length; t++) {
      if (values[t] > 0) activeMap.set(indices[t], values[t]);
    }
    for (const ft of results) {
      const val = activeMap.get(ft.featureIdx);
      if (val === undefined) continue;
      ft.fireCount++;
      ft.topRows.push({
        rowIdx,
        activation: val,
        text: rowToLine.get(rowIdx) ?? "<?>",
      });
      ft.topRows.sort((a, b) => b.activation - a.activation);
      if (ft.topRows.length > 5) ft.topRows.length = 5;
    }
  }

  const lines: string[] = [];
  lines.push(`# SAE feature preview`);
  lines.push("");
  lines.push(
    `Generated from ${sampleSize.toLocaleString()} sampled corpus rows, showing top-5 activating strings for ${featuresToPreview} random features.`,
  );
  lines.push("");
  for (const r of results) {
    lines.push(`## Feature ${r.featureIdx}`);
    lines.push(`- fires on ${r.fireCount} / ${sampleSize} sampled rows`);
    if (r.topRows.length === 0) {
      lines.push(`- **dead** (no activations in sample)`);
    } else {
      for (const t of r.topRows) {
        lines.push(`- \`${t.activation.toFixed(4)}\`  ${t.text}`);
      }
    }
    lines.push("");
  }
  ensureDir(outputFile);
  fs.writeFileSync(outputFile, lines.join("\n"), "utf8");
  process.stderr.write(`Wrote ${outputFile}\n`);
}

function ensureDir(file: string): void {
  const dir = path.dirname(file);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
