#!/usr/bin/env npx ts-node
/**
 * Generate a small synthetic corpus + matching Float32 embeddings so the full
 * training pipeline (train-sae.ts) can run end-to-end without the Wikipedia
 * dumps. The embeddings are structured (each "concept" has a random basis
 * vector and each sample is a sparse combination) so the SAE should actually
 * learn something meaningful.
 *
 * Outputs:
 *   data/wiki/corpus.txt
 *   data/wiki/embeddings.f32.bin
 *   data/wiki/embeddings-manifest.json
 *
 * Pure dev helper; NOT for production weights.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

const DIMS = 384;
const N_CONCEPTS = 48;
const N_SAMPLES = 4096;
const SEED = 0xA5A5A5;

const OUT_DIR = "data/wiki";
const CORPUS = path.join(OUT_DIR, "corpus.txt");
const EMB = path.join(OUT_DIR, "embeddings.f32.bin");
const MANIFEST = path.join(OUT_DIR, "embeddings-manifest.json");

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

function gauss(rng: () => number): number {
  let u1 = rng();
  if (u1 < 1e-12) u1 = 1e-12;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const rng = mulberry32(SEED);

// Build random concept basis vectors (unit norm)
const concepts: Float32Array[] = [];
for (let c = 0; c < N_CONCEPTS; c++) {
  const v = new Float32Array(DIMS);
  let sq = 0;
  for (let d = 0; d < DIMS; d++) {
    v[d] = gauss(rng);
    sq += v[d] * v[d];
  }
  const n = Math.sqrt(sq);
  for (let d = 0; d < DIMS; d++) v[d] /= n;
  concepts.push(v);
}

if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

// Samples + matching text strings
const corpusLines: string[] = [];
const embBuf = Buffer.alloc(N_SAMPLES * DIMS * 4);

for (let i = 0; i < N_SAMPLES; i++) {
  const a = Math.floor(rng() * N_CONCEPTS);
  let b = Math.floor(rng() * N_CONCEPTS);
  if (b === a) b = (b + 1) % N_CONCEPTS;
  const w1 = rng();
  const w2 = rng();
  const x = new Float32Array(DIMS);
  for (let d = 0; d < DIMS; d++) {
    x[d] = w1 * concepts[a][d] + w2 * concepts[b][d] + 0.05 * gauss(rng);
  }
  let sq = 0;
  for (let d = 0; d < DIMS; d++) sq += x[d] * x[d];
  const n = Math.sqrt(sq);
  for (let d = 0; d < DIMS; d++) x[d] /= n;

  for (let d = 0; d < DIMS; d++) {
    embBuf.writeFloatLE(x[d], (i * DIMS + d) * 4);
  }
  corpusLines.push(`concept_${a}_${b}_sample_${i}`);
}

fs.writeFileSync(CORPUS, corpusLines.join("\n") + "\n");
fs.writeFileSync(EMB, embBuf);

const inputSha = crypto
  .createHash("sha256")
  .update(fs.readFileSync(CORPUS))
  .digest("hex");

fs.writeFileSync(
  MANIFEST,
  JSON.stringify(
    {
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      model: "synthetic",
      dims: DIMS,
      totalRows: N_SAMPLES,
      rowsWritten: N_SAMPLES,
      inputFile: CORPUS,
      inputSha256: inputSha,
      complete: true,
    },
    null,
    2,
  ),
);

console.log(`Wrote ${N_SAMPLES} synthetic samples to ${EMB} (${DIMS}-dim)`);
console.log(`Corpus: ${CORPUS}, manifest: ${MANIFEST}`);
