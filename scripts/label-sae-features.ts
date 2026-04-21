#!/usr/bin/env npx ts-node
/**
 * Vocabulary projection: label each SAE decoder atom (column of W_dec) with
 * the top-3 Numberbatch concept strings whose embeddings (same model as
 * `npm run vocab:embed`, default EmbeddingGemma 768-d) are most cosine-similar
 * to that atom.
 *
 * Prereqs (run in order):
 *   npm run vocab:build   →  data/vocab.txt
 *   npm run vocab:embed   →  data/vocab-embeddings.f32.bin + data/vocab-manifest.json
 *
 * Usage:
 *   npm run label:sae-features
 *   npm run label:sae-features -- --weights=assets/sae-weights.bin \
 *     --vocab=data/vocab.txt --vocab-embeddings=data/vocab-embeddings.f32.bin \
 *     --min-score=0.25 --output=assets/sae-feature-labels.json \
 *     --preview=assets/sae-feature-labels-preview.md
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { SparseAutoencoder } from "../src/resolver/sae";

const LABELS_PER_FEATURE = 3;

interface CliOptions {
  weights: string;
  vocab: string;
  vocabEmbeddings: string;
  minScore: number;
  output: string;
  preview: string;
}

function parseCli(argv: string[]): CliOptions {
  const opts: Record<string, string> = {};
  for (const arg of argv.slice(2)) {
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) opts[m[1]] = m[2];
  }
  return {
    weights: opts["weights"] ?? "assets/sae-weights.bin",
    vocab: opts["vocab"] ?? "data/vocab.txt",
    vocabEmbeddings: opts["vocab-embeddings"] ?? "data/vocab-embeddings.f32.bin",
    minScore: opts["min-score"] ? Number(opts["min-score"]) : 0.25,
    output: opts["output"] ?? "assets/sae-feature-labels.json",
    preview: opts["preview"] ?? "assets/sae-feature-labels-preview.md",
  };
}

interface FeatureLabel {
  candidates: string[];
  scores: number[];
}

async function readVocab(file: string): Promise<string[]> {
  const rl = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
  const vocab: string[] = [];
  for await (const line of rl) if (line.trim()) vocab.push(line.trim());
  return vocab;
}

/** Find top-N indices in a Float32Array without fully sorting it (O(vocabSize * N)). */
function topN(arr: Float32Array, n: number): { idx: number; val: number }[] {
  const top: { idx: number; val: number }[] = [];
  for (let i = 0; i < arr.length; i++) {
    const val = arr[i];
    if (top.length < n) {
      top.push({ idx: i, val });
      if (top.length === n) top.sort((a, b) => a.val - b.val); // min-heap order
    } else if (val > top[0].val) {
      top[0] = { idx: i, val };
      // re-sort to maintain min at front
      let j = 0;
      while (j * 2 + 1 < n) {
        const l = j * 2 + 1, r = j * 2 + 2;
        const smallest = r < n && top[r].val < top[l].val ? r : l;
        if (top[smallest].val < top[j].val) {
          [top[j], top[smallest]] = [top[smallest], top[j]];
          j = smallest;
        } else break;
      }
    }
  }
  return top.sort((a, b) => b.val - a.val);
}

async function main(): Promise<void> {
  const opts = parseCli(process.argv);

  for (const [label, file] of [
    ["weights", opts.weights],
    ["vocab", opts.vocab],
    ["vocab-embeddings", opts.vocabEmbeddings],
  ] as const) {
    if (!fs.existsSync(file)) {
      throw new Error(`${label} file not found: ${file}`);
    }
  }

  process.stderr.write(`Loading SAE weights from ${opts.weights}...\n`);
  const weightsBuf = fs.readFileSync(opts.weights);
  const sae = SparseAutoencoder.deserialize(weightsBuf);
  const { dModel, dHidden } = sae;
  process.stderr.write(`SAE: dModel=${dModel}, dHidden=${dHidden}, k=${sae.k}\n`);

  process.stderr.write(`Reading vocab from ${opts.vocab}...\n`);
  const vocab = await readVocab(opts.vocab);
  const vocabSize = vocab.length;
  process.stderr.write(`Vocab size: ${vocabSize.toLocaleString()}\n`);

  process.stderr.write(`Loading vocab embeddings from ${opts.vocabEmbeddings}...\n`);
  const embBuf = fs.readFileSync(opts.vocabEmbeddings);
  const FLOAT_SIZE = 4;
  const expectedBytes = vocabSize * dModel * FLOAT_SIZE;
  if (embBuf.byteLength < expectedBytes) {
    throw new Error(
      `Embeddings file is ${embBuf.byteLength} bytes, expected at least ${expectedBytes} ` +
      `(${vocabSize} vocab × ${dModel} dims × 4). Run npm run vocab:embed first.`,
    );
  }
  const vocabEmbeddings = new Float32Array(embBuf.buffer, embBuf.byteOffset, vocabSize * dModel);

  process.stderr.write(`Labeling ${dHidden} SAE features...\n`);
  const labels: FeatureLabel[] = new Array(dHidden);
  let liveCount = 0;

  // Dot products buffer: one score per vocab entry.
  const scores = new Float32Array(vocabSize);

  for (let j = 0; j < dHidden; j++) {
    // Extract decoder column j: wDec is [dModel, dHidden] row-major.
    // atom[i] = wDec[i * dHidden + j]
    for (let v = 0; v < vocabSize; v++) {
      let dot = 0;
      for (let i = 0; i < dModel; i++) {
        dot += sae.wDec[i * dHidden + j] * vocabEmbeddings[v * dModel + i];
      }
      scores[v] = dot;
    }

    const top = topN(scores, LABELS_PER_FEATURE);
    if (top[0].val >= opts.minScore) {
      labels[j] = {
        candidates: top.map(t => vocab[t.idx]),
        scores: top.map(t => t.val),
      };
      liveCount++;
    } else {
      labels[j] = { candidates: [], scores: [] };
    }

    if ((j + 1) % 100 === 0 || j + 1 === dHidden) {
      process.stderr.write(`  ${j + 1}/${dHidden} features labeled (${liveCount} live)\n`);
    }
  }

  // Write output JSON.
  const outputJson = {
    dHidden,
    vocabSize,
    vocabSource: "numberbatch-en-19.08",
    minScore: opts.minScore,
    labelsPerFeature: LABELS_PER_FEATURE,
    labels,
  };
  fs.mkdirSync(path.dirname(opts.output), { recursive: true });
  fs.writeFileSync(opts.output, JSON.stringify(outputJson));
  process.stderr.write(`Wrote labels to ${opts.output} (${liveCount}/${dHidden} live)\n`);

  // Write preview markdown: 40 random live features.
  const liveIndices = labels
    .map((l, i) => ({ l, i }))
    .filter(x => x.l.candidates.length > 0)
    .map(x => x.i);

  // Shuffle with Fisher-Yates (seeded enough for preview purposes).
  for (let i = liveIndices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [liveIndices[i], liveIndices[j]] = [liveIndices[j], liveIndices[i]];
  }
  const sample = liveIndices.slice(0, 40);

  const lines: string[] = [
    `# SAE Feature Labels Preview`,
    ``,
    `Generated from: ${opts.weights}  `,
    `Vocab: ${vocabSource(opts.vocab)} (${vocabSize.toLocaleString()} concepts)  `,
    `Live features: ${liveCount} / ${dHidden}  `,
    `Min score: ${opts.minScore}  `,
    ``,
    `| Feature | Label | Scores |`,
    `|---------|-------|--------|`,
  ];
  for (const i of sample.sort((a, b) => a - b)) {
    const l = labels[i];
    const label = l.candidates.join(" · ");
    const scores = l.scores.map(s => s.toFixed(3)).join(", ");
    lines.push(`| ${i} | ${label} | ${scores} |`);
  }

  fs.mkdirSync(path.dirname(opts.preview), { recursive: true });
  fs.writeFileSync(opts.preview, lines.join("\n") + "\n", "utf8");
  process.stderr.write(`Wrote preview to ${opts.preview}\n`);
}

function vocabSource(file: string): string {
  return path.basename(file);
}

main().catch(err => { console.error(err); process.exit(1); });
