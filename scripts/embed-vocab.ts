#!/usr/bin/env npx ts-node
/**
 * Embed the Numberbatch vocabulary (one concept per line from build-vocab.ts)
 * using the same embedding model as the plugin (default EmbeddingGemma). Writes a raw Float32 LE
 * blob of shape [vocabSize, dims] and a manifest, resumable if interrupted.
 *
 * Usage:
 *   npm run vocab:embed
 *   npm run vocab:embed -- --input=data/vocab.txt --output=data/vocab-embeddings.f32.bin
 */

import * as crypto from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { createNodeEmbeddingProvider } from "./lib/node-providers";
import { readLinesAsync } from "./lib/read-lines";

const FLOAT_SIZE = 4;

function tsLog(line: string): void {
  process.stderr.write(`[${new Date().toISOString()}] ${line}\n`);
}

function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "?";
  const s = Math.floor(seconds % 60);
  const m = Math.floor((seconds / 60) % 60);
  const h = Math.floor(seconds / 3600);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function logRuntimeThreads(device: string): void {
  const cpuCount = os.cpus().length;
  const ap =
    typeof (os as { availableParallelism?: () => number }).availableParallelism === "function"
      ? (os as { availableParallelism: () => number }).availableParallelism()
      : cpuCount;
  tsLog(`Threads/cores: os.cpus().length=${cpuCount}, os.availableParallelism()=${ap}`);
  tsLog(
    `libuv pool: UV_THREADPOOL_SIZE=${process.env.UV_THREADPOOL_SIZE ?? "(default 4)"}`,
  );
  const threadEnvKeys = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
  ] as const;
  for (const k of threadEnvKeys) {
    if (process.env[k]) tsLog(`${k}=${process.env[k]}`);
  }
  try {
    const ort = require("onnxruntime-node") as { env?: { wasm?: { numThreads?: number } } };
    const wt = ort.env?.wasm?.numThreads;
    tsLog(`onnxruntime env.wasm.numThreads=${wt === undefined ? "(unused in Node native EP)" : String(wt)}`);
  } catch {
    tsLog("onnxruntime-node: env not read");
  }
  if (device === "coreml") {
    tsLog(
      "Inference note: CoreML EP uses Apple ANE/GPU; CPU/thread env vars mainly affect CPU fallback or host work.",
    );
  }
}

interface CliOptions {
  input: string;
  output: string;
  manifest: string;
  model: string;
  dims: number;
  batchSize: number;
  flushEvery: number;
  /** Transformers.js ONNX device (default: coreml on macOS, cpu elsewhere). */
  device: string;
}

interface Manifest {
  createdAt: string;
  updatedAt: string;
  model: string;
  dims: number;
  totalRows: number;
  rowsWritten: number;
  inputFile: string;
  inputSha256: string;
  complete: boolean;
}

function parseCli(argv: string[]): CliOptions {
  const opts: Record<string, string> = {};
  for (const arg of argv.slice(2)) {
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) opts[m[1]] = m[2];
  }
  const num = (k: string, d: number) => (opts[k] ? Number(opts[k]) : d);
  const str = (k: string, d: string) => opts[k] ?? d;
  const defaultDevice = process.platform === "darwin" ? "coreml" : "cpu";
  return {
    input: str("input", "data/vocab.txt"),
    output: str("output", "data/vocab-embeddings.f32.bin"),
    manifest: str("manifest", "data/vocab-manifest.json"),
    model: str("model", "onnx-community/embeddinggemma-300m-ONNX"),
    dims: num("dims", 768),
    batchSize: num("batch-size", 64),
    flushEvery: num("flush-every", 5_000),
    device: str("device", defaultDevice),
  };
}

function loadManifest(p: string): Manifest | null {
  if (!fs.existsSync(p)) return null;
  try { return JSON.parse(fs.readFileSync(p, "utf8")) as Manifest; } catch { return null; }
}

function saveManifest(p: string, m: Manifest): void {
  fs.writeFileSync(p, JSON.stringify(m, null, 2), "utf8");
}

function sha256File(p: string): string {
  const hash = crypto.createHash("sha256");
  hash.update(fs.readFileSync(p));
  return hash.digest("hex");
}

function countLines(p: string): number {
  const buf = fs.readFileSync(p);
  let n = 0;
  for (let i = 0; i < buf.length; i++) if (buf[i] === 0x0a) n++;
  if (buf.length > 0 && buf[buf.length - 1] !== 0x0a) n++;
  return n;
}

async function main(): Promise<void> {
  const tPipeline = Date.now();
  tsLog("embed-vocab starting");
  const opts = parseCli(process.argv);
  tsLog(`options: ${JSON.stringify(opts)}`);

  if (!fs.existsSync(opts.input)) {
    throw new Error(`Vocab file not found: ${opts.input}. Run npm run vocab:build first.`);
  }

  fs.mkdirSync(path.dirname(opts.output), { recursive: true });
  fs.mkdirSync(path.dirname(opts.manifest), { recursive: true });

  logRuntimeThreads(opts.device);

  tsLog("Counting vocab lines...");
  const totalRows = countLines(opts.input);
  tsLog(`vocab lines: ${totalRows.toLocaleString()} (${fmtDuration((Date.now() - tPipeline) / 1000)} since start)`);

  tsLog("Hashing input file...");
  const inputSha = sha256File(opts.input);
  tsLog(`input sha256: ${inputSha.slice(0, 16)}...`);

  const existing = loadManifest(opts.manifest);
  let rowsWritten = 0;

  if (existing && fs.existsSync(opts.output)) {
    if (
      existing.inputSha256 === inputSha &&
      existing.model === opts.model &&
      existing.dims === opts.dims &&
      existing.totalRows === totalRows
    ) {
      const onDisk = fs.statSync(opts.output).size;
      const rowBytes = opts.dims * FLOAT_SIZE;
      if (onDisk % rowBytes === 0) {
        rowsWritten = onDisk / rowBytes;
        tsLog(`Resuming from row ${rowsWritten.toLocaleString()} / ${totalRows.toLocaleString()}`);
        if (rowsWritten >= totalRows) {
          tsLog("Already complete.");
          saveManifest(opts.manifest, { ...existing, complete: true, rowsWritten });
          return;
        }
      }
    }
  }

  const provider = createNodeEmbeddingProvider(opts.model, opts.dims, { device: opts.device });
  const fd = fs.openSync(opts.output, rowsWritten > 0 ? "a" : "w");
  const tEmbedStart = Date.now();
  const rowBase = rowsWritten;
  tsLog(`Embedding loop starting (rows ${rowBase.toLocaleString()}→${totalRows.toLocaleString()}, batch=${opts.batchSize})`);

  const now = new Date().toISOString();
  const manifest: Manifest = {
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
    model: opts.model,
    dims: opts.dims,
    totalRows,
    rowsWritten,
    inputFile: opts.input,
    inputSha256: inputSha,
    complete: false,
  };
  saveManifest(opts.manifest, manifest);

  const allLines: string[] = [];
  for await (const line of readLinesAsync(opts.input)) {
    if (line.trim()) allLines.push(line.trim());
  }

  const toEmbed = allLines.slice(rowsWritten);
  let done = rowsWritten;

  for (let i = 0; i < toEmbed.length; i += opts.batchSize) {
    const batch = toEmbed.slice(i, i + opts.batchSize);
    const embeddings = await provider.embedBatch(batch);
    const buf = Buffer.allocUnsafe(embeddings.length * opts.dims * FLOAT_SIZE);
    for (let j = 0; j < embeddings.length; j++) {
      for (let d = 0; d < opts.dims; d++) {
        buf.writeFloatLE(embeddings[j][d], (j * opts.dims + d) * FLOAT_SIZE);
      }
    }
    fs.writeSync(fd, buf);
    done += embeddings.length;

    if (done % opts.flushEvery < opts.batchSize || done === totalRows) {
      manifest.rowsWritten = done;
      manifest.updatedAt = new Date().toISOString();
      saveManifest(opts.manifest, manifest);
      const elapsedSec = (Date.now() - tEmbedStart) / 1000;
      const sessionRows = done - rowBase;
      const rps = elapsedSec > 0.5 ? sessionRows / elapsedSec : 0;
      const remaining = totalRows - done;
      const etaSec = rps > 0 ? remaining / rps : Number.NaN;
      tsLog(
        `progress ${done.toLocaleString()} / ${totalRows.toLocaleString()} | ${rps.toFixed(1)} rows/s (this run) | elapsed ${fmtDuration(elapsedSec)} | ETA ~${fmtDuration(etaSec)}`,
      );
    }
  }

  fs.closeSync(fd);
  manifest.rowsWritten = totalRows;
  manifest.complete = true;
  manifest.updatedAt = new Date().toISOString();
  saveManifest(opts.manifest, manifest);
  await provider.dispose();

  tsLog(
    `Done. Wrote ${totalRows.toLocaleString()} embeddings to ${opts.output} (total wall ${fmtDuration((Date.now() - tPipeline) / 1000)})`,
  );
}

main().catch(err => { console.error(err); process.exit(1); });
