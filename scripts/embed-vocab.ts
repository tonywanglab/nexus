#!/usr/bin/env npx ts-node
/**
 * Embed the Numberbatch vocabulary (one concept per line from build-vocab.ts)
 * using the same Arctic model the plugin uses at runtime. Writes a raw Float32 LE
 * blob of shape [vocabSize, dims] and a manifest, resumable if interrupted.
 *
 * Usage:
 *   npm run vocab:embed
 *   npm run vocab:embed -- --input=data/vocab.txt --output=data/vocab-embeddings.f32.bin
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import * as crypto from "crypto";
import { PipelineEmbeddingProvider } from "./lib/node-providers";

const FLOAT_SIZE = 4;

interface CliOptions {
  input: string;
  output: string;
  manifest: string;
  model: string;
  dims: number;
  batchSize: number;
  flushEvery: number;
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
  return {
    input: str("input", "data/vocab.txt"),
    output: str("output", "data/vocab-embeddings.f32.bin"),
    manifest: str("manifest", "data/vocab-manifest.json"),
    model: str("model", "Snowflake/snowflake-arctic-embed-xs"),
    dims: num("dims", 384),
    batchSize: num("batch-size", 64),
    flushEvery: num("flush-every", 5_000),
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
  const opts = parseCli(process.argv);
  process.stderr.write(`embed-vocab options: ${JSON.stringify(opts, null, 2)}\n`);

  if (!fs.existsSync(opts.input)) {
    throw new Error(`Vocab file not found: ${opts.input}. Run npm run vocab:build first.`);
  }

  fs.mkdirSync(path.dirname(opts.output), { recursive: true });
  fs.mkdirSync(path.dirname(opts.manifest), { recursive: true });

  process.stderr.write(`Counting vocab lines... `);
  const totalRows = countLines(opts.input);
  process.stderr.write(`${totalRows.toLocaleString()}\n`);

  process.stderr.write(`Hashing input file... `);
  const inputSha = sha256File(opts.input);
  process.stderr.write(`${inputSha.slice(0, 16)}...\n`);

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
        process.stderr.write(`Resuming from row ${rowsWritten.toLocaleString()} / ${totalRows.toLocaleString()}\n`);
        if (rowsWritten >= totalRows) {
          process.stderr.write(`Already complete.\n`);
          saveManifest(opts.manifest, { ...existing, complete: true, rowsWritten });
          return;
        }
      }
    }
  }

  const provider = new PipelineEmbeddingProvider(opts.model, opts.dims);
  const fd = fs.openSync(opts.output, rowsWritten > 0 ? "a" : "w");

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

  const rl = readline.createInterface({ input: fs.createReadStream(opts.input), crlfDelay: Infinity });
  const allLines: string[] = [];
  for await (const line of rl) {
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
      process.stderr.write(`${done.toLocaleString()} / ${totalRows.toLocaleString()}\n`);
    }
  }

  fs.closeSync(fd);
  manifest.rowsWritten = totalRows;
  manifest.complete = true;
  manifest.updatedAt = new Date().toISOString();
  saveManifest(opts.manifest, manifest);
  await provider.dispose();

  process.stderr.write(`Done. Wrote ${totalRows.toLocaleString()} embeddings to ${opts.output}\n`);
}

main().catch(err => { console.error(err); process.exit(1); });
