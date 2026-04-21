#!/usr/bin/env npx ts-node
/**
 * Embed the training corpus one line at a time using the same model the plugin
 * uses at runtime (Snowflake/snowflake-arctic-embed-xs, 384-dim). Writes a raw
 * Float32 LE blob of shape [N, dims] to disk, resumable if interrupted.
 *
 * Usage:
 *   npm run wiki:embed
 *   npm run wiki:embed -- --batch-size=64 --flush-every=10000
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

function parseCli(argv: string[]): CliOptions {
  const opts: Record<string, string> = {};
  for (const arg of argv.slice(2)) {
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) opts[m[1]] = m[2];
  }
  const num = (k: string, d: number): number => (opts[k] ? Number(opts[k]) : d);
  const str = (k: string, d: string): string => opts[k] ?? d;
  return {
    input: str("input", "data/wiki/corpus.txt"),
    output: str("output", "data/wiki/embeddings.f32.bin"),
    manifest: str("manifest", "data/wiki/embeddings-manifest.json"),
    model: str("model", "Snowflake/snowflake-arctic-embed-xs"),
    dims: num("dims", 384),
    batchSize: num("batch-size", 64),
    flushEvery: num("flush-every", 10_000),
  };
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

function loadManifest(p: string): Manifest | null {
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf8")) as Manifest;
  } catch {
    return null;
  }
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
  // Non-newline-terminated tail line
  if (buf.length > 0 && buf[buf.length - 1] !== 0x0a) n++;
  return n;
}

function ensureDir(file: string): void {
  const dir = path.dirname(file);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

async function main(): Promise<void> {
  const opts = parseCli(process.argv);
  process.stderr.write(`Embedding corpus with options: ${JSON.stringify(opts, null, 2)}\n`);

  if (!fs.existsSync(opts.input)) {
    throw new Error(`Corpus file not found: ${opts.input}`);
  }

  ensureDir(opts.output);
  ensureDir(opts.manifest);

  process.stderr.write(`Counting corpus lines... `);
  const totalRows = countLines(opts.input);
  process.stderr.write(`${totalRows.toLocaleString()}\n`);

  process.stderr.write(`Hashing input file... `);
  const inputSha = sha256File(opts.input);
  process.stderr.write(`${inputSha.slice(0, 16)}...\n`);

  // Resume logic: trust the bin file size as the source of truth (manifest
  // may lag behind if the previous run crashed between batch writes and the
  // next checkpoint flush).
  const existing = loadManifest(opts.manifest);
  let rowsWritten = 0;
  let resume = false;

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
        resume = true;
        process.stderr.write(
          `Resuming from row ${rowsWritten.toLocaleString()} / ${totalRows.toLocaleString()} ` +
            `(manifest said ${existing.rowsWritten.toLocaleString()}, bin has ${rowsWritten.toLocaleString()})\n`,
        );
        if (rowsWritten >= totalRows) {
          process.stderr.write(`Bin file is complete; updating manifest and exiting.\n`);
          const finalized = {
            ...existing,
            rowsWritten,
            updatedAt: new Date().toISOString(),
            complete: true,
          };
          saveManifest(opts.manifest, finalized);
          return;
        }
      } else {
        process.stderr.write(
          `Bin size ${onDisk} not a multiple of row size ${rowBytes}; restarting\n`,
        );
      }
    } else {
      process.stderr.write(`Manifest mismatch (input changed); restarting\n`);
    }
  }

  if (!resume) {
    fs.writeFileSync(opts.output, new Uint8Array(0));
  }

  const manifest: Manifest = {
    createdAt: existing?.createdAt ?? new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    model: opts.model,
    dims: opts.dims,
    totalRows,
    rowsWritten,
    inputFile: opts.input,
    inputSha256: inputSha,
    complete: false,
  };
  saveManifest(opts.manifest, manifest);

  // Open the output file for appending raw Float32 bytes.
  const fd = fs.openSync(opts.output, "a");

  const provider = new PipelineEmbeddingProvider(opts.model, opts.dims);
  process.stderr.write(`Warming up ${opts.model}...\n`);
  await provider.embedBatch(["warmup"]);

  const rl = readline.createInterface({
    input: fs.createReadStream(opts.input),
    crlfDelay: Infinity,
  });

  let skipped = 0;
  let processed = rowsWritten;
  const startedAt = Date.now();
  let lastLog = Date.now();
  let lastFlush = rowsWritten;

  const batch: string[] = [];

  const flushBatch = async (): Promise<void> => {
    if (batch.length === 0) return;
    const embeddings = await provider.embedBatch(batch);
    if (embeddings.length !== batch.length) {
      throw new Error(`batch size mismatch: ${batch.length} -> ${embeddings.length}`);
    }
    const buf = Buffer.alloc(batch.length * opts.dims * FLOAT_SIZE);
    for (let i = 0; i < embeddings.length; i++) {
      const v = embeddings[i];
      if (v.length !== opts.dims) {
        throw new Error(`embedding dim ${v.length} != expected ${opts.dims}`);
      }
      for (let d = 0; d < opts.dims; d++) {
        buf.writeFloatLE(v[d], (i * opts.dims + d) * FLOAT_SIZE);
      }
    }
    fs.writeSync(fd, buf);
    processed += batch.length;
    batch.length = 0;
  };

  try {
    try {
      for await (const line of rl) {
        if (skipped < rowsWritten) {
          skipped++;
          continue;
        }
        batch.push(line);
        if (batch.length >= opts.batchSize) {
          await flushBatch();
        }

      if (Date.now() - lastLog > 2000) {
        lastLog = Date.now();
        const elapsed = (Date.now() - startedAt) / 1000;
        const done = processed - rowsWritten;
        const rate = done / Math.max(1, elapsed);
        const remaining = totalRows - processed;
        const eta = remaining / Math.max(0.1, rate);
        process.stderr.write(
          `  ${processed.toLocaleString()} / ${totalRows.toLocaleString()} ` +
            `(${(rate | 0).toLocaleString()} rows/s, ETA ${formatDuration(eta)})\n`,
        );
      }

        if (processed - lastFlush >= opts.flushEvery) {
          fs.fsyncSync(fd);
          manifest.rowsWritten = processed;
          manifest.updatedAt = new Date().toISOString();
          saveManifest(opts.manifest, manifest);
          lastFlush = processed;
        }
      }
    } catch (err) {
      // Node 24+ can throw ERR_USE_AFTER_CLOSE from the readline async iterator
      // if the underlying stream closes while an await is pending inside the
      // loop body. Treat as clean EOF; we flush whatever's left below.
      if ((err as NodeJS.ErrnoException | undefined)?.code !== "ERR_USE_AFTER_CLOSE") {
        throw err;
      }
    }
    await flushBatch();
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
    await provider.dispose();
  }

  manifest.rowsWritten = processed;
  manifest.updatedAt = new Date().toISOString();
  manifest.complete = processed === totalRows;
  saveManifest(opts.manifest, manifest);
  process.stderr.write(
    `\nDone: ${processed.toLocaleString()} rows in ${formatDuration((Date.now() - startedAt) / 1000)}\n`,
  );
  if (!manifest.complete) {
    process.stderr.write(
      `WARNING: processed ${processed} != totalRows ${totalRows}; not marked complete.\n`,
    );
  }
}

function formatDuration(s: number): string {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(2)}h`;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
