#!/usr/bin/env npx ts-node
/**
 * Extract concept strings from a Numberbatch word2vec text file and write a
 * deduplicated, normalized vocabulary list to disk.
 *
 * Numberbatch (English, gzipped ~250 MB):
 *   https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
 *
 * Usage:
 *   npm run vocab:build
 *   npm run vocab:fetch   # download the gz (if missing) + build vocab
 *   npm run vocab:build -- --input=data/numberbatch-en-19.08.txt.gz --output=data/vocab.txt
 *   npm run vocab:build -- --fetch   # same as vocab:fetch
 */

import { spawnSync } from "child_process";
import * as fs from "fs";
import * as zlib from "zlib";
import * as readline from "readline";
import * as path from "path";
import { STOPWORDS } from "../src/keyphrase/stopwords";

/** Official ConceptNet S3 mirror for Numberbatch English 19.08. */
export const NUMBERBATCH_EN_19_08_URL =
  "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz";

interface CliOptions {
  input: string;
  output: string;
  fetch: boolean;
}

function parseCli(argv: string[]): CliOptions {
  const opts: Record<string, string> = {};
  let fetch = false;
  for (const arg of argv.slice(2)) {
    if (arg === "--fetch") {
      fetch = true;
      continue;
    }
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) opts[m[1]] = m[2];
  }
  return {
    input: opts["input"] ?? "data/numberbatch-en-19.08.txt.gz",
    output: opts["output"] ?? "data/vocab.txt",
    fetch,
  };
}

function downloadNumberbatchGz(dest: string, url: string = NUMBERBATCH_EN_19_08_URL): void {
  const dir = path.dirname(dest);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${dest}.partial`;
  if (fs.existsSync(tmp)) fs.unlinkSync(tmp);

  const curl = spawnSync(
    "curl",
    ["-fL", "--connect-timeout", "30", "--retry", "3", "--retry-delay", "2", "-o", tmp, url],
    { stdio: "inherit" },
  );
  if (curl.status === 0) {
    fs.renameSync(tmp, dest);
    return;
  }

  if (curl.error && (curl.error as NodeJS.ErrnoException).code === "ENOENT") {
    const wget = spawnSync("wget", ["-O", tmp, url], { stdio: "inherit" });
    if (wget.status === 0) {
      fs.renameSync(tmp, dest);
      return;
    }
  }

  if (fs.existsSync(tmp)) fs.unlinkSync(tmp);
  throw new Error(
    `Download failed (curl exit ${curl.status}). Install curl or wget, or download manually:\n` +
      `  curl -fL -o ${dest} ${url}`,
  );
}

const ALLOWED_CHARS = /^[a-z0-9 \-']+$/;
const HAS_LETTER = /[a-z]/;

function normalizeConcept(raw: string): string {
  return raw.replace(/_/g, " ").toLowerCase();
}

function isValidConcept(concept: string): boolean {
  if (concept.length < 2 || concept.length > 64) return false;
  if (!HAS_LETTER.test(concept)) return false;
  if (!ALLOWED_CHARS.test(concept)) return false;
  // Drop single-token stopwords; keep multi-token entries even if they contain stopwords.
  const tokens = concept.split(" ");
  if (tokens.length === 1 && STOPWORDS.has(tokens[0])) return false;
  return true;
}

async function main() {
  const opts = parseCli(process.argv);

  if (!fs.existsSync(opts.input)) {
    if (opts.fetch) {
      console.error(`Fetching Numberbatch to ${opts.input} ...`);
      downloadNumberbatchGz(opts.input);
    } else {
      console.error(`Input file not found: ${opts.input}`);
      console.error("Run with --fetch to download automatically, or:");
      console.error(`  curl -fL -o ${opts.input} ${NUMBERBATCH_EN_19_08_URL}`);
      process.exit(1);
    }
  }

  fs.mkdirSync(path.dirname(opts.output), { recursive: true });

  const gunzip = zlib.createGunzip();
  const fileStream = fs.createReadStream(opts.input);
  const rl = readline.createInterface({ input: fileStream.pipe(gunzip), crlfDelay: Infinity });

  const seen = new Set<string>();
  const out = fs.createWriteStream(opts.output);

  let lineIdx = 0;
  let written = 0;

  for await (const line of rl) {
    if (lineIdx++ === 0) continue; // skip header (e.g. "516782 300")
    const concept = normalizeConcept(line.split(" ")[0]);
    if (!seen.has(concept) && isValidConcept(concept)) {
      seen.add(concept);
      out.write(concept + "\n");
      written++;
    }
  }

  await new Promise<void>((res, rej) => out.end((err: Error | null | undefined) => (err ? rej(err) : res())));
  console.log(`Wrote ${written} concepts to ${opts.output}`);
}

main().catch(err => { console.error(err); process.exit(1); });
