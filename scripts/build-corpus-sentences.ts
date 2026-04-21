#!/usr/bin/env npx ts-node
/**
 * Stream Wikipedia abstracts dump and emit deduped sentence lines for Gemma / SAE training.
 *
 * Prereq (download once, ~1GB):
 *   curl -L -o data/wiki/enwiki-latest-abstract.xml.gz \\
 *     https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz
 *
 * Usage:
 *   npm run corpus:build-sentences
 *   npm run corpus:build-sentences -- --input=data/wiki/enwiki-latest-abstract.xml.gz \\
 *     --output=data/wiki/corpus-sentences.txt --max-sentences=1000000
 */

import * as fs from "fs";
import * as path from "path";
import * as zlib from "zlib";
import * as sax from "sax";
import {
  stripBasicXmlEntities,
  splitIntoSentences,
  keepSentenceIfLengthOk,
} from "../src/corpus/wiki-abstract-sentences";

const MIN_LEN = 30;
const MAX_LEN = 200;

interface CliOptions {
  input: string;
  output: string;
  maxSentences: number;
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
    input: str("input", "data/wiki/enwiki-latest-abstract.xml.gz"),
    output: str("output", "data/wiki/corpus-sentences.txt"),
    maxSentences: num("max-sentences", 1_000_000),
  };
}

async function main(): Promise<void> {
  const opts = parseCli(process.argv);
  process.stderr.write(`build-corpus-sentences: ${JSON.stringify(opts, null, 2)}\n`);

  if (!fs.existsSync(opts.input)) {
    throw new Error(`Input not found: ${opts.input}`);
  }

  const dir = path.dirname(opts.output);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const seen = new Set<string>();
  let written = 0;
  const fd = fs.openSync(opts.output, "w");

  const flushAbstract = (raw: string): void => {
    if (written >= opts.maxSentences) return;
    const cleaned = stripBasicXmlEntities(raw);
    for (const sent of splitIntoSentences(cleaned)) {
      if (written >= opts.maxSentences) break;
      const kept = keepSentenceIfLengthOk(sent, MIN_LEN, MAX_LEN);
      if (!kept) continue;
      const key = kept.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      fs.writeSync(fd, Buffer.from(kept + "\n", "utf8"));
      written++;
      if (written % 50_000 === 0) {
        process.stderr.write(`  ${written.toLocaleString()} sentences…\n`);
      }
    }
  };

  await new Promise<void>((resolve, reject) => {
    const parser = sax.createStream(true, { lowercase: true });
    let abstractNest = 0;
    let abstractBuf = "";

    parser.on("opentag", (node) => {
      if (node.name === "abstract") {
        abstractNest++;
        if (abstractNest === 1) abstractBuf = "";
      }
    });

    parser.on("closetag", (name) => {
      if (name !== "abstract") return;
      abstractNest = Math.max(0, abstractNest - 1);
      if (abstractNest === 0) {
        flushAbstract(abstractBuf);
        abstractBuf = "";
      }
    });

    parser.on("text", (t) => {
      if (abstractNest > 0) abstractBuf += t;
    });

    parser.on("error", reject);
    parser.on("end", () => resolve());

    const gunzip = zlib.createGunzip();
    gunzip.on("error", reject);
    fs.createReadStream(opts.input).pipe(gunzip).pipe(parser);
  });

  fs.closeSync(fd);
  process.stderr.write(
    `Wrote ${written.toLocaleString()} deduped sentences to ${opts.output}\n`,
  );
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
