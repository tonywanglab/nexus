#!/usr/bin/env npx ts-node
/**
 * Merge multiple one-line-per-record corpus files into a single deduped output.
 * Case-insensitive dedup (keeps the first-seen casing), empty lines dropped.
 *
 * Usage:
 *   npm run corpus:merge
 *   npm run corpus:merge -- \
 *     --inputs=data/wiki/corpus.txt,data/wiki/corpus-sentences.txt \
 *     --output=data/wiki/corpus-titles-sentences.txt
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

interface CliOptions {
  inputs: string[];
  output: string;
}

function parseCli(argv: string[]): CliOptions {
  const opts: Record<string, string> = {};
  for (const arg of argv.slice(2)) {
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) opts[m[1]] = m[2];
  }
  const inputs = (opts["inputs"] ??
    "data/wiki/corpus.txt,data/wiki/corpus-sentences.txt")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const output = opts["output"] ?? "data/wiki/corpus-titles-sentences.txt";
  return { inputs, output };
}

async function streamLines(file: string, onLine: (line: string) => void): Promise<number> {
  let read = 0;
  const rl = readline.createInterface({
    input: fs.createReadStream(file),
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    read++;
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    onLine(trimmed);
  }
  return read;
}

async function main(): Promise<void> {
  const opts = parseCli(process.argv);
  process.stderr.write(`merge-corpora: ${JSON.stringify(opts, null, 2)}\n`);

  for (const f of opts.inputs) {
    if (!fs.existsSync(f)) {
      throw new Error(`Input not found: ${f}`);
    }
  }

  const dir = path.dirname(opts.output);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const seen = new Set<string>();
  const fd = fs.openSync(opts.output, "w");
  let written = 0;
  const perFileKept: Record<string, number> = {};

  try {
    for (const file of opts.inputs) {
      const before = written;
      const read = await streamLines(file, (line) => {
        const key = line.toLowerCase();
        if (seen.has(key)) return;
        seen.add(key);
        fs.writeSync(fd, Buffer.from(line + "\n", "utf8"));
        written++;
      });
      const kept = written - before;
      perFileKept[file] = kept;
      process.stderr.write(
        `  ${file}: read ${read.toLocaleString()}, kept ${kept.toLocaleString()} new\n`,
      );
    }
  } finally {
    fs.closeSync(fd);
  }

  process.stderr.write(
    `Wrote ${written.toLocaleString()} deduped lines to ${opts.output}\n`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
