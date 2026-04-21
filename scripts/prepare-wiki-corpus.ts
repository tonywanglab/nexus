#!/usr/bin/env npx ts-node
/**
 * Build the Wikipedia training corpus for SAE pretraining.
 *
 * Inputs (user-downloaded from https://dumps.wikimedia.org/enwiki/latest/):
 *   - data/wiki/enwiki-latest-all-titles-in-ns0.gz          (titles)
 *   - data/wiki/enwiki-latest-pages-articles-multistream.xml.bz2  (redirects + anchors)
 *
 * Outputs:
 *   - data/wiki/corpus.txt                (one UTF-8 string per line, deduplicated)
 *   - data/wiki/corpus-manifest.json      (per-source counts for reproducibility)
 *
 * Two sequential passes, three reservoirs (titles / redirects / anchors) with
 * user-configurable quotas. Final dedup is case-insensitive and keeps the
 * most-cased form. Early-terminates the XML pass once all reservoirs are full
 * AND at least --max-pages pages have been seen.
 *
 * Usage:
 *   npm run wiki:prepare
 *   npm run wiki:prepare -- --max-samples=500000 --max-pages=5000000
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import * as zlib from "zlib";
import { spawn } from "child_process";
import * as sax from "sax";

// ── CLI parsing ───────────────────────────────────────────────────────

interface CliOptions {
  titlesFile: string;
  xmlFile: string;
  output: string;
  manifest: string;
  maxSamples: number;
  maxPages: number;
  titleRatio: number;
  anchorRatio: number;
  redirectRatio: number;
  bzcat: string;
  seed: number;
  skipXml: boolean;
}

function parseCli(argv: string[]): CliOptions {
  const opts: Record<string, string> = {};
  const flags = new Set<string>();
  for (const arg of argv.slice(2)) {
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) {
      opts[m[1]] = m[2];
    } else if (arg.startsWith("--")) {
      flags.add(arg.slice(2));
    }
  }
  const num = (key: string, def: number): number => (opts[key] ? Number(opts[key]) : def);
  const str = (key: string, def: string): string => opts[key] ?? def;

  const skipXml = flags.has("skip-xml") || opts["skip-xml"] === "true";

  const maxSamples = num("max-samples", 500_000);
  let titleRatio = num("title-ratio", 0.4);
  let anchorRatio = num("anchor-ratio", 0.4);
  let redirectRatio = num("redirect-ratio", 0.2);

  // Titles-only mode: the titles dump already covers both canonical articles
  // and redirect pages (redirects are regular ns0 pages); funnel the entire
  // budget into titles rather than trying to sample from an absent XML file.
  if (skipXml) {
    titleRatio = 1.0;
    anchorRatio = 0;
    redirectRatio = 0;
  }

  const sum = titleRatio + anchorRatio + redirectRatio;
  if (Math.abs(sum - 1.0) > 1e-6) {
    throw new Error(`ratios must sum to 1.0, got ${sum}`);
  }

  return {
    titlesFile: str("titles-file", "data/wiki/enwiki-latest-all-titles-in-ns0.gz"),
    xmlFile: str("xml-file", "data/wiki/enwiki-latest-pages-articles-multistream.xml.bz2"),
    output: str("output", "data/wiki/corpus.txt"),
    manifest: str("manifest", "data/wiki/corpus-manifest.json"),
    maxSamples,
    maxPages: num("max-pages", 5_000_000),
    titleRatio,
    anchorRatio,
    redirectRatio,
    bzcat: str("bzcat", "bzcat"),
    seed: num("seed", 0x1234_5678),
    skipXml,
  };
}

// ── Reservoir sampling ────────────────────────────────────────────────

class Reservoir {
  private items: string[] = [];
  private rng: () => number;
  public seen = 0;
  public readonly capacity: number;

  constructor(capacity: number, seed: number) {
    this.capacity = capacity;
    this.rng = mulberry32(seed);
  }

  add(item: string): void {
    this.seen++;
    if (this.items.length < this.capacity) {
      this.items.push(item);
      return;
    }
    const j = Math.floor(this.rng() * this.seen);
    if (j < this.capacity) this.items[j] = item;
  }

  isFull(): boolean {
    return this.items.length >= this.capacity;
  }

  get all(): ReadonlyArray<string> {
    return this.items;
  }
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

// ── String filters ────────────────────────────────────────────────────

const NAMESPACE_PREFIXES = [
  "File:",
  "Category:",
  "Template:",
  "Wikipedia:",
  "Help:",
  "Portal:",
  "Draft:",
  "Module:",
  "MediaWiki:",
  "User:",
  "Talk:",
  "Special:",
];

function cleanAndFilter(raw: string): string | null {
  if (!raw) return null;
  let s = raw.replace(/_/g, " ").trim();
  if (s.length < 3 || s.length > 80) return null;
  if (/^[0-9]+$/.test(s)) return null;
  for (const prefix of NAMESPACE_PREFIXES) {
    if (s.startsWith(prefix)) return null;
  }
  if (/\(disambiguation\)/i.test(s)) return null;
  if (/^List of /i.test(s)) return null;
  if (/^Index of /i.test(s)) return null;
  if (/^Outline of /i.test(s)) return null;

  // ASCII-ratio filter: keep strings that are mostly Latin.
  let nonSpace = 0;
  let ascii = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 32 || c === 9) continue;
    nonSpace++;
    if (c < 128) ascii++;
  }
  if (nonSpace === 0) return null;
  if (ascii / nonSpace < 0.8) return null;

  return s;
}

// ── Titles pass ───────────────────────────────────────────────────────

async function streamTitles(
  gzPath: string,
  reservoir: Reservoir,
): Promise<{ totalLines: number; filtered: number }> {
  if (!fs.existsSync(gzPath)) {
    throw new Error(`Titles file not found: ${gzPath}`);
  }
  const stream = fs.createReadStream(gzPath).pipe(zlib.createGunzip());
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  let totalLines = 0;
  let filtered = 0;
  let lastLog = Date.now();

  for await (const line of rl) {
    totalLines++;
    const clean = cleanAndFilter(line);
    if (clean) {
      filtered++;
      reservoir.add(clean);
    }
    if (totalLines % 500_000 === 0 && Date.now() - lastLog > 2000) {
      lastLog = Date.now();
      process.stderr.write(
        `  titles: ${totalLines.toLocaleString()} read, ${filtered.toLocaleString()} kept\n`,
      );
    }
  }
  return { totalLines, filtered };
}

// ── XML pass ──────────────────────────────────────────────────────────

interface PageBuffer {
  title: string;
  redirect: string | null;
  text: string;
}

function extractAnchors(text: string): string[] {
  const out: string[] = [];
  // Simple regex - handles [[Target]] and [[Target|Anchor]]. Misses nested
  // brackets but that's fine for a training corpus.
  const re = /\[\[([^\[\]|]+)(?:\|([^\[\]]+))?\]\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const anchor = (m[2] ?? m[1]).split("#")[0].trim();
    if (anchor) out.push(anchor);
  }
  return out;
}

interface XmlPassStats {
  pagesRead: number;
  redirectsSeen: number;
  anchorsSeen: number;
}

async function streamPages(
  bz2Path: string,
  bzcatBinary: string,
  maxPages: number,
  redirectRes: Reservoir,
  anchorRes: Reservoir,
  onFull: () => boolean,
): Promise<XmlPassStats> {
  if (!fs.existsSync(bz2Path)) {
    throw new Error(`XML dump not found: ${bz2Path}`);
  }

  return new Promise((resolve, reject) => {
    const stats: XmlPassStats = { pagesRead: 0, redirectsSeen: 0, anchorsSeen: 0 };
    const child = spawn(bzcatBinary, [bz2Path], { stdio: ["ignore", "pipe", "pipe"] });
    const parser = sax.createStream(true, { lowercase: true });

    let current: PageBuffer = { title: "", redirect: null, text: "" };
    let inTitle = false;
    let inText = false;
    let depth = 0;
    let stopped = false;
    let lastLog = Date.now();

    const stop = (): void => {
      if (stopped) return;
      stopped = true;
      try {
        parser.end();
      } catch {
        // ignore
      }
      try {
        child.kill("SIGTERM");
      } catch {
        // ignore
      }
      resolve(stats);
    };

    parser.on("opentag", (node) => {
      depth++;
      const name = node.name;
      if (name === "page") {
        current = { title: "", redirect: null, text: "" };
      } else if (name === "title" && depth === 3) {
        inTitle = true;
      } else if (name === "redirect" && depth === 3) {
        const t = node.attributes.title;
        current.redirect = typeof t === "string" ? t : null;
      } else if (name === "text") {
        inText = true;
      }
    });

    parser.on("closetag", (name) => {
      if (name === "title") inTitle = false;
      if (name === "text") inText = false;
      if (name === "page") {
        stats.pagesRead++;

        if (current.redirect) {
          stats.redirectsSeen++;
          // Both sides of the redirect are valuable synonym samples.
          const clean1 = cleanAndFilter(current.title);
          const clean2 = cleanAndFilter(current.redirect);
          if (clean1) redirectRes.add(clean1);
          if (clean2) redirectRes.add(clean2);
        } else if (current.text) {
          const anchors = extractAnchors(current.text);
          stats.anchorsSeen += anchors.length;
          for (const a of anchors) {
            const clean = cleanAndFilter(a);
            if (clean) anchorRes.add(clean);
          }
        }

        if (stats.pagesRead % 100_000 === 0 && Date.now() - lastLog > 2000) {
          lastLog = Date.now();
          process.stderr.write(
            `  xml: ${stats.pagesRead.toLocaleString()} pages, ` +
              `redirects ${stats.redirectsSeen.toLocaleString()}, ` +
              `anchors ${stats.anchorsSeen.toLocaleString()}\n`,
          );
        }

        if (stats.pagesRead >= maxPages && redirectRes.isFull() && anchorRes.isFull() && onFull()) {
          stop();
        }
      }
      depth--;
    });

    parser.on("text", (t) => {
      if (inTitle) current.title += t;
      if (inText) current.text += t;
    });

    parser.on("error", (err) => {
      if (stopped) return;
      stopped = true;
      try {
        child.kill("SIGTERM");
      } catch {
        // ignore
      }
      reject(err);
    });
    parser.on("end", () => {
      if (!stopped) {
        stopped = true;
        resolve(stats);
      }
    });

    child.stderr.on("data", (data: Buffer) => {
      process.stderr.write(`  [${bzcatBinary}] ${data}`);
    });
    child.on("error", (err) => {
      if (stopped) return;
      stopped = true;
      reject(err);
    });
    child.on("exit", (code, signal) => {
      // SIGTERM is expected when we early-terminate.
      if (!stopped && code !== 0 && signal !== "SIGTERM") {
        stopped = true;
        reject(new Error(`${bzcatBinary} exited with code ${code} signal ${signal}`));
      }
    });
    child.stdout.pipe(parser);
  });
}

// ── Dedup + write ─────────────────────────────────────────────────────

function dedupPreferCased(items: Iterable<string>): string[] {
  const byLower = new Map<string, string>();
  for (const s of items) {
    const lower = s.toLowerCase();
    const existing = byLower.get(lower);
    if (!existing) {
      byLower.set(lower, s);
    } else {
      const existingUpper = countUpper(existing);
      const newUpper = countUpper(s);
      if (newUpper > existingUpper) byLower.set(lower, s);
    }
  }
  return Array.from(byLower.values());
}

function countUpper(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c >= 65 && c <= 90) n++;
  }
  return n;
}

function ensureDir(file: string): void {
  const dir = path.dirname(file);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// ── Main ──────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseCli(process.argv);
  process.stderr.write(`Preparing corpus with options: ${JSON.stringify(opts, null, 2)}\n`);

  const titleQuota = Math.floor(opts.maxSamples * opts.titleRatio);
  const anchorQuota = Math.floor(opts.maxSamples * opts.anchorRatio);
  const redirectQuota = Math.floor(opts.maxSamples * opts.redirectRatio);

  const titleRes = new Reservoir(titleQuota, opts.seed);
  const anchorRes = new Reservoir(anchorQuota, opts.seed + 1);
  const redirectRes = new Reservoir(redirectQuota, opts.seed + 2);

  process.stderr.write(
    `Quotas: titles=${titleQuota}, anchors=${anchorQuota}, redirects=${redirectQuota}\n`,
  );

  // Pass 1: titles
  const t0 = Date.now();
  process.stderr.write(`\n[1/2] Streaming titles from ${opts.titlesFile}\n`);
  const titleStats = await streamTitles(opts.titlesFile, titleRes);
  process.stderr.write(
    `  done in ${((Date.now() - t0) / 1000).toFixed(1)}s: ` +
      `${titleStats.totalLines.toLocaleString()} lines, ` +
      `${titleStats.filtered.toLocaleString()} passed filters, ` +
      `reservoir ${titleRes.all.length}/${titleQuota}\n`,
  );

  // Pass 2: XML (optional - titles.gz already covers both canonical articles
  // and redirect pages, so skip-xml is valid for a titles-only corpus).
  let xmlStats: XmlPassStats = { pagesRead: 0, redirectsSeen: 0, anchorsSeen: 0 };
  if (!opts.skipXml) {
    const t1 = Date.now();
    process.stderr.write(
      `\n[2/2] Streaming XML from ${opts.xmlFile} via ${opts.bzcat} (max ${opts.maxPages.toLocaleString()} pages)\n`,
    );
    xmlStats = await streamPages(
      opts.xmlFile,
      opts.bzcat,
      opts.maxPages,
      redirectRes,
      anchorRes,
      () => true,
    );
    process.stderr.write(
      `  done in ${((Date.now() - t1) / 1000).toFixed(1)}s: ` +
        `${xmlStats.pagesRead.toLocaleString()} pages, ` +
        `redirects ${redirectRes.all.length}/${redirectQuota}, ` +
        `anchors ${anchorRes.all.length}/${anchorQuota}\n`,
    );
  } else {
    process.stderr.write(`\n[2/2] --skip-xml set; using titles-only corpus.\n`);
  }

  // Dedup + write
  process.stderr.write(`\nMerging reservoirs and deduplicating (case-insensitive)...\n`);
  const merged = dedupPreferCased(
    concatIterables(titleRes.all, redirectRes.all, anchorRes.all),
  );
  merged.sort();
  process.stderr.write(`  ${merged.length.toLocaleString()} unique strings after dedup\n`);

  ensureDir(opts.output);
  fs.writeFileSync(opts.output, merged.join("\n") + "\n", "utf8");

  const manifest = {
    createdAt: new Date().toISOString(),
    options: opts,
    stats: {
      titlesSeen: titleStats.totalLines,
      titlesFiltered: titleStats.filtered,
      titlesSampled: titleRes.all.length,
      pagesRead: xmlStats.pagesRead,
      redirectsSeen: xmlStats.redirectsSeen,
      redirectsSampled: redirectRes.all.length,
      anchorsSeen: xmlStats.anchorsSeen,
      anchorsSampled: anchorRes.all.length,
      totalAfterDedup: merged.length,
    },
  };
  ensureDir(opts.manifest);
  fs.writeFileSync(opts.manifest, JSON.stringify(manifest, null, 2), "utf8");

  process.stderr.write(`\nWrote ${opts.output} and ${opts.manifest}\n`);
}

function* concatIterables<T>(...iters: Array<Iterable<T>>): Iterable<T> {
  for (const it of iters) yield* it;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
