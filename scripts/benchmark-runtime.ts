#!/usr/bin/env npx ts-node
/**
 * Runtime benchmark for the keyphrase → resolver pipeline.
 *
 * Sweeps the most interesting latency knobs over `senior-thesis-vault/`,
 * times each per-note pass in 7 phases, and writes JSON + Markdown summaries.
 *
 * Usage:
 *   npm run bench:runtime                    # default tier-1 sweep, full vault
 *   npm run bench:runtime:quick              # tier-1 sweep, 20-note sample
 *   npm run bench:runtime:full               # all sweeps
 *
 *   npm run bench:runtime -- --sweep=tier2   # device + dtype + threshold
 *   npm run bench:runtime -- --sweep=all     # all of the above
 *   npm run bench:runtime -- --sample=20     # subset of vault
 *   npm run bench:runtime -- --device=cpu --dtype=q8
 *   npm run bench:runtime -- --warm=false    # cold pass only (skips second pass)
 *   npm run bench:runtime -- --vault=/path/to/vault
 */

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { SpanExtractor } from "../src/keyphrase/span-extractor";
import { YakeLite } from "../src/keyphrase/yake-lite";
import { AliasResolver } from "../src/resolver/index";
import {
  EmbeddingResolver,
  PhaseName,
  PhaseRecorder,
} from "../src/resolver/embedding-resolver";
import { EmbeddingProvider } from "../src/resolver/embedding-provider";
import { SparseAutoencoder } from "../src/resolver/sae";
import { SAEFeatureLabels } from "../src/resolver/sae-feature-labels";
import { mergeByTarget } from "../src/resolver/merge-by-target";
import { ExtractedPhrase, VaultContext, CandidateEdge } from "../src/types";
import { loadVault, VaultFile } from "./lib/eval-vault";
import { createNodeEmbeddingProvider } from "./lib/node-providers";

const GEMMA_MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const GEMMA_DIMS = 768;
const STRONG_LCS_DEFAULT = 0.95;
const MAX_EMBED_DEFAULT = 60;
const SIMILARITY_DEFAULT = 0.82;

// ── CLI ─────────────────────────────────────────────────────────

interface CliOptions {
  sweep: "tier1" | "tier2" | "all";
  sample: number;          // 0 = full vault
  device: string;
  dtype: string;
  vault: string;
  warm: boolean;
  out: string;
  md: string;
  saeSweep: boolean;       // sweep across all SAE configs in assets/runs/
  sae: string;             // explicit SAE weights path override
}

function parseCli(argv: string[]): CliOptions {
  const opts: Record<string, string> = {};
  for (const arg of argv.slice(2)) {
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) opts[m[1]] = m[2];
  }
  const defaultDevice = process.platform === "darwin" ? "coreml" : "cpu";
  return {
    sweep: (opts["sweep"] as CliOptions["sweep"]) ?? "tier1",
    sample: opts["sample"] ? Number(opts["sample"]) : 0,
    device: opts["device"] ?? defaultDevice,
    dtype: opts["dtype"] ?? "q8",
    vault: opts["vault"] ?? path.join(__dirname, "..", "senior-thesis-vault"),
    warm: opts["warm"] !== "false",
    out: opts["out"] ?? "data/benchmark-runtime.json",
    md: opts["md"] ?? "assets/benchmark-runtime.md",
    saeSweep: opts["sae-sweep"] === "true",
    sae: opts["sae"] ?? "",
  };
}

// ── Config knobs ────────────────────────────────────────────────

type Pipeline = "lcs-only" | "lcs+dense" | "lcs+dense+sparse";

interface BenchConfig {
  label: string;
  extractor: "span" | "yake";
  maxNgram: number;
  pipeline: Pipeline;
  maxEmbed: number;          // Infinity = no cap
  strongLcs: number;         // 1.0 = disabled (no phrase ever matches)
  similarityThreshold: number;
  device: string;
  dtype: string;
}

function baseline(device: string, dtype: string): BenchConfig {
  return {
    label: "",
    extractor: "span",
    maxNgram: 10,
    pipeline: "lcs+dense+sparse",
    maxEmbed: MAX_EMBED_DEFAULT,
    strongLcs: STRONG_LCS_DEFAULT,
    similarityThreshold: SIMILARITY_DEFAULT,
    device,
    dtype,
  };
}

function labelFor(c: BenchConfig): string {
  const pipe = c.pipeline === "lcs-only"
    ? "LCS"
    : c.pipeline === "lcs+dense"
    ? "LCS+Dense"
    : "LCS+Dense+Sparse";
  const ext = c.extractor === "span" ? "Span" : "YAKE";
  const cap = c.maxEmbed === Infinity ? "∞" : String(c.maxEmbed);
  return `${ext}+${pipe} | maxN=${c.maxNgram} | MAX_EMBED=${cap} | STRONG_LCS=${c.strongLcs} | thr=${c.similarityThreshold}`;
}

function enumerateTier1(device: string, dtype: string): BenchConfig[] {
  const out: BenchConfig[] = [];
  // Extractor sweep
  for (const ext of ["span", "yake"] as const) {
    out.push({ ...baseline(device, dtype), extractor: ext });
  }
  // maxNgram sweep (Span)
  for (const n of [1, 3, 5, 10]) {
    out.push({ ...baseline(device, dtype), maxNgram: n });
  }
  // Pipeline sweep
  for (const p of ["lcs-only", "lcs+dense", "lcs+dense+sparse"] as const) {
    out.push({ ...baseline(device, dtype), pipeline: p });
  }
  // MAX_EMBED sweep. Skipping Infinity here — the benchmark previously
  // OOM-killed on batches of 800+ phrases × 16-token sequences via CoreML.
  // If you want to characterize unbounded behavior, run a separate invocation
  // with a smaller --sample.
  for (const m of [10, 30, 60, 200, 500]) {
    out.push({ ...baseline(device, dtype), maxEmbed: m });
  }
  // STRONG_LCS sweep
  for (const s of [0.85, 0.95, 1.0]) {
    out.push({ ...baseline(device, dtype), strongLcs: s });
  }
  // Dedup by label (baseline appears in multiple sweeps).
  const seen = new Set<string>();
  const dedup: BenchConfig[] = [];
  for (const c of out) {
    c.label = labelFor(c);
    if (seen.has(c.label)) continue;
    seen.add(c.label);
    dedup.push(c);
  }
  return dedup;
}

function enumerateTier2(device: string, dtype: string): BenchConfig[] {
  const out: BenchConfig[] = [];
  // Threshold sweep (device/dtype held at user choice)
  for (const t of [0.7, 0.82, 0.9]) {
    const c = { ...baseline(device, dtype), similarityThreshold: t };
    c.label = labelFor(c);
    out.push(c);
  }
  // Note: device and dtype sweeps are not enumerated automatically because
  // each requires a fresh model load. Run them via separate invocations:
  //   npm run bench:runtime -- --device=cpu
  //   npm run bench:runtime -- --device=coreml --dtype=q4
  return out;
}

function enumerateConfigs(args: CliOptions): BenchConfig[] {
  if (args.sweep === "tier1") return enumerateTier1(args.device, args.dtype);
  if (args.sweep === "tier2") return enumerateTier2(args.device, args.dtype);
  return [...enumerateTier1(args.device, args.dtype), ...enumerateTier2(args.device, args.dtype)];
}

// ── Phase recorder ──────────────────────────────────────────────

const PHASE_NAMES = [
  "extract",
  "lcs",
  "titleEmbed",
  "phraseEmbed",
  "denseMatch",
  "sparseEncode",
  "sparseMatch",
  "merge",
] as const;
type Phase = typeof PHASE_NAMES[number];

class TimingsRecorder implements PhaseRecorder {
  private byPhase = new Map<Phase, number[]>();
  /** Index of current note in the per-note arrays — written by runPass. */
  currentNoteIdx = -1;
  /** Per-note totals for each phase. byPhase[phase][noteIdx] = sum-of-records-for-that-note. */
  perNote = new Map<Phase, number[]>();

  constructor(noteCount: number) {
    for (const p of PHASE_NAMES) {
      this.perNote.set(p, new Array(noteCount).fill(0));
    }
  }

  record(phase: PhaseName | Phase, ms: number): void {
    if (this.currentNoteIdx < 0) return;
    const arr = this.perNote.get(phase as Phase);
    if (arr) arr[this.currentNoteIdx] += ms;
  }
}

interface PhaseStats {
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
  sum: number;
}

function summarize(values: number[]): PhaseStats {
  if (values.length === 0) {
    return { mean: 0, p50: 0, p95: 0, p99: 0, max: 0, sum: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((s, v) => s + v, 0);
  const pct = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  return {
    mean: Math.round((sum / sorted.length) * 100) / 100,
    p50: Math.round(pct(0.5) * 100) / 100,
    p95: Math.round(pct(0.95) * 100) / 100,
    p99: Math.round(pct(0.99) * 100) / 100,
    max: Math.round(sorted[sorted.length - 1] * 100) / 100,
    sum: Math.round(sum * 100) / 100,
  };
}

interface PassResult {
  totalMs: number;
  perNote: PhaseStats;
  perPhase: Record<Phase, PhaseStats>;
}

function buildPassResult(rec: TimingsRecorder, totalMs: number): PassResult {
  const perPhase = {} as Record<Phase, PhaseStats>;
  const noteTotals: number[] = [];
  for (const p of PHASE_NAMES) {
    const arr = rec.perNote.get(p)!;
    perPhase[p] = summarize(arr);
    if (noteTotals.length === 0) {
      for (let i = 0; i < arr.length; i++) noteTotals.push(arr[i]);
    } else {
      for (let i = 0; i < arr.length; i++) noteTotals[i] += arr[i];
    }
  }
  return {
    totalMs: Math.round(totalMs),
    perNote: summarize(noteTotals),
    perPhase,
  };
}

// ── Single pass over the vault ──────────────────────────────────

interface PassDeps {
  cfg: BenchConfig;
  vault: VaultFile[];
  noteTitles: string[];
  vaultContext: VaultContext;
  provider: EmbeddingProvider;
  sae: SparseAutoencoder;
  featureLabels: SAEFeatureLabels;
  resolver: EmbeddingResolver;
  recorder: TimingsRecorder;
}

async function runPass(deps: PassDeps): Promise<PassResult> {
  const { cfg, vault, noteTitles, vaultContext, sae, featureLabels, resolver, recorder } = deps;
  const aliasResolver = new AliasResolver();
  const extractor = cfg.extractor === "span"
    ? new SpanExtractor({ maxNgramSize: cfg.maxNgram })
    : new YakeLite({ maxNgramSize: cfg.maxNgram });

  const t0 = performance.now();

  for (let i = 0; i < vault.length; i++) {
    recorder.currentNoteIdx = i;
    const file = vault[i];
    const sourcePath = file.basename + ".md";

    // 1. extract
    const tExtract = performance.now();
    const phrases = await extractor.extract(file.content, vaultContext);
    recorder.record("extract", performance.now() - tExtract);

    if (phrases.length === 0) continue;

    // 2. lcs
    const tLcs = performance.now();
    const detEdges = await aliasResolver.resolve(phrases, noteTitles, sourcePath);
    recorder.record("lcs", performance.now() - tLcs);

    // Strong-LCS skip + MAX_EMBED cap (mirrors main.ts:144-164)
    let phrasesForEmbedding: ExtractedPhrase[] = phrases;
    if (cfg.strongLcs < 1.0) {
      const stronglyMatched = new Set<string>();
      for (const e of detEdges) {
        if (e.similarity >= cfg.strongLcs) stronglyMatched.add(e.phrase.phrase);
      }
      if (stronglyMatched.size > 0) {
        phrasesForEmbedding = phrases.filter((p) => !stronglyMatched.has(p.phrase));
      }
    }
    if (phrasesForEmbedding.length > cfg.maxEmbed) {
      phrasesForEmbedding = phrasesForEmbedding
        .slice()
        .sort((a, b) => a.phrase.length - b.phrase.length)
        .slice(0, cfg.maxEmbed);
    }

    // 3-5. dense (embedTitles + embedPhrases + denseMatch are all internal to resolve())
    let denseEdges: CandidateEdge[] = [];
    if (cfg.pipeline !== "lcs-only" && phrasesForEmbedding.length > 0) {
      denseEdges = await resolver.resolve(phrasesForEmbedding, noteTitles, sourcePath, "normal");
    }

    // 6-7. sparse (encode + match, both internal to resolveBySparseFeatures())
    let sparseEdges: CandidateEdge[] = [];
    if (cfg.pipeline === "lcs+dense+sparse" && phrasesForEmbedding.length > 0) {
      sparseEdges = await resolver.resolveBySparseFeatures(
        phrasesForEmbedding,
        noteTitles,
        sourcePath,
        featureLabels,
        { similarityThreshold: 0.75 },
      );
    }

    // 8. merge
    const tMerge = performance.now();
    mergeByTarget([...detEdges, ...denseEdges, ...sparseEdges]);
    recorder.record("merge", performance.now() - tMerge);
  }

  const totalMs = performance.now() - t0;
  recorder.currentNoteIdx = -1;
  return buildPassResult(recorder, totalMs);
}

// ── Output ──────────────────────────────────────────────────────

interface ConfigResult {
  label: string;
  knobs: Omit<BenchConfig, "label">;
  cold: PassResult;
  warm?: PassResult;
}

interface BenchOutput {
  schemaVersion: 1;
  generatedAt: string;
  host: { platform: string; arch: string; node: string; cpu: string };
  model: string;
  device: string;
  dtype: string;
  vault: { path: string; noteCount: number };
  startup: { vaultLoadMs: number; modelLoadMs: number };
  configurations: ConfigResult[];
}

function writeJson(out: BenchOutput, dest: string): void {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, JSON.stringify(out, null, 2));
}

function writeMarkdown(out: BenchOutput, dest: string): void {
  const lines: string[] = [];
  lines.push("# Runtime Benchmark — Nexus Pipeline\n");
  lines.push(`Generated: ${out.generatedAt}  `);
  lines.push(`Host: ${out.host.platform}/${out.host.arch}, Node ${out.host.node}, ${out.host.cpu}  `);
  lines.push(`Model: \`${out.model}\` device=\`${out.device}\` dtype=\`${out.dtype}\`  `);
  lines.push(`Vault: \`${out.vault.path}\` (${out.vault.noteCount} notes)  `);
  lines.push(`Startup: vault load ${out.startup.vaultLoadMs}ms, model load ${out.startup.modelLoadMs}ms\n`);

  // Sort by warm p50 (fallback to cold) for the headline table
  const sorted = [...out.configurations].sort(
    (a, b) => (a.warm?.perNote.p50 ?? a.cold.perNote.p50) - (b.warm?.perNote.p50 ?? b.cold.perNote.p50),
  );

  lines.push("## Headline (sorted by warm p50, ascending)\n");
  lines.push("| Configuration | Cold p50 | Warm p50 | Warm p95 | Title embed (cold sum) | Phrase embed (warm sum) |");
  lines.push("|---|---:|---:|---:|---:|---:|");
  for (const r of sorted) {
    const coldP50 = `${r.cold.perNote.p50}ms`;
    const warmP50 = r.warm ? `${r.warm.perNote.p50}ms` : "—";
    const warmP95 = r.warm ? `${r.warm.perNote.p95}ms` : "—";
    const titleEmbedCold = `${Math.round(r.cold.perPhase.titleEmbed.sum)}ms`;
    const phraseEmbedWarm = r.warm ? `${Math.round(r.warm.perPhase.phraseEmbed.sum)}ms` : "—";
    lines.push(`| ${r.label} | ${coldP50} | ${warmP50} | ${warmP95} | ${titleEmbedCold} | ${phraseEmbedWarm} |`);
  }
  lines.push("");

  lines.push("## Per-phase breakdown (warm pass, p50 ms)\n");
  lines.push("| Configuration | extract | lcs | titleEmbed | phraseEmbed | denseMatch | sparseEncode | sparseMatch | merge |");
  lines.push("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const r of sorted) {
    const p = r.warm?.perPhase ?? r.cold.perPhase;
    lines.push(
      `| ${r.label} | ${p.extract.p50} | ${p.lcs.p50} | ${p.titleEmbed.p50} | ${p.phraseEmbed.p50} | ${p.denseMatch.p50} | ${p.sparseEncode.p50} | ${p.sparseMatch.p50} | ${p.merge.p50} |`,
    );
  }
  lines.push("");

  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, lines.join("\n"));
}

// ── Main ────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseCli(process.argv);
  console.log(`Benchmark: sweep=${args.sweep} sample=${args.sample || "full"} device=${args.device} dtype=${args.dtype} warm=${args.warm}`);

  // Load vault
  const tVault = performance.now();
  let vault = loadVault(args.vault);
  const vaultLoadMs = Math.round(performance.now() - tVault);
  if (args.sample > 0 && args.sample < vault.length) {
    vault = vault.slice(0, args.sample);
  }
  console.log(`Loaded ${vault.length} notes in ${vaultLoadMs}ms`);

  const noteTitles = vault.map((f) => f.basename);
  const vaultContext: VaultContext = { noteTitles };

  // Load embedding model + warmup (excluded from per-note stats)
  process.env.NEXUS_EMBED_DEVICE = args.device;
  const provider = createNodeEmbeddingProvider(GEMMA_MODEL_ID, GEMMA_DIMS, { device: args.device });
  const tModel = performance.now();
  await provider.embedBatch(["warmup"]);
  const modelLoadMs = Math.round(performance.now() - tModel);
  console.log(`Model warmup ${modelLoadMs}ms`);

  // ── SAE sweep across all configs in assets/runs/ ──────────────────────────
  if (args.saeSweep) {
    const runsDir = path.join(__dirname, "..", "assets", "runs");
    const runDirs = fs.readdirSync(runsDir)
      .filter(n => n.match(/^k\d+-\d+x$/) && fs.existsSync(path.join(runsDir, n, "sae-weights.bin")))
      .sort();
    console.log(`\nSAE sweep: ${runDirs.length} configs\n`);
    const sweepResults: Array<{ config: string; k: number; dHidden: number; coldMs: number; warmP50Ms: number; warmP95Ms: number }> = [];
    const baseConfig = enumerateConfigs(args).find(c => c.pipeline === "lcs+dense+sparse" && c.maxEmbed === MAX_EMBED_DEFAULT)!;
    for (const runName of runDirs) {
      const weightsPath = path.join(runsDir, runName, "sae-weights.bin");
      const labelPath   = path.join(runsDir, runName, "sae-feature-labels.json");
      const saeBin = fs.readFileSync(weightsPath);
      const runSae = SparseAutoencoder.deserialize(saeBin);
      const runLabels = fs.existsSync(labelPath)
        ? SAEFeatureLabels.fromJSON(JSON.parse(fs.readFileSync(labelPath, "utf-8")), runSae.dHidden)
        : new SAEFeatureLabels(runSae.dHidden);
      console.log(`[${runName}] k=${runSae.k} dHidden=${runSae.dHidden} labels=${runLabels.liveCount}`);
      const recorder = new TimingsRecorder(vault.length);
      const resolver = new EmbeddingResolver({
        embeddingProvider: provider,
        sae: runSae,
        featureLabels: runLabels,
        similarityThreshold: baseConfig.similarityThreshold,
        phaseTimings: recorder,
      });
      const { coldMs, warmMs } = await runOneConfig({ cfg: baseConfig, vault, noteTitles, vaultContext, sae: runSae, featureLabels: runLabels, resolver, recorder });
      const p50 = warmMs.sort((a, b) => a - b)[Math.floor(warmMs.length * 0.5)] ?? 0;
      const p95 = warmMs[Math.floor(warmMs.length * 0.95)] ?? 0;
      sweepResults.push({ config: runName, k: runSae.k, dHidden: runSae.dHidden, coldMs, warmP50Ms: p50, warmP95Ms: p95 });
      console.log(`  cold=${coldMs.toFixed(1)}ms  warm p50=${p50.toFixed(2)}ms  p95=${p95.toFixed(2)}ms`);
    }
    console.log("\n=== SAE sweep summary ===");
    console.log(`${"config".padEnd(14)} ${"k".padStart(4)}  ${"d_hidden".padStart(8)}  ${"cold".padStart(8)}  ${"warm p50".padStart(9)}  ${"warm p95".padStart(9)}`);
    for (const r of sweepResults) {
      console.log(`${r.config.padEnd(14)} ${String(r.k).padStart(4)}  ${String(r.dHidden).padStart(8)}  ${r.coldMs.toFixed(1).padStart(7)}ms  ${r.warmP50Ms.toFixed(2).padStart(8)}ms  ${r.warmP95Ms.toFixed(2).padStart(8)}ms`);
    }
    return;
  }

  // Load SAE + labels (cheap, ~100ms)
  const saeBin = fs.readFileSync(
    args.sae ? args.sae : path.join(__dirname, "..", "assets", "sae-weights-v2.bin")
  );
  const sae = SparseAutoencoder.deserialize(saeBin);
  const featureLabels = new SAEFeatureLabels(sae.dHidden);
  console.log(`SAE: dHidden=${sae.dHidden} k=${sae.k}, labels live=${featureLabels.liveCount}`);

  // Run configs
  const configurations: ConfigResult[] = [];
  const configs = enumerateConfigs(args);
  console.log(`\nRunning ${configs.length} configurations × ${args.warm ? 2 : 1} pass(es)\n`);

  for (let ci = 0; ci < configs.length; ci++) {
    const cfg = configs[ci];
    process.stderr.write(`\n[${ci + 1}/${configs.length}] ${cfg.label}\n`);

    // Fresh resolver per config so the cache state is fair (cold starts cold).
    // For lcs+dense (no sparse), construct resolver WITHOUT sae — otherwise
    // resolveWithSparse() still computes sparse encodings internally for
    // the dense-edge explanation feature, contaminating the dense-only number.
    const recorder = new TimingsRecorder(vault.length);
    const resolver = new EmbeddingResolver({
      embeddingProvider: provider,
      sae: cfg.pipeline === "lcs+dense+sparse" ? sae : undefined,
      featureLabels: cfg.pipeline === "lcs+dense+sparse" ? featureLabels : undefined,
      similarityThreshold: cfg.similarityThreshold,
      phaseTimings: recorder,
    });

    const cold = await runPass({
      cfg, vault, noteTitles, vaultContext, provider, sae, featureLabels, resolver, recorder,
    });
    process.stderr.write(`  cold:  total=${cold.totalMs}ms p50=${cold.perNote.p50}ms p95=${cold.perNote.p95}ms\n`);

    let warm: PassResult | undefined;
    if (args.warm) {
      const recorderWarm = new TimingsRecorder(vault.length);
      // Reuse the same resolver (caches now warm) but swap in a new recorder.
      // Bypass the (private) phaseTimings field via a fresh resolver wrapping
      // the same caches: easier to just reach in.
      (resolver as unknown as { phaseTimings: TimingsRecorder }).phaseTimings = recorderWarm;
      warm = await runPass({
        cfg, vault, noteTitles, vaultContext, provider, sae, featureLabels, resolver, recorder: recorderWarm,
      });
      process.stderr.write(`  warm:  total=${warm.totalMs}ms p50=${warm.perNote.p50}ms p95=${warm.perNote.p95}ms\n`);
    }

    configurations.push({ label: cfg.label, knobs: cfg, cold, warm });
  }

  // Write output
  const benchOut: BenchOutput = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    host: {
      platform: process.platform,
      arch: process.arch,
      node: process.version,
      cpu: os.cpus()[0]?.model ?? "unknown",
    },
    model: GEMMA_MODEL_ID,
    device: args.device,
    dtype: args.dtype,
    vault: { path: args.vault, noteCount: vault.length },
    startup: { vaultLoadMs, modelLoadMs },
    configurations,
  };

  writeJson(benchOut, args.out);
  writeMarkdown(benchOut, args.md);
  console.log(`\nWrote ${args.out}`);
  console.log(`Wrote ${args.md}`);

  // Console summary
  console.log("\nHeadline (sorted by warm p50):");
  const rows = configurations
    .map((c) => ({
      config: c.label.length > 70 ? c.label.slice(0, 67) + "..." : c.label,
      coldP50: c.cold.perNote.p50,
      warmP50: c.warm?.perNote.p50 ?? "-",
      warmP95: c.warm?.perNote.p95 ?? "-",
    }))
    .sort((a, b) => {
      const av = typeof a.warmP50 === "number" ? a.warmP50 : a.coldP50;
      const bv = typeof b.warmP50 === "number" ? b.warmP50 : b.coldP50;
      return av - bv;
    });
  console.table(rows);

  // Sanity: warn on monotonicity violations and phase-sum drift
  warnOnSanityIssues(configurations);

  await provider.dispose?.();
}

function warnOnSanityIssues(configs: ConfigResult[]): void {
  // MAX_EMBED monotonicity: holding everything else at baseline, runtime should
  // be non-decreasing as MAX_EMBED grows.
  const maxEmbedSeries = configs
    .filter((c) =>
      c.knobs.extractor === "span" &&
      c.knobs.maxNgram === 10 &&
      c.knobs.pipeline === "lcs+dense+sparse" &&
      c.knobs.strongLcs === STRONG_LCS_DEFAULT,
    )
    .sort((a, b) => a.knobs.maxEmbed - b.knobs.maxEmbed);
  let prev = -Infinity;
  for (const c of maxEmbedSeries) {
    const t = c.warm?.perNote.p50 ?? c.cold.perNote.p50;
    if (t + 5 < prev) {
      console.warn(`⚠ Monotonicity violation: ${c.label} (${t}ms) < previous in MAX_EMBED series (${prev}ms)`);
    }
    prev = Math.max(prev, t);
  }

  // Phase-sum sanity: sum of phase sums should be close to totalMs (within 5%).
  for (const c of configs) {
    for (const passLabel of ["cold", "warm"] as const) {
      const pass = c[passLabel];
      if (!pass) continue;
      const phaseSum = Object.values(pass.perPhase).reduce((s, p) => s + p.sum, 0);
      const drift = Math.abs(phaseSum - pass.totalMs) / Math.max(pass.totalMs, 1);
      if (drift > 0.05) {
        console.warn(
          `⚠ Phase-sum drift > 5% for ${c.label} (${passLabel}): phaseSum=${Math.round(phaseSum)}ms total=${pass.totalMs}ms`,
        );
      }
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
