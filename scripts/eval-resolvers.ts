#!/usr/bin/env npx ts-node
/**
 * Standalone evaluation script for extractor × resolver combinations.
 *
 * Runs outside Jest to avoid ONNX/VM sandbox incompatibility.
 *
 * Usage:
 *   npm run eval                                    # all 6 combos
 *   npm run eval -- --extractor=span                # SpanExtractor × all resolvers
 *   npm run eval -- --resolver=lcs                  # all extractors × LCS
 *   npm run eval -- --extractor=yake --resolver=lcs # single combo
 *
 * Extractor values: span, yake, all (default: all)
 * Resolver values:  lcs, arctic, gemma, all (default: all)
 */

import * as fs from "fs";
import * as path from "path";
import { SpanExtractor } from "../src/keyphrase/span-extractor";
import { YakeLite } from "../src/keyphrase/yake-lite";
import { AliasResolver } from "../src/resolver/index";
import { EmbeddingResolver } from "../src/resolver/embedding-resolver";
import { EmbeddingProvider } from "../src/resolver/embedding-provider";
import { cosineSimilarity } from "../src/resolver/cosine";
import { normalize } from "../src/resolver/normalization";
import { ExtractedPhrase, CandidateEdge, VaultContext } from "../src/types";

// ── Types ───────────────────────────────────────────────────────

interface VaultFile {
  path: string;
  basename: string;
  content: string;
}

interface PerFileResult {
  file: string;
  groundTruth: Set<string>;
  predicted: Set<string>;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
  tpList: string[];
  fpList: string[];
  fnList: string[];
}

interface AggregateResult {
  name: string;
  filesEvaluated: number;
  totalTP: number;
  totalFP: number;
  totalFN: number;
  precision: number;
  recall: number;
  f1: number;
  perFile: PerFileResult[];
}

// ── Vault loading ───────────────────────────────────────────────

function loadVault(vaultDir: string): VaultFile[] {
  const files: VaultFile[] = [];
  function walk(dir: string): void {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        files.push({
          path: fullPath,
          basename: entry.name.replace(/\.md$/, ""),
          content: fs.readFileSync(fullPath, "utf-8"),
        });
      }
    }
  }
  walk(vaultDir);
  return files;
}

// ── Ground truth extraction ─────────────────────────────────────

function extractGroundTruth(content: string, sourceBasename: string): Set<string> {
  const targets = new Set<string>();
  const normalizedSource = normalize(sourceBasename);
  let text = content.replace(/^---\n[\s\S]*?\n---\n?/, "");
  text = text.replace(/```[\s\S]*?```/g, "");
  const footerIdx = text.indexOf("%% wiki footer");
  if (footerIdx !== -1) text = text.slice(0, footerIdx);

  const re = /(!?)\[\[([^\]]+)\]\]/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    if (match[1] === "!") continue;
    const inner = match[2];
    const pipeIdx = inner.indexOf("|");
    const targetPart = pipeIdx !== -1 ? inner.slice(0, pipeIdx) : inner;
    const hashIdx = targetPart.indexOf("#");
    const title = hashIdx !== -1 ? targetPart.slice(0, hashIdx) : targetPart;
    if (!title.trim()) continue;
    const normalizedTarget = normalize(title);
    if (normalizedTarget === normalizedSource) continue;
    targets.add(normalizedTarget);
  }
  return targets;
}

// ── Vault preparation (mirrors strip-unlinked-titles.ts) ────────

/** Split content into frontmatter, body, and wiki footer (never touch those zones). */
function splitProtectedZones(content: string): { frontmatter: string; body: string; footer: string } {
  let frontmatter = "";
  let body = content;
  if (content.startsWith("---")) {
    const endIdx = content.indexOf("\n---", 3);
    if (endIdx !== -1) {
      const fmEnd = endIdx + 4;
      frontmatter = content.slice(0, fmEnd);
      body = content.slice(fmEnd);
    }
  }
  let footer = "";
  const footerIdx = body.indexOf("%% wiki footer");
  if (footerIdx !== -1) {
    footer = body.slice(footerIdx);
    body = body.slice(0, footerIdx);
  }
  return { frontmatter, body, footer };
}

/**
 * Pass 1: remove aliased wikilinks [[target|display]] entirely.
 * These cause FNs because the display text doesn't match the target title.
 */
function removeAliasedWikilinks(content: string): string {
  const { frontmatter, body, footer } = splitProtectedZones(content);
  const cleaned = body.replace(/(?<!!)\[\[([^\]|]+)\|([^\]]+)\]\]/g, "");
  return frontmatter + cleaned + footer;
}

/**
 * Pass 2: strip plain-text occurrences of vault titles (longest-first, word-boundary).
 * Protects existing wikilinks so they aren't double-processed.
 */
function stripUnlinkedMentions(content: string, allTitles: string[], ownTitle: string): string {
  const { frontmatter, body, footer } = splitProtectedZones(content);
  const titlesToStrip = allTitles
    .filter((t) => t !== ownTitle)
    .sort((a, b) => b.length - a.length);

  const links: string[] = [];
  let work = body.replace(/\[\[[^\]]*\]\]/g, (match) => {
    links.push(match);
    return `\x00LINK_${links.length - 1}\x00`;
  });

  for (const title of titlesToStrip) {
    const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(?<![a-zA-Z0-9])${escaped}(?![a-zA-Z0-9])`, "gi");
    work = work.replace(re, "");
  }

  work = work.replace(/\x00LINK_(\d+)\x00/g, (_, idx) => links[Number(idx)]);
  return frontmatter + work + footer;
}

// ── Wikilink stripping ──────────────────────────────────────────

function stripWikilinksForEval(content: string): string {
  let text = content.replace(/!\[\[[^\]]*\]\]/g, "");
  text = text.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2");
  text = text.replace(/\[\[([^\]#]+)#[^\]]*\]\]/g, "$1");
  text = text.replace(/\[\[([^\]]+)\]\]/g, "$1");
  return text;
}

// ── Evaluation engine ───────────────────────────────────────────

type ResolverFn = (
  phrases: ExtractedPhrase[],
  noteTitles: string[],
  sourcePath: string,
) => CandidateEdge[] | Promise<CandidateEdge[]>;

async function evaluate(
  name: string,
  extractFn: (content: string, vaultContext: VaultContext) => ExtractedPhrase[],
  files: VaultFile[],
  noteTitles: string[],
  resolveFn: ResolverFn,
  vaultContext: VaultContext,
): Promise<AggregateResult> {
  const normalizedTitles = new Set(noteTitles.map(normalize));
  const perFile: PerFileResult[] = [];

  for (const file of files) {
    // Pass 1: remove aliased wikilinks before ground truth extraction
    const pass1Content = removeAliasedWikilinks(file.content);
    const rawGroundTruth = extractGroundTruth(pass1Content, file.basename);
    const groundTruth = new Set<string>();
    for (const target of rawGroundTruth) {
      if (normalizedTitles.has(target)) groundTruth.add(target);
    }
    if (groundTruth.size === 0) continue;

    // Pass 2: strip unlinked title mentions, then convert remaining wikilinks to plain text
    const pass2Content = stripUnlinkedMentions(pass1Content, noteTitles, file.basename);
    const strippedContent = stripWikilinksForEval(pass2Content);
    const phrases = extractFn(strippedContent, vaultContext);
    const candidates = await resolveFn(phrases, noteTitles, file.basename + ".md");

    const predicted = new Set<string>(candidates.map((c) => normalize(c.targetPath)));
    const tpList: string[] = [];
    const fpList: string[] = [];
    const fnList: string[] = [];

    for (const target of predicted) {
      if (groundTruth.has(target)) tpList.push(target);
      else fpList.push(target);
    }
    for (const target of groundTruth) {
      if (!predicted.has(target)) fnList.push(target);
    }

    const tp = tpList.length;
    const fp = fpList.length;
    const fn = fnList.length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    perFile.push({
      file: file.basename, groundTruth, predicted,
      tp, fp, fn, precision, recall, f1,
      tpList, fpList, fnList,
    });
  }

  const totalTP = perFile.reduce((s, r) => s + r.tp, 0);
  const totalFP = perFile.reduce((s, r) => s + r.fp, 0);
  const totalFN = perFile.reduce((s, r) => s + r.fn, 0);
  const precision = totalTP + totalFP > 0 ? totalTP / (totalTP + totalFP) : 0;
  const recall = totalTP + totalFN > 0 ? totalTP / (totalTP + totalFN) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { name, filesEvaluated: perFile.length, totalTP, totalFP, totalFN, precision, recall, f1, perFile };
}

// ── Node.js Embedding Providers ─────────────────────────────────

/**
 * Pipeline-based provider for models that work with the `feature-extraction` pipeline
 * (e.g., Snowflake arctic-embed).
 */
class PipelineEmbeddingProvider implements EmbeddingProvider {
  readonly dims: number;
  private modelId: string;
  private extractor: any = null;

  constructor(modelId: string, dims: number) {
    this.modelId = modelId;
    this.dims = dims;
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (!this.extractor) {
      const { pipeline, env } = require("@huggingface/transformers");
      env.allowLocalModels = true;
      env.useBrowserCache = false;
      this.extractor = await pipeline(
        "feature-extraction",
        this.modelId,
        { quantized: true },
      );
    }
    const results: Float32Array[] = [];
    for (const text of texts) {
      const output = await this.extractor(text, { pooling: "mean", normalize: true });
      results.push(new Float32Array(output.data));
    }
    return results;
  }

  async dispose(): Promise<void> {
    this.extractor = null;
  }
}

/**
 * AutoModel-based provider for EmbeddingGemma (uses AutoModel + AutoTokenizer
 * instead of the pipeline API).
 */
class GemmaEmbeddingProvider implements EmbeddingProvider {
  readonly dims: number;
  private modelId: string;
  private dtype: string;
  private model: any = null;
  private tokenizer: any = null;

  constructor(modelId: string, dims: number, dtype: string = "q4") {
    this.modelId = modelId;
    this.dims = dims;
    this.dtype = dtype;
  }

  private async ensureModel(): Promise<void> {
    if (this.model) return;
    const { AutoModel, AutoTokenizer, env } = require("@huggingface/transformers");
    env.allowLocalModels = true;
    env.useBrowserCache = false;
    console.log(`  Loading ${this.modelId} (${this.dtype})...`);
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);
    this.model = await AutoModel.from_pretrained(this.modelId, {
      dtype: this.dtype,
    });
    console.log(`  Model loaded.`);
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    await this.ensureModel();
    const results: Float32Array[] = [];
    for (const text of texts) {
      const inputs = await this.tokenizer(text, { padding: true, truncation: true });
      const output = await this.model(inputs);
      // EmbeddingGemma outputs last_hidden_state; take mean pooling
      const embeddings = output.last_hidden_state;
      const dims = embeddings.dims;  // [batch, seq_len, hidden_dim]
      const data = embeddings.data as Float32Array;
      const seqLen = dims[1];
      const hiddenDim = dims[2];
      // Mean pooling over sequence length
      const pooled = new Float32Array(hiddenDim);
      for (let s = 0; s < seqLen; s++) {
        for (let d = 0; d < hiddenDim; d++) {
          pooled[d] += data[s * hiddenDim + d];
        }
      }
      // Normalize
      let norm = 0;
      for (let d = 0; d < hiddenDim; d++) {
        pooled[d] /= seqLen;
        norm += pooled[d] * pooled[d];
      }
      norm = Math.sqrt(norm);
      if (norm > 0) {
        for (let d = 0; d < hiddenDim; d++) pooled[d] /= norm;
      }
      results.push(pooled);
    }
    return results;
  }

  async dispose(): Promise<void> {
    this.model = null;
    this.tokenizer = null;
  }
}

// ── Reporting ───────────────────────────────────────────────────

function printResult(result: AggregateResult): void {
  console.table({
    Resolver: result.name,
    Files: result.filesEvaluated,
    TP: result.totalTP,
    FP: result.totalFP,
    FN: result.totalFN,
    P: result.precision.toFixed(4),
    R: result.recall.toFixed(4),
    F1: result.f1.toFixed(4),
  });
}

function printWorstPrecision(result: AggregateResult, n = 10): void {
  const worst = [...result.perFile]
    .filter((r) => r.predicted.size >= 2)
    .sort((a, b) => a.precision - b.precision)
    .slice(0, n);

  console.log(`\n=== ${result.name}: Worst Precision ===`);
  console.table(
    worst.map((r) => ({
      File: r.file,
      Predicted: r.predicted.size,
      TP: r.tp,
      FP: r.fp,
      Precision: r.precision.toFixed(4),
      "False Positives": r.fpList.join(", "),
    }))
  );
}

function printWorstRecall(result: AggregateResult, n = 10): void {
  const worst = [...result.perFile]
    .filter((r) => r.groundTruth.size >= 2)
    .sort((a, b) => a.recall - b.recall)
    .slice(0, n);

  console.log(`\n=== ${result.name}: Worst Recall ===`);
  console.table(
    worst.map((r) => ({
      File: r.file,
      "Ground Truth": r.groundTruth.size,
      TP: r.tp,
      FN: r.fn,
      Recall: r.recall.toFixed(4),
      Missed: r.fnList.join(", "),
    }))
  );
}

// ── Main ────────────────────────────────────────────────────────

// ── CLI argument parsing ────────────────────────────────────────

function parseArgs(): { extractor: string; resolver: string } {
  let extractor = "all";
  let resolver = "all";
  for (const arg of process.argv.slice(2)) {
    const match = arg.match(/^--(\w+)=(\w+)$/);
    if (match) {
      if (match[1] === "extractor") extractor = match[2].toLowerCase();
      if (match[1] === "resolver") resolver = match[2].toLowerCase();
    }
  }
  return { extractor, resolver };
}

async function main(): Promise<void> {
  const args = parseArgs();

  const VAULT_DIR = path.resolve(__dirname, "../senior-thesis-vault");
  if (!fs.existsSync(VAULT_DIR)) {
    console.error(`Vault not found at ${VAULT_DIR}`);
    process.exit(1);
  }

  // Load vault
  const files = loadVault(VAULT_DIR);
  const noteTitles = files.map((f) => f.basename);
  const vaultContext: VaultContext = { noteTitles };

  // ── Extractors ──
  const spanExtractor = new SpanExtractor();
  const spanExtractFn = (content: string, ctx: VaultContext) => spanExtractor.extract(content, ctx);

  const yakeLite = new YakeLite();
  const yakeExtractFn = (content: string, ctx: VaultContext) => yakeLite.extract(content, ctx);

  const allExtractors = [
    { key: "span", name: "SpanExtractor", fn: spanExtractFn },
    { key: "yake", name: "YakeLite", fn: yakeExtractFn },
  ];

  const extractors = args.extractor === "all"
    ? allExtractors
    : allExtractors.filter((e) => e.key === args.extractor);

  if (extractors.length === 0) {
    console.error(`Unknown extractor: "${args.extractor}". Use: span, yake, all`);
    process.exit(1);
  }

  // ── Resolvers (only instantiate what's needed) ──
  const deterministicResolver = new AliasResolver();

  let arcticProvider: PipelineEmbeddingProvider | null = null;
  let arcticResolver: EmbeddingResolver | null = null;
  let gemmaProvider: GemmaEmbeddingProvider | null = null;
  let gemmaResolver: EmbeddingResolver | null = null;

  if (args.resolver === "all" || args.resolver === "arctic") {
    arcticProvider = new PipelineEmbeddingProvider("Snowflake/snowflake-arctic-embed-xs", 384);
    arcticResolver = new EmbeddingResolver({
      embeddingProvider: arcticProvider,
      similarityThreshold: 0.95,
    });
  }

  if (args.resolver === "all" || args.resolver === "gemma") {
    gemmaProvider = new GemmaEmbeddingProvider("onnx-community/embeddinggemma-300m-ONNX", 768, "q4");
    gemmaResolver = new EmbeddingResolver({
      embeddingProvider: gemmaProvider,
      similarityThreshold: 0.95,
    });
  }

  const allResolvers = [
    { key: "lcs", name: "LCS", fn: (p: ExtractedPhrase[], t: string[], s: string) => deterministicResolver.resolve(p, t, s) },
    ...(arcticResolver ? [{ key: "arctic", name: "Arctic-xs (384d)", fn: (p: ExtractedPhrase[], t: string[], s: string) => arcticResolver!.resolve(p, t, s) }] : []),
    ...(gemmaResolver ? [{ key: "gemma", name: "Gemma-300m q4 (768d)", fn: (p: ExtractedPhrase[], t: string[], s: string) => gemmaResolver!.resolve(p, t, s) }] : []),
  ];

  const resolvers = args.resolver === "all"
    ? allResolvers
    : allResolvers.filter((r) => r.key === args.resolver);

  if (resolvers.length === 0) {
    console.error(`Unknown resolver: "${args.resolver}". Use: lcs, arctic, gemma, all`);
    process.exit(1);
  }

  console.log(`Running: extractors=[${extractors.map((e) => e.name).join(", ")}] × resolvers=[${resolvers.map((r) => r.name).join(", ")}]`);

  // ── Run extractor × resolver combinations ──
  const allResults: Array<{ extractor: string; resolver: string; result: AggregateResult }> = [];

  for (const ext of extractors) {
    for (const res of resolvers) {
      const label = `${ext.name} + ${res.name}`;
      console.log("\n" + "=".repeat(60));
      console.log(label);
      console.log("=".repeat(60));

      const result = await evaluate(label, ext.fn, files, noteTitles, res.fn, vaultContext);
      printResult(result);
      printWorstPrecision(result);
      printWorstRecall(result);
      allResults.push({ extractor: ext.name, resolver: res.name, result });
    }
  }

  // ── Cleanup embedding providers ──
  if (arcticProvider) await arcticProvider.dispose();
  if (gemmaProvider) await gemmaProvider.dispose();

  // ── Side-by-side comparison (only when multiple combos ran) ──
  if (allResults.length > 1) {
    console.log("\n" + "=".repeat(60));
    console.log("SIDE-BY-SIDE COMPARISON");
    console.log("=".repeat(60));

    console.table(allResults.map(({ extractor, resolver, result: r }) => ({
      Extractor: extractor,
      Resolver: resolver,
      TP: r.totalTP, FP: r.totalFP, FN: r.totalFN,
      P: r.precision.toFixed(4),
      R: r.recall.toFixed(4),
      F1: r.f1.toFixed(4),
    })));
  }

  // ── Per-file deltas: best SpanExtractor vs best YakeLite (by F1) ──
  const spanBest = allResults.filter((r) => r.extractor === "SpanExtractor")
    .sort((a, b) => b.result.f1 - a.result.f1)[0];
  const yakeBest = allResults.filter((r) => r.extractor === "YakeLite")
    .sort((a, b) => b.result.f1 - a.result.f1)[0];

  if (spanBest && yakeBest) {
    console.log(`\n=== Per-file deltas: ${spanBest.extractor}+${spanBest.resolver} vs ${yakeBest.extractor}+${yakeBest.resolver} ===`);
    const deltas: Array<{ file: string; spanF1: string; yakeF1: string; delta: string }> = [];
    for (const spanFile of spanBest.result.perFile) {
      const yakeFile = yakeBest.result.perFile.find((r) => r.file === spanFile.file);
      if (!yakeFile) continue;
      const d = yakeFile.f1 - spanFile.f1;
      if (Math.abs(d) > 0.01) {
        deltas.push({
          file: spanFile.file,
          spanF1: spanFile.f1.toFixed(3),
          yakeF1: yakeFile.f1.toFixed(3),
          delta: (d > 0 ? "+" : "") + d.toFixed(3),
        });
      }
    }
    deltas.sort((a, b) => parseFloat(b.delta) - parseFloat(a.delta));
    console.log("\nYakeLite wins:");
    console.table(deltas.filter((d) => parseFloat(d.delta) > 0).slice(0, 15));
    console.log("\nSpanExtractor wins:");
    console.table(deltas.filter((d) => parseFloat(d.delta) < 0).slice(0, 15));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
