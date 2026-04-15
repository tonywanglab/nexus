#!/usr/bin/env npx ts-node
/**
 * Standalone evaluation script for both deterministic (LCS) and stochastic (embedding) resolvers.
 *
 * Runs outside Jest to avoid ONNX/VM sandbox incompatibility.
 *
 * Usage:
 *   npx ts-node scripts/eval-resolvers.ts
 */

import * as fs from "fs";
import * as path from "path";
import { SpanExtractor } from "../src/keyphrase/span-extractor";
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
    const rawGroundTruth = extractGroundTruth(file.content, file.basename);
    const groundTruth = new Set<string>();
    for (const target of rawGroundTruth) {
      if (normalizedTitles.has(target)) groundTruth.add(target);
    }
    if (groundTruth.size === 0) continue;

    const strippedContent = stripWikilinksForEval(file.content);
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

async function main(): Promise<void> {
  const VAULT_DIR = path.resolve(__dirname, "../senior-thesis-vault");
  if (!fs.existsSync(VAULT_DIR)) {
    console.error(`Vault not found at ${VAULT_DIR}`);
    process.exit(1);
  }

  // Load vault
  const files = loadVault(VAULT_DIR);
  const noteTitles = files.map((f) => f.basename);
  const vaultContext: VaultContext = { noteTitles };

  const spanExtractor = new SpanExtractor();
  const extractFn = (content: string, ctx: VaultContext) => spanExtractor.extract(content, ctx);

  // ── 1. Deterministic (LCS) resolver ──
  console.log("\n" + "=".repeat(60));
  console.log("DETERMINISTIC (LCS) RESOLVER");
  console.log("=".repeat(60));

  const deterministicResolver = new AliasResolver();
  const detResult = await evaluate(
    "Deterministic (LCS)",
    extractFn, files, noteTitles,
    (phrases, titles, src) => deterministicResolver.resolve(phrases, titles, src),
    vaultContext,
  );
  printResult(detResult);
  printWorstPrecision(detResult);
  printWorstRecall(detResult);

  // ── 2. Stochastic: Arctic-embed-xs (384d) ──
  console.log("\n" + "=".repeat(60));
  console.log("STOCHASTIC: Arctic-embed-xs (384d, threshold=0.95)");
  console.log("=".repeat(60));

  const arcticProvider = new PipelineEmbeddingProvider("Snowflake/snowflake-arctic-embed-xs", 384);
  const arcticResolver = new EmbeddingResolver({
    embeddingProvider: arcticProvider,
    similarityThreshold: 0.95,
  });

  const arcticResult = await evaluate(
    "Arctic-embed-xs (384d)",
    extractFn, files, noteTitles,
    (phrases, titles, src) => arcticResolver.resolve(phrases, titles, src),
    vaultContext,
  );
  printResult(arcticResult);
  printWorstPrecision(arcticResult);
  printWorstRecall(arcticResult);
  await arcticProvider.dispose();

  // ── 3. Stochastic: EmbeddingGemma-300m q4 (768d) ──
  console.log("\n" + "=".repeat(60));
  console.log("STOCHASTIC: EmbeddingGemma-300m q4 (768d, threshold=0.95)");
  console.log("=".repeat(60));

  const gemmaProvider = new GemmaEmbeddingProvider("onnx-community/embeddinggemma-300m-ONNX", 768, "q4");
  const gemmaResolver = new EmbeddingResolver({
    embeddingProvider: gemmaProvider,
    similarityThreshold: 0.95,
  });

  const gemmaResult = await evaluate(
    "EmbeddingGemma-300m q4 (768d)",
    extractFn, files, noteTitles,
    (phrases, titles, src) => gemmaResolver.resolve(phrases, titles, src),
    vaultContext,
  );
  printResult(gemmaResult);
  printWorstPrecision(gemmaResult);
  printWorstRecall(gemmaResult);
  await gemmaProvider.dispose();

  // ── 4. Side-by-side comparison ──
  console.log("\n" + "=".repeat(60));
  console.log("SIDE-BY-SIDE COMPARISON");
  console.log("=".repeat(60));

  const allResults = [
    { name: "Deterministic (LCS)", result: detResult },
    { name: "Arctic-embed-xs (384d)", result: arcticResult },
    { name: "EmbeddingGemma q4 (768d)", result: gemmaResult },
  ];

  console.table(allResults.map(({ name, result: r }) => ({
    Resolver: name,
    TP: r.totalTP, FP: r.totalFP, FN: r.totalFN,
    P: r.precision.toFixed(4),
    R: r.recall.toFixed(4),
    F1: r.f1.toFixed(4),
  })));

  // ── 5. Per-file deltas (Gemma vs LCS) ──
  console.log("\n=== Files where Gemma beats LCS (by F1) ===");
  const deltas: Array<{ file: string; lcsF1: string; gemmaF1: string; delta: string }> = [];
  for (const detFile of detResult.perFile) {
    const gemmaFile = gemmaResult.perFile.find((r) => r.file === detFile.file);
    if (!gemmaFile) continue;
    const d = gemmaFile.f1 - detFile.f1;
    if (Math.abs(d) > 0.01) {
      deltas.push({
        file: detFile.file,
        lcsF1: detFile.f1.toFixed(3),
        gemmaF1: gemmaFile.f1.toFixed(3),
        delta: (d > 0 ? "+" : "") + d.toFixed(3),
      });
    }
  }
  deltas.sort((a, b) => parseFloat(b.delta) - parseFloat(a.delta));
  console.table(deltas.slice(0, 15));

  console.log("\n=== Files where LCS beats Gemma (by F1) ===");
  console.table(deltas.filter(d => parseFloat(d.delta) < 0).slice(0, 15));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
