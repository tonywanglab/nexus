// precision/Recall evaluation framework for the Nexus link discovery pipeline.
//
// uses existing wikilinks in the senior-thesis-vault as ground truth:
// if a user already linked [[Data Warehouse]], the system should discover that link too.
//
// for each note:
// 1. Extract ground truth — parse all [[wikilinks]] from raw markdown
// 2. Strip wikilinks — so extractors see plain text (no link hints)
// 3. Run pipeline — extractor → resolver → candidate edges
// 4. Compare — predicted vs ground truth → TP, FP, FN → precision, recall, F1

import * as fs from "fs";
import * as path from "path";
import { SpanExtractor } from "../keyphrase/span-extractor";
import { AliasResolver } from "../resolver/index";
import { normalize } from "../resolver/normalization";
import { ExtractedPhrase, CandidateEdge, VaultContext } from "../types";

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
  //  FN targets that were reached via a display alias (e.g. [[Target|alias]]).
  fnAliasedList: string[];
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

// returns both the ground-truth target set and a set of normalized targets
// that were reached via an alias (i.e., [[target|display]] where target ≠ display).
function extractGroundTruth(
  content: string,
  sourceBasename: string,
): { targets: Set<string>; aliasedTargets: Set<string> } {
  const targets = new Set<string>();
  const aliasedTargets = new Set<string>();
  const normalizedSource = normalize(sourceBasename);

  // strip YAML frontmatter
  let text = content.replace(/^---\n[\s\S]*?\n---\n?/, "");

  // strip fenced code blocks
  text = text.replace(/```[\s\S]*?```/g, "");

  // truncate at %% wiki footer
  const footerIdx = text.indexOf("%% wiki footer");
  if (footerIdx !== -1) {
    text = text.slice(0, footerIdx);
  }

  // parse wikilinks
  const re = /(!?)\[\[([^\]]+)\]\]/g;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    // skip embeds (![[...]])
    if (match[1] === "!") continue;

    const inner = match[2];

    // split on | → take target part (before pipe)
    const pipeIdx = inner.indexOf("|");
    const targetPart = pipeIdx !== -1 ? inner.slice(0, pipeIdx) : inner;
    const displayPart = pipeIdx !== -1 ? inner.slice(pipeIdx + 1) : null;

    // split on # → take title part (before hash)
    const hashIdx = targetPart.indexOf("#");
    const title = hashIdx !== -1 ? targetPart.slice(0, hashIdx) : targetPart;

    // skip empty titles (heading-only self-links like [[#heading]])
    if (!title.trim()) continue;

    const normalizedTarget = normalize(title);

    // skip self-links
    if (normalizedTarget === normalizedSource) continue;

    targets.add(normalizedTarget);

    // track aliased links: [[target|display]] where display differs from target
    if (displayPart !== null && normalize(displayPart) !== normalizedTarget) {
      aliasedTargets.add(normalizedTarget);
    }
  }

  return { targets, aliasedTargets };
}

// ── Wikilink stripping ──────────────────────────────────────────

function stripWikilinksForEval(content: string): string {
  // remove transclusions/embeds entirely
  let text = content.replace(/!\[\[[^\]]*\]\]/g, "");

  // [[target|display]] → "display target"
  // inject the target title so the vault-title scan can find it even when the
  // display text is an abbreviation or alias (e.g. [[Online Analytical Processing|OLAP]]).
  text = text.replace(/\[\[([^\]|#]+)(?:#[^\]|]*)?\|([^\]]+)\]\]/g, "$2 $1");

  // [[target#heading]] → target
  text = text.replace(/\[\[([^\]#]+)#[^\]]*\]\]/g, "$1");

  // [[target]] → target
  text = text.replace(/\[\[([^\]]+)\]\]/g, "$1");

  return text;
}

// ── Vault context builder ───────────────────────────────────────

function buildVaultContext(_files: VaultFile[], noteTitles: string[]): VaultContext {
  return { noteTitles };
}

// ── Evaluation engine ───────────────────────────────────────────

async function evaluateExtractor(
  name: string,
  extractFn: (content: string, vaultContext: VaultContext) => Promise<ExtractedPhrase[]>,
  files: VaultFile[],
  noteTitles: string[],
  resolver: AliasResolver,
  vaultContext: VaultContext,
): Promise<AggregateResult> {
  const normalizedTitles = new Set(noteTitles.map(normalize));
  const perFile: PerFileResult[] = [];

  for (const file of files) {
    // 1. Extract ground truth from raw content
    const { targets: rawTargets, aliasedTargets } = extractGroundTruth(
      file.content,
      file.basename,
    );

    // 2. Filter ground truth to targets that exist in the vault
    const groundTruth = new Set<string>();
    for (const target of rawTargets) {
      if (normalizedTitles.has(target)) {
        groundTruth.add(target);
      }
    }

    // skip files with no valid ground truth
    if (groundTruth.size === 0) continue;

    // 3. Strip wikilinks, run extractor + resolver
    const strippedContent = stripWikilinksForEval(file.content);
    const phrases = await extractFn(strippedContent, vaultContext);
    const candidates = await resolver.resolve(phrases, noteTitles, file.basename + ".md");

    // 4. Build predicted target set (normalized)
    const predicted = new Set<string>(
      candidates.map((c) => normalize(c.targetPath))
    );

    // 5. Compute TP/FP/FN
    const tpList: string[] = [];
    const fpList: string[] = [];
    const fnList: string[] = [];
    const fnAliasedList: string[] = [];

    for (const target of predicted) {
      if (groundTruth.has(target)) {
        tpList.push(target);
      } else {
        fpList.push(target);
      }
    }
    for (const target of groundTruth) {
      if (!predicted.has(target)) {
        fnList.push(target);
        if (aliasedTargets.has(target)) fnAliasedList.push(target);
      }
    }

    const tp = tpList.length;
    const fp = fpList.length;
    const fn = fnList.length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    perFile.push({
      file: file.basename,
      groundTruth,
      predicted,
      tp, fp, fn,
      precision, recall, f1,
      tpList, fpList, fnList, fnAliasedList,
    });
  }

  // aggregate
  const totalTP = perFile.reduce((s, r) => s + r.tp, 0);
  const totalFP = perFile.reduce((s, r) => s + r.fp, 0);
  const totalFN = perFile.reduce((s, r) => s + r.fn, 0);
  const precision = totalTP + totalFP > 0 ? totalTP / (totalTP + totalFP) : 0;
  const recall = totalTP + totalFN > 0 ? totalTP / (totalTP + totalFN) : 0;
  const f1 = precision + recall > 0
    ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    name,
    filesEvaluated: perFile.length,
    totalTP, totalFP, totalFN,
    precision, recall, f1,
    perFile,
  };
}


// ── Tests ───────────────────────────────────────────────────────

const VAULT_DIR = path.resolve(__dirname, "../../senior-thesis-vault");

describe("Evaluation Framework", () => {
  let files: VaultFile[];
  let noteTitles: string[];
  let vaultContext: VaultContext;
  let spanResult: AggregateResult;

  const vaultExists = fs.existsSync(VAULT_DIR);

  beforeAll(async () => {
    if (!vaultExists) return;

    // load vault
    files = loadVault(VAULT_DIR);
    if (files.length === 0) return;

    noteTitles = files.map((f) => f.basename);
    vaultContext = buildVaultContext(files, noteTitles);

    // run SpanExtractor + Deterministic (LCS) resolver
    const spanExtractor = new SpanExtractor();
    spanResult = await evaluateExtractor(
      "SpanExtractor",
      (content, ctx) => spanExtractor.extract(content, ctx),
      files, noteTitles,
      new AliasResolver(),
      vaultContext,
    );
  }, 120_000);

  const skipReason = "senior-thesis-vault/ not found or empty — skipping eval tests";

  describe("SpanExtractor", () => {
    it("achieves non-zero precision", () => {
      if (!vaultExists || !files?.length) return console.warn(skipReason);
      expect(spanResult.precision).toBeGreaterThan(0);
    });

    it("achieves non-zero recall", () => {
      if (!vaultExists || !files?.length) return console.warn(skipReason);
      expect(spanResult.recall).toBeGreaterThan(0);
    });

    it("reports aggregate metrics", () => {
      if (!vaultExists || !files?.length) return console.warn(skipReason);
      console.table({
        Extractor: spanResult.name,
        Files: spanResult.filesEvaluated,
        TP: spanResult.totalTP,
        FP: spanResult.totalFP,
        FN: spanResult.totalFN,
        P: spanResult.precision.toFixed(4),
        R: spanResult.recall.toFixed(4),
        F1: spanResult.f1.toFixed(4),
      });
    });
  });

  describe("Per-file breakdown", () => {
    it("shows worst recall files (SpanExtractor)", () => {
      if (!vaultExists || !files?.length) return console.warn(skipReason);
      const worstRecall = [...spanResult.perFile]
        .filter((r) => r.groundTruth.size >= 2)
        .sort((a, b) => a.recall - b.recall)
        .slice(0, 10);

      console.log("\n=== SpanExtractor: Worst Recall ===");
      console.table(
        worstRecall.map((r) => ({
          File: r.file,
          "Ground Truth": r.groundTruth.size,
          TP: r.tp,
          FN: r.fn,
          Recall: r.recall.toFixed(4),
          "Missed": r.fnList.join(", "),
        }))
      );
    });

    it("shows worst precision files (SpanExtractor)", () => {
      if (!vaultExists || !files?.length) return console.warn(skipReason);
      const worstPrecision = [...spanResult.perFile]
        .filter((r) => r.predicted.size >= 2)
        .sort((a, b) => a.precision - b.precision)
        .slice(0, 10);

      console.log("\n=== SpanExtractor: Worst Precision ===");
      console.table(
        worstPrecision.map((r) => ({
          File: r.file,
          Predicted: r.predicted.size,
          TP: r.tp,
          FP: r.fp,
          Precision: r.precision.toFixed(4),
          "False Positives": r.fpList.join(", "),
        }))
      );
    });

    it("shows TP/FP/FN details per file", () => {
      if (!vaultExists || !files?.length) return console.warn(skipReason);
      const detailed = [...spanResult.perFile]
        .sort((a, b) => b.groundTruth.size - a.groundTruth.size)
        .slice(0, 15);

      console.log("\n=== SpanExtractor: Per-File Details (top by ground truth size) ===");
      for (const r of detailed) {
        console.log(`\n--- ${r.file} ---`);
        console.log(`  Ground truth (${r.groundTruth.size}): ${[...r.groundTruth].join(", ")}`);
        console.log(`  Predicted (${r.predicted.size}): ${[...r.predicted].join(", ")}`);
        console.log(`  TP (${r.tp}): ${r.tpList.join(", ")}`);
        console.log(`  FP (${r.fp}): ${r.fpList.join(", ")}`);
        console.log(`  FN (${r.fn}): ${r.fnList.join(", ")}`);
        console.log(`  P=${r.precision.toFixed(3)} R=${r.recall.toFixed(3)} F1=${r.f1.toFixed(3)}`);
      }
    });
  });
});
