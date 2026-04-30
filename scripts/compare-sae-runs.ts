#!/usr/bin/env npx ts-node
/**
 * Compare SAE ablation runs across k and expansion factor configurations.
 *
 * Usage:
 *   npm run ablation:compare
 */

import * as fs from "fs";
import * as path from "path";

interface TrainingReport {
  finalValMSE?: number;
  finalMetrics?: { valLoss: number; deadFeatures: number; avgL0: number };
  epochs?: { valLoss: number; deadFeatures: number; avgL0: number }[];
  hyperparameters?: { dModel: number; dHidden: number; k: number; epochs: number };
  dModel?: number;
  dHidden?: number;
  k?: number;
  trainWallSec?: number;
  corpusSize?: number;
}

interface FeatureLabels {
  dHidden: number;
  labels: { candidates: string[]; scores: number[] }[];
}

interface RunResult {
  config: string;
  k: number;
  dHidden: number;
  expansion: string;
  valMSE: number | null;
  deadFeatures: number | null;
  liveFeatures: number | null;
  totalFeatures: number | null;
  coveragePct: string;
  weightsMB: number | null;
}

const RUNS: { config: string; dir: string; k: number; dHidden: number }[] = [
  { config: "baseline (k=6, 21×)", dir: "modal-out-v2",    k: 6, dHidden: 16128 },
  { config: "k3-21×",              dir: "modal-out-k3-21x", k: 3, dHidden: 16128 },
  { config: "k6-64×",              dir: "modal-out-k6-64x", k: 6, dHidden: 49152 },
  { config: "k3-64×",              dir: "modal-out-k3-64x", k: 3, dHidden: 49152 },
];

function loadReport(dir: string): TrainingReport | null {
  const p = path.join(dir, "sae-training-report.json");
  if (!fs.existsSync(p)) return null;
  try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch { return null; }
}

function loadLabels(dir: string): FeatureLabels | null {
  const p = path.join(dir, "sae-feature-labels.json");
  if (!fs.existsSync(p)) return null;
  try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch { return null; }
}

function weightsMB(dir: string): number | null {
  const p = path.join(dir, "sae-weights.bin");
  if (!fs.existsSync(p)) return null;
  return Math.round(fs.statSync(p).size / (1024 * 1024));
}

function getValMSE(r: TrainingReport): number | null {
  if (r.finalValMSE != null) return r.finalValMSE;
  if (r.finalMetrics?.valLoss != null) return r.finalMetrics.valLoss;
  if (r.epochs?.length) return r.epochs[r.epochs.length - 1].valLoss;
  return null;
}

function getDeadFeatures(r: TrainingReport): number | null {
  if (r.finalMetrics?.deadFeatures != null) return r.finalMetrics.deadFeatures;
  if (r.epochs?.length) return r.epochs[r.epochs.length - 1].deadFeatures;
  return null;
}

function pad(s: string, n: number): string {
  return s.length >= n ? s : s + " ".repeat(n - s.length);
}

function main(): void {
  const results: RunResult[] = [];

  for (const run of RUNS) {
    const report = loadReport(run.dir);
    const labels = loadLabels(run.dir);
    const mb = weightsMB(run.dir);

    const valMSE = report ? getValMSE(report) : null;
    const dead = report ? getDeadFeatures(report) : null;
    const total = run.dHidden;
    const live = labels ? labels.labels.filter(l => l.candidates.length > 0).length : null;
    const expansion = `${Math.round(run.dHidden / 768)}×`;

    results.push({
      config: run.config,
      k: run.k,
      dHidden: run.dHidden,
      expansion,
      valMSE,
      deadFeatures: dead,
      liveFeatures: live,
      totalFeatures: total,
      coveragePct: live != null ? `${((live / total) * 100).toFixed(1)}%` : "—",
      weightsMB: mb,
    });
  }

  const cols = ["Config", "k", "Expansion", "ValMSE", "Dead", "Live/Total", "Coverage", "Weights"];
  const rows = results.map(r => [
    r.config,
    String(r.k),
    r.expansion,
    r.valMSE != null ? r.valMSE.toExponential(3) : "—",
    r.deadFeatures != null ? String(r.deadFeatures) : "—",
    r.liveFeatures != null ? `${r.liveFeatures.toLocaleString()} / ${r.totalFeatures!.toLocaleString()}` : "—",
    r.coveragePct,
    r.weightsMB != null ? `${r.weightsMB} MB` : "—",
  ]);

  const widths = cols.map((c, i) => Math.max(c.length, ...rows.map(r => r[i].length)));
  const sep = widths.map(w => "-".repeat(w + 2)).join("+");
  const header = cols.map((c, i) => pad(c, widths[i])).join(" | ");

  console.log(`\nSAE Ablation Comparison\n`);
  console.log(`| ${header} |`);
  console.log(`|-${sep}-|`);
  for (const row of rows) {
    console.log(`| ${row.map((c, i) => pad(c, widths[i])).join(" | ")} |`);
  }
  console.log();

  const missing = results.filter(r => r.valMSE === null).map(r => r.config);
  if (missing.length) {
    console.log(`Note: runs not yet completed — ${missing.join(", ")}`);
    console.log(`Run the corresponding train:sae-* and label:autointerp:* npm scripts first.\n`);
  }
}

main();
