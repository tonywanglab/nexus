#!/usr/bin/env npx ts-node
/**
 * One-shot helper that writes a random-initialized SAE to assets/sae-weights.bin.
 *
 * Intended as a placeholder so the plugin builds before real pretraining is
 * run. Calling `npm run train:sae` after the Wikipedia pipeline will overwrite
 * this file with actual trained weights.
 *
 * Usage:
 *   npx ts-node --compiler-options '{"module":"CommonJS","esModuleInterop":true}' \
 *     scripts/bootstrap-sae-weights.ts
 */

import * as fs from "fs";
import * as path from "path";
import { SparseAutoencoder } from "../src/resolver/sae";

// Placeholder shipped with the plugin — small d_hidden keeps the binary ~9MB (768×1536×2 fp32).
// Production training overrides via `npm run train:sae -- --d-hidden=16384 --k=6`.
const D_MODEL = 768;
const D_HIDDEN = 1536;
const K = 32;
const SEED = 0xC0FFEE;
const OUT = "assets/sae-weights.bin";

const sae = SparseAutoencoder.randomInit(D_MODEL, D_HIDDEN, K, SEED);
const bytes = sae.serialize();

const dir = path.dirname(OUT);
if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
fs.writeFileSync(OUT, bytes);

console.log(
  `Wrote placeholder SAE weights (d_model=${D_MODEL}, d_hidden=${D_HIDDEN}, k=${K}, seed=${SEED}) to ${OUT}`,
);
console.log(`File size: ${bytes.length.toLocaleString()} bytes`);
console.log(
  `NOTE: These are RANDOM weights. Run npm run wiki:prepare && wiki:embed && train:sae for the real pretraining.`,
);
