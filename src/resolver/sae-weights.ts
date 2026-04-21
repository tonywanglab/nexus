import weightsBin from "../../assets/sae-weights.bin";
import { SparseAutoencoder } from "./sae";

/**
 * Load the SAE weights bundled into the plugin (produced by
 * `scripts/train-sae.ts` from the Wikipedia pretraining pipeline).
 *
 * Synchronous - the bytes are inlined into main.js via esbuild's binary loader.
 */
export function loadBundledSAE(): SparseAutoencoder {
  return SparseAutoencoder.deserialize(weightsBin);
}
