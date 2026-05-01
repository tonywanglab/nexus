import weights from "../../assets/sae-weights-v3.bin";
import { SparseAutoencoder } from "./sae";

export function loadBundledSAE(): SparseAutoencoder {
  return SparseAutoencoder.deserialize(weights);
}
