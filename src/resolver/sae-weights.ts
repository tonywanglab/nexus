import weights from "../../assets/sae-weights-v2.bin";
import { SparseAutoencoder } from "./sae";

export function loadBundledSAE(): SparseAutoencoder {
  return SparseAutoencoder.deserialize(weights);
}
