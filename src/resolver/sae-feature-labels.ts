import labelsJson from "../../assets/sae-feature-labels.json";
import { SparseEncoding } from "./sae";

export interface FeatureLabel { candidates: string[]; scores: number[]; }

export interface LabeledFeatures {
  indices: number[];
  values: number[];
  labels: string[];
}

export class SAEFeatureLabels {
  private _labels: FeatureLabel[];

  constructor() {
    this._labels = (labelsJson as any).labels as FeatureLabel[];
  }

  get liveCount(): number {
    return this._labels.filter(l => l.candidates.length > 0).length;
  }

  /** Returns "candidate1 · candidate2 · candidate3" for live features, null for dead. */
  labelFor(idx: number): string | null {
    const l = this._labels[idx];
    if (!l || l.candidates.length === 0) return null;
    return l.candidates.join(" · ");
  }

  /**
   * All active features in `enc` that have a label, sorted by activation desc.
   * Use this for matching — retains full sparse signal.
   */
  pickAllLabeled(enc: SparseEncoding): LabeledFeatures {
    return this._pick(enc, Number.POSITIVE_INFINITY);
  }

  /**
   * Top-4 labeled active features, sorted by activation desc.
   * Use this for the display payload on CandidateEdge.
   */
  pickTop4Labeled(enc: SparseEncoding): LabeledFeatures {
    return this._pick(enc, 4);
  }

  private _pick(enc: SparseEncoding, limit: number): LabeledFeatures {
    const k = enc.indices.length;
    const order = Array.from({ length: k }, (_, i) => i)
      .sort((a, b) => enc.values[b] - enc.values[a]);
    const indices: number[] = [];
    const values: number[] = [];
    const labels: string[] = [];
    for (const i of order) {
      if (indices.length >= limit) break;
      const idx = enc.indices[i];
      const lbl = this.labelFor(idx);
      if (lbl !== null) {
        indices.push(idx);
        values.push(enc.values[i]);
        labels.push(lbl);
      }
    }
    return { indices, values, labels };
  }
}
