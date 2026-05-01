import labels from "../../assets/sae-feature-labels-v3.json";
import { SparseEncoding } from "./sae";

export interface FeatureLabel { candidates: string[]; scores: number[]; topTerms?: string[]; }

export interface LabeledFeatures {
  indices: number[];
  values: number[];
  labels: string[];
  /** Top vocab terms per feature (evidence behind each label). Parallel to indices/labels. */
  terms: Array<string[] | null>;
}

export class SAEFeatureLabels {
  private _labels: FeatureLabel[];
  private _overrides: Map<number, string>;

  constructor(saeDHidden?: number) {
    const raw = (labels as any).labels as FeatureLabel[];
    const empty = (): FeatureLabel => ({ candidates: [], scores: [] });
    const target = saeDHidden ?? raw.length;
    this._labels = raw.slice(0, target);
    while (this._labels.length < target) {
      this._labels.push(empty());
    }
    this._overrides = new Map();
  }

  setOverrides(overrides: Record<number, string>): void {
    this._overrides = new Map(Object.entries(overrides).map(([k, v]) => [Number(k), v]));
  }

  setOverride(idx: number, label: string): void {
    this._overrides.set(idx, label);
  }

  /** Replace labels from a parsed sae-feature-labels.json object. */
  loadFromJSON(data: { labels: FeatureLabel[] }): void {
    const empty = (): FeatureLabel => ({ candidates: [], scores: [] });
    const target = this._labels.length;
    this._labels = data.labels.slice(0, target);
    while (this._labels.length < target) this._labels.push(empty());
  }

  /** Create an instance from a parsed sae-feature-labels.json without the static default. */
  static fromJSON(data: { labels: FeatureLabel[] }, dHidden: number): SAEFeatureLabels {
    const instance = Object.create(SAEFeatureLabels.prototype) as SAEFeatureLabels;
    const empty = (): FeatureLabel => ({ candidates: [], scores: [] });
    (instance as any)._labels = data.labels.slice(0, dHidden);
    while ((instance as any)._labels.length < dHidden) (instance as any)._labels.push(empty());
    return instance;
  }

  get liveCount(): number {
    return this._labels.filter(l => l.candidates.length > 0).length;
  }

  /** Returns the synthesized concept label for live features, null for dead. */
  labelFor(idx: number): string | null {
    if (this._overrides.has(idx)) return this._overrides.get(idx)!;
    const l = this._labels[idx];
    if (!l || l.candidates.length === 0) return null;
    return l.candidates.join(" · ");
  }

  /** Returns the top-10 raw vocab terms for the atom (evidence behind the label), or null. */
  termsFor(idx: number): string[] | null {
    const l = this._labels[idx];
    if (!l || l.candidates.length === 0) return null;
    return l.topTerms ?? null;
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
    const terms: Array<string[] | null> = [];
    for (const i of order) {
      if (indices.length >= limit) break;
      const idx = enc.indices[i];
      const lbl = this.labelFor(idx);
      if (lbl !== null) {
        indices.push(idx);
        values.push(enc.values[i]);
        labels.push(lbl);
        terms.push(this.termsFor(idx));
      }
    }
    return { indices, values, labels, terms };
  }
}
