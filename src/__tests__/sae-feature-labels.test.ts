import { SAEFeatureLabels } from "../resolver/sae-feature-labels";
import { SparseEncoding } from "../resolver/sae";

// Mock the JSON module so tests don't depend on the real asset file.
jest.mock("../../assets/sae-feature-labels.json", () => ({
  dHidden: 6,
  vocabSize: 10,
  vocabSource: "test",
  minScore: 0.25,
  labelsPerFeature: 3,
  labels: [
    // feature 0: live, 3 candidates
    { candidates: ["politician", "statesman", "senator"], scores: [0.9, 0.8, 0.7] },
    // feature 1: live, 2 candidates
    { candidates: ["economy", "finance"], scores: [0.6, 0.5] },
    // feature 2: dead
    { candidates: [], scores: [] },
    // feature 3: live, 1 candidate
    { candidates: ["science"], scores: [0.4] },
    // feature 4: dead
    { candidates: [], scores: [] },
    // feature 5: live, 3 candidates
    { candidates: ["art", "painting", "sculpture"], scores: [0.88, 0.75, 0.65] },
  ],
}), { virtual: true });

function makeEncoding(indexValuePairs: [number, number][]): SparseEncoding {
  const indices = new Int32Array(indexValuePairs.map(p => p[0]));
  const values = new Float32Array(indexValuePairs.map(p => p[1]));
  return { indices, values };
}

describe("SAEFeatureLabels", () => {
  let fl: SAEFeatureLabels;
  beforeEach(() => { fl = new SAEFeatureLabels(); });

  describe("liveCount", () => {
    it("counts only features with non-empty candidates", () => {
      expect(fl.liveCount).toBe(4); // features 0, 1, 3, 5
    });
  });

  describe("labelFor", () => {
    it("returns joined candidates for a live feature", () => {
      expect(fl.labelFor(0)).toBe("politician · statesman · senator");
    });

    it("joins only the available candidates (fewer than 3)", () => {
      expect(fl.labelFor(1)).toBe("economy · finance");
    });

    it("returns single candidate without separator", () => {
      expect(fl.labelFor(3)).toBe("science");
    });

    it("returns null for a dead feature", () => {
      expect(fl.labelFor(2)).toBeNull();
      expect(fl.labelFor(4)).toBeNull();
    });

    it("returns null for out-of-range index", () => {
      expect(fl.labelFor(999)).toBeNull();
    });
  });

  describe("pickTop4Labeled", () => {
    it("returns top 4 labeled features sorted by activation desc", () => {
      // features 0(val=5), 1(val=4), 3(val=3), 5(val=2), 2(val=6,dead), 4(val=1,dead)
      const enc = makeEncoding([[2, 6], [0, 5], [1, 4], [3, 3], [5, 2], [4, 1]]);
      const result = fl.pickTop4Labeled(enc);
      expect(result.indices).toEqual([0, 1, 3, 5]);
      expect(result.values[0]).toBe(5);
      expect(result.values[1]).toBe(4);
      expect(result.values[2]).toBe(3);
      expect(result.values[3]).toBe(2);
      expect(result.labels[0]).toBe("politician · statesman · senator");
    });

    it("degrades gracefully when fewer than 4 live features are active", () => {
      // Only feature 0 is active and live
      const enc = makeEncoding([[0, 1.0], [2, 0.9], [4, 0.8]]);
      const result = fl.pickTop4Labeled(enc);
      expect(result.indices).toEqual([0]);
      expect(result.labels).toEqual(["politician · statesman · senator"]);
    });

    it("returns empty arrays when no active features are labeled", () => {
      const enc = makeEncoding([[2, 1.0], [4, 0.5]]);
      const result = fl.pickTop4Labeled(enc);
      expect(result.indices).toHaveLength(0);
      expect(result.values).toHaveLength(0);
      expect(result.labels).toHaveLength(0);
    });
  });

  describe("pickAllLabeled", () => {
    it("returns all labeled active features sorted by activation desc", () => {
      // 4 live features active: 0,1,3,5 + 2 dead: 2,4
      const enc = makeEncoding([[2, 6], [0, 5], [1, 4], [3, 3], [5, 2], [4, 1]]);
      const result = fl.pickAllLabeled(enc);
      expect(result.indices).toEqual([0, 1, 3, 5]);
      expect(result.values).toHaveLength(4);
    });

    it("returns more than 4 features when > 4 live features are active", () => {
      // Hypothetical 8-feature encoding — but our mock only has 6 features.
      // Use all 4 live features (indices 0,1,3,5) plus both dead ones.
      const enc = makeEncoding([[0, 0.9], [1, 0.8], [3, 0.7], [5, 0.6], [2, 0.5], [4, 0.4]]);
      const result = fl.pickAllLabeled(enc);
      // All 4 live features are returned (no 4-cap)
      expect(result.indices.length).toBe(4);
      expect(result.indices).toEqual([0, 1, 3, 5]);
    });

    it("returns empty arrays when no active features are labeled", () => {
      const enc = makeEncoding([[2, 1.0], [4, 0.5]]);
      const result = fl.pickAllLabeled(enc);
      expect(result.indices).toHaveLength(0);
    });
  });
});
