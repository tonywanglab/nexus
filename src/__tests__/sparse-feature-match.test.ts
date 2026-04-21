import { sparseCosine } from "../resolver/sparse-feature-match";

describe("sparseCosine", () => {
  it("returns 1.0 for identical non-empty vectors", () => {
    const a = { indices: [0, 1, 2], values: [1.0, 2.0, 3.0] };
    expect(sparseCosine(a, a)).toBeCloseTo(1.0, 6);
  });

  it("returns 0 for disjoint support", () => {
    const a = { indices: [0, 1], values: [1.0, 1.0] };
    const b = { indices: [2, 3], values: [1.0, 1.0] };
    expect(sparseCosine(a, b)).toBe(0);
  });

  it("returns 0 when either side is empty", () => {
    const a = { indices: [0], values: [1.0] };
    const empty = { indices: [] as number[], values: [] as number[] };
    expect(sparseCosine(a, empty)).toBe(0);
    expect(sparseCosine(empty, a)).toBe(0);
  });

  it("returns correct value for partial overlap (closed-form)", () => {
    // a = [2, 0, 0, 3]  (indices 0 and 3)
    // b = [0, 0, 0, 4]  (index 3)
    // dot = 3*4 = 12
    // normA = sqrt(4+9) = sqrt(13), normB = 4
    // cos = 12 / (sqrt(13)*4)
    const a = { indices: [0, 3], values: [2.0, 3.0] };
    const b = { indices: [3], values: [4.0] };
    const expected = 12 / (Math.sqrt(13) * 4);
    expect(sparseCosine(a, b)).toBeCloseTo(expected, 6);
  });

  it("is symmetric", () => {
    const a = { indices: [0, 1, 4], values: [0.5, 0.8, 0.3] };
    const b = { indices: [1, 3], values: [0.7, 0.2] };
    expect(sparseCosine(a, b)).toBeCloseTo(sparseCosine(b, a), 6);
  });

  it("is scale-invariant", () => {
    const a = { indices: [0, 1], values: [1.0, 2.0] };
    const b = { indices: [0, 1], values: [2.0, 4.0] };
    expect(sparseCosine(a, b)).toBeCloseTo(1.0, 6);
  });
});
