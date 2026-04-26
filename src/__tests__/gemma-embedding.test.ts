import {
  isEmbeddingGemmaModelId,
  meanPoolNormalizeL2BatchedFromLastHidden,
  meanPoolNormalizeL2FromLastHidden,
} from "../resolver/gemma-embedding";

describe("isEmbeddingGemmaModelId", () => {
  it("matches embeddinggemma id case-insensitively", () => {
    expect(isEmbeddingGemmaModelId("onnx-community/embeddinggemma-300m-ONNX")).toBe(true);
    expect(isEmbeddingGemmaModelId("EmbeddingGemma")).toBe(true);
  });

  it("rejects non-gemma models", () => {
    expect(isEmbeddingGemmaModelId("Snowflake/snowflake-arctic-embed-xs")).toBe(false);
  });
});

describe("meanPoolNormalizeL2FromLastHidden", () => {
  it("mean-pools seq length and L2-normalizes", () => {
    const hiddenDim = 3;
    const seqLen = 2;
    const data = new Float32Array([
      2, 0, 0,
      0, 2, 0,
    ]);
    const out = meanPoolNormalizeL2FromLastHidden(data, seqLen, hiddenDim);
    expect(out.length).toBe(3);
    expect(out[0]).toBeCloseTo(0.70710677, 5);
    expect(out[1]).toBeCloseTo(0.70710677, 5);
    expect(out[2]).toBe(0);
    let n = 0;
    for (let d = 0; d < 3; d++) n += out[d] * out[d];
    expect(n).toBeCloseTo(1, 6);
  });
});

describe("meanPoolNormalizeL2BatchedFromLastHidden", () => {
  it("matches single-sequence pooling when mask is all ones", () => {
    const hiddenDim = 3;
    const seqLen = 2;
    const batch = 1;
    const data = new Float32Array([
      2, 0, 0,
      0, 2, 0,
    ]);
    const mask = new Int32Array([1, 1]);
    const [batched] = meanPoolNormalizeL2BatchedFromLastHidden(data, batch, seqLen, hiddenDim, mask);
    const single = meanPoolNormalizeL2FromLastHidden(data, seqLen, hiddenDim);
    expect([...batched]).toEqual([...single]);
  });

  it("ignores padded positions in the mask", () => {
    const hiddenDim = 2;
    const seqLen = 3;
    const batch = 2;
    const data = new Float32Array([
      1, 0, 3, 0, 999, 999,
      0, 2, 0, 0, 0, 2,
    ]);
    const mask = new Int32Array([
      1, 1, 0,
      1, 0, 1,
    ]);
    const out = meanPoolNormalizeL2BatchedFromLastHidden(data, batch, seqLen, hiddenDim, mask);
    expect(out[0][0]).toBeCloseTo(1, 6);
    expect(out[0][1]).toBe(0);
    expect(out[1][0]).toBeCloseTo(0, 6);
    expect(out[1][1]).toBeCloseTo(1, 6);
  });
});
