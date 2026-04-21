import {
  isEmbeddingGemmaModelId,
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
