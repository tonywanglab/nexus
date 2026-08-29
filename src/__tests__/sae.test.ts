import { SparseAutoencoder, topK, topNOf, SAEWeights } from "../resolver/sae";

function makeTrivialWeights(
  dModel: number,
  dHidden: number,
  k: number,
): SAEWeights {
  return {
    dModel,
    dHidden,
    k,
    wEnc: new Float32Array(dHidden * dModel),
    bEnc: new Float32Array(dHidden),
    bPre: new Float32Array(dModel),
    wDec: new Float32Array(dModel * dHidden),
    bDec: new Float32Array(dModel),
  };
}

describe("topK", () => {
  it("returns k largest values", () => {
    const v = new Float32Array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]);
    const { indices, values } = topK(v, 3);
    const pairs = Array.from(indices).map((idx, i) => ({ idx, val: values[i] }));
    pairs.sort((a, b) => b.val - a.val);
    expect(pairs[0].val).toBe(9);
    expect(pairs[1].val).toBe(6);
    expect(pairs[2].val).toBe(5);
  });

  it("handles k === n", () => {
    const v = new Float32Array([3, 1, 4, 1]);
    const { indices, values } = topK(v, 4);
    expect(indices.length).toBe(4);
    // values+indices should cover every element
    const seen = new Set<number>();
    for (let i = 0; i < 4; i++) {
      seen.add(indices[i]);
      expect(values[i]).toBe(v[indices[i]]);
    }
    expect(seen.size).toBe(4);
  });

  it("throws when k > n", () => {
    expect(() => topK(new Float32Array([1, 2]), 3)).toThrow();
  });
});

describe("SparseAutoencoder constructor validation", () => {
  it("rejects mismatched wEnc length", () => {
    expect(
      () =>
        new SparseAutoencoder({
          dModel: 4,
          dHidden: 8,
          k: 2,
          wEnc: new Float32Array(31), // should be 32
          bEnc: new Float32Array(8),
          bPre: new Float32Array(4),
          wDec: new Float32Array(32),
          bDec: new Float32Array(4),
        }),
    ).toThrow();
  });

  it("rejects k > dHidden", () => {
    expect(
      () =>
        new SparseAutoencoder({
          dModel: 4,
          dHidden: 8,
          k: 10,
          wEnc: new Float32Array(32),
          bEnc: new Float32Array(8),
          bPre: new Float32Array(4),
          wDec: new Float32Array(32),
          bDec: new Float32Array(4),
        }),
    ).toThrow();
  });
});

describe("SparseAutoencoder.encode", () => {
  const dModel = 4;
  const dHidden = 6;
  const k = 2;

  it("produces exactly k non-zeros for positive inputs", () => {
    const w = makeTrivialWeights(dModel, dHidden, k);
    // set wEnc so each hidden unit just reads one input dim with known weight
    // wEnc[i * dModel + j] = identity-ish + some distinguishing signal
    for (let i = 0; i < dHidden; i++) {
      for (let j = 0; j < dModel; j++) {
        w.wEnc[i * dModel + j] = i === j ? 1 : 0;
      }
    }
    // so output[i] = ReLU(x[i]) for i < dModel, else ReLU(0)
    const sae = new SparseAutoencoder(w);

    const x = new Float32Array([5, 3, 1, 2]);
    const sparse = sae.encode(x);
    let nonzero = 0;
    for (let i = 0; i < sparse.length; i++) if (sparse[i] !== 0) nonzero++;
    expect(nonzero).toBe(k);
    // the top-2 should be indices 0 and 1 (values 5 and 3).
    expect(sparse[0]).toBe(5);
    expect(sparse[1]).toBe(3);
  });

  it("returns length dHidden", () => {
    const sae = SparseAutoencoder.randomInit(8, 32, 4, 123);
    const out = sae.encode(new Float32Array(8).fill(0.5));
    expect(out.length).toBe(32);
  });

  it("encodeSparse returns indices+values consistent with encode", () => {
    const sae = SparseAutoencoder.randomInit(8, 32, 4, 123);
    const x = new Float32Array([0.1, 0.2, 0.3, -0.1, 0.5, 0.0, -0.4, 0.7]);
    const dense = sae.encode(x);
    const compact = sae.encodeSparse(x);
    expect(compact.indices.length).toBe(4);
    expect(compact.values.length).toBe(4);

    let nonzeroInDense = 0;
    for (let i = 0; i < dense.length; i++) if (dense[i] !== 0) nonzeroInDense++;
    expect(nonzeroInDense).toBe(4);

    // every (index, value) from compact should match dense[index]
    for (let t = 0; t < compact.indices.length; t++) {
      expect(dense[compact.indices[t]]).toBeCloseTo(compact.values[t], 6);
    }
  });
});

describe("SparseAutoencoder.decode", () => {
  it("reconstructs b_dec + b_pre when z is zero", () => {
    const w = makeTrivialWeights(3, 5, 2);
    w.bDec[0] = 0.5;
    w.bDec[1] = -0.25;
    w.bDec[2] = 0.1;
    w.bPre[0] = 0.2;
    const sae = new SparseAutoencoder(w);
    const out = sae.decode(new Float32Array(5));
    expect(out[0]).toBeCloseTo(0.7, 6);
    expect(out[1]).toBeCloseTo(-0.25, 6);
    expect(out[2]).toBeCloseTo(0.1, 6);
  });

  it("adds one atom when z has a single 1", () => {
    const w = makeTrivialWeights(3, 4, 2);
    // decoder column 1 = [1, 2, 3]
    w.wDec[0 * 4 + 1] = 1;
    w.wDec[1 * 4 + 1] = 2;
    w.wDec[2 * 4 + 1] = 3;
    const sae = new SparseAutoencoder(w);
    const z = new Float32Array([0, 1, 0, 0]);
    const out = sae.decode(z);
    expect(out[0]).toBeCloseTo(1);
    expect(out[1]).toBeCloseTo(2);
    expect(out[2]).toBeCloseTo(3);
  });
});

describe("SparseAutoencoder.normalizeDecoder", () => {
  it("projects each decoder column to unit L2 norm", () => {
    const w = makeTrivialWeights(4, 6, 2);
    // fill decoder with non-unit columns
    for (let i = 0; i < w.wDec.length; i++) w.wDec[i] = (i + 1) * 0.1;
    const sae = new SparseAutoencoder(w);
    sae.normalizeDecoder();

    for (let j = 0; j < 6; j++) {
      let sumSq = 0;
      for (let i = 0; i < 4; i++) {
        const v = sae.wDec[i * 6 + j];
        sumSq += v * v;
      }
      expect(Math.sqrt(sumSq)).toBeCloseTo(1, 5);
    }
  });

  it("leaves zero columns untouched", () => {
    const w = makeTrivialWeights(3, 4, 1);
    w.wDec[0 * 4 + 0] = 3;
    w.wDec[1 * 4 + 0] = 4;
    // column 0 has norm 5; columns 1-3 are zero
    const sae = new SparseAutoencoder(w);
    sae.normalizeDecoder();
    expect(sae.wDec[0 * 4 + 0]).toBeCloseTo(3 / 5);
    expect(sae.wDec[1 * 4 + 0]).toBeCloseTo(4 / 5);
    for (let j = 1; j < 4; j++) {
      for (let i = 0; i < 3; i++) {
        expect(sae.wDec[i * 4 + j]).toBe(0);
      }
    }
  });
});

describe("SparseAutoencoder serialize/deserialize round-trip", () => {
  it("preserves all weights and hyperparameters", () => {
    const sae = SparseAutoencoder.randomInit(16, 48, 6, 42);
    const bytes = sae.serialize();
    const round = SparseAutoencoder.deserialize(bytes);

    expect(round.dModel).toBe(sae.dModel);
    expect(round.dHidden).toBe(sae.dHidden);
    expect(round.k).toBe(sae.k);
    expect(Array.from(round.wEnc)).toEqual(Array.from(sae.wEnc));
    expect(Array.from(round.bEnc)).toEqual(Array.from(sae.bEnc));
    expect(Array.from(round.bPre)).toEqual(Array.from(sae.bPre));
    expect(Array.from(round.wDec)).toEqual(Array.from(sae.wDec));
    expect(Array.from(round.bDec)).toEqual(Array.from(sae.bDec));
  });

  it("produces identical encodings after round-trip", () => {
    const sae = SparseAutoencoder.randomInit(8, 32, 4, 7);
    const round = SparseAutoencoder.deserialize(sae.serialize());
    const x = new Float32Array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]);
    const a = sae.encode(x);
    const b = round.encode(x);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("rejects invalid magic bytes", () => {
    const bytes = new Uint8Array(16);
    expect(() => SparseAutoencoder.deserialize(bytes)).toThrow();
  });
});

describe("topNOf", () => {
  function makeEncoding(indexValuePairs: [number, number][]) {
    const indices = new Int32Array(indexValuePairs.map(p => p[0]));
    const values = new Float32Array(indexValuePairs.map(p => p[1]));
    return { indices, values };
  }

  it("returns n highest activations when n < k", () => {
    const enc = makeEncoding([[5, 9], [2, 3], [7, 6], [1, 1]]);
    const result = topNOf(enc, 2);
    expect(result.indices.length).toBe(2);
    // top 2 by value: index 5 (val 9) and index 7 (val 6)
    const pairs = Array.from(result.indices).map((idx, i) => ({ idx, val: result.values[i] }));
    pairs.sort((a, b) => b.val - a.val);
    expect(pairs[0].val).toBe(9);
    expect(pairs[1].val).toBe(6);
  });

  it("returns all k when n === k", () => {
    const enc = makeEncoding([[0, 3], [1, 1], [2, 4]]);
    const result = topNOf(enc, 3);
    expect(result.indices.length).toBe(3);
    const seen = new Set(Array.from(result.indices));
    expect(seen.has(0)).toBe(true);
    expect(seen.has(1)).toBe(true);
    expect(seen.has(2)).toBe(true);
  });

  it("throws when n > k", () => {
    const enc = makeEncoding([[0, 1], [1, 2]]);
    expect(() => topNOf(enc, 3)).toThrow();
  });

  it("preserves Int32Array and Float32Array types", () => {
    const enc = makeEncoding([[3, 5], [7, 2]]);
    const result = topNOf(enc, 1);
    expect(result.indices).toBeInstanceOf(Int32Array);
    expect(result.values).toBeInstanceOf(Float32Array);
  });

  it("values in result match the original values at the selected indices", () => {
    const enc = makeEncoding([[10, 0.7], [20, 0.5], [30, 0.9]]);
    const result = topNOf(enc, 2);
    for (let i = 0; i < result.indices.length; i++) {
      const origPos = Array.from(enc.indices).indexOf(result.indices[i]);
      expect(result.values[i]).toBeCloseTo(enc.values[origPos], 6);
    }
  });
});

describe("SparseAutoencoder.randomInit", () => {
  it("is deterministic given a seed", () => {
    const a = SparseAutoencoder.randomInit(8, 32, 4, 777);
    const b = SparseAutoencoder.randomInit(8, 32, 4, 777);
    expect(Array.from(a.wEnc)).toEqual(Array.from(b.wEnc));
    expect(Array.from(a.wDec)).toEqual(Array.from(b.wDec));
  });

  it("starts with unit-norm decoder columns", () => {
    const sae = SparseAutoencoder.randomInit(16, 64, 8, 33);
    for (let j = 0; j < 64; j++) {
      let sumSq = 0;
      for (let i = 0; i < 16; i++) {
        const v = sae.wDec[i * 64 + j];
        sumSq += v * v;
      }
      expect(Math.sqrt(sumSq)).toBeCloseTo(1, 4);
    }
  });
});
