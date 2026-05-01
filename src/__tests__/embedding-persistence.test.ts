import {
  serializeEmbeddings,
  deserializeEmbeddings,
} from "../embedding-persistence";

function makeVec(dim: number, seed: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = seed * 0.01 + i * 0.001;
  return v;
}

describe("embedding-persistence", () => {
  it("round-trips a small map with exact float equality", () => {
    const map = new Map<string, Float32Array>();
    map.set("alpha", makeVec(4, 1));
    map.set("beta", makeVec(4, 2));
    map.set("gamma", makeVec(4, 3));

    const buf = serializeEmbeddings(map, 4);
    const { dim, map: out } = deserializeEmbeddings(buf);

    expect(dim).toBe(4);
    expect(out.size).toBe(3);
    for (const [k, v] of map) {
      const got = out.get(k);
      expect(got).toBeDefined();
      expect(Array.from(got!)).toEqual(Array.from(v));
    }
  });

  it("round-trips an empty map", () => {
    const buf = serializeEmbeddings(new Map(), 8);
    const { dim, map } = deserializeEmbeddings(buf);
    expect(dim).toBe(8);
    expect(map.size).toBe(0);
  });

  it("preserves UTF-8 titles (accents, emoji, CJK)", () => {
    const map = new Map<string, Float32Array>();
    map.set("café ☕", makeVec(3, 1));
    map.set("日本語", makeVec(3, 2));
    map.set("résumé 📄", makeVec(3, 3));
    map.set("", makeVec(3, 4)); // zero-length title

    const buf = serializeEmbeddings(map, 3);
    const { map: out } = deserializeEmbeddings(buf);
    expect(out.size).toBe(4);
    expect(Array.from(out.get("café ☕")!)).toEqual(Array.from(map.get("café ☕")!));
    expect(Array.from(out.get("日本語")!)).toEqual(Array.from(map.get("日本語")!));
    expect(Array.from(out.get("résumé 📄")!)).toEqual(Array.from(map.get("résumé 📄")!));
    expect(Array.from(out.get("")!)).toEqual(Array.from(map.get("")!));
  });

  it("preserves insertion order on deserialization", () => {
    const keys = ["z", "a", "m", "k", "b"];
    const map = new Map<string, Float32Array>();
    for (const k of keys) map.set(k, makeVec(2, k.charCodeAt(0)));

    const buf = serializeEmbeddings(map, 2);
    const { map: out } = deserializeEmbeddings(buf);
    expect([...out.keys()]).toEqual(keys);
  });

  it("throws on dim mismatch during serialization", () => {
    const map = new Map<string, Float32Array>();
    map.set("ok", makeVec(4, 1));
    map.set("bad", makeVec(5, 2));
    expect(() => serializeEmbeddings(map, 4)).toThrow(/dim mismatch/);
  });

  it("throws on bad magic", () => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, 0xdeadbeef, true); // wrong magic
    view.setUint32(4, 1, true);
    view.setUint32(8, 4, true);
    view.setUint32(12, 0, true);
    expect(() => deserializeEmbeddings(buf)).toThrow(/bad magic/);
  });

  it("throws on unsupported version", () => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, 0x4245584e, true); // "NXEB"
    view.setUint32(4, 99, true); // unsupported version
    view.setUint32(8, 4, true);
    view.setUint32(12, 0, true);
    expect(() => deserializeEmbeddings(buf)).toThrow(/unsupported version/);
  });

  it("throws on truncated header", () => {
    const buf = new ArrayBuffer(8); // less than 16 header bytes
    expect(() => deserializeEmbeddings(buf)).toThrow(/too small/);
  });

  it("produces independent Float32Array buffers (no aliasing into the file buffer)", () => {
    const original = makeVec(4, 1);
    const map = new Map([["x", original]]);
    const buf = serializeEmbeddings(map, 4);
    const { map: out } = deserializeEmbeddings(buf);
    const got = out.get("x")!;
    // mutating the deserialized vector must not affect the source buffer.
    got[0] = 999;
    const { map: out2 } = deserializeEmbeddings(buf);
    expect(out2.get("x")![0]).not.toBe(999);
  });

  it("handles larger-than-trivial sizes (1000 entries)", () => {
    const map = new Map<string, Float32Array>();
    for (let i = 0; i < 1000; i++) map.set(`title-${i}`, makeVec(16, i));
    const buf = serializeEmbeddings(map, 16);
    const { map: out } = deserializeEmbeddings(buf);
    expect(out.size).toBe(1000);
    // spot-check a few entries at known indices.
    for (const i of [0, 1, 500, 999]) {
      expect(Array.from(out.get(`title-${i}`)!)).toEqual(Array.from(makeVec(16, i)));
    }
  });
});
