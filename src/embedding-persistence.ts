/**
 * Binary persistence for text→embedding maps (title embeddings today, phrase
 * embeddings possibly later). Skips JSON inflation of Float32 arrays and
 * keeps load/save cheap for vaults with thousands of titles.
 *
 * Format (little-endian):
 *   magic:  "NXEB" (4 bytes)
 *   version: u32 (currently 1)
 *   dim: u32
 *   count: u32
 *   entries[count]:
 *     textLen: u32 (bytes of UTF-8)
 *     textBytes: u8[textLen]
 *     vec: f32[dim]
 */

const MAGIC = 0x4245584e; // "NXEB" little-endian
const VERSION = 1;

export function serializeEmbeddings(map: Map<string, Float32Array>, dim: number): ArrayBuffer {
  const encoder = new TextEncoder();
  const entries = [...map.entries()].map(([text, vec]) => {
    if (vec.length !== dim) {
      throw new Error(`embedding dim mismatch: expected ${dim}, got ${vec.length} for "${text}"`);
    }
    return { bytes: encoder.encode(text), vec };
  });

  let total = 16; // header
  for (const e of entries) total += 4 + e.bytes.length + dim * 4;

  const buf = new ArrayBuffer(total);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  let o = 0;
  view.setUint32(o, MAGIC, true); o += 4;
  view.setUint32(o, VERSION, true); o += 4;
  view.setUint32(o, dim, true); o += 4;
  view.setUint32(o, entries.length, true); o += 4;

  for (const e of entries) {
    view.setUint32(o, e.bytes.length, true); o += 4;
    bytes.set(e.bytes, o); o += e.bytes.length;
    const vecBytes = new Uint8Array(e.vec.buffer, e.vec.byteOffset, dim * 4);
    bytes.set(vecBytes, o); o += dim * 4;
  }

  return buf;
}

export interface DeserializedEmbeddings {
  dim: number;
  map: Map<string, Float32Array>;
}

export function deserializeEmbeddings(buf: ArrayBuffer): DeserializedEmbeddings {
  if (buf.byteLength < 16) throw new Error("embedding file too small");
  const view = new DataView(buf);
  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) throw new Error("embedding file: bad magic");
  const version = view.getUint32(4, true);
  if (version !== VERSION) throw new Error(`embedding file: unsupported version ${version}`);
  const dim = view.getUint32(8, true);
  const count = view.getUint32(12, true);

  const decoder = new TextDecoder();
  const map = new Map<string, Float32Array>();
  let o = 16;
  for (let i = 0; i < count; i++) {
    const textLen = view.getUint32(o, true); o += 4;
    const text = decoder.decode(new Uint8Array(buf, o, textLen)); o += textLen;
    // Copy into a fresh Float32Array — slicing the underlying buffer keeps the
    // mmap-style reference alive and misaligned reads can fail on some runtimes.
    const vec = new Float32Array(dim);
    const src = new Float32Array(buf.slice(o, o + dim * 4));
    vec.set(src);
    o += dim * 4;
    map.set(text, vec);
  }
  return { dim, map };
}
