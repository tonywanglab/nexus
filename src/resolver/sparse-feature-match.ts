export function sparseCosine(
  a: { indices: number[]; values: number[] },
  b: { indices: number[]; values: number[] },
): number {
  if (a.indices.length === 0 || b.indices.length === 0) return 0;
  const bMap = new Map<number, number>();
  for (let i = 0; i < b.indices.length; i++) bMap.set(b.indices[i], b.values[i]);
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.indices.length; i++) {
    const bv = bMap.get(a.indices[i]);
    if (bv !== undefined) dot += a.values[i] * bv;
    normA += a.values[i] * a.values[i];
  }
  for (let i = 0; i < b.indices.length; i++) normB += b.values[i] * b.values[i];
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom < 1e-12 ? 0 : dot / denom;
}
