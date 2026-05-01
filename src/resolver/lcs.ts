// longest Common Subsequence length between two strings.
// uses single-row DP optimization: O(n·m) time, O(min(n,m)) space.
// expects pre-normalized (e.g. lowercased) input — no case folding is applied.
export function lcsLength(a: string, b: string): number {
  // ensure b is the shorter string for space optimization
  const [short, long] = a.length < b.length ? [a, b] : [b, a];
  const m = short.length;
  const n = long.length;

  const prev = new Uint16Array(m + 1);

  for (let i = 1; i <= n; i++) {
    let prevDiag = 0;
    for (let j = 1; j <= m; j++) {
      const temp = prev[j];
      if (long[i - 1] === short[j - 1]) {
        prev[j] = prevDiag + 1;
      } else {
        prev[j] = Math.max(prev[j], prev[j - 1]);
      }
      prevDiag = temp;
    }
  }

  return prev[m];
}

// normalized similarity in [0, 1] using LCS.
// returns lcsLength(a,b) / max(a.length, b.length), or 0 for empty inputs.
export function normalizedSimilarity(a: string, b: string): number {
  if (a.length === 0 || b.length === 0) return 0;
  return lcsLength(a, b) / Math.max(a.length, b.length);
}
