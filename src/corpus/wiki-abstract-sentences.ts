/**
 * Helpers for building a sentence-level Wikipedia corpus (offline scripts + tests).
 */

const ENTITY_MAP: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
};

export function stripBasicXmlEntities(s: string): string {
  return s.replace(/&(#x?[0-9a-fA-F]+|[^;]+);/g, (full, inner) => {
    if (inner[0] === "#") {
      const hex = inner[1] === "x" || inner[1] === "X";
      const num = hex ? parseInt(inner.slice(2), 16) : parseInt(inner.slice(1), 10);
      if (!Number.isFinite(num)) return full;
      return String.fromCodePoint(num);
    }
    const named = ENTITY_MAP[inner];
    return named !== undefined ? named : full;
  });
}

/** Split on . ! ? when followed by space or end; trim fragments. */
export function splitIntoSentences(text: string): string[] {
  const out: string[] = [];
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "." || c === "!" || c === "?") {
      const next = text[i + 1];
      if (next === undefined || next === " " || next === "\n" || next === "\r" || next === "\t") {
        const piece = text.slice(start, i + 1).trim();
        if (piece) out.push(piece);
        start = i + 1;
        while (start < text.length && /\s/.test(text[start])) start++;
        i = start - 1;
      }
    }
  }
  const tail = text.slice(start).trim();
  if (tail) out.push(tail);
  return out;
}

const HAS_LETTER = /[a-zA-Z]/;

export function keepSentenceIfLengthOk(s: string, minLen: number, maxLen: number): string | null {
  const t = s.trim();
  if (t.length < minLen || t.length > maxLen) return null;
  if (!HAS_LETTER.test(t)) return null;
  return t;
}
