/**
 * Text normalization utilities for deterministic alias resolution.
 */

/**
 * Normalize text for comparison:
 * lowercase → collapse whitespace → strip possessives → strip diacritics → trim.
 */
export function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/'s\b/g, "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();
}

/**
 * Check if two strings are equal after normalization.
 */
export function exactMatch(phrase: string, title: string): boolean {
  return normalize(phrase) === normalize(title);
}

/**
 * Check if the phrase contains the title after normalization.
 * e.g. phrase="machine learning systems", title="machine learning" → true
 * This is a strong signal: the phrase is a superset of the title.
 */
export function phraseContainsTitle(phrase: string, title: string): boolean {
  const np = normalize(phrase);
  const nt = normalize(title);
  return np.includes(nt) && np !== nt;
}
