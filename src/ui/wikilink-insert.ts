export interface WikilinkResult {
  content: string;
  replaced: boolean;
  reason?: "span-drift" | "already-linked";
}

// attempts to replace the phrase at [startOffset, endOffset) with a wikilink.
//
// guards:
// - Verifies the span still contains the phrase text (catches drift after edits).
// - Detects if the span is already inside [[...]] or [text](url) and short-circuits.
//
// alias rule: if phrase (case-insensitive) equals targetTitle, use [[Title]].
// otherwise use [[Title|phrase]].
export function buildWikilinkReplacement(
  content: string,
  phrase: { phrase: string; startOffset: number; endOffset: number },
  targetTitle: string,
): WikilinkResult {
  const { phrase: text, startOffset: start, endOffset: end } = phrase;

  // 1. Verify span still matches.
  const span = content.slice(start, end);
  if (span !== text) {
    return { content, replaced: false, reason: "span-drift" };
  }

  // 2. Detect if span is inside an existing wikilink [[...]].
  //    The preprocessor keeps the display text but strips [[ ]], so the span
  //    points inside a wikilink if the two chars before start are "[[".
  if (start >= 2 && content.slice(start - 2, start) === "[[") {
    return { content, replaced: false, reason: "already-linked" };
  }
  // also check if ]] follows the span end.
  if (content.slice(end, end + 2) === "]]") {
    return { content, replaced: false, reason: "already-linked" };
  }

  // 3. Detect if span is inside a markdown link [text](url).
  //    The preprocessor keeps "text" and strips [...](...), so the char before
  //    start is "[" and somewhere after end we see "](" before a newline.
  if (start >= 1 && content[start - 1] === "[") {
    // look forward from end for ]( within the same line.
    const lineEnd = content.indexOf("\n", end);
    const searchTo = lineEnd === -1 ? content.length : lineEnd;
    const after = content.slice(end, searchTo);
    if (after.startsWith("](")) {
      return { content, replaced: false, reason: "already-linked" };
    }
  }

  // 4. Build replacement wikilink.
  const phraseNormalized = text.toLowerCase().trim();
  const titleNormalized = targetTitle.toLowerCase().trim();
  const wikilink =
    phraseNormalized === titleNormalized
      ? `[[${targetTitle}]]`
      : `[[${targetTitle}|${text}]]`;

  const newContent = content.slice(0, start) + wikilink + content.slice(end);
  return { content: newContent, replaced: true };
}
