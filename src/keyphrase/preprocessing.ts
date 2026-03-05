/**
 * Obsidian-aware preprocessing for YAKE-lite.
 *
 * Strips markdown/wiki syntax while tracking character offsets so
 * extracted phrases can be mapped back to the original content.
 */

/** A token with its position in the original (pre-stripped) content. */
export interface Token {
  /** Lowercased form for matching / scoring. */
  lower: string;
  /** Original form (preserves case for the Casing feature). */
  original: string;
  /** Character offset in the original content. */
  startOffset: number;
  /** Character offset (exclusive) in the original content. */
  endOffset: number;
  /** Structural importance: 0 = body, 1 = bold, 2 = heading, 3 = title. */
  structuralBoost: number;
}

/** A character range in the original content carrying a structural boost. */
export interface StructuralRange {
  start: number;
  end: number;
  boost: number;
}

/** A sentence is a list of tokens plus its index (0-based). */
export interface Sentence {
  index: number;
  tokens: Token[];
}

// ── Structural detection ─────────────────────────────────────

/**
 * Scans original markdown content and returns character ranges that
 * carry structural importance boosts:
 * - First `# ` heading (title) → boost 3
 * - `##`+ headings → boost 2
 * - `**…**` bold spans → boost 1
 */
export function detectStructuralRanges(content: string): StructuralRange[] {
  const ranges: StructuralRange[] = [];
  let seenTitle = false;

  const lines = content.split("\n");
  let offset = 0;

  for (const line of lines) {
    const lineEnd = offset + line.length;

    // Heading detection
    const headingMatch = line.match(/^(#{1,6})\s+(.*)/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const textStart = offset + headingMatch[1].length + 1; // skip "# "
      const textEnd = lineEnd;
      if (level === 1 && !seenTitle) {
        ranges.push({ start: textStart, end: textEnd, boost: 3 });
        seenTitle = true;
      } else {
        ranges.push({ start: textStart, end: textEnd, boost: 2 });
      }
    }

    // Bold detection within the line
    const boldRe = /\*\*(.+?)\*\*/g;
    let boldMatch: RegExpExecArray | null;
    while ((boldMatch = boldRe.exec(line)) !== null) {
      const boldStart = offset + boldMatch.index;
      const boldEnd = boldStart + boldMatch[0].length;
      ranges.push({ start: boldStart, end: boldEnd, boost: 1 });
    }

    offset = lineEnd + 1; // +1 for the \n
  }

  return ranges;
}

/**
 * Returns the maximum structural boost for a character range in the original content.
 */
export function getBoostForRange(
  start: number,
  end: number,
  ranges: StructuralRange[],
): number {
  let maxBoost = 0;
  for (const r of ranges) {
    if (start < r.end && end > r.start) {
      maxBoost = Math.max(maxBoost, r.boost);
    }
  }
  return maxBoost;
}

// ── Stripping ────────────────────────────────────────────────

/**
 * Strips Obsidian / Markdown syntax from content.
 * Returns a character-level mapping so offsets in the stripped text
 * can be converted back to offsets in the original.
 *
 * What gets stripped:
 * - YAML frontmatter (`---…---` at the top)
 * - Wikilinks `[[target|display]]` → keeps display text (or target if no alias)
 * - Markdown links `[text](url)` → keeps text
 * - Images `![alt](url)` → removed entirely
 * - Code blocks (fenced ``` and inline `)
 * - Bold / italic markers (`**`, `*`, `__`, `_`)
 * - Heading markers (`# `)
 * - URLs (bare http/https)
 * - HTML tags
 */
export function stripMarkdown(content: string): { stripped: string; offsetMap: number[] } {
  const offsetMap: number[] = [];
  let stripped = "";

  // Each replacement returns the text to keep and the offset within the
  // full match where that kept text starts (so offset mapping is accurate).
  const replacements: Array<{
    pattern: RegExp;
    replace: (match: RegExpMatchArray) => { text: string; offsetInMatch: number };
  }> = [
    // YAML frontmatter (only at very start)
    { pattern: /^---\n[\s\S]*?\n---\n?/, replace: () => ({ text: "", offsetInMatch: 0 }) },
    // Fenced code blocks
    { pattern: /^```[\s\S]*?```/m, replace: () => ({ text: "", offsetInMatch: 0 }) },
    // Inline code
    { pattern: /`[^`\n]+`/, replace: () => ({ text: "", offsetInMatch: 0 }) },
    // Images
    { pattern: /!\[[^\]]*\]\([^)]*\)/, replace: () => ({ text: "", offsetInMatch: 0 }) },
    // Wikilinks with alias [[target|display]]
    {
      pattern: /\[\[([^\]|]+)\|([^\]]+)\]\]/,
      replace: (m) => ({ text: m[2], offsetInMatch: m[0].indexOf(m[2]) }),
    },
    // Wikilinks plain [[target]]
    {
      pattern: /\[\[([^\]]+)\]\]/,
      replace: (m) => ({ text: m[1], offsetInMatch: 2 }), // skip [[
    },
    // Markdown links [text](url)
    {
      pattern: /\[([^\]]*)\]\([^)]*\)/,
      replace: (m) => ({ text: m[1], offsetInMatch: 1 }), // skip [
    },
    // HTML tags
    { pattern: /<[^>]+>/, replace: () => ({ text: "", offsetInMatch: 0 }) },
    // Bare URLs
    { pattern: /https?:\/\/[^\s)]+/, replace: () => ({ text: "", offsetInMatch: 0 }) },
    // Heading markers (start of line)
    { pattern: /^#{1,6}\s/m, replace: () => ({ text: "", offsetInMatch: 0 }) },
    // Bold/italic markers
    { pattern: /(\*{1,3}|_{1,2})/, replace: () => ({ text: "", offsetInMatch: 0 }) },
  ];

  let i = 0;
  outer: while (i < content.length) {
    // Try each pattern at the current position
    const remaining = content.slice(i);
    for (const { pattern, replace } of replacements) {
      // Only match at the start of remaining
      const anchored = new RegExp("^(?:" + pattern.source + ")", pattern.flags);
      const m = remaining.match(anchored);
      if (m && m.index === 0) {
        const { text: replacementText, offsetInMatch } = replace(m);
        // Map each character of the replacement to its original offset
        for (let j = 0; j < replacementText.length; j++) {
          offsetMap.push(i + offsetInMatch + j);
          stripped += replacementText[j];
        }
        i += m[0].length;
        continue outer;
      }
    }
    // No pattern matched — keep the character
    offsetMap.push(i);
    stripped += content[i];
    i++;
  }

  return { stripped, offsetMap };
}

// ── Sentence splitting ───────────────────────────────────────

/**
 * Splits stripped text into sentences.
 *
 * Boundaries:
 * - `.` `!` `?` followed by whitespace or end of string
 * - Double newlines (`\n\n`)
 */
export function splitSentences(text: string): Array<{ start: number; end: number }> {
  const boundaries: Array<{ start: number; end: number }> = [];
  let sentenceStart = 0;

  // Skip leading whitespace
  while (sentenceStart < text.length && /\s/.test(text[sentenceStart])) {
    sentenceStart++;
  }

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];

    // Double newline boundary
    if (ch === "\n" && i + 1 < text.length && text[i + 1] === "\n") {
      if (i > sentenceStart) {
        boundaries.push({ start: sentenceStart, end: i });
      }
      // Skip all contiguous newlines/whitespace
      i += 2;
      while (i < text.length && /\s/.test(text[i])) i++;
      sentenceStart = i;
      i--; // the loop will increment
      continue;
    }

    // Sentence-ending punctuation followed by whitespace or end
    if ((ch === "." || ch === "!" || ch === "?") &&
        (i + 1 >= text.length || /\s/.test(text[i + 1]))) {
      if (i >= sentenceStart) {
        boundaries.push({ start: sentenceStart, end: i + 1 });
      }
      // Skip whitespace to find next sentence start
      let next = i + 1;
      while (next < text.length && /\s/.test(text[next])) next++;
      sentenceStart = next;
    }
  }

  // Remaining text is a sentence
  if (sentenceStart < text.length) {
    // Trim trailing whitespace
    let end = text.length;
    while (end > sentenceStart && /\s/.test(text[end - 1])) end--;
    if (end > sentenceStart) {
      boundaries.push({ start: sentenceStart, end });
    }
  }

  return boundaries;
}

// ── Tokenization ─────────────────────────────────────────────

/**
 * Tokenizes a sentence span into words.
 * Splits on whitespace and punctuation boundaries.
 * Each token retains its offset into the stripped text.
 */
export function tokenize(text: string, spanStart: number): Token[] {
  const tokens: Token[] = [];
  // Match sequences of word characters (letters, digits, apostrophes within words)
  const wordRe = /[a-zA-Z0-9]+(?:['\u2019][a-zA-Z]+)*/g;
  let m: RegExpExecArray | null;
  while ((m = wordRe.exec(text)) !== null) {
    tokens.push({
      lower: m[0].toLowerCase(),
      original: m[0],
      startOffset: spanStart + m.index,
      endOffset: spanStart + m.index + m[0].length,
      structuralBoost: 0,
    });
  }
  return tokens;
}

// ── Full pipeline ────────────────────────────────────────────

/**
 * Runs the full preprocessing pipeline:
 * strip markdown → split sentences → tokenize.
 *
 * Returns sentences with tokens whose offsets point into the
 * **original** (pre-stripped) content.
 */
export function preprocess(content: string): {
  sentences: Sentence[];
  offsetMap: number[];
  structuralRanges: StructuralRange[];
  stripped: string;
} {
  const structuralRanges = detectStructuralRanges(content);
  const { stripped, offsetMap } = stripMarkdown(content);
  const sentenceSpans = splitSentences(stripped);

  const sentences: Sentence[] = sentenceSpans.map((span, idx) => {
    const sentenceText = stripped.slice(span.start, span.end);
    const rawTokens = tokenize(sentenceText, span.start);

    // Remap offsets from stripped-space to original-space
    const tokens: Token[] = rawTokens.map((t) => {
      const origStart = offsetMap[t.startOffset] ?? t.startOffset;
      const origEnd = offsetMap[t.endOffset - 1] !== undefined
        ? offsetMap[t.endOffset - 1] + 1
        : t.endOffset;
      return {
        ...t,
        startOffset: origStart,
        endOffset: origEnd,
        structuralBoost: getBoostForRange(origStart, origEnd, structuralRanges),
      };
    });

    return { index: idx, tokens };
  });

  return { sentences, offsetMap, structuralRanges, stripped };
}
