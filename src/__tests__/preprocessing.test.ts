import { stripMarkdown, splitSentences, tokenize, preprocess } from "../keyphrase/preprocessing";

describe("stripMarkdown", () => {
  it("strips YAML frontmatter", () => {
    const content = "---\ntitle: Test\ntags: [a, b]\n---\nHello world";
    const { stripped } = stripMarkdown(content);
    expect(stripped).toBe("Hello world");
  });

  it("strips wikilinks and keeps display text", () => {
    const { stripped } = stripMarkdown("See [[Some Note]] for details");
    expect(stripped).toBe("See Some Note for details");
  });

  it("strips wikilinks with alias and keeps alias", () => {
    const { stripped } = stripMarkdown("See [[Some Note|the note]] for details");
    expect(stripped).toBe("See the note for details");
  });

  it("strips markdown links and keeps text", () => {
    const { stripped } = stripMarkdown("Visit [Google](https://google.com) now");
    expect(stripped).toBe("Visit Google now");
  });

  it("strips images entirely", () => {
    const { stripped } = stripMarkdown("Before ![alt](img.png) after");
    expect(stripped).toBe("Before  after");
  });

  it("strips fenced code blocks", () => {
    const { stripped } = stripMarkdown("Before\n```js\nconsole.log('hi')\n```\nAfter");
    expect(stripped).toBe("Before\n\nAfter");
  });

  it("strips inline code", () => {
    const { stripped } = stripMarkdown("Use `const x = 1` here");
    expect(stripped).toBe("Use  here");
  });

  it("strips heading markers", () => {
    const { stripped } = stripMarkdown("## My Heading\nSome text");
    expect(stripped).toBe("My Heading\nSome text");
  });

  it("strips bold and italic markers", () => {
    const { stripped } = stripMarkdown("This is **bold** and *italic*");
    expect(stripped).toBe("This is bold and italic");
  });

  it("strips bare URLs", () => {
    const { stripped } = stripMarkdown("Check https://example.com/path for info");
    expect(stripped).toBe("Check  for info");
  });

  it("preserves offset mapping", () => {
    const content = "# Title\nSome text";
    const { stripped, offsetMap } = stripMarkdown(content);
    expect(stripped).toBe("Title\nSome text");
    // 'T' in stripped is at index 0, should map to index 2 in original
    expect(offsetMap[0]).toBe(2);
  });
});

describe("splitSentences", () => {
  it("splits on period followed by space", () => {
    const spans = splitSentences("First sentence. Second sentence.");
    expect(spans).toHaveLength(2);
  });

  it("splits on exclamation and question marks", () => {
    const spans = splitSentences("Wow! Really? Yes.");
    expect(spans).toHaveLength(3);
  });

  it("splits on double newlines", () => {
    const spans = splitSentences("Paragraph one\n\nParagraph two");
    expect(spans).toHaveLength(2);
  });

  it("handles trailing text without punctuation", () => {
    const spans = splitSentences("Hello world");
    expect(spans).toHaveLength(1);
  });

  it("skips empty segments", () => {
    const spans = splitSentences("Hello.\n\n\n\nWorld.");
    expect(spans).toHaveLength(2);
  });
});

describe("tokenize", () => {
  it("splits on whitespace and punctuation", () => {
    const tokens = tokenize("Hello, world!", 0);
    expect(tokens).toHaveLength(2);
    expect(tokens[0].lower).toBe("hello");
    expect(tokens[1].lower).toBe("world");
  });

  it("preserves original case", () => {
    const tokens = tokenize("TypeScript API", 0);
    expect(tokens[0].original).toBe("TypeScript");
    expect(tokens[1].original).toBe("API");
  });

  it("tracks offsets relative to spanStart", () => {
    const tokens = tokenize("hello world", 10);
    expect(tokens[0].startOffset).toBe(10);
    expect(tokens[0].endOffset).toBe(15);
    expect(tokens[1].startOffset).toBe(16);
    expect(tokens[1].endOffset).toBe(21);
  });

  it("handles contractions", () => {
    const tokens = tokenize("don't won't", 0);
    expect(tokens[0].lower).toBe("don't");
    expect(tokens[1].lower).toBe("won't");
  });
});

describe("preprocess (full pipeline)", () => {
  it("returns sentences with tokens mapped to original offsets", () => {
    const content = "---\ntitle: Test\n---\n# Heading\nThis is a test sentence.";
    const { sentences } = preprocess(content);

    expect(sentences.length).toBeGreaterThan(0);

    // all tokens should have valid offsets into the original content
    for (const sentence of sentences) {
      for (const token of sentence.tokens) {
        expect(token.startOffset).toBeGreaterThanOrEqual(0);
        expect(token.endOffset).toBeGreaterThan(token.startOffset);
        expect(token.endOffset).toBeLessThanOrEqual(content.length);
      }
    }
  });

  it("strips wikilinks and maps offsets correctly", () => {
    const content = "The [[Concept]] is important";
    const { sentences } = preprocess(content);

    expect(sentences.length).toBeGreaterThan(0);
    const allTokens = sentences.flatMap((s) => s.tokens);
    const conceptToken = allTokens.find((t) => t.lower === "concept");
    expect(conceptToken).toBeDefined();
    // the word "Concept" in original starts at index 6 (inside [[Concept]])
    expect(conceptToken!.startOffset).toBe(6);
  });

  it("produces sentence indices starting at 0", () => {
    const content = "First sentence. Second sentence. Third sentence.";
    const { sentences } = preprocess(content);
    expect(sentences[0].index).toBe(0);
    expect(sentences[1].index).toBe(1);
    expect(sentences[2].index).toBe(2);
  });
});
