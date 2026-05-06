import { buildWikilinkReplacement } from "../ui/wikilink-insert";

describe("buildWikilinkReplacement", () => {
  describe("span match", () => {
    it("replaces phrase with [[Target]] when phrase equals title (case-insensitive)", () => {
      const content = "I love neural networks today.";
      const phrase = { phrase: "neural networks", startOffset: 7, endOffset: 22 };
      const result = buildWikilinkReplacement(content, phrase, "Neural Networks");
      expect(result.replaced).toBe(true);
      expect(result.content).toBe("I love [[Neural Networks]] today.");
    });

    it("replaces phrase with [[Target|phrase]] when phrase differs from title", () => {
      const content = "I love neural nets today.";
      const phrase = { phrase: "neural nets", startOffset: 7, endOffset: 18 };
      const result = buildWikilinkReplacement(content, phrase, "Neural Networks");
      expect(result.replaced).toBe(true);
      expect(result.content).toBe("I love [[Neural Networks|neural nets]] today.");
    });
  });

  describe("span drift", () => {
    it("returns replaced=false when offsets no longer match phrase text", () => {
      const content = "Something else entirely here.";
      const phrase = { phrase: "neural nets", startOffset: 0, endOffset: 11 };
      const result = buildWikilinkReplacement(content, phrase, "Neural Networks");
      expect(result.replaced).toBe(false);
      expect(result.content).toBe(content);
    });
  });

  describe("already-linked guard", () => {
    it("returns already-linked when span is inside [[...]]", () => {
      // "See [[neural nets]] for more."
      //  0123456789012345678
      // "neural nets" starts at 6, ends at 17
      const content = "See [[neural nets]] for more.";
      const phrase = { phrase: "neural nets", startOffset: 6, endOffset: 17 };
      const result = buildWikilinkReplacement(content, phrase, "Neural Networks");
      expect(result.replaced).toBe(false);
      expect(result.reason).toBe("already-linked");
      expect(result.content).toBe(content);
    });

    it("returns already-linked when span is inside [text](url) markdown link", () => {
      // "See [neural nets](http://example.com) for more."
      //  01234567890123456
      // "neural nets" starts at 5, ends at 16
      const content = "See [neural nets](http://example.com) for more.";
      const phrase = { phrase: "neural nets", startOffset: 5, endOffset: 16 };
      const result = buildWikilinkReplacement(content, phrase, "Neural Networks");
      expect(result.replaced).toBe(false);
      expect(result.reason).toBe("already-linked");
    });

    it("replaces normally when phrase is adjacent to but outside a wikilink", () => {
      // "[[Foo]] neural nets are cool."
      //  01234567890123456789
      // "neural nets" starts at 8, ends at 19
      const content = "[[Foo]] neural nets are cool.";
      const phrase = { phrase: "neural nets", startOffset: 8, endOffset: 19 };
      const result = buildWikilinkReplacement(content, phrase, "Neural Networks");
      expect(result.replaced).toBe(true);
      expect(result.content).toBe("[[Foo]] [[Neural Networks|neural nets]] are cool.");
    });
  });

  describe("multiple occurrences", () => {
    it("replaces only the occurrence at the specified offsets", () => {
      const content = "neural nets first, neural nets second.";
      // Offset points to the second "neural nets" at position 19
      const phrase = { phrase: "neural nets", startOffset: 19, endOffset: 30 };
      const result = buildWikilinkReplacement(content, phrase, "Neural Networks");
      expect(result.replaced).toBe(true);
      expect(result.content).toBe("neural nets first, [[Neural Networks|neural nets]] second.");
    });
  });
});
