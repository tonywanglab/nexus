import {
  stripBasicXmlEntities,
  splitIntoSentences,
  keepSentenceIfLengthOk,
} from "../corpus/wiki-abstract-sentences";

describe("stripBasicXmlEntities", () => {
  it("decodes common entities", () => {
    expect(stripBasicXmlEntities("a&amp;b &lt; c &gt; &quot;x&quot; &apos;y&apos;")).toBe(
      "a&b < c > \"x\" 'y'",
    );
  });
});

describe("splitIntoSentences", () => {
  it("splits on punctuation boundaries", () => {
    expect(splitIntoSentences("First. Second! Third?")).toEqual(["First.", "Second!", "Third?"]);
  });

  it("strips whitespace", () => {
    expect(splitIntoSentences("  A.   B  ")).toEqual(["A.", "B"]);
  });
});

describe("keepSentenceIfLengthOk", () => {
  it("requires a letter and length bounds", () => {
    const ok = "abcdefghijklmnopqrstuvwxyz01234";
    expect(ok.length).toBeGreaterThanOrEqual(30);
    expect(keepSentenceIfLengthOk(ok, 30, 200)).toBe(ok);
    expect(keepSentenceIfLengthOk("short", 30, 200)).toBeNull();
    expect(keepSentenceIfLengthOk("x".repeat(201), 30, 200)).toBeNull();
    expect(keepSentenceIfLengthOk("123456789012345678901234567890", 30, 200)).toBeNull();
  });
});
