# Plan: YAKE-lite Keyphrase Extraction

## Context

Implementing the keyphrase extraction module for Nexus. The goal is a lightweight, from-scratch TypeScript implementation of YAKE tailored for Obsidian note content. The thesis proposal emphasizes positional/contextual cues over frequency, and notes that "keyphrase frequency is not a reliable signal for link relevance in personal notes."

Reference: [Rust YAKE implementation walkthrough](https://dev.to/tugascript/rust-keyword-extraction-creating-the-yake-algorithm-from-scratch-4n2l)

## Full YAKE has 5 features — here's what we keep and why

| # | YAKE Feature | Keep? | Rationale |
|---|-------------|-------|-----------|
| 1 | **Casing** (`max(TF_U, TF_C) / (1 + log(TF))`) | **Yes** | Capitalized/uppercase words in notes often signal proper nouns, concepts, or note titles — exactly what we want to link. |
| 2 | **Position** (`log(3 + median(sentenceIndices))`) | **Yes** | Thesis explicitly calls this out. First sentences of a note typically state its topic. |
| 3 | **Frequency** (`TF / (mean(allTF) + std(allTF))`) | **Downweighted** | Thesis says frequency is unreliable in personal notes. Keep it in the formula but reduce its influence (multiply by a dampening factor, e.g. 0.5). |
| 4 | **Relatedness** (`1 + ((WDL/WIL + WDR/WIR) * (TF/maxTF))`) | **Yes** | This is the "contextual cue" the thesis highlights. Words co-occurring with many different neighbors are likely stopword-ish; words with focused context are meaningful. |
| 5 | **DifSentence** (`uniqueSentences / totalOccurrences`) | **Yes** | Cheap to compute and helps distinguish topical terms from repeated filler. |

### YAKE-lite term score (modified from original)

```
H = (Relatedness × Position) / (Casing + (α × Frequency / Relatedness) + (DifSentence / Relatedness))
```

Where `α = 0.5` is the frequency dampening factor (configurable).

### N-gram scoring (kept from original)

```
S(kw) = ∏(H_i) / (TF_kw × (1 + ∑(H_i)))
```

Product of constituent term scores divided by n-gram frequency times (1 + sum of term scores). Lower S = more important.

### What we skip entirely
- **Levenshtein deduplication** — not needed now; the alias resolver downstream already handles fuzzy matching
- **POS tagging** — thesis says "if time allows"; skip for initial implementation, add later as a filter

## Preprocessing pipeline (Obsidian-aware)

1. **Strip YAML frontmatter** (`---...---` block at top)
2. **Strip wikilinks** — remove `[[...]]` syntax so existing links don't become candidates
3. **Strip markdown formatting** — headings `#`, bold `**`, italic `*`, code blocks, URLs
4. **Sentence splitting** — split on `.!?` followed by whitespace or newline; also treat `\n\n` as sentence boundary
5. **Tokenization** — split on whitespace + punctuation; normalize to lowercase for matching but preserve original case for the Casing feature
6. **Stopword filtering** — small built-in English stopword list (~175 words)

## Output format

Matches existing `ExtractedPhrase` interface:
```ts
{ phrase: string, score: number, startOffset: number, endOffset: number, spanId: string }
```

- `startOffset`/`endOffset` — character positions in the **original** (pre-stripped) content, so wikilinks can be inserted accurately
- `spanId` — deterministic ID: `"${startOffset}:${endOffset}"`
- `score` — lower = more important (YAKE convention), normalized to [0, 1]

## Files to create/modify

- `src/keyphrase/yake-lite.ts` — main extractor class (rewrite stub)
- `src/keyphrase/preprocessing.ts` — markdown stripping, sentence splitting, tokenization
- `src/keyphrase/stopwords.ts` — English stopword list
- `src/__tests__/yake-lite.test.ts` — unit tests
- `src/__tests__/preprocessing.test.ts` — preprocessing tests

## Configuration

```ts
interface YakeLiteOptions {
  maxNgramSize?: number;      // default 3 (unigrams, bigrams, trigrams)
  windowSize?: number;        // co-occurrence window, default 3
  topN?: number;              // max phrases to return, default 20
  frequencyDampening?: number; // α factor, default 0.5 — CONFIGURABLE per user request
}
```

## Decisions confirmed
- **N-gram size**: up to 3 (unigrams, bigrams, trigrams)
- **Frequency dampening α**: exposed as a configurable option (default 0.5)

## Verification

1. `npm test` — all new + existing tests pass
2. Manual test: feed a real Obsidian note's content into `YakeLite.extract()` and verify the returned phrases are sensible candidates for linking
