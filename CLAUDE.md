# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Nexus is an Obsidian plugin that performs **local-first automatic link discovery**. It watches vault changes, extracts keyphrases from notes (YAKE-lite), resolves them against existing note titles (LCS + embeddings), and surfaces candidate wikilinks for user approval. All processing is local — no external APIs.

## Commands

```bash
npm run dev          # esbuild watch mode → outputs to vault plugin dir (reads VAULT_PATH from .env)
npm run build        # type-check + production bundle → main.js
npm test             # run all Jest tests
npm test -- --testPathPattern=job-queue   # run a single test file
```

## Local Development

Set `VAULT_PATH` in `.env` to your Obsidian vault's root (see `.env.example`). `npm run dev` writes `main.js` + `manifest.json` to `$VAULT_PATH/.obsidian/plugins/nexus/` and watches for changes. Reload Obsidian (Cmd+R) or use the Hot Reload community plugin to pick up changes.

## Architecture

```
vault event → EventListener → JobQueue (debounced) → YakeLite → AliasResolver → SuggestionModal
```

1. **EventListener** (`src/event-listener.ts`) — subscribes to Obsidian vault `create`/`modify`/`delete`/`rename` hooks; filters to `.md` files only.
2. **JobQueue** (`src/job-queue.ts`) — per-file debounced queue. Active file gets priority (shorter delay). Cancel on delete, reindex on rename.
3. **YakeLite** (`src/keyphrase/yake-lite.ts`) — lightweight YAKE variant. Outputs `{phrase, score, startOffset, endOffset, spanId}`.
4. **AliasResolver** (`src/resolver/`) — deterministic (LCS normalization) + stochastic (local embeddings via Ollama/llama.cpp). `resolver/lcs.ts` for string similarity, `resolver/index.ts` orchestrates.
5. **SuggestionModal** (`src/ui/suggestion-modal.ts`) — Obsidian Modal for approve/reject of candidate edges.
6. **NexusPlugin** (`src/main.ts`) — plugin entry point, wires all components.

## Testing

Tests live in `src/__tests__/`. Obsidian API is mocked in `src/__tests__/__mocks__/obsidian.ts`. Jest is configured to map `obsidian` imports to this mock via `jest.config.js`. Job queue tests use `jest.useFakeTimers()` for deterministic debounce testing.

## Key Types (`src/types.ts`)

- `QueueJob` — `{filePath, type, priority, enqueuedAt}`
- `ExtractedPhrase` — `{phrase, score, startOffset, endOffset, spanId}`
- `CandidateEdge` — `{sourcePath, phrase, targetPath, similarity, approved?}`
