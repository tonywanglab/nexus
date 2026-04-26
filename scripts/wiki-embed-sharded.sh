#!/usr/bin/env bash
# Line-split the wiki corpus, embed each text shard in parallel, then merge with
# scripts/merge-embedding-shards.ts (streaming line count + SHA-256, shard manifest checks).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Defaults (override with env: CORPUS, SH, OUT_BIN, OUT_MANIFEST, SHARD_N, DEVICE, BATCH, FLUSH)
CORPUS="${CORPUS:-data/wiki/corpus.txt}"
SH="${SH:-data/wiki/shards}"
OUT_BIN="${OUT_BIN:-data/wiki/embeddings.f32.bin}"
OUT_MANIFEST="${OUT_MANIFEST:-data/wiki/embeddings-manifest.json}"
SHARD_N="${SHARD_N:-4}"
# macOS: coreml; Linux: cpu
DEVICE="${DEVICE:-$([ "$(uname -s 2>/dev/null)" = Darwin ] && echo coreml || echo cpu)}"
BATCH="${BATCH:-256}"
FLUSH="${FLUSH:-10000}"

log() { echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" | tee -a "$SH/runner.log" >&2; }

if [[ ! -f "$CORPUS" ]]; then
  log "ERROR: corpus not found: $CORPUS"
  exit 1
fi

mkdir -p "$SH"

log "Splitting corpus → $SHARD_N line-based shards (streaming count + split)…"
npm run wiki:split-corpus -- --input="$CORPUS" --out-dir="$SH" --n="$SHARD_N"

log "Starting parallel shard embeds (device=$DEVICE batch=$BATCH)…"
pids=()
for c in $(ls -1 "$SH"/corpus_* 2>/dev/null | sort); do
  base=$(basename "$c")
  suf=${base#corpus_}
  out_bin="$SH/embed_${suf}.f32.bin"
  out_man="$SH/embed_${suf}-manifest.json"
  log "  shard $suf  input=$c  →  $out_bin"
  (
    npm run wiki:embed -- \
      --input="$c" \
      --output="$out_bin" \
      --manifest="$out_man" \
      --device="$DEVICE" \
      --batch-size="$BATCH" \
      --flush-every="$FLUSH" \
  ) >"$SH/stdout_${suf}.log" 2>&1 &
  pids+=($!)
done
if ((${#pids[@]} == 0)); then
  log "ERROR: no data/wiki/shards/corpus_* files after split"
  exit 1
fi
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    log "ERROR: one of the embed jobs failed (see $SH/stdout_*.log)"
    exit 1
  fi
done

log "Merging $SHARD_N shard .f32.bin files into $OUT_BIN (merge-embedding-shards)…"
npm run wiki:embed:merge-shards -- --corpus="$CORPUS" \
  --out-bin="$OUT_BIN" \
  --out-manifest="$OUT_MANIFEST" \
  --shard-dir="$SH" \
  --shard-glob="embed_*.f32.bin"

log "OK: sharded embed + merge complete → $OUT_BIN + $OUT_MANIFEST"
