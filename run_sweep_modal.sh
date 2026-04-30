#!/bin/bash
# SAE k-sweep on Modal: train + label all configs in parallel.
# All 7 configs submit simultaneously — Modal provisions GPUs automatically.
#
# Usage:
#   tmux new -s sweep
#   ./run_sweep_modal.sh

set -euo pipefail

VOCAB="data/vocab.txt"
VOCAB_EMB="data/vocab-embeddings.f32.bin"
D_HIDDEN=16128  # 21x expansion (768 * 21)
EPOCHS=25
LR=0.0003
BATCH=256

PYTHON=$(command -v python3 || command -v python || true)
MODAL=$(command -v modal || true)

# ── Preflight checks ──────────────────────────────────────────────────────────

echo "=== Preflight checks ==="

# Python
if [ -z "$PYTHON" ]; then
  echo "ERROR: python3 not found." >&2; exit 1
fi
echo "Python: $($PYTHON --version)"

# Modal CLI
if [ -z "$MODAL" ]; then
  echo "ERROR: modal CLI not found. Install with: pip install modal" >&2; exit 1
fi
echo "Modal: $($MODAL --version)"

# Modal auth
if ! $MODAL profile current &>/dev/null; then
  echo "ERROR: not authenticated with Modal. Run: modal token new" >&2; exit 1
fi
echo "Modal auth: OK ($(modal profile current))"

# Required scripts
for f in scripts/modal_train_sae_v2.py scripts/modal_autointerp_vocab.py; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found" >&2; exit 1
  fi
done
echo "Scripts: OK"

# Required data files
for f in "$VOCAB" "$VOCAB_EMB"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found" >&2; exit 1
  fi
done
echo "Data files: OK"

echo ""
echo "=== All checks passed — submitting sweep to Modal ==="
echo ""

mkdir -p logs

# ── Per-config runner ─────────────────────────────────────────────────────────

run_config() {
  local k=$1
  local out="modal-out-k${k}-21x"

  echo "[k=$k] $(date '+%H:%M:%S') Training..."
  $MODAL run scripts/modal_train_sae_v2.py \
    --vocab "$VOCAB" \
    --out-dir "./$out" \
    --k "$k" \
    --d-hidden "$D_HIDDEN" \
    --epochs "$EPOCHS" \
    --batch-size-train "$BATCH" \
    --lr "$LR"

  if [ ! -f "$out/sae-weights.bin" ]; then
    echo "[k=$k] ERROR: sae-weights.bin not produced — skipping labeling" >&2
    return 1
  fi

  echo "[k=$k] $(date '+%H:%M:%S') Labeling..."
  $MODAL run scripts/modal_autointerp_vocab.py \
    --weights "$out/sae-weights.bin" \
    --vocab "$VOCAB" \
    --vocab-embeddings "$VOCAB_EMB" \
    --out "$out/sae-feature-labels.json"

  echo "[k=$k] $(date '+%H:%M:%S') Done"
}

export -f run_config
export VOCAB VOCAB_EMB D_HIDDEN EPOCHS LR BATCH PYTHON MODAL

# ── Submit all configs in parallel ────────────────────────────────────────────
# Modal provisions its own GPUs — no local GPU assignment needed.

for k in 16 32 48 96 128; do
  (
    set -euo pipefail
    run_config "$k"
  ) &> "logs/k${k}.txt" &
  echo "  k=$k submitted (PID $!)"
done

echo ""
echo "All 5 configs running on Modal."
echo "Monitor with:  tail -f logs/k<N>.txt"
echo ""

# ── Wait and report ───────────────────────────────────────────────────────────

wait
FAILED=()
for k in 16 32 48 96 128; do
  if [ -f "modal-out-k${k}-21x/sae-weights.bin" ]; then
    echo "k=$k: done"
  else
    echo "k=$k: FAILED — check logs/k${k}.txt" >&2
    FAILED+=($k)
  fi
done

echo ""
echo "=== Sweep complete ==="
echo ""
echo "Results:"
for k in 16 32 48 96 128; do
  out="modal-out-k${k}-21x"
  if [ -f "$out/sae-feature-labels.json" ]; then
    live=$($PYTHON -c "
import json
d = json.load(open('$out/sae-feature-labels.json'))
live = sum(1 for l in d['labels'] if l['candidates'])
print(f\"{live}/{d['dHidden']} labeled ({100*live/d['dHidden']:.1f}%)\")
" 2>/dev/null || echo "?")
    mse=$($PYTHON -c "
import json
d = json.load(open('$out/sae-training-report.json'))
print(f\"{d['finalValMSE']:.3e}\")
" 2>/dev/null || echo "?")
    echo "  k=$k: valMSE=$mse  labels=$live"
  else
    echo "  k=$k: MISSING"
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "Failed configs: ${FAILED[*]}"
  exit 1
fi
