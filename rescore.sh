#!/usr/bin/env bash
# Form 2 (GPU): regenerate the per-item predictions from the models, then reproduce.
# Scores the 11 models at FP16/int8/NF4 over the cultural and control items with
# deterministic constrained log-likelihood and 5 cyclic option rotations, writing
# the same per-group prediction files the analysis consumes. Resumable: a completed
# (model, precision) file is skipped. After scoring, runs reproduce.sh.
#
# Needs the project environment (uv sync) and a >=16 GB GPU. Predictions land in
# data/predictions/{cult,ctrl}/, overwriting the committed run of record by default;
# set OUT to write elsewhere and CQ_OUT to point reproduce.sh at it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/data/predictions}"

MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B \
  Qwen/Qwen3-4B TucanoBR/Tucano-1b1 TucanoBR/Tucano-2b4 microsoft/Phi-3.5-mini-instruct \
  mistralai/Mistral-7B-Instruct-v0.3}"
PRECISIONS="${PRECISIONS:-fp16 int8 nf4}"

for grp in cultural control; do
  sub=$([ "$grp" = cultural ] && echo cult || echo ctrl)
  echo "== scoring $grp -> $OUT/$sub =="
  uv run python -m culturaquant.score_cyclic \
    --models $MODELS --precisions $PRECISIONS \
    --items-file "$HERE/data/items/$grp.jsonl" --group "$grp" \
    --out-dir "$OUT/$sub"
done

echo "== re-running analysis on the fresh predictions =="
CQ_OUT="$OUT" "$HERE/reproduce.sh"
