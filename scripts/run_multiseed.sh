#!/usr/bin/env bash
# Detached, resumable multi-seed grid runner for ONE GPU host.
# Each (model x precision x seed) unit is resumable via the embedded unit-hash;
# re-running skips completed units. One model is loaded at a time. Outputs go to
# per-seed dirs under $CQ_DATA/out/multiseed/s<SEED>/ so seeds never collide.
#
# Env:
#   CQ_DATA   bind-mount data dir (default ~/cq_data)
#   IMAGE     docker image (default culturaquant:1.0)
#   SEEDS     space-separated seeds (default the 8-seed set)
#   MODELS    space-separated model ids (default all 8)
#   NPROV     proverbs to sample (default 80)
#   TAG       label for the log file
set -uo pipefail

IMAGE="${IMAGE:-culturaquant:1.0}"
CQ_DATA="${CQ_DATA:-$HOME/cq_data}"
NPROV="${NPROV:-80}"
TAG="${TAG:-$(hostname)}"
SEEDS="${SEEDS:-20260607 11 22 33 44 55 66 77}"
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B \
TucanoBR/Tucano-1b1 TucanoBR/Tucano-2b4}"
PRECISIONS="${PRECISIONS:-fp16 int8 nf4}"

mkdir -p "$CQ_DATA/out/multiseed" "$CQ_DATA/hf"
LOG="$CQ_DATA/multiseed_${TAG}.log"
echo "[start] $(date -Is) host=$(hostname) image=$IMAGE seeds=[$SEEDS] nprov=$NPROV" >> "$LOG"

for SEED in $SEEDS; do
  OUT="multiseed/s${SEED}"
  echo "[seed $SEED] $(date -Is) -> $OUT" >> "$LOG"
  # one container call per seed; run.py loops models x precisions internally and
  # skips already-finished units. --no-guard because shards may be partial.
  docker run --rm --gpus all \
    -v "$CQ_DATA/out:/data/out" -v "$CQ_DATA/hf:/data/hf" -e HF_HOME=/data/hf \
    "$IMAGE" python -m culturaquant.run \
    --models $MODELS --precisions $PRECISIONS \
    --data-dir /app/data --out-dir "/data/out/$OUT" \
    --n-proverbs "$NPROV" --seed "$SEED" --no-guard \
    >> "$LOG" 2>&1
  rc=$?
  echo "[seed $SEED] done rc=$rc $(date -Is)" >> "$LOG"
  # make outputs user-owned so they can be pulled back via rsync
  docker run --rm -v "$CQ_DATA/out:/data/out" --entrypoint chown \
    "$IMAGE" -R "$(id -u):$(id -g)" "/data/out/$OUT" >/dev/null 2>&1 || true
done
echo "[all-done] $(date -Is) host=$(hostname)" >> "$LOG"
