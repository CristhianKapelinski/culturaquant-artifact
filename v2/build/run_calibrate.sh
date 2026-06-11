#!/usr/bin/env bash
# Step 2: fp16 mid-band calibration of the rarity gradient, in the Blackwell image
# (culturaquant:bw) on an RTX 5080. fp16 ONLY; 5 cyclic rotations/item; per-bucket acc.
# Meant to run ON the GPU host against its synced ~/cq-run/artifact tree.
# Resumable: completed per-model summaries are skipped.
set -uo pipefail

ART="${ART:-$HOME/cq-run/artifact}"
DATAFILE="${DATAFILE:-gradient_cultural.jsonl}"   # basename inside v2/data
OUTSUB="${OUTSUB:-calib}"                          # subdir under v2/data/results
IMAGE="${IMAGE:-culturaquant:bw}"
MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
        Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B}"
mkdir -p "$ART/v2/data/results/$OUTSUB"

docker run --rm --gpus all \
  -v "$ART/v2/build/calibrate_gradient.py:/app/scripts/calibrate_gradient.py:ro" \
  -v "$ART/src:/app/src:ro" \
  -v "$ART/v2/data:/app/v2data" \
  -v cq-hf:/data/hf \
  -e HF_HOME=/data/hf \
  "$IMAGE" \
  python /app/scripts/calibrate_gradient.py \
    --models $MODELS \
    --data "/app/v2data/$DATAFILE" \
    --out-dir "/app/v2data/results/$OUTSUB"
