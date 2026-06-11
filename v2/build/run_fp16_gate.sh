#!/usr/bin/env bash
# Phase-2 fp16 difficulty-match gate for CulturaQuant-v2, run in the Blackwell image
# (culturaquant:bw) on an idle GPU. fp16 ONLY; 5 cyclic rotations/item.
#
# Runs ON the GPU host (it has the image + GPU). Copy the v2 artifact dir there first,
# or invoke remotely. The script bind-mounts the v2 data + the gate script into /app and
# uses the image's installed culturaquant package (src) and the cq-hf model cache volume.
#
# Resumable: completed per-model summaries are skipped.
set -uo pipefail

V2="${V2:-$(cd "$(dirname "$0")/.." && pwd)}"
SRC="${SRC:-$(cd "$V2/.." && pwd)/src}"
OUT="$V2/data/results/fp16_gate"
mkdir -p "$OUT"

MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
        Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B}"
IMAGE="${IMAGE:-culturaquant:bw}"

docker run --rm --gpus all \
  -v "$V2/build/fp16_gate.py:/app/scripts/fp16_gate.py:ro" \
  -v "$SRC:/app/src:ro" \
  -v "$V2/data:/app/v2data" \
  -v cq-hf:/data/hf \
  -e HF_HOME=/data/hf \
  "$IMAGE" \
  python /app/scripts/fp16_gate.py \
    --models $MODELS \
    --data-dir /app/v2data \
    --out-dir /app/v2data/results/fp16_gate
