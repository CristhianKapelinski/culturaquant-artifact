#!/usr/bin/env bash
# Cyclic position-bias probe on a local Blackwell (sm_120) GPU using the locally built
# culturaquant:bw image. HF cache -> named docker volume cq-hf; outputs -> bind-mounted
# into the artifact data/results dir. Resumable: completed (model,precision) units skip.
set -uo pipefail
ART="${ART:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="$ART/data/results/cyclic_bias"
mkdir -p "$OUT"

MODELS="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
        Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B"
PRECISIONS="fp16 int8 nf4"

docker run --rm --gpus all \
  -v "$ART/scripts/cyclic_position_bias.py:/app/scripts/cyclic_position_bias.py:ro" \
  -v "$ART/data/cultural_strata.jsonl:/app/data/cultural_strata.jsonl:ro" \
  -v "$ART/data/control_generic.jsonl:/app/data/control_generic.jsonl:ro" \
  -v "$OUT:/data/out" \
  -v cq-hf:/data/hf \
  -e HF_HOME=/data/hf \
  culturaquant:bw \
  python /app/scripts/cyclic_position_bias.py \
    --models $MODELS --precisions $PRECISIONS \
    --data-dir /app/data --out-dir /data/out
