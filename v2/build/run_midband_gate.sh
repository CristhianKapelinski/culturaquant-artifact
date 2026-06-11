#!/usr/bin/env bash
# Step 4: fp16 GATE on the final mid-band BR set vs the rarity-matched control,
# in the Blackwell image (culturaquant:bw) on an RTX 5080. fp16 ONLY; 5 cyclic
# rotations/item. Reuses the paper's fp16_gate.py: it reads br_rare.jsonl +
# control_matched.jsonl from --data-dir, so we stage the midband files under those
# names in a dedicated gate dir. Reports BR vs control fp16 accuracy + RStd-relevant
# per-band, two-proportion z, Wilson CIs, GREEN/RED verdict.
# Resumable: completed per-model summaries are skipped.
set -uo pipefail

ART="${ART:-$HOME/cq-run/artifact}"
GATE="${GATE:-$ART/v2/data/midband_gate}"
IMAGE="${IMAGE:-culturaquant:bw}"
MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
        Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B}"

mkdir -p "$GATE/in" "$GATE/out"
cp "$ART/v2/data/midband_cultural.jsonl" "$GATE/in/br_rare.jsonl"
cp "$ART/v2/data/midband_control.jsonl"  "$GATE/in/control_matched.jsonl"

docker run --rm --gpus all \
  -v "$ART/v2/build/fp16_gate.py:/app/scripts/fp16_gate.py:ro" \
  -v "$ART/src:/app/src:ro" \
  -v "$GATE/in:/app/in:ro" \
  -v "$GATE/out:/app/out" \
  -v cq-hf:/data/hf \
  -e HF_HOME=/data/hf \
  "$IMAGE" \
  python /app/scripts/fp16_gate.py \
    --models $MODELS \
    --data-dir /app/in \
    --out-dir /app/out
