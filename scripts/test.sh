#!/usr/bin/env bash
# Minimal test: run the unit-test suite (no GPU, no network).
# Host path (uv-managed) by default; pass --docker to run in the pinned image,
# which uses the image's baked environment (no bind mount over /app).
set -euo pipefail

IMAGE="${IMAGE:-culturaquant:1.0}"

if [[ "${1:-}" == "--docker" ]]; then
  exec docker run --rm "$IMAGE" python -m pytest tests/ -q
fi

exec uv run pytest tests/ -q
