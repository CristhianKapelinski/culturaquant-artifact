# Pinned CUDA base for the Ampere (sm_86) SLM grid where bitsandbytes int8/nf4
# kernels are well-supported. Everything runs in this image; nothing on the host.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/data/hf \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# uv for environment management (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md reproduce.sh rescore.sh ./
COPY src ./src
COPY analysis ./analysis
COPY data ./data
COPY results ./results
COPY tests ./tests
COPY scripts ./scripts
COPY docs ./docs

# Resolve and install into a project venv; commit uv.lock for reproducibility.
RUN uv sync --extra dev --frozen || uv sync --extra dev

ENTRYPOINT ["uv", "run"]
CMD ["python", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"]
