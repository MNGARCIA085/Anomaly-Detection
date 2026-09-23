FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Select the PyTorch wheel source at build time.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Package metadata + source code
COPY pyproject.toml .
COPY src ./src

# 1. Pre-install PyTorch from the specific CPU/GPU index
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    --index-url ${TORCH_INDEX_URL}

# 2. Install the rest of the package dependencies using PyTorch index as an extra index 
# so it reuses the installed PyTorch wheel instead of downloading CUDA versions
RUN pip install --no-cache-dir --extra-index-url ${TORCH_INDEX_URL} .

# Scripts are needed by both images
COPY scripts ./scripts


# ============================================================
# Inference
# ============================================================

FROM base AS inference

CMD ["bash"]


# ============================================================
# Training
# ============================================================

FROM base AS train

# Hydra configuration is only needed for training
COPY config ./config

# Install training dependencies
RUN pip install --no-cache-dir ".[train]"

CMD ["bash"]