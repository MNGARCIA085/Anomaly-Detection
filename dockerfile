FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Select the PyTorch wheel source at build time.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Package metadata + source code
COPY pyproject.toml .
COPY src ./src

# Install PyTorch explicitly from the selected index
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    --index-url ${TORCH_INDEX_URL}

# Install the package without resolving dependencies again.
# This keeps the explicitly installed PyTorch version.
RUN pip install --no-cache-dir --no-deps .

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