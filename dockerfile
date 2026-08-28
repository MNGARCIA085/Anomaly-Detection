FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Package metadata + source code
COPY pyproject.toml .
COPY src ./src

# Install runtime dependencies and the package
RUN pip install --no-cache-dir .

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