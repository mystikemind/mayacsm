# Project Maya - Production Dockerfile
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Base image with CUDA support
# =============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Create non-root user
RUN useradd --create-home --shell /bin/bash maya
USER maya
WORKDIR /home/maya/app

# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base AS deps

# Copy only requirements first for caching
COPY --chown=maya:maya pyproject.toml ./

# Create virtual environment and install dependencies
RUN python -m venv /home/maya/venv
ENV PATH="/home/maya/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
RUN pip install --no-cache-dir -e ".[prod]"

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM base AS production

# Copy virtual environment from deps stage
COPY --from=deps --chown=maya:maya /home/maya/venv /home/maya/venv
ENV PATH="/home/maya/venv/bin:$PATH"

# Copy application code
COPY --chown=maya:maya src/ ./src/
COPY --chown=maya:maya config/ ./config/
COPY --chown=maya:maya pyproject.toml ./

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for models and logs
RUN mkdir -p /home/maya/app/models /home/maya/app/logs

# Environment variables
ENV MAYA_ENVIRONMENT=production
ENV MAYA_LOG_JSON=true
ENV MAYA_LOG_LEVEL=INFO

# Model cache directory
ENV HF_HOME=/home/maya/app/models
ENV TRANSFORMERS_CACHE=/home/maya/app/models

# Expose ports
EXPOSE 8080 8765 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "maya.api.app"]

# =============================================================================
# Stage 4: Development image (optional)
# =============================================================================
FROM production AS development

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER maya

# Install dev dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov black isort mypy

ENV MAYA_ENVIRONMENT=development
ENV MAYA_LOG_LEVEL=DEBUG
ENV MAYA_DEBUG=true

CMD ["python", "-m", "maya.api.app"]
