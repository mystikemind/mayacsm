FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone Sesame CSM library
RUN git clone https://github.com/SesameAILabs/csm /app/csm

# Copy project files
COPY requirements.txt ./
COPY maya/ ./maya/
COPY assets/ ./assets/
COPY server.py ./
COPY .env ./

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Install CSM library deps
RUN pip install --no-cache-dir -r /app/csm/requirements.txt

ENV MAYA_PROJECT_ROOT=/app
ENV MAYA_CSM_ROOT=/app/csm
ENV MAYA_GPU_INDEX=0
ENV PORT=8002

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "server.py"]
