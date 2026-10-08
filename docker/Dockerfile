# ============================================================
# GenD – Deepfake Detection Gradio App
# Docker image (CPU / CUDA-optional)
# Build (from the REPO ROOT, not from docker/):
#   docker build -f docker/Dockerfile -t gend-app .
# Run:
#   docker run --rm -p 7860:7860 gend-app
# ============================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/cache/huggingface \
    GRADIO_TEMP_DIR=/tmp/gradio \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

# ---- System dependencies -----------------------------------
# ffmpeg       -> imageio / PyAV video decoding
# libglib2.0-0 -> OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ------------------------------------
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# On Apple Silicon (linux/arm64) the 'onnxruntime-gpu' wheel does not
# exist; fall back to CPU 'onnxruntime'. The RetinaFace detector works
# identically on both backends.
RUN arch=$(uname -m) \
    && if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then \
        sed 's/onnxruntime-gpu==/onnxruntime==/' /tmp/requirements.txt > /tmp/requirements.docker.txt; \
    else \
        cp /tmp/requirements.txt /tmp/requirements.docker.txt; \
    fi \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.docker.txt

# ---- Application code ---------------------------------------
COPY . /app

# Pre-create writable dirs used by the app
RUN mkdir -p /cache/huggingface /cache/torch /tmp/gradio outputs

# Default: run the Gradio app
CMD ["python", "app/run.py"]
