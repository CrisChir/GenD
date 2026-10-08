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
# curl, wget   -> fallback downloads (RetinaFace ONNX)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        git \
        curl \
        wget \
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

# ---- Pre-download RetinaFace ONNX model ---------------------
# src/retinaface.py:prepare_model() downloads det_10g.onnx with wget at
# runtime if missing. We fetch it at BUILD time so the container needs
# no network access on startup and no reliance on shell tools.
RUN mkdir -p /app/weights/models/buffalo_l \
    && curl -fsSL -o /app/weights/models/buffalo_l/det_10g.onnx \
       "https://huggingface.co/datasets/theanhntp/Liblib/resolve/ae4357741af379482690fe3e0f2fa6fd32ba33b4/insightface/models/buffalo_l/det_10g.onnx" \
    && python -c "import os; s=os.path.getsize('/app/weights/models/buffalo_l/det_10g.onnx'); assert s > 1_000_000, f'det_10g.onnx too small: {s}'" \
    || (echo "WARNING: RetinaFace det_10g.onnx could not be pre-downloaded; prepare_model() will retry with wget at runtime" && true)

# Pre-create writable dirs used by the app
RUN mkdir -p /cache/huggingface /cache/torch /tmp/gradio outputs

# Default: run the REST API + Gradio UI (single service, single port).
# The API is served at /api/v1/*, the Gradio UI at /ui.
# To run the UI-only app instead: docker run ... gend-app python app/run.py
CMD ["python", "app/api.py"]
