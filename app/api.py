"""GenD Deepfake Detection - REST API.

A FastAPI service that exposes the detector programmatically so it can be
integrated into a larger application, alongside the Gradio UI (mounted at /ui).

Endpoints (prefix /api/v1):
  GET  /api/v1/health    - service status
  GET  /api/v1/models   - available model identifiers
  POST /api/v1/analyze   - analyze an uploaded image/video; returns scores,
                           verdict and the rendered score graph (with legend)

Run (from the repo root, same as the Docker CMD):
  python app/api.py
"""

import base64
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

# Make sure the app directory (run.py, graphs.py) is importable when this
# file is executed as `python app/api.py` from the repo root.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import run
from run import (
    DEFAULT_CKPT,
    DETECTOR,
    HF_MODELS,
    IMAGE_EXTS,
    VIDEO_EXTS,
    MediaProcessor,
    is_image,
    is_video,
)

API_OUTPUT_DIR = Path("outputs") / "api"
API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Serialize model inference: the detector is heavy and the model cache in
# run.py holds a single model at a time.
_infer_lock = threading.Lock()

GRAPH_STYLES = ("Organic", "Radar 12 sectors", "None")


def _b64_png(path: str) -> Optional[str]:
    """Return a data URI for a PNG file, or None if the file is missing."""
    if not path or not Path(path).is_file():
        return None
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _run_video(processor: MediaProcessor, vid_path: str, detector, model, preproc, dtype,
               scale: float, target_size: Optional[int], out_dir: Path,
               stride: int, max_frames: int, max_faces: Optional[int]):
    """Consume the process_video generator and return its final return value."""
    gen = processor.process_video(
        vid_path,
        detector,
        model,
        preproc,
        dtype,
        scale,
        target_size,
        out_dir,
        stride=stride,
        max_frames=max_frames,
        max_faces=max_faces,
        stream_every=0,
    )
    try:
        while True:
            next(gen)
    except StopIteration as st:
        return st.value
    raise RuntimeError("process_video did not return a result")


def _analyze_sync(in_path: str, media_type: str, job_dir: Path, params: dict) -> dict:
    """Run the full analysis pipeline (blocking). Returns the API payload."""
    model_source = params["model_source"]
    model_id = params["hf_model"] if model_source == "Hugging Face" else params["local_ckpt"]
    target_size = None if params["target_size"] in (-1, 0, None) else int(params["target_size"])
    max_faces = None if params["max_faces"] in (-1, 0, None) else int(params["max_faces"])

    model, preproc, dtype = DETECTOR.load_model(model_source, model_id)
    face_detector = DETECTOR.load_detector(params["face_thresh"])
    processor = MediaProcessor(DETECTOR)

    if media_type == "image":
        out_path, metrics, p_fake_vals = processor.process_image(
            in_path, face_detector, model, preproc, dtype,
            params["scale"], target_size, job_dir, max_faces=max_faces,
        )
    else:
        out_path, metrics, p_fake_vals = _run_video(
            processor, in_path, face_detector, model, preproc, dtype,
            params["scale"], target_size, job_dir,
            params["stride"], params["max_frames"], max_faces,
        )

    scores = [float(v) for v in p_fake_vals]
    mean_p_fake = float(metrics.get("avg_p_fake") or 0.0)
    threshold = float(params["verdict_threshold"])
    verdict = "FAKE" if mean_p_fake >= threshold else "REAL"

    graph_path = run.render_score_graph(
        scores,
        params["graph_style"],
        job_dir,
        f"{Path(in_path).stem}_score",
        title="GenD API",
        legend=params["legend"],
    )

    def rel_url(p: str) -> Optional[str]:
        if not p:
            return None
        rel = Path(p).resolve().relative_to(Path("outputs").resolve()).as_posix()
        return f"/api/v1/files/{rel}"

    graph_block = None
    if graph_path:
        graph_block = {
            "style": params["graph_style"],
            "legend": bool(params["legend"]),
            "filename": Path(graph_path).name,
            "url": rel_url(graph_path),
            "base64_png": _b64_png(graph_path),
        }

    annotated_block = {
        "filename": Path(out_path).name,
        "url": rel_url(out_path),
    }
    if params["include_base64"] and is_image(out_path):
        b64 = _b64_png(out_path)
        if b64:
            annotated_block["base64_png"] = b64

    return {
        "filename": Path(in_path).name,
        "media_type": media_type,
        "num_frames": int(metrics.get("num_frames", 0)),
        "num_faces": int(metrics.get("num_faces", 0)),
        "avg_p_fake": mean_p_fake,
        "median_p_fake": float(metrics.get("median_p_fake") or 0.0),
        "p_fake_scores": scores,
        "verdict": {
            "label": verdict,
            "p_fake": mean_p_fake,
            "threshold": threshold,
        },
        "graph": graph_block,
        "annotated_media": annotated_block,
    }


app = FastAPI(
    title="GenD Deepfake Detection API",
    description="REST API for deepfake detection with circular score graphs (Organic / Radar 12 sectors).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("API_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/v1/files", StaticFiles(directory="outputs"), name="outputs")


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "device": run.DEVICE,
        "models_cached": list(DETECTOR.model_cache.keys()),
        "time": time.time(),
    }


@app.get("/api/v1/models")
async def models():
    return {
        "hugging_face": HF_MODELS,
        "default_local_checkpoint": DEFAULT_CKPT,
        "graph_styles": list(GRAPH_STYLES),
    }


@app.post("/api/v1/analyze")
async def analyze(
    file: UploadFile = File(..., description="Image or video to analyze"),
    model_source: str = Form("Hugging Face"),
    hf_model: str = Form(HF_MODELS[-1]),
    local_ckpt: str = Form(DEFAULT_CKPT),
    face_thresh: float = Form(0.5),
    scale: float = Form(1.3),
    target_size: int = Form(-1),
    max_faces: int = Form(-1),
    stride: int = Form(1),
    max_frames: int = Form(-1),
    graph_style: str = Form("Organic"),
    legend: bool = Form(True),
    verdict_threshold: float = Form(0.5),
    include_base64: bool = Form(False),
):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in IMAGE_EXTS and suffix not in VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(IMAGE_EXTS | VIDEO_EXTS)}",
        )
    if model_source not in ("Hugging Face", "Local Checkpoint"):
        raise HTTPException(status_code=400, detail="model_source must be 'Hugging Face' or 'Local Checkpoint'")
    if graph_style not in GRAPH_STYLES:
        raise HTTPException(status_code=400, detail=f"graph_style must be one of {GRAPH_STYLES}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = API_OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    in_path = job_dir / f"input{suffix}"
    with open(in_path, "wb") as fh:
        fh.write(await file.read())

    media_type = "image" if is_image(str(in_path)) else "video"
    params = {
        "model_source": model_source,
        "hf_model": hf_model,
        "local_ckpt": local_ckpt,
        "face_thresh": face_thresh,
        "scale": scale,
        "target_size": target_size,
        "max_faces": max_faces,
        "stride": max(1, int(stride)),
        "max_frames": int(max_frames),
        "graph_style": graph_style,
        "legend": legend,
        "verdict_threshold": verdict_threshold,
        "include_base64": include_base64,
    }

    started = time.time()
    try:
        with _infer_lock:
            result = await run_in_threadpool(
                _analyze_sync, str(in_path), media_type, job_dir, params
            )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    result["job_id"] = job_id
    result["processing_seconds"] = round(time.time() - started, 3)
    return JSONResponse(result)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")


# Mount the full Gradio UI alongside the API.
demo = run.build_ui()
gr.mount_gradio_app(app, demo, path="/ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 7860)),
    )
