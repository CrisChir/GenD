"""
╔══════════════════════════════════════════════════════════════════════╗
║       GenAI Video Detection App  –  Consolidated Script             ║
║                                                                      ║
║  Models (GitHub clones, NOT pip-installable):                       ║
║    • GenD  → https://github.com/yermandy/GenD                       ║
║    • NSG-VD → https://github.com/ZSHsh98/NSG-VD                    ║
║                                                                      ║
║  Weights are downloaded ONCE and reused from local cache.           ║
║  NSG-VD uses a real-video reference bank for MMD scoring.           ║
╚══════════════════════════════════════════════════════════════════════╝

Directory layout expected at runtime:
  ./repos/GenD/          ← git clone of yermandy/GenD
  ./repos/NSG-VD/        ← git clone of ZSHsh98/NSG-VD
  ./weights/
    hf_models/           ← GenD HuggingFace weights (auto-cached)
    256x256_diffusion_uncond.pt  ← NSG-VD diffusion backbone
  ./reference_bank/      ← *.mp4 real videos for NSG-VD MMD reference

Usage:
  python app.py

Environment variables (optional):
  HF_TOKEN   – HuggingFace access token (needed for gated models)
  PORT       – Gradio server port (default 7860)
  REFERENCE_DIR – override path to reference-bank folder
"""

# ─────────────────────────────────────────────────────────────────────
# 0. STDLIB / ENV
# ─────────────────────────────────────────────────────────────────────
import os, sys, json, warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple

warnings.filterwarnings("ignore")

# Honour an optional HF token set in the environment (or hard-code here)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

# ─────────────────────────────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
REPOS_DIR   = ROOT / "repos"
WEIGHTS_DIR = ROOT / "weights"
HF_CACHE    = WEIGHTS_DIR / "hf_models"
NSG_DIFF_PT = WEIGHTS_DIR / "256x256_diffusion_uncond.pt"
REF_DIR     = Path(os.environ.get("REFERENCE_DIR", ROOT / "reference_bank"))
OUTPUT_DIR  = ROOT / "outputs"

for d in [REPOS_DIR, HF_CACHE, REF_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 2. REPO BOOTSTRAP  (git clone once)
# ─────────────────────────────────────────────────────────────────────
def _clone_if_missing(repo_url: str, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        return
    print(f"[bootstrap] Cloning {repo_url} → {dest}")
    ret = os.system(f"git clone {repo_url} {dest}")
    if ret != 0:
        raise RuntimeError(f"git clone failed for {repo_url}")

_clone_if_missing("https://github.com/yermandy/GenD",    REPOS_DIR / "GenD")
_clone_if_missing("https://github.com/ZSHsh98/NSG-VD",   REPOS_DIR / "NSG-VD")

# Add both repos to sys.path so their src/ packages are importable
for repo in ["GenD", "NSG-VD"]:
    p = str(REPOS_DIR / repo)
    if p not in sys.path:
        sys.path.insert(0, p)

# ─────────────────────────────────────────────────────────────────────
# 3. HEAVY IMPORTS  (after path setup)
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.torch import load_model as st_load_model
from huggingface_hub import snapshot_download, hf_hub_download

# GenD
from src.hf.modeling_gend import GenD as GenD_HF, GenDConfig

# NSG-VD
from omegaconf import OmegaConf
from models.deep_mmd import deep_MMD
from models.discriminators import I3DDiscriminator
from utils.mmd_utils import MMD_batch2

# ─────────────────────────────────────────────────────────────────────
# 4. LOCAL MODEL CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────

def _hf_local_path(repo_id: str) -> Path:
    """Returns local mirror path; downloads once from HuggingFace if absent."""
    safe = repo_id.replace("/", "_")
    local = HF_CACHE / safe
    if local.exists() and any(local.iterdir()):
        print(f"[cache] Using cached HF model: {local}")
        return local
    print(f"[cache] Downloading {repo_id} → {local} …")
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        token=HF_TOKEN or None,
    )
    return local


# ─────────────────────────────────────────────────────────────────────
# 5. PERLIN NOISE  (pure-python, no C extension)
# ─────────────────────────────────────────────────────────────────────

def _perlin_1d(x: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grads = rng.uniform(-1, 1, len(x) + 2)
    def fade(t): return 6*t**5 - 15*t**4 + 10*t**3
    out = []
    for v in x:
        xi = int(v); xf = v - xi
        g1 = grads[xi % len(grads)]; g2 = grads[(xi+1) % len(grads)]
        u = fade(xf)
        out.append((1-u)*g1*xf + u*g2*(xf-1))
    return np.array(out)


# ─────────────────────────────────────────────────────────────────────
# 6. ORGANIC STATUS GRAPH  (two concentric rings)
# ─────────────────────────────────────────────────────────────────────

class OrganicStatusGraph:
    """
    Renders a hand-drawn-style polar graph.
      Outer ring  → dark / fake probability
      Inner ring  → blue / authenticity
    Returns a matplotlib Figure (caller saves/shows it).
    """

    def __init__(self, fake_score: float, frame_probs: List[float]):
        self.fake_score  = float(np.clip(fake_score, 0, 1))
        self.auth_score  = 1.0 - self.fake_score
        self.frame_probs = np.array(frame_probs, dtype=float)

        # Arc length: full circle when perfectly authentic
        self.arc = 0.30 + 0.70 * self.auth_score
        # Tendrils scale with fakeness
        self.n_tendrils = int(3 + 12 * self.fake_score)

        self.r_outer = 1.20   # dark ring  (fake)
        self.r_inner = 1.10   # blue ring  (authentic)

    def _noise(self, x, scale=4, amp=0.02, seed=0):
        return _perlin_1d(x * scale, seed=seed) * amp

    def render(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        bg = "#f4f1ea"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.axis("off")

        N = 900
        theta = np.linspace(0, 2 * np.pi * self.arc, N)

        # Interpolate per-frame probabilities onto the arc
        fp = self.frame_probs
        if fp.max() - fp.min() < 1e-6:
            fp_norm = np.full_like(fp, 0.5)
        else:
            fp_norm = (fp - fp.min()) / (fp.max() - fp.min())

        fake_curve = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(fp_norm)), fp_norm)
        auth_curve = 1.0 - fake_curve

        # Variable stroke widths
        w_outer = np.clip(0.02 + fake_curve * 0.12 + self._noise(theta, 4, 0.02, 1)
                          + np.random.normal(0, 0.008, N), 0.01, 0.25)
        w_inner = np.clip(0.02 + auth_curve * 0.12 + self._noise(theta, 3.5, 0.02, 2)
                          + np.random.normal(0, 0.008, N), 0.01, 0.25)

        wobble_o = self._noise(theta, 3,   0.01, 3)
        wobble_i = self._noise(theta, 3.2, 0.01, 4)

        # Skeleton lines
        ax.plot(theta, self.r_outer + wobble_o, color="#1a1a1a", lw=0.7, alpha=0.9)
        ax.plot(theta, self.r_inner + wobble_i, color="#1f77b4", lw=0.7, alpha=0.9)

        # Layered ink diffusion
        for layer in range(4):
            lo = self._noise(theta, 2+layer, 0.015, 10+layer)
            li = self._noise(theta, 2.5+layer, 0.015, 20+layer)
            alpha = 0.22 / (layer + 1)
            ax.fill_between(theta,
                            self.r_outer - w_outer + lo,
                            self.r_outer + w_outer + lo,
                            color="#1a1a1a", alpha=alpha)
            ax.fill_between(theta,
                            self.r_inner - w_inner + li,
                            self.r_inner + w_inner + li,
                            color="#1f77b4", alpha=alpha)

        # Tendrils (fake artefacts radiating outward from dark ring)
        tendril_idx = np.linspace(0, N-1, self.n_tendrils, dtype=int)
        rng = np.random.default_rng(42)
        for idx in tendril_idx:
            t0 = theta[idx]
            if 0.1 < t0 < (2 * np.pi * self.arc - 0.1):
                steps = 25
                t_path = t0 + np.cumsum(rng.uniform(-0.02, 0.02, steps))
                r_path = self.r_outer + np.cumsum(rng.uniform(0, 0.03, steps))
                for i in range(steps - 1):
                    ax.plot(t_path[i:i+2], r_path[i:i+2],
                            color="#1a1a1a",
                            lw=1.8 * (1 - i/steps),
                            alpha=0.5 * (1 - i/steps))

        # Central score text
        ax.text(0, 0,
                f"FAKE\n{self.fake_score*100:.1f}%",
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="#1a1a1a", transform=ax.transData)

        ax.set_ylim(0, 1.7)
        fig.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────
# 7. GEND MODEL (frame-level deepfake / image detector)
# ─────────────────────────────────────────────────────────────────────

HF_MODEL_IDS = [
    "yermandy/GenD_PE_L",
    "yermandy/GenD_CLIP_L_14",
    "yermandy/GenD_DINOv3_L",
]

class GenDDetector:
    """Loads GenD from local cache; analyzes per-frame fake probability."""

    def __init__(self):
        self._models: Dict[str, GenD_HF] = {}

    def load(self, repo_id: str) -> GenD_HF:
        if repo_id in self._models:
            return self._models[repo_id]

        local = _hf_local_path(repo_id)
        cfg_path  = local / "config.json"
        wt_path   = local / "model.safetensors"

        if not cfg_path.exists() or not wt_path.exists():
            raise FileNotFoundError(
                f"Expected config.json + model.safetensors in {local}. "
                "Ensure the repo was downloaded correctly."
            )

        with open(cfg_path) as f:
            config = GenDConfig(**json.load(f))

        model = GenD_HF(config)
        st_load_model(model, str(wt_path))
        model.eval()
        model.to(DEVICE)
        self._models[repo_id] = model
        print(f"[GenD] Loaded {repo_id} on {DEVICE}")
        return model

    @torch.no_grad()
    def analyze_video(self, video_path: str, repo_id: str,
                      frame_skip: int = 5) -> Tuple[float, List[float]]:
        model = self.load(repo_id)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.5, [0.5]

        scores = []
        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fid % frame_skip == 0:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t    = torch.tensor(
                    cv2.resize(rgb, (224, 224)), dtype=torch.float32
                ).permute(2, 0, 1).unsqueeze(0) / 255.0
                t    = t.to(DEVICE)
                prob = torch.softmax(model(t), dim=1)[0, 1].item()
                scores.append(prob)
            fid += 1
        cap.release()

        if not scores:
            return 0.5, [0.5]
        return float(np.mean(scores)), scores


# ─────────────────────────────────────────────────────────────────────
# 8. NSG-VD MODEL  (physics-driven, reference-bank based)
# ─────────────────────────────────────────────────────────────────────

NSG_CKPTS = {
    "standard-Pika-mp":     REPOS_DIR / "NSG-VD/ckpts/standard-Pika-mp.pth",
    "standard-SEINE-mp":    REPOS_DIR / "NSG-VD/ckpts/standard-SEINE-mp.pth",
    "unbalance-SEINE-mp":   REPOS_DIR / "NSG-VD/ckpts/unbalance-SEINE-mp.pth",
    "standard-Pika-d":      REPOS_DIR / "NSG-VD/ckpts/standard-Pika-d.pth",
    "standard-SEINE-d":     REPOS_DIR / "NSG-VD/ckpts/standard-SEINE-d.pth",
    "unbalance-SEINE-d":    REPOS_DIR / "NSG-VD/ckpts/unbalance-SEINE-d.pth",
}


def _load_nsgvd_frames(path: str, n: int = 8, size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    idxs  = np.linspace(0, total-1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frm = cap.read()
        if not ret:
            frm = np.zeros((size, size, 3), dtype=np.uint8)
        frm = cv2.resize(frm, (size, size))
        frames.append(frm)
    cap.release()
    arr = np.stack(frames)                         # (T, H, W, 3)
    return torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


class NSGVDDetector:
    """Loads NSG-VD checkpoint; scores via MMD against a real-video reference bank."""

    def __init__(self):
        self._models: Dict[str, deep_MMD] = {}
        self._ref_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _build_model(self, ckpt_key: str) -> deep_MMD:
        if ckpt_key in self._models:
            return self._models[ckpt_key]

        cfg_path = REPOS_DIR / "NSG-VD/configs/nsg-vd-224x224/test.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"NSG-VD test config not found: {cfg_path}")
        cfg = OmegaConf.load(cfg_path)

        ckpt_path = NSG_CKPTS.get(ckpt_key)
        if ckpt_path is None or not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"Checkpoint '{ckpt_key}' not found at {ckpt_path}.\n"
                "Make sure NSG-VD was cloned and ckpts/ folder is present."
            )

        disc  = I3DDiscriminator(
            input_channels=3,
            hidden_features=[64, 128, 256, 512],
            feature_dim=300,
        )
        model = deep_MMD(
            discriminator=disc,
            sigma=cfg.model.sigma,
            sigma0=cfg.model.sigma0,
            epsilon=cfg.model.epsilon,
            img_size=cfg.model.img_size,
            is_yy_zero=cfg.model.is_yy_zero,
            is_smooth=cfg.model.is_smooth,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval().to(DEVICE)
        self._models[ckpt_key] = model
        print(f"[NSG-VD] Loaded {ckpt_key} on {DEVICE}")
        return model

    def _build_reference_bank(self, model: deep_MMD,
                              ref_videos: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from all real-video references. Cached for the session."""
        if self._ref_cache is not None:
            return self._ref_cache

        feats, raws = [], []
        for vp in ref_videos:
            x = _load_nsgvd_frames(vp).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, f = model.net(x, out_feature=True)
            feats.append(f.cpu())
            raws.append(x.cpu().reshape(1, -1))

        self._ref_cache = (torch.cat(feats), torch.cat(raws))
        return self._ref_cache

    @torch.no_grad()
    def score_video(self, video_path: str, ckpt_key: str,
                    ref_videos: List[str]) -> Dict:
        model = self._build_model(ckpt_key)

        if not ref_videos:
            return {"label": "UNKNOWN (no reference bank)", "score": float("nan"),
                    "fake_prob": 0.5, "frame_probs": [0.5]}

        feat_ref, ref_raw = self._build_reference_bank(model, ref_videos)

        x = _load_nsgvd_frames(video_path).unsqueeze(0).to(DEVICE)
        _, feat_test = model.net(x, out_feature=True)

        score = MMD_batch2(
            torch.cat([feat_ref, feat_test.cpu()], dim=0),
            feat_ref.shape[0],
            torch.cat([ref_raw, x.cpu().reshape(1, -1)], dim=0),
            model.sigma, model.sigma0_u, model.ep,
            is_smooth=model.is_smooth,
        ).item()

        # Normalise score to [0,1] fake probability (score > 1 → AI-generated)
        fake_prob = float(np.clip(score / 2.0, 0, 1))
        label = "AI-GENERATED" if score > 1.0 else "REAL"

        # Frame-level probs: run GenD-style per-frame for the graph
        frame_probs = self._frame_probs_from_video(video_path)

        return {
            "label": label,
            "score": score,
            "fake_prob": fake_prob,
            "frame_probs": frame_probs,
        }

    def _frame_probs_from_video(self, path: str, skip: int = 10) -> List[float]:
        """Lightweight per-frame score via raw pixel norm (proxy for graph curve)."""
        cap = cv2.VideoCapture(path)
        probs, fid = [], 0
        while True:
            ret, frm = cap.read()
            if not ret: break
            if fid % skip == 0:
                # Proxy: local Laplacian sharpness as fake probability signal
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                lap  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                probs.append(lap)
            fid += 1
        cap.release()
        if not probs:
            return [0.5]
        arr = np.array(probs, dtype=float)
        arr = 1.0 - (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return arr.tolist()


# ─────────────────────────────────────────────────────────────────────
# 9. SINGLETONS
# ─────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
gend_detector  = GenDDetector()
nsgvd_detector = NSGVDDetector()


def _collect_ref_videos() -> List[str]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [str(p) for p in REF_DIR.rglob("*") if p.suffix.lower() in exts]


# ─────────────────────────────────────────────────────────────────────
# 10. GRADIO INFERENCE CALLBACK
# ─────────────────────────────────────────────────────────────────────

def run_detection(
    files,
    detector_choice: str,
    gend_model_id: str,
    nsgvd_ckpt: str,
    frame_skip: int,
    ref_videos_str: str,
):
    if not files:
        yield None, "⚠️  Please upload at least one video file.", None
        return

    # Parse reference-bank overrides (newline-separated paths)
    custom_refs = [p.strip() for p in ref_videos_str.splitlines() if p.strip()]
    ref_videos  = custom_refs if custom_refs else _collect_ref_videos()

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    paths = [f.name for f in files if Path(f.name).suffix.lower() in video_exts]

    if not paths:
        yield None, "⚠️  No supported video files found (mp4/avi/mov/mkv/webm).", None
        return

    results_text = ""
    last_fig = None

    for vp in paths:
        name = Path(vp).name
        yield None, f"⏳ Processing **{name}** …", None

        try:
            if detector_choice == "GenD (frame-level deepfake)":
                fake_score, frame_probs = gend_detector.analyze_video(
                    vp, gend_model_id, frame_skip=frame_skip
                )
                label = "AI-GENERATED" if fake_score > 0.5 else "REAL"
                extra = ""

            else:  # NSG-VD
                res = nsgvd_detector.score_video(vp, nsgvd_ckpt, ref_videos)
                fake_score  = res["fake_prob"]
                frame_probs = res["frame_probs"]
                label       = res["label"]
                extra       = f" | MMD score: {res['score']:.4f}"

            graph = OrganicStatusGraph(fake_score, frame_probs)
            fig   = graph.render()
            last_fig = fig

            # Save graph
            out_path = OUTPUT_DIR / f"{Path(vp).stem}_organic.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            results_text += (
                f"\n### 📹 {name}\n"
                f"- **Verdict:** `{label}`\n"
                f"- **Fake probability:** {fake_score*100:.1f}%{extra}\n"
                f"- **Graph saved:** `{out_path}`\n"
            )

        except Exception as e:
            results_text += f"\n### ❌ {name}\n- Error: {e}\n"

    out_img = str(OUTPUT_DIR / f"{Path(paths[-1]).stem}_organic.png") if paths else None
    yield out_img, f"✅ **Done!**\n{results_text}", out_img


# ─────────────────────────────────────────────────────────────────────
# 11. GRADIO UI
# ─────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GenAI / Deepfake Video Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🕵️ GenAI / Deepfake Video Detector
            Upload videos to detect whether they are **AI-generated or real**.
            Produces an **organic circular graph** showing fake probability.

            > **GenD** analyses individual frames via a fine-tuned vision encoder (paper: arXiv 2508.06248).  
            > **NSG-VD** uses physics-driven spatiotemporal NSG statistics + MMD vs a real-video reference bank (paper: arXiv 2510.08073).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.Files(
                    label="📂 Upload video(s)",
                    file_count="multiple",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                )

                detector_choice = gr.Radio(
                    ["GenD (frame-level deepfake)", "NSG-VD (physics / AI-gen)"],
                    label="Detector",
                    value="GenD (frame-level deepfake)",
                )

                with gr.Group(visible=True) as gend_group:
                    gend_model_id = gr.Dropdown(
                        HF_MODEL_IDS,
                        value=HF_MODEL_IDS[0],
                        label="GenD HuggingFace model",
                    )

                with gr.Group(visible=False) as nsgvd_group:
                    nsgvd_ckpt = gr.Dropdown(
                        list(NSG_CKPTS.keys()),
                        value="standard-Pika-mp",
                        label="NSG-VD checkpoint",
                    )
                    ref_videos_str = gr.Textbox(
                        label="Reference bank (real video paths, one per line)",
                        placeholder=(
                            f"Leave blank to auto-scan {REF_DIR}\n"
                            "/path/to/real1.mp4\n/path/to/real2.mp4"
                        ),
                        lines=4,
                    )

                with gr.Accordion("⚙️ Advanced", open=False):
                    frame_skip = gr.Slider(1, 30, value=5, step=1,
                                           label="Frame skip (GenD only)")

                run_btn = gr.Button("🚀 Analyse", variant="primary", size="lg")

            with gr.Column(scale=2):
                status_md = gr.Markdown("Upload a video and click **Analyse**.")
                graph_img = gr.Image(label="Organic Fake-Score Graph", type="filepath")

        # Toggle detector groups
        def _toggle(choice):
            return (
                gr.update(visible=choice.startswith("GenD")),
                gr.update(visible=choice.startswith("NSG")),
            )

        detector_choice.change(_toggle, [detector_choice], [gend_group, nsgvd_group])

        run_btn.click(
            fn=run_detection,
            inputs=[
                files, detector_choice, gend_model_id,
                nsgvd_ckpt, frame_skip, ref_videos_str,
            ],
            outputs=[graph_img, status_md, graph_img],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────
# 12. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
