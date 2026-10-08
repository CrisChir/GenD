"""Circular score graphs for the Gradio app.

Two styles, ported from the prototype applications:
  - OrganicGraph: hand-drawn style polar plot with two concentric rings
    (outer = fake probability, inner = authenticity), Perlin noise,
    ink diffusion and tendrils. Arc length reflects authenticity.
  - RadarGraph: dark 12-sector polar polygon with 24 score points.

Both take a list of per-face / per-frame p_fake scores in [0, 1].
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _perlin(x: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.uniform(-1, 1, len(x) + 2)
    fade = lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3  # noqa: E731
    out = []
    for v in x:
        xi, xf = int(v), v - int(v)
        u = fade(xf)
        out.append((1 - u) * g[xi % len(g)] * xf + u * g[(xi + 1) % len(g)] * (xf - 1))
    return np.array(out)


class OrganicGraph:
    """Hand-drawn style polar graph (outer ring = fake, inner ring = authentic)."""

    def __init__(self, fake: float, frame_probs: List[float]):
        self.fake = float(np.clip(fake, 0, 1))
        self.auth = 1.0 - self.fake
        self.fp = np.array(frame_probs, dtype=float)
        self.arc = 0.30 + 0.70 * self.auth
        self.ntend = int(3 + 12 * self.fake)
        self.ro, self.ri = 1.20, 1.10

    def _n(self, x, sc=4, am=0.02, sd=0):
        return _perlin(x * sc, seed=sd) * am

    def render(self, legend: bool = True) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("#f4f1ea")
        ax.set_facecolor("#f4f1ea")
        ax.axis("off")
        N = 900
        th = np.linspace(0, 2 * np.pi * self.arc, N)
        fp = self.fp
        fn = (
            (fp - fp.min()) / (fp.max() - fp.min() + 1e-6)
            if fp.max() > fp.min()
            else np.full_like(fp, 0.5)
        )
        fc = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(fn)), fn)
        ac = 1.0 - fc
        rng2 = np.random.default_rng(1)
        wo = np.clip(0.02 + fc * 0.12 + self._n(th, 4, 0.02, 1) + rng2.normal(0, 0.008, N), 0.01, 0.25)
        wi = np.clip(0.02 + ac * 0.12 + self._n(th, 3.5, 0.02, 2) + rng2.normal(0, 0.008, N), 0.01, 0.25)
        ax.plot(th, self.ro + self._n(th, 3, 0.01, 3), color="#1a1a1a", lw=0.7, alpha=0.9)
        ax.plot(th, self.ri + self._n(th, 3.2, 0.01, 4), color="#1f77b4", lw=0.7, alpha=0.9)
        for ly in range(4):
            a = 0.22 / (ly + 1)
            ax.fill_between(th, self.ro - wo + self._n(th, 2 + ly, 0.015, 10 + ly),
                            self.ro + wo + self._n(th, 2 + ly, 0.015, 10 + ly), color="#1a1a1a", alpha=a)
            ax.fill_between(th, self.ri - wi + self._n(th, 2.5 + ly, 0.015, 20 + ly),
                            self.ri + wi + self._n(th, 2.5 + ly, 0.015, 20 + ly), color="#1f77b4", alpha=a)
        rng3 = np.random.default_rng(42)
        for idx in np.linspace(0, N - 1, self.ntend, dtype=int):
            t0 = th[idx]
            if 0.1 < t0 < 2 * np.pi * self.arc - 0.1:
                steps = 25
                tp = t0 + np.cumsum(rng3.uniform(-0.02, 0.02, steps))
                rp = self.ro + np.cumsum(rng3.uniform(0, 0.03, steps))
                for i in range(steps - 1):
                    ax.plot(tp[i:i + 2], rp[i:i + 2], color="#1a1a1a",
                            lw=1.8 * (1 - i / steps), alpha=0.5 * (1 - i / steps))
        ax.text(0, 0, f"FAKE\n{self.fake * 100:.1f}%", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#1a1a1a", transform=ax.transData)
        ax.set_ylim(0, 1.7)
        if legend:
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor="#1a1a1a", alpha=0.45, label="P(fake) - outer ring"),
                Patch(facecolor="#1f77b4", alpha=0.45, label="Authenticity - inner ring"),
            ]
            ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.06),
                frameon=False,
                fontsize=9,
                labelcolor="#1a1a1a",
            )
        fig.tight_layout()
        return fig


class RadarGraph:
    """Dark 12-sector radar polygon with 24 interpolated score points."""

    @staticmethod
    def create(
        scores: List[float],
        title: str,
        filename: str,
        color: str = "#00ffcc",
        legend: bool = True,
    ) -> str:
        s = np.asarray(scores, dtype=float)
        if s.size == 0:
            s = np.array([0.5])
        if s.size == 1:
            s = np.full(24, float(s[0]))
        else:
            s = np.interp(np.linspace(0, 1, 24), np.linspace(0, 1, len(s)), s)
        s = np.clip(s, 0, 1)

        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        theta = np.append(theta, theta[0])
        radii = np.append(s, s[0])

        plt.figure(figsize=(6, 6), facecolor="#0b0b0b")
        ax = plt.subplot(111, projection="polar")
        ax.set_facecolor("#0b0b0b")
        ax.fill(theta, radii, color=color, alpha=0.2)
        ax.plot(theta, radii, color=color, lw=2, marker="o", markersize=4, alpha=0.8)
        for i in range(12):
            a = (i / 12) * 2 * np.pi
            ax.plot([a, a], [0, 1], color="white", alpha=0.1, lw=0.8)
        ax.set_ylim(0, 1)
        ax.grid(color="white", alpha=0.1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        fake = float(np.mean(s))
        ax.set_title(f"{title} - FAKE {fake * 100:.1f}%", color="white", size=12, pad=20)
        if legend:
            from matplotlib.lines import Line2D

            handles = [
                Line2D([0], [0], color=color, lw=2, marker="o", markersize=4, alpha=0.8,
                       label="P(fake) per sample (24 sectors)"),
                Line2D([0], [0], color="white", alpha=0.25, lw=0.8, label="Sector boundaries"),
            ]
            ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.14),
                frameon=False,
                fontsize=8,
                labelcolor="white",
                facecolor="#0b0b0b",
            )
        path = str(filename)
        plt.savefig(path, dpi=110, facecolor="#0b0b0b")
        plt.close()
        return path


def render_score_graph(
    scores: List[float],
    style: str,
    out_dir,
    name: str,
    title: str = "GenD",
    legend: bool = True,
) -> Optional[str]:
    """Render the score graph in the requested style; returns the image path."""
    if not scores or style not in ("Organic", "Radar 12 sectors"):
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fake = float(np.clip(np.mean(scores), 0, 1))

    if style == "Organic":
        fig = OrganicGraph(fake, list(scores)).render(legend=legend)
        path = out_dir / f"{name}_organic.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    path = out_dir / f"{name}_radar.png"
    return RadarGraph.create(list(scores), title, path, legend=legend)
