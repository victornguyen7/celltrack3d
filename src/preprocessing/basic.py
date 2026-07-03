"""Basic preprocessing: denoise + intensity normalization.

Run: python src/preprocessing/basic.py
"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_volume

VOLUME_PATH = "data/raw/train/44b6_0113de3b.zarr"


def denoise(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Light Gaussian smoothing to suppress shot noise before segmentation."""
    return ndimage.gaussian_filter(frame.astype(np.float32), sigma=sigma)


def normalize(frame: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    """Percentile clip + rescale to [0, 1] — robust to a few hot pixels."""
    lo, hi = np.percentile(frame, [low_pct, high_pct])
    clipped = np.clip(frame, lo, hi)
    return (clipped - lo) / max(hi - lo, 1e-6)


def preprocess_frame(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return normalize(denoise(frame, sigma))


def main():
    volume = load_volume(VOLUME_PATH)
    t = 50
    raw = np.asarray(volume[t])  # (Z, Y, X)

    processed = preprocess_frame(raw)

    raw_mip = raw.max(axis=0)
    processed_mip = processed.max(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(raw_mip, cmap="gray")
    axes[0].set_title(f"Raw (t={t}, MIP)")
    axes[1].imshow(processed_mip, cmap="gray")
    axes[1].set_title(f"Denoised + normalized (t={t}, MIP)")
    for ax in axes:
        ax.axis("off")

    out_path = "reports/preprocessing_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Raw range: [{raw.min()}, {raw.max()}]")
    print(f"Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()