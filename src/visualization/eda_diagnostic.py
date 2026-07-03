"""Diagnostic: verify labeled centroids actually land on bright nuclei.

Run: python src/visualization/eda_diagnostic.py
"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.loaders import load_volume, load_tracks

VOLUME_PATH = "data/raw/train/44b6_0113de3b.zarr"
GEFF_PATH = "data/raw/train/44b6_0113de3b.geff"


def main():
    volume = load_volume(VOLUME_PATH)
    nodes_df, edges_df = load_tracks(GEFF_PATH)

    # Middle of the long, stable chain (t=27..75) — less likely to be a
    # volume-boundary artifact than the short chain at t=0..2, z=63.
    row = nodes_df[nodes_df["t"] == 50].iloc[0]
    t, z, y, x = int(row.t), int(row.z), int(row.y), int(row.x)
    print(f"Checking node {row.node_id} at t={t}, z={z}, y={y}, x={x}")

    frame = np.asarray(volume[t])   # (Z, Y, X)
    mip = frame.max(axis=0)          # (Y, X)
    exact_slice = frame[z]           # (Y, X) — the node's own z-plane

    win = 20
    y0, y1 = max(0, y - win), min(mip.shape[0], y + win)
    x0, x1 = max(0, x - win), min(mip.shape[1], x + win)

    print("Pixel value at labeled point (MIP):", mip[y, x])
    print("Pixel value at labeled point (exact z-slice):", exact_slice[y, x])
    print("Local window max (MIP):", mip[y0:y1, x0:x1].max())
    print("Image-wide max (MIP):", mip.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, label in zip(
        axes,
        [mip[y0:y1, x0:x1], exact_slice[y0:y1, x0:x1]],
        ["MIP crop", f"z={z} slice crop"],
    ):
        ax.imshow(img, cmap="gray")
        ax.scatter([x - x0], [y - y0], s=150, facecolors="none",
                   edgecolors="red", linewidths=2)
        ax.set_title(label)
        ax.axis("off")

    out_path = "reports/eda_diagnostic_crop.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()