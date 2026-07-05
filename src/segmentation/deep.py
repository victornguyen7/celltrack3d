"""Deep-learning segmentation: Cellpose-SAM in 3D mode (full frame, timed).

Run: python src/segmentation/deep.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from cellpose import models
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_volume
from src.utils.constants import VOXEL_SCALE_UM

VOLUME_PATH = "data/raw/train/44b6_0113de3b.zarr"
ANISOTROPY = VOXEL_SCALE_UM[0] / VOXEL_SCALE_UM[1]  # 4.0


def segment_frame_dl(frame: np.ndarray, model: models.CellposeModel) -> np.ndarray:
    masks, flows, styles = model.eval(
        frame,
        do_3D=True,
        z_axis=0,
        anisotropy=ANISOTROPY,
        flow3D_smooth=1,
    )
    return masks


def main():
    volume = load_volume(VOLUME_PATH)
    t = 50
    raw = np.asarray(volume[t]).astype(np.float32)  # full 256x256, no crop

    print("Loading Cellpose-SAM (cpsam) model with MPS...")
    model = models.CellposeModel(gpu=True)

    print(f"Segmenting FULL frame t={t} in 3D...")
    start = time.time()
    labels = segment_frame_dl(raw, model)
    elapsed = time.time() - start
    print(f"Full-frame 3D segmentation took {elapsed:.1f}s")

    props = regionprops(labels)
    sizes = sorted((p.area for p in props), reverse=True)
    print(f"Detected {len(props)} instances at t={t} (classical baseline: 172)")
    if sizes:
        print(f"Instance sizes (voxels) — min/median/max: "
              f"{sizes[-1]}/{sizes[len(sizes)//2]}/{sizes[0]} "
              f"(classical baseline: 54/1605/8073)")

    raw_mip = raw.max(axis=0)
    labels_mip = labels.max(axis=0)
    boundaries = find_boundaries(labels_mip, mode="thin")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(raw_mip, cmap="gray")
    ax.contour(boundaries, colors="lime", linewidths=0.5)
    ax.set_title(f"Cellpose-SAM segmentation — t={t}, full frame ({len(props)} cells)")
    ax.axis("off")

    out_path = "reports/segmentation_deep_full.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    np.save("data/processed/cellpose_labels_t50_full.npy", labels)


if __name__ == "__main__":
    main()