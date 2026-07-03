"""Classical segmentation: Otsu threshold -> distance transform -> watershed.

Run: python src/segmentation/classical.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, watershed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_volume

from src.preprocessing.basic import preprocess_frame
from src.utils.constants import VOXEL_SCALE_UM

VOLUME_PATH = "data/raw/train/44b6_0113de3b.zarr"


def segment_frame(
    frame: np.ndarray,
    min_size: int = 50,
    min_distance_um: float = 3.0,
) -> np.ndarray:
    """Segment one denoised+normalized 3D (Z,Y,X) frame into labeled instances.

    NOTE: peak_local_max's min_distance is isotropic (voxel counts, no
    per-axis spacing support), so we approximate using the finest axis
    (xy). This under-suppresses seeds along z relative to true physical
    distance — a known simplification, revisit if z-adjacent cells get
    over-split.
    """
    thresh = threshold_otsu(frame)
    # Note: skimage 0.26 renamed `min_size` -> `max_size` for this function,
    # but kept the same "remove objects at/below this size" semantics.
    binary = remove_small_objects(frame > thresh, max_size=min_size)

    distance = np.asarray(
        ndimage.distance_transform_edt(binary, sampling=VOXEL_SCALE_UM),
        dtype=np.float32,
    )

    min_distance_px = max(1, int(min_distance_um / min(VOXEL_SCALE_UM)))
    coords = peak_local_max(distance, min_distance=min_distance_px, labels=binary)
    markers = np.zeros(distance.shape, dtype=int)
    for i, (z, y, x) in enumerate(coords, start=1):
        markers[z, y, x] = i

    labels = watershed(-distance, markers, mask=binary)

    # Drop tiny post-split fragments, then relabel IDs to be contiguous.
    cleaned = remove_small_objects(labels, max_size=min_size)
    from skimage.segmentation import relabel_sequential
    labels, _, _ = relabel_sequential(cleaned)

    return labels


def main():
    volume = load_volume(VOLUME_PATH)
    t = 50
    raw = np.asarray(volume[t])
    processed = preprocess_frame(raw)

    labels = segment_frame(processed)
    props = regionprops(labels)
    sizes = sorted((p.area for p in props), reverse=True)
    print(f"Detected {len(props)} instances at t={t}")
    print(f"Instance sizes (voxels) — min/median/max: "
          f"{sizes[-1]}/{sizes[len(sizes)//2]}/{sizes[0]}")

    # Quick rough visual: crude max-projection of the 3D labels (not a real
    # 2D segmentation, just enough to eyeball whether cells got split).
    mip_raw = raw.max(axis=0)
    mip_labels = labels.max(axis=0)
    boundaries = find_boundaries(mip_labels, mode="thin")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mip_raw, cmap="gray")
    ax.contour(boundaries, colors="red", linewidths=0.5)
    ax.set_title(f"Classical segmentation boundaries — t={t} ({len(props)} cells)")
    ax.axis("off")

    out_path = "reports/segmentation_classical.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()