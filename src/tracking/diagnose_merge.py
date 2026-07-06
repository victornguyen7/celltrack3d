"""Check whether the nearest detection at each frame is a plausible single
cell or an oversized merged blob.

Run: python src/tracking/diagnose_merge.py
"""
from pathlib import Path

import numpy as np
from skimage.measure import regionprops
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_tracks
from src.utils.constants import VOXEL_SCALE_UM
import networkx as nx

SAMPLE = "44b6_0b24845f"


def main():
    nodes_df, edges_df = load_tracks(f"data/raw/train/{SAMPLE}.geff")
    g = nx.DiGraph()
    g.add_edges_from(edges_df[["source_id", "target_id"]].itertuples(index=False))
    chains = list(nx.weakly_connected_components(g))
    chain0 = nodes_df[nodes_df["node_id"].isin(chains[0])].sort_values("t")

    labels_dir = Path(f"data/processed/{SAMPLE}_cellpose_labels")
    scale = np.array(VOXEL_SCALE_UM)

    # Typical single-cell size from earlier stats: median ~486 voxels.
    MERGED_THRESHOLD = 486 * 3  # ~3x median = likely a fused multi-cell blob

    for _, gt in chain0.iterrows():
        t = int(gt["t"])
        labels = np.load(labels_dir / f"t{t:03d}.npy")
        props = regionprops(labels)
        centroids = np.array([p.centroid for p in props])
        areas = np.array([p.area for p in props])

        gt_pos = np.array([gt["z"], gt["y"], gt["x"]]) * scale
        pred_pos = centroids * scale
        dists = np.sqrt(((pred_pos - gt_pos) ** 2).sum(axis=1))
        best = dists.argmin()

        flag = "MERGED?" if areas[best] > MERGED_THRESHOLD else ""
        print(f"t={t}: dist={dists[best]:.2f}um, nearest instance size={areas[best]:.0f} voxels {flag}")


if __name__ == "__main__":
    main()