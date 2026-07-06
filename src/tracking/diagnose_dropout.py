"""Check whether the ground-truth cell itself is going undetected in some
frames (segmentation dropout) vs. always detected but mis-linked.

Run: python src/tracking/diagnose_dropout.py
"""
from pathlib import Path

import numpy as np
import networkx as nx
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_tracks
from src.utils.constants import VOXEL_SCALE_UM

SAMPLE = "44b6_0b24845f"
GENEROUS_RADIUS_UM = 15.0  # much looser than the 6um tracking gate -- just
                           # checking "does ANY detection exist nearby at all"


def main():
    nodes_df, edges_df = load_tracks(f"data/raw/train/{SAMPLE}.geff")
    g = nx.DiGraph()
    g.add_edges_from(edges_df[["source_id", "target_id"]].itertuples(index=False))
    chains = list(nx.weakly_connected_components(g))
    chain0 = nodes_df[nodes_df["node_id"].isin(chains[0])].sort_values("t")

    labels_dir = Path(f"data/processed/{SAMPLE}_cellpose_labels")
    scale = np.array(VOXEL_SCALE_UM)

    missing_frames = []
    for _, gt in chain0.iterrows():
        t = int(gt["t"])
        labels = np.load(labels_dir / f"t{t:03d}.npy")
        from skimage.measure import regionprops
        centroids = np.array([p.centroid for p in regionprops(labels)])
        if len(centroids) == 0:
            missing_frames.append(t)
            continue
        gt_pos = np.array([gt["z"], gt["y"], gt["x"]]) * scale
        pred_pos = centroids * scale
        dists = np.sqrt(((pred_pos - gt_pos) ** 2).sum(axis=1))
        if dists.min() > GENEROUS_RADIUS_UM:
            missing_frames.append(t)
        print(f"t={t}: nearest detection {dists.min():.2f} um away")

    print(f"\nFrames with NO detection within {GENEROUS_RADIUS_UM}um: {missing_frames}")
    print(f"({len(missing_frames)} / {len(chain0)} frames)")


if __name__ == "__main__":
    main()