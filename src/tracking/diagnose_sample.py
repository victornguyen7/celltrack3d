"""Diagnose why 44b6_0b24845f's tracking is so much worse than other samples.

Run: python src/tracking/diagnose_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from laptrack import LapTrack
from skimage.measure import regionprops
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_tracks
from src.utils.constants import VOXEL_SCALE_UM

SAMPLE = "44b6_0b24845f"
T_START, T_END = 11, 50
MAX_LINK_DISTANCE_UM = 6.0


def build_detections_df(sample, t_start, t_end):
    labels_dir = Path(f"data/processed/{sample}_cellpose_labels")
    scale = np.array(VOXEL_SCALE_UM)
    rows = []
    for t in range(t_start, t_end + 1):
        labels = np.load(labels_dir / f"t{t:03d}.npy")
        for p in regionprops(labels):
            z, y, x = p.centroid
            zs, ys, xs = np.array([z, y, x]) * scale
            rows.append({"frame": t, "label": p.label, "area": p.area,
                         "z": z, "y": y, "x": x, "z_um": zs, "y_um": ys, "x_um": xs})
    return pd.DataFrame(rows)


def main():
    df = build_detections_df(SAMPLE, T_START, T_END)

    # 1. Detection density per frame -- compare against the healthy sample's
    # rough scale (a few hundred per frame). Wildly different counts would
    # point at over/under-segmentation rather than a linking problem.
    counts = df.groupby("frame").size()
    print("Detections per frame -- min/median/max:",
          counts.min(), counts.median(), counts.max())
    sizes = df["area"]
    print("Instance sizes (voxels) -- min/median/max:",
          sizes.min(), sizes.median(), sizes.max())

    # 2. Ground truth: where exactly do the two chains sit spatially/temporally?
    # Overlapping chains competing for the same space is a plausible cause of
    # the extreme switch rate -- check their centroid distance over time.
    nodes_df, edges_df = load_tracks(f"data/raw/train/{SAMPLE}.geff")
    import networkx as nx
    g = nx.DiGraph()
    g.add_edges_from(edges_df[["source_id", "target_id"]].itertuples(index=False))
    chains = list(nx.weakly_connected_components(g))

    chain_dfs = [nodes_df[nodes_df["node_id"].isin(c)].sort_values("t") for c in chains]
    scale = np.array(VOXEL_SCALE_UM)
    overlap_t = set(chain_dfs[0]["t"]) & set(chain_dfs[1]["t"])
    print(f"\nChains overlap in time for {len(overlap_t)} frames: {sorted(overlap_t)[:10]}...")

    for t in sorted(overlap_t)[:5]:
        p0 = chain_dfs[0][chain_dfs[0]["t"] == t][["z", "y", "x"]].to_numpy()[0] * scale
        p1 = chain_dfs[1][chain_dfs[1]["t"] == t][["z", "y", "x"]].to_numpy()[0] * scale
        dist = np.sqrt(((p0 - p1) ** 2).sum())
        print(f"  t={t}: distance between the two ground-truth cells = {dist:.2f} um")


if __name__ == "__main__":
    main()