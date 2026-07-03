"""Quick EDA: visualize one timepoint + overlay ground-truth centroids,
and summarize track structure from the sparse annotation graph.

Run: python src/visualization/eda_overview.py
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.loaders import load_volume, load_tracks

VOLUME_PATH = "data/raw/train/44b6_0113de3b.zarr"
GEFF_PATH = "data/raw/train/44b6_0113de3b.geff"


def build_track_chains(edges_df):
    """Group nodes into connected chains (tracks) from the edge list."""
    g = nx.DiGraph()
    g.add_edges_from(edges_df[["source_id", "target_id"]].itertuples(index=False))
    return list(nx.weakly_connected_components(g))


def main():
    volume = load_volume(VOLUME_PATH)
    nodes_df, edges_df = load_tracks(GEFF_PATH)

    print(f"Volume: shape={volume.shape}, dtype={volume.dtype}")
    print(f"Labeled nodes: {len(nodes_df)}, edges: {len(edges_df)}")
    print(f"Timepoints with any labeled node: {sorted(nodes_df['t'].unique())}")

    chains = build_track_chains(edges_df)
    lengths = sorted((len(c) for c in chains), reverse=True)
    print(f"Distinct track chains: {len(chains)}, lengths: {lengths}")

    # Visualize the timepoint with the most labeled nodes.
    t = int(nodes_df["t"].value_counts().idxmax())
    frame = np.asarray(volume[t])  # (Z, Y, X)
    mip = frame.max(axis=0)  # max-intensity projection over z

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mip, cmap="gray")
    frame_nodes = nodes_df[nodes_df["t"] == t]
    ax.scatter(frame_nodes["x"], frame_nodes["y"], s=60, facecolors="none",
               edgecolors="red", linewidths=1.5)
    for _, row in frame_nodes.iterrows():
        ax.annotate(str(row["node_id"]), (row["x"], row["y"]),
                    color="yellow", fontsize=7)
    ax.set_title(f"44b6_0113de3b — timepoint {t} (MIP) with labeled cells")
    ax.axis("off")

    out_path = "reports/eda_sample_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()