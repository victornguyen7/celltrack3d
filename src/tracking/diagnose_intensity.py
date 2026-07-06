"""Check whether the failing chain's ground-truth cell is unusually dim
compared to the rest of the frame -- the likely reason it's under-detected.

Run: python src/tracking/diagnose_intensity.py
"""
import numpy as np
import networkx as nx
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_volume, load_tracks

SAMPLE = "44b6_0b24845f"
VOLUME_PATH = f"data/raw/train/{SAMPLE}.zarr"


def main():
    volume = load_volume(VOLUME_PATH)
    nodes_df, edges_df = load_tracks(f"data/raw/train/{SAMPLE}.geff")
    g = nx.DiGraph()
    g.add_edges_from(edges_df[["source_id", "target_id"]].itertuples(index=False))
    chains = list(nx.weakly_connected_components(g))
    chain0 = nodes_df[nodes_df["node_id"].isin(chains[0])].sort_values("t")

    for _, gt in chain0.iloc[::5].iterrows():  # sample every 5th frame
        t, z, y, x = int(gt["t"]), int(gt["z"]), int(gt["y"]), int(gt["x"])
        frame = np.asarray(volume[t])
        # small window around the labeled point vs. the whole frame
        z0, z1 = max(0, z - 2), min(frame.shape[0], z + 3)
        y0, y1 = max(0, y - 5), min(frame.shape[1], y + 6)
        x0, x1 = max(0, x - 5), min(frame.shape[2], x + 6)
        local_max = frame[z0:z1, y0:y1, x0:x1].max()
        print(f"t={t}: cell peak intensity={local_max}, "
              f"frame overall: median={np.median(frame):.0f}, max={frame.max()}, "
              f"p95={np.percentile(frame, 95):.0f}")


if __name__ == "__main__":
    main()