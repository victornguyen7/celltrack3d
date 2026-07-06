"""Check whether the Week 5 tracking result (DL masks + LapTrack) generalizes
beyond the first sample. Handles multiple labeled chains per sample.

Run: python src/tracking/validate_multi_sample.py
(after downloading + extracting both new samples' label zips into data/processed/)
"""
from pathlib import Path

import networkx as nx
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

MAX_LINK_DISTANCE_UM = 6.0

SAMPLES = [
    {"name": "44b6_0113de3b", "t_start": 27, "t_end": 75},  # original Week 5 sample
    {"name": "44b6_0b24845f", "t_start": 11, "t_end": 50},
    {"name": "6bba_05b6850b", "t_start": 0, "t_end": 99},
]


def build_detections_df(sample: str, t_start: int, t_end: int) -> pd.DataFrame:
    labels_dir = Path(f"data/processed/{sample}_cellpose_labels")
    scale = np.array(VOXEL_SCALE_UM)
    rows = []
    for t in range(t_start, t_end + 1):
        labels = np.load(labels_dir / f"t{t:03d}.npy")
        for p in regionprops(labels):
            z, y, x = p.centroid
            zs, ys, xs = np.array([z, y, x]) * scale
            rows.append({"frame": t, "label": p.label, "z": z, "y": y, "x": x,
                         "z_um": zs, "y_um": ys, "x_um": xs})
    return pd.DataFrame(rows)


def get_chains(edges_df: pd.DataFrame) -> list:
    g = nx.DiGraph()
    g.add_edges_from(edges_df[["source_id", "target_id"]].itertuples(index=False))
    return list(nx.weakly_connected_components(g))


def evaluate_chain(track_df, nodes_df, chain_node_ids, t_start, t_end):
    labeled = nodes_df[
        nodes_df["node_id"].isin(chain_node_ids)
        & nodes_df["t"].between(t_start, t_end)
    ].sort_values("t")
    scale = np.array(VOXEL_SCALE_UM)
    matches = []
    for _, gt in labeled.iterrows():
        frame_preds = track_df[track_df["frame"] == gt["t"]]
        if len(frame_preds) == 0:
            continue
        gt_pos = np.array([gt["z"], gt["y"], gt["x"]]) * scale
        pred_pos = frame_preds[["z", "y", "x"]].to_numpy() * scale
        dists = np.sqrt(((pred_pos - gt_pos) ** 2).sum(axis=1))
        best = frame_preds.iloc[dists.argmin()]
        matches.append((int(gt["t"]), int(best["track_id"]), float(dists.min())))
    return matches


def main():
    results = []
    for spec in SAMPLES:
        sample, t_start, t_end = spec["name"], spec["t_start"], spec["t_end"]
        print(f"\n=== {sample} (t={t_start}..{t_end}) ===")

        geff_path = f"data/raw/train/{sample}.geff"
        nodes_df, edges_df = load_tracks(geff_path)
        chains = get_chains(edges_df)

        df = build_detections_df(sample, t_start, t_end)
        print(f"{len(df)} detections, {len(chains)} labeled chain(s)")

        lt = LapTrack(
            track_cost_cutoff=MAX_LINK_DISTANCE_UM ** 2, # type: ignore
            splitting_cost_cutoff=MAX_LINK_DISTANCE_UM ** 2, # type: ignore
        )
        track_df, split_df, merge_df = lt.predict_dataframe(
            df, coordinate_cols=["z_um", "y_um", "x_um"], only_coordinate_cols=False,
        )
        track_df = track_df.reset_index()

        for i, chain in enumerate(chains):
            matches = evaluate_chain(track_df, nodes_df, chain, t_start, t_end)
            if len(matches) < 2:
                continue
            track_ids_seen = sorted(set(m[1] for m in matches))
            switches = len(track_ids_seen) - 1
            n_transitions = len(matches) - 1
            rate = switches / n_transitions
            print(f"  chain {i}: {len(matches)} frames, "
                  f"{switches}/{n_transitions} switches = {rate:.1%}")
            results.append({"sample": sample, "chain": i, "frames": len(matches),
                             "switches": switches, "transitions": n_transitions,
                             "switch_rate": rate})

    results_df = pd.DataFrame(results)
    print("\n=== Summary across all samples/chains ===")
    print(results_df.to_string(index=False))
    overall_rate = results_df["switches"].sum() / results_df["transitions"].sum()
    print(f"\nOverall pooled ID switch rate: {overall_rate:.1%}")


if __name__ == "__main__":
    main()