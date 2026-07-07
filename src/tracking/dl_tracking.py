"""Track using precomputed Cellpose (DL) segmentation masks — directly
comparable to src/tracking/classical.py (same linker, same gate).

Run: python src/tracking/dl_tracking.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_tracks
from src.utils.constants import VOXEL_SCALE_UM

GEFF_PATH = "data/raw/train/44b6_0113de3b.geff"
DL_LABELS_DIR = Path("data/processed/44b6_0113de3b_cellpose_labels")

MAX_LINK_DISTANCE_UM = 6.0  # same gate as classical's second (tightened) run


def load_dl_labels(t: int) -> np.ndarray:
    return np.load(DL_LABELS_DIR / f"t{t:03d}.npy")


def detect_centroids(t: int) -> pd.DataFrame:
    labels = load_dl_labels(t)
    rows = [
        {"t": t, "z": p.centroid[0], "y": p.centroid[1], "x": p.centroid[2], "label": p.label}
        for p in regionprops(labels)
    ]
    return pd.DataFrame(rows)


def physical_distance_matrix(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    scale = np.array(VOXEL_SCALE_UM)
    pa = a[["z", "y", "x"]].to_numpy() * scale
    pb = b[["z", "y", "x"]].to_numpy() * scale
    diff = pa[:, None, :] - pb[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def link_frames(prev, curr, max_dist=MAX_LINK_DISTANCE_UM):
    if len(prev) == 0 or len(curr) == 0:
        return []
    cost = physical_distance_matrix(prev, curr)
    row_idx, col_idx = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(row_idx, col_idx) if cost[r, c] <= max_dist]


def track_range(t_start: int, t_end: int) -> pd.DataFrame:
    all_frames = {t: detect_centroids(t) for t in range(t_start, t_end + 1)}

    next_track_id = 0
    first = all_frames[t_start]
    first["track_id"] = range(next_track_id, next_track_id + len(first))
    next_track_id += len(first)

    for t in range(t_start, t_end):
        prev, curr = all_frames[t], all_frames[t + 1]
        links = link_frames(prev, curr)
        curr["track_id"] = -1
        for r, c in links:
            curr.at[curr.index[int(c)], "track_id"] = prev.iloc[r]["track_id"]
        unlinked = curr["track_id"] == -1
        n_new = int(unlinked.sum())
        curr.loc[unlinked, "track_id"] = range(next_track_id, next_track_id + n_new)
        next_track_id += n_new

    return pd.concat(all_frames.values(), ignore_index=True)


def evaluate_against_labeled_chain(predicted, nodes_df, t_start, t_end):
    labeled = nodes_df[(nodes_df["t"] >= t_start) & (nodes_df["t"] <= t_end)]
    scale = np.array(VOXEL_SCALE_UM)
    matches = []
    for _, gt in labeled.iterrows():
        frame_preds = predicted[predicted["t"] == gt["t"]]
        if len(frame_preds) == 0:
            continue
        gt_pos = np.array([gt["z"], gt["y"], gt["x"]]) * scale
        pred_pos = frame_preds[["z", "y", "x"]].to_numpy() * scale
        dists = np.sqrt(((pred_pos - gt_pos) ** 2).sum(axis=1))
        best = frame_preds.iloc[dists.argmin()]
        matches.append((int(gt["t"]), int(best["track_id"]), float(dists.min())))
    return matches


def main():
    nodes_df, edges_df = load_tracks(GEFF_PATH)

    t_start, t_end = 0, 99
    print(f"Loading DL labels and linking across t={t_start}..{t_end}...")
    predicted = track_range(t_start, t_end)
    print(f"Detected+linked {len(predicted)} instances total")

    matches = evaluate_against_labeled_chain(predicted, nodes_df, t_start, t_end)
    track_ids_seen = sorted(set(m[1] for m in matches))
    switches = len(track_ids_seen) - 1
    n_transitions = len(matches) - 1
    print(f"\nGround-truth cell matched to {len(track_ids_seen)} distinct predicted "
          f"track ID(s) across {len(matches)} labeled frames")
    print(f"ID switch rate: {switches}/{n_transitions} = {switches / n_transitions:.1%}")
    print("(classical baseline for comparison: 19 IDs, 18/48 = 37.5%)")
    for t, tid, dist in matches:
        print(f"  t={t}: predicted track_id={tid}, distance={dist:.2f} um")


if __name__ == "__main__":
    main()