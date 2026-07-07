"""Classical tracking: nearest-neighbor linking via Hungarian assignment.

Run: python src/tracking/classical.py
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
from src.io.loaders import load_volume
from src.io.loaders import load_volume, load_tracks
from src.preprocessing.basic import preprocess_frame
from src.segmentation.classical import segment_frame
from src.utils.constants import VOXEL_SCALE_UM

VOLUME_PATH = "data/raw/train/44b6_0113de3b.zarr"
GEFF_PATH = "data/raw/train/44b6_0113de3b.geff"

MAX_LINK_DISTANCE_UM = 6.0  # ~half a nucleus diameter — reject cross-cell jumps


def detect_centroids(volume, t: int) -> pd.DataFrame:
    frame = np.asarray(volume[t])
    processed = preprocess_frame(frame)
    labels = segment_frame(processed)
    rows = []
    for p in regionprops(labels):
        z, y, x = p.centroid
        rows.append({"t": t, "z": z, "y": y, "x": x, "label": p.label})
    return pd.DataFrame(rows)


def physical_distance_matrix(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    scale = np.array(VOXEL_SCALE_UM)
    pa = a[["z", "y", "x"]].to_numpy() * scale
    pb = b[["z", "y", "x"]].to_numpy() * scale
    diff = pa[:, None, :] - pb[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def link_frames(prev: pd.DataFrame, curr: pd.DataFrame, max_dist: float = MAX_LINK_DISTANCE_UM):
    """Return list of (prev_index, curr_index) accepted links."""
    if len(prev) == 0 or len(curr) == 0:
        return []
    cost = physical_distance_matrix(prev, curr)
    row_idx, col_idx = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(row_idx, col_idx) if cost[r, c] <= max_dist]


def track_range(volume, t_start: int, t_end: int) -> pd.DataFrame:
    """Detect + link across [t_start, t_end], assigning a track_id per chain."""
    all_frames = {t: detect_centroids(volume, t) for t in range(t_start, t_end + 1)}

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
    """Check whether the one labeled cell maps to a SINGLE consistent
    predicted track_id across the whole window (no identity switch)."""
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
    volume = load_volume(VOLUME_PATH)
    nodes_df, edges_df = load_tracks(GEFF_PATH)

    t_start, t_end = 0, 99  # the long labeled chain's frame range
    print(f"Detecting + linking across t={t_start}..{t_end} (this takes a bit)...")
    predicted = track_range(volume, t_start, t_end)
    print(f"Detected+linked {len(predicted)} instances total")

    matches = evaluate_against_labeled_chain(predicted, nodes_df, t_start, t_end)
    track_ids_seen = sorted(set(m[1] for m in matches))
    print(f"\nGround-truth cell matched to {len(track_ids_seen)} distinct predicted "
          f"track ID(s) across {len(matches)} labeled frames")
    print("(1 = perfect tracking, no identity switch; >1 = track got dropped/switched)")
    print(f"track IDs seen: {track_ids_seen}")
    for t, tid, dist in matches:
        print(f"  t={t}: predicted track_id={tid}, distance={dist:.2f} um")


if __name__ == "__main__":
    main()