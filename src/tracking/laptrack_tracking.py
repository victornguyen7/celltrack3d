"""Motion-aware tracking via LapTrack on precomputed DL (Cellpose) masks.

Run: python src/tracking/laptrack_tracking.py
"""
import numpy as np
import pandas as pd
from laptrack import LapTrack
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

MAX_LINK_DISTANCE_UM = 6.0  # same gate as the two previous baselines


def build_detections_df(t_start: int, t_end: int) -> pd.DataFrame:
    scale = np.array(VOXEL_SCALE_UM)
    rows = []
    for t in range(t_start, t_end + 1):
        labels = np.load(DL_LABELS_DIR / f"t{t:03d}.npy")
        for p in regionprops(labels):
            z, y, x = p.centroid
            zs, ys, xs = np.array([z, y, x]) * scale  # physical-µm coords
            rows.append({"frame": t, "label": p.label, "z": z, "y": y, "x": x,
                         "z_um": zs, "y_um": ys, "x_um": xs})
    return pd.DataFrame(rows)


def evaluate_against_labeled_chain(predicted, nodes_df, t_start, t_end):
    """predicted must have columns: frame, track_id, z, y, x (voxel units)."""
    labeled = nodes_df[(nodes_df["t"] >= t_start) & (nodes_df["t"] <= t_end)]
    scale = np.array(VOXEL_SCALE_UM)
    matches = []
    for _, gt in labeled.iterrows():
        frame_preds = predicted[predicted["frame"] == gt["t"]]
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
    print("Building detections dataframe from DL masks...")
    df = build_detections_df(t_start, t_end)
    print(f"{len(df)} total detections across {t_end - t_start + 1} frames")

    lt = LapTrack(
        track_cost_cutoff=MAX_LINK_DISTANCE_UM ** 2,       # squared distance, per LapTrack's API # type: ignore
        splitting_cost_cutoff=MAX_LINK_DISTANCE_UM ** 2,   # allow cell divisions # type: ignore
    )
    track_df, split_df, merge_df = lt.predict_dataframe(
        df, coordinate_cols=["z_um", "y_um", "x_um"], only_coordinate_cols=False,
    )
    track_df = track_df.reset_index()

    matches = evaluate_against_labeled_chain(track_df, nodes_df, t_start, t_end)
    track_ids_seen = sorted(set(m[1] for m in matches))
    switches = len(track_ids_seen) - 1
    n_transitions = len(matches) - 1
    print(f"\nGround-truth cell matched to {len(track_ids_seen)} distinct predicted "
          f"track ID(s) across {len(matches)} labeled frames")
    print(f"ID switch rate: {switches}/{n_transitions} = {switches / n_transitions:.1%}")
    print("(classical+classical-seg: 37.5%, classical-linker+DL-seg: 31.2%)")
    for t, tid, dist in matches:
        print(f"  t={t}: predicted track_id={tid}, distance={dist:.2f} um")


if __name__ == "__main__":
    main()