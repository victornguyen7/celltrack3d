"""Build the hero GIF: tracked ground-truth cell trajectory over the raw MIP,
using the best pipeline (DL segmentation + LapTrack).

Run: python src/visualization/hero_animation.py
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from laptrack import LapTrack
from skimage.measure import regionprops
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.loaders import load_volume, load_tracks
from src.utils.constants import VOXEL_SCALE_UM

SAMPLE = "44b6_0113de3b"
VOLUME_PATH = f"data/raw/train/{SAMPLE}.zarr"
GEFF_PATH = f"data/raw/train/{SAMPLE}.geff"
DL_LABELS_DIR = Path(f"data/processed/{SAMPLE}_cellpose_labels")
T_START, T_END = 27, 75
MAX_LINK_DISTANCE_UM = 6.0


def build_detections_df(t_start, t_end):
    scale = np.array(VOXEL_SCALE_UM)
    rows = []
    for t in range(t_start, t_end + 1):
        labels = np.load(DL_LABELS_DIR / f"t{t:03d}.npy")
        for p in regionprops(labels):
            z, y, x = p.centroid
            zs, ys, xs = np.array([z, y, x]) * scale
            rows.append({"frame": t, "z": z, "y": y, "x": x,
                         "z_um": zs, "y_um": ys, "x_um": xs})
    return pd.DataFrame(rows)


def track_best_pipeline(t_start, t_end):
    df = build_detections_df(t_start, t_end)
    lt = LapTrack(
        track_cost_cutoff=MAX_LINK_DISTANCE_UM ** 2, # type: ignore
        splitting_cost_cutoff=MAX_LINK_DISTANCE_UM ** 2, # type: ignore
    )
    track_df, _, _ = lt.predict_dataframe(
        df, coordinate_cols=["z_um", "y_um", "x_um"], only_coordinate_cols=False,
    )
    return track_df.reset_index()


def find_gt_track_id(track_df, nodes_df, t_start, t_end):
    """Which predicted track_id matches the ground-truth cell in each frame?"""
    scale = np.array(VOXEL_SCALE_UM)
    labeled = nodes_df[nodes_df["t"].between(t_start, t_end)].sort_values("t")
    matched_ids = []
    for _, gt in labeled.iterrows():
        frame_preds = track_df[track_df["frame"] == gt["t"]]
        if len(frame_preds) == 0:
            matched_ids.append(None)
            continue
        gt_pos = np.array([gt["z"], gt["y"], gt["x"]]) * scale
        pred_pos = frame_preds[["z", "y", "x"]].to_numpy() * scale
        dists = np.sqrt(((pred_pos - gt_pos) ** 2).sum(axis=1))
        matched_ids.append(frame_preds.iloc[dists.argmin()]["track_id"])
    return matched_ids


def main():
    volume = load_volume(VOLUME_PATH)
    nodes_df, edges_df = load_tracks(GEFF_PATH)

    print("Re-running DL+LapTrack pipeline for visualization...")
    track_df = track_best_pipeline(T_START, T_END)
    matched_ids = find_gt_track_id(track_df, nodes_df, T_START, T_END)
    print(f"{len(matched_ids)} frames to animate")

    fig, ax = plt.subplots(figsize=(8, 8))
    trail_x, trail_y = [], []

    def update(i):
        ax.clear()
        t = T_START + i
        frame = np.asarray(volume[t])
        mip = frame.max(axis=0)
        ax.imshow(mip, cmap="gray")

        frame_dets = track_df[track_df["frame"] == t]
        ax.scatter(frame_dets["x"], frame_dets["y"], s=8, c="cyan", alpha=0.25)

        current_id = matched_ids[i] if i < len(matched_ids) else None
        if current_id is not None:
            row = frame_dets[frame_dets["track_id"] == current_id]
            if len(row):
                trail_x.append(row["x"].values[0])
                trail_y.append(row["y"].values[0])
        ax.plot(trail_x, trail_y, "-", c="red", linewidth=1.5)
        if trail_x:
            ax.scatter([trail_x[-1]], [trail_y[-1]], s=100, facecolors="none",
                       edgecolors="red", linewidths=2)

        switched = i > 0 and matched_ids[i] != matched_ids[i - 1]
        title = f"{SAMPLE} — t={t} — tracked cell (red)"
        if switched:
            title += "  [ID SWITCH]"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ani = animation.FuncAnimation(fig, update, frames=len(matched_ids), interval=150)
    out_path = "reports/hero_tracking.gif"
    ani.save(out_path, writer="pillow", fps=6)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()