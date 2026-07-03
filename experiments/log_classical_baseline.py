"""Log the classical baseline's tracking result as an experiment record.

Run: python -m experiments.log_classical_baseline
"""
import json
from datetime import date
from pathlib import Path

EXPERIMENT_DIR = Path(f"experiments/{date.today()}_classical_baseline")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

config = {
    "sample": "44b6_0113de3b",
    "t_range": [27, 75],
    "segmentation": "otsu + watershed (scikit-image)",
    "linking": "nearest-neighbor, Hungarian assignment, physical-distance gated",
    "max_link_distance_um": 6.0,
}

metrics = {
    "labeled_frames": 49,
    "distinct_track_ids_for_ground_truth_cell": 19,
    "id_switch_rate": 18 / 48,  # switches / possible frame-to-frame transitions
    "note": (
        "ID switches driven by global Hungarian assignment competition in "
        "dense tissue, not gate distance — segmentation-to-groundtruth "
        "distances were identical across gate values 6um and 15um. "
        "Motivates motion-aware linking (Week 5) over further gate tuning."
    ),
}

(EXPERIMENT_DIR / "config.json").write_text(json.dumps(config, indent=2))
(EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Logged to {EXPERIMENT_DIR}")