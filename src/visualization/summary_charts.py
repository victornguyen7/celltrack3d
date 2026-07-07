"""Build the two summary figures for the final report:
1. Pipeline progression (classical -> DL seg -> DL+LapTrack) on sample 1
2. Multi-sample variance -- the honest generalization picture

Run: python src/visualization/summary_charts.py
"""
import matplotlib.pyplot as plt
import numpy as np

# --- Figure 1: pipeline progression on the primary sample ---
pipelines = ["Classical\n(Otsu+watershed,\nHungarian)",
             "DL segmentation\n(Cellpose-SAM,\nHungarian)",
             "Full hybrid\n(Cellpose-SAM,\nLapTrack)"]
switch_rates = [37.5, 31.2, 16.7]
colors = ["#c0392b", "#e67e22", "#27ae60"]

fig1, ax1 = plt.subplots(figsize=(7, 5))
bars = ax1.bar(pipelines, switch_rates, color=colors)
for bar, rate in zip(bars, switch_rates):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
              f"{rate}%", ha="center", fontsize=11, fontweight="bold")
ax1.set_ylabel("ID switch rate (%)")
ax1.set_title("Tracking accuracy improves at each pipeline stage\n(sample 44b6_0113de3b, 49 labeled frames)")
ax1.set_ylim(0, 45)
fig1.tight_layout()
fig1.savefig("reports/summary_pipeline_progression.png", dpi=150)
print("Saved reports/summary_pipeline_progression.png")

# --- Figure 2: multi-sample variance (the honest limitation) ---
# Raw per-chain switch rates from the Week 5.5 validation run.
sample_data = {
    "44b6_0113de3b\n(1 chain)": [16.7],
    "44b6_0b24845f\n(2 chains)": [53.8, 70.0],
    "6bba_05b6850b\n(16 chains)": [5.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.9,
                                    10.0, 0.0, 8.6, 0.0, 23.5, 29.4, 23.1, 38.5],
}

fig2, ax2 = plt.subplots(figsize=(8, 5))
positions = range(len(sample_data))
box_data = list(sample_data.values())
bp = ax2.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#5dade2")
for i, data in enumerate(box_data):
    jitter = np.random.normal(0, 0.04, size=len(data))
    ax2.scatter(np.full(len(data), i) + jitter, data, color="black", s=20, zorder=3)

ax2.set_xticks(list(positions))
ax2.set_xticklabels(sample_data.keys())
ax2.set_ylabel("ID switch rate (%)")
ax2.axhline(8.1, color="gray", linestyle="--", linewidth=1, label="Overall pooled rate (8.1%)")
ax2.set_title("Generalization is uneven across samples\n(full hybrid pipeline, 18 chains total)")
ax2.legend()
fig2.tight_layout()
fig2.savefig("reports/summary_multi_sample_variance.png", dpi=150)
print("Saved reports/summary_multi_sample_variance.png")