# celltrack3d — Project Planning

**3D + time cell detection and tracking in zebrafish embryo light-sheet microscopy**

- **Goal:** portfolio-ready, end-to-end pipeline
- **Approach:** hybrid — classical CV baseline first, then a deep-learning upgrade layered on top
- **Timeline:** 6 weeks, structured in weekly milestones
- **Data:** Kaggle — [Biohub: Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

---

## 1. Project Overview

Cells in a developing embryo divide, move, and rearrange in three dimensions over time. Turning raw light-sheet microscopy recordings into a usable record of "which cell went where, and which cell it came from" is one of the core bottlenecks in developmental biology. This project builds a full pipeline — from raw volumetric time-lapse data to evaluated, visualized cell tracks — using a real, currently-open research dataset as the driver.

The dataset comes from a Kaggle competition launched by the **Chan Zuckerberg Biohub** (Loïc Royer's group) on June 29, 2026: detect and track zebrafish embryo cells through 3D space and time, using high-resolution light-sheet microscopy videos. It's a code competition with a $60,000 prize pool and an entry deadline of September 22, 2026. The same group published **Ultrack** (Nature Methods, 2025), an AI-powered cell segmentation/tracking algorithm that sets the current bar for the field — a natural "compare yourself to the state of the art" reference point for this project.

This project does **not** need to win the competition. The competition is the excuse to build something real; the deliverable is a clean, well-documented, portfolio-grade pipeline.

---

## 2. Objectives & Success Criteria

| Objective | What "done" looks like |
|---|---|
| Understand the biology | Can explain in plain language what light-sheet microscopy captures and why 3D cell tracking matters for developmental biology |
| Working classical baseline | Segmentation + tracking pipeline runs end-to-end on a subset of the data using only classical CV |
| Working DL upgrade | Same pipeline with a deep-learning segmentation model (and optionally a smarter tracker) swapped in, with a head-to-head comparison against the classical baseline |
| Reproducibility | Anyone can clone the repo, run `pip install -r requirements.txt`, and reproduce the main result from raw data |
| Evaluation | Quantitative metrics (not just "it looks right") comparing predicted tracks to ground truth |
| Portfolio artifact | A README with a pipeline diagram, results table, and a GIF/render of tracked cells moving in 3D — something you'd genuinely show in an interview |

---

## 3. Background / Domain Context

- **Organism & imaging:** zebrafish embryos are a standard model organism for studying vertebrate development — transparent, fast-developing, and well suited to light-sheet microscopy, which images a thin plane of the sample at a time to build up 3D volumes rapidly with low phototoxicity. A recording is effectively a 4D array: `(time, z, y, x)`, sometimes with an extra channel dimension.
- **Why tracking is hard:** cells are densely packed, divide (one track becomes two), touch and separate, and move in and out of focus. Segmentation errors compound into tracking errors, so the two problems have to be thought about together, not in isolation.
- **The established convention:** the field's reference benchmark is the [Cell Tracking Challenge](https://celltrackingchallenge.net/), which scores submissions with a **SEG** metric (segmentation overlap accuracy) and a **TRA** metric (tracking accuracy, based on the cost of correcting a predicted track graph into the ground-truth graph). This project will implement metrics in that spirit; the *exact* Kaggle scoring rule should be double-checked directly on the competition's Evaluation page once you're logged in, since that page is JS-rendered and I couldn't confirm the precise formula from outside.
- **Reference tooling from the same lab:** [Ultrack](https://github.com/royerlab/ultrack) (segmentation-based, multi-hypothesis cell tracking) and [inTRACKtive](https://github.com/royerlab/inTRACKtive) (a Zarr-based viewer for tracking results) are both from the Royer Group. inTRACKtive documents a simple, portable track table schema — columns `track_id, t, z, y, x, parent_track_id` (z and parent_track_id optional) — which is a good default schema to adopt for this project's own track outputs, and likely close to what the competition's ground truth uses.

---

## 4. Data Source

```python
import kagglehub

# Download latest version
path = kagglehub.competition_download('biohub-cell-tracking-during-development')
print("Path to competition files:", path)
```

**What to expect (confirm against the actual download — see Open Questions):**
- Raw imaging data: 3D+time light-sheet microscopy volumes of developing zebrafish embryos. Given the source lab's usual tooling, expect either chunked volumetric arrays (Zarr/OME-Zarr) or TIFF stacks, likely large enough that lazy/chunked loading matters.
- Ground truth: cell positions and lineage over time, most plausibly as a CSV/Parquet table of `track_id, t, z, y, x, parent_track_id` (parent_track_id encodes divisions).
- License: the dataset has been described as released under an open (CC0) license.

**Data folder roles:**
- `data/raw/` — the untouched kagglehub download. Never edit these files directly.
- `data/processed/` — cropped/denoised/normalized volumes, cached segmentation masks, cached track tables. Anything regenerable from `raw/` + code belongs here, not in git.
- `data/external/` — anything brought in for comparison, e.g. a small Cell Tracking Challenge dataset for sanity-checking your metrics implementation, or Ultrack's own output on a shared crop for benchmarking.

---

## 5. Repository Structure

```
celltrack3d/
│── data/
│   ├── raw/            # untouched kagglehub download
│   ├── processed/      # cached intermediate artifacts (masks, cropped volumes, track tables)
│   └── external/       # reference data for benchmarking (e.g. CTC sample, Ultrack output)
│
│── notebooks/           # exploratory + narrative notebooks (numbered, one topic each)
│
│── src/
│   ├── io/              # load raw volumes & tracks, kagglehub wrapper, format conversion
│   ├── preprocessing/    # denoising, normalization, cropping/downsampling
│   ├── visualization/    # 2D slice/MIP plots, 3D renders, track overlays, GIF export
│   ├── segmentation/     # classical (threshold+watershed) and DL (Cellpose/StarDist) segmenters
│   ├── tracking/         # classical (NN/Hungarian, trackpy) and upgraded (laptrack, motion model) linkers
│   ├── evaluation/       # detection & tracking metrics, scorecards
│   └── utils/            # config loading, logging, path helpers
│
│── experiments/          # one subfolder per run: config used + resulting metrics/logs
│
│── models/               # trained/fine-tuned model weights (gitignored, tracked via README pointer)
│
│── reports/              # final write-up, figures, comparison tables
│
│── requirements.txt
│── README.md
```

Each `src/` module should be import-only (no top-level execution) so notebooks and scripts can both call into it — that's what makes the difference between "notebook full of scattered code" and "portfolio-grade codebase."

---

## 6. Pipeline Overview

```
Raw volumes (zarr/tiff, 4D)         Ground-truth tracks (csv)
        │                                    │
        ▼                                    │
 [src/io]  load & cache                       │
        │                                    │
        ▼                                    │
 [src/preprocessing]  denoise, normalize, crop │
        │                                    │
        ▼                                    │
 [src/segmentation]  classical → DL masks       │
        │                                    │
        ▼                                    │
 [src/tracking]  classical → upgraded linking    │
        │                                    │
        ▼                                    ▼
              [src/evaluation]  compare to ground truth
                        │
                        ▼
              [src/visualization]  figures, 3D renders, GIFs
                        │
                        ▼
        [reports/] + [README.md]  portfolio write-up
```

---

## 7. Detailed Step Breakdown

### Step 1 — Data acquisition & exploration
- **What happens:** download via `kagglehub`, inventory the files (shapes, dtypes, number of timepoints/z-slices, channels), inspect the ground-truth track table's schema, visualize a few raw frames as max-intensity projections.
- **Where:** `src/io/` for loaders, `notebooks/01_eda.ipynb` for the narrative.
- **Tools:** `kagglehub`, `tifffile` or `zarr`, `pandas`, `matplotlib`.
- **Output:** a short data dictionary documenting the confirmed schema (update Section 4 once verified).

### Step 2 — Preprocessing
- **What happens:** denoise (Gaussian/median filter), normalize intensity (percentile clipping / contrast stretch), optionally crop to a manageable sub-volume and downsample for fast iteration during development.
- **Where:** `src/preprocessing/`.
- **Tools:** `scipy.ndimage`, `scikit-image`.
- **Output:** cached arrays in `data/processed/`.

### Step 3 — Segmentation (classical baseline, Weeks 2–3)
- **What happens:** Otsu/adaptive thresholding → distance transform → watershed to separate touching cells → morphological cleanup → connected-component labeling → `regionprops` for centroids/volume/intensity features.
- **Where:** `src/segmentation/classical.py`.
- **Tools:** `scikit-image` (`filters`, `segmentation.watershed`, `morphology`, `measure`).
- **Output:** per-frame instance masks + a feature table of detected cells.

### Step 4 — Tracking (classical baseline, Week 3)
- **What happens:** link detected centroids across consecutive frames. Simplest version: nearest-neighbor matching with a Hungarian/linear-sum-assignment solver and a max-distance gate to reject bad matches. `trackpy` is a solid drop-in for this since it's built for exactly this kind of particle-linking problem in 3D.
- **Where:** `src/tracking/classical.py`.
- **Tools:** `scipy.optimize.linear_sum_assignment`, `trackpy`.
- **Output:** a track table in the `track_id, t, z, y, x, parent_track_id` schema (matches Section 3's reference convention).

### Step 5 — Evaluation harness (Week 3)
- **What happens:** compare predicted tracks/masks to ground truth. Start simple: detection precision/recall at a distance threshold, fraction of correctly linked consecutive pairs, number of ID switches, track length distribution vs. ground truth. Stretch: implement a TRA/AOGM-style graph-matching metric to mirror the Cell Tracking Challenge convention.
- **Where:** `src/evaluation/metrics.py`.
- **Output:** a scorecard (dict/CSV) that every later phase gets compared against.

### Step 6 — Segmentation upgrade: deep learning (Week 4)
- **What happens:** swap the classical segmenter for a pretrained cell-segmentation model, run it on the same frames, and compare masks/metrics against the classical baseline. **Cellpose** is the recommended starting point (PyTorch-based, has a 3D mode, easy to run without fine-tuning first); **StarDist** (TensorFlow-based, star-convex shapes) is a reasonable alternative especially if the cells/nuclei are roughly round. Pick one as primary to avoid installing two heavy DL backends side by side.
- **Where:** `src/segmentation/deep.py`.
- **Output:** DL masks + an updated scorecard, and a side-by-side visual comparison against classical masks.

### Step 7 — Tracking upgrade (Week 5)
- **What happens:** feed the (better) DL masks into a smarter linker — e.g. `laptrack` (linear-assignment-problem based, lightweight, handles divisions) or a simple constant-velocity motion model to improve gating. Optional stretch: benchmark a small crop against Ultrack's own output as an "how close did I get to the state of the art" comparison.
- **Where:** `src/tracking/advanced.py`.
- **Output:** updated track table + updated scorecard, full classical-vs-hybrid comparison.

### Step 8 — Visualization (ongoing, polished in Week 6)
- **What happens:** 2D max-intensity-projection overlays with track trails (matplotlib, cheap and always works); interactive 3D inspection for exploration (napari, run locally) and/or a notebook-embeddable interactive 3D scatter of trajectories (plotly) for something that renders in a shared notebook without a GUI session; export a short GIF/MP4 of cells moving and dividing over time for the README.
- **Where:** `src/visualization/`.
- **Output:** figures for `reports/`, one hero GIF for `README.md`.

### Step 9 — Reporting & portfolio packaging (Week 6)
- **What happens:** write up methodology and results in `reports/`, finalize `README.md` with the pipeline diagram, results table (classical vs. DL), and the hero visualization.
- **Output:** the actual portfolio artifact.

---

## 8. Evaluation Metrics

| Metric | What it measures | Phase introduced |
|---|---|---|
| Detection precision/recall | Are predicted cell centroids close enough to ground-truth centroids at each timepoint? | Week 3 |
| Track linking accuracy | Fraction of consecutive-frame links that match ground truth | Week 3 |
| ID switches | How often a track's identity gets swapped with another cell's | Week 3 |
| Segmentation overlap (IoU / SEG-style) | How well predicted masks overlap ground-truth masks, once mask-level ground truth is available | Week 4 |
| TRA/AOGM-style graph accuracy | Cost to correct the predicted track graph into the ground-truth graph — the Cell Tracking Challenge standard | Stretch, Week 5–6 |

---

## 9. Milestones & Timeline (6 weeks)

| Week | Focus | Key deliverables |
|---|---|---|
| 1 | Setup + data exploration | Repo scaffolded, data downloaded & inventoried, `01_eda.ipynb`, confirmed data schema |
| 2 | Preprocessing + classical segmentation | `src/preprocessing`, classical segmenter, mask overlay visualizations |
| 3 | Classical tracking + evaluation harness | `src/tracking/classical.py`, `src/evaluation/metrics.py`, baseline scorecard |
| 4 | DL segmentation upgrade | Cellpose/StarDist integrated, segmentation comparison vs. classical |
| 5 | DL/advanced tracking upgrade | `laptrack`-based linker, full hybrid pipeline run, updated scorecard |
| 6 | Visualization + reporting + polish | 3D trajectory visuals, hero GIF, final `reports/` write-up, polished README |

This is a plan, not a contract — some weeks will run long (Week 4 especially, DL install/debug tends to eat time) and that's fine as long as the classical baseline from Weeks 1–3 stays a solid, working fallback the whole way through.

---

## 10. Experiment Tracking Convention

Each pipeline run gets its own folder under `experiments/`, e.g.:

```
experiments/
└── 2026-07-14_classical_baseline/
    ├── config.yaml       # what was run
    ├── metrics.json       # scorecard from src/evaluation
    └── notes.md            # what changed, what you observed
```

This is a small habit that makes the eventual "compare classical vs. DL" story trivial to write, and reads well in an interview as evidence of methodical experimentation.

---

## 11. Environment & Dependencies (draft `requirements.txt`)

```
numpy
pandas
matplotlib
scipy
scikit-image
scikit-learn
trackpy
zarr
dask
tifffile
plotly
napari          # optional — interactive 3D inspection, local GUI session
cellpose        # DL segmentation (Week 4); brings in torch
laptrack        # DL/graph-based tracking upgrade (Week 5)
kagglehub
tqdm
pyyaml
jupyterlab
pytest
```

Pin exact versions once the environment is built and working — don't lock versions speculatively before you've confirmed compatibility, especially around `cellpose`'s `torch` dependency.

---

## 12. Deliverables / Portfolio Checklist

- [ ] Modular, docstring'd `src/` codebase (no stray notebook-only logic for anything reused twice)
- [ ] 4–5 focused notebooks: EDA, segmentation dev, tracking dev, evaluation, final demo
- [ ] `requirements.txt` that reproduces the environment from scratch
- [ ] `README.md`: problem statement, pipeline diagram, results table, hero GIF/render
- [ ] `reports/`: methodology + classical-vs-DL comparison write-up
- [ ] `experiments/`: versioned runs showing iterative progress
- [ ] A handful of `pytest` unit tests for core logic (linking function, metric calculations)
- [ ] (Optional) short blog-style post on the biological motivation + technical approach

---

## 13. Open Questions / Assumptions to Verify After Downloading Data

- **Exact raw file format** — likely Zarr/OME-Zarr or TIFF stacks given the source lab's usual tooling, but confirm actual format and dimension order once `kagglehub` finishes downloading.
- **Exact ground-truth track schema** — assumed `track_id, t, z, y, x, parent_track_id` based on the same lab's inTRACKtive convention; confirm real column names.
- **Official Kaggle evaluation metric** — the competition's Evaluation page is JS-rendered and wasn't fetchable directly; check it in-browser and update Section 8 accordingly if it differs from the TRA/AOGM-style metric assumed here.
- **Compute budget** — light-sheet time-lapse volumes can be large; decide early whether you're working locally, on Colab, or in Kaggle Notebooks, and set a downsampling/crop strategy for fast dev iteration accordingly.
- **Whether to actually submit to the leaderboard** — not required for the portfolio goal, but the entry deadline (Sept 22, 2026) leaves room if you want to treat it as a stretch goal.

---

## 14. Stretch Goals

- Submit a real entry to the Kaggle leaderboard.
- Benchmark your hybrid pipeline against Ultrack's output on a shared crop — "how close did a 6-week solo project get to the lab's own state-of-the-art tool" is a strong portfolio narrative.
- Small interactive dashboard (Streamlit or a napari plugin) to scrub through time and inspect tracks.
- Short technical write-up/blog post pairing the biological motivation with the engineering approach.

---

## 15. References

- Biohub — Cell Tracking During Development (Kaggle competition)
- Ultrack — Royer Group, published in *Nature Methods*, 2025
- inTRACKtive — Zarr-based cell tracking visualization tool, Royer Group
- Cell Tracking Challenge — celltrackingchallenge.net (SEG/TRA metric convention)
- Cellpose, StarDist — general-purpose deep-learning cell segmentation
- trackpy, laptrack — Python particle/cell linking libraries