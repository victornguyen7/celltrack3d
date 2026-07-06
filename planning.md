# celltrack3d — Project Planning

**3D + time cell detection and tracking in zebrafish embryo light-sheet microscopy**

- **Goal:** portfolio-ready, end-to-end pipeline (not just a leaderboard script)
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

**Confirmed schema (from the Kaggle Data tab, verified):**
- **Images:** each sample is a `.zarr` (Zarr v3) directory with a single array at path `0/`, shape `(T, Z, Y, X)` — typically `(100, 64, 256, 256)`, dtype `uint16`. Chunked one timepoint at a time: `(1, 64, 256, 256)`, blosc/zstd compressed. Chunk for timepoint `t` lives at `0/c/{t}/0/0/0`; array metadata (shape/dtype/codecs) is in `0/zarr.json`.
- **Voxel scale (anisotropic):** z = 1.625 µm/voxel, y = x = 0.40625 µm/voxel — z spacing is ~4× coarser than xy. Any physical-distance calculation (watershed seed spacing, nearest-neighbor linking distance) must scale z accordingly or it will be badly wrong.
- **Ground truth (train only):** `.geff` directories (graph exchange format, also Zarr v3): `nodes/ids`, `nodes/props/{t,z,y,x}/values` (integer voxel centroids per node), `edges/ids` (shape `(N, 2)`, columns `source_id, target_id`). A track is just a chain of edges; a division is a node with two outgoing edges.
- **Annotations are sparse** — not every visible cell in every frame is labeled. `estimated_number_of_nodes` (in the `.geff` metadata) gives the true total cell count estimate per sample. This matters for evaluation (see Section 8) — precision/recall must be computed against labeled nodes only, not full detection density.
- **Embryo identity:** folder names are `{embryo_id}_{hash}` (e.g. `44b6_0113de3b`). Train/test are embryo-disjoint. `train/` has ~380+ paired `.zarr`+`.geff` samples; `test/` shown publicly is only 4 example samples (image-only, copies from train) — the real hidden test set is swapped in at submission time, roughly the same size as train.
- **Submission format:** `sample_submission.csv` columns are `id, dataset, row_type, node_id, t, z, y, x, source_id, target_id` — one CSV with mixed `node`/`edge` rows per `dataset` (sample name), i.e. a flattened version of the same graph structure as `.geff`.
- **Total size:** 87.61 GB across 24,886 files. License: CC0 (public domain).
- Each individual `.zarr`+`.geff` sample pair is only on the order of a few hundred MB — small enough to download and work with locally.

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

**Confirmed result — ID switch rate on sample `44b6_0113de3b`, ground-truth chain t=27..75 (49 labeled frames, 48 possible transitions):**

| Pipeline | Segmentation | Linking | Distinct track IDs | ID switch rate |
|---|---|---|---|---|
| Classical baseline | Otsu + watershed | Hungarian, physical-distance gated | 19 | 18/48 = 37.5% |
| DL segmentation only | Cellpose-SAM (2D+stitch) | Hungarian, physical-distance gated | 16 | 15/48 = 31.2% |
| Full hybrid (Week 5) | Cellpose-SAM (2D+stitch) | LapTrack (LAP with birth/death) | 9 | 8/48 = 16.7% |

Segmentation quality alone gave a modest improvement; switching to a proper LAP-based linker (which allows a detection to go unmatched at a fixed cost instead of forcing a same-size bipartite match) accounted for most of the gain — likely more than any implicit motion modeling. This is a reasoned inference from the mechanism, not something separately isolated by an ablation — worth stating as such in the final report.

**Multi-sample validation (Week 5.5) — does the result generalize?**

Testing the same DL-segmentation + LapTrack pipeline on 3 samples (18 labeled chains total, 1044 frame-transitions) revealed the single-sample result does **not** generalize uniformly:

| Sample | Chains | Switch rate range | Notes |
|---|---|---|---|
| `44b6_0113de3b` | 1 | 16.7% | Original Week 5 result |
| `44b6_0b24845f` | 2 | 53.8% – 70.0% | Same embryo as above, different FOV — much worse |
| `6bba_05b6850b` | 16 | 0.0% – 38.5% (mostly under 10%) | Different embryo, mostly strong, some weak chains |

Pooled overall: 8.1% — but reporting only this number would hide the real finding, which is the *variance*, not the average.

**Root-cause investigation of the `44b6_0b24845f` failure** (methodical, not guessed): tested three hypotheses in sequence against direct evidence, not assumption —
1. *Merged blobs* (segmentation fusing touching cells) — mostly ruled out; only 2/40 frames showed oversized instances near the failing cell.
2. *Simple dropout* (cell not detected at all) — partially true (15/40 frames had no detection within 15µm), but doesn't explain the persistent 10-30µm offsets in frames that *did* have a nearby detection.
3. *Low intensity* (dim cell gets missed) — not supported; the cell's peak intensity sat consistently between the frame median and 95th percentile (moderately bright, not dim).

**Honest conclusion:** the failure looks like local competition among several visually-similar, moderately-bright neighboring cells rather than one single clean, fixable mechanism. This is a legitimate limitation to report as-is — the pipeline's reliability is not uniform across samples, and the reason is more nuanced than any single hypothesis. Worth stating plainly in the final report rather than cherry-picking only the best-case sample.

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

## 13. Open Questions / Assumptions

**Resolved (confirmed via the Kaggle Data tab, July 2026):**
- ~~Exact raw file format~~ → Zarr v3, `(T,Z,Y,X)` uint16, one-timepoint chunks. See Section 4.
- ~~Ground-truth track schema~~ → `.geff` node/edge graph, not a flat CSV. See Section 4.
- ~~(t,z,y,x) axis alignment between volume and graph~~ → verified empirically on sample `44b6_0113de3b`: a labeled centroid lands directly on a bright nucleus in both the MIP and its exact z-slice.

**Still open:**
- **Official Kaggle evaluation metric** — the Data tab describes the format but not the scoring formula; check the competition's Evaluation page directly. Given the sparse-annotation graph format, it's likely some form of graph-matching accuracy (TRA/AOGM-style) restricted to labeled nodes, but confirm before building Section 8's metrics around it.
- **`zarr` library version** — this is Zarr v3 format, which needs `zarr-python >= 3.0`; pin that explicitly rather than letting pip resolve an older v2-only version.
- **`.geff` reader** — check whether the `geff` Python package (graph exchange format, used by the Royer lab tooling) is available on PyPI to read these directly, versus reading the underlying Zarr arrays by hand (`nodes/ids`, `nodes/props/.../values`, `edges/ids`) — the manual route always works as a fallback.
- **Compute budget** — confirmed, not just a guess: Cellpose-SAM 3D inference on one 100×64×256×256 frame took **1940s (~32 min) on an M-series Mac, even with MPS enabled** (`gpu=True`) — MPS didn't help much here, likely because Cellpose's mask-creation step isn't yet accelerated on Apple Silicon. At that rate, segmenting a 49-frame window would take ~26 hours locally. **Decision: all further multi-frame Cellpose runs move to the Kaggle Notebook (real GPU)** — local dev stays limited to single-frame sanity checks.
- **Whether to actually submit to the leaderboard** — optional stretch, deadline noted in Section 1.

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