"""Access the Biohub cell-tracking competition data.

Strategy for this project: prototype locally against ONE sample pair
(~a few hundred MB), then scale the full 87.61GB pipeline run on a Kaggle
Notebook (data is pre-mounted there, no download needed, and free GPU is
available).

Auth — pick ONE (see https://github.com/Kaggle/kagglehub#authenticate):
  1. kagglehub.login()          -> interactive username/token prompt
  2. ~/.kaggle/kaggle.json       -> standard Kaggle API token file
  3. env vars KAGGLE_USERNAME / KAGGLE_KEY

Note: you must have joined the competition and accepted its rules on
kaggle.com in your browser first, or any download will be rejected even
with valid credentials.

Confirmed sample layout (from the Kaggle Data tab):
  train/{embryo_id}_{hash}.zarr   -- image volume, shape (T,Z,Y,X) uint16
  train/{embryo_id}_{hash}.geff   -- ground-truth tracking graph
One paired sample is small (a few hundred MB), unlike the full dataset.
"""

from pathlib import Path

import kagglehub

COMPETITION = "biohub-cell-tracking-during-development"

# One confirmed real sample pair, seen in the Data tab's file browser.
SAMPLE_ZARR = "train/44b6_0113de3b.zarr"
SAMPLE_GEFF = "train/44b6_0113de3b.geff"


def download_subset(file_path: str, output_dir: str | Path = "data/raw") -> Path:
    """Download a single file/folder for local prototyping.

    `file_path` must match a name/prefix from:
        kaggle competitions files -c biohub-cell-tracking-during-development
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.competition_download(
        COMPETITION, path=file_path, output_dir=str(output_dir)
    )
    print(f"Downloaded {file_path} to: {path}")
    return Path(path)


GEFF_NODE_PROPS = ("t", "z", "y", "x")


def download_sample_pair(output_dir: str | Path = "data/raw") -> tuple[Path, Path]:
    """Download one train sample's image volume + ground truth graph."""
    zarr_dir = download_zarr_array(SAMPLE_ZARR, array_path="0", output_dir=output_dir)
    geff_dir = download_geff(SAMPLE_GEFF, output_dir=output_dir)
    return zarr_dir, geff_dir


def download_geff(store_relpath: str, output_dir: str | Path = "data/raw") -> Path:
    """Download a full .geff ground-truth graph store.

    Layout confirmed via the Kaggle Data tab file browser:
        zarr.json
        edges/zarr.json, edges/ids (array), edges/props/zarr.json
        nodes/zarr.json, nodes/ids (array), nodes/props/zarr.json
        nodes/props/{t,z,y,x}/zarr.json + values (array)
    Reuses download_zarr_array() for every actual array — it only needs
    metadata + chunk math, not a directory listing.
    """
    download_subset(f"{store_relpath}/zarr.json", output_dir)

    download_subset(f"{store_relpath}/edges/zarr.json", output_dir)
    download_zarr_array(store_relpath, array_path="edges/ids", output_dir=output_dir)
    try:
        download_subset(f"{store_relpath}/edges/props/zarr.json", output_dir)
    except Exception as e:
        print(f"(no edge props, continuing: {e})")

    download_subset(f"{store_relpath}/nodes/zarr.json", output_dir)
    download_zarr_array(store_relpath, array_path="nodes/ids", output_dir=output_dir)
    download_subset(f"{store_relpath}/nodes/props/zarr.json", output_dir)
    for prop in GEFF_NODE_PROPS:
        download_subset(f"{store_relpath}/nodes/props/{prop}/zarr.json", output_dir)
        download_zarr_array(
            store_relpath, array_path=f"nodes/props/{prop}/values", output_dir=output_dir
        )

    return Path(output_dir) / store_relpath


def download_zarr_array(
    store_relpath: str, array_path: str = "0", output_dir: str | Path = "data/raw"
) -> Path:
    """Download one Zarr v3 array's metadata + every chunk it has.

    Works without listing any files: reads shape/chunk_shape from the
    array's own zarr.json, computes every chunk's path, and downloads
    each one individually. Only dimension 0 (time) is expected to have
    more than one chunk for this dataset — see Section 4 of planning.md.
    """
    import json

    # Best-effort: root group metadata (some zarr writers omit it).
    try:
        download_subset(f"{store_relpath}/zarr.json", output_dir)
    except Exception as e:
        print(f"(no root zarr.json, continuing: {e})")

    meta_remote = f"{store_relpath}/{array_path}/zarr.json"
    meta_local = download_subset(meta_remote, output_dir)
    meta = json.loads(Path(meta_local).read_text())

    shape = meta["shape"]
    chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
    n_chunks = [-(-s // c) for s, c in zip(shape, chunk_shape)]  # ceil division
    print(f"{store_relpath}/{array_path}: shape={shape}, chunk_shape={chunk_shape} "
          f"-> chunks/dim={n_chunks}")

    for t in range(n_chunks[0]):
        coords = [t] + [0] * (len(shape) - 1)
        chunk_remote = f"{store_relpath}/{array_path}/c/{'/'.join(map(str, coords))}"
        download_subset(chunk_remote, output_dir)

    return Path(output_dir) / store_relpath


def download_raw_data(output_dir: str | Path = "data/raw") -> Path:
    """Download the FULL competition archive (87.61GB).

    Only run this locally if you really want the whole dataset on disk.
    For the full-scale pass, prefer a Kaggle Notebook instead — the data
    is already mounted there under /kaggle/input/, no download required.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.competition_download(COMPETITION, output_dir=str(output_dir))
    print(f"Downloaded to: {path}")
    return Path(path)


if __name__ == "__main__":
    download_sample_pair()
