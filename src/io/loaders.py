"""Load the confirmed Zarr v3 image volumes and .geff ground-truth graphs.

Image volume:
    <sample>.zarr/0   -> array, shape (T, Z, Y, X), uint16

Ground truth graph:
    <sample>.geff     -> directed graph; node attrs t,z,y,x; edges = links
                          between consecutive-frame detections of the same
                          cell (a division = a node with 2 outgoing edges)
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import zarr


def load_volume(sample_zarr_path: str | Path) -> zarr.Array:
    """Open a sample's image array lazily (no full read into RAM).

    Returns a zarr Array you can slice like a numpy array, e.g.
    `volume[t]` pulls just that timepoint's (Z, Y, X) chunk.
    """
    store = zarr.open(str(sample_zarr_path), mode="r")
    return store["0"] # type: ignore


def load_tracks(sample_geff_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a .geff ground-truth graph into (nodes_df, edges_df).

    nodes_df columns: node_id, t, z, y, x
    edges_df columns: source_id, target_id
    """
    import geff

    result = geff.read(str(sample_geff_path), backend="networkx")
    # Current geff versions return (graph, metadata); older ones returned
    # just the graph. Handle both so this doesn't break on the next release.
    graph: nx.DiGraph = result[0] if isinstance(result, tuple) else result # type: ignore

    nodes = [
        {"node_id": n, **{k: graph.nodes[n][k] for k in ("t", "z", "y", "x")}}
        for n in graph.nodes
    ]
    edges = [{"source_id": u, "target_id": v} for u, v in graph.edges]

    return pd.DataFrame(nodes), pd.DataFrame(edges)


if __name__ == "__main__":
    # Quick sanity check once data/raw has the sample pair downloaded.
    vol = load_volume("data/raw/train/44b6_0113de3b.zarr")
    print("Volume shape:", vol.shape, "dtype:", vol.dtype)

    nodes_df, edges_df = load_tracks("data/raw/train/44b6_0113de3b.geff")
    print(f"Nodes: {len(nodes_df)}, Edges: {len(edges_df)}")
    print(nodes_df.head())
