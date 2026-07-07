"""Download a sample's .geff and report its longest labeled track chain,
so we know which frame range is worth spending Cellpose compute on.

Run: python src/io/inspect_sample.py 44b6_0b24845f
"""
import networkx as nx
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io.download import download_geff
from src.io.loaders import load_tracks


def longest_chain_ranges(sample: str, top_n: int = 3):
    geff_relpath = f"train/{sample}.geff"
    geff_local = f"data/raw/train/{sample}.geff"

    download_geff(geff_relpath, output_dir="data/raw")

    nodes_df, edges_df = load_tracks(geff_local)
    print(f"{sample}: {len(nodes_df)} nodes, {len(edges_df)} edges")

    g = nx.DiGraph()
    g.add_edges_from(edges_df[["source_id", "target_id"]].itertuples(index=False))
    chains = sorted(nx.weakly_connected_components(g), key=len, reverse=True)

    for i, chain in enumerate(chains[:top_n]):
        chain_nodes = nodes_df[nodes_df["node_id"].isin(chain)]
        t_min, t_max = int(chain_nodes["t"].min()), int(chain_nodes["t"].max())
        print(f"  chain {i}: {len(chain)} nodes, t={t_min}..{t_max} "
              f"(span {t_max - t_min + 1} frames)")


if __name__ == "__main__":
    longest_chain_ranges(sys.argv[1])