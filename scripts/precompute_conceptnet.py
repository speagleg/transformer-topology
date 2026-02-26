#!/usr/bin/env python3
"""Download and preprocess ConceptNet for Phase 7 KG tasks.

Usage:
    python scripts/precompute_conceptnet.py [--csv-path PATH] [--output-dir DIR]

Downloads ConceptNet 5.7 assertions if not present, filters to English,
builds the graph, and saves as pickle for fast loading during training.
"""
import argparse
import urllib.request
from collections import Counter
from pathlib import Path

from src.data.conceptnet import load_conceptnet_graph, save_conceptnet_graph


CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"


def download_conceptnet(output_path: Path):
    """Download ConceptNet assertions CSV if not present."""
    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ConceptNet to {output_path}...")
    print(f"  URL: {CONCEPTNET_URL}")
    print(f"  This is ~300MB, may take a few minutes...")
    urllib.request.urlretrieve(CONCEPTNET_URL, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"Downloaded: {output_path} ({size_mb:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ConceptNet for Phase 7")
    parser.add_argument("--csv-path",
                        default="data/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
    parser.add_argument("--output-dir", default="data/conceptnet")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    download_conceptnet(csv_path)

    # Load and filter
    print("\nLoading and filtering ConceptNet (English, weight >= 1.0)...")
    G = load_conceptnet_graph(csv_path)

    # Save processed graph
    pkl_path = output_dir / "conceptnet_en.pkl"
    save_conceptnet_graph(G, pkl_path)
    print(f"\nSaved processed graph to {pkl_path}")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # Print relation distribution
    rels = Counter(d.get("relation", "Other") for _, _, d in G.edges(data=True))
    print(f"\nRelation distribution:")
    for rel, count in rels.most_common():
        print(f"  {rel:20s} {count:>8,d} ({100 * count / G.number_of_edges():.1f}%)")

    # Print some stats
    degrees = [G.degree(n) for n in G.nodes()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    print(f"\nAvg degree: {avg_degree:.1f}")
    print(f"Max degree: {max(degrees) if degrees else 0}")
    high_degree = sum(1 for d in degrees if d >= 5)
    print(f"Nodes with degree >= 5: {high_degree:,} ({100 * high_degree / len(degrees):.1f}%)")


if __name__ == "__main__":
    main()
