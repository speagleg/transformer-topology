#!/usr/bin/env python3
"""Pre-generate KG task datasets from ConceptNet for Phase D training.

Usage:
    python scripts/pregenerate_kg_datasets.py
    python scripts/pregenerate_kg_datasets.py --output-dir data/kg_datasets --train-samples 5000

Generates train + val splits for all 5 KG tasks and saves as .pt files.
These can then be rsynced to the training instance and loaded via --pregenerated-dir.
"""

import argparse
import time
from pathlib import Path

from src.data.conceptnet import load_cached_graph, load_conceptnet_graph, save_conceptnet_graph
from src.benchmarks.benchmark_dataset import BenchmarkDataset


KG_TASKS = ["kg_relation", "kg_concept", "kg_pathvalid", "kg_analogy", "kg_cluster"]


def main():
    parser = argparse.ArgumentParser(description="Pre-generate KG datasets")
    parser.add_argument("--conceptnet-path", default="data/conceptnet/conceptnet_en.pkl")
    parser.add_argument("--conceptnet-csv", default="data/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
    parser.add_argument("--output-dir", default="data/dsm_datasets")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--n-nodes-min", type=int, default=16)
    parser.add_argument("--n-nodes-max", type=int, default=32)
    parser.add_argument("--tasks", nargs="+", default=KG_TASKS,
                        help="Which KG tasks to generate (default: all 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_range = (args.n_nodes_min, args.n_nodes_max)
    range_str = f"n{n_range[0]}-{n_range[1]}"

    # Load ConceptNet
    cn_path = Path(args.conceptnet_path)
    if cn_path.exists():
        print(f"Loading cached ConceptNet from {cn_path}...")
        G = load_cached_graph(cn_path)
    elif Path(args.conceptnet_csv).exists():
        print(f"Parsing ConceptNet from {args.conceptnet_csv}...")
        G = load_conceptnet_graph(args.conceptnet_csv)
        cn_path.parent.mkdir(parents=True, exist_ok=True)
        save_conceptnet_graph(G, cn_path)
        print(f"Saved to {cn_path}")
    else:
        print(f"ERROR: Neither {cn_path} nor {args.conceptnet_csv} found.")
        print(f"Run: python scripts/precompute_conceptnet.py")
        return

    print(f"ConceptNet: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    for task in args.tasks:
        if task not in KG_TASKS:
            print(f"Skipping unknown task: {task}")
            continue

        for split, num_samples in [("train", args.train_samples), ("val", args.val_samples)]:
            save_path = output_dir / f"{task}_{split}_{range_str}.pt"
            if save_path.exists():
                print(f"\n  {save_path.name} already exists, skipping")
                continue

            print(f"\n  Generating {task} {split} ({num_samples} samples, {range_str})...")
            t0 = time.time()
            ds = BenchmarkDataset(
                num_samples, task, args.n_nodes_min, args.embedding_dim,
                n_nodes_range=n_range,
                conceptnet_graph=G,
            )
            ds.save(str(save_path))
            elapsed = time.time() - t0
            print(f"  Saved {save_path.name} ({len(ds)} samples, {elapsed:.1f}s)")

    print("\nDone! Rsync to instance with:")
    print(f"  rsync -avz -e 'ssh -i ~/.ssh/vastai -p 34701' {output_dir}/ root@136.59.129.136:~/transformer-topology/{output_dir}/")


if __name__ == "__main__":
    main()
