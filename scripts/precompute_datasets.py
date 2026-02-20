#!/usr/bin/env python3
"""Precompute all datasets for DSM curriculum training.

Runs on CPU. Generates and serializes BenchmarkDataset files for each
task at multiple sizes and topology splits.

Usage:
    python scripts/precompute_datasets.py config/dsm_training.yaml \
        --output-dir data/dsm_datasets
    python scripts/precompute_datasets.py config/dsm_training.yaml \
        --output-dir data/dsm_datasets --tasks diverse bfs
"""

import argparse
import time
from pathlib import Path

import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset

ALL_TASKS = [
    # Phase A — structural
    "diverse",
    "bfs",
    "hodge_class",
    "spectral_gap",
    "path_counting",
    # Phase B — semantic
    "graph_completion",
    "labeled_reasoning",
    # Phase C — transfer
    "analogical_transfer",
]


def _generate_and_save(
    output_dir: Path,
    task: str,
    split: str,
    num_samples: int,
    emb_dim: int,
    n_nodes: int,
    topologies: list[str],
    n_nodes_range: tuple[int, int] | None = None,
) -> None:
    if n_nodes_range:
        suffix = f"n{n_nodes_range[0]}-{n_nodes_range[1]}"
    else:
        suffix = f"n{n_nodes}"
    filename = f"{task}_{split}_{suffix}.pt"
    path = output_dir / filename

    if path.exists():
        print(f"  SKIP {filename} (already exists)")
        return

    t0 = time.time()
    print(f"  Generating {filename} ({num_samples} samples)...", end="", flush=True)
    ds = BenchmarkDataset(
        num_samples,
        task,
        n_nodes,
        emb_dim,
        topologies=topologies,
        n_nodes_range=n_nodes_range,
    )
    ds.save(str(path))
    elapsed = time.time() - t0
    print(f" done ({elapsed:.1f}s)")


def precompute(
    config_path: str,
    output_dir: str,
    tasks: list[str] | None = None,
) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    bc = config["benchmark"]
    emb_dim = config["model"]["embedding_dim"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tasks = tasks or ALL_TASKS
    train_samples = bc.get("train_samples", 5000)
    val_samples = bc.get("val_samples", 500)
    test_samples = bc.get("test_samples", 500)
    n_min = bc.get("train_n_nodes_min", 16)
    n_max = bc.get("train_n_nodes_max", 32)
    train_topos = bc.get(
        "train_topologies", ["ba", "ws", "grid", "tree", "ladder", "sbm"]
    )
    test_topos = bc.get("test_topologies", ["er", "caveman"])
    all_topos = train_topos + test_topos

    total_t0 = time.time()
    print(f"Config: {config_path}")
    print(f"Output: {output}")
    print(f"Tasks:  {tasks}")
    print(f"Train:  {train_samples} samples, n={n_min}-{n_max}, topos={all_topos}")
    print(f"Val:    {val_samples} samples, n={n_min}-{n_max}")
    print(f"Test:   {test_samples} samples, n=20/40/80 + topo transfer")

    for task in tasks:
        print(f"\n{'─' * 60}")
        print(f"Task: {task}")
        print(f"{'─' * 60}")

        # Training set (mixed size, all topologies)
        _generate_and_save(
            output,
            task,
            "train",
            train_samples,
            emb_dim,
            n_nodes=20,
            topologies=all_topos,
            n_nodes_range=(n_min, n_max),
        )

        # Validation set (mixed size, all topologies)
        _generate_and_save(
            output,
            task,
            "val",
            val_samples,
            emb_dim,
            n_nodes=20,
            topologies=all_topos,
            n_nodes_range=(n_min, n_max),
        )

        # Test sets — in-distribution
        _generate_and_save(
            output,
            task,
            "test",
            test_samples,
            emb_dim,
            n_nodes=20,
            topologies=all_topos,
        )

        # Test sets — size OOD
        _generate_and_save(
            output,
            task,
            "test",
            test_samples,
            emb_dim,
            n_nodes=40,
            topologies=all_topos,
        )
        _generate_and_save(
            output,
            task,
            "test",
            test_samples,
            emb_dim,
            n_nodes=80,
            topologies=all_topos,
        )

        # Test set — topology transfer (held-out topologies only)
        _generate_and_save(
            output,
            task,
            "test_topo",
            test_samples,
            emb_dim,
            n_nodes=20,
            topologies=test_topos,
        )

    total_elapsed = time.time() - total_t0
    print(f"\n{'═' * 60}")
    print(f"All done in {total_elapsed:.1f}s")
    print(f"Datasets saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute DSM training datasets")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument(
        "--output-dir",
        default="data/dsm_datasets",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific tasks to generate (default: all)",
    )
    args = parser.parse_args()
    precompute(args.config, args.output_dir, args.tasks)
