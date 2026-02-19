#!/usr/bin/env python3
"""Pre-generate all datasets for Llama 3.2 curriculum training.

Generates train/val/test splits for every task in the curriculum,
with mixed-size training, topology control, and seed reproducibility.
Saves to disk so training loads instantly without regeneration.

Usage:
    python scripts/generate_llama_datasets.py [config_path]
    python scripts/generate_llama_datasets.py config/llama_training.yaml --output-dir data/llama_datasets
    python scripts/generate_llama_datasets.py --tasks diverse bfs hodge_class
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, TASK_REGISTRY, get_max_classes


ALL_TOPOLOGIES = ['ba', 'ws', 'sbm', 'grid', 'tree', 'ladder', 'caveman', 'er']
DEFAULT_TRAIN_TOPOS = ['ba', 'ws', 'sbm', 'er']
DEFAULT_TEST_TOPOS = ['grid', 'tree', 'ladder', 'caveman']


def compute_class_distribution(dataset: BenchmarkDataset) -> dict[int, int]:
    """Count class occurrences in a dataset."""
    counts: Counter = Counter()
    for sample in dataset.samples:
        counts[sample[3]] += 1
    return dict(sorted(counts.items()))


def generate_task_datasets(
    task_type: str,
    output_dir: Path,
    num_train: int,
    num_val: int,
    num_test: int,
    embedding_dim: int,
    train_n_nodes: int,
    train_n_nodes_range: tuple[int, int] | None,
    test_n_nodes_list: list[int],
    train_topologies: list[str],
    test_topologies: list[str],
    seed: int,
) -> dict:
    """Generate and save all splits for a single task type."""
    all_topos = train_topologies + test_topologies
    max_classes = get_max_classes(task_type)
    stats: dict = {}

    range_str = (f"n{train_n_nodes_range[0]}-{train_n_nodes_range[1]}"
                 if train_n_nodes_range else f"n{train_n_nodes}")

    print(f"\n{'=' * 60}")
    print(f"Task: {task_type} ({max_classes} classes)")
    print(f"{'=' * 60}")

    # --- Train split (mixed-size, all topologies) ---
    random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    train_ds = BenchmarkDataset(
        num_train, task_type, train_n_nodes, embedding_dim,
        topologies=all_topos, n_nodes_range=train_n_nodes_range,
    )
    train_path = output_dir / f"{task_type}_train_{range_str}.pt"
    train_ds.save(str(train_path))
    dist = compute_class_distribution(train_ds)
    elapsed = time.time() - t0
    print(f"  Train: {num_train} samples, {elapsed:.1f}s")
    print(f"    Class dist: {dist}")
    stats["train"] = {"count": num_train, "class_dist": dist, "time": round(elapsed, 1)}

    # --- Val split (mixed-size, all topologies) ---
    random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    t0 = time.time()
    val_ds = BenchmarkDataset(
        num_val, task_type, train_n_nodes, embedding_dim,
        topologies=all_topos, n_nodes_range=train_n_nodes_range,
    )
    val_path = output_dir / f"{task_type}_val_{range_str}.pt"
    val_ds.save(str(val_path))
    dist = compute_class_distribution(val_ds)
    elapsed = time.time() - t0
    print(f"  Val:   {num_val} samples, {elapsed:.1f}s")
    print(f"    Class dist: {dist}")
    stats["val"] = {"count": num_val, "class_dist": dist, "time": round(elapsed, 1)}

    # --- Test: ID (fixed-size, all topologies) per test size ---
    for i, test_n in enumerate(test_n_nodes_list):
        random.seed(seed + 100 + i)
        torch.manual_seed(seed + 100 + i)
        t0 = time.time()
        test_ds = BenchmarkDataset(
            num_test, task_type, test_n, embedding_dim,
            topologies=all_topos,
        )
        test_path = output_dir / f"{task_type}_test_n{test_n}.pt"
        test_ds.save(str(test_path))
        dist = compute_class_distribution(test_ds)
        elapsed = time.time() - t0
        print(f"  Test n={test_n}: {num_test} samples, {elapsed:.1f}s | dist: {dist}")
        stats[f"test_n{test_n}"] = {"count": num_test, "class_dist": dist}

    # --- Test: Topology transfer (fixed-size, held-out topologies only) ---
    random.seed(seed + 200)
    torch.manual_seed(seed + 200)
    t0 = time.time()
    topo_test_ds = BenchmarkDataset(
        num_test, task_type, train_n_nodes, embedding_dim,
        topologies=test_topologies,
    )
    topo_path = output_dir / f"{task_type}_test_topo_n{train_n_nodes}.pt"
    topo_test_ds.save(str(topo_path))
    dist = compute_class_distribution(topo_test_ds)
    elapsed = time.time() - t0
    print(f"  Topo transfer: {num_test} samples, {elapsed:.1f}s | dist: {dist}")
    stats["topo_transfer"] = {"count": num_test, "class_dist": dist}

    return stats


def generate_all_datasets(config_path: str, output_dir: str | None = None,
                          seed: int | None = None, task_filter: list[str] | None = None):
    """Main entry point: generate datasets for all curriculum tasks."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    bc = config["benchmark"]
    cc = config.get("curriculum", {})

    seed = seed or bc.get("seed", 42)
    output_dir_path = Path(output_dir or bc.get("datasets_dir", "data/llama_datasets"))
    output_dir_path.mkdir(parents=True, exist_ok=True)

    emb_dim = mc["embedding_dim"]
    train_n = bc["train_n_nodes"]
    num_train = bc.get("num_train", 5000)
    num_val = bc.get("num_val", 500)
    num_test = bc.get("num_test", 500)

    # Collect all unique tasks from curriculum phases
    all_tasks: set[str] = set()
    for phase_key in ["phase_a", "phase_b", "phase_c"]:
        phase = cc.get(phase_key, {})
        if phase.get("enabled", True):
            all_tasks.update(phase.get("tasks", []))

    if task_filter:
        all_tasks = all_tasks & set(task_filter)

    # Mixed-size range
    n_min = bc.get("train_n_nodes_min")
    n_max = bc.get("train_n_nodes_max")
    train_range = (n_min, n_max) if n_min is not None and n_max is not None else None

    train_topos = bc.get("train_topologies", DEFAULT_TRAIN_TOPOS)
    test_topos = bc.get("test_topologies", DEFAULT_TEST_TOPOS)
    test_sizes = bc.get("test_n_nodes", [train_n])
    if isinstance(test_sizes, int):
        test_sizes = [test_sizes]

    print("=" * 72)
    print("Llama Dataset Generation")
    print("=" * 72)
    print(f"  Config: {config_path}")
    print(f"  Tasks: {sorted(all_tasks)}")
    print(f"  Train: {num_train} samples, size range: {train_range or train_n}")
    print(f"  Val: {num_val} samples")
    print(f"  Test: {num_test} samples per split")
    print(f"  Test sizes: {test_sizes}")
    print(f"  Topologies: train={train_topos}, test={test_topos}")
    print(f"  Output: {output_dir_path}")
    print(f"  Seed: {seed}")

    task_list = sorted(TASK_REGISTRY.keys())
    all_stats: dict = {}
    total_t0 = time.time()

    for task_type in sorted(all_tasks):
        task_seed = seed + task_list.index(task_type) * 1000
        stats = generate_task_datasets(
            task_type, output_dir_path, num_train, num_val, num_test,
            emb_dim, train_n, train_range, test_sizes,
            train_topos, test_topos, task_seed,
        )
        all_stats[task_type] = stats

    total_elapsed = time.time() - total_t0

    # Save manifest
    manifest = {
        "seed": seed,
        "config": config_path,
        "tasks": all_stats,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "train_n_nodes_range": list(train_range) if train_range else None,
        "test_n_nodes": test_sizes,
        "train_topologies": train_topos,
        "test_topologies": test_topos,
        "total_time_seconds": round(total_elapsed, 1),
    }
    manifest_path = output_dir_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'=' * 72}")
    print(f"Done! Generated {len(all_stats)} task datasets in {total_elapsed:.0f}s")
    print(f"Manifest: {manifest_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate Llama training datasets")
    parser.add_argument("config", nargs="?", default="config/llama_training.yaml",
                        help="Path to YAML config")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Generate only these task types")
    args = parser.parse_args()
    generate_all_datasets(args.config, args.output_dir, args.seed, args.tasks)
