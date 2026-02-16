#!/usr/bin/env python3
"""Hodge class scaling experiment — single job worker.

Trains ONE filter at ONE training size, evaluates on all test sizes.
Designed to be launched in parallel via gpu_launcher.

Usage:
    python3 scripts/run_hodge_scaling.py --filter schrodinger --train-size 40 \
        --datasets-dir DIR --results-file result.json
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
from src.benchmarks.run_comparison import (
    HierarchicalMultiHopModel,
    train_epoch,
    evaluate,
    evaluate_per_class,
)


TEST_SIZES = [20, 40, 80, 160, 320]
TASK = "hodge_class"
CONFIG_PATH = "config/benchmark_4b_schrodinger.yaml"


def resolve_device(tc):
    dev = tc.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def build_model(mc, max_classes, device, wc):
    model = HierarchicalMultiHopModel(
        embedding_dim=mc["embedding_dim"],
        gnn_hidden=mc["gnn_hidden"],
        gnn_spatial_layers=mc["gnn_spatial_layers"],
        gnn_spectral_layers=mc["gnn_spectral_layers"],
        max_freqs=mc["max_freqs"],
        tat_layers=mc["tat_layers"],
        tat_spatial_heads=mc["tat_spatial_heads"],
        tat_spectral_heads=mc["tat_spectral_heads"],
        tat_ff_dim=mc["tat_ff_dim"],
        max_classes=max_classes,
        max_iterations=mc["max_iterations"],
        convergence_threshold=mc["convergence_threshold"],
        use_topological_pe=mc.get("use_topological_pe", False),
        use_structural_features=mc.get("use_structural_features", False),
        use_wave_dynamics=mc.get("use_wave_dynamics", True),
        use_higher_order=mc.get("use_higher_order", True),
        wave_config=wc,
    )
    return model.to(device)


def load_dataset(ds_dir, name, n_nodes, emb_dim, num_samples, topologies):
    """Load dataset from cache, or generate and save."""
    if ds_dir is not None:
        path = ds_dir / f"hodge_scaling_{name}_n{n_nodes}.pt"
        if path.exists():
            print(f"    Loading {path.name}...")
            return BenchmarkDataset.load(str(path))

    ds = BenchmarkDataset(num_samples, TASK, n_nodes, emb_dim, topologies=topologies)

    if ds_dir is not None:
        ds_dir.mkdir(parents=True, exist_ok=True)
        path = ds_dir / f"hodge_scaling_{name}_n{n_nodes}.pt"
        ds.save(str(path))
        print(f"    Saved {path.name}")

    return ds


def train_model(model, train_ds, val_ds, tc, device):
    """Train with early stopping, return best model + info."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5,
    )

    label_smoothing = tc.get("label_smoothing", 0.1)
    best_val_acc = 0.0
    best_train_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    curve = []

    for epoch in range(tc["epochs"]):
        t0 = time.time()
        train_loss, grad_norm = train_epoch(
            model, train_ds, optimizer,
            max_norm=tc.get("max_norm", 5.0),
            accumulation_steps=tc.get("accumulation_steps", 4),
            label_smoothing=label_smoothing,
            device=device,
        )
        val_acc, val_loss = evaluate(model, val_ds, label_smoothing=label_smoothing,
                                     device=device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        curve.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'val_loss': round(val_loss, 5),
            'val_acc': round(val_acc, 4),
            'grad_norm': round(grad_norm, 4),
            'lr': lr,
            'time': round(elapsed, 1),
        })

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        print(f"      Ep {epoch:3d} | loss {train_loss:.4f} | "
              f"val {val_acc:.3f} ({val_loss:.4f}) | "
              f"gn {grad_norm:.2f} | lr {lr:.1e} | {elapsed:.0f}s{marker}")

        if patience_counter >= tc["patience"]:
            print(f"      Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'best_val_acc': best_val_acc,
        'best_train_loss': best_train_loss,
        'best_epoch': best_epoch,
        'total_epochs': len(curve),
        'training_curve': curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Hodge scaling — single job worker")
    parser.add_argument("--filter", required=True, help="Filter type (schrodinger, wave_cosine)")
    parser.add_argument("--train-size", type=int, required=True, help="Training graph size")
    parser.add_argument("--datasets-dir", required=True, help="Directory with cached datasets")
    parser.add_argument("--results-file", required=True, help="Output JSON file")
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-test", type=int, default=500)
    parser.add_argument("--test-sizes", default=None,
                        help="Comma-separated test sizes (default: 20,40,80,160,320)")
    parser.add_argument("--config", default=None,
                        help="Override config file (default: benchmark_4b_schrodinger.yaml)")
    args = parser.parse_args()

    test_sizes = [int(x) for x in args.test_sizes.split(",")] if args.test_sizes else TEST_SIZES
    filter_type = args.filter
    train_n = args.train_size

    config_path = args.config or CONFIG_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    base_wc = config.get("wave", {})
    wc = dict(base_wc)
    # Only override filter_type for spectral mode (not sheaf)
    if wc.get("wave_mode") != "sheaf":
        wc["filter_type"] = filter_type

    emb_dim = mc["embedding_dim"]
    topologies = ["ba", "ws", "sbm", "er"]
    max_classes = get_max_classes(TASK)
    device = resolve_device(tc)
    seed = 42
    ds_dir = Path(args.datasets_dir)

    print("=" * 72)
    print(f"Hodge Scaling: filter={filter_type}, train_n={train_n}")
    print(f"  Test sizes: {test_sizes}")
    print(f"  Device: {device}")
    print("=" * 72)

    # Load datasets
    print("\n--- Loading datasets ---")
    train_ds = load_dataset(ds_dir, f"train_{train_n}", train_n, emb_dim, args.num_train, topologies)
    val_ds = load_dataset(ds_dir, f"val_{train_n}", train_n, emb_dim, args.num_test, topologies)

    test_datasets = {}
    for tn in test_sizes:
        test_datasets[tn] = load_dataset(ds_dir, f"test_{tn}", tn, emb_dim, args.num_test, topologies)

    # Train
    print(f"\n--- Training (filter={filter_type}, n={train_n}) ---")
    random.seed(seed)
    torch.manual_seed(seed)

    model = build_model(mc, max_classes, device, wc)
    params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {params:,}")

    t0 = time.time()
    train_info = train_model(model, train_ds, val_ds, tc, device)
    train_time = time.time() - t0

    # Evaluate on all test sizes
    print("\n--- Evaluating ---")
    size_results = {}
    for tn in test_sizes:
        acc, _ = evaluate(model, test_datasets[tn], device=device)
        size_results[str(tn)] = round(acc, 4)
        print(f"    Test n={tn}: {acc:.1%}")

    # Per-class at training size
    pc_info = evaluate_per_class(model, test_datasets[train_n], max_classes, device)

    # Save
    output = {
        "experiment": "hodge_class_scaling",
        "filter": filter_type,
        "train_size": train_n,
        "test_sizes": test_sizes,
        "params": params,
        "train_time": round(train_time, 1),
        "best_val_acc": train_info["best_val_acc"],
        "best_epoch": train_info["best_epoch"],
        "total_epochs": train_info["total_epochs"],
        "training_curve": train_info["training_curve"],
        "test_accuracies": size_results,
        "per_class": pc_info["per_class"],
        "confusion": pc_info["confusion"],
    }

    rf = Path(args.results_file)
    rf.parent.mkdir(parents=True, exist_ok=True)
    with open(rf, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {rf}")


if __name__ == "__main__":
    main()
