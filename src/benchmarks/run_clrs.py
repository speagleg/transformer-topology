"""CLRS-comparable benchmark runner.

Trains on small ER graphs (n=16), evaluates on n=16/32/64 to measure
size generalization. Reports CLRS-comparable metrics for BFS and Dijkstra.
"""

import json
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.clrs_tasks import CLRSDataset
from src.benchmarks.run_comparison import (
    SymmetricMultiHopModel,
    HierarchicalMultiHopModel,
    train_epoch,
    evaluate,
    _resolve_device,
)


def _build_model(arch, mc, max_classes, device):
    """Build a model of the given architecture type."""
    use_topo_pe = mc.get("use_topological_pe", False)
    use_struct = mc.get("use_structural_features", False)

    if arch == "symmetric":
        model = SymmetricMultiHopModel(
            embedding_dim=mc["embedding_dim"], gnn_hidden=mc["gnn_hidden"],
            gnn_spatial_layers=mc["gnn_spatial_layers"],
            gnn_spectral_layers=mc["gnn_spectral_layers"],
            max_freqs=mc["max_freqs"], tat_layers=mc["tat_layers"],
            tat_spatial_heads=mc["tat_spatial_heads"],
            tat_spectral_heads=mc["tat_spectral_heads"],
            tat_ff_dim=mc["tat_ff_dim"], max_classes=max_classes,
            max_iterations=mc["max_iterations"],
            convergence_threshold=mc["convergence_threshold"],
            use_topological_pe=use_topo_pe,
            use_structural_features=use_struct,
        )
    elif arch == "hierarchical":
        model = HierarchicalMultiHopModel(
            embedding_dim=mc["embedding_dim"], gnn_hidden=mc["gnn_hidden"],
            gnn_spatial_layers=mc["gnn_spatial_layers"],
            gnn_spectral_layers=mc["gnn_spectral_layers"],
            max_freqs=mc["max_freqs"], tat_layers=mc["tat_layers"],
            tat_spatial_heads=mc["tat_spatial_heads"],
            tat_spectral_heads=mc["tat_spectral_heads"],
            tat_ff_dim=mc["tat_ff_dim"], max_classes=max_classes,
            max_iterations=mc["max_iterations"],
            convergence_threshold=mc["convergence_threshold"],
            use_wave_dynamics=mc.get("use_wave_dynamics", True),
            use_topological_pe=use_topo_pe,
            use_structural_features=use_struct,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model.to(device)


def _train_model(model, train_ds, val_ds, tc, device):
    """Train a model with early stopping on validation loss."""
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
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(tc["epochs"]):
        start = time.time()
        train_loss, _ = train_epoch(
            model, train_ds, optimizer,
            max_norm=tc.get("max_norm", 5.0),
            accumulation_steps=tc.get("accumulation_steps", 4),
            label_smoothing=label_smoothing,
            device=device,
        )
        val_acc, val_loss = evaluate(
            model, val_ds, label_smoothing=label_smoothing, device=device,
        )
        elapsed = time.time() - start
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_accuracy": val_acc, "time": elapsed,
        })
        print(f"    Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_acc


@torch.no_grad()
def _evaluate_per_distance(model, dataset, device):
    """Evaluate accuracy broken down by ground-truth distance bucket."""
    model.eval()
    buckets: dict[int, list[bool]] = {}
    total_correct = 0

    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        cc = cc.clone().to(device)
        logits = model(cc, query, target)
        pred = logits.argmax().item()
        correct = pred == answer
        total_correct += int(correct)
        buckets.setdefault(answer, []).append(correct)

    overall_acc = total_correct / len(dataset) if len(dataset) > 0 else 0.0
    per_distance = {
        d: sum(hits) / len(hits)
        for d, hits in sorted(buckets.items())
    }
    return overall_acc, per_distance


def run_clrs_benchmark(config_path: str = "config/clrs.yaml"):
    """Run the CLRS-comparable benchmark."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    cc = config["clrs"]

    device = _resolve_device(tc)
    max_distance = cc["max_distance"]
    max_classes = max_distance

    print("=" * 70)
    print("CLRS-Comparable Benchmark: BFS & Dijkstra")
    print(f"Device: {device}")
    print(f"Train size: n={cc['train_size']}, Test sizes: {cc['test_sizes']}")
    print("=" * 70)

    all_results = {}

    for task_type in cc["task_types"]:
        print(f"\n{'=' * 70}")
        print(f"Task: {task_type.upper()}")
        print("=" * 70)

        task_kwargs = {
            "embedding_dim": mc["embedding_dim"],
            "max_distance": max_distance,
        }
        if task_type == "dijkstra":
            task_kwargs["max_edge_weight"] = cc["max_edge_weight"]

        # Generate datasets
        print(f"\nGenerating datasets...")
        train_n = cc["train_size"]
        train_ds = CLRSDataset(
            num_samples=cc["num_train"], task_type=task_type,
            n_nodes=train_n, **task_kwargs,
        )
        val_ds = CLRSDataset(
            num_samples=cc["num_val"], task_type=task_type,
            n_nodes=train_n, **task_kwargs,
        )

        test_datasets = {}
        for test_n in cc["test_sizes"]:
            test_datasets[test_n] = CLRSDataset(
                num_samples=cc["num_test"], task_type=task_type,
                n_nodes=test_n, **task_kwargs,
            )
        print(f"  Train: {len(train_ds)} (n={train_n}), Val: {len(val_ds)} (n={train_n})")
        for n, ds in test_datasets.items():
            print(f"  Test n={n}: {len(ds)}")

        task_results = {}

        for arch in ["symmetric", "hierarchical"]:
            print(f"\n--- {arch.title()} ---")
            model = _build_model(arch, mc, max_classes, device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_params:,}")

            # Train
            history, best_val_acc = _train_model(model, train_ds, val_ds, tc, device)

            # Evaluate on all test sizes
            size_results = {}
            for test_n, test_ds in test_datasets.items():
                acc, per_dist = _evaluate_per_distance(model, test_ds, device)
                label = "ID" if test_n == train_n else f"{test_n // train_n}x OOD"
                print(f"  n={test_n} ({label}): {acc:.3f}")
                size_results[test_n] = {
                    "accuracy": acc,
                    "per_distance": {str(k): v for k, v in per_dist.items()},
                }

            # Generalization gap
            id_acc = size_results[train_n]["accuracy"]
            ood_accs = {n: r["accuracy"] for n, r in size_results.items() if n != train_n}
            max_ood_n = max(cc["test_sizes"])
            gen_gap = id_acc - size_results.get(max_ood_n, size_results[train_n])["accuracy"]

            task_results[arch] = {
                "params": total_params,
                "history": history,
                "best_val_accuracy": best_val_acc,
                "test_results": size_results,
                "generalization_gap": gen_gap,
            }

        all_results[task_type] = task_results

    # Print summary table
    print(f"\n{'=' * 70}")
    print("CLRS-Comparable Results")
    print("=" * 70)

    train_n = cc["train_size"]
    for task_type in cc["task_types"]:
        print(f"\n{task_type.upper()}:")
        header = f"{'':15s}"
        for n in cc["test_sizes"]:
            if n == train_n:
                label = f"n={n} (ID)"
            else:
                label = f"n={n} ({n // train_n}x)"
            header += f"  {label:>12s}"
        print(header)

        for arch in ["symmetric", "hierarchical"]:
            row = f"{arch:15s}"
            for n in cc["test_sizes"]:
                acc = all_results[task_type][arch]["test_results"][n]["accuracy"]
                row += f"  {acc:>11.1%}"
            print(row)

    print(f"\nCLRS Reference (Triplet-GMPNN, BFS):")
    print(f"  n=16: 96.1%   n=64: ~45%")

    # Save results
    Path("data").mkdir(exist_ok=True)
    results_path = "data/clrs_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/clrs.yaml"
    run_clrs_benchmark(config)
