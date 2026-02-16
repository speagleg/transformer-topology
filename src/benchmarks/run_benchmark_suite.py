"""Comprehensive benchmark suite for meta-cognition architecture evaluation.

Evaluation axes:
1. In-Distribution (ID): train and test on all topologies, n=train_n_nodes
2. Topology Transfer (OOD-Topo): train on some topologies, test on held-out
3. Size Generalization (OOD-Size): train on n=20, test on n=20/40/80

Tier 1 (core): diverse, propagation_delay, spectral_gap, hodge_class, bfs
Tier 2 (extended): blocking, interference, cycle_detection, betti_number, path_counting, dijkstra

Model variants: hierarchical, symmetric, hierarchical_nowave
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, TASK_REGISTRY, get_max_classes
from src.benchmarks.diagnostics import DiagnosticCollector
from src.benchmarks.run_comparison import (
    SymmetricMultiHopModel,
    HierarchicalMultiHopModel,
    train_epoch,
    evaluate,
    evaluate_per_class,
)

TIER_1_TASKS = ["diverse", "propagation_delay", "spectral_gap", "hodge_class", "bfs"]
TIER_2_TASKS = ["blocking", "interference", "cycle_detection", "betti_number",
                "path_counting", "dijkstra"]
TIER_3_TASKS = ["graph_completion", "labeled_reasoning", "analogical_transfer"]


def _resolve_device(tc: dict) -> torch.device:
    dev = tc.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _build_model(variant: str, mc: dict, max_classes: int, device: torch.device,
                  wave_config: dict | None = None, llm_config: dict | None = None):
    """Build a model for the given variant."""
    common = dict(
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
    )

    if variant == "symmetric":
        model = SymmetricMultiHopModel(**common)
    elif variant == "hierarchical":
        model = HierarchicalMultiHopModel(
            **common,
            use_wave_dynamics=mc.get("use_wave_dynamics", True),
            use_higher_order=mc.get("use_higher_order", True),
            wave_config=wave_config,
        )
    elif variant == "hierarchical_nowave":
        model = HierarchicalMultiHopModel(
            **common,
            use_wave_dynamics=False,
            use_higher_order=mc.get("use_higher_order", True),
        )
    elif variant == "hierarchical_llm":
        model = HierarchicalMultiHopModel(
            **common,
            use_wave_dynamics=mc.get("use_wave_dynamics", True),
            use_higher_order=mc.get("use_higher_order", True),
            wave_config=wave_config,
            use_llm=True,
            llm_config=llm_config,
        )
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    return model.to(device)


def _train_and_evaluate(model, train_ds, val_ds, tc, device, checkpoint_path=None):
    """Train model with early stopping on val loss.

    Args:
        checkpoint_path: If set, save best model checkpoint to this path
            after each improvement. Allows recovery from SIGTERM/crashes.

    Returns:
        dict with best_val_acc, best_train_loss, best_epoch, training_curve.
    """
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
    training_curve = []

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

        training_curve.append({
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
            # Checkpoint on improvement so SIGTERM doesn't lose progress
            if checkpoint_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state,
                    'val_acc': best_val_acc,
                    'training_curve': training_curve,
                }, checkpoint_path)
        else:
            patience_counter += 1

        print(f"      Ep {epoch:3d} | loss {train_loss:.4f} | "
              f"val {val_acc:.3f} ({val_loss:.4f}) | "
              f"gn {grad_norm:.2f} | lr {lr:.1e} | {elapsed:.0f}s{marker}")

        if patience_counter >= tc["patience"]:
            print(f"      Early stop at epoch {epoch}")
            break

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'best_val_acc': best_val_acc,
        'best_train_loss': best_train_loss,
        'best_epoch': best_epoch,
        'total_epochs': len(training_curve),
        'training_curve': training_curve,
    }


def _evaluate_dataset(model, dataset, device):
    """Evaluate model on a dataset, return accuracy."""
    acc, _ = evaluate(model, dataset, device=device)
    return acc


def _evaluate_per_topology(model, task_type, bc, mc, topologies, max_classes, device,
                           datasets_dir=None):
    """Evaluate model separately on each test topology.

    Generates (or loads) a small per-topology test set and evaluates.
    Returns dict {topology_name: accuracy}.
    """
    emb_dim = mc["embedding_dim"]
    train_n = bc["train_n_nodes"]
    # Use a smaller per-topology sample count (1/len split of total)
    per_topo_n = max(50, bc["num_test"] // max(len(topologies), 1))
    ds_dir = Path(datasets_dir) if datasets_dir else None

    breakdown = {}
    for topo in topologies:
        ds = _load_or_generate(
            ds_dir, f"topo_test_{topo}", task_type,
            per_topo_n, train_n, emb_dim,
            topologies=[topo],
        )
        acc = _evaluate_dataset(model, ds, device)
        breakdown[topo] = round(acc, 4)

    return breakdown


def _dataset_path(datasets_dir: Path, task_type: str, split_name: str, n_nodes: int,
                   n_nodes_range: tuple[int, int] | None = None) -> Path:
    """Convention for dataset file paths.

    Mixed-size datasets get a different filename (e.g. ``_n16-32.pt``) so they
    can't accidentally load stale fixed-size data.
    """
    if n_nodes_range is not None:
        return datasets_dir / f"{task_type}_{split_name}_n{n_nodes_range[0]}-{n_nodes_range[1]}.pt"
    return datasets_dir / f"{task_type}_{split_name}_n{n_nodes}.pt"


def _load_or_generate(
    datasets_dir: Path | None,
    split_name: str,
    task_type: str,
    num_samples: int,
    n_nodes: int,
    emb_dim: int,
    topologies: list[str] | None,
    n_nodes_range: tuple[int, int] | None = None,
) -> BenchmarkDataset:
    """Load dataset from disk if available, else generate and optionally save."""
    if datasets_dir is not None:
        path = _dataset_path(datasets_dir, task_type, split_name, n_nodes,
                             n_nodes_range=n_nodes_range)
        if path.exists():
            print(f"    Loading {path.name}...")
            return BenchmarkDataset.load(str(path))

    ds = BenchmarkDataset(
        num_samples, task_type, n_nodes, emb_dim,
        topologies=topologies, n_nodes_range=n_nodes_range,
    )

    if datasets_dir is not None:
        datasets_dir.mkdir(parents=True, exist_ok=True)
        path = _dataset_path(datasets_dir, task_type, split_name, n_nodes,
                             n_nodes_range=n_nodes_range)
        ds.save(str(path))
        print(f"    Saved {path.name}")

    return ds


def _generate_datasets(task_type, bc, mc, datasets_dir=None):
    """Generate train/val/test datasets for all splits.

    If datasets_dir is provided, datasets are saved/loaded from disk as .pt files.
    Mixed-size training is enabled when train_n_nodes_min and train_n_nodes_max are set.
    """
    emb_dim = mc["embedding_dim"]
    train_n = bc["train_n_nodes"]
    all_topos = bc.get("train_topologies", []) + bc.get("test_topologies", [])
    ds_dir = Path(datasets_dir) if datasets_dir else None

    # Mixed-size training range (None if not configured)
    n_min = bc.get("train_n_nodes_min")
    n_max = bc.get("train_n_nodes_max")
    train_range = (n_min, n_max) if n_min is not None and n_max is not None else None

    datasets = {}

    # ID split: train/val use mixed sizes if configured; test is fixed-size
    size_str = f"n={n_min}-{n_max}" if train_range else f"n={train_n}"
    print(f"  Generating ID datasets ({size_str} train, n={train_n} test, all topologies)...")
    for split, count in [("id_train", "num_train"), ("id_val", "num_val")]:
        datasets[split] = _load_or_generate(
            ds_dir, split, task_type, bc[count], train_n, emb_dim,
            topologies=all_topos or None,
            n_nodes_range=train_range,
        )
    # Test datasets are always fixed-size for clean evaluation
    datasets["id_test"] = _load_or_generate(
        ds_dir, "id_test", task_type, bc["num_test"], train_n, emb_dim,
        topologies=all_topos or None,
    )

    # Topology transfer: train on train_topologies, test on test_topologies
    train_topos = bc.get("train_topologies")
    test_topos = bc.get("test_topologies")
    if train_topos and test_topos:
        print(f"  Generating topo-transfer datasets (train={train_topos}, test={test_topos})...")
        for split, count, topos in [
            ("topo_train", "num_train", train_topos),
            ("topo_val", "num_val", train_topos),
        ]:
            datasets[split] = _load_or_generate(
                ds_dir, split, task_type, bc[count], train_n, emb_dim,
                topologies=topos,
                n_nodes_range=train_range,
            )
        # Topo test is fixed-size
        datasets["topo_test"] = _load_or_generate(
            ds_dir, "topo_test", task_type, bc["num_test"], train_n, emb_dim,
            topologies=test_topos,
        )

    # Size generalization: same topologies as ID, different sizes
    test_sizes = bc.get("test_n_nodes", [])
    for size in test_sizes:
        if size != train_n:
            print(f"  Generating size-{size} test dataset...")
            key = f"size_{size}"
            datasets[key] = _load_or_generate(
                ds_dir, key, task_type, bc["num_test"], size, emb_dim,
                topologies=all_topos or None,
            )

    return datasets


def run_benchmark_suite(
    config_path: str = "config/benchmark_suite.yaml",
    task_filter: str | None = None,
    variant_filter: str | None = None,
    datasets_dir: str | None = None,
    generate_only: bool = False,
    results_file: str | None = None,
):
    """Run the full benchmark suite, or a filtered subset.

    Args:
        config_path: Path to YAML config.
        task_filter: If set, run only this task type.
        variant_filter: If set, run only this model variant.
        datasets_dir: If set, save/load datasets as .pt files in this directory.
        generate_only: If True, generate datasets and exit (no training).
        results_file: If set, save results to this specific JSON path.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    wc = config.get("wave")  # Optional wave config section
    lc = config.get("llm")   # Optional LLM config section

    device = _resolve_device(tc)
    base_seed = bc.get("seed", 42)

    tier = bc.get("tier", 1)
    tasks = list(TIER_1_TASKS)
    if tier >= 2:
        tasks.extend(TIER_2_TASKS)
    if tier >= 3:
        tasks.extend(TIER_3_TASKS)

    if task_filter:
        if task_filter not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_filter}. Available: {list(TASK_REGISTRY.keys())}")
        tasks = [task_filter]

    variants = bc.get("model_variants", ["hierarchical", "symmetric", "hierarchical_nowave"])
    if variant_filter:
        variants = [variant_filter]

    collect_diag = bc.get("collect_diagnostics", True)
    output_dir = Path(bc.get("output_dir", "data/benchmark_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = bc.get("checkpoint_dir")
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Per-job seeding: deterministic and non-overlapping
    all_task_list = list(TIER_1_TASKS) + list(TIER_2_TASKS) + list(TIER_3_TASKS)
    all_variant_list = ["hierarchical", "symmetric", "hierarchical_nowave", "hierarchical_llm"]

    print("=" * 72)
    mode = "generate-only" if generate_only else "train+eval"
    print(f"Benchmark Suite ({mode}, {len(tasks)} tasks, {len(variants)} variants)")
    print(f"Device: {device}")
    if datasets_dir:
        print(f"Datasets dir: {datasets_dir}")
    print("=" * 72)

    all_results = {}
    all_diagnostics = {}

    for task_type in tasks:
        # Compute per-task seed
        task_idx = all_task_list.index(task_type) if task_type in all_task_list else 0
        task_seed = base_seed + task_idx * 100
        random.seed(task_seed)
        torch.manual_seed(task_seed)

        print(f"\n{'─' * 72}")
        print(f"Task: {task_type} (seed={task_seed})")
        print(f"{'─' * 72}")

        max_classes = get_max_classes(task_type)
        datasets = _generate_datasets(task_type, bc, mc, datasets_dir=datasets_dir)
        task_results = {}

        if generate_only:
            print(f"  Datasets generated. Skipping training (--generate-only).")
            continue

        # For tier 2 tasks, only run hierarchical (unless variant_filter overrides)
        task_variants = variants if tier < 2 or task_type in TIER_1_TASKS else ["hierarchical"]
        if variant_filter:
            task_variants = [variant_filter]

        for variant in task_variants:
            # Per-variant seed
            variant_idx = all_variant_list.index(variant) if variant in all_variant_list else 0
            variant_seed = task_seed + variant_idx
            random.seed(variant_seed)
            torch.manual_seed(variant_seed)

            print(f"\n  Model: {variant} (seed={variant_seed})")
            start = time.time()

            # --- ID evaluation ---
            model = _build_model(variant, mc, max_classes, device, wave_config=wc, llm_config=lc)
            params = sum(p.numel() for p in model.parameters())
            print(f"    Parameters: {params:,}")

            print(f"    Training on ID split...")
            ckpt_path = (
                str(Path(checkpoint_dir) / f"{task_type}_{variant}_id.pt")
                if checkpoint_dir else None
            )
            train_info = _train_and_evaluate(
                model, datasets["id_train"], datasets["id_val"], tc, device,
                checkpoint_path=ckpt_path,
            )
            id_acc = _evaluate_dataset(model, datasets["id_test"], device)
            print(f"    ID accuracy: {id_acc:.3f} "
                  f"(val: {train_info['best_val_acc']:.3f}, "
                  f"ep: {train_info['best_epoch']}/{train_info['total_epochs']})")

            result = {
                "id_accuracy": id_acc,
                "params": params,
                "best_val_acc": train_info['best_val_acc'],
                "best_train_loss": train_info['best_train_loss'],
                "best_epoch": train_info['best_epoch'],
                "total_epochs": train_info['total_epochs'],
                "training_curve": train_info['training_curve'],
            }

            # --- Per-class accuracy ---
            print(f"    Per-class accuracy...")
            pc_info = evaluate_per_class(model, datasets["id_test"], max_classes, device)
            result["per_class"] = pc_info["per_class"]
            result["confusion"] = pc_info["confusion"]
            # Print summary of worst classes
            sorted_classes = sorted(
                pc_info["per_class"].items(),
                key=lambda x: x[1]["accuracy"],
            )
            if sorted_classes:
                worst_3 = sorted_classes[:3]
                best_3 = sorted_classes[-3:]
                w_parts = [f"c{c}={v['accuracy']:.0%}({v['total']})" for c, v in worst_3]
                b_parts = [f"c{c}={v['accuracy']:.0%}({v['total']})" for c, v in best_3]
                print(f"      Worst: {', '.join(w_parts)}")
                print(f"      Best:  {', '.join(b_parts)}")

            # Size generalization (using the ID-trained model)
            test_sizes = bc.get("test_n_nodes", [])
            for size in test_sizes:
                key = f"size_{size}"
                if key in datasets:
                    acc = _evaluate_dataset(model, datasets[key], device)
                    result[f"size_{size}_accuracy"] = acc
                    print(f"    Size n={size}: {acc:.3f}")

            # --- Topology transfer ---
            if "topo_train" in datasets:
                topo_model = _build_model(variant, mc, max_classes, device, wave_config=wc, llm_config=lc)
                print(f"    Training on topo-restricted split...")
                topo_ckpt_path = (
                    str(Path(checkpoint_dir) / f"{task_type}_{variant}_topo.pt")
                    if checkpoint_dir else None
                )
                topo_train_info = _train_and_evaluate(
                    topo_model, datasets["topo_train"], datasets["topo_val"], tc, device,
                    checkpoint_path=topo_ckpt_path,
                )
                topo_acc = _evaluate_dataset(topo_model, datasets["topo_test"], device)
                result["topo_transfer_accuracy"] = topo_acc
                result["topo_training_curve"] = topo_train_info['training_curve']
                print(f"    Topo transfer: {topo_acc:.3f}")

                # --- Per-topology breakdown ---
                test_topos = bc.get("test_topologies", [])
                if test_topos:
                    topo_breakdown = _evaluate_per_topology(
                        topo_model, task_type, bc, mc, test_topos, max_classes, device,
                        datasets_dir=datasets_dir,
                    )
                    result["topo_breakdown"] = topo_breakdown
                    for tname, tacc in topo_breakdown.items():
                        print(f"      {tname}: {tacc:.3f}")

            elapsed = time.time() - start
            result["time"] = elapsed
            task_results[variant] = result
            print(f"    Time: {elapsed:.0f}s")

            # --- Diagnostics (hierarchical variants only) ---
            diag_summary = None
            if collect_diag and variant.startswith("hierarchical") and hasattr(model, 'executive_loop'):
                print(f"    Collecting diagnostics...")
                collector = DiagnosticCollector()
                collector.collect(model, datasets["id_test"], device)
                diag_summary = collector.summarize()
                all_diagnostics.setdefault(task_type, {})[variant] = diag_summary

            # Per-job results file (for multi-GPU mode)
            if results_file:
                job_data = {
                    "task": task_type,
                    "variant": variant,
                    "result": result,
                    "diagnostics": diag_summary,
                }
                rf = Path(results_file)
                rf.parent.mkdir(parents=True, exist_ok=True)
                with open(rf, "w") as f:
                    json.dump(job_data, f, indent=2, default=str)
                print(f"    Results saved to {rf}")

        all_results[task_type] = task_results

    if generate_only:
        print("\nDataset generation complete.")
        return all_results, all_diagnostics

    # Print summary tables (only useful when running multiple tasks)
    if len(tasks) > 1 or len(variants) > 1:
        _print_results_table(all_results, bc)
        _print_diagnostics_table(all_diagnostics)

    # Save results (unless per-job results_file was used)
    if not results_file:
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")

        if all_diagnostics:
            diag_path = output_dir / "diagnostics.json"
            with open(diag_path, "w") as f:
                json.dump(all_diagnostics, f, indent=2, default=str)
            print(f"Diagnostics saved to {diag_path}")

    return all_results, all_diagnostics


def _print_results_table(results: dict, bc: dict):
    """Print the main results table."""
    test_sizes = bc.get("test_n_nodes", [])

    print(f"\n{'=' * 72}")
    print("Benchmark Suite Results")
    print(f"{'=' * 72}")

    header = f"{'Task':<18} {'Model':<12} {'ID (n=20)':>10}"
    if any("topo_transfer_accuracy" in r.get(v, {})
           for r in results.values() for v in r):
        header += f" {'Topo Xfer':>10}"
    for size in test_sizes:
        if size != bc["train_n_nodes"]:
            header += f" {'n=' + str(size):>10}"
    print(header)
    print("─" * len(header))

    for task, task_results in results.items():
        for i, (variant, res) in enumerate(task_results.items()):
            task_label = task if i == 0 else ""
            var_short = {"hierarchical": "hier", "symmetric": "symm",
                         "hierarchical_nowave": "nowave"}.get(variant, variant)
            line = f"{task_label:<18} {var_short:<12} {res['id_accuracy']:>9.1%}"
            if "topo_transfer_accuracy" in res:
                line += f" {res['topo_transfer_accuracy']:>9.1%}"
            elif any("topo_transfer_accuracy" in r.get(v, {})
                     for r in results.values() for v in r):
                line += f" {'':>10}"
            for size in test_sizes:
                key = f"size_{size}_accuracy"
                if size != bc["train_n_nodes"]:
                    if key in res:
                        line += f" {res[key]:>9.1%}"
                    else:
                        line += f" {'':>10}"
            print(line)


def _print_diagnostics_table(diagnostics: dict):
    """Print executive diagnostics summary table."""
    if not diagnostics:
        return

    print(f"\n{'=' * 72}")
    print("Executive Diagnostics (Hierarchical)")
    print(f"{'=' * 72}")
    print(f"{'Task':<18} {'fg_ent':>8} {'sf_gini':>8} {'conf':>8} "
          f"{'iters':>8} {'diff_t':>8}")
    print("─" * 58)

    for task, variants in diagnostics.items():
        for variant, summary in variants.items():
            if variant != "hierarchical":
                continue
            fg_ent = summary.get("frequency_gate", {}).get("entropy", 0)
            sf_gini = summary.get("spatial_focus", {}).get("gini", 0)
            conf = summary.get("confidence", {}).get("mean", 0)
            iters = summary.get("num_iterations", {}).get("mean", 0)
            diff_t = summary.get("diffusion_time", {}).get("mean", 0)
            print(f"{task:<18} {fg_ent:>8.3f} {sf_gini:>8.3f} {conf:>8.3f} "
                  f"{iters:>8.1f} {diff_t:>8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark suite runner")
    parser.add_argument("config", nargs="?", default="config/benchmark_suite.yaml",
                        help="Path to YAML config (default: config/benchmark_suite.yaml)")
    parser.add_argument("--task", default=None,
                        help="Run only this task type (e.g. 'diverse', 'hodge_class')")
    parser.add_argument("--variant", default=None,
                        help="Run only this model variant (e.g. 'hierarchical')")
    parser.add_argument("--datasets-dir", default=None,
                        help="Save/load pre-generated datasets in this directory")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate datasets and exit (no training)")
    parser.add_argument("--results-file", default=None,
                        help="Save this job's results to a specific JSON file")
    args = parser.parse_args()

    run_benchmark_suite(
        config_path=args.config,
        task_filter=args.task,
        variant_filter=args.variant,
        datasets_dir=args.datasets_dir,
        generate_only=args.generate_only,
        results_file=args.results_file,
    )
