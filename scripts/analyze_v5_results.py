"""Post-training analysis for DSM v5 curriculum results.

Evaluates the final model on all tasks across multiple axes:
  1. Per-task accuracy (all 14 tasks)
  2. Per-class breakdown for semantic tasks
  3. Size generalization (n=20, 40, 80)
  4. Topology transfer (train topos vs held-out)
  5. DSM contribution analysis (with vs without DSM)
  6. Control signal statistics across task types
"""

import copy
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, TASK_REGISTRY, get_max_classes
from src.benchmarks.diagnostics import DiagnosticCollector
from src.benchmarks.run_benchmark_suite import _build_model, _resolve_device
from src.benchmarks.run_comparison import evaluate, evaluate_per_class, _unpack_sample


def _load_or_generate(task, split, n_nodes, embedding_dim, topologies=None,
                      n_nodes_range=None, num_samples=200, pregen_dir=None):
    if pregen_dir is not None:
        range_str = f"n{n_nodes_range[0]}-{n_nodes_range[1]}" if n_nodes_range else f"n{n_nodes}"
        path = pregen_dir / f"{task}_{split}_{range_str}.pt"
        if path.exists():
            ds = BenchmarkDataset.load(str(path))
            if len(ds) > num_samples:
                ds.samples = ds.samples[:num_samples]
            return ds
    return BenchmarkDataset(num_samples, task, n_nodes, embedding_dim,
                            topologies=topologies, n_nodes_range=n_nodes_range)


def _rebuild_classifier(model, task):
    num_classes = get_max_classes(task)
    if model.classifier[-1].out_features == num_classes:
        return
    old_in = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(old_in, num_classes)
    model.classifier[-1].to(next(model.parameters()).device)


def run_analysis(config_path="config/dsm_training.yaml",
                 model_path="data/dsm_results/dsm_model_final.pt",
                 pregen_dir_str="data/dsm_datasets"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    wc = config.get("wave")
    lc = config.get("llm")

    device = _resolve_device(tc)
    emb_dim = mc["embedding_dim"]
    train_n = bc["train_n_nodes"]
    pregen_dir = Path(pregen_dir_str) if pregen_dir_str else None

    train_topos = bc.get("train_topologies", [])
    test_topos = bc.get("test_topologies", [])
    all_topos = (train_topos + test_topos) or None

    n_min = bc.get("train_n_nodes_min")
    n_max = bc.get("train_n_nodes_max")
    train_range = (n_min, n_max) if n_min and n_max else None

    # Build and load model
    initial_classes = get_max_classes("diverse")
    model = _build_model("hierarchical_llm", mc, initial_classes, device,
                         wave_config=wc, llm_config=lc)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    # Filter out classifier keys with shape mismatch (checkpoint may have different num_classes)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape}
    skipped = set(state.keys()) - set(filtered.keys())
    model.load_state_dict(filtered, strict=False)
    del state
    if skipped:
        print(f"Skipped {len(skipped)} keys with shape mismatch: {skipped}")
    print(f"Loaded model from {model_path}")
    print(f"Device: {device}")

    results = {}

    # ================================================================
    # 1. Per-task accuracy on all available tasks
    # ================================================================
    print("\n" + "=" * 72)
    print("1. PER-TASK ACCURACY (val set, n=train range)")
    print("=" * 72)

    all_tasks = ["diverse", "bfs", "hodge_class", "spectral_gap", "path_counting",
                 "graph_completion", "labeled_reasoning", "analogical_transfer",
                 "cycle_detection", "betti_number", "blocking", "interference"]

    task_results = {}
    for task in all_tasks:
        try:
            _rebuild_classifier(model, task)
            ds = _load_or_generate(task, "val", train_n, emb_dim,
                                   topologies=all_topos, n_nodes_range=train_range,
                                   num_samples=500, pregen_dir=pregen_dir)
            acc, loss = evaluate(model, ds, device=device)
            task_results[task] = {"accuracy": acc, "loss": loss, "n_samples": len(ds)}
            print(f"  {task:25s} acc={acc:.3f}  loss={loss:.4f}  (n={len(ds)})")
        except Exception as e:
            print(f"  {task:25s} ERROR: {e}")
            task_results[task] = {"error": str(e)}
    results["per_task"] = task_results

    # ================================================================
    # 2. Per-class breakdown for semantic tasks
    # ================================================================
    print("\n" + "=" * 72)
    print("2. PER-CLASS BREAKDOWN (semantic tasks)")
    print("=" * 72)

    semantic_tasks = ["graph_completion", "labeled_reasoning", "analogical_transfer"]
    per_class_results = {}
    for task in semantic_tasks:
        try:
            _rebuild_classifier(model, task)
            num_classes = get_max_classes(task)
            ds = _load_or_generate(task, "val", train_n, emb_dim,
                                   topologies=all_topos, n_nodes_range=train_range,
                                   num_samples=500, pregen_dir=pregen_dir)
            breakdown = evaluate_per_class(model, ds, num_classes, device=device)
            per_class_results[task] = breakdown["per_class"]
            print(f"\n  {task} ({num_classes} classes):")
            for c, info in sorted(breakdown["per_class"].items()):
                print(f"    class {c}: {info['correct']}/{info['total']} = {info['accuracy']:.3f}")
        except Exception as e:
            print(f"  {task}: ERROR: {e}")
    results["per_class"] = per_class_results

    # ================================================================
    # 3. Size generalization
    # ================================================================
    print("\n" + "=" * 72)
    print("3. SIZE GENERALIZATION")
    print("=" * 72)

    size_tasks = ["diverse", "bfs", "graph_completion", "analogical_transfer"]
    test_sizes = bc.get("test_n_nodes", [20, 40, 80])
    size_results = {}
    for task in size_tasks:
        size_results[task] = {}
        _rebuild_classifier(model, task)
        for n in test_sizes:
            try:
                ds = _load_or_generate(task, f"test", n, emb_dim,
                                       topologies=all_topos, num_samples=200,
                                       pregen_dir=pregen_dir)
                acc, loss = evaluate(model, ds, device=device)
                size_results[task][str(n)] = {"accuracy": acc, "loss": loss}
                print(f"  {task:25s} n={n:3d}  acc={acc:.3f}  loss={loss:.4f}")
            except Exception as e:
                print(f"  {task:25s} n={n:3d}  ERROR: {e}")
                size_results[task][str(n)] = {"error": str(e)}
    results["size_generalization"] = size_results

    # ================================================================
    # 4. Topology transfer
    # ================================================================
    print("\n" + "=" * 72)
    print("4. TOPOLOGY TRANSFER (train topos vs held-out)")
    print("=" * 72)

    topo_tasks = ["diverse", "bfs", "graph_completion"]
    topo_results = {}
    for task in topo_tasks:
        topo_results[task] = {}
        _rebuild_classifier(model, task)
        for label, topos in [("train_topos", train_topos), ("test_topos", test_topos)]:
            if not topos:
                continue
            try:
                ds = BenchmarkDataset(200, task, train_n, emb_dim,
                                      topologies=topos, n_nodes_range=train_range)
                acc, loss = evaluate(model, ds, device=device)
                topo_results[task][label] = {"accuracy": acc, "loss": loss,
                                             "topologies": topos}
                print(f"  {task:25s} {label:12s} acc={acc:.3f}  topos={topos}")
            except Exception as e:
                print(f"  {task:25s} {label:12s} ERROR: {e}")
    results["topology_transfer"] = topo_results

    # ================================================================
    # 5. DSM ablation: with vs without DSM
    # ================================================================
    print("\n" + "=" * 72)
    print("5. DSM ABLATION (bypass_llm=True vs False)")
    print("=" * 72)

    ablation_tasks = ["diverse", "graph_completion", "labeled_reasoning",
                      "analogical_transfer"]
    ablation_results = {}
    for task in ablation_tasks:
        _rebuild_classifier(model, task)
        ds = _load_or_generate(task, "val", train_n, emb_dim,
                               topologies=all_topos, n_nodes_range=train_range,
                               num_samples=200, pregen_dir=pregen_dir)

        # With DSM
        model.bypass_llm = False
        acc_with, loss_with = evaluate(model, ds, device=device)

        # Without DSM
        model.bypass_llm = True
        acc_without, loss_without = evaluate(model, ds, device=device)

        model.bypass_llm = False  # restore

        delta = acc_with - acc_without
        ablation_results[task] = {
            "with_dsm": {"accuracy": acc_with, "loss": loss_with},
            "without_dsm": {"accuracy": acc_without, "loss": loss_without},
            "delta": delta,
        }
        marker = "+" if delta > 0 else ""
        print(f"  {task:25s} with={acc_with:.3f}  without={acc_without:.3f}  "
              f"delta={marker}{delta:.3f}")
    results["dsm_ablation"] = ablation_results

    # ================================================================
    # 6. Control signal diagnostics
    # ================================================================
    print("\n" + "=" * 72)
    print("6. CONTROL SIGNAL DIAGNOSTICS")
    print("=" * 72)

    diag_tasks = ["diverse", "bfs", "graph_completion", "labeled_reasoning",
                  "analogical_transfer"]
    diag_results = {}
    for task in diag_tasks:
        _rebuild_classifier(model, task)
        ds = _load_or_generate(task, "val", train_n, emb_dim,
                               topologies=all_topos, n_nodes_range=train_range,
                               num_samples=50, pregen_dir=pregen_dir)
        collector = DiagnosticCollector()
        collector.collect(model, ds, device)
        summary = collector.summarize()
        diag_results[task] = summary

        gate = summary.get("semantic_weight", {})
        conf = summary.get("confidence", {})
        freq = summary.get("frequency_gate", {})
        diff = summary.get("diffusion_time", {})
        damp = summary.get("wave_damping", {})
        filt = summary.get("filter_weights_entropy", {})
        print(f"  {task:25s} gate={gate.get('mean',0):.3f}+/-{gate.get('std',0):.3f}  "
              f"conf={conf.get('mean',0):.3f}  "
              f"diff_t={diff.get('mean',0):.3f}  "
              f"damp={damp.get('mean',0):.3f}  "
              f"filt_H={filt.get('mean',0):.3f}")
    results["diagnostics"] = diag_results

    # ================================================================
    # Save
    # ================================================================
    out_path = Path("data/dsm_results/v5_full_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull analysis saved to {out_path}")

    return results


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config/dsm_training.yaml"
    model = sys.argv[2] if len(sys.argv) > 2 else "data/dsm_results/dsm_model_final.pt"
    pregen = sys.argv[3] if len(sys.argv) > 3 else "data/dsm_datasets"
    run_analysis(config, model, pregen)
