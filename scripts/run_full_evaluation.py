#!/usr/bin/env python3
"""Full evaluation of Llama 3.2 1B curriculum training checkpoints.

Evaluates each checkpoint on its own task across 4 evaluation axes:
  - ID_n20:        In-distribution (same size, same topologies)
  - OOD_n48:       Size OOD (2.4x training size)
  - OOD_n80:       Size OOD (5x training size)
  - topo_transfer:  Topology transfer (held-out topologies, same size)

Also runs analogical_transfer task analysis (class distribution, structural
detectability of roles) and diagnostic collection (control signals, filter
weights, gate values).

Usage:
    python -u scripts/run_full_evaluation.py config/llama_training.yaml \
        --checkpoint-dir data/llama_checkpoints \
        --dataset-dir data/llama_datasets \
        [--smoke-test]
"""

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
from src.benchmarks.diagnostics import DiagnosticCollector
from src.benchmarks.run_benchmark_suite import _build_model, _resolve_device
from src.benchmarks.run_comparison import evaluate, evaluate_per_class

# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

# Maps logical name -> (filename, task it was trained on, num_classes for task)
CHECKPOINTS = {
    "phase_a": ("phase_a_spectral_gap_original.pt", "spectral_gap", 8),
    "phase_b_gc": ("phase_b_graph_completion_best.pt", "graph_completion", 2),
    "phase_b_pc": ("phase_b_path_counting_best.pt", "path_counting", 5),
    "phase_c_lr": ("phase_c_labeled_reasoning_best.pt", "labeled_reasoning", 3),
    "phase_c_gc": ("phase_c_graph_completion_best.pt", "graph_completion", 2),
}

# Eval axes: name -> dataset filename pattern ({task} placeholder)
EVAL_AXES = {
    "ID_n20": "{task}_test_n20.pt",
    "OOD_n48": "{task}_test_n48.pt",
    "OOD_n80": "{task}_test_n80.pt",
    "topo_transfer": "{task}_test_topo_n20.pt",
}


def _load_checkpoint(path: Path):
    """Load a checkpoint, handling both raw state_dict and wrapped formats."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        return state["model_state"]
    return state


def _try_load_dataset(dataset_dir: Path, task: str, axis_pattern: str):
    """Load a pregenerated test set, return None if missing."""
    filename = axis_pattern.format(task=task)
    path = dataset_dir / filename
    if not path.exists():
        return None
    return BenchmarkDataset.load(str(path))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(config_path: str, checkpoint_dir: str, dataset_dir: str,
                   smoke_test: bool = False, mock_llm: bool = False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    wc = config.get("wave")
    lc = config.get("llm") or {}

    if mock_llm:
        lc = dict(lc)  # don't mutate original
        lc["backend"] = "mock"
        lc["llm_dim"] = lc.get("llm_dim", 128)

    device = _resolve_device(tc)
    ckpt_dir = Path(checkpoint_dir)
    ds_dir = Path(dataset_dir)

    # Model uses global max_classes (same as curriculum training)
    max_classes = bc.get("max_classes", 16)

    print("=" * 72)
    print("Full Evaluation: Llama 3.2 1B Curriculum Checkpoints")
    print(f"Device: {device}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Datasets: {ds_dir}")
    print(f"Max classes: {max_classes}")
    if mock_llm:
        print("** MOCK LLM MODE (mock backend, no transformers needed) **")
    if smoke_test:
        print("** SMOKE TEST MODE (max 20 samples per eval) **")
    print("=" * 72)

    all_results = {}
    all_diagnostics = {}

    for ckpt_name, (ckpt_file, task, task_classes) in CHECKPOINTS.items():
        ckpt_path = ckpt_dir / ckpt_file
        if not ckpt_path.exists():
            print(f"\n  SKIP {ckpt_name}: {ckpt_path} not found")
            continue

        print(f"\n{'─' * 72}")
        print(f"Checkpoint: {ckpt_name} ({ckpt_file})")
        print(f"Task: {task} ({task_classes} classes)")
        print(f"{'─' * 72}")

        # Build fresh model and load weights
        model = _build_model("hierarchical_llm", mc, max_classes, device,
                             wave_config=wc, llm_config=lc)
        state = _load_checkpoint(ckpt_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys")

        # Enable LLM path for Phase B/C checkpoints
        if "phase_b" in ckpt_name or "phase_c" in ckpt_name:
            model.bypass_llm = False
        else:
            # Phase A was trained with bypass_llm=True
            model.bypass_llm = True

        model.eval()
        ckpt_results = {}

        for axis_name, axis_pattern in EVAL_AXES.items():
            ds = _try_load_dataset(ds_dir, task, axis_pattern)
            if ds is None:
                print(f"  {axis_name}: dataset not found, skipping")
                ckpt_results[axis_name] = {"accuracy": None, "loss": None}
                continue

            if smoke_test and len(ds) > 20:
                ds.samples = ds.samples[:20]

            t0 = time.time()
            acc, loss = evaluate(model, ds, device=device)
            elapsed = time.time() - t0

            per_class = evaluate_per_class(model, ds, task_classes, device=device)

            ckpt_results[axis_name] = {
                "accuracy": acc,
                "loss": loss,
                "elapsed_sec": round(elapsed, 1),
                "num_samples": len(ds),
                "per_class": per_class["per_class"],
                "confusion": per_class["confusion"],
            }
            print(f"  {axis_name}: acc={acc:.3f}  loss={loss:.3f}  "
                  f"({len(ds)} samples, {elapsed:.1f}s)")

            # Per-class breakdown
            for cls_id in sorted(per_class["per_class"].keys()):
                info = per_class["per_class"][cls_id]
                print(f"    class {cls_id}: {info['correct']}/{info['total']} "
                      f"= {info['accuracy']:.3f}")

        # Diagnostics on ID test set
        id_ds = _try_load_dataset(ds_dir, task, EVAL_AXES["ID_n20"])
        if id_ds is not None:
            if smoke_test and len(id_ds) > 20:
                id_ds.samples = id_ds.samples[:20]
            print(f"\n  Collecting diagnostics on ID_n20...")
            collector = DiagnosticCollector()
            collector.collect(model, id_ds, device)
            summary = collector.summarize()
            all_diagnostics[ckpt_name] = summary
            _print_diagnostics(summary)

        all_results[ckpt_name] = {
            "checkpoint": ckpt_file,
            "task": task,
            "task_classes": task_classes,
            "axes": ckpt_results,
        }

        # Free GPU memory between checkpoints
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Analogical transfer task analysis (CPU, no model)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("Analogical Transfer Task Analysis")
    print(f"{'=' * 72}")
    at_analysis = _analyze_analogical_transfer(
        n_samples=50 if smoke_test else 500,
        n_nodes=20,
        embedding_dim=mc["embedding_dim"],
    )
    all_results["analogical_transfer_analysis"] = at_analysis

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("SUMMARY TABLE")
    print(f"{'=' * 72}")
    _print_summary_table(all_results)

    # Save results
    output_path = ckpt_dir / "eval_results.json"
    # Convert non-serializable keys (int confusion matrix keys) to strings
    serializable = _make_serializable(all_results)
    serializable["diagnostics"] = _make_serializable(all_diagnostics)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


# ---------------------------------------------------------------------------
# Analogical transfer task analysis
# ---------------------------------------------------------------------------

def _analyze_analogical_transfer(n_samples: int, n_nodes: int, embedding_dim: int):
    """Analyze analogical_transfer task properties without a model."""
    from src.benchmarks.llm_tasks import generate_analogical_transfer_task

    topologies = ["ba", "ws", "sbm", "er", "grid", "tree", "ladder", "caveman"]
    random.seed(42)

    class_counts = collections.Counter()
    domain_pair_counts = collections.Counter()
    role_name_counts = collections.Counter()
    degree_stats_by_role = collections.defaultdict(list)  # role_idx -> [degree]
    degree_stats_non_role = []
    graph_sizes = []

    for _ in range(n_samples):
        cc, query, target, answer, meta = generate_analogical_transfer_task(
            n_nodes, embedding_dim, topologies=topologies,
        )
        class_counts[answer] += 1
        pair_key = ",".join(meta["domain_a"]) + " / " + ",".join(meta["domain_b"])
        domain_pair_counts[pair_key] += 1
        role_name_counts[meta["role_name"]] += 1

        # Check structural distinguishability of role nodes
        # Role nodes get random assignment, so their degree distribution
        # should be similar to non-role nodes
        n_actual = cc.num_cells(0)
        graph_sizes.append(n_actual)
        if cc.num_cells(1) > 0:
            try:
                B1 = cc.boundary_operator(1)
                degrees = B1.abs().sum(dim=1)  # node degrees from incidence
                query_deg = degrees[query].item() if query < len(degrees) else 0
                degree_stats_by_role[answer].append(query_deg)
            except (RuntimeError, ValueError):
                pass

    # Class distribution
    total = sum(class_counts.values())
    print(f"\n  Class distribution ({n_samples} samples):")
    for cls in sorted(class_counts.keys()):
        pct = class_counts[cls] / total * 100
        print(f"    class {cls}: {class_counts[cls]} ({pct:.1f}%)")

    # Random baseline
    num_classes = 5
    random_baseline = 1.0 / num_classes
    majority_class = max(class_counts.values()) / total
    print(f"\n  Random baseline: {random_baseline:.1%}")
    print(f"  Majority class baseline: {majority_class:.1%}")

    # Domain pair distribution
    print(f"\n  Domain pairs ({len(domain_pair_counts)} unique):")
    for pair, count in domain_pair_counts.most_common():
        print(f"    {pair}: {count}")

    # Structural analysis: role node degree vs average
    print(f"\n  Structural analysis (role node degree vs average):")
    for role_idx in sorted(degree_stats_by_role.keys()):
        degs = degree_stats_by_role[role_idx]
        if degs:
            avg = np.mean(degs)
            std = np.std(degs)
            print(f"    role {role_idx} query degree: {avg:.2f} +/- {std:.2f}")

    # Key finding
    print(f"\n  KEY FINDING: Roles are randomly assigned (line 237 of llm_tasks.py).")
    print(f"  Zero structural signal for GNN/TAT. Task requires LLM to parse")
    print(f"  'query_role=<name>' from metadata task_prompt.")

    return {
        "class_distribution": dict(class_counts),
        "random_baseline": random_baseline,
        "majority_class_baseline": majority_class,
        "domain_pair_counts": dict(domain_pair_counts),
        "role_name_counts": dict(role_name_counts),
        "graph_sizes_mean": float(np.mean(graph_sizes)),
        "degree_by_role": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in degree_stats_by_role.items() if v
        },
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_diagnostics(summary: dict):
    """Print diagnostic summary in a compact format."""
    if not summary:
        return

    ni = summary.get("num_iterations", {})
    print(f"    iterations: mean={ni.get('mean', 0):.1f}  std={ni.get('std', 0):.1f}")

    he = summary.get("harmonic_energy", {})
    print(f"    harmonic_energy: initial={he.get('initial_mean', 0):.4f}  "
          f"final={he.get('final_mean', 0):.4f}  "
          f"delta={he.get('delta_mean', 0):.4f}")

    fg = summary.get("frequency_gate", {})
    if fg:
        print(f"    freq_gate: mean={fg.get('mean', 0):.3f}  "
              f"entropy={fg.get('entropy', 0):.3f}")

    sf = summary.get("spatial_focus", {})
    if sf:
        print(f"    spatial_focus: mean={sf.get('mean', 0):.3f}  "
              f"gini={sf.get('gini', 0):.3f}")

    conf = summary.get("confidence", {})
    if conf:
        print(f"    confidence: mean={conf.get('mean', 0):.3f}  "
              f"std={conf.get('std', 0):.3f}")

    dt = summary.get("diffusion_time", {})
    wd = summary.get("wave_damping", {})
    if dt:
        print(f"    diffusion_time: mean={dt.get('mean', 0):.4f}  "
              f"wave_damping: mean={wd.get('mean', 0):.4f}")

    lg = summary.get("semantic_weight", {})
    if lg:
        print(f"    semantic_weight: mean={lg.get('mean', 0):.3f}  "
              f"std={lg.get('std', 0):.3f}")

    fw = summary.get("filter_weights", {})
    if fw:
        avg_w = fw.get("avg_weights", [])
        w_str = ", ".join(f"{w:.3f}" for w in avg_w)
        print(f"    filter_weights: [{w_str}]  "
              f"entropy={fw.get('entropy_mean', 0):.3f}")


def _print_summary_table(all_results: dict):
    """Print a compact summary table of accuracy across checkpoints and axes."""
    header = f"{'Checkpoint':<16} {'Task':<20} "
    for axis in EVAL_AXES:
        header += f"{axis:>14} "
    print(header)
    print("-" * len(header))

    for ckpt_name, info in all_results.items():
        if ckpt_name == "analogical_transfer_analysis":
            continue
        task = info["task"]
        row = f"{ckpt_name:<16} {task:<20} "
        for axis in EVAL_AXES:
            acc = info["axes"].get(axis, {}).get("accuracy")
            if acc is not None:
                row += f"{acc:>13.1%} "
            else:
                row += f"{'N/A':>14} "
        print(row)


def _make_serializable(obj):
    """Recursively convert dict keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate curriculum checkpoints")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory containing checkpoint .pt files")
    parser.add_argument("--dataset-dir", required=True,
                        help="Directory containing pregenerated test .pt files")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run with 20 samples per eval")
    parser.add_argument("--mock-llm", action="store_true",
                        help="Use mock LLM backend (no transformers needed)")
    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.dataset_dir,
        smoke_test=args.smoke_test,
        mock_llm=args.mock_llm,
    )
