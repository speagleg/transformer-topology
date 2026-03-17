"""Inverse-scaling experiment: evaluate trained checkpoints at OOD sizes.

Loads a checkpoint trained at n=16-32 and evaluates hodge_class accuracy
at sizes [20, 40, 80, 120, 160, 200, 320]. Reports per-class accuracy
(gradient/curl/harmonic) at each size -- this is the key metric for
detecting inverse scaling.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/evaluate_ood.py \
        scripts/experiments/inverse_scaling/configs/baseline.yaml

    # Evaluate a single seed:
    PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/evaluate_ood.py \
        scripts/experiments/inverse_scaling/configs/baseline.yaml --seed 42
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
from src.benchmarks.run_benchmark_suite import _build_model
from src.benchmarks.run_comparison import _unpack_sample


TEST_SIZES = [20, 40, 80, 120, 160, 200, 320]
NUM_TEST_SAMPLES = 500
CLASS_NAMES = ["gradient", "curl", "harmonic"]


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_at_size(model, n_nodes, embedding_dim, device, num_samples=NUM_TEST_SAMPLES):
    """Generate test data at n_nodes and evaluate with per-class breakdown.

    Uses a custom eval loop (not evaluate_batched) to collect per-sample
    predictions for per-class accuracy reporting.

    Returns:
        dict with keys: n_nodes, accuracy, balanced_accuracy, per_class,
        predictions (list of {true, pred} dicts)
    """
    model.eval()

    print(f"    Generating {num_samples} samples at n={n_nodes}...", flush=True)
    dataset = BenchmarkDataset(
        num_samples=num_samples,
        task_type="hodge_class",
        n_nodes=n_nodes,
        embedding_dim=embedding_dim,
    )

    correct = 0
    n_total = 0
    class_correct = {}
    class_total = {}
    predictions = []

    for i in range(len(dataset)):
        cc, query, target, answer, metadata = _unpack_sample(dataset[i])
        cc = cc.clone().to(device)
        logits = model(cc, query, target, metadata=metadata,
                       topo_features=None, task="hodge_class")
        pred = logits.argmax().item()

        predictions.append({"true": answer, "pred": pred})

        if pred == answer:
            correct += 1
            class_correct[answer] = class_correct.get(answer, 0) + 1
        class_total[answer] = class_total.get(answer, 0) + 1
        n_total += 1

    accuracy = correct / max(n_total, 1)

    per_class_accs = []
    per_class_info = {}
    for c in sorted(class_total.keys()):
        c_correct = class_correct.get(c, 0)
        c_total = class_total[c]
        c_acc = c_correct / c_total if c_total > 0 else 0.0
        per_class_accs.append(c_acc)
        per_class_info[c] = {
            "name": CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}",
            "correct": c_correct,
            "total": c_total,
            "accuracy": round(c_acc, 4),
        }

    balanced_accuracy = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0.0

    return {
        "n_nodes": n_nodes,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "per_class": per_class_info,
        "predictions": predictions,
    }


def evaluate_checkpoint(config, seed, device):
    """Load checkpoint for a seed and evaluate at all OOD sizes."""
    exp = config["experiment"]
    mc = config["model"]
    wc = config.get("wave")
    out = config["output"]

    task = exp["task"]
    max_classes = get_max_classes(task)
    emb_dim = mc["embedding_dim"]

    # Load checkpoint
    ckpt_dir = Path(out["checkpoint_dir"])
    ckpt_path = ckpt_dir / f"{exp['name']}_seed{seed}_best.pt"
    if not ckpt_path.exists():
        print(f"  WARNING: Checkpoint not found: {ckpt_path}")
        return None

    print(f"  Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Build model with same config
    model = _build_model(
        variant="hierarchical",
        mc=mc,
        max_classes=max_classes,
        device=device,
        wave_config=wc,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_val_acc = checkpoint.get("val_acc", "N/A")
    train_epoch = checkpoint.get("epoch", "N/A")
    print(f"  Checkpoint from epoch {train_epoch}, val_acc={train_val_acc}")

    # Seed for reproducible test data
    random.seed(seed + 10000)
    torch.manual_seed(seed + 10000)
    torch.cuda.manual_seed_all(seed + 10000)

    # Evaluate at each OOD size
    size_results = []
    for n_nodes in TEST_SIZES:
        t0 = time.time()
        result = evaluate_at_size(model, n_nodes, emb_dim, device)
        elapsed = time.time() - t0

        pc = result["per_class"]
        pc_str = " | ".join(
            f"{pc[c]['name'][:4]}={pc[c]['accuracy']:.0%}"
            for c in sorted(pc.keys())
        )
        print(f"    n={n_nodes:4d}: acc={result['accuracy']:.3f} "
              f"bal={result['balanced_accuracy']:.3f} | {pc_str} | {elapsed:.0f}s",
              flush=True)

        # Strip predictions for summary (keep full detail in per-seed file)
        size_results.append(result)

    return {
        "seed": seed,
        "train_val_acc": train_val_acc if isinstance(train_val_acc, float) else None,
        "train_epoch": train_epoch,
        "size_results": size_results,
    }


def main(config_path: str = None):
    """Main entry point."""
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Evaluate hodge_class model at OOD sizes"
        )
        parser.add_argument("config", help="Path to YAML config file")
        parser.add_argument("--seed", type=int, default=None,
                            help="Evaluate only this seed (default: all)")
        parser.add_argument("--test-sizes", type=int, nargs="+", default=None,
                            help="Override test sizes (default: 20-320)")
        parser.add_argument("--num-samples", type=int, default=NUM_TEST_SAMPLES,
                            help=f"Samples per size (default: {NUM_TEST_SAMPLES})")
        args = parser.parse_args()
        config_path = args.config
        single_seed = args.seed
        if args.test_sizes:
            global TEST_SIZES
            TEST_SIZES = args.test_sizes
        global NUM_TEST_SAMPLES
        NUM_TEST_SAMPLES = args.num_samples
    else:
        single_seed = None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = _resolve_device()
    exp = config["experiment"]
    out = config["output"]

    seeds = exp["seeds"]
    if single_seed is not None:
        seeds = [single_seed]

    print("=" * 72)
    print(f"OOD Size Evaluation: {exp['name']}")
    print(f"Test sizes: {TEST_SIZES}")
    print(f"Samples per size: {NUM_TEST_SAMPLES}")
    print(f"Seeds: {seeds}, Device: {device}")
    print("=" * 72)

    results_dir = Path(out["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in seeds:
        print(f"\n{'─' * 72}")
        print(f"Seed: {seed}")
        print(f"{'─' * 72}")

        result = evaluate_checkpoint(config, seed, device)
        if result is not None:
            all_results.append(result)

            # Save per-seed results (with predictions)
            seed_path = results_dir / f"{exp['name']}_ood_seed{seed}.json"
            with open(seed_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

    if not all_results:
        print("\nNo checkpoints found. Run train.py first.")
        return

    # Aggregate across seeds
    aggregated = _aggregate_results(all_results)

    # Save aggregated results (without per-sample predictions)
    agg_path = results_dir / f"{exp['name']}_ood_aggregated.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    # Print summary table
    _print_summary(aggregated, exp["name"])
    print(f"\nResults saved to {results_dir}/")


def _aggregate_results(all_results):
    """Aggregate per-seed results into mean/std across seeds."""
    aggregated = {"num_seeds": len(all_results), "sizes": {}}

    # Collect per-size, per-class accuracies across seeds
    for result in all_results:
        for size_result in result["size_results"]:
            n = size_result["n_nodes"]
            key = str(n)
            if key not in aggregated["sizes"]:
                aggregated["sizes"][key] = {
                    "accuracy": [],
                    "balanced_accuracy": [],
                    "per_class": {},
                }
            aggregated["sizes"][key]["accuracy"].append(size_result["accuracy"])
            aggregated["sizes"][key]["balanced_accuracy"].append(
                size_result["balanced_accuracy"]
            )
            for c, info in size_result["per_class"].items():
                c_key = str(c)
                if c_key not in aggregated["sizes"][key]["per_class"]:
                    aggregated["sizes"][key]["per_class"][c_key] = {
                        "name": info["name"],
                        "accuracies": [],
                    }
                aggregated["sizes"][key]["per_class"][c_key]["accuracies"].append(
                    info["accuracy"]
                )

    # Compute means and stds
    import statistics
    for key, data in aggregated["sizes"].items():
        accs = data["accuracy"]
        bal_accs = data["balanced_accuracy"]
        data["mean_accuracy"] = round(statistics.mean(accs), 4)
        data["std_accuracy"] = round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0
        data["mean_balanced_accuracy"] = round(statistics.mean(bal_accs), 4)
        data["std_balanced_accuracy"] = (
            round(statistics.stdev(bal_accs), 4) if len(bal_accs) > 1 else 0.0
        )
        for c_key, c_data in data["per_class"].items():
            c_accs = c_data["accuracies"]
            c_data["mean"] = round(statistics.mean(c_accs), 4)
            c_data["std"] = round(statistics.stdev(c_accs), 4) if len(c_accs) > 1 else 0.0

    return aggregated


def _print_summary(aggregated, name):
    """Print a summary table of results."""
    print(f"\n{'=' * 80}")
    print(f"OOD Evaluation Summary: {name} ({aggregated['num_seeds']} seeds)")
    print(f"{'=' * 80}")

    # Header
    header = f"{'n':>6}  {'Acc':>12}  {'BalAcc':>12}"
    class_names_sorted = None
    for key in sorted(aggregated["sizes"].keys(), key=int):
        data = aggregated["sizes"][key]
        if class_names_sorted is None:
            class_names_sorted = sorted(data["per_class"].keys(), key=int)
            for c_key in class_names_sorted:
                cname = data["per_class"][c_key]["name"][:8]
                header += f"  {cname:>12}"
        break
    print(header)
    print("-" * len(header))

    for key in sorted(aggregated["sizes"].keys(), key=int):
        data = aggregated["sizes"][key]
        line = (f"{key:>6}  "
                f"{data['mean_accuracy']:.3f}+/-{data['std_accuracy']:.3f}  "
                f"{data['mean_balanced_accuracy']:.3f}+/-{data['std_balanced_accuracy']:.3f}")
        for c_key in class_names_sorted:
            c = data["per_class"][c_key]
            line += f"  {c['mean']:.3f}+/-{c['std']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
