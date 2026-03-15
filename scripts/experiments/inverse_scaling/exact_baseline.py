"""Exp 0b: Exact Hodge decomposition baseline for hodge_class task.

For each test size, generates 500 samples and classifies each by running the
exact Hodge decomposition on the edge signal encoded in the cell complex.
The classification is argmax([gradient.norm(), curl.norm(), harmonic.norm()]).

This establishes the non-learned upper bound — it should achieve near-100%
accuracy at all sizes since it computes the answer directly.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/exact_baseline.py
"""

import sys

import torch

from src.benchmarks.benchmark_dataset import BenchmarkDataset
from src.spectral.decomposition import hodge_decomposition

TEST_SIZES = [20, 40, 80, 120, 160, 200, 320]
NUM_SAMPLES = 500
EMBEDDING_DIM = 64
CLASS_NAMES = ["gradient", "curl", "harmonic"]


def exact_classify(cc, signal: torch.Tensor) -> int:
    """Classify signal by argmax of Hodge component norms."""
    try:
        gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
        norms = [gradient.norm().item(), curl.norm().item(), harmonic.norm().item()]
        return int(norms.index(max(norms)))
    except (RuntimeError, ValueError) as e:
        # If decomposition fails, return -1 (counted as wrong)
        return -1


def evaluate_size(n_nodes: int) -> dict:
    """Generate NUM_SAMPLES samples, run exact decomposition, return accuracy stats."""
    print(f"  Generating {NUM_SAMPLES} samples at n={n_nodes}...", flush=True)
    dataset = BenchmarkDataset(
        num_samples=NUM_SAMPLES,
        task_type="hodge_class",
        n_nodes=n_nodes,
        embedding_dim=EMBEDDING_DIM,
    )

    per_class_correct = {0: 0, 1: 0, 2: 0}
    per_class_total = {0: 0, 1: 0, 2: 0}
    total_correct = 0
    total = 0
    errors = 0

    for i in range(len(dataset)):
        cc, src, tgt, answer = dataset.samples[i]
        gt = int(answer)

        n_edges = cc.num_cells(1)
        if n_edges == 0:
            # No edges — skip
            continue

        # Extract signal from edge embedding dim 0
        signal = cc.get_embeddings(1)[:, 0]

        pred = exact_classify(cc, signal)

        total += 1
        if gt in per_class_total:
            per_class_total[gt] += 1

        if pred == -1:
            errors += 1
        elif pred == gt:
            total_correct += 1
            if gt in per_class_correct:
                per_class_correct[gt] += 1

    overall_acc = total_correct / total if total > 0 else 0.0
    per_class_acc = {}
    for cls in range(3):
        n = per_class_total[cls]
        c = per_class_correct[cls]
        per_class_acc[cls] = c / n if n > 0 else float("nan")

    return {
        "n_nodes": n_nodes,
        "total": total,
        "correct": total_correct,
        "errors": errors,
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "per_class_total": per_class_total,
    }


def print_results(all_results: list[dict]):
    """Print formatted accuracy table."""
    header = (
        f"{'n':>6} | {'overall':>9} | "
        f"{'gradient':>10} {'curl':>10} {'harmonic':>10} | "
        f"{'errors':>7} | {'total':>7}"
    )
    sep = "-" * len(header)
    print()
    print("Exact Hodge Decomposition Baseline Results")
    print(sep)
    print(header)
    print(sep)
    for r in all_results:
        n = r["n_nodes"]
        acc = r["overall_acc"]
        pca = r["per_class_acc"]
        errs = r["errors"]
        total = r["total"]

        def fmt_acc(v):
            if v != v:  # NaN check
                return "     N/A  "
            return f"{100.0 * v:>9.1f}%"

        print(
            f"{n:>6} | {100.0 * acc:>8.1f}% | "
            f"{fmt_acc(pca[0])} {fmt_acc(pca[1])} {fmt_acc(pca[2])} | "
            f"{errs:>7} | {total:>7}"
        )
    print(sep)
    print()
    print("Near-100% overall = exact decomposition works correctly.")
    print("If curl accuracy is low at small n → task has fallback to gradient (expected).")
    print("If gradient accuracy is low → decomposition bug or noisy signal.")


def main():
    print(f"Exact Hodge baseline: {NUM_SAMPLES} samples per size")
    print(f"Sizes: {TEST_SIZES}")
    print()

    all_results = []
    for n_nodes in TEST_SIZES:
        try:
            result = evaluate_size(n_nodes)
            all_results.append(result)
            pca = result["per_class_acc"]
            print(
                f"    n={n_nodes}: overall={100.0*result['overall_acc']:.1f}%  "
                f"gradient={100.0*pca[0]:.1f}%  "
                f"curl={100.0*pca[1]:.1f}%  "
                f"harmonic={100.0*pca[2]:.1f}%  "
                f"(errors={result['errors']})"
            )
        except Exception as e:
            print(f"    n={n_nodes}: ERROR — {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            all_results.append({
                "n_nodes": n_nodes,
                "total": 0,
                "correct": 0,
                "errors": 0,
                "overall_acc": 0.0,
                "per_class_acc": {0: float("nan"), 1: float("nan"), 2: float("nan")},
                "per_class_total": {0: 0, 1: 0, 2: 0},
            })

    print_results(all_results)


if __name__ == "__main__":
    main()
