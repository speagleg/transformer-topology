"""Exp 0a: Audit hodge_class dataset class balance across graph sizes.

Generates 1000 samples at each test size and reports gradient/curl/harmonic
distribution. Critical for identifying whether curl is underrepresented at
small graph sizes (few triangles → curl falls back to gradient).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/audit_class_balance.py
"""

import sys
from collections import Counter

from src.benchmarks.benchmark_dataset import BenchmarkDataset

TEST_SIZES = [20, 40, 80, 120, 160, 200, 320]
NUM_SAMPLES = 1000
EMBEDDING_DIM = 64
CLASS_NAMES = {0: "gradient", 1: "curl", 2: "harmonic"}


def audit_size(n_nodes: int) -> Counter:
    """Generate NUM_SAMPLES hodge_class samples at n_nodes, return class counts."""
    print(f"  Generating {NUM_SAMPLES} samples at n={n_nodes}...", flush=True)
    dataset = BenchmarkDataset(
        num_samples=NUM_SAMPLES,
        task_type="hodge_class",
        n_nodes=n_nodes,
        embedding_dim=EMBEDDING_DIM,
    )
    counts: Counter = Counter()
    for i in range(len(dataset)):
        cc, src, tgt, answer = dataset.samples[i]
        counts[int(answer)] += 1
    return counts


def print_table(results: list[tuple[int, Counter]]):
    """Print a formatted table of class distribution results."""
    header = f"{'n':>6} | {'gradient':>10} {'curl':>10} {'harmonic':>10} | {'curl%':>7} | {'gradient%':>10} | {'harmonic%':>10}"
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for n_nodes, counts in results:
        total = sum(counts.values())
        grad = counts.get(0, 0)
        curl = counts.get(1, 0)
        harm = counts.get(2, 0)
        curl_pct = 100.0 * curl / total if total > 0 else 0.0
        grad_pct = 100.0 * grad / total if total > 0 else 0.0
        harm_pct = 100.0 * harm / total if total > 0 else 0.0
        print(
            f"{n_nodes:>6} | {grad:>10} {curl:>10} {harm:>10} | {curl_pct:>6.1f}% | {grad_pct:>9.1f}% | {harm_pct:>9.1f}%"
        )
    print(sep)
    print()
    print("Expected balanced: ~33.3% each class.")
    print("Low curl% at small n → curl falls back to gradient (no triangles).")


def main():
    print(f"Hodge class balance audit: {NUM_SAMPLES} samples per size")
    print(f"Sizes: {TEST_SIZES}")
    print()

    results = []
    for n_nodes in TEST_SIZES:
        try:
            counts = audit_size(n_nodes)
            results.append((n_nodes, counts))
            total = sum(counts.values())
            print(
                f"    n={n_nodes}: gradient={counts[0]} curl={counts[1]} harmonic={counts[2]} "
                f"(curl={100.0*counts[1]/total:.1f}%)"
            )
        except Exception as e:
            print(f"    n={n_nodes}: ERROR — {e}", file=sys.stderr)
            results.append((n_nodes, Counter()))

    print_table(results)


if __name__ == "__main__":
    main()
