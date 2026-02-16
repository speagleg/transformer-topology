#!/usr/bin/env python3
"""Run a single benchmark task+variant from a config file.

Thin wrapper around run_benchmark_suite that outputs results to a JSON file.

Usage:
    python3 scripts/run_single_benchmark.py \
        --config config/benchmark_4b_sheaf.yaml \
        --task hodge_class --variant hierarchical \
        --output results/sheaf_hodge_class.json
"""

import argparse
from src.benchmarks.run_benchmark_suite import run_benchmark_suite


def main():
    parser = argparse.ArgumentParser(description="Run single benchmark task")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--task", required=True, help="Task type (bfs, diverse, etc.)")
    parser.add_argument("--variant", default="hierarchical", help="Model variant")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--datasets-dir", default=None, help="Cache datasets directory")
    args = parser.parse_args()

    run_benchmark_suite(
        config_path=args.config,
        task_filter=args.task,
        variant_filter=args.variant,
        datasets_dir=args.datasets_dir,
        results_file=args.output,
    )


if __name__ == "__main__":
    main()
