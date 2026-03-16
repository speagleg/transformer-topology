"""Inverse-scaling experiment orchestrator: run all configs in sequence.

Trains and evaluates each experiment configuration (baseline, multiscale,
sheaf_only, sheaf_ensemble) one by one.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/run_all.py

    # Evaluate only (skip training):
    PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/run_all.py --eval-only

    # Run a subset of configs:
    PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/run_all.py \
        --configs baseline multiscale
"""

import argparse
import sys
import time
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent / "configs"

DEFAULT_CONFIGS = ["baseline", "multiscale", "sheaf_only", "sheaf_ensemble"]


def run_training(config_path: str):
    """Run training for a single config."""
    from scripts.experiments.inverse_scaling.train import main as train_main
    print(f"\n{'#' * 72}")
    print(f"# TRAINING: {config_path}")
    print(f"{'#' * 72}\n")
    train_main(config_path)


def run_evaluation(config_path: str):
    """Run OOD evaluation for a single config."""
    from scripts.experiments.inverse_scaling.evaluate_ood import main as eval_main
    print(f"\n{'#' * 72}")
    print(f"# EVALUATION: {config_path}")
    print(f"{'#' * 72}\n")
    eval_main(config_path)


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate all inverse-scaling experiments"
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help=f"Config names to run (default: {DEFAULT_CONFIGS})")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only run OOD evaluation")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip evaluation, only run training")
    args = parser.parse_args()

    config_paths = []
    for name in args.configs:
        path = CONFIGS_DIR / f"{name}.yaml"
        if not path.exists():
            print(f"ERROR: Config not found: {path}")
            sys.exit(1)
        config_paths.append(str(path))

    print("=" * 72)
    print("Inverse Scaling Experiment Suite")
    print(f"Configs: {args.configs}")
    print(f"Mode: {'eval-only' if args.eval_only else 'train-only' if args.train_only else 'train+eval'}")
    print("=" * 72)

    total_start = time.time()

    for config_path in config_paths:
        exp_start = time.time()

        if not args.eval_only:
            run_training(config_path)

        if not args.train_only:
            run_evaluation(config_path)

        exp_elapsed = time.time() - exp_start
        print(f"\n  Experiment time: {exp_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 72}")
    print(f"All experiments complete. Total time: {total_elapsed/60:.1f} min")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
