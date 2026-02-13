"""Merge per-job benchmark result JSON files into a single results.json.

Usage:
    python scripts/merge_results.py data/benchmark_jobs/results/ [--output results.json]
"""

import argparse
import json
from pathlib import Path


def merge_results(results_dir: str, output_path: str | None = None) -> dict:
    """Merge per-job result_<task>_<variant>.json into a single results dict.

    Each per-job file should contain: {"task": str, "variant": str, "result": dict}
    Output format is compatible with _print_results_table():
        {task: {variant: {id_accuracy: ..., ...}, ...}, ...}
    """
    results_dir = Path(results_dir)
    merged = {}
    diagnostics = {}

    for path in sorted(results_dir.glob("result_*.json")):
        with open(path) as f:
            data = json.load(f)

        task = data["task"]
        variant = data["variant"]
        result = data["result"]

        merged.setdefault(task, {})[variant] = result

        if "diagnostics" in data and data["diagnostics"]:
            diagnostics.setdefault(task, {})[variant] = data["diagnostics"]

    if output_path is None:
        output_path = str(results_dir / "results.json")

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Merged {len(list(results_dir.glob('result_*.json')))} files -> {output_path}")

    if diagnostics:
        diag_path = str(Path(output_path).parent / "diagnostics.json")
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2, default=str)
        print(f"Diagnostics -> {diag_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge per-job benchmark results")
    parser.add_argument("results_dir", help="Directory containing result_*.json files")
    parser.add_argument("--output", default=None, help="Output path (default: results_dir/results.json)")
    args = parser.parse_args()

    merge_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
