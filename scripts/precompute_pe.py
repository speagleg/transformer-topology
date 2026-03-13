#!/usr/bin/env python
"""Add precomputed TopologicalPE features to existing v10 datasets.

Run once after deploying the GPU optimization code. Adds 'precomputed_pe'
to each sample's metadata, eliminating gudhi calls during training.

Usage:
    PYTHONPATH=. python scripts/precompute_pe.py data/v10_datasets
"""

import sys
import torch
from pathlib import Path
from src.benchmarks.benchmark_dataset import BenchmarkDataset, precompute_pe_for_sample


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/precompute_pe.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.exists():
        print(f"Directory not found: {dataset_dir}")
        sys.exit(1)

    for pt_file in sorted(dataset_dir.glob("*.pt")):
        print(f"Processing {pt_file.name}...")
        ds = BenchmarkDataset.load(str(pt_file))

        already_done = 0
        updated = 0
        for i, sample in enumerate(ds.samples):
            if len(sample) == 5:
                cc, q, t, ans, meta = sample
            else:
                cc, q, t, ans = sample
                meta = {}

            if meta is None:
                meta = {}

            if 'precomputed_pe' in meta:
                already_done += 1
                continue

            meta['precomputed_pe'] = precompute_pe_for_sample(cc)
            ds.samples[i] = (cc, q, t, ans, meta)
            updated += 1

            if (updated % 500) == 0:
                print(f"  {updated} samples updated...")

        if updated > 0:
            ds.save(str(pt_file))
            print(f"  Saved: {updated} updated, {already_done} already had PE")
        else:
            print(f"  Skipped: all {already_done} samples already have PE")


if __name__ == "__main__":
    main()
