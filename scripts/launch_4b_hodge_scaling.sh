#!/bin/bash
# Phase 4b — Hodge Class Scaling Experiment (parallel across GPUs)
#
# Tests hypothesis: hodge_class accuracy improves with graph size.
# 2 filters × 4 sizes = 8 jobs, run in parallel across 8 GPUs.
#
# Step 1: Generate all datasets (sequential, ~10 min)
# Step 2: Launch 8 training jobs via gpu_launcher (~1.5 hrs)
# Step 3: Merge results into cross-size matrix
#
# Usage:
#   ./scripts/launch_4b_hodge_scaling.sh

set -e

DATASETS_DIR="data/benchmark_4b_datasets_hodge_scaling"
RESULTS_DIR="data/benchmark_4b_results/hodge_scaling"
LOG_DIR="data/benchmark_4b_results/logs_hodge_scaling"
JOBS_FILE="data/benchmark_4b_results/hodge_scaling_jobs.txt"
RESULTS_FILE="data/benchmark_4b_results/hodge_scaling_results.json"

FILTERS="schrodinger wave_cosine"
TRAIN_SIZES="20 40 80 160"

echo "============================================================"
echo "Phase 4b — Hodge Class Scaling Experiment"
echo "============================================================"
echo "  Filters:     $FILTERS"
echo "  Train sizes: $TRAIN_SIZES"
echo "  Test sizes:  20, 40, 80, 160, 320"
echo "  Datasets:    $DATASETS_DIR"
echo "============================================================"

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Step 1: Pre-generate all datasets (must be sequential to avoid race conditions)
echo ""
echo "=== Step 1: Generating datasets ==="
python3 -c "
from pathlib import Path
from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes

ds_dir = Path('$DATASETS_DIR')
ds_dir.mkdir(parents=True, exist_ok=True)
topologies = ['ba', 'ws', 'sbm', 'er']
emb_dim = 32
max_classes = get_max_classes('hodge_class')

# Train + val for each training size
for n in [20, 40, 80, 160]:
    for split, count in [('train', 2000), ('val', 500)]:
        name = f'hodge_scaling_{split}_{n}_n{n}.pt'
        path = ds_dir / name
        if path.exists():
            print(f'  {name} exists, skipping')
            continue
        print(f'  Generating {name} ({count} samples, n={n})...')
        ds = BenchmarkDataset(count, 'hodge_class', n, emb_dim, topologies=topologies)
        ds.save(str(path))

# Test datasets at all evaluation sizes
for n in [20, 40, 80, 160, 320]:
    name = f'hodge_scaling_test_{n}_n{n}.pt'
    path = ds_dir / name
    if path.exists():
        print(f'  {name} exists, skipping')
        continue
    print(f'  Generating {name} (500 samples, n={n})...')
    ds = BenchmarkDataset(500, 'hodge_class', n, emb_dim, topologies=topologies)
    ds.save(str(path))

print('Dataset generation complete.')
"

# Step 2: Build job list
echo ""
echo "=== Step 2: Building job list ==="
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

> "$JOBS_FILE"
for FILTER in $FILTERS; do
    for SIZE in $TRAIN_SIZES; do
        RESULT="$RESULTS_DIR/result_${FILTER}_n${SIZE}.json"
        CMD="python3 -u scripts/run_hodge_scaling.py"
        CMD="$CMD --filter $FILTER --train-size $SIZE"
        CMD="$CMD --datasets-dir $DATASETS_DIR"
        CMD="$CMD --results-file $RESULT"
        echo "$CMD" >> "$JOBS_FILE"
    done
done

NUM_JOBS=$(wc -l < "$JOBS_FILE")
echo "Generated $NUM_JOBS jobs -> $JOBS_FILE"
cat "$JOBS_FILE"

# Step 3: Launch in parallel
echo ""
echo "=== Step 3: Launching ($NUM_JOBS jobs in parallel) ==="

LAUNCHER_ARGS="--jobs $JOBS_FILE --log-dir $LOG_DIR"
if [ -n "$GPU_IDS" ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --gpus $GPU_IDS"
fi

python3 scripts/gpu_launcher.py $LAUNCHER_ARGS

# Step 4: Merge into cross-size matrix
echo ""
echo "=== Step 4: Merging results ==="
python3 -c "
import json, glob
from pathlib import Path

files = sorted(glob.glob('$RESULTS_DIR/result_*.json'))
print(f'Found {len(files)} result files')

filters = set()
train_sizes = set()
results = {}

for f in files:
    d = json.load(open(f))
    ft = d['filter']
    ts = d['train_size']
    filters.add(ft)
    train_sizes.add(ts)
    results.setdefault(ft, {})[str(ts)] = d

train_sizes = sorted(train_sizes)
filters = sorted(filters)
test_sizes = [20, 40, 80, 160, 320]

# Print cross-size matrices
for ft in filters:
    print(f'\nCross-Size Matrix: {ft}')
    header = '{:<12}'.format('train\\\test')
    for tn in test_sizes:
        header += '{:>8}'.format(f'n={tn}')
    print(header)
    print('-' * (12 + 8 * len(test_sizes)))
    for ts in train_sizes:
        row = '{:<12}'.format(f'n={ts}')
        r = results.get(ft, {}).get(str(ts), {})
        accs = r.get('test_accuracies', {})
        for tn in test_sizes:
            acc = accs.get(str(tn), 0)
            row += '{:>7.1%} '.format(acc)
        print(row)

# Save merged
output = {
    'experiment': 'hodge_class_scaling',
    'filters': filters,
    'train_sizes': train_sizes,
    'test_sizes': test_sizes,
    'results': results,
}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f'\nMerged results saved to $RESULTS_FILE')
"

echo ""
echo "=== Done! ==="
echo "  Results: $RESULTS_FILE"
