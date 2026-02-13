#!/bin/bash
# Multi-GPU benchmark suite launcher
# 4-phase pipeline: generate datasets → build job list → parallel train → merge results
#
# Usage:
#   ./scripts/launch_benchmark_multi_gpu.sh [config.yaml]
#
# Environment variables:
#   GPU_IDS       - Comma-separated GPU IDs (default: auto-detect)
#   MAX_PARALLEL  - Max concurrent jobs (default: num GPUs)
#   DATASETS_DIR  - Dataset cache directory (default: data/benchmark_datasets)
#   RESULTS_DIR   - Per-job results directory (default: data/benchmark_jobs/results)
#   LOG_DIR       - Job log directory (default: data/benchmark_jobs/logs)
set -e

CONFIG="${1:-config/benchmark_suite.yaml}"
DATASETS_DIR="${DATASETS_DIR:-data/benchmark_datasets}"
RESULTS_DIR="${RESULTS_DIR:-data/benchmark_jobs/results}"
LOG_DIR="${LOG_DIR:-data/benchmark_jobs/logs}"
JOBS_FILE="data/benchmark_jobs/jobs.txt"

# Read tasks and variants from config
TIER=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c['benchmark'].get('tier', 1))
")

TIER1_TASKS="diverse propagation_delay spectral_gap hodge_class bfs"
TIER2_TASKS="blocking interference cycle_detection betti_number path_counting dijkstra"

if [ "$TIER" -ge 2 ]; then
    TASKS="$TIER1_TASKS $TIER2_TASKS"
else
    TASKS="$TIER1_TASKS"
fi

VARIANTS=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
vs = c['benchmark'].get('model_variants', ['hierarchical', 'symmetric', 'hierarchical_nowave'])
print(' '.join(vs))
")

echo "============================================================"
echo "Multi-GPU Benchmark Suite"
echo "============================================================"
echo "Config:     $CONFIG"
echo "Tier:       $TIER"
echo "Tasks:      $TASKS"
echo "Variants:   $VARIANTS"
echo "Datasets:   $DATASETS_DIR"
echo "Results:    $RESULTS_DIR"
echo "Logs:       $LOG_DIR"
echo "GPU_IDS:    ${GPU_IDS:-auto}"
echo "============================================================"

# GPU check
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Phase 1: Generate datasets (serial, CPU-bound, fast)
echo ""
echo "=== Phase 1: Generating datasets ==="
mkdir -p "$DATASETS_DIR"

for TASK in $TASKS; do
    echo "--- $TASK ---"
    python3 -u -m src.benchmarks.run_benchmark_suite "$CONFIG" \
        --task "$TASK" \
        --datasets-dir "$DATASETS_DIR" \
        --generate-only
done

# Phase 2: Generate job list
echo ""
echo "=== Phase 2: Building job list ==="
mkdir -p "$(dirname "$JOBS_FILE")" "$RESULTS_DIR"

> "$JOBS_FILE"
for TASK in $TASKS; do
    # Tier 2 tasks only get hierarchical variant
    if echo "$TIER2_TASKS" | grep -qw "$TASK"; then
        JOB_VARIANTS="hierarchical"
    else
        JOB_VARIANTS="$VARIANTS"
    fi

    for VARIANT in $JOB_VARIANTS; do
        RESULT_FILE="$RESULTS_DIR/result_${TASK}_${VARIANT}.json"
        CMD="python3 -u -m src.benchmarks.run_benchmark_suite $CONFIG"
        CMD="$CMD --task $TASK --variant $VARIANT"
        CMD="$CMD --datasets-dir $DATASETS_DIR"
        CMD="$CMD --results-file $RESULT_FILE"
        echo "$CMD" >> "$JOBS_FILE"
    done
done

NUM_JOBS=$(wc -l < "$JOBS_FILE")
echo "Generated $NUM_JOBS jobs -> $JOBS_FILE"

# Phase 3: Parallel training
echo ""
echo "=== Phase 3: Parallel training ($NUM_JOBS jobs) ==="
mkdir -p "$LOG_DIR"

LAUNCHER_ARGS="--jobs $JOBS_FILE --log-dir $LOG_DIR"
if [ -n "$GPU_IDS" ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --gpus $GPU_IDS"
fi
if [ -n "$MAX_PARALLEL" ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --max-parallel $MAX_PARALLEL"
fi

python3 scripts/gpu_launcher.py $LAUNCHER_ARGS

# Phase 4: Merge results
echo ""
echo "=== Phase 4: Merging results ==="
OUTPUT_DIR=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c['benchmark'].get('output_dir', 'data/benchmark_results'))
")
mkdir -p "$OUTPUT_DIR"

python3 scripts/merge_results.py "$RESULTS_DIR" --output "$OUTPUT_DIR/results.json"

echo ""
echo "=== Done! ==="
echo "Results: $OUTPUT_DIR/results.json"
echo "Logs:    $LOG_DIR/"
