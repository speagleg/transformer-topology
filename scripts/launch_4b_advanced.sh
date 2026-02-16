#!/bin/bash
# Phase 4b — magnetic laplacian + sheaf diffusion experiments
# Reuses datasets from the main benchmark run.
# 2 modes × 5 tasks × 1 variant = 10 jobs
# Expected runtime: ~30-60 min on 8x A100
#
# Usage:
#   ./scripts/launch_4b_advanced.sh

set -e

DATASETS_DIR="data/benchmark_4b_datasets"
RESULTS_BASE="data/benchmark_4b_results"
LOG_DIR="$RESULTS_BASE/logs_advanced"
JOBS_FILE="$RESULTS_BASE/jobs_advanced.txt"

MODES="magnetic sheaf"
TASKS="diverse propagation_delay spectral_gap hodge_class bfs"

echo "============================================================"
echo "Phase 4b — Magnetic Laplacian + Sheaf Diffusion"
echo "============================================================"
echo "Modes:     $MODES"
echo "Tasks:     $TASKS"
echo "Datasets:  $DATASETS_DIR (reusing existing)"
echo "============================================================"

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Verify datasets exist
EXPECTED=$(ls "$DATASETS_DIR"/*.pt 2>/dev/null | wc -l)
if [ "$EXPECTED" -lt 25 ]; then
    echo "ERROR: Expected 25+ dataset files in $DATASETS_DIR, found $EXPECTED"
    exit 1
fi
echo "Found $EXPECTED dataset files — good."

# Build job list
echo ""
echo "=== Building job list ==="
mkdir -p "$LOG_DIR"

> "$JOBS_FILE"
for MODE in $MODES; do
    CONFIG="config/benchmark_4b_${MODE}.yaml"
    RESULTS_DIR="$RESULTS_BASE/jobs_${MODE}"
    mkdir -p "$RESULTS_DIR"
    for TASK in $TASKS; do
        RESULT_FILE="$RESULTS_DIR/result_${TASK}_hierarchical.json"
        CMD="python3 -u -m src.benchmarks.run_benchmark_suite $CONFIG"
        CMD="$CMD --task $TASK --variant hierarchical"
        CMD="$CMD --datasets-dir $DATASETS_DIR"
        CMD="$CMD --results-file $RESULT_FILE"
        echo "$CMD" >> "$JOBS_FILE"
    done
done

NUM_JOBS=$(wc -l < "$JOBS_FILE")
echo "Generated $NUM_JOBS jobs -> $JOBS_FILE"

# Launch
echo ""
echo "=== Launching ($NUM_JOBS jobs) ==="

LAUNCHER_ARGS="--jobs $JOBS_FILE --log-dir $LOG_DIR"
if [ -n "$GPU_IDS" ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --gpus $GPU_IDS"
fi
if [ -n "$MAX_PARALLEL" ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --max-parallel $MAX_PARALLEL"
fi

python3 scripts/gpu_launcher.py $LAUNCHER_ARGS

# Merge results
echo ""
echo "=== Merging results ==="
for MODE in $MODES; do
    RESULTS_DIR="$RESULTS_BASE/jobs_${MODE}"
    if ls "$RESULTS_DIR"/result_*.json 1>/dev/null 2>&1; then
        python3 scripts/merge_results.py "$RESULTS_DIR" --output "$RESULTS_BASE/results_${MODE}.json"
    fi
done

echo ""
echo "=== Done! ==="
for MODE in $MODES; do
    echo "  $MODE: $RESULTS_BASE/results_${MODE}.json"
done
