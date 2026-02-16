#!/bin/bash
# Phase 4b validation — lean benchmark focused on size generalization
# + Schrödinger filter experiment running in parallel
#
# Expected runtime: ~2 hours on 8x A100 (15 jobs, 2 rounds)
#   - 5 Tier-1 tasks × 2 variants (hierarchical + nowave) = 10 heat jobs
#   - 5 Tier-1 tasks × 1 variant (hierarchical only) = 5 Schrödinger jobs
#   - 2000 train samples, patience=5, max 30 epochs
#   - Mixed-size training (n=16-32), test on n=20/40/80
#   - No topo-transfer (focus on size gen)
#   - Checkpointing enabled (survives SIGTERM)
#
# Usage:
#   ./scripts/launch_4b_validate.sh              # all 15 jobs
#   SKIP_SCHRODINGER=1 ./scripts/launch_4b_validate.sh  # heat only (10 jobs)
#
# IMPORTANT: Uses separate dataset dir to avoid stale cached data from old runs

set -e

CONFIG_HEAT="config/benchmark_4b_validate.yaml"
CONFIG_SCHRODINGER="config/benchmark_4b_schrodinger.yaml"
# Separate datasets dir — old n=20-fixed datasets are incompatible with mixed-size
DATASETS_DIR="data/benchmark_4b_datasets"
RESULTS_DIR="data/benchmark_4b_results/jobs"
RESULTS_DIR_SCHRODINGER="data/benchmark_4b_results/jobs_schrodinger"
LOG_DIR="data/benchmark_4b_results/logs"
JOBS_FILE="data/benchmark_4b_results/jobs.txt"

TASKS="diverse propagation_delay spectral_gap hodge_class bfs"
HEAT_VARIANTS="hierarchical hierarchical_nowave"
SCHRODINGER_VARIANTS="hierarchical"

echo "============================================================"
echo "Phase 4b Validation Benchmark + Schrödinger Experiment"
echo "============================================================"
echo "Heat config:       $CONFIG_HEAT"
echo "Schrödinger config: $CONFIG_SCHRODINGER"
echo "Tasks:             $TASKS"
echo "Heat variants:     $HEAT_VARIANTS"
echo "Schrödinger:       $SCHRODINGER_VARIANTS (wave-only)"
echo "Datasets:          $DATASETS_DIR (fresh — mixed-size)"
echo "Results:           $RESULTS_DIR"
if [ -n "$SKIP_SCHRODINGER" ]; then
    echo "*** Schrödinger experiment SKIPPED (SKIP_SCHRODINGER=1) ***"
fi
echo "============================================================"

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Phase 1: Generate datasets (serial, CPU-bound)
# Both configs share the same dataset params, so one generation pass suffices
echo ""
echo "=== Phase 1: Generating datasets ==="
mkdir -p "$DATASETS_DIR"

for TASK in $TASKS; do
    echo "--- $TASK ---"
    python3 -u -m src.benchmarks.run_benchmark_suite "$CONFIG_HEAT" \
        --task "$TASK" \
        --datasets-dir "$DATASETS_DIR" \
        --generate-only
done

# Phase 2: Build job list
echo ""
echo "=== Phase 2: Building job list ==="
mkdir -p "$(dirname "$JOBS_FILE")" "$RESULTS_DIR" "$RESULTS_DIR_SCHRODINGER"

> "$JOBS_FILE"

# Heat filter jobs (hierarchical + nowave)
for TASK in $TASKS; do
    for VARIANT in $HEAT_VARIANTS; do
        RESULT_FILE="$RESULTS_DIR/result_${TASK}_${VARIANT}.json"
        CMD="python3 -u -m src.benchmarks.run_benchmark_suite $CONFIG_HEAT"
        CMD="$CMD --task $TASK --variant $VARIANT"
        CMD="$CMD --datasets-dir $DATASETS_DIR"
        CMD="$CMD --results-file $RESULT_FILE"
        echo "$CMD" >> "$JOBS_FILE"
    done
done

# Schrödinger filter jobs (hierarchical only — nowave is identical)
if [ -z "$SKIP_SCHRODINGER" ]; then
    for TASK in $TASKS; do
        for VARIANT in $SCHRODINGER_VARIANTS; do
            RESULT_FILE="$RESULTS_DIR_SCHRODINGER/result_${TASK}_${VARIANT}.json"
            CMD="python3 -u -m src.benchmarks.run_benchmark_suite $CONFIG_SCHRODINGER"
            CMD="$CMD --task $TASK --variant $VARIANT"
            CMD="$CMD --datasets-dir $DATASETS_DIR"
            CMD="$CMD --results-file $RESULT_FILE"
            echo "$CMD" >> "$JOBS_FILE"
        done
    done
fi

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
OUTPUT_DIR="data/benchmark_4b_results"
mkdir -p "$OUTPUT_DIR"

python3 scripts/merge_results.py "$RESULTS_DIR" --output "$OUTPUT_DIR/results_heat.json"

if [ -z "$SKIP_SCHRODINGER" ] && [ -d "$RESULTS_DIR_SCHRODINGER" ] && ls "$RESULTS_DIR_SCHRODINGER"/result_*.json 1>/dev/null 2>&1; then
    python3 scripts/merge_results.py "$RESULTS_DIR_SCHRODINGER" --output "$OUTPUT_DIR/results_schrodinger.json"
fi

echo ""
echo "=== Done! ==="
echo "Results (heat):        $OUTPUT_DIR/results_heat.json"
echo "Results (schrodinger): $OUTPUT_DIR/results_schrodinger.json"
echo "Checkpoints:           data/benchmark_4b_checkpoints/"
echo "Logs:                  $LOG_DIR/"
