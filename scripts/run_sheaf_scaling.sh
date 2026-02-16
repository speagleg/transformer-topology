#!/bin/bash
# Phase 4b — Sheaf Stability + Hodge Scaling Experiment
#
# Designed for a single cheap GPU (1x A10 or similar).
# Runs sequentially: sheaf across 5 tasks, then hodge scaling for 4 filters.
#
# Total estimate: ~6-8 hours on 1x A10
#
# Usage:
#   ./scripts/run_sheaf_scaling.sh
#
# Self-terminates instance after completion (set SELF_TERMINATE=1).

set -e
cd ~/transformer-topology
source .venv/bin/activate

RESULTS="data/sheaf_scaling_results"
DATASETS="data/sheaf_scaling_datasets"
LOG="$RESULTS/experiment.log"
TOKEN_FILE="$HOME/.cache/lambda_token"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

mkdir -p "$RESULTS" "$DATASETS"

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" | tee -a "$LOG"

############################
# Part 1: Sheaf Stability Test (all 5 tasks)
############################
log "=========================================="
log "Part 1: Sheaf Stability Test"
log "  Config: benchmark_4b_sheaf.yaml (with stability fixes)"
log "  Tasks: bfs, diverse, hodge_class, propagation_delay, spectral_gap"
log "=========================================="

for TASK in bfs diverse hodge_class propagation_delay spectral_gap; do
    RESULT_FILE="$RESULTS/sheaf_${TASK}.json"
    if [ -f "$RESULT_FILE" ]; then
        log "  $TASK: already done, skipping"
        continue
    fi
    log "  Starting sheaf $TASK..."
    python3 -u scripts/run_single_benchmark.py \
        --config config/benchmark_4b_sheaf.yaml \
        --task "$TASK" \
        --variant hierarchical \
        --output "$RESULT_FILE" \
        2>&1 | tee -a "$RESULTS/sheaf_${TASK}.log"
    log "  $TASK complete -> $RESULT_FILE"
done

log "Part 1 complete."

############################
# Part 2: Hodge Scaling Experiment
############################
log ""
log "=========================================="
log "Part 2: Hodge Class Scaling"
log "  Filters: sheaf, schrodinger, wave_cosine, chebyshev"
log "  Train sizes: 20, 40, 80"
log "  Test sizes: 20, 40, 80, 160, 320"
log "=========================================="

# Generate datasets first (sequential)
log "Generating datasets..."
python3 -u << 'PYEOF'
from pathlib import Path
from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
import time

ds_dir = Path("data/sheaf_scaling_datasets")
ds_dir.mkdir(parents=True, exist_ok=True)
topologies = ["ba", "ws", "sbm", "er"]
emb_dim = 32

# Train + val for each training size
for n in [20, 40, 80]:
    for split, count in [("train", 2000), ("val", 500)]:
        name = f"hodge_scaling_{split}_{n}_n{n}.pt"
        path = ds_dir / name
        if path.exists():
            print(f"  {name} exists, skipping")
            continue
        t0 = time.time()
        print(f"  Generating {name} ({count} samples, n={n})...")
        ds = BenchmarkDataset(count, "hodge_class", n, emb_dim, topologies=topologies)
        ds.save(str(path))
        print(f"    Done in {time.time()-t0:.0f}s")

# Test datasets at all sizes
for n in [20, 40, 80, 160, 320]:
    name = f"hodge_scaling_test_{n}_n{n}.pt"
    path = ds_dir / name
    if path.exists():
        print(f"  {name} exists, skipping")
        continue
    t0 = time.time()
    print(f"  Generating {name} (500 samples, n={n})...")
    ds = BenchmarkDataset(500, "hodge_class", n, emb_dim, topologies=topologies)
    ds.save(str(path))
    print(f"    Done in {time.time()-t0:.0f}s")

print("Dataset generation complete.")
PYEOF

log "Datasets ready."

# Run hodge scaling jobs sequentially
FILTERS="schrodinger wave_cosine chebyshev sheaf"
TRAIN_SIZES="20 40 80"

for FILTER in $FILTERS; do
    for SIZE in $TRAIN_SIZES; do
        RESULT_FILE="$RESULTS/hodge_${FILTER}_n${SIZE}.json"
        if [ -f "$RESULT_FILE" ]; then
            log "  hodge ${FILTER} n=${SIZE}: already done, skipping"
            continue
        fi
        log "  Starting hodge scaling: filter=$FILTER, train_n=$SIZE..."

        if [ "$FILTER" = "sheaf" ]; then
            # Sheaf uses its own config (wave_mode=sheaf)
            CONFIG="config/benchmark_4b_sheaf.yaml"
        else
            CONFIG="config/benchmark_4b_schrodinger.yaml"
        fi

        python3 -u scripts/run_hodge_scaling.py \
            --filter "$FILTER" \
            --train-size "$SIZE" \
            --datasets-dir "$DATASETS" \
            --results-file "$RESULT_FILE" \
            --config "$CONFIG" \
            2>&1 | tee -a "$RESULTS/hodge_${FILTER}_n${SIZE}.log"
        log "  Done: $RESULT_FILE"
    done
done

############################
# Part 3: Compile Results
############################
log ""
log "=========================================="
log "Part 3: Compiling Results"
log "=========================================="

python3 -u << 'PYEOF'
import json, glob
from pathlib import Path

results_dir = Path("data/sheaf_scaling_results")

# Sheaf stability results
print("\n=== Sheaf Stability (all 5 tasks) ===")
print(f"{'Task':<20s} {'ID':>6s} {'n=40':>6s} {'n=80':>6s} {'Ret':>6s} {'Epochs':>7s} {'NaN?':>5s}")
print("-" * 60)
for task in ["bfs", "diverse", "hodge_class", "propagation_delay", "spectral_gap"]:
    f = results_dir / f"sheaf_{task}.json"
    if not f.exists():
        print(f"  {task:<20s} MISSING")
        continue
    d = json.load(open(f))
    r = d.get("result", d)
    id_acc = r["id_accuracy"]
    s40 = r.get("size_40_accuracy", 0)
    s80 = r.get("size_80_accuracy", 0)
    ret = f"{s80/id_acc:.0%}" if id_acc > 0 else "N/A"
    # Check for NaN
    has_nan = any(
        c.get("train_loss") != c.get("train_loss")
        for c in r.get("training_curve", [])
        if isinstance(c.get("train_loss"), float)
    )
    total_ep = r.get("total_epochs", "?")
    print(f"  {task:<20s} {id_acc:>5.1%} {s40:>5.1%} {s80:>5.1%} {ret:>5s} {total_ep:>6} {'YES' if has_nan else 'no':>5s}")

# Hodge scaling cross-size matrices
print("\n=== Hodge Scaling Cross-Size Matrices ===")
filters = ["schrodinger", "wave_cosine", "chebyshev", "sheaf"]
train_sizes = [20, 40, 80]
test_sizes = [20, 40, 80, 160, 320]

for ft in filters:
    print(f"\n  Filter: {ft}")
    header = f"  {'train\\test':<10s}"
    for tn in test_sizes:
        header += f"  n={tn:>3d}"
    print(header)
    print("  " + "-" * (10 + 7 * len(test_sizes)))
    for ts in train_sizes:
        f = results_dir / f"hodge_{ft}_n{ts}.json"
        row = f"  n={ts:<6d}"
        if f.exists():
            d = json.load(open(f))
            accs = d.get("test_accuracies", {})
            for tn in test_sizes:
                acc = accs.get(str(tn), 0)
                row += f"  {acc:>5.1%}"
        else:
            row += "  MISSING" * len(test_sizes)
        print(row)

# Save merged
output = {"experiment": "sheaf_scaling", "results": {}}
for f in sorted(results_dir.glob("*.json")):
    if f.name == "merged.json":
        continue
    d = json.load(open(f))
    output["results"][f.stem] = d
with open(results_dir / "merged.json", "w") as fh:
    json.dump(output, fh, indent=2, default=str)
print(f"\nMerged -> {results_dir}/merged.json")
PYEOF

echo "COMPLETE $(date)" > "$RESULTS/TRAINING_COMPLETE"
log "ALL DONE. Results in $RESULTS/"

############################
# Part 4: Self-terminate (if configured)
############################
if [ "${SELF_TERMINATE:-0}" = "1" ] && [ -f "$TOKEN_FILE" ]; then
    log "Waiting for SYNC_CONFIRMED or 2-hour timeout..."
    WAIT_START=$(date +%s)
    MAX_WAIT=7200
    while true; do
        ELAPSED=$(( $(date +%s) - WAIT_START ))
        if [ -f "$RESULTS/SYNC_CONFIRMED" ]; then
            log "SYNC_CONFIRMED received! Terminating in 60s..."
            sleep 60
            break
        fi
        if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
            log "WARNING: 2-hour timeout. Terminating anyway."
            break
        fi
        sleep 120
    done

    # Lambda instance ID stored during setup
    INSTANCE_ID=$(cat "$HOME/.cache/lambda_instance_id" 2>/dev/null || echo "unknown")
    TOKEN=$(cat "$TOKEN_FILE")
    log "Self-terminating..."
    curl -s -H "Authorization: Bearer $TOKEN" \
        -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
        -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}" \
        -H "Content-Type: application/json"
    log "Terminate request sent."
fi
