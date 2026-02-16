#!/bin/bash
# Single reliable orchestrator — polls result files, runs stages, self-terminates.
# Will NOT terminate until results are confirmed synced or 2-hour timeout.
set -e
cd ~/transformer-topology
source .venv/bin/activate

TOKEN_FILE="$HOME/.cache/lambda_token"
INSTANCE_ID="0cf36204c61f43fcb9e5dbc55edcc65c"
RESULTS="data/benchmark_4b_results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

############################
# Stage 1: Wait for filter jobs (20 results)
############################
log "Stage 1: Waiting for 20 filter results..."
while true; do
    COUNT=0
    for d in jobs_bandpass jobs_wave_cosine jobs_chebyshev jobs_wavelet; do
        n=$(ls "$RESULTS/$d"/result_*.json 2>/dev/null | wc -l)
        COUNT=$((COUNT + n))
    done
    log "  Filter results: $COUNT/20"
    if [ "$COUNT" -ge 20 ]; then
        log "All filter jobs complete!"
        break
    fi
    sleep 120
done

# Merge filter results
log "Merging filter results..."
for f in bandpass wave_cosine chebyshev wavelet; do
    DIR="$RESULTS/jobs_${f}"
    if ls "$DIR"/result_*.json 1>/dev/null 2>&1; then
        python3 scripts/merge_results.py "$DIR" --output "$RESULTS/results_${f}.json" 2>/dev/null || true
    fi
done

############################
# Stage 2: Run magnetic + sheaf (10 jobs)
############################
log "Stage 2: Launching magnetic + sheaf jobs..."
bash scripts/launch_4b_advanced.sh 2>&1 | tee "$RESULTS/run_advanced.log"
log "Advanced jobs complete!"

############################
# Stage 3: Hodge scaling experiment (parallel)
############################
log "Stage 3: Launching hodge scaling experiment (parallel across GPUs)..."
bash scripts/launch_4b_hodge_scaling.sh 2>&1 | tee "$RESULTS/hodge_scaling.log"
log "Hodge scaling complete!"

############################
# Stage 4: Wait for sync confirmation, then self-terminate
############################
log "ALL TRAINING COMPLETE"
echo "ALL_COMPLETE $(date)" > "$RESULTS/TRAINING_COMPLETE"
log "Created TRAINING_COMPLETE marker."
log ""
log "Waiting for SYNC_CONFIRMED marker or 2-hour timeout..."
log "To confirm sync, run on local machine:"
log "  ssh ubuntu@132.145.133.228 'touch ~/transformer-topology/data/benchmark_4b_results/SYNC_CONFIRMED'"

WAIT_START=$(date +%s)
MAX_WAIT=7200  # 2 hours

while true; do
    ELAPSED=$(( $(date +%s) - WAIT_START ))

    if [ -f "$RESULTS/SYNC_CONFIRMED" ]; then
        log "SYNC_CONFIRMED received! Terminating in 60s..."
        sleep 60
        break
    fi

    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        log "WARNING: 2-hour timeout reached without SYNC_CONFIRMED."
        log "Terminating anyway to avoid cost overrun."
        break
    fi

    REMAINING=$(( (MAX_WAIT - ELAPSED) / 60 ))
    if [ $((ELAPSED % 600)) -lt 120 ]; then
        log "  Waiting for sync... ($REMAINING min until timeout)"
    fi

    sleep 120
done

# Self-terminate
log "Self-terminating instance $INSTANCE_ID..."
if [ -f "$TOKEN_FILE" ]; then
    TOKEN=$(cat "$TOKEN_FILE")
    curl -s -H "Authorization: Bearer $TOKEN" \
        -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
        -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}"
    log "Terminate request sent."
else
    log "ERROR: No token file — cannot self-terminate!"
fi
