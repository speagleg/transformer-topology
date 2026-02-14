#!/bin/bash
# Pull benchmark results from Lambda Labs instance.
#
# Usage:
#   ./scripts/pull_lambda_results.sh [IP_ADDRESS]
#
# Pulls logs, job files, configs, and result JSONs (not datasets).

set -euo pipefail

LAMBDA_KEY="$HOME/.ssh/lambda_gpu"
LAMBDA_USER="ubuntu"
LAMBDA_IP="${1:-147.224.195.184}"
REMOTE_DIR="transformer-topology"
LOCAL_BASE="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DATA="$LOCAL_BASE/data/benchmark_jobs"

echo "Pulling from $LAMBDA_USER@$LAMBDA_IP:~/$REMOTE_DIR/"
echo "Local destination: $LOCAL_DATA/"
echo ""

# Ensure local dirs exist
mkdir -p "$LOCAL_DATA/results"

# Pull log directories (the main data)
for log_dir in logs logs_round2 logs_hodge_fix logs_hodge_quick; do
    echo "--- Syncing $log_dir ---"
    rsync -avz --progress \
        -e "ssh -i $LAMBDA_KEY -o StrictHostKeyChecking=no" \
        "$LAMBDA_USER@$LAMBDA_IP:~/$REMOTE_DIR/data/benchmark_jobs/$log_dir/" \
        "$LOCAL_DATA/$log_dir/" 2>/dev/null || echo "  (not found, skipping)"
done

# Pull job list files and launcher logs
echo "--- Syncing job files ---"
rsync -avz --progress \
    -e "ssh -i $LAMBDA_KEY -o StrictHostKeyChecking=no" \
    --include='jobs*.txt' --include='launcher*.log' --exclude='*' \
    "$LAMBDA_USER@$LAMBDA_IP:~/$REMOTE_DIR/data/benchmark_jobs/" \
    "$LOCAL_DATA/" 2>/dev/null

# Pull result JSONs (if any have been written)
echo "--- Syncing result JSONs ---"
rsync -avz --progress \
    -e "ssh -i $LAMBDA_KEY -o StrictHostKeyChecking=no" \
    "$LAMBDA_USER@$LAMBDA_IP:~/$REMOTE_DIR/data/benchmark_jobs/results/" \
    "$LOCAL_DATA/results/" 2>/dev/null || echo "  (empty or not found)"

# Pull Lambda-only configs
echo "--- Syncing configs ---"
rsync -avz --progress \
    -e "ssh -i $LAMBDA_KEY -o StrictHostKeyChecking=no" \
    --include='benchmark*.yaml' --exclude='*' \
    "$LAMBDA_USER@$LAMBDA_IP:~/$REMOTE_DIR/config/" \
    "$LOCAL_BASE/config/" 2>/dev/null

echo ""
echo "Done. Run analysis with:"
echo "  python scripts/analyze_benchmark.py data/benchmark_jobs"
