#!/bin/bash
# Backup DSM checkpoints from vast.ai to local machine.
# Run periodically during training to prevent checkpoint loss.
#
# Usage:
#   ./scripts/backup_checkpoints.sh
#   ./scripts/backup_checkpoints.sh --all   # include _latest.pt (large, has optimizer state)
#
# Backs up: _best.pt files + phase completion checkpoints (phase_a_*.pt etc.)
# Skips: _latest.pt (2.4 GB each with optimizer state) unless --all

set -e

REMOTE_HOST="root@76.66.207.49"
REMOTE_PORT=43668
SSH_KEY="$HOME/.ssh/vastai"
REMOTE_DIR="/root/transformer-topology/data/dsm_checkpoints/"
LOCAL_DIR="$(dirname "$0")/../data/dsm_checkpoints/"

mkdir -p "$LOCAL_DIR"

if [ "$1" = "--all" ]; then
    echo "Backing up ALL checkpoints (including _latest.pt)..."
    rsync -avz --progress \
        -e "ssh -i $SSH_KEY -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
        "$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR"
else
    echo "Backing up _best.pt and phase completion checkpoints..."
    rsync -avz --progress \
        --include='*_best.pt' \
        --include='phase_a_*.pt' --include='phase_b_*.pt' --include='phase_c_*.pt' \
        --exclude='*_latest.pt' \
        -e "ssh -i $SSH_KEY -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
        "$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR"
fi

echo ""
echo "=== Local checkpoint inventory ==="
ls -lh "$LOCAL_DIR"/*.pt 2>/dev/null || echo "No .pt files yet"
echo ""
echo "=== Remote checkpoint inventory ==="
ssh -i "$SSH_KEY" -p "$REMOTE_PORT" -o StrictHostKeyChecking=no "$REMOTE_HOST" \
    "ls -lh $REMOTE_DIR*.pt 2>/dev/null || echo 'No .pt files'"
