#!/bin/bash
# CLRS benchmark training on Lambda Labs GPU
# Generates full datasets + trains both architectures on BFS and Dijkstra
set -e

cd ~/transformer-topology
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== GPU Check ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Running CLRS Benchmark (full config) ==="
python3 -u -m src.benchmarks.run_clrs config/clrs.yaml

echo ""
echo "=== Done! ==="
