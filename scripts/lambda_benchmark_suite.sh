#!/bin/bash
# Comprehensive benchmark suite on Lambda Labs GPU
# 5 tier-1 tasks × 3 model variants × 3 eval axes
# ~5000 train / 1000 val / 2000 test per task, 50 epochs
set -e

cd ~/transformer-topology
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== GPU Check ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Running Benchmark Suite (Tier 1) ==="
python3 -u -m src.benchmarks.run_benchmark_suite config/benchmark_suite.yaml

echo ""
echo "=== Results ==="
cat data/benchmark_results/results.json
echo ""
echo "=== Done! ==="
