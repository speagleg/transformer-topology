# GPU Training Reference: Transformer-Topology + Llama 3.2 1B

Quick reference for deploying and running curriculum training on cloud GPUs.

## Instance Specs

| Provider | GPU | VRAM | Disk | Cost |
|----------|-----|------|------|------|
| vast.ai | RTX 4090 | 24 GB | 83 GB overlay | ~$0.30/hr |
| Lambda Labs | A100 SXM4 | 40 GB | varies | ~$1.10/hr |

### vast.ai Connection
```bash
ssh -i ~/.ssh/vastai -p <PORT> root@<IP>
# Current instance: ssh -i ~/.ssh/vastai -p 43668 root@76.66.207.49
```

### Lambda Labs Connection
```bash
ssh -i ~/.ssh/lambda_gpu ubuntu@<IP>
```

---

## Model Memory Budget (RTX 4090, 24 GB)

| Component | VRAM |
|-----------|------|
| Llama 3.2 1B (bf16, frozen) | ~2.5 GB |
| GNN Executive + TAT + Wave Ensemble | ~0.2 GB |
| TopoBridge (projector + prefix + extractor) | ~0.05 GB |
| LoRA adapters (rank 16) | ~0.05 GB |
| **Total model** | **~2.8 GB** |
| Forward pass activations | +0.2 GB |
| Backward pass (with gradient checkpointing) | +0.3 GB |
| Optimizer state (AdamW, ~50M trainable params) | ~0.4 GB |
| **Peak training** | **~4-6.5 GB** |

Headroom: ~17 GB free for batch data and PyTorch allocator overhead.

### Critical: Trainable Param Count

- **Total params**: 1,286,633,073 (1.287B)
- **Trainable params**: 50,818,673 (50.8M) — LoRA + TopoBridge + cross-attn
- **Frozen**: 1,235,814,400 — Llama base weights

If optimizer uses >1 GB, something is wrong — check `_unfreeze_llm`.

---

## Setup on New Instance

```bash
cd ~/transformer-topology
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torch-geometric gudhi torchdiffeq networkx pyyaml
pip install transformers peft sentencepiece   # for Llama

# Sync from local
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
  -e 'ssh -i ~/.ssh/vastai -p <PORT>' \
  ./ root@<IP>:~/transformer-topology/
```

### Lambda-specific
```bash
# Lambda has Python 3.10 — skip editable install, direct imports work
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torch-geometric gudhi torchdiffeq networkx pyyaml
```

---

## Launch Training

```bash
cd ~/transformer-topology
source .venv/bin/activate
export PYTHONPATH=/root/transformer-topology
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Full curriculum (Phase A → B → C)
nohup python -u scripts/run_phase4c_curriculum.py \
    config/llama_training.yaml \
    --pregenerated-dir data/llama_datasets \
    > data/llama_checkpoints/training.log 2>&1 &

# Resume from Phase B or C
nohup python -u scripts/run_phase4c_curriculum.py \
    config/llama_training.yaml \
    --resume-phase c \
    --pregenerated-dir data/llama_datasets \
    > data/llama_checkpoints/training.log 2>&1 &
```

### Monitor
```bash
tail -f data/llama_checkpoints/training.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ps aux | grep run_phase4c | grep -v grep
```

---

## Curriculum Phases

### Phase A: Regression Lock
- **Purpose**: Train GNN/TAT core without LLM interference
- **TopoBridge**: on CPU (`bypass_llm=True`)
- **LR**: 0.001 (full)
- **Gate penalty**: 1.0 (penalizes llm_gate opening)
- **Tasks**: diverse, bfs, hodge_class, spectral_gap

### Phase B: Graph Completion
- **Purpose**: Introduce LLM pathway on graph-structural tasks
- **TopoBridge**: on GPU, unfrozen
- **LR**: 0.0005 (half), LLM params at 0.5x scale
- **Gate penalty**: 0.0
- **Tasks**: graph_completion, path_counting

### Phase C: Language + Analogy
- **Purpose**: Full fine-tuning on language-grounded tasks
- **LR**: 0.00025 (quarter), LLM params at 1.0x scale
- **Tasks**: labeled_reasoning, analogical_transfer, graph_completion

### Resume Behavior
`--resume-phase c` loads `phase_a_{last_task}.pt` from `checkpoint_dir`, then skips A and B.

**Gotcha**: It only loads the Phase A checkpoint. If you need Phase B/C weights, manually copy the desired checkpoint over `phase_a_spectral_gap.pt`:
```bash
cp data/llama_checkpoints/phase_c_labeled_reasoning_best.pt \
   data/llama_checkpoints/phase_a_spectral_gap.pt
```

---

## Training Results (Feb 2026, vast.ai RTX 4090)

### Phase A (bypass_llm=True, 1000 train, n=16-48)

| Task | Classes | Best Val | Epochs | Time/Epoch |
|------|---------|----------|--------|------------|
| diverse | 11 | 71.0% | 30 (patience) | ~830s |
| bfs | 16 | 100% | 9 (perfect) | ~1550s |
| hodge_class | 3 | 82.0% | 25 (patience) | ~1050s |
| spectral_gap | 8 | 73.0% | 17 (patience) | ~1220s |

### Phase B (TopoBridge on GPU, LLM unfrozen)

| Task | Classes | Best Val | Epochs | Time/Epoch |
|------|---------|----------|--------|------------|
| graph_completion | 2 | 80.5% | 18 (patience) | ~1070s |
| path_counting | 5 | 98.0% | 21 (patience) | ~1100s |

### Phase C (fine-tuning LR)

| Task | Classes | Best Val | Epochs | Status |
|------|---------|----------|--------|--------|
| labeled_reasoning | 3 | 76.5% | 20 (patience) | Done |
| analogical_transfer | 6 | 18.5% | — | Killed (near random) |
| graph_completion | 2 | in progress | — | Running |

### Epoch Timing by Task Complexity
- **Simple tasks** (diverse, hodge): ~800-1050s/epoch
- **BFS** (16 classes, larger graphs): ~1550s/epoch
- **Spectral gap** (8 classes): ~1220s/epoch
- **LLM tasks** (TopoBridge active): ~1050-1150s/epoch

---

## Pregenerated Datasets

```bash
# Generate all datasets (run on CPU, ~10 min)
python scripts/generate_llama_datasets.py config/llama_training.yaml

# Generate specific tasks
python scripts/generate_llama_datasets.py config/llama_training.yaml --tasks diverse bfs
```

### File naming convention
```
data/llama_datasets/
  {task}_train_n16-48.pt    # 5000 samples, mixed sizes, all topologies
  {task}_val_n16-48.pt      # 500 samples
  {task}_test_n20.pt        # 500 samples, fixed size (ID test)
  {task}_test_n48.pt        # 500 samples (size OOD)
  {task}_test_n80.pt        # 500 samples (large OOD)
  {task}_test_topo_n20.pt   # 500 samples, held-out topologies
  manifest.json             # stats, class distributions
```

### Topology Split
- **Train**: ba, ws, sbm, er
- **Test (transfer)**: grid, tree, ladder, caveman

---

## Checkpoint Management

### File sizes
- `_best.pt` (model state_dict only): ~2.5 GB
- `_latest.pt` (model + optimizer + epoch): ~2.5 GB with correct `_unfreeze_llm`, was **7.5 GB** with the bug

### Checkpoint naming
```
data/llama_checkpoints/
  phase_a_{task}.pt              # saved after each Phase A task
  phase_b_{task}_best.pt         # best val during Phase B
  phase_b_{task}_latest.pt       # rolling (overwritten each epoch)
  phase_c_{task}_best.pt         # best val during Phase C
  phase_c_{task}_latest.pt       # rolling
```

### Backup to local
```bash
scp -i ~/.ssh/vastai -P <PORT> root@<IP>:~/transformer-topology/data/llama_checkpoints/phase_c_*_best.pt \
    data/llama_checkpoints/
```

### Disk management
Rolling `_latest.pt` files accumulate. Clean periodically:
```bash
# Keep only _best.pt files
rm data/llama_checkpoints/*_latest.pt
df -h /
```

---

## Bugs & Fixes (Lessons Learned)

### 1. `_unfreeze_llm` unfreezing ALL Llama params
**Symptom**: CUDA OOM during optimizer.step(), ~13 GB VRAM used.
**Root cause**: `if 'topo_bridge' in name or 'llm' in name` matched frozen Llama base weights nested under topo_bridge.
**Fix**: Precise name matching:
```python
def _unfreeze_llm(model):
    for name, param in model.named_parameters():
        if 'lora' in name or 'topo_cross_attn' in name:
            param.requires_grad = True
        elif 'topo_bridge' in name and 'llama' not in name:
            param.requires_grad = True
```
**Verify**: `sum(p.numel() for p in model.parameters() if p.requires_grad)` should be ~50M, not ~1.3B.

### 2. Disk full causing silent kills
**Symptom**: Process dies with no Python traceback, `dmesg` shows OOM killer.
**Root cause**: 83 GB overlay filesystem filled by 67 GB of checkpoints (7.5 GB each from bug #1).
**Fix**: Monitor disk with `df -h /`. Clean old `_latest.pt` files.

### 3. `torch_dtype` deprecation
**Symptom**: Warning from unsloth/Llama loader.
**Fix**: Change `torch_dtype=torch.bfloat16` to `dtype=torch.bfloat16` in `llama_backend.py`.

### 4. bf16 AMP for memory savings
**What**: `torch.amp.autocast('cuda', dtype=torch.bfloat16)` reduces backward pass memory ~50%.
**Safe**: Spectral ops (eigh, svd, lstsq) stay fp32 automatically. GradScaler not needed for bf16.
**Config**: `use_amp = device.type == 'cuda'` in curriculum script.

### 5. PYTHONPATH required on vast.ai
**Symptom**: `ModuleNotFoundError: No module named 'src'`
**Fix**: `export PYTHONPATH=/root/transformer-topology` before running scripts.

### 6. torchdiffeq diffusion_time=0 assertion
**Symptom**: `AssertionError` in wave propagation.
**Fix**: `diffusion_time.clamp(min=1e-4)` in WavePropagation + SheafWaveDynamics.

### 7. Multiple competing GPU processes
**Symptom**: GPU memory shows >10 GB before training starts.
**Fix**: Always `pkill -f run_phase4c` before launching. Check with `nvidia-smi`.

---

## Config Reference

### Key differences: Mock vs Llama

| Setting | Mock (`benchmark_4c_llm.yaml`) | Llama (`llama_training.yaml`) |
|---------|-------------------------------|-------------------------------|
| `llm.backend` | (default=mock) | `llama` |
| `llm.llm_dim` | 128 | 2048 |
| `llm.model_name` | — | `unsloth/Llama-3.2-1B` |
| `llm.lora_rank` | — | 16 |
| `llm.num_prefix` | 4 | 8 |
| `benchmark.num_train` | 500 | 1000-2000 |
| `benchmark.n_nodes_range` | — | 16-48 |
| `training.accumulation_steps` | 4 | 4 |
| `training.patience` | 7 | 10 |

### Wave Ensemble Config
```yaml
wave:
  wave_mode: ensemble
  filter_types: [chebyshev, wave_cosine, heat]
  include_identity: true
  include_sheaf: true       # 5 filters total
  wave_strength_gate: true
  use_neural_ode: false
```

---

## Quick Troubleshooting

| Problem | Check | Fix |
|---------|-------|-----|
| OOM at optimizer.step() | `nvidia-smi` shows >10 GB | Fix `_unfreeze_llm`, verify 50M trainable |
| Silent process death | `df -h /` shows >90% disk | Clean old checkpoints |
| No module 'src' | `echo $PYTHONPATH` | `export PYTHONPATH=~/transformer-topology` |
| Process stuck, no output | `ps -o etime -p <PID>` | First epoch takes ~17 min, wait |
| GPU 0% but process runs | `top` shows 100% CPU | Model loading or data prep phase |
| Low val accuracy on resume | Check which checkpoint loaded | Copy correct `_best.pt` to `phase_a_spectral_gap.pt` |
| NaN gradient norms | Occasional, self-correcting | Monitor; if persistent, reduce LR |
