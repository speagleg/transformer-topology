"""Quick diagnostic: check metacog gate values on validation samples."""
import torch
import yaml
from pathlib import Path

# Load config
with open("config/v7_metacognition.yaml") as f:
    config = yaml.safe_load(f)

mc = config["model"]
wc = config.get("wave")
lc = config.get("llm")

from src.benchmarks.run_benchmark_suite import _build_model, _resolve_device
from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
from src.benchmarks.run_comparison import _unpack_sample

device = _resolve_device(config["training"])

# Build model and load latest checkpoint
model = _build_model("hierarchical_llm", mc, get_max_classes("kg_relation"), device,
                      wave_config=wc, llm_config=lc)

from scripts.run_dsm_curriculum import _load_state_filtered

ckpt_path = "data/v6_track2/checkpoints/phase_d_kg_relation_best.pt"
if Path(ckpt_path).exists():
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    _load_state_filtered(model, state)
    print("Loaded: " + ckpt_path)

# Load text embedding cache
cache_path = "data/dsm_datasets_25k/text_embedding_cache.pt"
if Path(cache_path).exists() and hasattr(model, "topo_bridge") and model.topo_bridge is not None:
    model.topo_bridge.text_extractor.load_cache(cache_path)
    model.topo_bridge.text_extractor.build_gpu_cache(device)

model.eval()

# Load a few val samples
val_path = "data/dsm_datasets_25k/kg_relation_val_n16-32.pt"
val_ds = BenchmarkDataset.load(val_path)

# Run forward on 50 samples and collect gate values
text_gates = []
structure_gates = []
uncertainties = []
strategy_weights_list = []
semantic_weights = []
predictions = []
answers_list = []

SEP = "=" * 60

with torch.no_grad():
    for i in range(min(50, len(val_ds))):
        cc, query, target, answer, metadata = _unpack_sample(val_ds[i])
        cc = cc.clone().to(device)
        logits = model(cc, query, target, metadata=metadata, task="kg_relation")

        pred = logits.argmax().item()
        predictions.append(pred)
        answers_list.append(answer)

        ctrl = getattr(model, "_last_control", None)
        if ctrl is not None:
            if ctrl.text_gate is not None:
                text_gates.append(ctrl.text_gate.item())
            if ctrl.structure_gate is not None:
                structure_gates.append(ctrl.structure_gate.item())
            if ctrl.uncertainty is not None:
                uncertainties.append(ctrl.uncertainty.item())
            if ctrl.strategy_weights is not None:
                strategy_weights_list.append(ctrl.strategy_weights.detach().cpu())
            if ctrl.semantic_weight is not None:
                semantic_weights.append(ctrl.semantic_weight.item())

print()
print(SEP)
print("Gate Diagnostics (n=%d samples)" % len(text_gates))
print(SEP)

if text_gates:
    tg = torch.tensor(text_gates)
    print("text_gate:      mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (
        tg.mean(), tg.std(), tg.min(), tg.max()))

if structure_gates:
    sg = torch.tensor(structure_gates)
    print("structure_gate: mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (
        sg.mean(), sg.std(), sg.min(), sg.max()))

if uncertainties:
    uc = torch.tensor(uncertainties)
    print("uncertainty:    mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (
        uc.mean(), uc.std(), uc.min(), uc.max()))

if semantic_weights:
    sw = torch.tensor(semantic_weights)
    print("semantic_wt:    mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (
        sw.mean(), sw.std(), sw.min(), sw.max()))

if strategy_weights_list:
    sw_stack = torch.stack(strategy_weights_list)
    print("strategy_weights (4 dims):")
    for d in range(sw_stack.shape[1]):
        col = sw_stack[:, d]
        print("  dim[%d]:       mean=%.4f  std=%.4f" % (d, col.mean(), col.std()))

# Check gate values for correct vs wrong predictions
correct_tg = []
wrong_tg = []
correct_sg = []
wrong_sg = []

for i in range(len(predictions)):
    if i < len(text_gates):
        if predictions[i] == answers_list[i]:
            correct_tg.append(text_gates[i])
            if i < len(structure_gates):
                correct_sg.append(structure_gates[i])
        else:
            wrong_tg.append(text_gates[i])
            if i < len(structure_gates):
                wrong_sg.append(structure_gates[i])

print()
print("Gate values by correctness:")
if correct_tg:
    ct = torch.tensor(correct_tg)
    cs = torch.tensor(correct_sg) if correct_sg else torch.tensor([0.0])
    print("  Correct (n=%d): text_gate=%.4f  structure_gate=%.4f" % (
        len(correct_tg), ct.mean(), cs.mean()))
if wrong_tg:
    wt = torch.tensor(wrong_tg)
    ws = torch.tensor(wrong_sg) if wrong_sg else torch.tensor([0.0])
    print("  Wrong   (n=%d): text_gate=%.4f  structure_gate=%.4f" % (
        len(wrong_tg), wt.mean(), ws.mean()))

# Class distribution of predictions vs answers
from collections import Counter
pred_dist = Counter(predictions)
ans_dist = Counter(answers_list)
print()
print("Prediction distribution: %s" % dict(sorted(pred_dist.items())))
print("Answer distribution:     %s" % dict(sorted(ans_dist.items())))
