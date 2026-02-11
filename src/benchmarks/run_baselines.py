"""Baseline comparison: train and evaluate all models on the same dataset.

Generates a fixed-seed dataset, trains the main model (TopologyAwareTransformer
+ GNN Executive + ReasoningLoop) alongside three baselines (Vanilla Transformer,
GAT, GCN), and prints a comparison table with per-hop accuracy breakdown.

Usage:
    source .venv/bin/activate
    python -m src.benchmarks.run_baselines
"""

import random
import time
import torch
import torch.nn as nn

from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.baselines import (
    VanillaTransformerBaseline,
    GATBaseline,
    GCNBaseline,
)
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.trainer import train_epoch, evaluate


# ---------------------------------------------------------------------------
# Configuration — matches config/quick_test.yaml
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
MAX_HOPS = 10
MIN_HOPS = 2
NUM_DISTRACTORS = 20

NUM_TRAIN = 500
NUM_VAL = 100
NUM_TEST = 200

LR = 0.001
MAX_EPOCHS = 20
PATIENCE = 7
SEED = 42

# Main model config (from quick_test.yaml)
MAIN_MODEL_CONFIG = dict(
    embedding_dim=EMBEDDING_DIM,
    gnn_hidden=EMBEDDING_DIM * 2,
    gnn_spatial_layers=NUM_LAYERS,
    gnn_spectral_layers=NUM_LAYERS,
    max_freqs=EMBEDDING_DIM // 2,
    tat_layers=NUM_LAYERS,
    tat_spatial_heads=NUM_HEADS,
    tat_spectral_heads=NUM_HEADS,
    tat_ff_dim=256,
    max_hops=MAX_HOPS,
    max_iterations=3,
)

# Per-hop evaluation points (even hops from 2 to 10)
EVAL_HOPS = [2, 4, 6, 8, 10]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_datasets():
    """Generate train/val/test datasets with fixed seed."""
    set_seed(SEED)
    train_ds = MultiHopDataset(
        num_samples=NUM_TRAIN, min_hops=MIN_HOPS, max_hops=MAX_HOPS,
        num_distractors=NUM_DISTRACTORS, embedding_dim=EMBEDDING_DIM,
    )
    val_ds = MultiHopDataset(
        num_samples=NUM_VAL, min_hops=MIN_HOPS, max_hops=MAX_HOPS,
        num_distractors=NUM_DISTRACTORS, embedding_dim=EMBEDDING_DIM,
    )
    test_ds = MultiHopDataset(
        num_samples=NUM_TEST, min_hops=MIN_HOPS, max_hops=MAX_HOPS,
        num_distractors=NUM_DISTRACTORS, embedding_dim=EMBEDDING_DIM,
    )
    return train_ds, val_ds, test_ds


def generate_per_hop_datasets():
    """Generate per-hop evaluation datasets with fresh fixed seed."""
    datasets = {}
    for h in EVAL_HOPS:
        set_seed(SEED + h)  # deterministic but distinct per hop
        datasets[h] = MultiHopDataset(
            num_samples=100, min_hops=h, max_hops=h,
            num_distractors=NUM_DISTRACTORS, embedding_dim=EMBEDDING_DIM,
        )
    return datasets


def train_model(name: str, model: nn.Module, train_ds, val_ds) -> nn.Module:
    """Train a model with early stopping, return best weights loaded."""
    print(f"\n{'─' * 60}")
    print(f"Training: {name}  ({format_params(count_params(model))} params)")
    print(f"{'─' * 60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        start = time.time()
        train_loss = train_epoch(model, train_ds, optimizer)
        val_acc, val_loss = evaluate(model, val_ds)
        elapsed = time.time() - start

        scheduler.step(val_loss)

        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_ds, per_hop_datasets):
    """Evaluate on test set and per-hop datasets. Returns (test_acc, hop_accs)."""
    test_acc, _ = evaluate(model, test_ds)
    hop_accs = {}
    for h, ds in per_hop_datasets.items():
        acc, _ = evaluate(model, ds)
        hop_accs[h] = acc
    return test_acc, hop_accs


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    # Header
    hop_cols = "".join(f" | {h:>5}-hop" for h in EVAL_HOPS)
    header = f"{'Model':<22} | {'Params':>7} | {'Test Acc':>8}{hop_cols}"
    sep = "─" * len(header)

    print(f"\n{'=' * len(header)}")
    print("BASELINE COMPARISON RESULTS")
    print(f"{'=' * len(header)}")
    print(f"Dataset: {NUM_TRAIN} train / {NUM_VAL} val / {NUM_TEST} test | "
          f"Hops {MIN_HOPS}-{MAX_HOPS} | {NUM_DISTRACTORS} distractors | seed={SEED}")
    print(f"Training: lr={LR}, {MAX_EPOCHS} epochs, patience={PATIENCE}")
    print(f"Random chance: ~{1/(MAX_HOPS - MIN_HOPS + 1):.3f} ({MAX_HOPS - MIN_HOPS + 1} classes)")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        hop_vals = "".join(f" | {r['hop_accs'].get(h, 0):>8.3f}" for h in EVAL_HOPS)
        print(f"{r['name']:<22} | {format_params(r['params']):>7} | "
              f"{r['test_acc']:>8.3f}{hop_vals}")

    print(sep)


def main():
    print("=" * 60)
    print("Baseline Comparison: Multi-Hop Graph Traversal")
    print("=" * 60)

    # Generate datasets (same seed for all models)
    print("\nGenerating datasets with seed=42...")
    train_ds, val_ds, test_ds = generate_datasets()
    per_hop_datasets = generate_per_hop_datasets()
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Build all models — baselines are sized to roughly match the main model's
    # ~2.9M parameter count.  GNN architectures are inherently more parameter-
    # efficient per layer, so they need wider hidden dims / more layers.
    models = {
        "TT (ours)": MultiHopReasoningModel(**MAIN_MODEL_CONFIG),
        "Vanilla Transformer": VanillaTransformerBaseline(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=240,        # 2.96M params with 4 layers
            num_layers=4,
            num_heads=4,
            max_hops=MAX_HOPS,
        ),
        "GAT": GATBaseline(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=512,        # 3.27M params with 12 layers, 8 heads
            num_layers=12,
            num_heads=8,
            max_hops=MAX_HOPS,
        ),
        "GCN": GCNBaseline(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=512,        # 2.73M params with 10 layers
            num_layers=10,
            max_hops=MAX_HOPS,
        ),
    }

    # Print param counts
    print("\nModel parameter counts:")
    for name, model in models.items():
        print(f"  {name:<22}: {count_params(model):>10,} ({format_params(count_params(model))})")

    # Train and evaluate each model
    results = []
    for name, model in models.items():
        # Reset random seed so each model sees data in same order
        set_seed(SEED)
        model = train_model(name, model, train_ds, val_ds)
        test_acc, hop_accs = evaluate_model(model, test_ds, per_hop_datasets)
        print(f"  >> Test Accuracy: {test_acc:.3f}")
        results.append({
            "name": name,
            "params": count_params(model),
            "test_acc": test_acc,
            "hop_accs": hop_accs,
        })

    # Print comparison table
    print_comparison_table(results)


if __name__ == "__main__":
    main()
