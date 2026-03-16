"""Inverse-scaling experiment: train hodge_class model at small sizes (n=16-32).

Trains a HierarchicalMultiHopModel on hodge_class (3 classes) with mixed-size
training, saving checkpoints and per-epoch metrics as JSON.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/train.py \
        scripts/experiments/inverse_scaling/configs/baseline.yaml
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
from src.benchmarks.run_benchmark_suite import _build_model
from src.benchmarks.run_comparison import _unpack_sample


CLASS_NAMES = ["gradient", "curl", "harmonic"]


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataset, optimizer, tc, device):
    """Train one epoch, returns (avg_loss, grad_norm)."""
    model.train()
    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()

    total_loss = 0.0
    n_samples = 0
    grad_norms = []

    batch_size = tc.get("batch_size", 8)
    max_norm = tc.get("max_norm", 5.0)
    label_smoothing = tc.get("label_smoothing", 0.1)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    for batch_indices in batches:
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_count = 0

        for idx in batch_indices:
            cc, query, target, answer, metadata = _unpack_sample(dataset[idx])
            cc = cc.clone().to(device)
            logits = model(cc, query, target, metadata=metadata,
                           topo_features=None, task="hodge_class")
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([answer], device=device),
                label_smoothing=label_smoothing,
            )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                batch_loss = batch_loss + loss
                valid_count += 1

        if valid_count > 0:
            batch_loss = batch_loss / valid_count
            batch_loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if torch.isfinite(gn):
                optimizer.step()
                grad_norms.append(gn.item())

            total_loss += batch_loss.item() * valid_count
            n_samples += valid_count

    avg_loss = total_loss / max(n_samples, 1)
    avg_gn = sum(grad_norms) / max(len(grad_norms), 1)
    return avg_loss, avg_gn


@torch.no_grad()
def evaluate_with_per_class(model, dataset, device, label_smoothing=0.1):
    """Evaluate model, returning accuracy, loss, balanced acc, and per-class breakdown.

    Returns:
        dict with keys: accuracy, avg_loss, balanced_accuracy, per_class
        where per_class is {class_id: {correct, total, accuracy}}
    """
    model.eval()
    correct = 0
    total_loss = 0.0
    n_valid = 0
    n_total = 0

    class_correct = {}
    class_total = {}

    for i in range(len(dataset)):
        cc, query, target, answer, metadata = _unpack_sample(dataset[i])
        cc = cc.clone().to(device)
        logits = model(cc, query, target, metadata=metadata,
                       topo_features=None, task="hodge_class")

        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([answer], device=device),
            label_smoothing=label_smoothing,
        )
        loss_val = loss.item()
        if loss_val == loss_val and loss_val != float('inf'):
            total_loss += loss_val
            n_valid += 1

        pred = logits.argmax().item()
        if pred == answer:
            correct += 1
            class_correct[answer] = class_correct.get(answer, 0) + 1
        class_total[answer] = class_total.get(answer, 0) + 1
        n_total += 1

    accuracy = correct / max(n_total, 1)
    avg_loss = total_loss / max(n_valid, 1)

    per_class_accs = []
    per_class_info = {}
    for c in sorted(class_total.keys()):
        c_correct = class_correct.get(c, 0)
        c_total = class_total[c]
        c_acc = c_correct / c_total if c_total > 0 else 0.0
        per_class_accs.append(c_acc)
        per_class_info[c] = {
            "name": CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}",
            "correct": c_correct,
            "total": c_total,
            "accuracy": round(c_acc, 4),
        }

    balanced_accuracy = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0.0

    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class_info,
    }


def train_single_seed(config, seed, device):
    """Train a model for one seed, return results dict."""
    # Seed everything
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    exp = config["experiment"]
    mc = config["model"]
    wc = config.get("wave")
    tc = config["training"]
    out = config["output"]

    task = exp["task"]
    max_classes = get_max_classes(task)
    epochs = exp["epochs"]
    patience = exp["patience"]

    # Build model
    model = _build_model(
        variant="hierarchical",
        mc=mc,
        max_classes=max_classes,
        device=device,
        wave_config=wc,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params:,}")

    # Generate datasets with mixed-size training
    n_min = tc["train_n_min"]
    n_max = tc["train_n_max"]
    emb_dim = mc["embedding_dim"]

    print(f"  Generating train dataset ({tc['train_samples']} samples, n={n_min}-{n_max})...")
    train_ds = BenchmarkDataset(
        num_samples=tc["train_samples"],
        task_type=task,
        n_nodes=n_min,
        embedding_dim=emb_dim,
        n_nodes_range=(n_min, n_max),
    )

    print(f"  Generating val dataset ({tc['val_samples']} samples, n={n_min}-{n_max})...")
    val_ds = BenchmarkDataset(
        num_samples=tc["val_samples"],
        task_type=task,
        n_nodes=n_min,
        embedding_dim=emb_dim,
        n_nodes_range=(n_min, n_max),
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5,
    )

    # Checkpoint path
    ckpt_dir = Path(out["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{exp['name']}_seed{seed}_best.pt"

    # Training loop
    best_val_acc = 0.0
    best_train_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    training_curve = []

    print(f"  Training for up to {epochs} epochs (patience={patience})...")
    for epoch in range(epochs):
        t0 = time.time()

        train_loss, grad_norm = train_epoch(model, train_ds, optimizer, tc, device)
        val_info = evaluate_with_per_class(model, val_ds, device,
                                           label_smoothing=tc.get("label_smoothing", 0.1))
        elapsed = time.time() - t0

        val_acc = val_info["accuracy"]
        val_loss = val_info["avg_loss"]
        bal_acc = val_info["balanced_accuracy"]
        lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
            "val_bal_acc": round(bal_acc, 4),
            "grad_norm": round(grad_norm, 4),
            "lr": lr,
            "time": round(elapsed, 1),
            "per_class": val_info["per_class"],
        }
        training_curve.append(epoch_record)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "val_acc": best_val_acc,
                "val_bal_acc": bal_acc,
                "training_curve": training_curve,
                "config": config,
                "seed": seed,
            }, ckpt_path)
        else:
            patience_counter += 1

        # Per-class summary
        pc = val_info["per_class"]
        pc_str = " | ".join(
            f"{pc[c]['name'][:4]}={pc[c]['accuracy']:.0%}"
            for c in sorted(pc.keys())
        )

        print(f"    Ep {epoch:3d} | loss {train_loss:.4f} | "
              f"val {val_acc:.3f} bal {bal_acc:.3f} ({val_loss:.4f}) | "
              f"{pc_str} | gn {grad_norm:.2f} | lr {lr:.1e} | {elapsed:.0f}s{marker}",
              flush=True)

        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    result = {
        "seed": seed,
        "best_val_acc": round(best_val_acc, 4),
        "best_val_bal_acc": round(training_curve[best_epoch]["val_bal_acc"], 4),
        "best_train_loss": round(best_train_loss, 5),
        "best_epoch": best_epoch,
        "total_epochs": len(training_curve),
        "params": params,
        "checkpoint_path": str(ckpt_path),
        "training_curve": training_curve,
    }

    return result


def main(config_path: str = None):
    """Main entry point. Can be called with a config path or uses argparse."""
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Train hodge_class model for inverse scaling experiment"
        )
        parser.add_argument("config", help="Path to YAML config file")
        parser.add_argument("--seed", type=int, default=None,
                            help="Run only this seed (default: all seeds in config)")
        args = parser.parse_args()
        config_path = args.config
        single_seed = args.seed
    else:
        single_seed = None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = _resolve_device()
    exp = config["experiment"]
    out = config["output"]

    seeds = exp["seeds"]
    if single_seed is not None:
        seeds = [single_seed]

    print("=" * 72)
    print(f"Inverse Scaling Experiment: {exp['name']}")
    print(f"Task: {exp['task']}, Seeds: {seeds}, Device: {device}")
    print("=" * 72)

    all_results = []
    for seed in seeds:
        print(f"\n{'─' * 72}")
        print(f"Seed: {seed}")
        print(f"{'─' * 72}")

        result = train_single_seed(config, seed, device)
        all_results.append(result)

        # Save incremental results
        results_dir = Path(out["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"{exp['name']}_train_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "experiment": exp["name"],
                "config_path": str(config_path),
                "results": all_results,
            }, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 72}")
    print(f"Training Summary: {exp['name']}")
    print(f"{'=' * 72}")
    accs = [r["best_val_acc"] for r in all_results]
    bal_accs = [r["best_val_bal_acc"] for r in all_results]
    print(f"  Val accuracy:  {sum(accs)/len(accs):.3f} "
          f"(min={min(accs):.3f}, max={max(accs):.3f})")
    print(f"  Balanced acc:  {sum(bal_accs)/len(bal_accs):.3f} "
          f"(min={min(bal_accs):.3f}, max={max(bal_accs):.3f})")
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
