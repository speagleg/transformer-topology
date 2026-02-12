"""Phase 3 Experiment: Temporal/causal reasoning with executive loop and wave dynamics."""

import torch
import torch.nn as nn
import yaml
import json
import time
from pathlib import Path
from src.benchmarks.phase3_model import TemporalReasoningModel
from src.benchmarks.temporal_tasks import TemporalDataset


def make_model(mc, max_classes):
    return TemporalReasoningModel(
        embedding_dim=mc["embedding_dim"],
        gnn_hidden=mc["gnn_hidden"],
        gnn_spatial_layers=mc["gnn_spatial_layers"],
        gnn_spectral_layers=mc["gnn_spectral_layers"],
        max_freqs=mc["max_freqs"],
        tat_layers=mc["tat_layers"],
        tat_spatial_heads=mc["tat_spatial_heads"],
        tat_spectral_heads=mc["tat_spectral_heads"],
        tat_ff_dim=mc["tat_ff_dim"],
        max_classes=max_classes,
        max_iterations=mc["max_iterations"],
        convergence_threshold=mc["convergence_threshold"],
        use_wave_dynamics=mc["use_wave_dynamics"],
    )


def train_epoch(model, dataset, optimizer, max_norm=5.0, accumulation_steps=4):
    """Train for one epoch over dataset samples."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        logits = model(cc, query, target)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([answer]))
        loss = loss / accumulation_steps
        loss.backward()
        total_loss += loss.item() * accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataset):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / len(dataset)


@torch.no_grad()
def evaluate(model, dataset):
    """Evaluate model accuracy and loss on a dataset."""
    model.eval()
    correct = 0
    total_loss = 0.0
    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        logits = model(cc, query, target)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([answer]))
        total_loss += loss.item()
        pred = logits.argmax().item()
        if pred == answer:
            correct += 1
    accuracy = correct / len(dataset)
    avg_loss = total_loss / len(dataset)
    return accuracy, avg_loss


def run_task_experiment(task_type, model, tc, dc, tag=""):
    """Run a single task experiment: generate data, train, evaluate."""
    max_classes = dc.get("max_delay", 10)
    if task_type == "interference":
        max_classes = 2  # binary: constructive vs destructive

    task_kwargs = {
        "n_nodes": dc.get("n_nodes", 10),
        "max_delay": dc.get("max_delay", 10),
    }
    # interference task does not use max_delay
    if task_type == "interference":
        task_kwargs.pop("max_delay")

    print(f"\n--- {task_type} ---")
    print("Generating datasets...")
    train_ds = TemporalDataset(
        num_samples=dc["num_train"], task_type=task_type,
        embedding_dim=model.embedding_dim, **task_kwargs,
    )
    test_ds = TemporalDataset(
        num_samples=dc["num_test"], task_type=task_type,
        embedding_dim=model.embedding_dim, **task_kwargs,
    )
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_test_acc = 0.0
    patience_counter = 0
    results = []

    max_norm = tc.get("max_norm", 5.0)
    accumulation_steps = tc.get("accumulation_steps", 4)

    print("Training...")
    for epoch in range(tc["epochs"]):
        start = time.time()
        train_loss = train_epoch(model, train_ds, optimizer,
                                 max_norm=max_norm,
                                 accumulation_steps=accumulation_steps)
        test_acc, test_loss = evaluate(model, test_ds)
        elapsed = time.time() - start

        scheduler.step(test_loss)

        result = {
            "epoch": epoch, "train_loss": train_loss,
            "test_loss": test_loss, "test_accuracy": test_acc, "time": elapsed,
        }
        results.append(result)

        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.3f} | {elapsed:.1f}s")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"models/best_phase3_{task_type}.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model and report final test accuracy
    best_path = f"models/best_phase3_{task_type}.pt"
    if Path(best_path).exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
    test_acc, test_loss = evaluate(model, test_ds)
    print(f"  Final Test Accuracy: {test_acc:.3f} | Test Loss: {test_loss:.4f}")

    return {"task": task_type, "results": results,
            "test_accuracy": test_acc, "test_loss": test_loss}


def run_experiment(config_path: str = "config/phase3.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]

    print("=" * 60)
    print("Phase 3: Temporal/Causal Reasoning Benchmarks")
    print("=" * 60)

    task_types = ["propagation_delay", "blocking", "interference"]
    all_results = {}

    for task_type in task_types:
        max_classes = dc.get("max_delay", 10)
        if task_type == "interference":
            max_classes = 2

        model = make_model(mc, max_classes)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters ({task_type}): {total_params:,}")

        task_results = run_task_experiment(task_type, model, tc, dc)
        all_results[task_type] = task_results

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task, res in all_results.items():
        print(f"  {task}: {res['test_accuracy']:.3f}")

    Path("data").mkdir(exist_ok=True)
    with open("data/phase3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to data/phase3_results.json")


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/phase3.yaml"
    run_experiment(config)
