"""Phase 2 Experiment: Topological reasoning benchmarks."""

import torch
import yaml
import json
import time
from pathlib import Path
from src.benchmarks.phase2_model import TopologicalReasoningModel
from src.benchmarks.topological_tasks import TopologicalDataset
from src.benchmarks.trainer import train_epoch, evaluate


def make_model(mc, rc, max_classes):
    return TopologicalReasoningModel(
        embedding_dim=mc["embedding_dim"],
        gnn_hidden=mc["embedding_dim"] * 2,
        gnn_spatial_layers=mc["num_gnn_layers"],
        gnn_spectral_layers=mc["num_gnn_layers"],
        max_freqs=mc["embedding_dim"] // 2,
        tat_layers=mc["num_tat_layers"],
        tat_spatial_heads=mc["num_spatial_heads"],
        tat_spectral_heads=mc["num_spectral_heads"],
        tat_ff_dim=mc["tat_ff_dim"],
        max_classes=max_classes,
        max_iterations=rc["max_iterations"],
        convergence_threshold=rc.get("convergence_threshold", 0.1),
    )


def run_task_experiment(task_type: str, model, tc, bc, tag: str = ""):
    max_classes = bc.get("max_classes", 6)
    task_kwargs = {}
    if task_type == "cycle_detection":
        task_kwargs["num_nodes"] = bc.get("num_nodes", 10)
    elif task_type == "path_counting":
        pass  # uses defaults
    elif task_type == "betti_number":
        task_kwargs["max_beta"] = bc.get("max_beta", 5)

    print(f"\n--- {task_type} ---")
    print("Generating datasets...")
    train_ds = TopologicalDataset(
        num_samples=bc["num_train"], task_type=task_type,
        embedding_dim=model.embedding_dim, **task_kwargs,
    )
    val_ds = TopologicalDataset(
        num_samples=bc["num_val"], task_type=task_type,
        embedding_dim=model.embedding_dim, **task_kwargs,
    )
    test_ds = TopologicalDataset(
        num_samples=bc["num_test"], task_type=task_type,
        embedding_dim=model.embedding_dim, **task_kwargs,
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    patience_counter = 0
    results = []

    max_norm = tc.get("max_norm", 5.0)
    accumulation_steps = tc.get("accumulation_steps", 4)

    print("Training...")
    for epoch in range(tc["max_epochs"]):
        start = time.time()
        train_loss, _ = train_epoch(model, train_ds, optimizer,
                                    max_norm=max_norm,
                                    accumulation_steps=accumulation_steps)
        val_acc, val_loss = evaluate(model, val_ds)
        elapsed = time.time() - start

        scheduler.step(val_loss)

        result = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_accuracy": val_acc, "time": elapsed,
        }
        results.append(result)

        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"models/best_phase2_{task_type}.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(f"models/best_phase2_{task_type}.pt", weights_only=True))
    test_acc, test_loss = evaluate(model, test_ds)
    print(f"  Test Accuracy: {test_acc:.3f} | Test Loss: {test_loss:.4f}")

    return {"task": task_type, "results": results,
            "test_accuracy": test_acc, "test_loss": test_loss}


def run_experiment(config_path: str = "config/phase2.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    rc = config["reasoning_loop"]

    print("=" * 60)
    print("Phase 2: Topological Reasoning Benchmarks")
    print("=" * 60)

    max_classes = bc.get("max_classes", 6)
    all_results = {}

    for task_type in bc["tasks"]:
        model = make_model(mc, rc, max_classes)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters ({task_type}): {total_params:,}")

        task_results = run_task_experiment(task_type, model, tc, bc)
        all_results[task_type] = task_results

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task, res in all_results.items():
        print(f"  {task}: {res['test_accuracy']:.3f}")

    Path("data").mkdir(exist_ok=True)
    with open("data/phase2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/phase2.yaml"
    run_experiment(config)
