"""Phase 1 Experiment: Multi-hop graph traversal benchmark."""

import torch
import yaml
import json
import time
from pathlib import Path
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.trainer import train_epoch, evaluate


def _resolve_device(tc):
    """Resolve training device from config."""
    dev = tc.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def run_experiment(config_path: str = "config/default.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    rc = config["reasoning_loop"]

    device = _resolve_device(tc)

    print("=" * 60)
    print("Phase 1: Multi-Hop Graph Traversal Benchmark")
    print(f"Device: {device}")
    print("=" * 60)

    task_type = bc.get("task_type", "chain")
    n_nodes_range = tuple(bc.get("n_nodes_range", [30, 80]))

    print(f"\nGenerating datasets (task_type={task_type})...")
    train_ds = MultiHopDataset(
        num_samples=bc["num_train"], min_hops=bc["min_hops"],
        max_hops=bc["max_hops"], num_distractors=bc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
        task_type=task_type, n_nodes_range=n_nodes_range,
    )
    val_ds = MultiHopDataset(
        num_samples=bc["num_val"], min_hops=bc["min_hops"],
        max_hops=bc["max_hops"], num_distractors=bc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
        task_type=task_type, n_nodes_range=n_nodes_range,
    )
    test_ds = MultiHopDataset(
        num_samples=bc["num_test"], min_hops=bc["min_hops"],
        max_hops=bc["max_hops"], num_distractors=bc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
        task_type=task_type, n_nodes_range=n_nodes_range,
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    model = MultiHopReasoningModel(
        embedding_dim=mc["embedding_dim"],
        gnn_hidden=mc["embedding_dim"] * 2,
        gnn_spatial_layers=mc["num_gnn_layers"],
        gnn_spectral_layers=mc["num_gnn_layers"],
        max_freqs=mc["embedding_dim"] // 2,
        tat_layers=mc["num_tat_layers"],
        tat_spatial_heads=mc["num_spatial_heads"],
        tat_spectral_heads=mc["num_spectral_heads"],
        tat_ff_dim=mc["tat_ff_dim"],
        max_hops=bc["max_hops"],
        max_iterations=rc["max_iterations"],
        convergence_threshold=rc.get("convergence_threshold", 0.1),
    )

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                                   weight_decay=tc.get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    patience_counter = 0
    results = []

    label_smoothing = tc.get("label_smoothing", 0.1)

    print("\nTraining...")
    for epoch in range(tc["max_epochs"]):
        start = time.time()
        train_loss, _ = train_epoch(model, train_ds, optimizer,
                                    label_smoothing=label_smoothing,
                                    device=device)
        val_acc, val_loss = evaluate(model, val_ds, label_smoothing=label_smoothing,
                                     device=device)
        elapsed = time.time() - start

        scheduler.step(val_loss)

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "time": elapsed,
        }
        results.append(result)

        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/best_phase1.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("models/best_phase1.pt", weights_only=True,
                                     map_location=device))
    test_acc, test_loss = evaluate(model, test_ds, label_smoothing=label_smoothing,
                                   device=device)
    print(f"\nTest Accuracy: {test_acc:.3f} | Test Loss: {test_loss:.4f}")

    print("\nPer-hop accuracy:")
    for h in range(bc["min_hops"], bc["max_hops"] + 1):
        hop_ds = MultiHopDataset(
            num_samples=100, min_hops=h, max_hops=h,
            num_distractors=bc["num_distractors"],
            embedding_dim=mc["embedding_dim"],
        )
        hop_acc, _ = evaluate(model, hop_ds, label_smoothing=label_smoothing,
                              device=device)
        print(f"  {h}-hop: {hop_acc:.3f}")

    Path("data").mkdir(exist_ok=True)
    with open("data/phase1_results.json", "w") as f:
        json.dump({"results": results, "test_accuracy": test_acc, "test_loss": test_loss}, f, indent=2)


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    run_experiment(config)
