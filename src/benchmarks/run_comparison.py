"""Comparison experiment: Symmetric ReasoningLoop vs Hierarchical ExecutiveReasoningLoop.

Runs both architectures on the same multi-hop graph traversal task,
then runs the hierarchical model on Phase 3 temporal tasks.
"""

import torch
import torch.nn as nn
import yaml
import json
import time
from pathlib import Path

from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.loop import ReasoningLoop
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.temporal_tasks import TemporalDataset
from src.spectral.decomposition import hodge_decomposition


class SymmetricMultiHopModel(nn.Module):
    """Phase 1 architecture: symmetric GNN <-> TAT co-processing."""

    def __init__(self, embedding_dim, gnn_hidden, gnn_spatial_layers,
                 gnn_spectral_layers, max_freqs, tat_layers,
                 tat_spatial_heads, tat_spectral_heads, tat_ff_dim,
                 max_classes, max_iterations, convergence_threshold):
        super().__init__()
        self.reasoning_loop = ReasoningLoop(
            embedding_dim=embedding_dim, gnn_hidden=gnn_hidden,
            gnn_spatial_layers=gnn_spatial_layers,
            gnn_spectral_layers=gnn_spectral_layers,
            max_freqs=max_freqs, tat_layers=tat_layers,
            tat_spatial_heads=tat_spatial_heads,
            tat_spectral_heads=tat_spectral_heads,
            tat_ff_dim=tat_ff_dim, max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * embedding_dim, 4 * embedding_dim),
            nn.LayerNorm(4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.LayerNorm(2 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, max_classes),
        )

    def forward(self, cc, query_node, target_node):
        output, num_iters = self.reasoning_loop(cc)
        query_emb = output[query_node]
        target_emb = output[target_node]
        diff_emb = query_emb - target_emb
        combined = torch.cat([query_emb, target_emb, diff_emb])
        return self.classifier(combined)


class HierarchicalMultiHopModel(nn.Module):
    """Phase 3 architecture: hierarchical GNN -> TAT with control signals + wave dynamics."""

    def __init__(self, embedding_dim, gnn_hidden, gnn_spatial_layers,
                 gnn_spectral_layers, max_freqs, tat_layers,
                 tat_spatial_heads, tat_spectral_heads, tat_ff_dim,
                 max_classes, max_iterations, convergence_threshold,
                 use_wave_dynamics=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.executive_loop = ExecutiveReasoningLoop(
            embedding_dim=embedding_dim, gnn_hidden=gnn_hidden,
            gnn_spatial_layers=gnn_spatial_layers,
            gnn_spectral_layers=gnn_spectral_layers,
            max_freqs=max_freqs, tat_layers=tat_layers,
            tat_spatial_heads=tat_spatial_heads,
            tat_spectral_heads=tat_spectral_heads,
            tat_ff_dim=tat_ff_dim, max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            use_wave_dynamics=use_wave_dynamics,
        )

        # Same classifier dimensions as symmetric + hodge(3) + wave_energy(1)
        classifier_input_dim = 3 * embedding_dim + 3 + 1
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 4 * embedding_dim),
            nn.LayerNorm(4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.LayerNorm(2 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, max_classes),
        )

    def _compute_hodge_features(self, cc):
        if cc.num_cells(1) == 0 or cc.num_cells(0) == 0:
            return torch.zeros(3)
        try:
            edge_embs = cc.get_embeddings(1)
            signal = edge_embs.mean(dim=1)
            gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
            return torch.tensor([
                gradient.abs().mean().item(),
                curl.abs().mean().item(),
                harmonic.abs().mean().item(),
            ])
        except (RuntimeError, ValueError):
            return torch.zeros(3)

    def _compute_wave_energy(self, diagnostics):
        control_signals = diagnostics.get('control_signals', [])
        if not control_signals:
            return torch.zeros(1)
        last_control = control_signals[-1]
        dt = last_control.diffusion_time
        wd = last_control.wave_damping
        energy = dt.pow(2) + wd.pow(2)
        return energy.unsqueeze(0)

    def forward(self, cc, query_node, target_node):
        output, num_iters, diagnostics = self.executive_loop(cc)
        query_emb = output[query_node]
        target_emb = output[target_node]
        diff_emb = query_emb - target_emb
        hodge_features = self._compute_hodge_features(cc)
        wave_energy = self._compute_wave_energy(diagnostics)
        combined = torch.cat([query_emb, target_emb, diff_emb, hodge_features, wave_energy])
        return self.classifier(combined)


def train_epoch(model, dataset, optimizer, max_norm=5.0, accumulation_steps=4):
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


def run_model(name, model, train_ds, test_ds, tc):
    """Train and evaluate a single model."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {name}: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_test_acc = 0.0
    patience_counter = 0
    results = []

    for epoch in range(tc["epochs"]):
        start = time.time()
        train_loss = train_epoch(model, train_ds, optimizer,
                                 max_norm=tc.get("max_norm", 5.0),
                                 accumulation_steps=tc.get("accumulation_steps", 4))
        test_acc, test_loss = evaluate(model, test_ds)
        elapsed = time.time() - start

        scheduler.step(test_loss)

        result = {
            "epoch": epoch, "train_loss": train_loss,
            "test_loss": test_loss, "test_accuracy": test_acc, "time": elapsed,
        }
        results.append(result)

        print(f"    Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.3f} | {elapsed:.1f}s")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"    Early stopping at epoch {epoch}")
                break

    return {"results": results, "best_accuracy": best_test_acc, "params": total_params}


def run_comparison(config_path: str = "config/comparison.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]

    print("=" * 70)
    print("Comparison: Symmetric (Phase 1) vs Hierarchical (Phase 3)")
    print("=" * 70)

    # ---- Part 1: Multi-hop task (apples-to-apples comparison) ----
    print("\n" + "-" * 70)
    print("PART 1: Multi-Hop Graph Traversal (same task, different architectures)")
    print("-" * 70)

    max_hops = dc["max_hops"]
    max_classes = max_hops + 1  # answers 0..max_hops

    print("\nGenerating multi-hop datasets...")
    train_ds = MultiHopDataset(
        num_samples=dc["num_train"], min_hops=dc["min_hops"],
        max_hops=max_hops, num_distractors=dc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
    )
    test_ds = MultiHopDataset(
        num_samples=dc["num_test"], min_hops=dc["min_hops"],
        max_hops=max_hops, num_distractors=dc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
    )
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Model A: Symmetric
    print("\n--- A) Symmetric ReasoningLoop ---")
    sym_model = SymmetricMultiHopModel(
        embedding_dim=mc["embedding_dim"], gnn_hidden=mc["gnn_hidden"],
        gnn_spatial_layers=mc["gnn_spatial_layers"],
        gnn_spectral_layers=mc["gnn_spectral_layers"],
        max_freqs=mc["max_freqs"], tat_layers=mc["tat_layers"],
        tat_spatial_heads=mc["tat_spatial_heads"],
        tat_spectral_heads=mc["tat_spectral_heads"],
        tat_ff_dim=mc["tat_ff_dim"], max_classes=max_classes,
        max_iterations=mc["max_iterations"],
        convergence_threshold=mc["convergence_threshold"],
    )
    sym_results = run_model("Symmetric", sym_model, train_ds, test_ds, tc)

    # Model B: Hierarchical (with wave dynamics)
    print("\n--- B) Hierarchical ExecutiveReasoningLoop (wave ON) ---")
    hier_model = HierarchicalMultiHopModel(
        embedding_dim=mc["embedding_dim"], gnn_hidden=mc["gnn_hidden"],
        gnn_spatial_layers=mc["gnn_spatial_layers"],
        gnn_spectral_layers=mc["gnn_spectral_layers"],
        max_freqs=mc["max_freqs"], tat_layers=mc["tat_layers"],
        tat_spatial_heads=mc["tat_spatial_heads"],
        tat_spectral_heads=mc["tat_spectral_heads"],
        tat_ff_dim=mc["tat_ff_dim"], max_classes=max_classes,
        max_iterations=mc["max_iterations"],
        convergence_threshold=mc["convergence_threshold"],
        use_wave_dynamics=True,
    )
    hier_results = run_model("Hierarchical", hier_model, train_ds, test_ds, tc)

    # Model C: Hierarchical without wave (to isolate effect of control signals vs wave)
    print("\n--- C) Hierarchical ExecutiveReasoningLoop (wave OFF) ---")
    hier_nowave = HierarchicalMultiHopModel(
        embedding_dim=mc["embedding_dim"], gnn_hidden=mc["gnn_hidden"],
        gnn_spatial_layers=mc["gnn_spatial_layers"],
        gnn_spectral_layers=mc["gnn_spectral_layers"],
        max_freqs=mc["max_freqs"], tat_layers=mc["tat_layers"],
        tat_spatial_heads=mc["tat_spatial_heads"],
        tat_spectral_heads=mc["tat_spectral_heads"],
        tat_ff_dim=mc["tat_ff_dim"], max_classes=max_classes,
        max_iterations=mc["max_iterations"],
        convergence_threshold=mc["convergence_threshold"],
        use_wave_dynamics=False,
    )
    nowave_results = run_model("Hierarchical (no wave)", hier_nowave, train_ds, test_ds, tc)

    # ---- Part 2: Temporal tasks (hierarchical-only) ----
    print("\n" + "-" * 70)
    print("PART 2: Temporal Tasks (hierarchical only — symmetric can't do these)")
    print("-" * 70)

    temporal_results = {}
    for task_type in ["propagation_delay", "blocking", "interference"]:
        task_max_classes = dc.get("max_delay", 10)
        if task_type == "interference":
            task_max_classes = 2

        task_kwargs = {"n_nodes": dc.get("n_nodes", 10), "max_delay": dc.get("max_delay", 10)}
        if task_type == "interference":
            task_kwargs.pop("max_delay")

        print(f"\n--- {task_type} ---")
        temp_train = TemporalDataset(
            num_samples=dc["num_train"], task_type=task_type,
            embedding_dim=mc["embedding_dim"], **task_kwargs,
        )
        temp_test = TemporalDataset(
            num_samples=dc["num_test"], task_type=task_type,
            embedding_dim=mc["embedding_dim"], **task_kwargs,
        )

        from src.benchmarks.phase3_model import TemporalReasoningModel
        temp_model = TemporalReasoningModel(
            embedding_dim=mc["embedding_dim"], gnn_hidden=mc["gnn_hidden"],
            gnn_spatial_layers=mc["gnn_spatial_layers"],
            gnn_spectral_layers=mc["gnn_spectral_layers"],
            max_freqs=mc["max_freqs"], tat_layers=mc["tat_layers"],
            tat_spatial_heads=mc["tat_spatial_heads"],
            tat_spectral_heads=mc["tat_spectral_heads"],
            tat_ff_dim=mc["tat_ff_dim"], max_classes=task_max_classes,
            max_iterations=mc["max_iterations"],
            convergence_threshold=mc["convergence_threshold"],
            use_wave_dynamics=True,
        )
        temporal_results[task_type] = run_model(
            f"Temporal ({task_type})", temp_model, temp_train, temp_test, tc,
        )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMulti-Hop Graph Traversal:")
    print(f"  A) Symmetric:                {sym_results['best_accuracy']:.3f}  ({sym_results['params']:,} params)")
    print(f"  B) Hierarchical (wave ON):   {hier_results['best_accuracy']:.3f}  ({hier_results['params']:,} params)")
    print(f"  C) Hierarchical (wave OFF):  {nowave_results['best_accuracy']:.3f}  ({nowave_results['params']:,} params)")

    print("\nTemporal Tasks (hierarchical only):")
    for task, res in temporal_results.items():
        print(f"  {task:25s}: {res['best_accuracy']:.3f}")

    # Timing comparison
    sym_total = sum(r["time"] for r in sym_results["results"])
    hier_total = sum(r["time"] for r in hier_results["results"])
    nowave_total = sum(r["time"] for r in nowave_results["results"])
    print(f"\nTotal training time:")
    print(f"  Symmetric:               {sym_total:.0f}s")
    print(f"  Hierarchical (wave ON):  {hier_total:.0f}s")
    print(f"  Hierarchical (wave OFF): {nowave_total:.0f}s")

    # Save results
    all_results = {
        "multi_hop": {
            "symmetric": sym_results,
            "hierarchical_wave": hier_results,
            "hierarchical_nowave": nowave_results,
        },
        "temporal": temporal_results,
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to data/comparison_results.json")


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/comparison.yaml"
    run_comparison(config)
