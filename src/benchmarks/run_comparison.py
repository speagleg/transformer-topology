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
from src.spectral.persistence import vectorize_persistence, compute_persistence_diagram
from src.benchmarks.phase3_model import PERSISTENCE_FEATURES
from src.llm.backend import MockLLMBackend
from src.llm.topo_bridge import TopoBridge


class SymmetricMultiHopModel(nn.Module):
    """ARCHIVED: Phase 1 symmetric architecture.

    Superseded by HierarchicalMultiHopModel. Retained for backward
    compatibility and baseline comparisons. See docs/plans/architecture-decision-symmetric.md.
    """

    def __init__(self, embedding_dim, gnn_hidden, gnn_spatial_layers,
                 gnn_spectral_layers, max_freqs, tat_layers,
                 tat_spatial_heads, tat_spectral_heads, tat_ff_dim,
                 max_classes, max_iterations, convergence_threshold,
                 use_topological_pe=False,
                 use_structural_features=False):
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
            use_topological_pe=use_topological_pe,
            use_structural_features=use_structural_features,
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

    def forward(self, cc, query_node, target_node, metadata=None):
        output, num_iters = self.reasoning_loop(cc)
        query_emb = output[query_node]
        target_emb = output[target_node]
        diff_emb = query_emb - target_emb
        combined = torch.cat([query_emb, target_emb, diff_emb])
        return self.classifier(combined)


class HierarchicalMultiHopModel(nn.Module):
    """Phase 3 architecture: hierarchical GNN -> TAT with control signals + wave dynamics + Phase 2."""

    def __init__(self, embedding_dim, gnn_hidden, gnn_spatial_layers,
                 gnn_spectral_layers, max_freqs, tat_layers,
                 tat_spatial_heads, tat_spectral_heads, tat_ff_dim,
                 max_classes, max_iterations, convergence_threshold,
                 use_wave_dynamics=True, use_higher_order=True,
                 use_topological_pe=False,
                 use_structural_features=False,
                 wave_config=None,
                 use_llm=False, llm_config=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_llm = use_llm
        wc = wave_config or {}
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
            use_higher_order=use_higher_order,
            use_topological_pe=use_topological_pe,
            use_structural_features=use_structural_features,
            wave_filter_type=wc.get('filter_type', 'heat'),
            wave_laplacian_dim=wc.get('laplacian_dim', 0),
            wave_strength_gate=wc.get('wave_strength_gate', False),
            wave_use_neural_ode=wc.get('use_neural_ode', True),
            wave_mode=wc.get('wave_mode', 'spectral'),
            wave_filter_kwargs={
                k: v for k, v in wc.items()
                if k not in ('filter_type', 'laplacian_dim', 'wave_strength_gate',
                             'use_neural_ode', 'wave_mode')
            },
        )

        # TopoBridge (LLM integration)
        if use_llm:
            lc = llm_config or {}
            llm_dim = lc.get('llm_dim', 2048)
            num_prefix = lc.get('num_prefix', 8)
            gate_threshold = lc.get('gate_threshold', 0.1)
            backend = MockLLMBackend(llm_dim=llm_dim, hidden_dim=lc.get('mock_hidden', 256))
            self.topo_bridge = TopoBridge(
                backend=backend,
                topo_dim=embedding_dim,
                llm_dim=llm_dim,
                num_prefix=num_prefix,
                gate_threshold=gate_threshold,
            )
        else:
            self.topo_bridge = None

        # hodge(3) + wave_energy(1) + persistence(32)
        classifier_input_dim = 3 * embedding_dim + 3 + 1 + PERSISTENCE_FEATURES
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

    def _compute_hodge_features(self, cc, initial_edge_embs=None):
        if cc.num_cells(1) == 0 or cc.num_cells(0) == 0:
            return torch.zeros(3)
        try:
            edge_embs = initial_edge_embs if initial_edge_embs is not None else cc.get_embeddings(1)
            # Use dim 0 where the task signal lives (delay, curl/grad/harmonic).
            signal = edge_embs[:, 0]
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

    def _compute_persistence_features(self, cc):
        try:
            diagrams = compute_persistence_diagram(cc, max_dimension=1)
            return vectorize_persistence(diagrams, num_features=16)
        except (RuntimeError, ValueError):
            return torch.zeros(PERSISTENCE_FEATURES)

    def forward(self, cc, query_node, target_node, metadata=None):
        # Capture initial edge embeddings before the executive loop overwrites
        # them (GNN edge_out is gradient-dominated, destroys curl content).
        initial_edge_embs = cc.get_embeddings(1).clone() if cc.num_cells(1) > 0 else None

        output, num_iters, diagnostics = self.executive_loop(cc)

        # LLM integration: blend executive output with TopoBridge output
        if self.use_llm and self.topo_bridge is not None:
            control_signals = diagnostics.get('control_signals', [])
            if control_signals:
                llm_gate = control_signals[-1].llm_gate
            else:
                llm_gate = torch.tensor(0.0, device=output.device)

            task_text = None
            if metadata and 'task_prompt' in metadata:
                task_text = metadata['task_prompt']

            llm_out = self.topo_bridge(output, llm_gate, task_text)

            # Blend: output = (1 - llm_gate) * executive_output + llm_gate * llm_out
            # TopoBridge already scales by llm_gate, so we just add
            output = (1 - llm_gate).unsqueeze(-1) * output + llm_out

        query_emb = output[query_node]
        target_emb = output[target_node]
        diff_emb = query_emb - target_emb
        dev = query_emb.device
        hodge_features = self._compute_hodge_features(
            cc, initial_edge_embs=initial_edge_embs,
        ).to(dev)
        wave_energy = self._compute_wave_energy(diagnostics).to(dev)
        persistence_features = self._compute_persistence_features(cc).to(dev)
        combined = torch.cat([query_emb, target_emb, diff_emb,
                              hodge_features, wave_energy, persistence_features])
        return self.classifier(combined)


def _unpack_sample(sample):
    """Unpack a 4-tuple or 5-tuple sample."""
    if len(sample) == 5:
        cc, query, target, answer, metadata = sample
        return cc, query, target, answer, metadata
    cc, query, target, answer = sample
    return cc, query, target, answer, None


def train_epoch(model, dataset, optimizer, max_norm=5.0, accumulation_steps=4,
                label_smoothing=0.0, device=None):
    if device is None:
        device = torch.device('cpu')
    model.train()
    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()
    total_loss = 0.0
    grad_norms = []
    optimizer.zero_grad()
    for i in range(len(dataset)):
        cc, query, target, answer, metadata = _unpack_sample(dataset[i])
        cc = cc.clone().to(device)
        logits = model(cc, query, target, metadata=metadata)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0),
                                           torch.tensor([answer], device=device),
                                           label_smoothing=label_smoothing)
        loss = loss / accumulation_steps
        # Skip backward if loss is NaN (prevents NaN gradient corruption)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        total_loss += loss.item() * accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataset):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            gn_val = gn.item() if isinstance(gn, torch.Tensor) else gn
            grad_norms.append(gn_val)
            # Skip optimizer step if gradients are NaN/Inf (prevents weight corruption)
            if not (gn_val != gn_val or gn_val == float('inf')):  # NaN != NaN is True
                optimizer.step()
            optimizer.zero_grad()
    avg_loss = total_loss / len(dataset)
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    return avg_loss, avg_grad_norm


@torch.no_grad()
def evaluate(model, dataset, label_smoothing=0.0, device=None):
    if device is None:
        device = torch.device('cpu')
    model.eval()
    correct = 0
    total_loss = 0.0
    n_valid = 0
    for i in range(len(dataset)):
        cc, query, target, answer, metadata = _unpack_sample(dataset[i])
        cc = cc.clone().to(device)
        logits = model(cc, query, target, metadata=metadata)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0),
                                           torch.tensor([answer], device=device),
                                           label_smoothing=label_smoothing)
        loss_val = loss.item()
        if loss_val == loss_val and loss_val != float('inf'):  # skip NaN/Inf
            total_loss += loss_val
            n_valid += 1
        pred = logits.argmax().item()
        if pred == answer:
            correct += 1
    accuracy = correct / len(dataset)
    avg_loss = total_loss / max(n_valid, 1)
    return accuracy, avg_loss


@torch.no_grad()
def evaluate_per_class(model, dataset, num_classes, device=None):
    """Evaluate model and return per-class accuracy breakdown.

    Returns:
        dict with 'per_class': {class_id: {correct, total, accuracy}},
                   'confusion': {true_class: {pred_class: count}}
    """
    if device is None:
        device = torch.device('cpu')
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    confusion = {}

    for i in range(len(dataset)):
        cc, query, target, answer, metadata = _unpack_sample(dataset[i])
        cc = cc.clone().to(device)
        logits = model(cc, query, target, metadata=metadata)
        pred = logits.argmax().item()

        if answer < num_classes:
            class_total[answer] += 1
            if pred == answer:
                class_correct[answer] += 1
            # Confusion matrix
            confusion.setdefault(answer, {})
            confusion[answer][pred] = confusion[answer].get(pred, 0) + 1

    per_class = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class[c] = {
                'correct': class_correct[c],
                'total': class_total[c],
                'accuracy': class_correct[c] / class_total[c],
            }

    return {'per_class': per_class, 'confusion': confusion}


def run_model(name, model, train_ds, test_ds, tc, device=None):
    """Train and evaluate a single model."""
    if device is None:
        device = torch.device('cpu')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {name}: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                                   weight_decay=tc.get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_test_acc = 0.0
    patience_counter = 0
    results = []

    for epoch in range(tc["epochs"]):
        start = time.time()
        label_smoothing = tc.get("label_smoothing", 0.1)
        train_loss, grad_norm = train_epoch(model, train_ds, optimizer,
                                 max_norm=tc.get("max_norm", 5.0),
                                 accumulation_steps=tc.get("accumulation_steps", 4),
                                 label_smoothing=label_smoothing,
                                 device=device)
        test_acc, test_loss = evaluate(model, test_ds, label_smoothing=label_smoothing,
                                       device=device)
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


def _resolve_device(tc):
    """Resolve training device from config."""
    dev = tc.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def run_comparison(config_path: str = "config/comparison.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]

    device = _resolve_device(tc)

    print("=" * 70)
    print("Comparison: Symmetric (Phase 1) vs Hierarchical (Phase 3)")
    print(f"Device: {device}")
    print("=" * 70)

    # ---- Part 1: Multi-hop task (apples-to-apples comparison) ----
    print("\n" + "-" * 70)
    print("PART 1: Multi-Hop Graph Traversal (same task, different architectures)")
    print("-" * 70)

    max_hops = dc["max_hops"]
    max_classes = max_hops + 1  # answers 0..max_hops

    task_type = dc.get("task_type", "chain")
    n_nodes_range = tuple(dc.get("n_nodes_range", [30, 80]))

    print(f"\nGenerating multi-hop datasets (task_type={task_type})...")
    train_ds = MultiHopDataset(
        num_samples=dc["num_train"], min_hops=dc["min_hops"],
        max_hops=max_hops, num_distractors=dc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
        task_type=task_type, n_nodes_range=n_nodes_range,
    )
    test_ds = MultiHopDataset(
        num_samples=dc["num_test"], min_hops=dc["min_hops"],
        max_hops=max_hops, num_distractors=dc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
        task_type=task_type, n_nodes_range=n_nodes_range,
    )
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    use_topo_pe = mc.get("use_topological_pe", False)
    use_struct = mc.get("use_structural_features", False)

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
        use_topological_pe=use_topo_pe,
        use_structural_features=use_struct,
    )
    sym_model.to(device)
    sym_results = run_model("Symmetric", sym_model, train_ds, test_ds, tc, device=device)

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
        use_topological_pe=use_topo_pe,
        use_structural_features=use_struct,
    )
    hier_model.to(device)
    hier_results = run_model("Hierarchical", hier_model, train_ds, test_ds, tc, device=device)

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
        use_topological_pe=use_topo_pe,
        use_structural_features=use_struct,
    )
    hier_nowave.to(device)
    nowave_results = run_model("Hierarchical (no wave)", hier_nowave, train_ds, test_ds, tc, device=device)

    # ---- Part 2: Temporal tasks (hierarchical-only) ----
    print("\n" + "-" * 70)
    print("PART 2: Temporal Tasks (hierarchical only — symmetric can't do these)")
    print("-" * 70)

    temporal_results = {}
    for task_type in ["propagation_delay", "blocking", "interference"]:
        task_max_classes = dc.get("max_delay", 10)
        if task_type == "interference":
            task_max_classes = 2

        task_kwargs = {
            "n_nodes": dc.get("n_nodes", 10),
            "max_delay": dc.get("max_delay", 10),
            "max_edge_delay": dc.get("max_edge_delay", 3),
            "use_diverse_topology": dc.get("use_diverse_topology", False),
        }
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
            use_topological_pe=use_topo_pe,
            use_structural_features=use_struct,
        )
        temp_model.to(device)
        temporal_results[task_type] = run_model(
            f"Temporal ({task_type})", temp_model, temp_train, temp_test, tc,
            device=device,
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
