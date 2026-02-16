"""Phase 4c curriculum training: three-phase LLM integration.

Phase A: Regression lock — train on existing tasks with llm_gate penalty.
         Validates that LLM infrastructure doesn't break existing performance.
Phase B: Graph completion — remove gate penalty, let model discover LLM utility.
Phase C: Language + Analogy — full training on all tasks, no gate constraints.

Usage:
    python scripts/run_phase4c_curriculum.py [config_path]
    python scripts/run_phase4c_curriculum.py config/benchmark_4c_llm.yaml
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
from src.benchmarks.diagnostics import DiagnosticCollector
from src.benchmarks.run_benchmark_suite import _build_model, _resolve_device
from src.benchmarks.run_comparison import _unpack_sample


def _llm_gate_penalty(model, weight: float = 1.0) -> torch.Tensor:
    """Auxiliary loss that penalizes llm_gate > 0.

    Encourages the model to keep the gate closed during Phase A
    (regression lock), so LLM doesn't hurt existing tasks.

    Reads the last control signal from the most recent forward pass
    stored in the executive loop's diagnostics.
    """
    if not hasattr(model, 'executive_loop'):
        return torch.tensor(0.0)

    # The executive loop stores control signals during forward
    # We access them via the model's last forward pass diagnostics
    # This is called right after model.forward(), so the diagnostics are fresh
    return weight * torch.tensor(0.0)  # placeholder — actual penalty applied in train loop


def train_epoch_with_gate_penalty(
    model, dataset, optimizer, gate_penalty_weight=0.0,
    max_norm=5.0, accumulation_steps=4, label_smoothing=0.0, device=None,
):
    """Training loop with optional llm_gate penalty.

    When gate_penalty_weight > 0, adds an auxiliary loss term that
    penalizes the llm_gate value, encouraging the model to keep the
    LLM pathway closed.
    """
    if device is None:
        device = torch.device('cpu')
    model.train()
    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()

    total_loss = 0.0
    total_gate_loss = 0.0
    grad_norms = []
    optimizer.zero_grad()

    for i in range(len(dataset)):
        cc, query, target, answer, metadata = _unpack_sample(dataset[i])
        cc = cc.clone().to(device)

        logits = model(cc, query, target, metadata=metadata)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([answer], device=device),
            label_smoothing=label_smoothing,
        )

        # Gate penalty: penalize llm_gate if weight > 0
        gate_loss = torch.tensor(0.0, device=device)
        if gate_penalty_weight > 0 and hasattr(model, 'executive_loop'):
            # Access the last control signal from the executive loop
            # The forward pass just ran, so diagnostics are available
            # We need to run the executive loop again to get the gate value
            # Actually, we extract it from the model's last forward
            if hasattr(model, 'use_llm') and model.use_llm:
                # Get gate from the last control signal
                # The executive_loop stores diagnostics during forward
                # We can get the llm_gate from the control head directly
                with torch.no_grad():
                    pass  # Gate is already used in forward

                # Recompute gate with grad for penalty
                # Use the pooled embedding approach
                node_emb = cc.get_embeddings(0).to(device)
                ctrl = model.executive_loop.gnn_executive.control_head(node_emb)
                gate_loss = gate_penalty_weight * ctrl.llm_gate

        loss = (ce_loss + gate_loss) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        total_loss += ce_loss.item()
        total_gate_loss += gate_loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataset):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            gn_val = gn.item() if isinstance(gn, torch.Tensor) else gn
            grad_norms.append(gn_val)
            if not (gn_val != gn_val or gn_val == float('inf')):
                optimizer.step()
            optimizer.zero_grad()

    n = len(dataset)
    avg_loss = total_loss / n
    avg_gate_loss = total_gate_loss / n
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    return avg_loss, avg_gate_loss, avg_grad_norm


def run_curriculum(config_path: str = "config/benchmark_4c_llm.yaml"):
    """Run the three-phase curriculum training."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    wc = config.get("wave")
    lc = config.get("llm")
    cc = config.get("curriculum", {})

    device = _resolve_device(tc)
    output_dir = Path(bc.get("output_dir", "data/phase4c_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 4c: LLM Integration Curriculum Training")
    print(f"Device: {device}")
    print("=" * 72)

    # Build model with LLM
    max_classes = max(
        get_max_classes("diverse"),
        get_max_classes("graph_completion"),
        get_max_classes("labeled_reasoning"),
        get_max_classes("analogical_transfer"),
    )
    # Use the largest max_classes across all tasks
    max_classes = bc.get("max_classes", max_classes)

    model = _build_model("hierarchical_llm", mc, max_classes, device,
                          wave_config=wc, llm_config=lc)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    emb_dim = mc["embedding_dim"]
    train_n = bc["train_n_nodes"]
    all_results = {}

    # ---- Phase A: Regression Lock ----
    phase_a = cc.get("phase_a", {})
    if phase_a.get("enabled", True):
        print(f"\n{'─' * 72}")
        print("Phase A: Regression Lock (existing tasks, gate penalty)")
        print(f"{'─' * 72}")

        gate_penalty = phase_a.get("gate_penalty_weight", 1.0)
        phase_a_tasks = phase_a.get("tasks", ["diverse", "bfs", "hodge_class", "spectral_gap"])
        phase_a_epochs = phase_a.get("epochs", tc["epochs"])

        # Generate datasets for phase A tasks
        for task in phase_a_tasks:
            task_classes = get_max_classes(task)
            print(f"\n  Task: {task} ({task_classes} classes)")
            train_ds = BenchmarkDataset(
                bc.get("num_train", 500), task, train_n, emb_dim,
            )
            val_ds = BenchmarkDataset(
                bc.get("num_val", 100), task, train_n, emb_dim,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=tc["learning_rate"],
                weight_decay=tc.get("weight_decay", 0.01),
            )

            best_val_acc = 0.0
            for epoch in range(phase_a_epochs):
                t0 = time.time()
                loss, gate_loss, gn = train_epoch_with_gate_penalty(
                    model, train_ds, optimizer,
                    gate_penalty_weight=gate_penalty,
                    label_smoothing=tc.get("label_smoothing", 0.1),
                    device=device,
                )
                # Quick eval
                from src.benchmarks.run_comparison import evaluate
                val_acc, val_loss = evaluate(model, val_ds, device=device)
                elapsed = time.time() - t0

                marker = ""
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    marker = " *"

                print(f"    Ep {epoch:3d} | loss {loss:.4f} gate_loss {gate_loss:.4f} | "
                      f"val {val_acc:.3f} | gn {gn:.2f} | {elapsed:.0f}s{marker}")

            all_results[f"phase_a_{task}"] = {"best_val_acc": best_val_acc}

    # ---- Phase B: Graph Completion ----
    phase_b = cc.get("phase_b", {})
    if phase_b.get("enabled", True):
        print(f"\n{'─' * 72}")
        print("Phase B: Graph Completion (no gate penalty)")
        print(f"{'─' * 72}")

        phase_b_epochs = phase_b.get("epochs", tc["epochs"])

        train_ds = BenchmarkDataset(
            bc.get("num_train", 500), "graph_completion", train_n, emb_dim,
        )
        val_ds = BenchmarkDataset(
            bc.get("num_val", 100), "graph_completion", train_n, emb_dim,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=tc["learning_rate"] * 0.5,
            weight_decay=tc.get("weight_decay", 0.01),
        )

        best_val_acc = 0.0
        for epoch in range(phase_b_epochs):
            t0 = time.time()
            loss, _, gn = train_epoch_with_gate_penalty(
                model, train_ds, optimizer,
                gate_penalty_weight=0.0,
                label_smoothing=tc.get("label_smoothing", 0.1),
                device=device,
            )
            from src.benchmarks.run_comparison import evaluate
            val_acc, val_loss = evaluate(model, val_ds, device=device)
            elapsed = time.time() - t0

            marker = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                marker = " *"

            print(f"    Ep {epoch:3d} | loss {loss:.4f} | "
                  f"val {val_acc:.3f} | gn {gn:.2f} | {elapsed:.0f}s{marker}")

        all_results["phase_b_graph_completion"] = {"best_val_acc": best_val_acc}

    # ---- Phase C: Language + Analogy ----
    phase_c = cc.get("phase_c", {})
    if phase_c.get("enabled", True):
        print(f"\n{'─' * 72}")
        print("Phase C: Language + Analogy (all tasks)")
        print(f"{'─' * 72}")

        phase_c_epochs = phase_c.get("epochs", tc["epochs"])
        phase_c_tasks = phase_c.get("tasks", [
            "labeled_reasoning", "analogical_transfer", "graph_completion",
        ])

        for task in phase_c_tasks:
            task_classes = get_max_classes(task)
            print(f"\n  Task: {task} ({task_classes} classes)")
            train_ds = BenchmarkDataset(
                bc.get("num_train", 500), task, train_n, emb_dim,
            )
            val_ds = BenchmarkDataset(
                bc.get("num_val", 100), task, train_n, emb_dim,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=tc["learning_rate"] * 0.25,
                weight_decay=tc.get("weight_decay", 0.01),
            )

            best_val_acc = 0.0
            for epoch in range(phase_c_epochs):
                t0 = time.time()
                loss, _, gn = train_epoch_with_gate_penalty(
                    model, train_ds, optimizer,
                    gate_penalty_weight=0.0,
                    label_smoothing=tc.get("label_smoothing", 0.1),
                    device=device,
                )
                from src.benchmarks.run_comparison import evaluate
                val_acc, val_loss = evaluate(model, val_ds, device=device)
                elapsed = time.time() - t0

                marker = ""
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    marker = " *"

                print(f"    Ep {epoch:3d} | loss {loss:.4f} | "
                      f"val {val_acc:.3f} | gn {gn:.2f} | {elapsed:.0f}s{marker}")

            all_results[f"phase_c_{task}"] = {"best_val_acc": best_val_acc}

    # ---- Diagnostics ----
    print(f"\n{'─' * 72}")
    print("Final Diagnostics")
    print(f"{'─' * 72}")

    diag_tasks = ["diverse", "graph_completion", "labeled_reasoning"]
    for task in diag_tasks:
        ds = BenchmarkDataset(50, task, train_n, emb_dim)
        collector = DiagnosticCollector()
        collector.collect(model, ds, device)
        summary = collector.summarize()
        gate_info = summary.get("llm_gate", {})
        conf_info = summary.get("confidence", {})
        print(f"  {task:25s} gate={gate_info.get('mean', 0):.3f}±{gate_info.get('std', 0):.3f} "
              f"conf={conf_info.get('mean', 0):.3f}")

    # Save results
    results_path = output_dir / "curriculum_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save model
    model_path = output_dir / "phase4c_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4c curriculum training")
    parser.add_argument("config", nargs="?", default="config/benchmark_4c_llm.yaml")
    args = parser.parse_args()
    run_curriculum(args.config)
