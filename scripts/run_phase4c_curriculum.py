"""Phase 4c curriculum training: three-phase LLM integration.

Phase A: Regression lock — train on existing tasks with semantic_weight penalty.
         Validates that LLM infrastructure doesn't break existing performance.
Phase B: Graph completion — remove gate penalty, let model discover LLM utility.
Phase C: Language + Analogy — full training on all tasks, no gate constraints.

Usage:
    python scripts/run_phase4c_curriculum.py [config_path]
    python scripts/run_phase4c_curriculum.py config/benchmark_4c_llm.yaml
    python scripts/run_phase4c_curriculum.py config/llama_training.yaml --pregenerated-dir data/llama_datasets
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


def _build_optimizer(model, lr: float, weight_decay: float = 0.01,
                     llm_lr_scale: float = 0.5, freeze_llm: bool = False):
    """Build optimizer with differential learning rates.

    GNN/TAT/classifier params get full LR; TopoBridge/LLM params get scaled LR.
    During Phase A (freeze_llm=True), LLM params are frozen entirely.
    """
    core_params = []
    llm_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'topo_bridge' in name or 'llm' in name:
            if freeze_llm:
                param.requires_grad = False
            else:
                llm_params.append(param)
        else:
            core_params.append(param)

    param_groups = [{'params': core_params, 'lr': lr}]
    if llm_params:
        param_groups.append({'params': llm_params, 'lr': lr * llm_lr_scale})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _unfreeze_llm(model):
    """Re-enable gradients on LLM params (after Phase A).

    Only unfreezes LoRA adapters, TopoCrossAttention, and TopoBridge encoder
    params. Frozen Llama base weights (under llama.*) stay frozen.
    """
    for name, param in model.named_parameters():
        if 'lora' in name or 'topo_cross_attn' in name:
            param.requires_grad = True
        elif 'topo_bridge' in name and 'llama' not in name:
            # TopoBridge encoder params (node_proj, prefix, extractor)
            # but NOT the nested Llama model weights
            param.requires_grad = True


def _semantic_weight_penalty(model, weight: float = 1.0) -> torch.Tensor:
    """Auxiliary loss that penalizes semantic_weight > 0.

    Encourages the model to keep the weight low during Phase A
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
    scaler=None, use_amp=False,
):
    """Training loop with optional semantic_weight penalty and AMP support.

    When gate_penalty_weight > 0, adds an auxiliary loss term that
    penalizes the semantic_weight value, encouraging the model to keep the
    LLM pathway closed.

    use_amp: enable bf16 autocast (reduces backward memory ~50%).
    """
    if device is None:
        device = torch.device('cpu')
    model.train()
    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()

    amp_enabled = (use_amp or scaler is not None) and device.type == 'cuda'
    total_loss = 0.0
    total_gate_loss = 0.0
    grad_norms = []
    optimizer.zero_grad()

    for i in range(len(dataset)):
        cc, query, target, answer, metadata = _unpack_sample(dataset[i])
        cc = cc.clone().to(device)

        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.bfloat16):
            logits = model(cc, query, target, metadata=metadata)
            ce_loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([answer], device=device),
                label_smoothing=label_smoothing,
            )

            # Gate penalty: penalize semantic_weight if weight > 0
            gate_loss = torch.tensor(0.0, device=device)
            if gate_penalty_weight > 0 and hasattr(model, 'executive_loop'):
                if hasattr(model, 'use_llm') and model.use_llm:
                    node_emb = cc.get_embeddings(0).to(device)
                    ctrl = model.executive_loop.gnn_executive.control_head(node_emb)
                    gate_loss = gate_penalty_weight * ctrl.semantic_weight

            loss = (ce_loss + gate_loss) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += ce_loss.item()
        total_gate_loss += gate_loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataset):
            if scaler is not None:
                scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            gn_val = gn.item() if isinstance(gn, torch.Tensor) else gn
            grad_norms.append(gn_val)
            if not (gn_val != gn_val or gn_val == float('inf')):
                if scaler is not None:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            if scaler is not None:
                scaler.update()
            optimizer.zero_grad()

    n = len(dataset)
    avg_loss = total_loss / n
    avg_gate_loss = total_gate_loss / n
    # Filter NaN/inf from grad norms (NaN steps are already skipped above)
    valid_norms = [g for g in grad_norms if g == g and g != float('inf')]
    avg_grad_norm = sum(valid_norms) / len(valid_norms) if valid_norms else 0.0
    return avg_loss, avg_gate_loss, avg_grad_norm


def _load_or_generate(
    pregenerated_dir: Path | None,
    task_type: str,
    split: str,
    num_samples: int,
    n_nodes: int,
    embedding_dim: int,
    topologies: list[str] | None = None,
    n_nodes_range: tuple[int, int] | None = None,
) -> BenchmarkDataset:
    """Load pregenerated dataset if available, else generate on-the-fly."""
    if pregenerated_dir is not None:
        range_str = (f"n{n_nodes_range[0]}-{n_nodes_range[1]}"
                     if n_nodes_range else f"n{n_nodes}")
        path = pregenerated_dir / f"{task_type}_{split}_{range_str}.pt"
        if path.exists():
            print(f"    Loading {path.name}...")
            ds = BenchmarkDataset.load(str(path))
            if len(ds) > num_samples:
                ds.samples = ds.samples[:num_samples]
            return ds
        else:
            print(f"    {path.name} not found, generating on-the-fly...")

    return BenchmarkDataset(
        num_samples, task_type, n_nodes, embedding_dim,
        topologies=topologies, n_nodes_range=n_nodes_range,
    )


def run_curriculum(config_path: str = "config/benchmark_4c_llm.yaml",
                   pregenerated_dir: str | None = None,
                   resume_phase: str | None = None):
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
    checkpoint_dir = Path(bc.get("checkpoint_dir", output_dir / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Mixed-size training range
    n_min = bc.get("train_n_nodes_min")
    n_max = bc.get("train_n_nodes_max")
    train_range = (n_min, n_max) if n_min is not None and n_max is not None else None

    # Topologies
    train_topos = bc.get("train_topologies")
    test_topos = bc.get("test_topologies")
    all_topos = ((train_topos or []) + (test_topos or [])) or None

    # Pregenerated data directory
    pregen_dir = Path(pregenerated_dir) if pregenerated_dir else None

    print("=" * 72)
    print("Phase 4c: LLM Integration Curriculum Training")
    print(f"Device: {device}")
    if pregen_dir:
        print(f"Pregenerated data: {pregen_dir}")
    if train_range:
        print(f"Mixed-size training: n={train_range[0]}-{train_range[1]}")
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

    # AMP with bf16 autocast: reduces backward pass peak memory ~50%.
    # Spectral ops (eigh, svd, lstsq) are not eligible for autocasting
    # and stay in their input dtype (fp32), so they work correctly.
    # GradScaler is NOT used with bf16 (only needed for fp16).
    use_amp = device.type == 'cuda'
    scaler = None

    # ---- Resume from checkpoint if requested ----
    skip_a = resume_phase in ("b", "B", "c", "C")
    skip_b = resume_phase in ("c", "C")
    if skip_a:
        # Load the last Phase A checkpoint to restore model weights
        phase_a_tasks = cc.get("phase_a", {}).get("tasks",
                        ["diverse", "bfs", "hodge_class", "spectral_gap"])
        last_task = phase_a_tasks[-1]
        ckpt_path = checkpoint_dir / f"phase_a_{last_task}.pt"
        if ckpt_path.exists():
            # Load to CPU first to avoid GPU OOM from fp32 checkpoint + bf16 model
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"  Resumed from checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: {ckpt_path} not found, starting from scratch")
        # Enable TopoBridge for Phase B/C — aggressively free memory first
        del state  # Free CPU checkpoint copy
        import gc; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model.bypass_llm = False
        if hasattr(model, 'topo_bridge') and model.topo_bridge is not None:
            model.topo_bridge.to(device)
            print("  TopoBridge moved to GPU")
        _unfreeze_llm(model)

        # Reset semantic gate bias (see comment in Phase A→B transition)
        for name, param in model.named_parameters():
            if 'semantic_weight_head.bias' in name:
                old_val = param.data.item()
                param.data.fill_(0.0)
                print(f"  Reset semantic gate bias: {old_val:.2f} -> 0.0 (sigmoid 0.05 -> 0.50)")
                break

    # ---- Phase A: Regression Lock ----
    phase_a = cc.get("phase_a", {})
    if phase_a.get("enabled", True) and not skip_a:
        print(f"\n{'─' * 72}")
        print("Phase A: Regression Lock (existing tasks, gate penalty)")
        print(f"{'─' * 72}")

        gate_penalty = phase_a.get("gate_penalty_weight", 1.0)
        model.bypass_llm = True  # Skip TopoBridge entirely during Phase A

        # Move Llama to CPU during Phase A to free VRAM
        if hasattr(model, 'topo_bridge') and model.topo_bridge is not None:
            model.topo_bridge.to('cpu')
            print("  TopoBridge moved to CPU (bypass_llm=True)")
        phase_a_tasks = phase_a.get("tasks", ["diverse", "bfs", "hodge_class", "spectral_gap"])
        phase_a_epochs = phase_a.get("epochs", tc["epochs"])

        # Generate datasets for phase A tasks
        for task in phase_a_tasks:
            task_classes = get_max_classes(task)
            print(f"\n  Task: {task} ({task_classes} classes)")
            train_ds = _load_or_generate(
                pregen_dir, task, "train",
                bc.get("num_train", 500), train_n, emb_dim,
                topologies=all_topos, n_nodes_range=train_range,
            )
            val_ds = _load_or_generate(
                pregen_dir, task, "val",
                bc.get("num_val", 100), train_n, emb_dim,
                topologies=all_topos, n_nodes_range=train_range,
            )

            # Phase A: freeze LLM, full LR for GNN/TAT core
            optimizer = _build_optimizer(
                model, lr=tc["learning_rate"],
                weight_decay=tc.get("weight_decay", 0.01),
                freeze_llm=True,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5,
            )

            best_val_acc = 0.0
            patience_counter = 0
            patience = tc.get("patience", 10)
            best_state = None
            for epoch in range(phase_a_epochs):
                t0 = time.time()
                loss, gate_loss, gn = train_epoch_with_gate_penalty(
                    model, train_ds, optimizer,
                    gate_penalty_weight=gate_penalty,
                    label_smoothing=tc.get("label_smoothing", 0.1),
                    accumulation_steps=tc.get("accumulation_steps", 4),
                    device=device,
                    scaler=scaler,
                    use_amp=use_amp,
                )
                # Quick eval
                from src.benchmarks.run_comparison import evaluate
                val_acc, val_loss = evaluate(model, val_ds, device=device)
                elapsed = time.time() - t0

                marker = ""
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    marker = " *"
                    # Save best checkpoint immediately
                    best_ckpt = checkpoint_dir / f"phase_a_{task}_best.pt"
                    torch.save(best_state, best_ckpt)
                else:
                    patience_counter += 1

                cur_lr = optimizer.param_groups[0]['lr']
                print(f"    Ep {epoch:3d} | loss {loss:.4f} gate_loss {gate_loss:.4f} | "
                      f"val {val_acc:.3f} | gn {gn:.2f} | lr {cur_lr:.6f} | {elapsed:.0f}s{marker}")
                scheduler.step(val_loss)

                # Rolling checkpoint every epoch (overwrites previous)
                rolling_ckpt = checkpoint_dir / f"phase_a_{task}_latest.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "patience_counter": patience_counter,
                }, rolling_ckpt)

                if best_val_acc >= 1.0 - 1e-6:
                    print(f"    Early stop: perfect val accuracy at epoch {epoch}")
                    break
                if patience_counter >= patience:
                    print(f"    Early stop: no improvement for {patience} epochs")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            all_results[f"phase_a_{task}"] = {"best_val_acc": best_val_acc}

            ckpt_path = checkpoint_dir / f"phase_a_{task}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")

        # Free Phase A memory before loading LLM onto GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"  VRAM freed: {torch.cuda.memory_reserved(device) / 1e9:.1f}GB reserved")

        # Unfreeze LLM params and enable TopoBridge for Phase B/C
        model.bypass_llm = False
        if hasattr(model, 'topo_bridge') and model.topo_bridge is not None:
            model.topo_bridge.to(device)
            print("  TopoBridge moved back to GPU")
        _unfreeze_llm(model)

        # Reset semantic gate bias so LLM has 50% influence at Phase B start.
        # Phase A penalized it to sigmoid(-3)=5%; without reset, bridge can't
        # bootstrap (cold-start trap: weak gate → weak gradients → gate stays weak).
        for name, param in model.named_parameters():
            if 'semantic_weight_head.bias' in name:
                old_val = param.data.item()
                param.data.fill_(0.0)  # sigmoid(0) = 0.5
                print(f"  Reset semantic gate bias: {old_val:.2f} -> 0.0 (sigmoid 0.05 -> 0.50)")
                break
        print("  Re-enabled DSM + unfroze params for Phase B")

    # ---- Phase B: Graph Completion ----
    phase_b = cc.get("phase_b", {})
    if phase_b.get("enabled", True) and not skip_b:
        print(f"\n{'─' * 72}")
        print("Phase B: Graph Completion (no gate penalty)")
        print(f"{'─' * 72}")

        phase_b_epochs = phase_b.get("epochs", tc["epochs"])
        phase_b_tasks = phase_b.get("tasks", ["graph_completion"])

        for task in phase_b_tasks:
            task_classes = get_max_classes(task)
            print(f"\n  Task: {task} ({task_classes} classes)")
            train_ds = _load_or_generate(
                pregen_dir, task, "train",
                bc.get("num_train", 500), train_n, emb_dim,
                topologies=all_topos, n_nodes_range=train_range,
            )
            val_ds = _load_or_generate(
                pregen_dir, task, "val",
                bc.get("num_val", 100), train_n, emb_dim,
                topologies=all_topos, n_nodes_range=train_range,
            )

            # Phase B: differential LR, LLM unfrozen
            optimizer = _build_optimizer(
                model, lr=tc["learning_rate"] * 0.5,
                weight_decay=tc.get("weight_decay", 0.01),
                llm_lr_scale=0.5, freeze_llm=False,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5,
            )

            best_val_acc = 0.0
            patience_counter = 0
            patience = tc.get("patience", 10)
            best_state = None
            for epoch in range(phase_b_epochs):
                t0 = time.time()
                loss, _, gn = train_epoch_with_gate_penalty(
                    model, train_ds, optimizer,
                    gate_penalty_weight=0.0,
                    label_smoothing=tc.get("label_smoothing", 0.1),
                    accumulation_steps=tc.get("accumulation_steps", 4),
                    device=device,
                    scaler=scaler,
                    use_amp=use_amp,
                )
                from src.benchmarks.run_comparison import evaluate
                val_acc, val_loss = evaluate(model, val_ds, device=device)
                elapsed = time.time() - t0

                marker = ""
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    marker = " *"
                    best_ckpt = checkpoint_dir / f"phase_b_{task}_best.pt"
                    torch.save(best_state, best_ckpt)
                else:
                    patience_counter += 1

                cur_lr = optimizer.param_groups[0]['lr']
                print(f"    Ep {epoch:3d} | loss {loss:.4f} | "
                      f"val {val_acc:.3f} | gn {gn:.2f} | lr {cur_lr:.6f} | {elapsed:.0f}s{marker}")
                scheduler.step(val_loss)

                rolling_ckpt = checkpoint_dir / f"phase_b_{task}_latest.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "patience_counter": patience_counter,
                }, rolling_ckpt)

                if best_val_acc >= 1.0 - 1e-6:
                    print(f"    Early stop: perfect val accuracy at epoch {epoch}")
                    break
                if patience_counter >= patience:
                    print(f"    Early stop: no improvement for {patience} epochs")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            all_results[f"phase_b_{task}"] = {"best_val_acc": best_val_acc}

            ckpt_path = checkpoint_dir / f"phase_b_{task}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")

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
            train_ds = _load_or_generate(
                pregen_dir, task, "train",
                bc.get("num_train", 500), train_n, emb_dim,
                topologies=all_topos, n_nodes_range=train_range,
            )
            val_ds = _load_or_generate(
                pregen_dir, task, "val",
                bc.get("num_val", 100), train_n, emb_dim,
                topologies=all_topos, n_nodes_range=train_range,
            )

            # Phase C: lower LR for fine-tuning, differential for LLM
            optimizer = _build_optimizer(
                model, lr=tc["learning_rate"] * 0.25,
                weight_decay=tc.get("weight_decay", 0.01),
                llm_lr_scale=1.0, freeze_llm=False,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5,
            )

            best_val_acc = 0.0
            patience_counter = 0
            patience = tc.get("patience", 10)
            best_state = None
            for epoch in range(phase_c_epochs):
                t0 = time.time()
                loss, _, gn = train_epoch_with_gate_penalty(
                    model, train_ds, optimizer,
                    gate_penalty_weight=0.0,
                    label_smoothing=tc.get("label_smoothing", 0.1),
                    accumulation_steps=tc.get("accumulation_steps", 4),
                    device=device,
                    scaler=scaler,
                    use_amp=use_amp,
                )
                from src.benchmarks.run_comparison import evaluate
                val_acc, val_loss = evaluate(model, val_ds, device=device)
                elapsed = time.time() - t0

                marker = ""
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    marker = " *"
                    best_ckpt = checkpoint_dir / f"phase_c_{task}_best.pt"
                    torch.save(best_state, best_ckpt)
                else:
                    patience_counter += 1

                cur_lr = optimizer.param_groups[0]['lr']
                print(f"    Ep {epoch:3d} | loss {loss:.4f} | "
                      f"val {val_acc:.3f} | gn {gn:.2f} | lr {cur_lr:.6f} | {elapsed:.0f}s{marker}")
                scheduler.step(val_loss)

                rolling_ckpt = checkpoint_dir / f"phase_c_{task}_latest.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "patience_counter": patience_counter,
                }, rolling_ckpt)

                if best_val_acc >= 1.0 - 1e-6:
                    print(f"    Early stop: perfect val accuracy at epoch {epoch}")
                    break
                if patience_counter >= patience:
                    print(f"    Early stop: no improvement for {patience} epochs")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            all_results[f"phase_c_{task}"] = {"best_val_acc": best_val_acc}

            ckpt_path = checkpoint_dir / f"phase_c_{task}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")

    # ---- Diagnostics ----
    print(f"\n{'─' * 72}")
    print("Final Diagnostics")
    print(f"{'─' * 72}")

    diag_tasks = ["diverse", "graph_completion", "labeled_reasoning"]
    for task in diag_tasks:
        ds = _load_or_generate(
            pregen_dir, task, "val", 50, train_n, emb_dim,
            topologies=all_topos, n_nodes_range=train_range,
        )
        collector = DiagnosticCollector()
        collector.collect(model, ds, device)
        summary = collector.summarize()
        gate_info = summary.get("semantic_weight", {})
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
    parser.add_argument("--pregenerated-dir", default=None,
                        help="Load pre-generated datasets from this directory")
    parser.add_argument("--resume-phase", default=None, choices=["b", "B", "c", "C"],
                        help="Skip earlier phases and resume from B or C (loads last checkpoint)")
    args = parser.parse_args()
    run_curriculum(args.config, pregenerated_dir=args.pregenerated_dir,
                   resume_phase=args.resume_phase)
