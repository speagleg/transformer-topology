"""DSM curriculum training: three-phase training for DSM-GNN symbiosis.

Phase A: Structural Foundation -- GNN/TAT + DSM on structural tasks (30 epochs)
         Tasks: diverse, bfs, hodge_class, spectral_gap, path_counting
         DSM is active but learns NOT to interfere on structural tasks.

Phase B: Semantic Integration -- Add semantic tasks with 20% replay (30 epochs)
         Tasks: graph_completion, labeled_reasoning + 20% Phase A replay

Phase C: Full Symbiosis -- All tasks mixed (30 epochs)
         Tasks: All tasks including analogical_transfer + 20% replay

Usage:
    python scripts/run_dsm_curriculum.py config/dsm_training.yaml
    python scripts/run_dsm_curriculum.py config/dsm_training.yaml --pregenerated-dir data/dsm_datasets
    python scripts/run_dsm_curriculum.py config/dsm_training.yaml --resume-phase b
"""

import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
from src.benchmarks.diagnostics import DiagnosticCollector
from src.benchmarks.run_benchmark_suite import _build_model, _resolve_device
from src.benchmarks.run_comparison import evaluate, _unpack_sample

# Phase task lists
PHASE_A_TASKS = ["diverse", "bfs", "hodge_class", "spectral_gap", "path_counting"]
PHASE_B_TASKS = ["graph_completion", "labeled_reasoning"]
PHASE_C_TASKS = ["analogical_transfer", "graph_completion", "labeled_reasoning"]


def _rebuild_classifier(model, task: str):
    """Rebuild classifier head for the current task's num_classes."""
    num_classes = get_max_classes(task)
    old_in = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(old_in, num_classes)
    model.classifier[-1].to(next(model.parameters()).device)


def _build_dsm_optimizer(model, config):
    """Build optimizer with 3 param groups: GNN/TAT, DSM, TopoBridge."""
    tc = config['training']
    gnn_tat_params = []
    dsm_params = []
    bridge_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'dsm' in name.lower() or 'distilled' in name.lower():
            dsm_params.append(param)
        elif 'topo_bridge' in name or 'bridge' in name:
            bridge_params.append(param)
        else:
            gnn_tat_params.append(param)

    param_groups = [
        {'params': gnn_tat_params, 'lr': tc['learning_rate']},
        {'params': dsm_params, 'lr': tc.get('dsm_learning_rate', 5e-4)},
        {'params': bridge_params, 'lr': tc.get('bridge_learning_rate', 1e-4)},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=tc.get('weight_decay', 0.01))


def _create_replay_dataset(phase_a_datasets, replay_fraction=0.2, target_size=None):
    """Sample replay_fraction of Phase A datasets for task replay.

    Returns a list of (sample, task_type) tuples so the training loop
    can rebuild the classifier for each replay sample's task.
    """
    replay_samples = []
    for task_type, ds in phase_a_datasets.items():
        n_replay = int(len(ds) * replay_fraction)
        indices = random.sample(range(len(ds)), min(n_replay, len(ds)))
        for idx in indices:
            replay_samples.append((ds.samples[idx], task_type))
    random.shuffle(replay_samples)
    if target_size is not None and len(replay_samples) > target_size:
        replay_samples = replay_samples[:target_size]
    return replay_samples


def train_epoch(model, dataset, optimizer, max_norm=5.0, accumulation_steps=4,
                label_smoothing=0.0, device=None, use_amp=False):
    """Standard training loop -- no gate penalty."""
    if device is None:
        device = torch.device('cpu')
    model.train()
    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()

    amp_enabled = use_amp and device.type == 'cuda'
    total_loss = 0.0
    optimizer.zero_grad()

    for i in range(len(dataset)):
        cc, query, target, answer, metadata = _unpack_sample(dataset[i])
        cc = cc.clone().to(device)

        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.bfloat16):
            logits = model(cc, query, target, metadata=metadata)
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([answer], device=device),
                label_smoothing=label_smoothing,
            ) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        total_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataset):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if torch.isfinite(gn):
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(len(dataset), 1)


def train_epoch_with_replay(model, dataset, replay_samples, optimizer,
                            max_norm=5.0, accumulation_steps=4,
                            label_smoothing=0.0, device=None, use_amp=False):
    """Training loop that interleaves replay samples from earlier phases.

    replay_samples is a list of (sample, task_type) tuples. For each replay
    sample, the classifier is temporarily rebuilt for that task.
    """
    if device is None:
        device = torch.device('cpu')
    model.train()
    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()

    amp_enabled = use_amp and device.type == 'cuda'
    total_loss = 0.0
    optimizer.zero_grad()

    # Interleave: main dataset samples + replay samples
    main_indices = list(range(len(dataset)))
    replay_indices = list(range(len(replay_samples)))
    random.shuffle(replay_indices)

    # Build combined schedule: (source, index) tuples
    schedule = [('main', i) for i in main_indices]
    schedule.extend([('replay', i) for i in replay_indices])
    random.shuffle(schedule)

    # Remember the current task's num_classes to avoid unnecessary rebuilds
    current_task = None
    main_task = None  # Will be set from dataset metadata or left as None

    step_count = 0
    for source, idx in schedule:
        if source == 'main':
            sample = dataset[idx]
            cc, query, target, answer, metadata = _unpack_sample(sample)
        else:
            sample, replay_task = replay_samples[idx]
            cc, query, target, answer, metadata = _unpack_sample(sample)
            # Rebuild classifier for replay task if needed
            if replay_task != current_task:
                _rebuild_classifier(model, replay_task)
                current_task = replay_task

        cc = cc.clone().to(device)

        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.bfloat16):
            logits = model(cc, query, target, metadata=metadata)
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([answer], device=device),
                label_smoothing=label_smoothing,
            ) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            step_count += 1
            continue

        loss.backward()
        total_loss += loss.item() * accumulation_steps
        step_count += 1

        if step_count % accumulation_steps == 0 or step_count == len(schedule):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if torch.isfinite(gn):
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(step_count, 1)


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


def _run_phase(phase_name, tasks, model, config, device, pregen_dir,
               train_range, all_topos, checkpoint_dir, use_amp,
               phase_a_datasets=None):
    """Run a single curriculum phase (A, B, or C).

    Args:
        phase_name: 'a', 'b', or 'c'
        tasks: list of task type strings
        model: the model to train
        config: full config dict
        device: torch device
        pregen_dir: optional Path to pregenerated data
        train_range: (min_n, max_n) tuple or None
        all_topos: list of topology names or None
        checkpoint_dir: Path for checkpoints
        use_amp: whether to use bf16 autocast
        phase_a_datasets: dict of {task: BenchmarkDataset} from Phase A (for replay)

    Returns:
        dict of {task: best_val_acc}, dict of {task: BenchmarkDataset} for datasets
    """
    tc = config['training']
    bc = config['benchmark']
    mc = config['model']
    emb_dim = mc['embedding_dim']
    train_n = bc['train_n_nodes']

    epochs_key = f'epochs_phase_{phase_name}'
    num_epochs = tc.get(epochs_key, 30)
    patience = tc.get('patience', 10)
    label_smoothing = tc.get('label_smoothing', 0.1)
    accumulation_steps = tc.get('accumulation_steps', 4)
    max_norm = tc.get('max_norm', 5.0)

    results = {}
    datasets = {}

    # Build replay samples if Phase A datasets provided
    replay_samples = []
    if phase_a_datasets:
        replay_samples = _create_replay_dataset(phase_a_datasets, replay_fraction=0.2)
        print(f"  Task replay: {len(replay_samples)} samples from Phase A")

    for task in tasks:
        task_classes = get_max_classes(task)
        print(f"\n  Task: {task} ({task_classes} classes)")

        # Rebuild classifier for this task
        _rebuild_classifier(model, task)

        # Load/generate datasets
        train_ds = _load_or_generate(
            pregen_dir, task, "train",
            bc.get('train_samples', 5000), train_n, emb_dim,
            topologies=all_topos, n_nodes_range=train_range,
        )
        val_ds = _load_or_generate(
            pregen_dir, task, "val",
            bc.get('val_samples', 500), train_n, emb_dim,
            topologies=all_topos, n_nodes_range=train_range,
        )
        datasets[task] = train_ds

        # Build per-component optimizer
        optimizer = _build_dsm_optimizer(model, config)

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs,
        )

        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(num_epochs):
            t0 = time.time()

            if replay_samples:
                # Rebuild classifier for main task before training
                _rebuild_classifier(model, task)
                loss = train_epoch_with_replay(
                    model, train_ds, replay_samples, optimizer,
                    max_norm=max_norm,
                    accumulation_steps=accumulation_steps,
                    label_smoothing=label_smoothing,
                    device=device,
                    use_amp=use_amp,
                )
            else:
                loss = train_epoch(
                    model, train_ds, optimizer,
                    max_norm=max_norm,
                    accumulation_steps=accumulation_steps,
                    label_smoothing=label_smoothing,
                    device=device,
                    use_amp=use_amp,
                )

            # Ensure classifier matches val task before evaluation
            _rebuild_classifier(model, task)
            val_acc, val_loss = evaluate(model, val_ds, device=device)
            elapsed = time.time() - t0

            marker = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                marker = " *"
                # Save best checkpoint immediately
                best_ckpt = checkpoint_dir / f"phase_{phase_name}_{task}_best.pt"
                torch.save(best_state, best_ckpt)
            else:
                patience_counter += 1

            cur_lrs = [g['lr'] for g in optimizer.param_groups]
            lr_str = "/".join(f"{lr:.6f}" for lr in cur_lrs)
            print(f"    Ep {epoch:3d} | loss {loss:.4f} | "
                  f"val {val_acc:.3f} | lr {lr_str} | {elapsed:.0f}s{marker}")
            scheduler.step()

            # Rolling checkpoint
            rolling_ckpt = checkpoint_dir / f"phase_{phase_name}_{task}_latest.pt"
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

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
        results[task] = best_val_acc

        # Save phase checkpoint
        ckpt_path = checkpoint_dir / f"phase_{phase_name}_{task}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"    Checkpoint saved: {ckpt_path}")

    return results, datasets


def run_curriculum(config_path: str = "config/dsm_training.yaml",
                   pregenerated_dir: str | None = None,
                   resume_phase: str | None = None):
    """Run the three-phase DSM curriculum training."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    wc = config.get("wave")
    lc = config.get("llm")

    device = _resolve_device(tc)
    output_dir = Path(bc.get("output_dir", "data/dsm_results"))
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
    print("DSM Curriculum Training: Three-Phase DSM-GNN Symbiosis")
    print(f"Device: {device}")
    if pregen_dir:
        print(f"Pregenerated data: {pregen_dir}")
    if train_range:
        print(f"Mixed-size training: n={train_range[0]}-{train_range[1]}")
    print("=" * 72)

    # Build model -- use initial max_classes from the first task, will be rebuilt per-task
    initial_max_classes = get_max_classes(PHASE_A_TASKS[0])
    model = _build_model("hierarchical_llm", mc, initial_max_classes, device,
                          wave_config=wc, llm_config=lc)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    use_amp = device.type == 'cuda'
    all_results = {}

    # ---- Resume from checkpoint if requested ----
    skip_a = resume_phase in ("b", "B", "c", "C")
    skip_b = resume_phase in ("c", "C")

    if skip_a:
        last_task = PHASE_A_TASKS[-1]
        ckpt_path = checkpoint_dir / f"phase_a_{last_task}.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=False)
            del state
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  Resumed from checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: {ckpt_path} not found, starting from scratch")

    if skip_b:
        last_task = PHASE_B_TASKS[-1]
        ckpt_path = checkpoint_dir / f"phase_b_{last_task}.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=False)
            del state
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  Resumed from checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: {ckpt_path} not found, starting from scratch")

    # ---- Phase A: Structural Foundation ----
    phase_a_datasets = {}
    if not skip_a:
        print(f"\n{'=' * 72}")
        print("Phase A: Structural Foundation (DSM active, learns not to interfere)")
        print(f"{'=' * 72}")

        phase_a_results, phase_a_datasets = _run_phase(
            'a', PHASE_A_TASKS, model, config, device, pregen_dir,
            train_range, all_topos, checkpoint_dir, use_amp,
        )
        for task, acc in phase_a_results.items():
            all_results[f"phase_a_{task}"] = {"best_val_acc": acc}

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- Phase B: Semantic Integration ----
    if not skip_b:
        print(f"\n{'=' * 72}")
        print("Phase B: Semantic Integration (graph_completion + labeled_reasoning + 20% replay)")
        print(f"{'=' * 72}")

        phase_b_results, phase_b_datasets = _run_phase(
            'b', PHASE_B_TASKS, model, config, device, pregen_dir,
            train_range, all_topos, checkpoint_dir, use_amp,
            phase_a_datasets=phase_a_datasets if phase_a_datasets else None,
        )
        for task, acc in phase_b_results.items():
            all_results[f"phase_b_{task}"] = {"best_val_acc": acc}

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- Phase C: Full Symbiosis ----
    print(f"\n{'=' * 72}")
    print("Phase C: Full Symbiosis (all tasks + 20% replay)")
    print(f"{'=' * 72}")

    phase_c_results, _ = _run_phase(
        'c', PHASE_C_TASKS, model, config, device, pregen_dir,
        train_range, all_topos, checkpoint_dir, use_amp,
        phase_a_datasets=phase_a_datasets if phase_a_datasets else None,
    )
    for task, acc in phase_c_results.items():
        all_results[f"phase_c_{task}"] = {"best_val_acc": acc}

    # ---- Diagnostics ----
    print(f"\n{'=' * 72}")
    print("Final Diagnostics")
    print(f"{'=' * 72}")

    emb_dim = mc["embedding_dim"]
    train_n = bc["train_n_nodes"]
    diag_tasks = ["diverse", "graph_completion", "labeled_reasoning"]
    for task in diag_tasks:
        _rebuild_classifier(model, task)
        ds = _load_or_generate(
            pregen_dir, task, "val", 50, train_n, emb_dim,
            topologies=all_topos, n_nodes_range=train_range,
        )
        collector = DiagnosticCollector()
        collector.collect(model, ds, device)
        summary = collector.summarize()
        gate_info = summary.get("semantic_weight", {})
        conf_info = summary.get("confidence", {})
        print(f"  {task:25s} gate={gate_info.get('mean', 0):.3f}+/-{gate_info.get('std', 0):.3f} "
              f"conf={conf_info.get('mean', 0):.3f}")

    # Save results
    results_path = output_dir / "dsm_curriculum_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save final model
    model_path = output_dir / "dsm_model_final.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSM curriculum training")
    parser.add_argument("config", nargs="?", default="config/dsm_training.yaml")
    parser.add_argument("--pregenerated-dir", default=None,
                        help="Load pre-generated datasets from this directory")
    parser.add_argument("--resume-phase", default=None, choices=["b", "B", "c", "C"],
                        help="Skip earlier phases and resume from B or C (loads last checkpoint)")
    args = parser.parse_args()
    run_curriculum(args.config, pregenerated_dir=args.pregenerated_dir,
                   resume_phase=args.resume_phase)
