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
from src.training.batch_utils import train_epoch_batched, evaluate_batched
from src.training.class_weights import compute_class_weights
from src.training.contrastive_loss import SemanticContrastiveLoss
from src.training.feature_replay import FeatureReplayBuffer
from src.topology_observer import TopologyObserver

# Phase task lists
PHASE_A_TASKS = ["diverse", "bfs", "hodge_class", "spectral_gap", "path_counting"]
PHASE_B_TASKS = ["graph_completion", "labeled_reasoning"]
PHASE_C_TASKS = ["analogical_transfer", "graph_completion", "labeled_reasoning"]
PHASE_D_TASKS = ["kg_relation", "kg_concept", "kg_pathvalid", "kg_analogy", "kg_cluster"]


def _load_state_filtered(model, state_dict):
    """Load state dict, skipping keys with shape mismatches (e.g. num_tasks changed)."""
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(model_state[k].shape)}")
            continue
        filtered[k] = v
    if skipped:
        print(f"  Skipped {len(skipped)} size-mismatched keys:")
        for s in skipped:
            print(f"    {s}")
    model.load_state_dict(filtered, strict=False)


def _rebuild_classifier(model, task: str):
    """Rebuild classifier head for the current task's num_classes.

    Idempotent: only creates a new Linear layer if the output dimension
    doesn't match.  This prevents destroying trained weights when called
    repeatedly within the same task.
    """
    num_classes = get_max_classes(task)
    if model.classifier[-1].out_features == num_classes:
        return  # Already correct size — keep trained weights
    old_in = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(old_in, num_classes)
    model.classifier[-1].to(next(model.parameters()).device)


def _build_dsm_optimizer(model, config):
    """Build optimizer with 3 param groups: GNN/TAT, DSM/adapter, TopoBridge.

    Handles both Track 1 (DSM) and Track 2 (Qwen adapter) backends.
    """
    tc = config['training']
    gnn_tat_params = []
    dsm_params = []
    bridge_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'dsm' in name.lower() or 'distilled' in name.lower():
            dsm_params.append(param)
        elif ('adapter' in name.lower() or 'graph_former' in name.lower()
              or 'graphformer' in name.lower()):
            # Track 2: Qwen adapter params get adapter-specific LR
            dsm_params.append(param)
        elif 'topo_bridge' in name or 'bridge' in name:
            bridge_params.append(param)
        else:
            gnn_tat_params.append(param)

    # Use adapter_learning_rate for Track 2, dsm_learning_rate for Track 1
    semantic_lr = tc.get('adapter_learning_rate',
                         tc.get('dsm_learning_rate', 5e-4))

    param_groups = [
        {'params': gnn_tat_params, 'lr': tc['learning_rate']},
        {'params': dsm_params, 'lr': semantic_lr},
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
                label_smoothing=0.0, device=None, use_amp=False,
                topo_features=None, task=None,
                class_weights=None,
                contrastive_fn=None, contrastive_weight=0.0,
                feature_replay=None, current_task=None):
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
            logits = model(cc, query, target, metadata=metadata,
                           topo_features=topo_features, task=task)
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([answer], device=device),
                label_smoothing=label_smoothing,
                weight=class_weights,
            ) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Contrastive loss on semantic features
        if contrastive_fn is not None and hasattr(model, '_last_semantic_features'):
            sf = model._last_semantic_features
            adj = model._last_adjacency.to(sf.device)
            c_loss = contrastive_fn(sf, adj) * contrastive_weight / accumulation_steps
            loss = loss + c_loss

        loss.backward()
        total_loss += loss.item() * accumulation_steps

        # Feature replay: store combined features
        if feature_replay is not None and hasattr(model, '_last_combined'):
            feature_replay.save(current_task or task or '', model._last_combined, answer)

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataset):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if torch.isfinite(gn):
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(len(dataset), 1)


def train_epoch_with_replay(model, dataset, replay_samples, optimizer,
                            max_norm=5.0, accumulation_steps=4,
                            label_smoothing=0.0, device=None, use_amp=False,
                            main_task=None, topo_features=None,
                            class_weights=None,
                            contrastive_fn=None, contrastive_weight=0.0,
                            feature_replay=None):
    """Training loop that interleaves replay samples from earlier phases.

    replay_samples is a list of (sample, task_type) tuples. With multi-head
    classifier, uses task= parameter. Falls back to _rebuild_classifier
    for legacy single-head classifier.
    """
    if device is None:
        device = torch.device('cpu')
    model.train()
    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()

    use_multi_head = getattr(model, 'use_multi_head_classifier', False)
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

    # Track whether a replay sample changed the classifier away from main task
    on_replay_task = False

    step_count = 0
    for source, idx in schedule:
        if source == 'main':
            sample = dataset[idx]
            cc, query, target, answer, metadata = _unpack_sample(sample)
            current_task = main_task
            # Legacy path: restore classifier if replay changed it
            if not use_multi_head and on_replay_task and main_task is not None:
                _rebuild_classifier(model, main_task)
                on_replay_task = False
        else:
            sample, replay_task = replay_samples[idx]
            cc, query, target, answer, metadata = _unpack_sample(sample)
            current_task = replay_task
            # Legacy path: rebuild classifier for replay task
            if not use_multi_head:
                _rebuild_classifier(model, replay_task)
                on_replay_task = True

        cc = cc.clone().to(device)

        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.bfloat16):
            logits = model(cc, query, target, metadata=metadata,
                           topo_features=topo_features,
                           task=current_task if use_multi_head else None)
            # Only apply class_weights for main task samples — replay samples
            # have different class counts and would cause shape mismatch
            wt = class_weights if source == 'main' else None
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([answer], device=device),
                label_smoothing=label_smoothing,
                weight=wt,
            ) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            step_count += 1
            continue

        # Contrastive loss on semantic features
        if contrastive_fn is not None and hasattr(model, '_last_semantic_features'):
            sf = model._last_semantic_features
            adj = model._last_adjacency.to(sf.device)
            c_loss = contrastive_fn(sf, adj) * contrastive_weight / accumulation_steps
            loss = loss + c_loss

        loss.backward()
        total_loss += loss.item() * accumulation_steps

        # Feature replay: store combined features
        if feature_replay is not None and hasattr(model, '_last_combined'):
            feature_replay.save(current_task or '', model._last_combined, answer)

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
    **extra_kwargs,
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
        **extra_kwargs,
    )


def _feature_replay_step(model, feature_replay, replay_ratio, optimizer,
                         device, label_smoothing, max_norm, use_multi_head, task):
    """Classifier-only replay from feature buffer.

    Replays stored penultimate-layer features through the classifier head(s)
    for tasks OTHER than the current task. Cheap alternative to full forward
    passes for catastrophic forgetting prevention.
    """
    import torch.nn.functional as F
    model.train()
    for replay_task in feature_replay.tasks():
        if replay_task == task:
            continue  # skip current task
        items = feature_replay.sample_ratio(replay_task, replay_ratio)
        if not items:
            continue
        for feat, label in items:
            feat = feat.to(device)
            if use_multi_head and model.multi_head_classifier is not None:
                logits = model.multi_head_classifier(feat.unsqueeze(0), replay_task).squeeze(0)
            else:
                logits = model.classifier(feat)
            r_loss = F.cross_entropy(logits.unsqueeze(0),
                                     torch.tensor([label], device=device),
                                     label_smoothing=label_smoothing)
            r_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if torch.isfinite(gn):
            optimizer.step()
        optimizer.zero_grad()


def _run_phase(phase_name, tasks, model, config, device, pregen_dir,
               train_range, all_topos, checkpoint_dir, use_amp,
               phase_a_datasets=None, observer=None, conceptnet_graph=None):
    """Run a single curriculum phase (A, B, C, or D).

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
        observer: optional TopologyObserver for periodic analysis

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
    # Phase D per-task epoch override from config
    phase_d_epoch_map = config.get('phase_d', {}).get('epochs', {})
    patience = tc.get('patience', 10)
    label_smoothing = tc.get('label_smoothing', 0.1)
    accumulation_steps = tc.get('accumulation_steps', 4)
    max_norm = tc.get('max_norm', 5.0)
    use_batched = tc.get('use_batched', False)
    batch_size = tc.get('batch_size', 8)

    results = {}
    datasets = {}

    # Build replay samples if Phase A datasets provided
    replay_samples = []
    if phase_a_datasets:
        replay_samples = _create_replay_dataset(phase_a_datasets, replay_fraction=0.2)
        print(f"  Task replay: {len(replay_samples)} samples from Phase A")

    use_multi_head = getattr(model, 'use_multi_head_classifier', False)

    for task in tasks:
        task_classes = get_max_classes(task)
        print(f"\n  Task: {task} ({task_classes} classes)")

        # Per-task resume: skip if checkpoint already exists
        task_ckpt = checkpoint_dir / f"phase_{phase_name}_{task}.pt"
        if task_ckpt.exists():
            print(f"    Checkpoint exists: {task_ckpt} — loading and skipping")
            state = torch.load(task_ckpt, map_location=device, weights_only=True)
            _load_state_filtered(model, state)
            results[task] = -1.0  # unknown val from previous run
            continue

        # Legacy: rebuild single-head classifier for this task
        if not use_multi_head:
            _rebuild_classifier(model, task)

        # Load/generate datasets
        extra_gen_kwargs = {}
        if conceptnet_graph is not None and task.startswith("kg_"):
            extra_gen_kwargs['conceptnet_graph'] = conceptnet_graph
            # Phase D may specify custom sample counts
            phase_d_cfg = config.get('phase_d', {})
            train_samples = phase_d_cfg.get('kg_train_samples', bc.get('train_samples', 5000))
            val_samples = phase_d_cfg.get('kg_val_samples', bc.get('val_samples', 500))
        else:
            train_samples = bc.get('train_samples', 5000)
            val_samples = bc.get('val_samples', 500)

        train_ds = _load_or_generate(
            pregen_dir, task, "train",
            train_samples, train_n, emb_dim,
            topologies=all_topos, n_nodes_range=train_range,
            **extra_gen_kwargs,
        )
        val_ds = _load_or_generate(
            pregen_dir, task, "val",
            val_samples, train_n, emb_dim,
            topologies=all_topos, n_nodes_range=train_range,
            **extra_gen_kwargs,
        )
        datasets[task] = train_ds

        # Class weights for imbalanced tasks
        class_wt = None
        if tc.get('use_class_weights', False):
            labels = torch.tensor([s[3] if len(s) == 5 else s[-1]
                                   for s in train_ds.samples])
            class_wt = compute_class_weights(labels, task_classes).to(device)

        # Contrastive loss on semantic features
        contrastive_weight = tc.get('contrastive_loss_weight', 0.0)
        contrastive_fn = SemanticContrastiveLoss() if contrastive_weight > 0 else None

        # Feature replay buffer
        replay_ratio = tc.get('replay_ratio', 0.0)
        feature_replay = FeatureReplayBuffer() if replay_ratio > 0 else None

        # Per-task epoch count (Phase D tasks may have individual counts)
        task_epochs = phase_d_epoch_map.get(task, num_epochs)

        # Build per-component optimizer
        optimizer = _build_dsm_optimizer(model, config)

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=task_epochs,
        )

        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        topo_config = config.get('topology_observer', {})
        active_mode = topo_config.get('active_mode', False)

        # Gradual unfreeze config (Track 1)
        lc = config.get('llm', {})
        gradual_unfreeze = lc.get('gradual_unfreeze', False)
        freeze_epochs = lc.get('freeze_epochs', 5)
        unfreeze_top_n_epochs = lc.get('unfreeze_top_n_epochs', 15)
        unfreeze_top_n = lc.get('unfreeze_top_n', 4)
        dsm_backend = None
        if gradual_unfreeze and hasattr(model, 'executive_loop'):
            tb = getattr(model.executive_loop, 'topo_bridge', None)
            if tb is not None:
                dsm_backend = getattr(tb, 'backend', None)

        for epoch in range(task_epochs):
            t0 = time.time()

            # Gradual unfreeze: freeze → partial → full
            if dsm_backend is not None and gradual_unfreeze:
                if epoch == 0:
                    dsm_backend.freeze_all()
                    print(f"    [UNFREEZE] DSM frozen (epoch 0-{freeze_epochs-1})")
                elif epoch == freeze_epochs:
                    dsm_backend.unfreeze_top_n(unfreeze_top_n)
                    optimizer = _build_dsm_optimizer(model, config)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=task_epochs - epoch,
                    )
                    print(f"    [UNFREEZE] Top {unfreeze_top_n} layers unfrozen")
                elif epoch == unfreeze_top_n_epochs:
                    dsm_backend.unfreeze_all()
                    optimizer = _build_dsm_optimizer(model, config)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=task_epochs - epoch,
                    )
                    print(f"    [UNFREEZE] All DSM layers unfrozen")

            # Active mode: get cached topo_features from observer
            topo_feat = None
            if active_mode and observer is not None:
                topo_feat = observer.get_topo_features()

            task_arg = task if use_multi_head else None

            if use_batched and not replay_samples:
                # Batched training (4-8x GPU utilization improvement)
                loss = train_epoch_batched(
                    model, train_ds, optimizer,
                    batch_size=batch_size, max_norm=max_norm,
                    label_smoothing=label_smoothing,
                    device=device, use_amp=use_amp,
                    task=task_arg, topo_features=topo_feat,
                    class_weights=class_wt,
                    contrastive_fn=contrastive_fn,
                    contrastive_weight=contrastive_weight,
                )
            elif replay_samples:
                # Replay with interleaving (sequential — replay mixes tasks)
                if not use_multi_head:
                    _rebuild_classifier(model, task)
                loss = train_epoch_with_replay(
                    model, train_ds, replay_samples, optimizer,
                    max_norm=max_norm,
                    accumulation_steps=accumulation_steps,
                    label_smoothing=label_smoothing,
                    device=device,
                    use_amp=use_amp,
                    main_task=task,
                    topo_features=topo_feat,
                    class_weights=class_wt,
                    contrastive_fn=contrastive_fn,
                    contrastive_weight=contrastive_weight,
                    feature_replay=feature_replay,
                )
            else:
                loss = train_epoch(
                    model, train_ds, optimizer,
                    max_norm=max_norm,
                    accumulation_steps=accumulation_steps,
                    label_smoothing=label_smoothing,
                    device=device,
                    use_amp=use_amp,
                    topo_features=topo_feat,
                    task=task_arg,
                    class_weights=class_wt,
                    contrastive_fn=contrastive_fn,
                    contrastive_weight=contrastive_weight,
                    feature_replay=feature_replay,
                    current_task=task,
                )

            # Feature replay: classifier-only rehearsal on previous tasks
            if feature_replay is not None and feature_replay.tasks():
                _feature_replay_step(model, feature_replay, replay_ratio,
                                     optimizer, device, label_smoothing,
                                     max_norm, use_multi_head, task)

            # Evaluation
            if not use_multi_head:
                _rebuild_classifier(model, task)
            if use_batched:
                val_acc, val_loss = evaluate_batched(
                    model, val_ds, batch_size=batch_size,
                    device=device, task=task_arg,
                )
            else:
                val_acc, val_loss = evaluate(
                    model, val_ds, device=device,
                    task=task_arg,
                )
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

            # Topology observer: periodic analysis
            if observer is not None and observer.should_analyze(epoch):
                sample = train_ds[0]
                topo_results = observer.run_analysis(epoch, sample)
                cg = topo_results.get('comp_graph', {})
                emb = topo_results.get('embedding', {})
                if 'error' not in cg:
                    print(f"    [TOPO] gap={cg.get('spectral_gap', 0):.4f} "
                          f"grad={cg.get('gradient_energy_ratio', 0):.3f} "
                          f"curl={cg.get('curl_energy_ratio', 0):.3f} "
                          f"harm={cg.get('harmonic_energy_ratio', 0):.3f} "
                          f"ops={cg.get('num_operations', 0)} "
                          f"flows={cg.get('num_data_flows', 0)}")
                else:
                    print(f"    [TOPO ERR comp_graph] {cg.get('error', 'unknown')}")
                if 'error' in emb:
                    print(f"    [TOPO ERR embedding] {emb.get('error', 'unknown')}")

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
    # Load pre-trained DSM weights if available (Track 1)
    pretrained_path = lc.get('pretrained_path')
    if pretrained_path and Path(pretrained_path).exists():
        dsm_backend = None
        if hasattr(model, 'executive_loop') and model.executive_loop.topo_bridge is not None:
            dsm_backend = model.executive_loop.topo_bridge.backend
        if dsm_backend is not None and hasattr(dsm_backend, 'load_pretrained'):
            state = torch.load(pretrained_path, map_location=device, weights_only=True)
            dsm_backend.load_pretrained(state)
            print(f"  Loaded pre-trained DSM from {pretrained_path}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # bf16 autocast causes NaN in spectral/ODE ops — disable until we add
    # per-op exclusions or GradScaler.  fp32 is ~2x slower but actually learns.
    use_amp = False
    all_results = {}

    # Topology observer: periodic analysis of DSM health + computation graph
    topo_config = config.get('topology_observer', {})
    observer = None
    if topo_config.get('enabled', False):
        criterion = torch.nn.CrossEntropyLoss()
        dsm = None
        if (model.executive_loop.topo_bridge is not None
                and hasattr(model.executive_loop.topo_bridge, 'backend')
                and hasattr(model.executive_loop.topo_bridge.backend, 'dsm')):
            dsm = model.executive_loop.topo_bridge.backend.dsm
        observer = TopologyObserver(model, criterion, dsm=dsm, config=topo_config)
        print(f"  Topology observer enabled (every {topo_config.get('analyze_every', 5)} epochs)"
              f"{', active mode' if topo_config.get('active_mode', False) else ''}")

    # ---- Resume from checkpoint if requested ----
    skip_a = resume_phase in ("b", "B", "c", "C", "d", "D")
    skip_b = resume_phase in ("c", "C", "d", "D")
    skip_c = resume_phase in ("d", "D")

    if skip_a:
        last_task = PHASE_A_TASKS[-1]
        ckpt_path = checkpoint_dir / f"phase_a_{last_task}.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            _load_state_filtered(model, state)
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
            _load_state_filtered(model, state)
            del state
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  Resumed from checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: {ckpt_path} not found, starting from scratch")

    if skip_c:
        # Try phase_d resume_from config first, then last Phase C checkpoint
        phase_d_cfg = config.get('phase_d', {})
        resume_from = phase_d_cfg.get('resume_from', None)
        if resume_from and Path(resume_from).exists():
            ckpt_path = Path(resume_from)
        else:
            last_task = PHASE_C_TASKS[-1]
            ckpt_path = checkpoint_dir / f"phase_c_{last_task}.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            _load_state_filtered(model, state)
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
        print("Phase A: Structural Foundation (DSM frozen, GNN/TAT learns)")
        print(f"{'=' * 72}")

        # Disable DSM entirely during Phase A — skip the 205M-param forward
        # pass (runs 5x per sample in the executive loop).  Phase A is pure
        # structural tasks; DSM contributes nothing (semantic_weight ≈ 0.05).
        # Also freeze params so backward skips them too.
        model.executive_loop.use_dsm = False
        frozen_params = []
        for name, param in model.named_parameters():
            if 'topo_bridge' in name or 'dsm' in name.lower():
                param.requires_grad_(False)
                frozen_params.append(name)
        print(f"  Disabled DSM forward + frozen {len(frozen_params)} param tensors")

        phase_a_results, phase_a_datasets = _run_phase(
            'a', PHASE_A_TASKS, model, config, device, pregen_dir,
            train_range, all_topos, checkpoint_dir, use_amp,
            observer=observer,
        )

        # Re-enable DSM + unfreeze for subsequent phases
        model.executive_loop.use_dsm = True
        for name, param in model.named_parameters():
            if 'topo_bridge' in name or 'dsm' in name.lower():
                param.requires_grad_(True)
        print(f"  Re-enabled DSM + unfroze params for Phase B")
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
            observer=observer,
        )
        for task, acc in phase_b_results.items():
            all_results[f"phase_b_{task}"] = {"best_val_acc": acc}

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- Phase C: Full Symbiosis ----
    if not skip_c:
        print(f"\n{'=' * 72}")
        print("Phase C: Full Symbiosis (all tasks + 20% replay)")
        print(f"{'=' * 72}")

        phase_c_results, _ = _run_phase(
            'c', PHASE_C_TASKS, model, config, device, pregen_dir,
            train_range, all_topos, checkpoint_dir, use_amp,
            phase_a_datasets=phase_a_datasets if phase_a_datasets else None,
            observer=observer,
        )
        for task, acc in phase_c_results.items():
            all_results[f"phase_c_{task}"] = {"best_val_acc": acc}

    # ---- Phase D: Meta-Cognition (KG tasks) ----
    phase_d_config = config.get("phase_d", {})
    phase_d_tasks = phase_d_config.get("tasks", [])
    if phase_d_tasks:
        print(f"\n{'=' * 72}")
        print("Phase D: Meta-Cognition (ConceptNet KG tasks)")
        print(f"{'=' * 72}")

        # Load ConceptNet graph
        cn_path = bc.get("conceptnet_path", "data/conceptnet/conceptnet_en.pkl")
        conceptnet_graph = None
        if Path(cn_path).exists():
            from src.data.conceptnet import load_cached_graph
            conceptnet_graph = load_cached_graph(cn_path)
            print(f"  Loaded ConceptNet: {conceptnet_graph.number_of_nodes():,} nodes, "
                  f"{conceptnet_graph.number_of_edges():,} edges")
        else:
            print(f"  WARNING: ConceptNet not found at {cn_path}")
            print(f"  Run: python scripts/precompute_conceptnet.py")

        if conceptnet_graph is not None:
            # Ensure DSM/LLM is enabled for Phase D
            if hasattr(model, 'executive_loop'):
                model.executive_loop.use_dsm = True

            phase_d_results, _ = _run_phase(
                'd', phase_d_tasks, model, config, device, pregen_dir,
                train_range, all_topos, checkpoint_dir, use_amp,
                phase_a_datasets=phase_a_datasets if phase_a_datasets else None,
                observer=observer,
                conceptnet_graph=conceptnet_graph,
            )
            for task, acc in phase_d_results.items():
                all_results[f"phase_d_{task}"] = {"best_val_acc": acc}

    # ---- Diagnostics ----
    print(f"\n{'=' * 72}")
    print("Final Diagnostics")
    print(f"{'=' * 72}")

    emb_dim = mc["embedding_dim"]
    train_n = bc["train_n_nodes"]
    use_multi_head = getattr(model, 'use_multi_head_classifier', False)
    diag_tasks = ["diverse", "graph_completion", "labeled_reasoning"]
    for task in diag_tasks:
        if not use_multi_head:
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
    parser.add_argument("--resume-phase", default=None, choices=["b", "B", "c", "C", "d", "D"],
                        help="Skip earlier phases and resume from B, C, or D (loads last checkpoint)")
    args = parser.parse_args()
    run_curriculum(args.config, pregenerated_dir=args.pregenerated_dir,
                   resume_phase=args.resume_phase)
