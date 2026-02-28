"""Batched training utilities for graph-level mini-batching.

Key optimization: instead of processing one graph at a time through the
205M-param DSM, we batch B graphs' prefix tokens into a single DSM forward
pass. The GNN and TAT operate on small graphs (20-30 nodes) and are fast
enough to run per-graph, so we only batch the expensive DSM call.

Usage:
    from src.training.batch_utils import train_epoch_batched, evaluate_batched

    loss = train_epoch_batched(
        model, dataset, optimizer, batch_size=8,
        device=device, use_amp=True,
    )
    acc, val_loss, bal_acc = evaluate_batched(model, dataset, batch_size=8, device=device)
"""

import random
import torch
import torch.nn as nn
from src.benchmarks.run_comparison import _unpack_sample
from src.training.focal_loss import focal_loss


def graph_collate_fn(samples):
    """Collate function for DataLoader: returns list of unpacked samples."""
    return [_unpack_sample(s) for s in samples]


def _forward_batch(model, batch, device, task=None, topo_features=None):
    """Forward pass for a batch of samples, using batched DSM when available.

    Processes a list of (cc, query, target, answer, metadata) tuples.
    Returns list of (logits, answer) pairs.

    Also sets model._last_semantic_features, model._last_adjacency, and
    model._last_combined from the LAST graph in the batch (for contrastive
    loss and feature replay).

    Args:
        task: Optional task name for multi-head classifier.
        topo_features: Optional topology features tensor for ControlHead.

    Uses model.executive_loop.forward_batched() when the model has DSM enabled,
    otherwise falls back to sequential per-graph forward passes.
    """
    results = []

    # Check if the model supports batched DSM forward
    has_batched_dsm = (
        hasattr(model, 'executive_loop')
        and hasattr(model.executive_loop, 'forward_batched')
        and model.executive_loop.use_dsm
        and model.executive_loop.topo_bridge is not None
        and not getattr(model, 'bypass_llm', False)
    )

    if has_batched_dsm:
        # Batched path: run executive loop on all graphs simultaneously
        ccs = []
        queries = []
        targets = []
        answers = []
        metadatas = []

        for cc, query, target, answer, metadata in batch:
            cc = cc.clone().to(device)
            ccs.append(cc)
            queries.append(query)
            targets.append(target)
            answers.append(answer)
            metadatas.append(metadata)

        # Capture initial edge embeddings for hodge features (per-graph)
        initial_edge_embs = []
        for cc in ccs:
            if cc.num_cells(1) > 0:
                initial_edge_embs.append(cc.get_embeddings(1).clone())
            else:
                initial_edge_embs.append(None)

        # Build per-graph topo_features list if provided
        topo_features_list = None
        if topo_features is not None:
            topo_features_list = [topo_features] * len(ccs)

        # Batched executive loop (GNN per-graph, DSM batched, TAT per-graph)
        loop_results = model.executive_loop.forward_batched(
            ccs, topo_features_list=topo_features_list,
        )

        # Per-graph classifier
        for g, (output, num_iters, diagnostics) in enumerate(loop_results):
            # Capture semantic features from DSM diagnostics
            if 'semantic_features' in diagnostics:
                model._last_semantic_features = diagnostics['semantic_features']
                model._last_adjacency = ccs[g].adjacency_matrix(0)

            # LLM integration at model level
            text_features = None
            if (model.use_llm and model.topo_bridge is not None
                    and not model.bypass_llm):
                if hasattr(model.topo_bridge, 'extract_text_features'):
                    # Approach C (Qwen): per-node text features direct to classifier
                    node_texts = getattr(ccs[g], 'node_texts', None) or None
                    if node_texts:
                        text_features = model.topo_bridge.extract_text_features(
                            node_texts, output.device,
                        )
                else:
                    # Legacy blend (mock/llama)
                    control_signals = diagnostics.get('control_signals', [])
                    if control_signals:
                        semantic_weight = control_signals[-1].semantic_weight
                    else:
                        semantic_weight = torch.tensor(0.0, device=output.device)
                    task_text = None
                    if metadatas[g] and 'task_prompt' in metadatas[g]:
                        task_text = metadatas[g]['task_prompt']
                    node_texts = getattr(ccs[g], 'node_texts', None) or None
                    llm_out, _, sem_feat, _ = model.topo_bridge(
                        output, semantic_weight, task_text, node_texts=node_texts,
                    )
                    model._last_semantic_features = sem_feat
                    model._last_adjacency = ccs[g].adjacency_matrix(0)
                    output = ((1 - semantic_weight).unsqueeze(-1) * output
                              + semantic_weight.unsqueeze(-1) * llm_out)

            query_emb = output[queries[g]]
            target_emb = output[targets[g]]
            diff_emb = query_emb - target_emb
            dev = query_emb.device

            hodge_features = model._compute_hodge_features(
                ccs[g], initial_edge_embs=initial_edge_embs[g],
            ).to(dev)
            wave_energy = model._compute_wave_energy(diagnostics).to(dev)
            persistence_features = model._compute_persistence_features(ccs[g]).to(dev)

            combined = torch.cat([query_emb, target_emb, diff_emb,
                                  hodge_features, wave_energy, persistence_features])

            # Approach C: concatenate text features for query/target/diff
            text_feat_dim = getattr(model, 'text_feat_dim', 0)
            if text_features is not None and text_feat_dim > 0:
                text_q = text_features[queries[g]]
                text_t = text_features[targets[g]]
                combined = torch.cat([combined, text_q, text_t, text_q - text_t])
            elif text_feat_dim > 0:
                combined = torch.cat([combined,
                                      torch.zeros(3 * text_feat_dim, device=dev)])

            model._last_combined = combined.detach()

            if task is not None and getattr(model, 'multi_head_classifier', None) is not None:
                logits = model.multi_head_classifier(combined.unsqueeze(0), task).squeeze(0)
            else:
                logits = model.classifier(combined)
            results.append((logits, answers[g]))
    else:
        # Sequential fallback: standard per-graph forward
        for cc, query, target, answer, metadata in batch:
            cc = cc.clone().to(device)
            logits = model(cc, query, target, metadata=metadata,
                           topo_features=topo_features, task=task)
            results.append((logits, answer))

    return results


def train_epoch_batched(
    model, dataset, optimizer, batch_size=8, max_norm=5.0,
    label_smoothing=0.0, device=None, use_amp=False,
    task=None, topo_features=None,
    class_weights=None,
    contrastive_fn=None, contrastive_weight=0.0,
    loss_fn='ce', focal_gamma=2.0,
    bridge_optimizer=None, bridge_max_norm=1.0,
    bridge_accumulation_steps=8,
):
    """Training epoch with graph-level mini-batching.

    Instead of processing one graph at a time, groups B graphs into a batch.
    When the model has DSM, prefix tokens are batched through DSM in one pass.
    Gradients are accumulated per-batch and a single optimizer step is taken.

    Args:
        model: HierarchicalMultiHopModel (or compatible).
        dataset: BenchmarkDataset with .samples and __getitem__.
        optimizer: PyTorch optimizer.
        batch_size: Number of graphs per mini-batch.
        max_norm: Gradient clipping norm.
        label_smoothing: Cross-entropy label smoothing.
        device: Target device.
        use_amp: Whether to use bf16 autocast.
        task: Optional task name for multi-head classifier.
        topo_features: Optional topology features tensor for ControlHead.
        class_weights: Optional per-class weights for cross-entropy.
        contrastive_fn: Optional SemanticContrastiveLoss instance.
        contrastive_weight: Weight for contrastive loss term.

    Returns:
        Average training loss for the epoch.
    """
    if device is None:
        device = torch.device('cpu')
    model.train()

    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()

    amp_enabled = use_amp and device.type == 'cuda'
    total_loss = 0.0
    n_samples = 0

    # Collect bridge params for separate clipping
    bridge_params = []
    main_params = []
    if bridge_optimizer is not None:
        for pg in bridge_optimizer.param_groups:
            bridge_params.extend(pg['params'])
        for pg in optimizer.param_groups:
            main_params.extend(pg['params'])

    # Create mini-batches
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    # Bridge accumulation counter (bridges step less frequently)
    bridge_batch_count = 0
    if bridge_optimizer is not None:
        bridge_optimizer.zero_grad()

    for batch_indices in batches:
        optimizer.zero_grad()
        samples = [_unpack_sample(dataset[i]) for i in batch_indices]

        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.bfloat16):
            results = _forward_batch(model, samples, device, task=task,
                                     topo_features=topo_features)

            # Compute batch loss
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            valid_count = 0
            for logits, answer in results:
                target = torch.tensor([answer], device=device)
                if loss_fn == 'focal':
                    loss = focal_loss(logits.unsqueeze(0), target,
                                      gamma=focal_gamma, alpha=class_weights,
                                      label_smoothing=label_smoothing)
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0), target,
                        label_smoothing=label_smoothing,
                        weight=class_weights,
                    )
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_loss = batch_loss + loss
                    valid_count += 1

            if valid_count > 0:
                batch_loss = batch_loss / valid_count

            # Contrastive loss (uses last graph's semantic features)
            if (contrastive_fn is not None and valid_count > 0
                    and hasattr(model, '_last_semantic_features')):
                sf = model._last_semantic_features
                adj = model._last_adjacency.to(sf.device)
                c_loss = contrastive_fn(sf, adj) * contrastive_weight
                batch_loss = batch_loss + c_loss

        if valid_count > 0 and not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
            batch_loss.backward()

            # Main optimizer: step every batch
            if bridge_optimizer is not None:
                gn = torch.nn.utils.clip_grad_norm_(main_params, max_norm)
            else:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if torch.isfinite(gn):
                optimizer.step()

            # Bridge optimizer: step every bridge_accumulation_steps batches
            if bridge_optimizer is not None:
                bridge_batch_count += 1
                if bridge_batch_count % bridge_accumulation_steps == 0:
                    bgn = torch.nn.utils.clip_grad_norm_(bridge_params, bridge_max_norm)
                    if torch.isfinite(bgn):
                        bridge_optimizer.step()
                    bridge_optimizer.zero_grad()

            total_loss += batch_loss.item() * valid_count
            n_samples += valid_count

    # Final bridge step for remaining accumulated gradients
    if bridge_optimizer is not None and bridge_batch_count % bridge_accumulation_steps != 0:
        bgn = torch.nn.utils.clip_grad_norm_(bridge_params, bridge_max_norm)
        if torch.isfinite(bgn):
            bridge_optimizer.step()
        bridge_optimizer.zero_grad()

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_batched(
    model, dataset, batch_size=8, label_smoothing=0.0, device=None,
    task=None,
):
    """Evaluate model with graph-level mini-batching.

    Args:
        model: HierarchicalMultiHopModel (or compatible).
        dataset: BenchmarkDataset.
        batch_size: Number of graphs per mini-batch.
        label_smoothing: Cross-entropy label smoothing.
        device: Target device.
        task: Optional task name for multi-head classifier.

    Returns:
        (accuracy, average_loss) tuple.
    """
    if device is None:
        device = torch.device('cpu')
    model.eval()

    correct = 0
    total_loss = 0.0
    n_valid = 0
    n_total = 0

    indices = list(range(len(dataset)))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    class_correct = {}
    class_total = {}

    for batch_indices in batches:
        samples = [_unpack_sample(dataset[i]) for i in batch_indices]
        results = _forward_batch(model, samples, device, task=task)

        for logits, answer in results:
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
    per_class = [class_correct.get(c, 0) / class_total[c] for c in class_total]
    bal_acc = sum(per_class) / len(per_class) if per_class else 0.0
    return accuracy, avg_loss, bal_acc
