"""Post-training analysis for DSM v6 curriculum results.

Evaluates the final model on all tasks across multiple axes:
  1. Per-task accuracy (all 14 tasks)
  2. Per-class breakdown for semantic tasks
  3. Size generalization (n=20, 40, 80)
  4. Topology transfer (train topos vs held-out)
  5. DSM contribution analysis (with vs without DSM)
  6. Control signal statistics across task types
  7. Semantic feature extraction (for t-SNE visualization)
  8. Topology alert history (embedding + computation graph)
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset, TASK_REGISTRY, get_max_classes
from src.benchmarks.diagnostics import DiagnosticCollector
from src.benchmarks.run_benchmark_suite import _build_model, _resolve_device
from src.benchmarks.run_comparison import evaluate, evaluate_per_class, _unpack_sample


def _load_or_generate(task, split, n_nodes, embedding_dim, topologies=None,
                      n_nodes_range=None, num_samples=200, pregen_dir=None):
    if pregen_dir is not None:
        range_str = f"n{n_nodes_range[0]}-{n_nodes_range[1]}" if n_nodes_range else f"n{n_nodes}"
        path = pregen_dir / f"{task}_{split}_{range_str}.pt"
        if path.exists():
            ds = BenchmarkDataset.load(str(path))
            if len(ds) > num_samples:
                ds.samples = ds.samples[:num_samples]
            return ds
    return BenchmarkDataset(num_samples, task, n_nodes, embedding_dim,
                            topologies=topologies, n_nodes_range=n_nodes_range)


def _use_multi_head(model):
    """Check if model has multi-head classifier."""
    return getattr(model, 'multi_head_classifier', None) is not None


def _rebuild_classifier(model, task):
    """Fallback classifier rebuild for models without multi-head classifier."""
    if _use_multi_head(model):
        return  # multi-head handles task switching
    num_classes = get_max_classes(task)
    if model.classifier[-1].out_features == num_classes:
        return
    old_in = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(old_in, num_classes)
    model.classifier[-1].to(next(model.parameters()).device)


def _eval_task(model, ds, device, task):
    """Evaluate with task= if multi-head, else fallback to legacy."""
    t = task if _use_multi_head(model) else None
    return evaluate(model, ds, device=device, task=t)


@torch.no_grad()
def _extract_semantic_features(model, dataset, device, num_samples=100):
    """Extract TopoBridge semantic features for t-SNE visualization.

    Returns:
        dict with 'semantic_features': (N, dim) tensor,
                   'graph_embeddings': (N, topo_dim) tensor,
                   'labels': list of int answers
    """
    if not (hasattr(model, 'topo_bridge') and model.topo_bridge is not None):
        return None

    model.eval()
    sem_features = []
    graph_embeddings = []
    labels = []
    captured = {}

    def hook_fn(module, inp, out):
        # TopoBridge.forward returns (semantic_out, semantic_bias, sem_feat, graph_emb)
        if isinstance(out, tuple) and len(out) >= 4:
            captured['sem_feat'] = out[2].detach().cpu()
            captured['graph_emb'] = out[3].detach().cpu()

    handle = model.topo_bridge.register_forward_hook(hook_fn)
    try:
        n = min(num_samples, len(dataset))
        for i in range(n):
            cc, query, target, answer, metadata = _unpack_sample(dataset[i])
            cc = cc.clone().to(device)
            captured.clear()
            model.bypass_llm = False
            model(cc, query, target, metadata=metadata)

            if 'sem_feat' in captured:
                # sem_feat: (seq, dim) → mean pool to (dim,)
                sf = captured['sem_feat']
                if sf.dim() > 1:
                    sf = sf.mean(dim=0)
                sem_features.append(sf)
            if 'graph_emb' in captured:
                graph_embeddings.append(captured['graph_emb'])
            labels.append(answer)
    finally:
        handle.remove()

    result = {'labels': labels}
    if sem_features:
        result['semantic_features'] = torch.stack(sem_features)
    if graph_embeddings:
        result['graph_embeddings'] = torch.stack(graph_embeddings)
    return result


@torch.no_grad()
def _run_topology_alerts(model, dataset, device, config):
    """Run topology observer on a few samples and collect alerts."""
    try:
        from src.topology_observer import TopologyObserver
    except ImportError:
        return {'error': 'TopologyObserver not available'}

    topo_config = config.get('topology_observer', {})
    criterion = nn.CrossEntropyLoss()

    dsm = None
    if hasattr(model, 'executive_loop') and hasattr(model.executive_loop, 'topo_bridge'):
        bridge = model.executive_loop.topo_bridge
        if bridge is not None and hasattr(bridge, 'backend') and hasattr(bridge.backend, 'dsm'):
            dsm = bridge.backend.dsm

    observer = TopologyObserver(model, criterion, dsm=dsm, config=topo_config)
    alert_history = []
    n = min(5, len(dataset))
    for i in range(n):
        result = observer.run_analysis(epoch=i, sample=dataset[i])
        alert_history.append(result)

    return {
        'analyses': alert_history,
        'num_profiles': len(observer.profile_history),
        'last_topo_features': (
            observer.get_topo_features().tolist()
            if observer.get_topo_features() is not None else None
        ),
    }


def run_analysis(config_path="config/dsm_training.yaml",
                 model_path="data/dsm_results/dsm_model_final.pt",
                 pregen_dir_str="data/dsm_datasets"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    wc = config.get("wave")
    lc = config.get("llm")

    device = _resolve_device(tc)
    emb_dim = mc["embedding_dim"]
    train_n = bc["train_n_nodes"]
    pregen_dir = Path(pregen_dir_str) if pregen_dir_str else None

    train_topos = bc.get("train_topologies", [])
    test_topos = bc.get("test_topologies", [])
    all_topos = (train_topos + test_topos) or None

    n_min = bc.get("train_n_nodes_min")
    n_max = bc.get("train_n_nodes_max")
    train_range = (n_min, n_max) if n_min and n_max else None

    # Build and load model
    initial_classes = get_max_classes("diverse")
    model = _build_model("hierarchical_llm", mc, initial_classes, device,
                         wave_config=wc, llm_config=lc)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape}
    skipped = set(state.keys()) - set(filtered.keys())
    model.load_state_dict(filtered, strict=False)
    del state
    if skipped:
        print(f"Skipped {len(skipped)} keys with shape mismatch: {skipped}")
    print(f"Loaded model from {model_path}")
    print(f"Device: {device}")
    multi_head = _use_multi_head(model)
    print(f"Multi-head classifier: {multi_head}")

    results = {}

    # ================================================================
    # 1. Per-task accuracy on all available tasks
    # ================================================================
    print("\n" + "=" * 72)
    print("1. PER-TASK ACCURACY (val set, n=train range)")
    print("=" * 72)

    all_tasks = ["diverse", "bfs", "hodge_class", "spectral_gap", "path_counting",
                 "graph_completion", "labeled_reasoning", "analogical_transfer",
                 "cycle_detection", "betti_number", "blocking", "interference"]

    task_results = {}
    for task in all_tasks:
        try:
            _rebuild_classifier(model, task)
            ds = _load_or_generate(task, "val", train_n, emb_dim,
                                   topologies=all_topos, n_nodes_range=train_range,
                                   num_samples=500, pregen_dir=pregen_dir)
            acc, loss = _eval_task(model, ds, device, task)
            task_results[task] = {"accuracy": acc, "loss": loss, "n_samples": len(ds)}
            print(f"  {task:25s} acc={acc:.3f}  loss={loss:.4f}  (n={len(ds)})")
        except Exception as e:
            print(f"  {task:25s} ERROR: {e}")
            task_results[task] = {"error": str(e)}
    results["per_task"] = task_results

    # ================================================================
    # 2. Per-class breakdown for semantic tasks
    # ================================================================
    print("\n" + "=" * 72)
    print("2. PER-CLASS BREAKDOWN (semantic tasks)")
    print("=" * 72)

    semantic_tasks = ["graph_completion", "labeled_reasoning", "analogical_transfer"]
    per_class_results = {}
    for task in semantic_tasks:
        try:
            _rebuild_classifier(model, task)
            num_classes = get_max_classes(task)
            ds = _load_or_generate(task, "val", train_n, emb_dim,
                                   topologies=all_topos, n_nodes_range=train_range,
                                   num_samples=500, pregen_dir=pregen_dir)
            breakdown = evaluate_per_class(model, ds, num_classes, device=device)
            per_class_results[task] = breakdown["per_class"]
            print(f"\n  {task} ({num_classes} classes):")
            for c, info in sorted(breakdown["per_class"].items()):
                print(f"    class {c}: {info['correct']}/{info['total']} = {info['accuracy']:.3f}")
        except Exception as e:
            print(f"  {task}: ERROR: {e}")
    results["per_class"] = per_class_results

    # ================================================================
    # 3. Size generalization
    # ================================================================
    print("\n" + "=" * 72)
    print("3. SIZE GENERALIZATION")
    print("=" * 72)

    size_tasks = ["diverse", "bfs", "graph_completion", "analogical_transfer"]
    test_sizes = bc.get("test_n_nodes", [20, 40, 80])
    size_results = {}
    for task in size_tasks:
        size_results[task] = {}
        _rebuild_classifier(model, task)
        for n in test_sizes:
            try:
                ds = _load_or_generate(task, "test", n, emb_dim,
                                       topologies=all_topos, num_samples=200,
                                       pregen_dir=pregen_dir)
                acc, loss = _eval_task(model, ds, device, task)
                size_results[task][str(n)] = {"accuracy": acc, "loss": loss}
                print(f"  {task:25s} n={n:3d}  acc={acc:.3f}  loss={loss:.4f}")
            except Exception as e:
                print(f"  {task:25s} n={n:3d}  ERROR: {e}")
                size_results[task][str(n)] = {"error": str(e)}
    results["size_generalization"] = size_results

    # ================================================================
    # 4. Topology transfer
    # ================================================================
    print("\n" + "=" * 72)
    print("4. TOPOLOGY TRANSFER (train topos vs held-out)")
    print("=" * 72)

    topo_tasks = ["diverse", "bfs", "graph_completion"]
    topo_results = {}
    for task in topo_tasks:
        topo_results[task] = {}
        _rebuild_classifier(model, task)
        for label, topos in [("train_topos", train_topos), ("test_topos", test_topos)]:
            if not topos:
                continue
            try:
                ds = BenchmarkDataset(200, task, train_n, emb_dim,
                                      topologies=topos, n_nodes_range=train_range)
                acc, loss = _eval_task(model, ds, device, task)
                topo_results[task][label] = {"accuracy": acc, "loss": loss,
                                             "topologies": topos}
                print(f"  {task:25s} {label:12s} acc={acc:.3f}  topos={topos}")
            except Exception as e:
                print(f"  {task:25s} {label:12s} ERROR: {e}")
    results["topology_transfer"] = topo_results

    # ================================================================
    # 5. DSM ablation: with vs without DSM
    # ================================================================
    print("\n" + "=" * 72)
    print("5. DSM ABLATION (bypass_llm=True vs False)")
    print("=" * 72)

    ablation_tasks = ["diverse", "graph_completion", "labeled_reasoning",
                      "analogical_transfer"]
    ablation_results = {}
    for task in ablation_tasks:
        _rebuild_classifier(model, task)
        ds = _load_or_generate(task, "val", train_n, emb_dim,
                               topologies=all_topos, n_nodes_range=train_range,
                               num_samples=200, pregen_dir=pregen_dir)

        # With DSM
        model.bypass_llm = False
        acc_with, loss_with = _eval_task(model, ds, device, task)

        # Without DSM
        model.bypass_llm = True
        acc_without, loss_without = _eval_task(model, ds, device, task)

        model.bypass_llm = False  # restore

        delta = acc_with - acc_without
        ablation_results[task] = {
            "with_dsm": {"accuracy": acc_with, "loss": loss_with},
            "without_dsm": {"accuracy": acc_without, "loss": loss_without},
            "delta": delta,
        }
        marker = "+" if delta > 0 else ""
        print(f"  {task:25s} with={acc_with:.3f}  without={acc_without:.3f}  "
              f"delta={marker}{delta:.3f}")
    results["dsm_ablation"] = ablation_results

    # ================================================================
    # 6. Control signal diagnostics
    # ================================================================
    print("\n" + "=" * 72)
    print("6. CONTROL SIGNAL DIAGNOSTICS")
    print("=" * 72)

    diag_tasks = ["diverse", "bfs", "graph_completion", "labeled_reasoning",
                  "analogical_transfer"]
    diag_results = {}
    for task in diag_tasks:
        _rebuild_classifier(model, task)
        ds = _load_or_generate(task, "val", train_n, emb_dim,
                               topologies=all_topos, n_nodes_range=train_range,
                               num_samples=50, pregen_dir=pregen_dir)
        collector = DiagnosticCollector()
        collector.collect(model, ds, device)
        summary = collector.summarize()
        diag_results[task] = summary

        gate = summary.get("semantic_weight", {})
        conf = summary.get("confidence", {})
        diff = summary.get("diffusion_time", {})
        damp = summary.get("wave_damping", {})
        filt = summary.get("filter_weights_entropy", {})
        print(f"  {task:25s} gate={gate.get('mean',0):.3f}+/-{gate.get('std',0):.3f}  "
              f"conf={conf.get('mean',0):.3f}  "
              f"diff_t={diff.get('mean',0):.3f}  "
              f"damp={damp.get('mean',0):.3f}  "
              f"filt_H={filt.get('mean',0):.3f}")
    results["diagnostics"] = diag_results

    # ================================================================
    # 7. Semantic feature extraction (for t-SNE)
    # ================================================================
    print("\n" + "=" * 72)
    print("7. SEMANTIC FEATURE EXTRACTION (for t-SNE)")
    print("=" * 72)

    sem_tasks = ["diverse", "graph_completion", "labeled_reasoning"]
    sem_results = {}
    for task in sem_tasks:
        _rebuild_classifier(model, task)
        ds = _load_or_generate(task, "val", train_n, emb_dim,
                               topologies=all_topos, n_nodes_range=train_range,
                               num_samples=100, pregen_dir=pregen_dir)
        feats = _extract_semantic_features(model, ds, device, num_samples=100)
        if feats is not None and 'semantic_features' in feats:
            shape = list(feats['semantic_features'].shape)
            sem_results[task] = {
                "n_samples": shape[0],
                "feature_dim": shape[1] if len(shape) > 1 else 1,
            }
            # Save raw tensors for offline t-SNE plotting
            feat_path = Path(f"data/dsm_results/sem_features_{task}.pt")
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feats, feat_path)
            print(f"  {task:25s} {shape[0]} samples, dim={shape[1] if len(shape) > 1 else 1}  "
                  f"saved to {feat_path}")
        else:
            sem_results[task] = {"status": "no TopoBridge / no features"}
            print(f"  {task:25s} no TopoBridge or features not available")
    results["semantic_features"] = sem_results

    # ================================================================
    # 8. Topology alert history
    # ================================================================
    print("\n" + "=" * 72)
    print("8. TOPOLOGY ALERT HISTORY")
    print("=" * 72)

    topo_alert_ds = _load_or_generate("diverse", "val", train_n, emb_dim,
                                       topologies=all_topos, n_nodes_range=train_range,
                                       num_samples=10, pregen_dir=pregen_dir)
    topo_alerts = _run_topology_alerts(model, topo_alert_ds, device, config)
    results["topology_alerts"] = topo_alerts
    if 'error' in topo_alerts:
        print(f"  {topo_alerts['error']}")
    else:
        print(f"  Ran {len(topo_alerts.get('analyses', []))} analyses")
        print(f"  Profiles collected: {topo_alerts.get('num_profiles', 0)}")
        tf = topo_alerts.get('last_topo_features')
        if tf is not None:
            print(f"  Last topo features: {[round(x, 4) for x in tf]}")
        else:
            print(f"  Last topo features: None (no DSM)")

    # ================================================================
    # Save
    # ================================================================
    out_path = Path("data/dsm_results/v6_full_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull analysis saved to {out_path}")

    return results


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config/dsm_training.yaml"
    model = sys.argv[2] if len(sys.argv) > 2 else "data/dsm_results/dsm_model_final.pt"
    pregen = sys.argv[3] if len(sys.argv) > 3 else "data/dsm_datasets"
    run_analysis(config, model, pregen)
