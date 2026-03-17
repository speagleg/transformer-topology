"""Tests for DSM curriculum training script."""

import torch
import yaml


class TestPerTaskMaxClasses:
    def test_diverse(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('diverse') == 11

    def test_bfs(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('bfs') == 16

    def test_graph_completion(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('graph_completion') == 2

    def test_analogical_transfer(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('analogical_transfer') == 3

    def test_hodge_class(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('hodge_class') == 3

    def test_spectral_gap(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('spectral_gap') == 8

    def test_path_counting(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('path_counting') == 5

    def test_labeled_reasoning(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('labeled_reasoning') == 3


_MC = {
    'embedding_dim': 32, 'gnn_hidden': 64, 'gnn_spatial_layers': 2,
    'gnn_spectral_layers': 1, 'max_freqs': 8, 'tat_layers': 1,
    'tat_spatial_heads': 4, 'tat_spectral_heads': 4, 'tat_ff_dim': 64,
    'use_higher_order': True, 'use_topological_pe': False,
    'use_structural_features': False, 'max_iterations': 5,
    'convergence_threshold': 0.1,
}
_LC = {
    'backend': 'dsm', 'dsm_dim': 64, 'num_heads': 4, 'ff_dim': 128,
    'num_layers': 2, 'cross_attn_layer': 0, 'num_prefix': 4,
}


class TestRebuildClassifier:
    def test_rebuild_to_graph_completion(self):
        from scripts.run_dsm_curriculum import _rebuild_classifier
        from src.benchmarks.run_benchmark_suite import _build_model
        model = _build_model('hierarchical_llm', _MC, 16, torch.device('cpu'),
                              llm_config=_LC)
        _rebuild_classifier(model, 'graph_completion')
        assert model.classifier[-1].out_features == 2

    def test_rebuild_to_diverse(self):
        from scripts.run_dsm_curriculum import _rebuild_classifier
        from src.benchmarks.run_benchmark_suite import _build_model
        model = _build_model('hierarchical_llm', _MC, 16, torch.device('cpu'),
                              llm_config=_LC)
        _rebuild_classifier(model, 'diverse')
        assert model.classifier[-1].out_features == 11

    def test_rebuild_preserves_in_features(self):
        from scripts.run_dsm_curriculum import _rebuild_classifier
        from src.benchmarks.run_benchmark_suite import _build_model
        model = _build_model('hierarchical_llm', _MC, 16, torch.device('cpu'),
                              llm_config=_LC)
        in_features = model.classifier[-1].in_features
        _rebuild_classifier(model, 'bfs')
        assert model.classifier[-1].in_features == in_features
        assert model.classifier[-1].out_features == 16

    def test_rebuild_multiple_times(self):
        from scripts.run_dsm_curriculum import _rebuild_classifier
        from src.benchmarks.run_benchmark_suite import _build_model
        model = _build_model('hierarchical_llm', _MC, 16, torch.device('cpu'),
                              llm_config=_LC)
        _rebuild_classifier(model, 'graph_completion')
        assert model.classifier[-1].out_features == 2
        _rebuild_classifier(model, 'diverse')
        assert model.classifier[-1].out_features == 11
        _rebuild_classifier(model, 'spectral_gap')
        assert model.classifier[-1].out_features == 8


class TestBuildDSMOptimizer:
    def test_param_groups(self):
        from scripts.run_dsm_curriculum import _build_dsm_optimizers
        from src.benchmarks.run_benchmark_suite import _build_model
        config = {'training': {'learning_rate': 1e-3, 'dsm_learning_rate': 5e-4,
                                'bridge_learning_rate': 1e-4, 'weight_decay': 0.01}}
        model = _build_model('hierarchical_llm', _MC, 11, torch.device('cpu'),
                              llm_config=_LC)
        main_opt, bridge_opt = _build_dsm_optimizers(model, config)
        # GNN/TAT + DSM/adapter + classifier (metacog only if use_metacog=True)
        assert len(main_opt.param_groups) >= 2
        assert len(main_opt.param_groups) <= 5  # max: GNN/TAT, DSM, metacog, classifier, Qwen
        assert bridge_opt is not None
        assert len(bridge_opt.param_groups) == 1  # bridge params

    def test_correct_learning_rates(self):
        from scripts.run_dsm_curriculum import _build_dsm_optimizers
        from src.benchmarks.run_benchmark_suite import _build_model
        config = {'training': {'learning_rate': 1e-3, 'dsm_learning_rate': 5e-4,
                                'bridge_learning_rate': 1e-4, 'weight_decay': 0.01}}
        model = _build_model('hierarchical_llm', _MC, 11, torch.device('cpu'),
                              llm_config=_LC)
        main_opt, bridge_opt = _build_dsm_optimizers(model, config)
        assert main_opt.param_groups[0]['lr'] == 1e-3   # GNN/TAT
        assert main_opt.param_groups[1]['lr'] == 5e-4   # DSM/adapter
        assert bridge_opt.param_groups[0]['lr'] == 1e-4  # bridge

    def test_all_params_covered(self):
        """Every trainable param is in exactly one group."""
        from scripts.run_dsm_curriculum import _build_dsm_optimizers
        from src.benchmarks.run_benchmark_suite import _build_model
        config = {'training': {'learning_rate': 1e-3, 'dsm_learning_rate': 5e-4,
                                'bridge_learning_rate': 1e-4, 'weight_decay': 0.01}}
        model = _build_model('hierarchical_llm', _MC, 11, torch.device('cpu'),
                              llm_config=_LC)
        main_opt, bridge_opt = _build_dsm_optimizers(model, config)
        opt_params = set()
        for group in main_opt.param_groups:
            for p in group['params']:
                opt_params.add(id(p))
        if bridge_opt is not None:
            for group in bridge_opt.param_groups:
                for p in group['params']:
                    opt_params.add(id(p))
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        assert opt_params == trainable


class TestConfig:
    def test_no_gate_penalty(self):
        with open('config/dsm_training.yaml') as f:
            config = yaml.safe_load(f)
        assert 'gate_penalty' not in config.get('training', {})
        assert 'gate_penalty_weight' not in config.get('training', {})

    def test_has_dsm_learning_rate(self):
        with open('config/dsm_training.yaml') as f:
            config = yaml.safe_load(f)
        assert 'dsm_learning_rate' in config['training']
        assert 'bridge_learning_rate' in config['training']

    def test_has_required_model_keys(self):
        with open('config/dsm_training.yaml') as f:
            config = yaml.safe_load(f)
        mc = config['model']
        assert 'max_iterations' in mc
        assert 'convergence_threshold' in mc

    def test_dsm_backend(self):
        with open('config/dsm_training.yaml') as f:
            config = yaml.safe_load(f)
        assert config['llm']['backend'] == 'dsm'

    def test_phase_epochs(self):
        with open('config/dsm_training.yaml') as f:
            config = yaml.safe_load(f)
        tc = config['training']
        assert tc['epochs_phase_a'] == 30
        assert tc['epochs_phase_b'] == 30
        assert tc['epochs_phase_c'] == 30


class TestTrainEpoch:
    def test_runs_without_crash(self):
        """A single training step completes and returns a positive loss."""
        from scripts.run_dsm_curriculum import train_epoch, _rebuild_classifier
        from src.benchmarks.run_benchmark_suite import _build_model
        from src.benchmarks.benchmark_dataset import BenchmarkDataset

        model = _build_model('hierarchical_llm', _MC, 11, torch.device('cpu'),
                              llm_config=_LC)
        _rebuild_classifier(model, 'diverse')
        ds = BenchmarkDataset(5, 'diverse', 10, 32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = train_epoch(model, ds, optimizer, device=torch.device('cpu'))
        assert isinstance(loss, float)
        assert loss > 0

    def test_with_dsm_optimizer(self):
        """Training works with the split DSM optimizers."""
        from scripts.run_dsm_curriculum import (
            train_epoch, _rebuild_classifier, _build_dsm_optimizers,
        )
        from src.benchmarks.run_benchmark_suite import _build_model
        from src.benchmarks.benchmark_dataset import BenchmarkDataset

        config = {'training': {'learning_rate': 1e-3, 'dsm_learning_rate': 5e-4,
                                'bridge_learning_rate': 1e-4, 'weight_decay': 0.01}}
        model = _build_model('hierarchical_llm', _MC, 11, torch.device('cpu'),
                              llm_config=_LC)
        _rebuild_classifier(model, 'diverse')
        ds = BenchmarkDataset(5, 'diverse', 10, 32)
        main_opt, bridge_opt = _build_dsm_optimizers(model, config)
        loss = train_epoch(model, ds, main_opt, device=torch.device('cpu'),
                           bridge_optimizer=bridge_opt)
        assert isinstance(loss, float)
        assert loss > 0


class TestReplayDataset:
    def test_create_replay_dataset(self):
        from scripts.run_dsm_curriculum import _create_replay_dataset
        from src.benchmarks.benchmark_dataset import BenchmarkDataset

        ds1 = BenchmarkDataset(10, 'diverse', 10, 32)
        ds2 = BenchmarkDataset(10, 'bfs', 10, 32)
        phase_a_datasets = {'diverse': ds1, 'bfs': ds2}
        replay = _create_replay_dataset(phase_a_datasets, replay_fraction=0.2)
        # 20% of 10 = 2 per task, so 4 total
        assert len(replay) == 4
        # Each item is a (sample, task_type) tuple
        for sample, task_type in replay:
            assert task_type in ('diverse', 'bfs')

    def test_replay_with_target_size(self):
        from scripts.run_dsm_curriculum import _create_replay_dataset
        from src.benchmarks.benchmark_dataset import BenchmarkDataset

        ds = BenchmarkDataset(20, 'diverse', 10, 32)
        phase_a_datasets = {'diverse': ds}
        replay = _create_replay_dataset(phase_a_datasets, replay_fraction=0.5,
                                         target_size=3)
        assert len(replay) <= 3


_V10_MC = {
    'embedding_dim': 32, 'gnn_hidden': 64, 'gnn_spatial_layers': 2,
    'gnn_spectral_layers': 1, 'max_freqs': 8, 'tat_layers': 1,
    'tat_spatial_heads': 4, 'tat_spectral_heads': 4, 'tat_ff_dim': 64,
    'use_higher_order': True, 'use_topological_pe': False,
    'use_structural_features': False, 'max_iterations': 2,
    'convergence_threshold': 0.05, 'use_dual_track': True,
    'use_multi_head_classifier': True, 'use_wave_dynamics': False,
}
_V10_LC = {
    'backend': 'qwen_contextual',
    'llm_dim': 64, 'output_dim': 32,
    'use_mock': True, 'num_tasks': 23,
}


class TestV10ModelBuild:
    def test_build_v10_model(self):
        from src.benchmarks.run_benchmark_suite import _build_model
        model = _build_model('hierarchical_llm', _V10_MC, 10,
                              torch.device('cpu'), llm_config=_V10_LC)
        assert model.qwen_encoder is not None
        assert model.attention_readout is not None
        assert model.topo_bridge is None
        assert model.executive_loop.use_dual_track is True
        # 3*32 + 4 + 1 + 2*32 = 165 (text_dim=embedding_dim now)
        assert model.classifier_input_dim == 165

    def test_v10_forward_pass(self):
        from src.benchmarks.run_benchmark_suite import _build_model
        from src.cell_complex.cell_complex import CellComplex
        model = _build_model('hierarchical_llm', _V10_MC, 16,
                              torch.device('cpu'), llm_config=_V10_LC)
        cc = CellComplex(embedding_dim=32)
        for i in range(10):
            cc.add_0_cell(torch.randn(32), 'node')
        for i in range(9):
            cc.add_1_cell(i, i+1, torch.randn(32), 'edge')
        cc.node_texts = [f'concept_{i}' for i in range(10)]
        logits = model(cc, 0, 3, task='bfs')
        assert logits.shape == (16,)

    def test_v10_gradient_flows(self):
        from src.benchmarks.run_benchmark_suite import _build_model
        from src.cell_complex.cell_complex import CellComplex
        model = _build_model('hierarchical_llm', _V10_MC, 5,
                              torch.device('cpu'), llm_config=_V10_LC)
        cc = CellComplex(embedding_dim=32)
        for i in range(8):
            cc.add_0_cell(torch.randn(32), 'node')
        for i in range(7):
            cc.add_1_cell(i, i+1, torch.randn(32), 'edge')
        cc.node_texts = [f'n_{i}' for i in range(8)]
        logits = model(cc.clone(), 0, 3, task='diverse')
        loss = logits.sum()
        loss.backward()
        # Gradients reach Qwen proj
        assert model.qwen_encoder.proj.weight.grad is not None
        # Gradients reach GNN
        gnn_grads = sum(1 for p in model.executive_loop.parameters()
                        if p.grad is not None)
        assert gnn_grads > 0

    def test_v10_bypass_llm(self):
        """bypass_llm=True skips Qwen encoding (Phase A behavior)."""
        from src.benchmarks.run_benchmark_suite import _build_model
        from src.cell_complex.cell_complex import CellComplex
        model = _build_model('hierarchical_llm', _V10_MC, 10,
                              torch.device('cpu'), llm_config=_V10_LC)
        model.bypass_llm = True
        cc = CellComplex(embedding_dim=32)
        for i in range(10):
            cc.add_0_cell(torch.randn(32), 'node')
        for i in range(9):
            cc.add_1_cell(i, i+1, torch.randn(32), 'edge')
        cc.node_texts = [f'concept_{i}' for i in range(10)]
        logits = model(cc, 0, 3, task='bfs')
        # bfs has 16 classes via multi_head_classifier
        assert logits.shape == (16,)


class TestV10Optimizer:
    def test_v10_param_groups(self):
        from scripts.run_dsm_curriculum import _build_dsm_optimizers
        from src.benchmarks.run_benchmark_suite import _build_model
        config = {'training': {'learning_rate': 3e-4, 'dsm_learning_rate': 1e-3,
                                'weight_decay': 0.01}}
        model = _build_model('hierarchical_llm', _V10_MC, 10,
                              torch.device('cpu'), llm_config=_V10_LC)
        main_opt, bridge_opt = _build_dsm_optimizers(model, config)
        # v10 has no bridge params
        assert bridge_opt is None
        # Should have GNN/TAT + cross-attn/Qwen groups
        assert len(main_opt.param_groups) >= 2

    def test_v10_all_params_covered(self):
        from scripts.run_dsm_curriculum import _build_dsm_optimizers
        from src.benchmarks.run_benchmark_suite import _build_model
        config = {'training': {'learning_rate': 3e-4, 'dsm_learning_rate': 1e-3,
                                'weight_decay': 0.01}}
        model = _build_model('hierarchical_llm', _V10_MC, 10,
                              torch.device('cpu'), llm_config=_V10_LC)
        main_opt, bridge_opt = _build_dsm_optimizers(model, config)
        opt_params = set()
        for group in main_opt.param_groups:
            for p in group['params']:
                opt_params.add(id(p))
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        assert opt_params == trainable
