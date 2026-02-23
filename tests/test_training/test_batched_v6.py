"""Tests for v6 batched training extensions."""
import torch
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.benchmark_dataset import BenchmarkDataset
from src.training.batch_utils import train_epoch_batched, evaluate_batched


class TestBatchedV6:
    def _make_model_and_data(self):
        model = HierarchicalMultiHopModel(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=32, max_classes=11,
            max_iterations=2, convergence_threshold=0.1,
            use_wave_dynamics=False, use_higher_order=False,
        )
        ds = BenchmarkDataset(8, "diverse", 20, 16)
        return model, ds

    def test_train_epoch_batched_with_task(self):
        model, ds = self._make_model_and_data()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = train_epoch_batched(model, ds, opt, batch_size=4, task="diverse")
        assert loss > 0

    def test_evaluate_batched_with_task(self):
        model, ds = self._make_model_and_data()
        acc, loss = evaluate_batched(model, ds, batch_size=4, task="diverse")
        assert 0 <= acc <= 1
        assert loss > 0

    def test_train_with_topo_features(self):
        model, ds = self._make_model_and_data()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        topo_feat = torch.randn(6)
        loss = train_epoch_batched(model, ds, opt, batch_size=4, topo_features=topo_feat)
        assert loss > 0
