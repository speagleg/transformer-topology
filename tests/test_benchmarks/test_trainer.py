import torch
import pytest
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.trainer import train_epoch, evaluate


class TestMultiHopModel:
    def test_output_shape(self):
        model = MultiHopReasoningModel(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_hops=10, max_iterations=2,
        )
        ds = MultiHopDataset(num_samples=1, min_hops=3, max_hops=3, num_distractors=5, embedding_dim=32)
        cc, query, target, answer = ds[0]
        logits = model(cc, query, target)
        assert logits.shape == (11,)  # max_hops + 1 classes (0 through max_hops)

    def test_gradient_flow(self):
        model = MultiHopReasoningModel(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=1,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_hops=10, max_iterations=1,
        )
        ds = MultiHopDataset(num_samples=1, min_hops=2, max_hops=2, num_distractors=3, embedding_dim=32)
        cc, query, target, answer = ds[0]
        logits = model(cc, query, target)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([answer]))
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0


class TestTraining:
    def test_train_epoch(self):
        model = MultiHopReasoningModel(
            embedding_dim=16, gnn_hidden=32, gnn_spatial_layers=1,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=32,
            max_hops=6, max_iterations=1,
        )
        ds = MultiHopDataset(num_samples=4, min_hops=2, max_hops=4, num_distractors=3, embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = train_epoch(model, ds, optimizer)
        assert loss > 0

    def test_evaluate(self):
        model = MultiHopReasoningModel(
            embedding_dim=16, gnn_hidden=32, gnn_spatial_layers=1,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=32,
            max_hops=6, max_iterations=1,
        )
        ds = MultiHopDataset(num_samples=4, min_hops=2, max_hops=4, num_distractors=3, embedding_dim=16)
        accuracy, avg_loss = evaluate(model, ds)
        assert 0.0 <= accuracy <= 1.0
        assert avg_loss > 0
