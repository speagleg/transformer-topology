import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.loop import ReasoningLoop


def make_chain(dim=32, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestReasoningLoop:
    def test_output_shape(self):
        loop = ReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=2, max_freqs=4, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_iterations=3, convergence_threshold=0.01,
        )
        cc = make_chain(dim=32, length=5)
        output, num_iters = loop(cc)
        assert output.shape == (5, 32)
        assert 1 <= num_iters <= 3

    def test_convergence(self):
        loop = ReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=1,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_iterations=10, convergence_threshold=100.0,
        )
        cc = make_chain(dim=32, length=5)
        output, num_iters = loop(cc)
        assert num_iters == 1

    def test_updates_cell_complex(self):
        loop = ReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=2, max_freqs=4, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_iterations=2, convergence_threshold=0.001,
        )
        cc = make_chain(dim=32, length=5)
        original_emb = cc.get_embeddings(0).clone()
        output, _ = loop(cc)
        assert not torch.allclose(output, original_emb, atol=0.01)
