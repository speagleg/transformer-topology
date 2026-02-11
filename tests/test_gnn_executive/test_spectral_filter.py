import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.spectral_filter import SpectralFilterLayer, SpectralGNN


def make_chain(dim=16, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestSpectralFilterLayer:
    def test_output_shape(self):
        layer = SpectralFilterLayer(in_dim=16, out_dim=16, num_freqs=5)
        cc = make_chain(dim=16, length=5)
        x = cc.get_embeddings(0)
        from src.spectral.decomposition import spectral_decomposition
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)
        out = layer(x, eigenvalues, eigenvectors)
        assert out.shape == (5, 16)

    def test_low_pass_behavior(self):
        layer = SpectralFilterLayer(in_dim=16, out_dim=16, num_freqs=5)
        cc = make_chain(dim=16, length=5)
        x = cc.get_embeddings(0)
        from src.spectral.decomposition import spectral_decomposition
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)
        out = layer(x, eigenvalues, eigenvectors)
        assert not torch.isnan(out).any()


class TestSpectralGNN:
    def test_output_shape(self):
        gnn = SpectralGNN(in_dim=16, hidden_dim=16, out_dim=16, num_layers=2, max_freqs=5)
        cc = make_chain(dim=16, length=5)
        out = gnn(cc)
        assert out.shape == (5, 16)

    def test_gradient_flow(self):
        gnn = SpectralGNN(in_dim=16, hidden_dim=16, out_dim=16, num_layers=2, max_freqs=5)
        cc = make_chain(dim=16, length=5)
        embs = cc.get_embeddings(0).clone().requires_grad_(True)
        cc.set_embeddings(0, embs)
        out = gnn(cc)
        loss = out.sum()
        loss.backward()
        assert embs.grad is not None
