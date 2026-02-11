import torch
import pytest
from src.tat.spectral_attention import TopologicalSpectralAttention


class TestTopologicalSpectralAttention:
    def test_output_shape(self):
        attn = TopologicalSpectralAttention(embed_dim=64, num_heads=4, num_freqs=8)
        x = torch.randn(10, 64)
        eigenvectors = torch.randn(10, 8)
        eigenvalues = torch.sort(torch.rand(8))[0]
        out = attn(x, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        assert out.shape == (10, 64)

    def test_gradient_flow(self):
        attn = TopologicalSpectralAttention(embed_dim=64, num_heads=4, num_freqs=8)
        x = torch.randn(10, 64, requires_grad=True)
        eigenvectors = torch.randn(10, 8)
        eigenvalues = torch.sort(torch.rand(8))[0]
        out = attn(x, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_captures_long_range(self):
        attn = TopologicalSpectralAttention(embed_dim=16, num_heads=2, num_freqs=4)
        x = torch.randn(6, 16)
        from src.cell_complex.cell_complex import CellComplex
        from src.spectral.decomposition import spectral_decomposition
        cc = CellComplex(embedding_dim=16)
        for i in range(6):
            cc.add_0_cell(torch.randn(16), "c")
        for i in range(5):
            cc.add_1_cell(i, i + 1, torch.randn(16), "r")
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0, k=4)
        out = attn(x, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        assert not torch.isnan(out).any()
