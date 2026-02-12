import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.tat.transformer import TopologyAwareTransformer


def make_chain(dim=64, length=6):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestTopologyAwareTransformer:
    def test_output_shape(self):
        tat = TopologyAwareTransformer(
            embedding_dim=64, num_layers=2, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=128, num_freqs=4,
        )
        cc = make_chain(dim=64, length=6)
        out = tat(cc)
        assert out.shape == (6, 64)

    def test_different_graph_sizes(self):
        tat = TopologyAwareTransformer(
            embedding_dim=32, num_layers=2, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=4,
        )
        for length in [3, 5, 10]:
            cc = make_chain(dim=32, length=length)
            out = tat(cc)
            assert out.shape == (length, 32)

    def test_gradient_flow(self):
        tat = TopologyAwareTransformer(
            embedding_dim=32, num_layers=2, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=4,
        )
        cc = make_chain(dim=32, length=5)
        out = tat(cc)
        loss = out.sum()
        loss.backward()
        for name, param in tat.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_param_count(self):
        tat = TopologyAwareTransformer(
            embedding_dim=256, num_layers=6, num_spatial_heads=4,
            num_spectral_heads=4, ff_dim=512, num_freqs=32,
        )
        total_params = sum(p.numel() for p in tat.parameters())
        assert 1_000_000 < total_params < 50_000_000

    def test_topological_pe_different_output(self):
        cc = make_chain(dim=32, length=5)
        tat_no_pe = TopologyAwareTransformer(
            embedding_dim=32, num_layers=1, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=4,
            use_topological_pe=False,
        )
        tat_with_pe = TopologyAwareTransformer(
            embedding_dim=32, num_layers=1, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=4,
            use_topological_pe=True,
        )
        # Copy shared weights
        tat_with_pe.blocks.load_state_dict(tat_no_pe.blocks.state_dict())
        out_no_pe = tat_no_pe(cc)
        out_with_pe = tat_with_pe(cc)
        # Outputs should differ due to PE addition
        assert not torch.allclose(out_no_pe, out_with_pe, atol=1e-4)

    def test_topological_pe_gradient_flow(self):
        cc = make_chain(dim=32, length=5)
        tat = TopologyAwareTransformer(
            embedding_dim=32, num_layers=1, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=4,
            use_topological_pe=True,
        )
        out = tat(cc)
        loss = out.sum()
        loss.backward()
        # Check that topo_pe params have gradients computed (not None)
        pe_params = [(n, p) for n, p in tat.named_parameters() if 'topo_pe' in n]
        assert len(pe_params) > 0, "No topo_pe parameters found"
        pe_with_grad = sum(1 for n, p in pe_params if p.grad is not None)
        assert pe_with_grad > 0, "No topo_pe parameters received gradients"
