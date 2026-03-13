"""Tests for precomputed TopologicalPE path (Task 2: GPU optimization)."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.positional_encoding import TopologicalPositionalEncoding
from src.benchmarks.benchmark_dataset import precompute_pe_for_sample


def _make_test_cc(dim=32):
    cc = CellComplex(dim)
    for i in range(5):
        cc.add_0_cell(torch.randn(dim), "node")
    for i in range(4):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    cc.add_1_cell(4, 0, torch.randn(dim), "edge")
    return cc


class TestPrecomputedPE:
    def test_precomputed_matches_live(self):
        """Precomputed path should produce identical output to live computation."""
        torch.manual_seed(42)
        cc = _make_test_cc()
        tpe = TopologicalPositionalEncoding(32, num_eigenvectors=8,
                                            num_persistence_features=8,
                                            max_2_cells=32)
        # Compute live
        live_pe = tpe(cc)
        # Precompute features
        precomputed = tpe.precompute_features(cc)
        # Use precomputed
        fast_pe = tpe(cc, precomputed_features=precomputed)
        assert torch.allclose(live_pe, fast_pe, atol=1e-5)

    def test_precomputed_shape(self):
        cc = _make_test_cc()
        tpe = TopologicalPositionalEncoding(32, num_eigenvectors=8,
                                            num_persistence_features=8,
                                            max_2_cells=32)
        precomputed = tpe.precompute_features(cc)
        assert precomputed.shape == (5, 8 + 8 + 32)  # eigvecs + persistence + membership

    def test_standalone_precompute_matches_module(self):
        """Standalone precompute_pe_for_sample should match module's precompute."""
        torch.manual_seed(42)
        cc = _make_test_cc()
        tpe = TopologicalPositionalEncoding(32, num_eigenvectors=8,
                                            num_persistence_features=8,
                                            max_2_cells=32)
        module_features = tpe.precompute_features(cc)
        standalone_features = precompute_pe_for_sample(cc)
        assert torch.allclose(module_features, standalone_features, atol=1e-5)

    def test_precomputed_forward_output_shape(self):
        cc = _make_test_cc()
        tpe = TopologicalPositionalEncoding(32, num_eigenvectors=8,
                                            num_persistence_features=8,
                                            max_2_cells=32)
        precomputed = tpe.precompute_features(cc)
        result = tpe(cc, precomputed_features=precomputed)
        assert result.shape == (5, 32)

    def test_tat_with_precomputed_pe(self):
        """Full TAT forward with precomputed PE should work."""
        from src.tat.transformer import TopologyAwareTransformer
        cc = _make_test_cc()
        tat = TopologyAwareTransformer(
            embedding_dim=32, num_layers=1, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=8,
            use_topological_pe=True,
        )
        tat.eval()  # Disable dropout so outputs are deterministic
        precomputed = tat.topo_pe.precompute_features(cc)
        # Without precomputed PE (live)
        out_live = tat(cc)
        # With precomputed PE (should match since precomputed == live features)
        out_fast = tat(cc, precomputed_pe=precomputed)
        assert torch.allclose(out_live, out_fast, atol=1e-5)

    def test_tat_without_pe_ignores_precomputed(self):
        """TAT without use_topological_pe should ignore precomputed_pe."""
        from src.tat.transformer import TopologyAwareTransformer
        cc = _make_test_cc()
        tat = TopologyAwareTransformer(
            embedding_dim=32, num_layers=1, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=8,
            use_topological_pe=False,
        )
        fake_pe = torch.randn(5, 48)
        out = tat(cc, precomputed_pe=fake_pe)
        assert out.shape == (5, 32)
