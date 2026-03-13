"""Tests for v10 batch forward with attention readout and dual-track support."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.attention_readout import AttentionReadout
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop


DIM = 32


def _make_small_cc(dim=DIM, n_nodes=5):
    cc = CellComplex(dim)
    for _ in range(n_nodes):
        cc.add_0_cell(torch.randn(dim), "node")
    for i in range(n_nodes - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    cc.add_1_cell(n_nodes - 1, 0, torch.randn(dim), "edge")
    return cc


class TestV10BatchForward:

    def test_attention_readout_builds_correct_dim(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        topo = torch.randn(4)
        fw = torch.tensor(0.5)
        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=5, topo_features=topo, fusion_weight=fw,
        )
        assert combined.shape == (101,)

    def test_attention_readout_without_optional_features(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        combined = readout.build_classifier_input(h_out, 0, 3, task_id=5)
        # 32 + 32 + 32 = 96 (no topo, no fusion)
        assert combined.shape == (96,)


class TestBatchedDualTrack:

    @pytest.fixture
    def loop(self):
        return ExecutiveReasoningLoop(
            embedding_dim=DIM, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=8, tat_layers=1, tat_spatial_heads=2,
            tat_spectral_heads=2, tat_ff_dim=64,
            max_iterations=2, convergence_threshold=0.05,
            use_wave_dynamics=False,
            use_topological_pe=False,
            use_structural_features=False,
            use_dual_track=True,
        )

    def test_batched_output_shapes(self, loop):
        ccs = [_make_small_cc(DIM, n) for n in [4, 6]]
        results = loop.forward_dual_track_batched(ccs)
        assert len(results) == 2
        out0, iters0, diag0 = results[0]
        out1, iters1, diag1 = results[1]
        assert out0.shape == (4, DIM)
        assert out1.shape == (6, DIM)
        assert iters0 == 2
        assert iters1 == 2

    def test_batched_diagnostics_populated(self, loop):
        ccs = [_make_small_cc(DIM, 5)]
        results = loop.forward_dual_track_batched(ccs)
        _, _, diag = results[0]
        assert len(diag['harmonic_energies']) == 2  # pre + post
        assert len(diag['convergence_deltas']) == 1
        assert len(diag['control_signals']) == 1
        assert 'fusion_weight' in diag

    def test_batched_with_text_embeddings(self, loop):
        ccs = [_make_small_cc(DIM, n) for n in [4, 5]]
        text_embs = [torch.randn(n, DIM) for n in [4, 5]]
        results = loop.forward_dual_track_batched(
            ccs, text_embeddings_list=text_embs,
        )
        assert len(results) == 2
        for out, iters, diag in results:
            assert out.shape[1] == DIM
            assert diag.get('fusion_weight', 0) > 0  # text should activate fusion

    def test_batched_with_precomputed_pe(self):
        """Batched path with precomputed PE should work."""
        loop = ExecutiveReasoningLoop(
            embedding_dim=DIM, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=8, tat_layers=1, tat_spatial_heads=2,
            tat_spectral_heads=2, tat_ff_dim=64,
            max_iterations=2, convergence_threshold=0.05,
            use_wave_dynamics=False,
            use_topological_pe=True,
            use_structural_features=False,
            use_dual_track=True,
        )
        ccs = [_make_small_cc(DIM, n) for n in [4, 5]]
        pe_list = [loop.tat.topo_pe.precompute_features(cc) for cc in ccs]
        results = loop.forward_dual_track_batched(
            ccs, precomputed_pe_list=pe_list,
        )
        assert len(results) == 2
        assert results[0][0].shape == (4, DIM)
        assert results[1][0].shape == (5, DIM)
