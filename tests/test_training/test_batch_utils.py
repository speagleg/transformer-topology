"""Tests for graph-level mini-batching utilities."""

import torch
import torch.nn as nn
import pytest

from src.cell_complex.cell_complex import CellComplex
from src.llm.dsm import DistilledSemanticModel, DSMBlock, DSMCrossAttentionBlock
from src.llm.dsm_backend import DSMBackend
from src.llm.topo_bridge import TopoBridge, TopoBridgeEncoder, TopoBridgeDecoder
from src.llm.backend import MockLLMBackend


def _make_simple_cc(n_nodes=5, embedding_dim=8):
    """Create a simple CellComplex for testing."""
    cc = CellComplex(embedding_dim)
    for i in range(n_nodes):
        cc.add_0_cell(torch.randn(embedding_dim), "node")
    for i in range(n_nodes - 1):
        cc.add_1_cell(i, i + 1, torch.randn(embedding_dim), "edge")
    return cc


class TestDSMBatchSupport:
    """Test that DSM accepts batched inputs with memory_key_padding_mask."""

    def test_dsm_block_unbatched(self):
        """DSMBlock works with unbatched 2D input (backward compat)."""
        block = DSMBlock(hidden_dim=32, num_heads=4, ff_dim=64)
        x = torch.randn(8, 32)  # (seq, dim)
        out = block(x)
        assert out.shape == (8, 32)

    def test_dsm_block_batched(self):
        """DSMBlock works with batched 3D input."""
        block = DSMBlock(hidden_dim=32, num_heads=4, ff_dim=64)
        x = torch.randn(8, 4, 32)  # (seq, batch, dim)
        out = block(x)
        assert out.shape == (8, 4, 32)

    def test_dsm_cross_attn_unbatched(self):
        """DSMCrossAttentionBlock works without mask (backward compat)."""
        block = DSMCrossAttentionBlock(hidden_dim=32, num_heads=4, ff_dim=64)
        x = torch.randn(8, 32)
        memory = torch.randn(10, 32)
        out = block(x, memory)
        assert out.shape == (8, 32)

    def test_dsm_cross_attn_batched_with_mask(self):
        """DSMCrossAttentionBlock handles batched input with padding mask."""
        block = DSMCrossAttentionBlock(hidden_dim=32, num_heads=4, ff_dim=64)
        B, seq, max_N, dim = 4, 8, 10, 32
        x = torch.randn(seq, B, dim)
        memory = torch.randn(max_N, B, dim)
        # Mask: True = ignore. Last 3 positions are padding for all samples
        mask = torch.zeros(B, max_N, dtype=torch.bool)
        mask[:, 7:] = True
        out = block(x, memory, memory_key_padding_mask=mask)
        assert out.shape == (seq, B, dim)

    def test_dsm_full_batched(self):
        """Full DSM forward with batched inputs and memory mask."""
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=4, cross_attn_layer=1,
        )
        B, K, max_N = 4, 8, 12
        prefix = torch.randn(K, B, 32)
        topo_memory = torch.randn(max_N, B, 32)
        mask = torch.zeros(B, max_N, dtype=torch.bool)
        mask[:, 10:] = True  # Last 2 positions padded
        out = dsm(prefix, topo_memory, memory_key_padding_mask=mask)
        assert out.shape == (K, B, 32)

    def test_dsm_unbatched_backward_compat(self):
        """DSM forward still works with unbatched 2D inputs."""
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(8, 32)
        topo_memory = torch.randn(10, 32)
        out = dsm(prefix, topo_memory)
        assert out.shape == (8, 32)

    def test_dsm_backend_batched(self):
        """DSMBackend passes memory_key_padding_mask through to DSM."""
        backend = DSMBackend(
            dsm_dim=32, num_heads=4, ff_dim=64,
            num_layers=4, cross_attn_layer=1,
        )
        B = 3
        prefix = torch.randn(8, B, 32)
        topo_memory = torch.randn(10, B, 32)
        mask = torch.zeros(B, 10, dtype=torch.bool)
        out = backend.forward(prefix, topo_memory, memory_key_padding_mask=mask)
        assert out.shape == (8, B, 32)


class TestTopoBridgeBatchSupport:
    """Test batched TopoBridge encoder/decoder/full."""

    def test_encoder_batched(self):
        """TopoBridgeEncoder.forward_batched pads and batches correctly."""
        enc = TopoBridgeEncoder(topo_dim=8, llm_dim=32, num_prefix=4)
        # 3 graphs with different node counts
        ne_list = [torch.randn(5, 8), torch.randn(8, 8), torch.randn(3, 8)]
        topo_memory, prefix, mask, sizes = enc.forward_batched(ne_list)

        assert sizes == [5, 8, 3]
        assert topo_memory.shape == (8, 3, 32)  # (max_N, B, llm_dim)
        assert prefix.shape == (4, 3, 32)  # (K, B, llm_dim)
        assert mask.shape == (3, 8)  # (B, max_N)
        # Check mask: graph 0 has 5 nodes → positions 5-7 masked
        assert not mask[0, :5].any()
        assert mask[0, 5:].all()
        # Graph 1 has 8 nodes → no masking
        assert not mask[1].any()
        # Graph 2 has 3 nodes → positions 3-7 masked
        assert not mask[2, :3].any()
        assert mask[2, 3:].all()

    def test_decoder_batched(self):
        """TopoBridgeDecoder.forward_batched unpads correctly."""
        dec = TopoBridgeDecoder(topo_dim=8, llm_dim=32)
        B, max_N, K = 3, 8, 4
        topo_memory = torch.randn(max_N, B, 32)
        llm_hidden = torch.randn(K, B, 32)
        mask = torch.zeros(B, max_N, dtype=torch.bool)
        sizes = [5, 8, 3]
        for i, n in enumerate(sizes):
            mask[i, n:] = True

        results = dec.forward_batched(topo_memory, llm_hidden, mask, sizes)
        assert len(results) == 3
        assert results[0][0].shape == (5, 8)  # (N_0, topo_dim)
        assert results[0][1].shape == (5, 5)  # semantic_bias
        assert results[1][0].shape == (8, 8)
        assert results[1][1].shape == (8, 8)
        assert results[2][0].shape == (3, 8)
        assert results[2][1].shape == (3, 3)

    def test_full_topo_bridge_batched(self):
        """TopoBridge.forward_batched end-to-end with DSM backend."""
        backend = DSMBackend(
            dsm_dim=32, num_heads=4, ff_dim=64,
            num_layers=4, cross_attn_layer=1,
        )
        bridge = TopoBridge(
            backend=backend, topo_dim=8, llm_dim=32, num_prefix=4,
        )
        ne_list = [torch.randn(5, 8), torch.randn(10, 8), torch.randn(7, 8)]
        sw_list = [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.8)]

        results = bridge.forward_batched(ne_list, sw_list)
        assert len(results) == 3
        for i, (llm_out, semantic_bias, _, _) in enumerate(results):
            n = ne_list[i].shape[0]
            assert llm_out.shape == (n, 8)
            assert semantic_bias.shape == (n, n)

    def test_batched_matches_sequential(self):
        """Batched forward should produce same-shaped outputs as sequential."""
        torch.manual_seed(42)
        backend = DSMBackend(
            dsm_dim=32, num_heads=4, ff_dim=64,
            num_layers=2, cross_attn_layer=0,
        )
        bridge = TopoBridge(
            backend=backend, topo_dim=8, llm_dim=32, num_prefix=4,
        )

        ne_list = [torch.randn(5, 8), torch.randn(7, 8)]
        sw_list = [torch.tensor(0.5), torch.tensor(0.5)]

        # Sequential
        seq_results = []
        for ne, sw in zip(ne_list, sw_list):
            out, bias, _, _ = bridge(ne, sw)
            seq_results.append((out.shape, bias.shape))

        # Batched
        batch_results = bridge.forward_batched(ne_list, sw_list)
        for i, (llm_out, bias, _, _) in enumerate(batch_results):
            assert llm_out.shape == seq_results[i][0]
            assert bias.shape == seq_results[i][1]

    def test_batched_gradients_flow(self):
        """Gradients flow through the batched TopoBridge."""
        backend = DSMBackend(
            dsm_dim=32, num_heads=4, ff_dim=64,
            num_layers=2, cross_attn_layer=0,
        )
        bridge = TopoBridge(
            backend=backend, topo_dim=8, llm_dim=32, num_prefix=4,
        )
        ne_list = [torch.randn(5, 8, requires_grad=True),
                    torch.randn(3, 8, requires_grad=True)]
        sw_list = [torch.tensor(0.5), torch.tensor(0.5)]

        results = bridge.forward_batched(ne_list, sw_list)
        loss = sum(r[0].sum() + r[1].sum() for r in results)
        loss.backward()

        # Check that gradients flow to inputs
        for ne in ne_list:
            assert ne.grad is not None
            assert ne.grad.abs().sum() > 0

        # Check bridge parameters got gradients
        grad_count = sum(1 for p in bridge.parameters() if p.grad is not None)
        assert grad_count > 0


class TestBatchedExecutiveLoop:
    """Test ExecutiveReasoningLoop.forward_batched."""

    def test_forward_batched_without_dsm(self):
        """Batched loop works without DSM (sequential fallback for all phases)."""
        from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop

        loop = ExecutiveReasoningLoop(
            embedding_dim=8, gnn_hidden=16,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=16, max_iterations=2,
            use_wave_dynamics=False,
        )

        ccs = [_make_simple_cc(5, 8), _make_simple_cc(7, 8)]
        results = loop.forward_batched(ccs)

        assert len(results) == 2
        emb0, iters0, diag0 = results[0]
        emb1, iters1, diag1 = results[1]
        assert emb0.shape == (5, 8)
        assert emb1.shape == (7, 8)
        assert iters0 == 2  # max_iterations
        assert iters1 == 2

    def test_forward_batched_with_dsm(self):
        """Batched loop with DSM batches the DSM call across graphs."""
        from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop

        loop = ExecutiveReasoningLoop(
            embedding_dim=8, gnn_hidden=16,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=16, max_iterations=2,
            use_wave_dynamics=False,
            use_dsm=True,
            dsm_config={
                'dsm_dim': 32, 'num_heads': 4, 'ff_dim': 64,
                'num_layers': 2, 'cross_attn_layer': 0,
                'num_prefix': 4,
            },
        )

        ccs = [_make_simple_cc(5, 8), _make_simple_cc(8, 8), _make_simple_cc(4, 8)]
        results = loop.forward_batched(ccs)

        assert len(results) == 3
        for g, (emb, iters, diag) in enumerate(results):
            expected_n = [5, 8, 4][g]
            assert emb.shape == (expected_n, 8)
            assert iters == 2
            assert len(diag['control_signals']) == 2

    def test_forward_batched_single_graph(self):
        """Batched loop handles batch_size=1 correctly."""
        from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop

        loop = ExecutiveReasoningLoop(
            embedding_dim=8, gnn_hidden=16,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=16, max_iterations=2,
            use_wave_dynamics=False,
        )

        ccs = [_make_simple_cc(6, 8)]
        results = loop.forward_batched(ccs)
        assert len(results) == 1
        assert results[0][0].shape == (6, 8)


class TestBatchedTraining:
    """Test batched training and evaluation utilities."""

    def _build_model_and_dataset(self, use_dsm=False):
        """Build a small model and dummy dataset for testing."""
        from src.benchmarks.run_benchmark_suite import _build_model
        from src.benchmarks.benchmark_dataset import BenchmarkDataset

        mc = {
            'embedding_dim': 8, 'gnn_hidden': 16,
            'gnn_spatial_layers': 1, 'gnn_spectral_layers': 1,
            'max_freqs': 4, 'tat_layers': 1,
            'tat_spatial_heads': 2, 'tat_spectral_heads': 2,
            'tat_ff_dim': 16, 'max_iterations': 2,
            'convergence_threshold': 0.1,
            'use_topological_pe': False,
            'use_structural_features': False,
            'use_wave_dynamics': False,
            'use_higher_order': False,
        }

        if use_dsm:
            llm_config = {
                'backend': 'dsm',
                'dsm_dim': 32, 'num_heads': 4, 'ff_dim': 64,
                'num_layers': 2, 'cross_attn_layer': 0,
                'num_prefix': 4,
            }
            model = _build_model('hierarchical_llm', mc, 11, torch.device('cpu'),
                                 llm_config=llm_config)
        else:
            model = _build_model('hierarchical', mc, 11, torch.device('cpu'))

        dataset = BenchmarkDataset(20, 'diverse', 10, 8)
        return model, dataset

    def test_train_epoch_batched_sequential(self):
        """Batched training epoch works with non-DSM model (sequential path)."""
        from src.training.batch_utils import train_epoch_batched

        model, dataset = self._build_model_and_dataset(use_dsm=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = train_epoch_batched(
            model, dataset, optimizer, batch_size=4,
            device=torch.device('cpu'),
        )
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_batched_dsm(self):
        """Batched training epoch works with DSM model (batched DSM path)."""
        from src.training.batch_utils import train_epoch_batched

        model, dataset = self._build_model_and_dataset(use_dsm=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = train_epoch_batched(
            model, dataset, optimizer, batch_size=4,
            device=torch.device('cpu'),
        )
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_batched(self):
        """Batched evaluation returns valid accuracy and loss."""
        from src.training.batch_utils import evaluate_batched

        model, dataset = self._build_model_and_dataset(use_dsm=False)
        acc, loss = evaluate_batched(
            model, dataset, batch_size=4,
            device=torch.device('cpu'),
        )
        assert 0.0 <= acc <= 1.0
        assert loss > 0

    def test_evaluate_batched_dsm(self):
        """Batched evaluation works with DSM model."""
        from src.training.batch_utils import evaluate_batched

        model, dataset = self._build_model_and_dataset(use_dsm=True)
        acc, loss = evaluate_batched(
            model, dataset, batch_size=4,
            device=torch.device('cpu'),
        )
        assert 0.0 <= acc <= 1.0
        assert loss > 0

    def test_batch_size_one(self):
        """Batch size 1 should work (degenerate case)."""
        from src.training.batch_utils import train_epoch_batched

        model, dataset = self._build_model_and_dataset(use_dsm=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = train_epoch_batched(
            model, dataset, optimizer, batch_size=1,
            device=torch.device('cpu'),
        )
        assert isinstance(loss, float)

    def test_batch_size_larger_than_dataset(self):
        """Batch size larger than dataset should work (single batch)."""
        from src.training.batch_utils import train_epoch_batched

        model, dataset = self._build_model_and_dataset(use_dsm=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = train_epoch_batched(
            model, dataset, optimizer, batch_size=100,
            device=torch.device('cpu'),
        )
        assert isinstance(loss, float)

    def test_collate_fn(self):
        """graph_collate_fn correctly unpacks samples."""
        from src.training.batch_utils import graph_collate_fn

        _, dataset = self._build_model_and_dataset(use_dsm=False)
        samples = [dataset[i] for i in range(3)]
        collated = graph_collate_fn(samples)
        assert len(collated) == 3
        for cc, query, target, answer, metadata in collated:
            assert isinstance(cc, CellComplex)
            assert isinstance(query, int)
            assert isinstance(target, int)
