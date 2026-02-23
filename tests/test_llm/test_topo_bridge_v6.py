"""Tests for v6 richer TopoBridge output."""
import torch
from src.llm.topo_bridge import TopoBridgeDecoder


class TestTopoBridgeDecoderV6:
    def test_forward_returns_four_outputs(self):
        """Decoder should return (semantic_out, semantic_bias, semantic_features, graph_embedding)."""
        dec = TopoBridgeDecoder(llm_dim=64, topo_dim=32)
        topo_memory = torch.randn(5, 64)  # 5 nodes
        llm_hidden = torch.randn(8, 64)  # 8 prefix tokens
        result = dec(topo_memory, llm_hidden)
        # Should be a 4-tuple now
        assert len(result) == 4
        semantic_out, semantic_bias, semantic_features, graph_embedding = result
        assert semantic_out.shape == (5, 32)
        assert semantic_bias.shape == (5, 5)
        assert semantic_features.shape == (5, 32)
        assert graph_embedding.shape == (32,)

    def test_semantic_bias_init_larger(self):
        """semantic_bias_proj should use std=0.1 (not 0.01)."""
        dec = TopoBridgeDecoder(llm_dim=64, topo_dim=32)
        w = dec.semantic_bias_proj.weight
        # With std=0.1, weights should have larger magnitude than old std=0.01
        assert w.std().item() > 0.05

    def test_batched_returns_four_tuples(self):
        """Batched decode should return list of 4-tuples."""
        dec = TopoBridgeDecoder(llm_dim=64, topo_dim=32)
        topo_memory = torch.randn(10, 2, 64)  # max_N=10, B=2
        llm_hidden = torch.randn(8, 2, 64)    # K=8, B=2
        memory_mask = torch.zeros(2, 10, dtype=torch.bool)
        memory_mask[1, 7:] = True  # second graph has 7 nodes
        sizes = [10, 7]
        results = dec.forward_batched(topo_memory, llm_hidden, memory_mask, sizes)
        assert len(results) == 2
        for i, (so, sb, sf, ge) in enumerate(results):
            n = sizes[i]
            assert so.shape == (n, 32)
            assert sb.shape == (n, n)
            assert sf.shape == (n, 32)
            assert ge.shape == (32,)
