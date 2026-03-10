"""Tests for QwenTextFeatureExtractor (Approach C) and updated QwenBridgeAdapter."""
import torch
import pytest

from src.llm.qwen_text_features import QwenTextFeatureExtractor
from src.llm.qwen_bridge_adapter import QwenBridgeAdapter


class TestQwenTextFeatureExtractor:
    def test_mock_forward_shape(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        out = ext(["dog", "cat", "animal"], device=torch.device('cpu'))
        assert out.shape == (3, 32)

    def test_mock_forward_none(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        assert ext(None, device=torch.device('cpu')) is None

    def test_mock_forward_empty(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        assert ext([], device=torch.device('cpu')) is None

    def test_cache_consistency(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        out1 = ext(["dog", "cat"], device=torch.device('cpu'))
        out2 = ext(["dog", "cat"], device=torch.device('cpu'))
        # Raw embeddings are cached and deterministic; proj weights are the same
        torch.testing.assert_close(out1, out2)

    def test_cache_shared_across_calls(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        _ = ext(["dog", "cat"], device=torch.device('cpu'))
        assert "dog" in ext._cache
        assert "cat" in ext._cache

    def test_clear_cache(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        _ = ext(["dog"], device=torch.device('cpu'))
        assert len(ext._cache) == 1
        ext.clear_cache()
        assert len(ext._cache) == 0

    def test_proj_has_gradients(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        out = ext(["dog", "cat"], device=torch.device('cpu'))
        loss = out.sum()
        loss.backward()
        # proj Linear should have gradients
        assert ext.proj[0].weight.grad is not None

    def test_embed_tokens_frozen(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        for p in ext._embed_tokens.parameters():
            assert not p.requires_grad

    def test_single_concept(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=16, use_mock=True)
        out = ext(["hello"], device=torch.device('cpu'))
        assert out.shape == (1, 16)

    def test_precompute(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        new_count = ext.precompute(["dog", "cat", "bird"])
        assert new_count == 3
        assert len(ext._cache) == 3
        # Re-precompute should add no new entries
        new_count2 = ext.precompute(["dog", "cat", "fish"])
        assert new_count2 == 1  # only "fish" is new
        assert len(ext._cache) == 4

    def test_save_load_cache(self, tmp_path):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        ext.precompute(["dog", "cat", "bird"])
        cache_path = tmp_path / "cache.pt"
        ext.save_cache(cache_path)
        assert cache_path.exists()

        ext2 = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        loaded = ext2.load_cache(cache_path)
        assert loaded == 3
        assert "dog" in ext2._cache
        # Embeddings should match
        torch.testing.assert_close(ext._cache["dog"], ext2._cache["dog"])

    def test_build_gpu_cache(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        ext.precompute(["dog", "cat", "bird"])
        ext.build_gpu_cache(torch.device('cpu'))
        assert ext._gpu_cache is not None
        assert ext._gpu_cache.shape == (3, 64)
        assert ext._concept_to_idx is not None
        assert len(ext._concept_to_idx) == 3

    def test_gpu_cache_forward_matches_slow(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        dev = torch.device('cpu')
        concepts = ["dog", "cat", "bird"]

        # Slow path
        out_slow = ext(concepts, device=dev)

        # Build GPU cache, re-run
        ext.build_gpu_cache(dev)
        out_fast = ext(concepts, device=dev)
        torch.testing.assert_close(out_slow, out_fast)

    def test_gpu_cache_unknown_concept(self):
        """GPU cache handles concepts not in the pre-computed set."""
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        ext.precompute(["dog", "cat"])
        ext.build_gpu_cache(torch.device('cpu'))
        # "bird" is unknown — should fall back to _get_raw_embedding
        out = ext(["dog", "bird"], device=torch.device('cpu'))
        assert out.shape == (2, 32)

    def test_clear_cache_clears_gpu(self):
        ext = QwenTextFeatureExtractor(llm_dim=64, text_feat_dim=32, use_mock=True)
        ext.precompute(["dog", "cat"])
        ext.build_gpu_cache(torch.device('cpu'))
        assert ext._gpu_cache is not None
        ext.clear_cache()
        assert ext._gpu_cache is None
        assert ext._concept_to_idx is None
        assert len(ext._cache) == 0


class TestQwenBridgeAdapter:
    def test_extract_text_features_shape(self):
        config = {'llm_dim': 64, 'use_mock': True}
        adapter = QwenBridgeAdapter(config, topo_dim=32)
        out = adapter.extract_text_features(["dog", "cat"], torch.device('cpu'))
        assert out.shape == (2, 32)

    def test_extract_text_features_none(self):
        config = {'llm_dim': 64, 'use_mock': True}
        adapter = QwenBridgeAdapter(config, topo_dim=32)
        assert adapter.extract_text_features(None, torch.device('cpu')) is None

    def test_forward_passthrough(self):
        config = {'llm_dim': 64, 'use_mock': True}
        adapter = QwenBridgeAdapter(config, topo_dim=32)
        node_embs = torch.randn(5, 32)
        sw = torch.tensor(0.5)
        out = adapter(node_embs, sw, task_text="test", node_texts=["a", "b", "c", "d", "e"])
        sem_feat, sem_bias, sem_feat2, graph_emb = out
        # Passthrough: sem_feat == node_embs
        torch.testing.assert_close(sem_feat, node_embs)
        assert sem_bias.shape == (5, 5)
        assert graph_emb.shape == (32,)

    def test_has_extract_text_features(self):
        config = {'llm_dim': 64, 'use_mock': True}
        adapter = QwenBridgeAdapter(config, topo_dim=32)
        assert hasattr(adapter, 'extract_text_features')


class TestApproachCIntegration:
    """Integration test: build HierarchicalMultiHopModel with Qwen Approach C."""

    def test_classifier_input_dim_with_text(self):
        from src.benchmarks.run_comparison import HierarchicalMultiHopModel
        model = HierarchicalMultiHopModel(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=16, tat_layers=2,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=128, max_classes=10,
            max_iterations=2, convergence_threshold=0.05,
            use_llm=True,
            llm_config={'backend': 'qwen', 'llm_dim': 64, 'use_mock': True},
        )
        # Dual-path: text injected into node embs + concatenated to classifier
        # base = 3*32 + 3 + 1 + 32 = 132, text = 3*32 = 96, text_reasoning = 3*64 = 192
        assert model.text_feat_dim == 32
        assert model.text_reasoning_dim == 64
        assert model.classifier_input_dim == 132 + 96 + 192

    def test_classifier_input_dim_without_text(self):
        from src.benchmarks.run_comparison import HierarchicalMultiHopModel
        model = HierarchicalMultiHopModel(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=16, tat_layers=2,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=128, max_classes=10,
            max_iterations=2, convergence_threshold=0.05,
        )
        assert model.text_feat_dim == 0
        assert model.classifier_input_dim == 132

    def test_forward_with_text_features(self):
        from src.benchmarks.run_comparison import HierarchicalMultiHopModel
        from src.cell_complex.cell_complex import CellComplex
        dim = 32
        model = HierarchicalMultiHopModel(
            embedding_dim=dim, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=16, tat_layers=2,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=128, max_classes=10,
            max_iterations=2, convergence_threshold=0.05,
            use_llm=True,
            llm_config={'backend': 'qwen', 'llm_dim': 64, 'use_mock': True},
        )
        cc = CellComplex(embedding_dim=dim)
        nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(5)]
        cc.add_1_cell(nodes[0], nodes[1], torch.randn(dim), "edge")
        cc.add_1_cell(nodes[1], nodes[2], torch.randn(dim), "edge")
        cc.add_1_cell(nodes[2], nodes[3], torch.randn(dim), "edge")
        cc.add_1_cell(nodes[3], nodes[4], torch.randn(dim), "edge")
        cc.node_texts = ["dog", "cat", "animal", "pet", "bird"]

        logits = model(cc, query_node=0, target_node=2)
        assert logits.shape == (10,)

    def test_gpu_cache_forward(self):
        """Model forward pass works with GPU cache enabled."""
        from src.benchmarks.run_comparison import HierarchicalMultiHopModel
        from src.cell_complex.cell_complex import CellComplex
        dim = 32
        model = HierarchicalMultiHopModel(
            embedding_dim=dim, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=16, tat_layers=2,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=128, max_classes=10,
            max_iterations=2, convergence_threshold=0.05,
            use_llm=True,
            llm_config={'backend': 'qwen', 'llm_dim': 64, 'use_mock': True},
        )
        cc = CellComplex(embedding_dim=dim)
        nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(5)]
        cc.add_1_cell(nodes[0], nodes[1], torch.randn(dim), "edge")
        cc.add_1_cell(nodes[1], nodes[2], torch.randn(dim), "edge")
        cc.add_1_cell(nodes[2], nodes[3], torch.randn(dim), "edge")
        cc.add_1_cell(nodes[3], nodes[4], torch.randn(dim), "edge")
        cc.node_texts = ["dog", "cat", "animal", "pet", "bird"]

        # Build GPU cache before forward
        text_ext = model.topo_bridge.text_extractor
        text_ext.precompute(["dog", "cat", "animal", "pet", "bird"])
        text_ext.build_gpu_cache(torch.device('cpu'))
        assert text_ext._gpu_cache is not None

        logits = model(cc, query_node=0, target_node=2)
        assert logits.shape == (10,)

    def test_forward_without_text_no_crash(self):
        from src.benchmarks.run_comparison import HierarchicalMultiHopModel
        from src.cell_complex.cell_complex import CellComplex
        dim = 32
        model = HierarchicalMultiHopModel(
            embedding_dim=dim, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=16, tat_layers=2,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=128, max_classes=10,
            max_iterations=2, convergence_threshold=0.05,
            use_llm=True,
            llm_config={'backend': 'qwen', 'llm_dim': 64, 'use_mock': True},
        )
        # No node_texts — should zero-pad
        cc = CellComplex(embedding_dim=dim)
        nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(5)]
        cc.add_1_cell(nodes[0], nodes[1], torch.randn(dim), "edge")
        cc.add_1_cell(nodes[1], nodes[2], torch.randn(dim), "edge")

        logits = model(cc, query_node=0, target_node=2)
        assert logits.shape == (10,)
