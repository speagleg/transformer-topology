"""Tests for EmbeddingManifoldAnalyzer."""

import torch
import pytest


class TestEmbeddingManifoldAnalyzer:
    def test_build_cell_complex_knn(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        hidden = torch.randn(10, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        assert lcc.cc.num_cells(0) == 10
        assert lcc.cc.num_cells(1) > 0
        assert lcc.neighborhood == "knn"
        assert lcc.layer_idx == 0

    def test_build_cell_complex_mutual_knn(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="mutual_knn", k=3)
        hidden = torch.randn(10, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        assert lcc.cc.num_cells(0) == 10
        assert lcc.neighborhood == "mutual_knn"

    def test_build_adds_triangles(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        # Use very clustered points so triangles form
        hidden = torch.zeros(6, 4)
        hidden[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        hidden[1] = torch.tensor([0.01, 0.0, 0.0, 0.0])
        hidden[2] = torch.tensor([0.0, 0.01, 0.0, 0.0])
        hidden[3] = torch.tensor([10.0, 10.0, 10.0, 10.0])
        hidden[4] = torch.tensor([10.01, 10.0, 10.0, 10.0])
        hidden[5] = torch.tensor([10.0, 10.01, 10.0, 10.0])
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        assert lcc.cc.num_cells(2) > 0

    def test_analyze_layer_returns_all_invariants(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        hidden = torch.randn(10, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        result = analyzer.analyze_layer(lcc)
        assert "persistence" in result
        assert "betti" in result
        assert "spectral_gap" in result
        assert "hodge_ratios" in result
        assert isinstance(result["betti"], tuple)
        assert len(result["betti"]) == 2

    def test_cosine_similarity_edge_signal(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        hidden = torch.randn(8, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        edge_embs = lcc.cc.get_embeddings(1)
        assert edge_embs.shape[0] == lcc.cc.num_cells(1)
        assert (edge_embs[:, 0] >= -1.01).all()
        assert (edge_embs[:, 0] <= 1.01).all()

    def test_small_input(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        hidden = torch.randn(2, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        assert lcc.cc.num_cells(0) == 2
        result = analyzer.analyze_layer(lcc)
        assert "betti" in result
