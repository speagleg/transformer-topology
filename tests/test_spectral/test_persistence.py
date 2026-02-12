import torch
import pytest
import numpy as np
from src.cell_complex.cell_complex import CellComplex
from src.spectral.persistence import (
    compute_persistence_diagram,
    vectorize_persistence,
    persistence_node_features,
    GUDHI_AVAILABLE,
)


def make_simple_complex(dim=8):
    cc = CellComplex(embedding_dim=dim)
    # Place nodes at known positions (use first 3 dims as coordinates)
    for i in range(5):
        emb = torch.zeros(dim)
        emb[0] = float(i)
        cc.add_0_cell(emb, "node")
    # Add edges
    for i in range(4):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    return cc


@pytest.mark.skipif(not GUDHI_AVAILABLE, reason="gudhi not installed")
class TestPersistenceDiagram:
    def test_diagram_shape(self):
        cc = make_simple_complex()
        diagrams = compute_persistence_diagram(cc, max_dimension=1)
        assert len(diagrams) == 2  # dim 0 and dim 1
        for d in diagrams:
            assert d.ndim == 2 or d.shape[0] == 0

    def test_diagram_birth_before_death(self):
        cc = make_simple_complex()
        diagrams = compute_persistence_diagram(cc, max_dimension=1)
        for d in diagrams:
            if d.shape[0] > 0:
                assert (d[:, 1] >= d[:, 0]).all()


@pytest.mark.skipif(not GUDHI_AVAILABLE, reason="gudhi not installed")
class TestVectorization:
    def test_vectorization_shape(self):
        cc = make_simple_complex()
        diagrams = compute_persistence_diagram(cc, max_dimension=1)
        vec = vectorize_persistence(diagrams, num_features=16)
        assert vec.shape == (32,)  # 2 dimensions * 16 features each


@pytest.mark.skipif(not GUDHI_AVAILABLE, reason="gudhi not installed")
class TestNodeFeatures:
    def test_node_features_shape(self):
        cc = make_simple_complex()
        features = persistence_node_features(cc, num_features=8)
        assert features.shape == (5, 8)


class TestGracefulFallback:
    def test_fallback_diagrams(self):
        """Even without gudhi, should return empty arrays."""
        cc = make_simple_complex()
        diagrams = compute_persistence_diagram(cc, max_dimension=1)
        assert len(diagrams) == 2

    def test_fallback_node_features(self):
        """Even without gudhi, should return zeros."""
        cc = make_simple_complex()
        features = persistence_node_features(cc, num_features=8)
        assert features.shape == (5, 8)
