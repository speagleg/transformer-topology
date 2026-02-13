import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.cell_complex.structural_features import StructuralFeatureEncoder


class TestComputeStructuralFeatures:
    def test_output_shape(self):
        cc = CellComplex(embedding_dim=16)
        cc.add_0_cell(torch.randn(16), "source")
        cc.add_0_cell(torch.randn(16), "concept")
        cc.add_0_cell(torch.randn(16), "target")
        cc.add_1_cell(0, 1, torch.randn(16), "r")
        cc.add_1_cell(1, 2, torch.randn(16), "r")
        features = cc.compute_structural_features()
        assert features.shape == (3, 5)

    def test_degree_normalization(self):
        cc = CellComplex(embedding_dim=8)
        cc.add_0_cell(torch.randn(8), "concept")
        cc.add_0_cell(torch.randn(8), "concept")
        cc.add_0_cell(torch.randn(8), "concept")
        cc.add_1_cell(0, 1, torch.randn(8), "r")
        cc.add_1_cell(0, 2, torch.randn(8), "r")
        features = cc.compute_structural_features()
        # Node 0 has degree 2 (max), nodes 1,2 have degree 1
        assert features[0, 0].item() == pytest.approx(1.0)
        assert features[1, 0].item() == pytest.approx(0.5)

    def test_type_indicators(self):
        cc = CellComplex(embedding_dim=8)
        cc.add_0_cell(torch.randn(8), "source")
        cc.add_0_cell(torch.randn(8), "target")
        cc.add_0_cell(torch.randn(8), "blocked")
        cc.add_0_cell(torch.randn(8), "source2")
        cc.add_0_cell(torch.randn(8), "concept")
        features = cc.compute_structural_features()
        assert features[0, 1] == 1.0  # is_source
        assert features[1, 2] == 1.0  # is_target
        assert features[2, 3] == 1.0  # is_blocked
        assert features[3, 4] == 1.0  # is_source2
        assert features[4, 1:].sum() == 0.0  # concept has no type indicators

    def test_empty_complex(self):
        cc = CellComplex(embedding_dim=8)
        features = cc.compute_structural_features()
        assert features.shape == (0, 5)


class TestStructuralFeatureEncoder:
    def test_output_shape(self):
        cc = CellComplex(embedding_dim=32)
        cc.add_0_cell(torch.randn(32), "source")
        cc.add_0_cell(torch.randn(32), "target")
        cc.add_1_cell(0, 1, torch.randn(32), "r")
        encoder = StructuralFeatureEncoder(embedding_dim=32)
        out = encoder(cc)
        assert out.shape == (2, 32)

    def test_gradient_flow(self):
        cc = CellComplex(embedding_dim=16)
        cc.add_0_cell(torch.randn(16), "source")
        cc.add_0_cell(torch.randn(16), "target")
        cc.add_1_cell(0, 1, torch.randn(16), "r")
        encoder = StructuralFeatureEncoder(embedding_dim=16)
        out = encoder(cc)
        loss = out.sum()
        loss.backward()
        assert encoder.proj.weight.grad is not None
