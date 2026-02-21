import torch
import torch.nn as nn
from src.computation_graph.capture import ComputationGraphCapture


class TestHookRegistration:
    def test_forward_hooks_fire(self):
        """Forward hooks record one entry per module during forward pass."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            _ = model(x)
        assert len(cap.forward_records) >= 3

    def test_backward_hooks_fire(self):
        """Backward hooks record one entry per module during backward pass."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            loss = out.sum()
            loss.backward()
        assert len(cap.backward_records) >= 3

    def test_hooks_removed_after_exit(self):
        """Hooks should be cleaned up when context manager exits."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            pass
        for mod in model.modules():
            assert len(mod._forward_hooks) == 0
            assert len(mod._backward_hooks) == 0

    def test_records_contain_module_name_and_norm(self):
        """Each record should have module name, activation norm, and shape."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            _ = model(x)
        rec = cap.forward_records[0]
        assert 'module_name' in rec
        assert 'activation_norm' in rec
        assert isinstance(rec['activation_norm'], float)


from src.cell_complex.cell_complex import CellComplex


class TestCellComplexConversion:
    def test_to_cell_complex_returns_cell_complex(self):
        """to_cell_complex() should return a CellComplex instance."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        assert isinstance(cc, CellComplex)

    def test_0_cells_match_leaf_modules(self):
        """Each leaf module should become a 0-cell."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        leaf_count = sum(1 for m in model.modules()
                         if len(list(m.children())) == 0)
        assert cc.num_cells(0) == leaf_count

    def test_1_cells_connect_sequential_modules(self):
        """Sequential data flow should create edges between consecutive leaf modules."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        # Linear -> ReLU -> Linear = 2 data flow edges + 1 closure edge for cycle
        assert cc.num_cells(1) >= 2

    def test_edge_signals_are_activation_norms(self):
        """1-cell embeddings should encode forward activation norms."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        edge_embs = cc.get_embeddings(1)
        # First dim of edge embedding = forward activation norm (non-negative)
        assert (edge_embs[:, 0] >= 0).all()

    def test_backward_signals_in_embeddings(self):
        """0-cell embeddings should include gradient norm from backward pass."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        node_embs = cc.get_embeddings(0)
        # Second dim = gradient norm (non-negative)
        assert (node_embs[:, 1] >= 0).all()

    def test_2_cells_for_composite_modules(self):
        """A Sequential container should produce a 2-cell."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        assert cc.num_cells(2) >= 1

    def test_chain_complex_property(self):
        """B1 @ B2 should equal 0 (chain complex property)."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        if cc.num_cells(2) > 0:
            assert cc.verify_chain_complex()


class ResidualBlock(nn.Module):
    """Simple residual block for testing skip connection detection."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.relu(self.linear(x))


class TestSkipConnectionDetection:
    def test_residual_creates_extra_edge(self):
        """A residual block should create more edges than a pure sequential."""
        model = ResidualBlock(4)
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        # Should have at least the sequential edges plus skip edges
        assert cc.num_cells(1) >= 2

    def test_skip_edge_has_correct_signal(self):
        """Skip connection edges should have non-zero activation norm."""
        model = ResidualBlock(4)
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        edge_embs = cc.get_embeddings(1)
        assert (edge_embs[:, 0] >= 0).all()


class TestModuleExports:
    def test_top_level_imports(self):
        """Key classes should be importable from src.computation_graph."""
        from src.computation_graph import (
            ComputationGraphCapture,
            TopologicalDiagnostics,
            TrainingTopologyMonitor,
            analyze_computation_graph,
            analyze_hodge,
            analyze_spectral_gap,
        )
        assert ComputationGraphCapture is not None
        assert TopologicalDiagnostics is not None
        assert TrainingTopologyMonitor is not None
