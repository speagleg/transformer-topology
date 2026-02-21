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
