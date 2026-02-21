import torch
import torch.nn as nn
from src.computation_graph import (
    ComputationGraphCapture,
    TopologicalConfidence,
    TopologicalDiagnostics,
    TrainingTopologyMonitor,
    analyze_computation_graph,
)


class TestEndToEnd:
    def test_full_pipeline(self):
        """Full pipeline: model -> capture -> analyze -> confidence."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        criterion = nn.CrossEntropyLoss()
        conf_head = TopologicalConfidence(num_classes=3)

        x = torch.randn(1, 4)
        target = torch.tensor([1])

        # 1. Get diagnostics
        diag = analyze_computation_graph(model, x, target, criterion)
        assert isinstance(diag, TopologicalDiagnostics)

        # 2. Get model output
        logits = model(x)

        # 3. Compute topology-grounded confidence
        topo_feats = torch.tensor([[
            diag.gradient_energy_ratio,
            diag.curl_energy_ratio,
            diag.harmonic_energy_ratio,
            diag.spectral_gap,
        ]])
        conf = conf_head(logits, topo_feats)
        assert 0.0 <= conf.item() <= 1.0

    def test_training_not_affected(self):
        """Analysis should not interfere with normal training."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Normal training step
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()

        # Analysis step (should not affect model)
        diag = analyze_computation_graph(model, x, target, criterion)

        # Another training step - should work normally
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()
        assert isinstance(loss.item(), float)
