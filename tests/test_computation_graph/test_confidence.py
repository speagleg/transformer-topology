import torch
import torch.nn as nn
from src.computation_graph.confidence import TopologicalConfidence


class TestTopologicalConfidence:
    def test_output_shape(self):
        head = TopologicalConfidence(num_classes=3)
        logits = torch.randn(1, 3)
        topo_features = torch.tensor([[0.7, 0.2, 0.1, 0.5]])
        conf = head(logits, topo_features)
        assert conf.shape == (1,)

    def test_output_range(self):
        head = TopologicalConfidence(num_classes=3)
        logits = torch.randn(1, 3)
        topo_features = torch.tensor([[0.7, 0.2, 0.1, 0.5]])
        conf = head(logits, topo_features)
        assert conf.item() >= 0.0
        assert conf.item() <= 1.0

    def test_gradient_flows(self):
        head = TopologicalConfidence(num_classes=3)
        logits = torch.randn(1, 3, requires_grad=True)
        topo_features = torch.tensor([[0.7, 0.2, 0.1, 0.5]])
        conf = head(logits, topo_features)
        conf.sum().backward()
        assert logits.grad is not None

    def test_batch_support(self):
        head = TopologicalConfidence(num_classes=3)
        logits = torch.randn(4, 3)
        topo_features = torch.randn(4, 4)
        conf = head(logits, topo_features)
        assert conf.shape == (4,)
