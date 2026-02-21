import torch
from src.gnn_executive.control_head import ControlHead, ControlSignal


class TestTopologicalFeedback:
    def test_control_head_accepts_topo_features(self):
        head = ControlHead(embedding_dim=8, num_freqs=4, use_topo_feedback=True)
        node_embs = torch.randn(5, 8)
        harmonic = torch.tensor(0.5)
        topo_feats = torch.tensor([0.7, 0.2, 0.5])
        signal = head(node_embs, harmonic, topo_features=topo_feats)
        assert isinstance(signal, ControlSignal)

    def test_control_head_works_without_topo_features(self):
        head = ControlHead(embedding_dim=8, num_freqs=4, use_topo_feedback=True)
        node_embs = torch.randn(5, 8)
        harmonic = torch.tensor(0.5)
        signal = head(node_embs, harmonic)
        assert isinstance(signal, ControlSignal)

    def test_backward_compat(self):
        head = ControlHead(embedding_dim=8, num_freqs=4)
        node_embs = torch.randn(5, 8)
        harmonic = torch.tensor(0.5)
        signal = head(node_embs, harmonic)
        assert isinstance(signal, ControlSignal)
