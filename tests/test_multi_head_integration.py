"""Test MultiHeadClassifier integration with HierarchicalMultiHopModel."""
import torch
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.multi_hop import MultiHopDataset


def _make_model():
    return HierarchicalMultiHopModel(
        embedding_dim=16, gnn_hidden=32,
        gnn_spatial_layers=1, gnn_spectral_layers=1,
        max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2,
        tat_ff_dim=32, max_classes=6,
        max_iterations=2, convergence_threshold=0.1,
        use_wave_dynamics=False, use_higher_order=False,
        use_multi_head_classifier=True,
    )


class TestMultiHeadIntegration:
    def test_model_has_multi_head_classifier(self):
        model = _make_model()
        assert hasattr(model, 'multi_head_classifier')

    def test_forward_with_task_name(self):
        model = _make_model()
        ds = MultiHopDataset(1, min_hops=1, max_hops=5, num_distractors=3, embedding_dim=16)
        cc, query, target, answer = ds[0][:4]
        logits = model(cc.clone(), query, target, task="diverse")
        assert logits.shape[-1] == 11  # diverse has 11 classes

    def test_forward_without_task_falls_back(self):
        model = _make_model()
        ds = MultiHopDataset(1, min_hops=1, max_hops=5, num_distractors=3, embedding_dim=16)
        cc, query, target, answer = ds[0][:4]
        logits = model(cc.clone(), query, target)
        assert logits.shape[-1] == 6  # max_classes default
