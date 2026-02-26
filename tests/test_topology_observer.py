"""Tests for TopologyObserver."""

import torch
import torch.nn as nn

from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.multi_hop import MultiHopDataset
from src.topology_observer import TopologyObserver


def _make_model_and_dataset():
    model = HierarchicalMultiHopModel(
        embedding_dim=16, gnn_hidden=32,
        gnn_spatial_layers=1, gnn_spectral_layers=1,
        max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2,
        tat_ff_dim=32, max_classes=6,
        max_iterations=2, convergence_threshold=0.1,
        use_wave_dynamics=False, use_higher_order=False,
    )
    ds = MultiHopDataset(
        num_samples=2, min_hops=1, max_hops=5,
        num_distractors=3, embedding_dim=16,
    )
    return model, ds


class TestTopologyObserver:
    def test_should_analyze_frequency(self):
        model, ds = _make_model_and_dataset()
        criterion = nn.CrossEntropyLoss()
        obs = TopologyObserver(model, criterion, config={'analyze_every': 3})
        assert obs.should_analyze(0) is True
        assert obs.should_analyze(1) is False
        assert obs.should_analyze(2) is False
        assert obs.should_analyze(3) is True
        assert obs.should_analyze(6) is True

    def test_run_analysis_no_dsm(self):
        model, ds = _make_model_and_dataset()
        criterion = nn.CrossEntropyLoss()
        obs = TopologyObserver(model, criterion, dsm=None)
        results = obs.run_analysis(0, ds[0])
        # No embedding analysis when dsm is None
        assert 'embedding' not in results
        # Comp graph should always be present
        assert 'comp_graph' in results

    def test_run_analysis_with_dsm(self):
        from src.llm.dsm import DistilledSemanticModel
        model, ds = _make_model_and_dataset()
        criterion = nn.CrossEntropyLoss()
        dsm = DistilledSemanticModel(
            hidden_dim=16, num_heads=2, ff_dim=32,
            num_layers=2, cross_attn_layer=1,
        )
        obs = TopologyObserver(model, criterion, dsm=dsm,
                               config={'dsm_num_heads': 2, 'dsm_hidden_dim': 16})
        results = obs.run_analysis(0, ds[0])
        assert 'embedding' in results
        assert 'comp_graph' in results

    def test_get_topo_features_none_without_analysis(self):
        model, ds = _make_model_and_dataset()
        criterion = nn.CrossEntropyLoss()
        obs = TopologyObserver(model, criterion)
        assert obs.get_topo_features() is None

    def test_get_topo_features_after_analysis(self):
        from src.llm.dsm import DistilledSemanticModel
        model, ds = _make_model_and_dataset()
        criterion = nn.CrossEntropyLoss()
        dsm = DistilledSemanticModel(
            hidden_dim=16, num_heads=2, ff_dim=32,
            num_layers=2, cross_attn_layer=1,
        )
        obs = TopologyObserver(model, criterion, dsm=dsm,
                               config={'dsm_num_heads': 2, 'dsm_hidden_dim': 16})
        obs.run_analysis(0, ds[0])
        feat = obs.get_topo_features()
        assert feat is not None
        assert feat.shape == (6,)
