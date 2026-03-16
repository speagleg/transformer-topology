"""Tests for TopologyAnalyzer and ReasoningHealthReport."""

import torch
from src.metacog.graph_constructor import ReasoningGraphConstructor
from src.metacog.topology_analyzer import TopologyAnalyzer
from src.metacog.health_report import ReasoningHealthReport, Action, decide_action


def _emb(dim=128, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(dim)


def test_linear_graph_analysis():
    """A linear chain of steps produces a valid health report."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    gc.add_step("A", _emb(128, seed=0), depends_on=[])
    gc.add_step("B", _emb(128, seed=1), depends_on=[1])
    gc.add_step("C", _emb(128, seed=2), depends_on=[2])
    gc.add_step("D", _emb(128, seed=3), depends_on=[3])

    analyzer = TopologyAnalyzer()
    report = analyzer.analyze(gc.get_snapshot())

    assert report.num_nodes == 4
    assert report.num_edges == 3
    assert report.connected_components == 1
    # Linear chain: betti_1 = edges - nodes + components = 3 - 4 + 1 = 0
    assert report.betti_1 == 0
    # Energy ratios should sum to ~1
    total = report.curl_energy + report.gradient_energy + report.harmonic_energy
    assert abs(total - 1.0) < 0.1 or total < 0.01  # near 1 or all near-zero signal
    assert report.density > 0


def test_health_report_construction():
    """ReasoningHealthReport can be constructed with all fields."""
    report = ReasoningHealthReport(
        curl_energy=0.1,
        gradient_energy=0.6,
        harmonic_energy=0.3,
        spectral_gap=0.5,
        connected_components=1,
        betti_1=0,
        density=0.5,
        num_nodes=4,
        num_edges=3,
    )
    assert report.curl_energy == 0.1
    assert report.num_nodes == 4


def test_action_continue_when_healthy():
    """Healthy report gets CONTINUE action."""
    report = ReasoningHealthReport(
        curl_energy=0.1,
        gradient_energy=0.7,
        harmonic_energy=0.2,
        spectral_gap=0.5,
        connected_components=1,
        betti_1=1,
        density=0.5,
        num_nodes=5,
        num_edges=6,
    )
    assert decide_action(report) == Action.CONTINUE


def test_action_intervene_high_curl():
    """High curl energy triggers INTERVENE."""
    report = ReasoningHealthReport(
        curl_energy=0.6,
        gradient_energy=0.2,
        harmonic_energy=0.2,
        spectral_gap=0.5,
        connected_components=1,
        betti_1=1,
        density=0.5,
        num_nodes=5,
        num_edges=6,
    )
    assert decide_action(report) == Action.INTERVENE


def test_action_intervene_low_spectral_gap():
    """Low spectral gap triggers INTERVENE."""
    report = ReasoningHealthReport(
        curl_energy=0.1,
        gradient_energy=0.7,
        harmonic_energy=0.2,
        spectral_gap=0.01,
        connected_components=1,
        betti_1=1,
        density=0.5,
        num_nodes=5,
        num_edges=6,
    )
    assert decide_action(report) == Action.INTERVENE


def test_action_intervene_fragmented():
    """Multiple connected components triggers INTERVENE."""
    report = ReasoningHealthReport(
        curl_energy=0.1,
        gradient_energy=0.7,
        harmonic_energy=0.2,
        spectral_gap=0.5,
        connected_components=3,
        betti_1=0,
        density=0.1,
        num_nodes=10,
        num_edges=7,
    )
    assert decide_action(report) == Action.INTERVENE


def test_action_intervene_high_betti():
    """High Betti-1 triggers INTERVENE."""
    report = ReasoningHealthReport(
        curl_energy=0.1,
        gradient_energy=0.7,
        harmonic_energy=0.2,
        spectral_gap=0.5,
        connected_components=1,
        betti_1=5,
        density=0.5,
        num_nodes=5,
        num_edges=6,
    )
    assert decide_action(report) == Action.INTERVENE
