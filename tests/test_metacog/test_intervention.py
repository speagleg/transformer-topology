"""Tests for InterventionGenerator."""

from src.metacog.health_report import ReasoningHealthReport
from src.metacog.intervention import generate_intervention


def test_no_intervention_when_healthy():
    """Healthy report produces no intervention (None)."""
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
    assert generate_intervention(report) is None


def test_intervention_on_curl():
    """High curl energy produces cycling intervention text."""
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
    text = generate_intervention(report)
    assert text is not None
    assert "cycling" in text.lower()


def test_intervention_on_fragmented():
    """Multiple components produce fragmentation intervention text."""
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
    text = generate_intervention(report)
    assert text is not None
    assert "fragmented" in text.lower()
    assert "3" in text


def test_intervention_on_spectral_gap():
    """Low spectral gap produces bottleneck intervention text."""
    report = ReasoningHealthReport(
        curl_energy=0.1,
        gradient_energy=0.7,
        harmonic_energy=0.2,
        spectral_gap=0.01,
        connected_components=1,
        betti_1=0,
        density=0.5,
        num_nodes=5,
        num_edges=6,
    )
    text = generate_intervention(report)
    assert text is not None
    assert "bottleneck" in text.lower()


def test_intervention_combines_multiple_issues():
    """Multiple threshold violations produce combined text."""
    report = ReasoningHealthReport(
        curl_energy=0.6,
        gradient_energy=0.1,
        harmonic_energy=0.4,
        spectral_gap=0.01,
        connected_components=2,
        betti_1=4,
        density=0.3,
        num_nodes=8,
        num_edges=10,
    )
    text = generate_intervention(report)
    assert text is not None
    assert "cycling" in text.lower()
    assert "bottleneck" in text.lower()
    assert "fragmented" in text.lower()
    assert "oscillation" in text.lower()
    assert "cycles" in text.lower()
