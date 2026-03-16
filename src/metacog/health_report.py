"""Reasoning health report dataclass and action decision logic."""

from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    CONTINUE = "CONTINUE"
    INTERVENE = "INTERVENE"


@dataclass
class ReasoningHealthReport:
    """Topological health metrics for a reasoning graph.

    Attributes:
        curl_energy: Fraction of edge signal energy in curl component (cycling).
        gradient_energy: Fraction of edge signal energy in gradient component (flow).
        harmonic_energy: Fraction of edge signal energy in harmonic component (global).
        spectral_gap: Fiedler value (2nd smallest eigenvalue of L0).
        connected_components: Number of connected components.
        betti_1: First Betti number (independent cycles).
        density: Graph density (edges / max possible edges).
        num_nodes: Number of reasoning steps.
        num_edges: Number of dependency edges.
    """
    curl_energy: float
    gradient_energy: float
    harmonic_energy: float
    spectral_gap: float
    connected_components: int
    betti_1: int
    density: float
    num_nodes: int
    num_edges: int


def decide_action(report: ReasoningHealthReport) -> Action:
    """Decide whether to intervene based on health report thresholds.

    Thresholds:
        - curl_energy > 0.4 -> INTERVENE (information cycling)
        - spectral_gap < 0.05 -> INTERVENE (information bottleneck)
        - harmonic_energy > 0.3 -> INTERVENE (trapped global modes)
        - connected_components > 1 -> INTERVENE (fragmented reasoning)
        - betti_1 > 2 -> INTERVENE (too many independent cycles)
    """
    if report.curl_energy > 0.4:
        return Action.INTERVENE
    if report.spectral_gap < 0.05:
        return Action.INTERVENE
    if report.harmonic_energy > 0.3:
        return Action.INTERVENE
    if report.connected_components > 1:
        return Action.INTERVENE
    if report.betti_1 > 2:
        return Action.INTERVENE
    return Action.CONTINUE
