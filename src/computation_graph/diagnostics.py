from __future__ import annotations

from dataclasses import dataclass

import torch

from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import hodge_decomposition, spectral_decomposition


@dataclass
class TopologicalDiagnostics:
    """Topological analysis results for a computation graph."""
    gradient_energy_ratio: float
    curl_energy_ratio: float
    harmonic_energy_ratio: float
    spectral_gap: float
    per_layer_spectral_gaps: dict[str, float]
    restriction_map_rank: float
    b1_alignment: float
    b2_alignment: float
    num_operations: int
    num_data_flows: int
    num_composites: int
    executive_iterations: int


def analyze_hodge(cc: CellComplex) -> tuple[float, float, float]:
    """Compute Hodge energy ratios from 1-cell embeddings.

    Uses embedding dim 2 (gradient norm) as the signal on edges.
    Returns (gradient_ratio, curl_ratio, harmonic_ratio) summing to 1.0.
    """
    num_edges = cc.num_cells(1)
    if num_edges < 2:
        return 0.0, 0.0, 0.0

    edge_embs = cc.get_embeddings(1)
    signal = edge_embs[:, 2] if edge_embs.shape[1] > 2 else edge_embs[:, 0]

    total_energy = (signal ** 2).sum().item()
    if total_energy < 1e-12:
        return 0.0, 0.0, 0.0

    try:
        gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
    except Exception:
        return 0.0, 0.0, 0.0

    grad_energy = (gradient ** 2).sum().item()
    curl_energy = (curl ** 2).sum().item()
    harm_energy = (harmonic ** 2).sum().item()

    total = grad_energy + curl_energy + harm_energy
    if total < 1e-12:
        return 0.0, 0.0, 0.0

    return grad_energy / total, curl_energy / total, harm_energy / total


def analyze_spectral_gap(cc: CellComplex) -> float:
    """Compute spectral gap (smallest nonzero eigenvalue) of L0."""
    if cc.num_cells(0) < 2:
        return 0.0

    try:
        eigenvalues, _ = spectral_decomposition(cc, dim=0)
    except Exception:
        return 0.0

    threshold = 1e-6
    nonzero = eigenvalues[eigenvalues > threshold]
    if len(nonzero) == 0:
        return 0.0

    return nonzero[0].item()
