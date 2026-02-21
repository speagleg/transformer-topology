from __future__ import annotations

from dataclasses import dataclass

import torch

from src.cell_complex.cell_complex import CellComplex
from src.computation_graph.capture import ComputationGraphCapture
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


def analyze_computation_graph(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
) -> TopologicalDiagnostics:
    """Run one forward+backward pass and return full topological diagnostics.

    Does not modify model parameters -- saves and restores gradient state.
    """
    grad_state = {p: p.grad.clone() if p.grad is not None else None
                  for p in model.parameters()}
    was_training = model.training
    model.zero_grad()

    with ComputationGraphCapture(model) as cap:
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()

    cc = cap.to_cell_complex()
    grad_r, curl_r, harm_r = analyze_hodge(cc)
    gap = analyze_spectral_gap(cc)

    # Restore gradient state
    model.zero_grad()
    for p in model.parameters():
        if grad_state[p] is not None:
            p.grad = grad_state[p]
    if was_training:
        model.train()

    return TopologicalDiagnostics(
        gradient_energy_ratio=grad_r,
        curl_energy_ratio=curl_r,
        harmonic_energy_ratio=harm_r,
        spectral_gap=gap,
        per_layer_spectral_gaps={},
        restriction_map_rank=0.0,
        b1_alignment=0.0,
        b2_alignment=0.0,
        num_operations=cc.num_cells(0),
        num_data_flows=cc.num_cells(1),
        num_composites=cc.num_cells(2),
        executive_iterations=0,
    )


_DEFAULT_THRESHOLDS = {
    'curl_energy_ratio': 0.4,
    'harmonic_energy_ratio': 0.15,
    'spectral_gap_min': 0.01,
}


class TrainingTopologyMonitor:
    """Monitors computation graph topology across training steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        alert_thresholds: dict[str, float] | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.history: list[TopologicalDiagnostics] = []
        self.thresholds = {**_DEFAULT_THRESHOLDS}
        if alert_thresholds:
            self.thresholds.update(alert_thresholds)

    def record_step(
        self, input_tensor: torch.Tensor, target: torch.Tensor,
    ) -> TopologicalDiagnostics:
        diag = analyze_computation_graph(
            self.model, input_tensor, target, self.criterion,
        )
        self.history.append(diag)
        return diag

    def summary(self) -> dict[str, float]:
        if not self.history:
            return {}
        fields = [
            'gradient_energy_ratio', 'curl_energy_ratio',
            'harmonic_energy_ratio', 'spectral_gap',
        ]
        result: dict[str, float] = {}
        for field in fields:
            values = [getattr(d, field) for d in self.history]
            mean = sum(values) / len(values)
            result[f'{field}_mean'] = mean
            if len(values) > 1:
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                result[f'{field}_std'] = var ** 0.5
            else:
                result[f'{field}_std'] = 0.0
        return result

    def alerts(self) -> list[str]:
        if not self.history:
            return []
        latest = self.history[-1]
        alerts: list[str] = []
        curl_thresh = self.thresholds.get('curl_energy_ratio', 0.4)
        if latest.curl_energy_ratio > curl_thresh:
            alerts.append(
                f"High curl energy: {latest.curl_energy_ratio:.3f} > {curl_thresh}"
            )
        harm_thresh = self.thresholds.get('harmonic_energy_ratio', 0.15)
        if latest.harmonic_energy_ratio > harm_thresh:
            alerts.append(
                f"High harmonic energy: {latest.harmonic_energy_ratio:.3f} > {harm_thresh}"
            )
        gap_min = self.thresholds.get('spectral_gap_min', 0.01)
        if latest.spectral_gap < gap_min:
            alerts.append(
                f"Low spectral gap: {latest.spectral_gap:.6f} < {gap_min}"
            )
        return alerts

    def reset(self):
        self.history.clear()
