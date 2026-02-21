import torch
import torch.nn as nn
from src.computation_graph.capture import ComputationGraphCapture
from src.computation_graph.diagnostics import (
    TopologicalDiagnostics,
    analyze_hodge,
    analyze_spectral_gap,
)
from src.cell_complex.cell_complex import CellComplex


class TestTopologicalDiagnostics:
    def test_dataclass_fields(self):
        diag = TopologicalDiagnostics(
            gradient_energy_ratio=0.5, curl_energy_ratio=0.3,
            harmonic_energy_ratio=0.2, spectral_gap=0.1,
            per_layer_spectral_gaps={}, restriction_map_rank=0.0,
            b1_alignment=0.0, b2_alignment=0.0,
            num_operations=10, num_data_flows=9,
            num_composites=2, executive_iterations=0,
        )
        assert diag.gradient_energy_ratio == 0.5
        assert diag.num_operations == 10

    def test_energy_ratios_sum_to_one(self):
        diag = TopologicalDiagnostics(
            gradient_energy_ratio=0.5, curl_energy_ratio=0.3,
            harmonic_energy_ratio=0.2, spectral_gap=0.0,
            per_layer_spectral_gaps={}, restriction_map_rank=0.0,
            b1_alignment=0.0, b2_alignment=0.0,
            num_operations=0, num_data_flows=0,
            num_composites=0, executive_iterations=0,
        )
        total = diag.gradient_energy_ratio + diag.curl_energy_ratio + diag.harmonic_energy_ratio
        assert abs(total - 1.0) < 1e-6


class TestAnalyzeHodge:
    def _capture_model(self):
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 2),
        )
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        return cap.to_cell_complex()

    def test_analyze_hodge_returns_ratios(self):
        cc = self._capture_model()
        grad_r, curl_r, harm_r = analyze_hodge(cc)
        assert grad_r >= 0.0
        assert curl_r >= 0.0
        assert harm_r >= 0.0

    def test_analyze_hodge_ratios_sum_to_one(self):
        cc = self._capture_model()
        grad_r, curl_r, harm_r = analyze_hodge(cc)
        total = grad_r + curl_r + harm_r
        assert abs(total - 1.0) < 1e-4 or total == 0.0

    def test_analyze_hodge_with_no_2cells(self):
        """No 2-cells means curl should be 0."""
        cc = CellComplex(embedding_dim=4)
        n0 = cc.add_0_cell(torch.randn(4), "a")
        n1 = cc.add_0_cell(torch.randn(4), "b")
        n2 = cc.add_0_cell(torch.randn(4), "c")
        cc.add_1_cell(n0, n1, torch.randn(4), "e0")
        cc.add_1_cell(n1, n2, torch.randn(4), "e1")
        grad_r, curl_r, harm_r = analyze_hodge(cc)
        assert curl_r == 0.0


class TestAnalyzeSpectralGap:
    def _capture_model(self):
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 2),
        )
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        return cap.to_cell_complex()

    def test_spectral_gap_is_non_negative(self):
        cc = self._capture_model()
        gap = analyze_spectral_gap(cc)
        assert gap >= 0.0

    def test_spectral_gap_connected_graph(self):
        cc = self._capture_model()
        gap = analyze_spectral_gap(cc)
        assert gap > 0.0

    def test_spectral_gap_single_node(self):
        cc = CellComplex(embedding_dim=4)
        cc.add_0_cell(torch.randn(4), "single")
        gap = analyze_spectral_gap(cc)
        assert gap == 0.0
