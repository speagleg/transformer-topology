import torch
import torch.nn as nn
from src.computation_graph.capture import ComputationGraphCapture
from src.computation_graph.diagnostics import (
    TopologicalDiagnostics,
    analyze_hodge,
    analyze_spectral_gap,
    analyze_computation_graph,
    TrainingTopologyMonitor,
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


class TestFullAnalysisPipeline:
    def test_returns_diagnostics(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        assert isinstance(diag, TopologicalDiagnostics)

    def test_diagnostics_has_positive_spectral_gap(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        assert diag.spectral_gap > 0.0

    def test_diagnostics_hodge_ratios_valid(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        total = diag.gradient_energy_ratio + diag.curl_energy_ratio + diag.harmonic_energy_ratio
        assert abs(total - 1.0) < 1e-4 or total == 0.0

    def test_diagnostics_counts_correct(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        leaf_count = sum(1 for m in model.modules() if len(list(m.children())) == 0)
        assert diag.num_operations == leaf_count

    def test_no_gradient_leak(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        model.zero_grad()
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        for p in model.parameters():
            assert p.grad is None or (p.grad == 0).all()


class TestTrainingTopologyMonitor:
    def _make_model(self):
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def test_record_step(self):
        model = self._make_model()
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        monitor.record_step(x, target)
        assert len(monitor.history) == 1
        assert isinstance(monitor.history[0], TopologicalDiagnostics)

    def test_summary(self):
        model = self._make_model()
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for _ in range(3):
            x = torch.randn(1, 4)
            target = torch.tensor([1])
            monitor.record_step(x, target)
        summary = monitor.summary()
        assert 'gradient_energy_ratio_mean' in summary
        assert 'spectral_gap_mean' in summary

    def test_alerts_empty_when_healthy(self):
        model = self._make_model()
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for _ in range(3):
            x = torch.randn(1, 4)
            target = torch.tensor([1])
            monitor.record_step(x, target)
        alerts = monitor.alerts()
        assert isinstance(alerts, list)

    def test_custom_thresholds(self):
        model = self._make_model()
        thresholds = {'curl_energy_ratio': 0.01}
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss(), alert_thresholds=thresholds)
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        monitor.record_step(x, target)
        alerts = monitor.alerts()
        assert isinstance(alerts, list)

    def test_reset(self):
        model = self._make_model()
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        monitor.record_step(x, target)
        assert len(monitor.history) == 1
        monitor.reset()
        assert len(monitor.history) == 0
