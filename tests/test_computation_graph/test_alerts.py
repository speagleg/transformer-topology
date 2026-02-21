import torch
import torch.nn as nn
from src.computation_graph.diagnostics import (
    TrainingTopologyMonitor,
    TopologicalDiagnostics,
)


class TestTrendAlerts:
    def test_rising_curl_alert(self):
        """Rising curl energy over 5+ steps should trigger suggestion."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for i in range(6):
            monitor.history.append(TopologicalDiagnostics(
                gradient_energy_ratio=0.8 - i * 0.05,
                curl_energy_ratio=0.1 + i * 0.05,
                harmonic_energy_ratio=0.1,
                spectral_gap=0.5,
                per_layer_spectral_gaps={},
                restriction_map_rank=0.0, b1_alignment=0.0, b2_alignment=0.0,
                num_operations=3, num_data_flows=2, num_composites=1,
                executive_iterations=0,
            ))
        suggestions = monitor.suggestions()
        assert any('curl' in s.lower() or 'damping' in s.lower() for s in suggestions)

    def test_declining_spectral_gap_alert(self):
        """Declining spectral gap should trigger bottleneck suggestion."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for i in range(6):
            monitor.history.append(TopologicalDiagnostics(
                gradient_energy_ratio=0.7, curl_energy_ratio=0.2,
                harmonic_energy_ratio=0.1,
                spectral_gap=0.5 - i * 0.08,
                per_layer_spectral_gaps={},
                restriction_map_rank=0.0, b1_alignment=0.0, b2_alignment=0.0,
                num_operations=3, num_data_flows=2, num_composites=1,
                executive_iterations=0,
            ))
        suggestions = monitor.suggestions()
        assert any('spectral' in s.lower() or 'bottleneck' in s.lower() for s in suggestions)

    def test_no_suggestions_when_stable(self):
        """Stable metrics should produce no suggestions."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for _ in range(6):
            monitor.history.append(TopologicalDiagnostics(
                gradient_energy_ratio=0.7, curl_energy_ratio=0.2,
                harmonic_energy_ratio=0.1, spectral_gap=0.5,
                per_layer_spectral_gaps={},
                restriction_map_rank=0.0, b1_alignment=0.0, b2_alignment=0.0,
                num_operations=3, num_data_flows=2, num_composites=1,
                executive_iterations=0,
            ))
        suggestions = monitor.suggestions()
        assert len(suggestions) == 0
