"""Unified topology observer for the DSM training pipeline.

Coordinates both the Embedding Topology Analyzer (DSM internal health)
and the Computation Graph Topology analyzer (autograd graph structure)
during training. Supports two modes:

- **Observer mode**: both analyzers run periodically, log diagnostics,
  and print alerts. Zero impact on the model forward pass.
- **Active mode**: the embedding topology's 6-feature vector is cached
  and fed into the ControlHead via ``topo_features``, letting the GNN
  executive adapt based on DSM health.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from src.topology_analyzer.dsm_wrapper import DSMAnalysisWrapper
from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
from src.topology_analyzer.feedback import TopologyFeedback, EmbeddingTopologyAdvisor
from src.computation_graph.diagnostics import TrainingTopologyMonitor
from src.computation_graph.model_wrapper import analyze_computation_graph_cc


def _unpack_sample(sample):
    """Unpack a 4-tuple or 5-tuple sample."""
    if len(sample) == 5:
        return sample
    cc, query, target, answer = sample
    return cc, query, target, answer, None


class TopologyObserver:
    """Coordinates embedding + computation graph topology analysis.

    Args:
        model: The HierarchicalMultiHopModel being trained.
        criterion: Loss function (e.g. CrossEntropyLoss).
        dsm: Optional DistilledSemanticModel for embedding analysis.
        config: Dict with keys ``analyze_every``, ``active_mode``,
                ``dsm_num_heads``, ``dsm_hidden_dim``.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dsm: nn.Module | None = None,
        config: dict | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.config = config or {}
        self.analyze_every = self.config.get('analyze_every', 5)

        # Embedding topology analyzer (DSM internal health)
        self.embedding_analyzer: TransformerTopologyAnalyzer | None = None
        if dsm is not None:
            wrapper = DSMAnalysisWrapper(dsm)
            num_heads = self.config.get('dsm_num_heads', 16)
            hidden_dim = self.config.get('dsm_hidden_dim', 1024)
            self.embedding_analyzer = TransformerTopologyAnalyzer(
                wrapper,
                model_name="dsm",
                num_heads=num_heads,
                hidden_dim=hidden_dim,
            )

        # Computation graph topology monitor
        self.comp_graph_monitor = TrainingTopologyMonitor(model, criterion)

        # Feedback converters
        self.feedback = TopologyFeedback()
        self.advisor = EmbeddingTopologyAdvisor()

        # State
        self.profile_history: list = []
        self._last_topo_features: torch.Tensor | None = None

    def should_analyze(self, epoch: int) -> bool:
        """Return True if analysis should run this epoch."""
        return epoch % self.analyze_every == 0

    def run_analysis(self, epoch: int, sample) -> dict:
        """Run both analyzers on a single sample.

        Args:
            epoch: Current epoch number (for logging).
            sample: A dataset sample (4 or 5-tuple).

        Returns:
            Dict with ``'embedding'`` and ``'comp_graph'`` sub-dicts.
        """
        results: dict = {}
        cc, query, target, answer, metadata = _unpack_sample(sample)

        # --- Embedding topology (DSM internal analysis) ---
        # no_grad is fine here — no backward needed
        if self.embedding_analyzer is not None:
            with torch.no_grad():
                node_embs = cc.get_embeddings(0)
                prefix = node_embs.unsqueeze(0)  # (1, N, dim)
                try:
                    profile = self.embedding_analyzer.analyze(prefix)
                    self.profile_history.append(profile)
                    self._last_topo_features = self.feedback.profile_to_features(profile)
                    results['embedding'] = profile.summary()

                    alerts = self.advisor.check_alerts(profile)
                    if alerts:
                        for a in alerts:
                            print(f"  [TOPO ALERT] {a}")
                except Exception as e:
                    results['embedding'] = {'error': str(e)}

        # --- Computation graph topology ---
        # NOTE: no torch.no_grad() here — analyze_computation_graph needs backward()
        try:
            diag = analyze_computation_graph_cc(
                self.model, cc, query, target, answer, self.criterion,
                metadata=metadata,
            )
            results['comp_graph'] = dataclasses.asdict(diag)

            # Record for monitor trend tracking
            self.comp_graph_monitor.history.append(diag)
            cg_alerts = self.comp_graph_monitor.alerts()
            if cg_alerts:
                for a in cg_alerts:
                    print(f"  [COMP GRAPH ALERT] {a}")
        except Exception as e:
            results['comp_graph'] = {'error': str(e)}

        return results

    def get_topo_features(self) -> torch.Tensor | None:
        """Return cached 6-dim topo feature tensor for active mode, or None.

        Returns 6-dim embedding topology features if DSM is available,
        otherwise builds a 6-dim vector from computation graph topology
        (gradient/curl/harmonic energy ratios + spectral gap) padded to 6.
        This is the executive's self-awareness signal.
        """
        if self._last_topo_features is not None:
            return self._last_topo_features
        # Fallback: computation graph topology → self-awareness
        if self.comp_graph_monitor.history:
            latest = self.comp_graph_monitor.history[-1]
            return torch.tensor([
                latest.gradient_energy_ratio,
                latest.curl_energy_ratio,
                latest.harmonic_energy_ratio,
                latest.spectral_gap,
                float(latest.num_operations) / 200.0,  # normalized op count
                float(latest.num_data_flows) / 200.0,   # normalized flow count
            ])
        return None
