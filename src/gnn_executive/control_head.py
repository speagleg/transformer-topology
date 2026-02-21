"""Control signals for hierarchical GNN executive → TAT communication."""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn


@dataclass
class ControlSignal:
    """Control signals produced by GNN executive to modulate TAT behavior.

    Attributes:
        frequency_gate: (num_freqs,) — which spectral bands TAT attends to [0,1].
        spatial_focus: (num_nodes,) — which nodes TAT prioritizes [0,1].
        confidence_weights: (num_nodes,) — how much to trust TAT output per node [0,1].
        diffusion_time: scalar — wave propagation time (positive).
        wave_damping: scalar — wave dissipation (positive).
        semantic_weight: scalar — soft weight for DSM attention bias [0,1]. Never penalized.
        filter_weights: (num_filters,) — softmax weights over spectral filter ensemble.
    """
    frequency_gate: torch.Tensor
    spatial_focus: torch.Tensor
    confidence_weights: torch.Tensor
    diffusion_time: torch.Tensor
    wave_damping: torch.Tensor
    semantic_weight: torch.Tensor = None
    filter_weights: torch.Tensor = None


class ControlHead(nn.Module):
    """Produces ControlSignal from pooled GNN embeddings.

    Takes the fused GNN node embeddings, pools them, and produces
    per-signal outputs with appropriate activations.
    """

    def __init__(self, embedding_dim: int, num_freqs: int, num_filters: int = 0,
                 use_topo_feedback: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.num_filters = num_filters
        self.use_topo_feedback = use_topo_feedback

        # Shared trunk: pool → project
        # +1 for harmonic energy, +1 for log(N) size feature
        # +3 for topo feedback (gradient_ratio, curl_ratio, spectral_gap) if enabled
        trunk_input_dim = embedding_dim + 2 + (3 if use_topo_feedback else 0)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.GELU(),
        )

        # Per-signal heads
        self.freq_head = nn.Linear(embedding_dim, num_freqs)
        self.spatial_head = nn.Linear(embedding_dim, 1)  # per-node, applied via broadcast
        self.confidence_head = nn.Linear(embedding_dim, 1)  # per-node
        self.time_head = nn.Linear(embedding_dim, 1)
        self.damping_head = nn.Linear(embedding_dim, 1)
        self.semantic_weight_head = nn.Linear(embedding_dim, 1)
        # Initialize bias to -3.0 so sigmoid starts near 0.05, not 0.5.
        # This prevents untrained DSM from corrupting TAT attention at init.
        nn.init.constant_(self.semantic_weight_head.bias, -3.0)

        # Filter ensemble weights (only when multi-filter ensemble is active)
        if num_filters > 0:
            self.filter_weights_head = nn.Linear(embedding_dim, num_filters)

    def forward(self, node_embeddings: torch.Tensor,
                harmonic_energy: torch.Tensor | None = None,
                topo_features: torch.Tensor | None = None) -> ControlSignal:
        """Produce control signals from GNN node embeddings.

        Args:
            node_embeddings: (N, embedding_dim) fused GNN output.
            harmonic_energy: scalar tensor, optional harmonic component energy.

        Returns:
            ControlSignal with appropriately shaped and activated tensors.
        """
        # Mean pool over nodes
        pooled = node_embeddings.mean(dim=0)  # (embedding_dim,)

        # Append harmonic energy
        if harmonic_energy is None:
            harmonic_energy = torch.zeros(1, device=pooled.device, dtype=pooled.dtype)
        else:
            harmonic_energy = harmonic_energy.reshape(1)

        # Append log(N) size feature, normalized by log(100) so ~0.3–0.5 for N=20–100
        n_nodes = node_embeddings.shape[0]
        log_n = torch.tensor(
            [math.log(max(n_nodes, 1)) / math.log(100)],
            device=pooled.device, dtype=pooled.dtype,
        )

        trunk_input = torch.cat([pooled, harmonic_energy, log_n])  # (embedding_dim + 2,)

        # Append topological feedback features if enabled
        if self.use_topo_feedback:
            if topo_features is not None:
                topo_features = topo_features.to(pooled.device, pooled.dtype)
                if topo_features.dim() == 0:
                    topo_features = topo_features.unsqueeze(0)
                trunk_input = torch.cat([trunk_input, topo_features[:3]])
            else:
                trunk_input = torch.cat([
                    trunk_input,
                    torch.zeros(3, device=pooled.device, dtype=pooled.dtype),
                ])

        features = self.trunk(trunk_input)  # (embedding_dim,)

        # Frequency gate: sigmoid → [0,1] per spectral band
        frequency_gate = torch.sigmoid(self.freq_head(features))  # (num_freqs,)

        # Spatial focus: per-node via projection of node embeddings
        spatial_scores = self.spatial_head(node_embeddings).squeeze(-1)  # (N,)
        spatial_focus = torch.sigmoid(spatial_scores)

        # Confidence weights: per-node
        conf_scores = self.confidence_head(node_embeddings).squeeze(-1)  # (N,)
        confidence_weights = torch.sigmoid(conf_scores)

        # Diffusion time and wave damping: positive scalars via softplus
        diffusion_time = torch.nn.functional.softplus(self.time_head(features).squeeze())
        wave_damping = torch.nn.functional.softplus(self.damping_head(features).squeeze())

        # Semantic weight: scalar [0,1] — soft weight for DSM attention bias
        semantic_weight = torch.sigmoid(self.semantic_weight_head(features).squeeze())

        # Filter ensemble weights: softmax over filter paths
        filter_weights = None
        if self.num_filters > 0:
            filter_weights = torch.softmax(
                self.filter_weights_head(features), dim=-1,
            )  # (num_filters,)

        return ControlSignal(
            frequency_gate=frequency_gate,
            spatial_focus=spatial_focus,
            confidence_weights=confidence_weights,
            diffusion_time=diffusion_time,
            wave_damping=wave_damping,
            semantic_weight=semantic_weight,
            filter_weights=filter_weights,
        )
