"""MetaCognitive Controller: task-conditional gating, confidence, and strategy selection.

Extends ControlHead with:
  - Task embedding input (nn.Embedding) so the controller knows WHAT task is running
  - Iteration context input (iter_ratio, prev_confidence, prev_delta)
  - text_gate / structure_gate: per-task feature gating
  - uncertainty: temperature-scaled calibrated uncertainty
  - iteration_budget: soft learned iteration limit
  - strategy_weights: softmax over reasoning strategies
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.gnn_executive.control_head import ControlSignal


class MetaCognitiveController(nn.Module):
    """Produces ControlSignal with metacognitive fields from node embeddings + task identity.

    Compared to ControlHead, adds:
      - task_embedding: nn.Embedding(num_tasks, task_embed_dim) -- task identity
      - iteration_context: (3,) -- [iter/max_iter, prev_confidence, prev_delta]
      - 5 new heads: text_gate, structure_gate, uncertainty, iteration_budget, strategy_weights
      - Learned confidence_temperature for calibrated uncertainty
    """

    NUM_STRATEGIES = 4  # spectral_dominant, spatial_dominant, balanced, text_dominant

    def __init__(self, embedding_dim: int, num_freqs: int, num_filters: int = 0,
                 num_tasks: int = 19, task_embed_dim: int = 128,
                 use_topo_feedback: bool = False,
                 use_embedding_topo_feedback: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.num_filters = num_filters
        self.num_tasks = num_tasks
        self.task_embed_dim = task_embed_dim

        # Task embedding: separate from GraphFormerEncoder's embedding
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)

        # Topo feedback dimension (same logic as ControlHead)
        if use_embedding_topo_feedback and use_topo_feedback:
            topo_dim = 6
        elif use_topo_feedback:
            topo_dim = 3
        else:
            topo_dim = 0
        self._topo_dim = topo_dim

        # Trunk input: mean_pool(emb_dim) + harmonic(1) + logN(1) + topo(0-6) + task(128) + iter_ctx(3)
        trunk_input_dim = embedding_dim + 2 + topo_dim + task_embed_dim + 3
        self._trunk_input_dim = trunk_input_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        trunk_out_dim = 128

        # === Existing heads (same as ControlHead) ===
        self.freq_head = nn.Linear(trunk_out_dim, num_freqs)
        # Per-node heads operate on raw node embeddings (embedding_dim), not trunk output
        self.spatial_head = nn.Linear(embedding_dim, 1)
        self.confidence_head = nn.Linear(embedding_dim, 1)
        self.time_head = nn.Linear(trunk_out_dim, 1)
        self.damping_head = nn.Linear(trunk_out_dim, 1)
        self.semantic_weight_head = nn.Linear(trunk_out_dim, 1)
        nn.init.constant_(self.semantic_weight_head.bias, -3.0)

        if num_filters > 0:
            self.filter_weights_head = nn.Linear(trunk_out_dim, num_filters)

        # === NEW metacognition heads ===
        # Layer 1: Task-conditional gating
        self.text_gate_head = nn.Linear(trunk_out_dim, 1)
        self.structure_gate_head = nn.Linear(trunk_out_dim, 1)
        nn.init.constant_(self.structure_gate_head.bias, 2.0)  # structural tasks dominate early

        # Layer 2: Confidence-aware reasoning
        self.uncertainty_head = nn.Linear(trunk_out_dim, 1)
        self.iteration_budget_head = nn.Linear(trunk_out_dim, 1)
        self.confidence_temperature = nn.Parameter(torch.tensor(1.5))

        # Layer 3: Strategy selection
        self.strategy_head = nn.Linear(trunk_out_dim, self.NUM_STRATEGIES)

    def forward(self, node_embeddings: torch.Tensor,
                task_id: int | None = None,
                harmonic_energy: torch.Tensor | None = None,
                topo_features: torch.Tensor | None = None,
                iteration_context: torch.Tensor | None = None) -> ControlSignal:
        """Produce control signals with metacognitive fields from GNN node embeddings.

        Args:
            node_embeddings: (N, embedding_dim) fused GNN output.
            task_id: integer task index, or None for zero task embedding.
            harmonic_energy: scalar tensor, optional harmonic component energy.
            topo_features: (topo_dim,) optional topological feedback features.
            iteration_context: (3,) tensor [iter_ratio, prev_confidence, prev_delta].

        Returns:
            ControlSignal with all metacognitive fields populated.
        """
        dev = node_embeddings.device
        dtype = node_embeddings.dtype

        # Mean pool over nodes
        pooled = node_embeddings.mean(dim=0)

        # Harmonic energy
        if harmonic_energy is None:
            harmonic_energy = torch.zeros(1, device=dev, dtype=dtype)
        else:
            harmonic_energy = harmonic_energy.reshape(1)

        # log(N) size feature
        n_nodes = node_embeddings.shape[0]
        log_n = torch.tensor(
            [math.log(max(n_nodes, 1)) / math.log(100)],
            device=dev, dtype=dtype,
        )

        # Topo features
        if self._topo_dim > 0:
            if topo_features is not None:
                topo_features = topo_features.to(dev, dtype)
                if topo_features.dim() == 0:
                    topo_features = topo_features.unsqueeze(0)
                topo_feat = topo_features[:self._topo_dim]
            else:
                topo_feat = torch.zeros(self._topo_dim, device=dev, dtype=dtype)
        else:
            topo_feat = torch.zeros(0, device=dev, dtype=dtype)

        # Task embedding
        if task_id is not None:
            task_idx = torch.tensor(task_id, device=dev, dtype=torch.long)
            task_emb = self.task_embedding(task_idx)
        else:
            task_emb = torch.zeros(self.task_embed_dim, device=dev, dtype=dtype)

        # Iteration context
        if iteration_context is None:
            iter_ctx = torch.zeros(3, device=dev, dtype=dtype)
        else:
            iter_ctx = iteration_context.to(dev, dtype)

        # Build trunk input
        trunk_input = torch.cat([pooled, harmonic_energy, log_n, topo_feat, task_emb, iter_ctx])
        features = self.trunk(trunk_input)

        # === Existing heads ===
        frequency_gate = torch.sigmoid(self.freq_head(features))
        spatial_focus = torch.sigmoid(self.spatial_head(node_embeddings).squeeze(-1))
        confidence_weights = torch.sigmoid(self.confidence_head(node_embeddings).squeeze(-1))
        diffusion_time = F.softplus(self.time_head(features).squeeze())
        wave_damping = F.softplus(self.damping_head(features).squeeze())
        semantic_weight = torch.sigmoid(self.semantic_weight_head(features).squeeze())

        filter_weights = None
        if self.num_filters > 0:
            filter_weights = torch.softmax(self.filter_weights_head(features), dim=-1)

        # === NEW metacognition heads ===
        text_gate = torch.sigmoid(self.text_gate_head(features).squeeze())
        structure_gate = torch.sigmoid(self.structure_gate_head(features).squeeze())

        raw_uncertainty = self.uncertainty_head(features).squeeze()
        temperature = self.confidence_temperature.clamp(min=0.1)
        uncertainty = torch.sigmoid(raw_uncertainty / temperature)

        iteration_budget = F.softplus(self.iteration_budget_head(features).squeeze())
        strategy_weights = torch.softmax(self.strategy_head(features), dim=-1)

        return ControlSignal(
            frequency_gate=frequency_gate,
            spatial_focus=spatial_focus,
            confidence_weights=confidence_weights,
            diffusion_time=diffusion_time,
            wave_damping=wave_damping,
            semantic_weight=semantic_weight,
            filter_weights=filter_weights,
            text_gate=text_gate,
            structure_gate=structure_gate,
            uncertainty=uncertainty,
            iteration_budget=iteration_budget,
            strategy_weights=strategy_weights,
        )
