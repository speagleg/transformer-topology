"""Top-level TransformerTopologyAnalyzer.

Orchestrates all three analysis layers and cross-layer sheaf
to produce a complete TopologicalProfile.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.topology_analyzer.profile import TopologicalProfile
from src.topology_analyzer.hooks import TransformerHookManager, _find_transformer_layers
from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer


class TransformerTopologyAnalyzer:
    """Full topological analysis of a transformer model.

    Runs a forward pass with hooks to capture internal representations,
    then analyzes embedding manifolds, attention flows, and weight spaces.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "unknown",
        num_heads: int = 1,
        hidden_dim: int = 64,
        analyze_attention: bool = False,
        analyze_weights: bool = False,
        embedding_k: int = 5,
        embedding_neighborhood: str = "knn",
    ):
        self.model = model
        self.model_name = model_name
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.analyze_attention = analyze_attention
        self.analyze_weights = analyze_weights

        self.embedding_analyzer = EmbeddingManifoldAnalyzer(
            neighborhood=embedding_neighborhood, k=embedding_k,
        )
        self.attention_analyzer = AttentionFlowAnalyzer()
        self.weight_analyzer = WeightSpaceAnalyzer()

    @torch.no_grad()
    def analyze(self, input_tensor: torch.Tensor) -> TopologicalProfile:
        """Run full topological analysis.

        Args:
            input_tensor: Model input (batch, seq_len, hidden_dim) or similar.

        Returns:
            TopologicalProfile with all computed invariants.
        """
        was_training = self.model.training
        self.model.eval()

        # Capture hidden states (and optionally attention maps)
        with TransformerHookManager(
            self.model, capture_attention=self.analyze_attention,
        ) as manager:
            _ = self.model(input_tensor)
            hidden_states = manager.get_hidden_states()
            attention_maps = manager.get_attention_maps() if self.analyze_attention else {}

        num_layers = len(hidden_states)
        seq_len = next(iter(hidden_states.values())).shape[0] if hidden_states else 0

        # Layer 1: Embedding Manifold
        per_layer_persistence = {}
        per_layer_betti = {}
        per_layer_spectral_gap = {}
        per_layer_hodge_ratios = {}

        embedding_features = {}  # for sheaf
        for layer_idx, h in hidden_states.items():
            lcc = self.embedding_analyzer.build_cell_complex(h, layer_idx)
            result = self.embedding_analyzer.analyze_layer(lcc)
            per_layer_persistence[layer_idx] = result["persistence"]
            per_layer_betti[layer_idx] = result["betti"]
            per_layer_spectral_gap[layer_idx] = result["spectral_gap"]
            per_layer_hodge_ratios[layer_idx] = result["hodge_ratios"]
            # Mean hidden state as feature for cross-layer sheaf
            embedding_features[layer_idx] = h.mean(dim=0)

        # Cross-layer embedding sheaf
        sheaf_dim = min(self.hidden_dim, 16)  # cap for efficiency
        cross_layer_sheaf_gap = 0.0
        if num_layers >= 2:
            sheaf_analyzer = CrossLayerSheafAnalyzer(
                feature_dim=sheaf_dim, num_layers=num_layers,
            )
            projected = {
                i: f[:sheaf_dim] for i, f in embedding_features.items()
            }
            cross_layer_sheaf_gap = sheaf_analyzer.compute_coherence(projected)

        # Layer 2: Attention Flow
        per_head_hodge_ratios = {}
        per_head_classification = {}
        per_layer_head_sheaf_gap = {}
        cross_layer_attention_wasserstein = {}

        if self.analyze_attention and attention_maps:
            for (layer_idx, head_idx), attn in attention_maps.items():
                lcc = self.attention_analyzer.build_cell_complex(
                    attn, layer_idx, head_idx,
                )
                result = self.attention_analyzer.analyze_head(lcc)
                per_head_hodge_ratios[(layer_idx, head_idx)] = result["hodge_ratios"]
                per_head_classification[(layer_idx, head_idx)] = result["classification"]

        # Layer 3: Weight Space
        per_layer_effective_rank = {}
        per_layer_condition_number = {}
        per_layer_sv_persistence = {}
        cross_layer_weight_sheaf_gap = 0.0

        if self.analyze_weights:
            weight_layers = self._find_weight_matrices()
            for layer_idx, W in weight_layers.items():
                result = self.weight_analyzer.analyze_weight_matrix(W, layer_idx)
                per_layer_effective_rank[layer_idx] = result["effective_rank"]
                per_layer_condition_number[layer_idx] = result["condition_number"]
                per_layer_sv_persistence[layer_idx] = result["sv_persistence"]

        if was_training:
            self.model.train()

        return TopologicalProfile(
            per_layer_persistence=per_layer_persistence,
            per_layer_betti=per_layer_betti,
            per_layer_spectral_gap=per_layer_spectral_gap,
            per_layer_hodge_ratios=per_layer_hodge_ratios,
            cross_layer_sheaf_gap=cross_layer_sheaf_gap,
            per_head_hodge_ratios=per_head_hodge_ratios,
            per_head_classification=per_head_classification,
            per_layer_head_sheaf_gap=per_layer_head_sheaf_gap,
            cross_layer_attention_wasserstein=cross_layer_attention_wasserstein,
            per_layer_effective_rank=per_layer_effective_rank,
            per_layer_condition_number=per_layer_condition_number,
            per_layer_sv_persistence=per_layer_sv_persistence,
            cross_layer_weight_sheaf_gap=cross_layer_weight_sheaf_gap,
            model_name=self.model_name,
            num_layers=num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            seq_len=seq_len,
            timestamp=time.time(),
        )

    def _find_weight_matrices(self) -> dict[int, torch.Tensor]:
        """Find the main weight matrices (e.g. in_proj_weight from self_attn)."""
        weights = {}
        layers = _find_transformer_layers(self.model)
        for i, layer in enumerate(layers):
            for name in ['self_attn.in_proj_weight', 'linear1.weight', 'linear2.weight']:
                parts = name.split('.')
                obj = layer
                try:
                    for part in parts:
                        obj = getattr(obj, part)
                    if isinstance(obj, torch.Tensor) and obj.dim() == 2:
                        weights[i] = obj
                        break
                except AttributeError:
                    continue
        return weights
