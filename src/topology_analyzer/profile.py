"""Data structures for transformer topology analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.cell_complex.cell_complex import CellComplex


@dataclass
class LayerCellComplex:
    """CellComplex built from one transformer layer's hidden states or attention."""
    cc: CellComplex
    layer_idx: int
    source: str  # "embedding" | "attention" | "weight"
    head_idx: int | None  # for attention complexes
    neighborhood: str  # "knn" | "epsilon" | "mutual_knn"
    k_or_epsilon: float


@dataclass
class TopologicalProfile:
    """Complete topological profile of a transformer at one point in time."""

    # Layer 1: Embedding Manifold
    per_layer_persistence: dict[int, list[np.ndarray]]
    per_layer_betti: dict[int, tuple[int, int]]
    per_layer_spectral_gap: dict[int, float]
    per_layer_hodge_ratios: dict[int, tuple[float, float, float]]
    cross_layer_sheaf_gap: float

    # Layer 2: Attention Flow
    per_head_hodge_ratios: dict[tuple[int, int], tuple[float, float, float]]
    per_head_classification: dict[tuple[int, int], str]
    per_layer_head_sheaf_gap: dict[int, float]
    cross_layer_attention_wasserstein: dict[int, float]

    # Layer 3: Weight Space
    per_layer_effective_rank: dict[int, float]
    per_layer_condition_number: dict[int, float]
    per_layer_sv_persistence: dict[int, list[np.ndarray]]
    cross_layer_weight_sheaf_gap: float

    # Metadata
    model_name: str
    num_layers: int
    num_heads: int
    hidden_dim: int
    seq_len: int
    timestamp: float

    def summary(self) -> dict[str, float]:
        """Return a flat dict of key scalar metrics."""
        result: dict[str, float] = {
            "cross_layer_sheaf_gap": self.cross_layer_sheaf_gap,
            "cross_layer_weight_sheaf_gap": self.cross_layer_weight_sheaf_gap,
        }
        for layer_idx, gap in self.per_layer_spectral_gap.items():
            result[f"layer_{layer_idx}_spectral_gap"] = gap
        for layer_idx, (g, c, h) in self.per_layer_hodge_ratios.items():
            result[f"layer_{layer_idx}_gradient_ratio"] = g
            result[f"layer_{layer_idx}_curl_ratio"] = c
            result[f"layer_{layer_idx}_harmonic_ratio"] = h
        for layer_idx, rank in self.per_layer_effective_rank.items():
            result[f"layer_{layer_idx}_effective_rank"] = rank
        for (li, hi), cls in self.per_head_classification.items():
            result[f"head_{li}_{hi}_type"] = {
                "gradient": 0.0, "curl": 1.0, "harmonic": 2.0,
            }.get(cls, -1.0)
        return result

    def active_features(self) -> torch.Tensor:
        """Return the 6-feature vector for ControlHead topo_feedback.

        Features:
            [0] gradient_ratio   - mean over per_layer_hodge_ratios
            [1] curl_ratio       - mean over per_layer_hodge_ratios
            [2] spectral_gap     - mean over per_layer_spectral_gap
            [3] embedding_coherence - cross_layer_sheaf_gap
            [4] attention_curl   - mean curl across all heads
            [5] weight_alignment - cross_layer_weight_sheaf_gap
        """
        features = torch.zeros(6)

        if self.per_layer_hodge_ratios:
            grads = [r[0] for r in self.per_layer_hodge_ratios.values()]
            curls = [r[1] for r in self.per_layer_hodge_ratios.values()]
            features[0] = sum(grads) / len(grads)
            features[1] = sum(curls) / len(curls)

        if self.per_layer_spectral_gap:
            gaps = list(self.per_layer_spectral_gap.values())
            features[2] = sum(gaps) / len(gaps)

        features[3] = self.cross_layer_sheaf_gap

        if self.per_head_hodge_ratios:
            curls = [r[1] for r in self.per_head_hodge_ratios.values()]
            features[4] = sum(curls) / len(curls)

        features[5] = self.cross_layer_weight_sheaf_gap

        return features
