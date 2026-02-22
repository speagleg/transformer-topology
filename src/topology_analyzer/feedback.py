"""Active mode: convert TopologicalProfile into ControlHead feedback + advisories."""

from __future__ import annotations

import torch

from src.topology_analyzer.profile import TopologicalProfile


_DEFAULT_THRESHOLDS = {
    "sheaf_gap_min": 0.01,
    "effective_rank_min": 3.0,
    "attention_curl_max": 0.6,
    "condition_number_max": 1000.0,
    "harmonic_ratio_max": 0.4,
}


class TopologyFeedback:
    """Convert TopologicalProfile into ControlHead topo_features vector."""

    def profile_to_features(self, profile: TopologicalProfile) -> torch.Tensor:
        """Return the 6-feature vector for ControlHead topo_feedback input.

        Same as profile.active_features() but callable from training loop.
        """
        return profile.active_features()


class EmbeddingTopologyAdvisor:
    """Epoch-level advisor: threshold-based alerts from TopologicalProfile."""

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = {**_DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

    def check_alerts(self, profile: TopologicalProfile) -> list[str]:
        """Check for concerning patterns in the topological profile."""
        alerts = []

        # Sheaf gap collapse
        gap_min = self.thresholds["sheaf_gap_min"]
        if profile.cross_layer_sheaf_gap < gap_min:
            alerts.append(
                f"Low cross-layer sheaf coherence: {profile.cross_layer_sheaf_gap:.4f} < {gap_min}. "
                f"Representation bottleneck between layers."
            )

        # Effective rank collapse
        rank_min = self.thresholds["effective_rank_min"]
        for layer_idx, rank in profile.per_layer_effective_rank.items():
            if rank < rank_min:
                alerts.append(
                    f"Low effective rank at layer {layer_idx}: {rank:.1f} < {rank_min}. "
                    f"Consider reinitializing or applying spectral regularization."
                )

        # High attention curl
        curl_max = self.thresholds["attention_curl_max"]
        if profile.per_head_hodge_ratios:
            avg_curl = sum(
                r[1] for r in profile.per_head_hodge_ratios.values()
            ) / len(profile.per_head_hodge_ratios)
            if avg_curl > curl_max:
                alerts.append(
                    f"High average attention curl: {avg_curl:.3f} > {curl_max}. "
                    f"Circular attention patterns detected."
                )

        # High condition number
        cond_max = self.thresholds["condition_number_max"]
        for layer_idx, cond in profile.per_layer_condition_number.items():
            if cond > cond_max:
                alerts.append(
                    f"High condition number at layer {layer_idx}: {cond:.1f} > {cond_max}. "
                    f"Numerically unstable weight matrix."
                )

        # High harmonic ratio (dead patterns)
        harm_max = self.thresholds["harmonic_ratio_max"]
        for layer_idx, (g, c, h) in profile.per_layer_hodge_ratios.items():
            if h > harm_max:
                alerts.append(
                    f"High harmonic ratio at layer {layer_idx}: {h:.3f} > {harm_max}. "
                    f"Dead representation zone."
                )

        return alerts

    def suggest_architecture_changes(
        self, history: list[TopologicalProfile], window: int = 5,
    ) -> list[str]:
        """Analyze trends across checkpoints and suggest changes."""
        if len(history) < 2:
            return []

        recent = history[-window:] if len(history) >= window else history
        suggestions = []

        # Declining sheaf gap
        gaps = [p.cross_layer_sheaf_gap for p in recent]
        if len(gaps) >= 2 and gaps[-1] < gaps[0] * 0.5:
            suggestions.append(
                f"Cross-layer sheaf gap declining ({gaps[0]:.4f} -> {gaps[-1]:.4f}). "
                f"Consider adding skip connections between bottleneck layers."
            )

        # Rising average curl
        avg_curls = []
        for p in recent:
            if p.per_head_hodge_ratios:
                avg_curls.append(
                    sum(r[1] for r in p.per_head_hodge_ratios.values())
                    / len(p.per_head_hodge_ratios)
                )
        if len(avg_curls) >= 2 and avg_curls[-1] > avg_curls[0] + 0.1:
            suggestions.append(
                f"Attention curl increasing ({avg_curls[0]:.3f} -> {avg_curls[-1]:.3f}). "
                f"Increase head dropout or wave damping."
            )

        # Declining effective rank
        for layer_idx in recent[0].per_layer_effective_rank:
            ranks = [p.per_layer_effective_rank.get(layer_idx, 0) for p in recent]
            if ranks and ranks[-1] < ranks[0] * 0.5:
                suggestions.append(
                    f"Layer {layer_idx} effective rank collapsing ({ranks[0]:.1f} -> {ranks[-1]:.1f}). "
                    f"Consider spectral regularization or reinitialization."
                )

        return suggestions
