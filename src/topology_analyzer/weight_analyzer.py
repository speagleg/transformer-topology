"""Layer 3: Weight Space Geometry Analyzer.

SVD analysis of weight matrices: effective rank, condition number,
singular value persistence, spectral gap.
"""

from __future__ import annotations

import torch
import numpy as np

from src.topology_analyzer.profile import TopologicalProfile

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


class WeightSpaceAnalyzer:
    """Analyze the topological structure of learned weight matrices."""

    def analyze_weight_matrix(self, W: torch.Tensor, layer_idx: int) -> dict:
        """Compute all weight space metrics for one weight matrix.

        Args:
            W: Weight matrix of shape (d_out, d_in).
            layer_idx: Which layer this weight belongs to.

        Returns:
            Dict with effective_rank, condition_number, sv_persistence, spectral_gap.
        """
        W = W.detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Effective rank: exp(entropy(p)) where p = S/sum(S)
        s = S.clamp(min=1e-10)
        p = s / s.sum()
        entropy = -(p * p.log()).sum().item()
        effective_rank = np.exp(entropy)

        # Condition number
        condition_number = (S[0] / S[-1]).item() if S[-1] > 1e-10 else float('inf')

        # Spectral gap of W^T @ W (gap between top 2 squared singular values)
        eigenvalues = S ** 2
        if len(eigenvalues) >= 2:
            spectral_gap = (eigenvalues[0] - eigenvalues[1]).item()
        else:
            spectral_gap = 0.0

        sv_persistence = self._sv_persistence(S.cpu().numpy())

        return {
            "effective_rank": effective_rank,
            "condition_number": condition_number,
            "spectral_gap": spectral_gap,
            "sv_persistence": sv_persistence,
            "layer_idx": layer_idx,
        }

    @staticmethod
    def _sv_persistence(singular_values: np.ndarray) -> list[np.ndarray]:
        """Compute persistence of singular values as a 1D point cloud."""
        if not GUDHI_AVAILABLE or len(singular_values) < 2:
            return [np.empty((0, 2))]

        points = singular_values.reshape(-1, 1)
        rips = gudhi.RipsComplex(points=points, max_edge_length=float('inf'))
        st = rips.create_simplex_tree(max_dimension=1)
        st.compute_persistence()

        pairs = st.persistence_intervals_in_dimension(0)
        if len(pairs) == 0:
            return [np.empty((0, 2))]

        finite = pairs[np.isfinite(pairs[:, 1])]
        if len(finite) == 0:
            return [np.empty((0, 2))]
        return [finite]

    def training_dynamics(self, profiles: list[TopologicalProfile]) -> dict:
        """Compute Wasserstein distances between consecutive profiles.

        Args:
            profiles: List of TopologicalProfile from consecutive checkpoints.

        Returns:
            Dict with wasserstein_distances and phase_transitions.
        """
        if len(profiles) < 2:
            return {"wasserstein_distances": [], "phase_transitions": []}

        distances = []
        for i in range(len(profiles) - 1):
            d = self._wasserstein_between(profiles[i], profiles[i + 1])
            distances.append(d)

        phase_transitions = []
        if distances:
            median_d = sorted(distances)[len(distances) // 2]
            for i, d in enumerate(distances):
                if d > 2 * median_d and median_d > 1e-6:
                    phase_transitions.append(i + 1)

        return {
            "wasserstein_distances": distances,
            "phase_transitions": phase_transitions,
        }

    @staticmethod
    def _wasserstein_between(p1: TopologicalProfile, p2: TopologicalProfile) -> float:
        """Approximate Wasserstein distance between SV persistence diagrams."""
        total_dist = 0.0
        count = 0

        for layer_idx in p1.per_layer_sv_persistence:
            if layer_idx not in p2.per_layer_sv_persistence:
                continue
            diags1 = p1.per_layer_sv_persistence[layer_idx]
            diags2 = p2.per_layer_sv_persistence[layer_idx]
            for d1, d2 in zip(diags1, diags2):
                if d1.shape[0] == 0 and d2.shape[0] == 0:
                    continue
                lt1 = np.sort(d1[:, 1] - d1[:, 0])[::-1] if d1.shape[0] > 0 else np.array([0.0])
                lt2 = np.sort(d2[:, 1] - d2[:, 0])[::-1] if d2.shape[0] > 0 else np.array([0.0])
                max_len = max(len(lt1), len(lt2))
                lt1_pad = np.zeros(max_len)
                lt2_pad = np.zeros(max_len)
                lt1_pad[:len(lt1)] = lt1
                lt2_pad[:len(lt2)] = lt2
                total_dist += np.sum(np.abs(lt1_pad - lt2_pad))
                count += 1

        return total_dist / max(count, 1)
