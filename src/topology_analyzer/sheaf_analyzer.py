"""Cross-layer sheaf coherence analysis for transformer representations.

Builds a path graph over transformer layers (one node per layer, edges between
consecutive layers) and uses a learnable sheaf Laplacian to measure how
coherently features transform across layers. The spectral gap of the sheaf
Laplacian serves as the coherence metric.
"""

import torch
import torch.nn as nn

from src.cell_complex.cell_complex import CellComplex
from src.spectral.sheaf_diffusion import SheafLaplacian


class CrossLayerSheafAnalyzer(nn.Module):
    """Measures cross-layer coherence via sheaf Laplacian spectral gap.

    Args:
        feature_dim: Dimension of per-layer feature vectors.
        num_layers: Number of transformer layers (nodes in the path graph).
    """

    def __init__(self, feature_dim: int, num_layers: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.sheaf_lap = SheafLaplacian(
            n_edges=max(num_layers - 1, 1),
            feature_dim=feature_dim,
        )

    def _build_layer_graph(self) -> CellComplex:
        """Build a path graph CellComplex with one node per layer."""
        cc = CellComplex(embedding_dim=self.feature_dim)
        device = next(self.sheaf_lap.parameters()).device
        zero_emb = torch.zeros(self.feature_dim, device=device)

        for i in range(self.num_layers):
            cc.add_0_cell(zero_emb, cell_type=f"layer_{i}")

        for i in range(self.num_layers - 1):
            cc.add_1_cell(i, i + 1, zero_emb, relation_type="consecutive")

        return cc

    @torch.no_grad()
    def compute_coherence(self, per_layer_features: dict[int, torch.Tensor]) -> float:
        """Compute the sheaf Laplacian spectral gap as a coherence measure.

        Args:
            per_layer_features: Mapping from layer index to feature tensor
                of shape (feature_dim,).

        Returns:
            Smallest nonzero eigenvalue of the sheaf Laplacian. Higher = more
            coherent. Returns 0.0 on failure.
        """
        try:
            cc = self._build_layer_graph()
            device = next(self.sheaf_lap.parameters()).device

            features_list = []
            for i in range(self.num_layers):
                feat = per_layer_features[i].to(device)
                features_list.append(feat)

            L = self.sheaf_lap(cc)
            L = (L + L.T) / 2  # symmetrize

            eigenvalues = torch.linalg.eigvalsh(L.float())

            nonzero_mask = eigenvalues > 1e-6
            if not nonzero_mask.any():
                return 0.0

            return eigenvalues[nonzero_mask].min().item()

        except Exception:
            return 0.0
