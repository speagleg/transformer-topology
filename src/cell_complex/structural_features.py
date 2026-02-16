import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex


class StructuralFeatureEncoder(nn.Module):
    """Encodes structural node features (degree, role indicators, size) into embedding space.

    Takes the 7-dimensional structural features from CellComplex.compute_structural_features()
    and projects them to embedding_dim, to be added to node embeddings.
    """

    def __init__(self, embedding_dim: int, num_features: int = 7):
        super().__init__()
        self.proj = nn.Linear(num_features, embedding_dim)

    def forward(self, cc: CellComplex) -> torch.Tensor:
        """Compute and encode structural features for all nodes.

        Args:
            cc: Cell complex with 0-cells.

        Returns:
            Tensor of shape (N, embedding_dim) to be added to node embeddings.
        """
        features = cc.compute_structural_features()
        return self.proj(features)
