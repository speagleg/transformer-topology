"""Higher-order message passing along boundary operators.

Implements bidirectional message passing between cells of adjacent dimensions
(0-cells <-> 1-cells, 1-cells <-> 2-cells) using boundary operators.
"""

import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex


class BoundaryMessagePassing(nn.Module):
    """Bidirectional message passing along a boundary operator.

    Messages flow down (B_k: k-cells -> (k-1)-cells) and up (B_k^T: (k-1)-cells -> k-cells)
    with normalized aggregation, residual connection, and LayerNorm.

    Args:
        embedding_dim: Dimension of cell embeddings.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.down_transform = nn.Linear(embedding_dim, embedding_dim)
        self.up_transform = nn.Linear(embedding_dim, embedding_dim)
        self.combine_lower = nn.Linear(2 * embedding_dim, embedding_dim)
        self.combine_upper = nn.Linear(2 * embedding_dim, embedding_dim)
        self.norm_lower = nn.LayerNorm(embedding_dim)
        self.norm_upper = nn.LayerNorm(embedding_dim)

    def forward(self, lower_features: torch.Tensor, upper_features: torch.Tensor,
                boundary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass messages between lower and upper dimensional cells.

        Args:
            lower_features: Features of (k-1)-cells, shape (n_lower, embedding_dim).
            upper_features: Features of k-cells, shape (n_upper, embedding_dim).
            boundary: Boundary operator B_k, shape (n_lower, n_upper).

        Returns:
            (updated_lower, updated_upper) feature tensors.
        """
        # Normalize boundary for aggregation (column-normalized for down, row-normalized for up)
        col_sum = boundary.abs().sum(dim=0, keepdim=True).clamp(min=1.0)
        B_down = boundary / col_sum  # normalized per upper cell

        row_sum = boundary.abs().sum(dim=1, keepdim=True).clamp(min=1.0)
        B_up = boundary / row_sum  # normalized per lower cell

        # Down messages: upper -> lower
        down_msg = B_down @ self.down_transform(upper_features)
        updated_lower = self.norm_lower(
            self.combine_lower(torch.cat([lower_features, down_msg], dim=-1)) + lower_features
        )

        # Up messages: lower -> upper
        up_msg = B_up.T @ self.up_transform(lower_features)
        updated_upper = self.norm_upper(
            self.combine_upper(torch.cat([upper_features, up_msg], dim=-1)) + upper_features
        )

        return updated_lower, updated_upper


class HigherOrderGNN(nn.Module):
    """Higher-order GNN that chains boundary message passing across dimensions.

    Applies boundary message passing for (0,1)-cells and (1,2)-cells when 2-cells exist.

    Args:
        embedding_dim: Dimension of cell embeddings.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.bmp_01 = BoundaryMessagePassing(embedding_dim)
        self.bmp_12 = BoundaryMessagePassing(embedding_dim)

    def forward(self, cc: CellComplex, node_features: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply higher-order message passing.

        Args:
            cc: Cell complex.
            node_features: Node features, shape (num_nodes, embedding_dim).

        Returns:
            (updated_nodes, updated_edges).
        """
        edge_features = cc.get_embeddings(1)
        B1 = cc.boundary_operator(1)

        # (0,1) message passing
        node_features, edge_features = self.bmp_01(node_features, edge_features, B1)

        # (1,2) message passing if 2-cells exist
        if cc.num_cells(2) > 0:
            face_features = cc.get_embeddings(2)
            B2 = cc.boundary_operator(2)
            edge_features, face_features = self.bmp_12(edge_features, face_features, B2)

        return node_features, edge_features
