import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition
from src.spectral.persistence import persistence_node_features, GUDHI_AVAILABLE


def laplacian_pe(cc: CellComplex, dim: int, k: int) -> torch.Tensor:
    """Compute Laplacian positional encodings for cells of the given dimension.

    Returns the k smallest eigenvectors of the Hodge Laplacian as positional
    encodings. Each row corresponds to a cell, each column to an eigenvector.

    Args:
        cc: Cell complex.
        dim: Cell dimension (0 or 1).
        k: Number of eigenvectors to use as positional encoding dimensions.

    Returns:
        Tensor of shape (num_cells, k) -- the positional encodings.
    """
    eigenvalues, eigenvectors = spectral_decomposition(cc, dim=dim, k=k)
    return eigenvectors


def cell_membership_encoding(cc: CellComplex) -> torch.Tensor:
    """Compute binary indicators of which 2-cells each node participates in.

    Args:
        cc: Cell complex with 0-cells and potentially 2-cells.

    Returns:
        Tensor of shape (num_nodes, num_2_cells). Zero matrix if no 2-cells.
    """
    n0 = cc.num_cells(0)
    n2 = cc.num_cells(2)
    if n2 == 0:
        return torch.zeros(n0, 0, device=cc.device)

    membership = torch.zeros(n0, n2, device=cc.device)
    for face_idx, boundary_edges in enumerate(cc._2_cell_boundaries):
        # Collect all nodes involved in this face's boundary edges
        nodes_in_face = set()
        for e in boundary_edges:
            nodes_in_face.add(cc._1_cell_sources[e])
            nodes_in_face.add(cc._1_cell_targets[e])
        for node in nodes_in_face:
            membership[node, face_idx] = 1.0

    return membership


class TopologicalPositionalEncoding(nn.Module):
    """Combined positional encoding from Laplacian PE + persistence + cell membership.

    Concatenates three feature sources and projects to embedding_dim:
    - Laplacian eigenvectors (spectral position)
    - Persistence features (topological scale)
    - Cell membership indicators (higher-order structure)

    Args:
        embedding_dim: Output embedding dimension.
        num_eigenvectors: Number of Laplacian eigenvectors to use.
        num_persistence_features: Number of persistence features per node.
        max_2_cells: Maximum number of 2-cells to encode membership for.
    """

    def __init__(self, embedding_dim: int, num_eigenvectors: int = 8,
                 num_persistence_features: int = 8, max_2_cells: int = 32):
        super().__init__()
        self.num_eigenvectors = num_eigenvectors
        self.num_persistence_features = num_persistence_features
        self.max_2_cells = max_2_cells
        input_dim = num_eigenvectors + num_persistence_features + max_2_cells
        self.proj = nn.Linear(input_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.act = nn.GELU()

    def forward(self, cc: CellComplex) -> torch.Tensor:
        """Compute topological positional encodings for all nodes.

        Args:
            cc: Cell complex with node embeddings.

        Returns:
            Tensor of shape (num_nodes, embedding_dim).
        """
        num_nodes = cc.num_cells(0)

        # Laplacian PE
        k = min(self.num_eigenvectors, num_nodes)
        lap_pe = laplacian_pe(cc, dim=0, k=k)
        if lap_pe.shape[1] < self.num_eigenvectors:
            pad = torch.zeros(num_nodes, self.num_eigenvectors - lap_pe.shape[1],
                              device=lap_pe.device)
            lap_pe = torch.cat([lap_pe, pad], dim=1)

        # Persistence features (computed on CPU via numpy/gudhi, move to device)
        pers_feat = persistence_node_features(cc, num_features=self.num_persistence_features)
        pers_feat = pers_feat.to(cc.device)

        # Cell membership
        membership = cell_membership_encoding(cc)
        if membership.shape[1] < self.max_2_cells:
            pad = torch.zeros(num_nodes, self.max_2_cells - membership.shape[1],
                              device=membership.device)
            membership = torch.cat([membership, pad], dim=1)
        elif membership.shape[1] > self.max_2_cells:
            membership = membership[:, :self.max_2_cells]

        combined = torch.cat([lap_pe, pers_feat, membership], dim=1)
        return self.act(self.norm(self.proj(combined)))
