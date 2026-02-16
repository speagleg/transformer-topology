"""Sheaf diffusion on cell complexes with learnable restriction maps.

Instead of scalar edge weights, each edge carries a pair of linear maps
(restriction maps) that specify how node features project onto the edge
stalk. This gives a block-structured sheaf Laplacian that generalizes
the standard graph Laplacian to vector-valued signals.

Reference: Bodnar, Di Giovanni et al. (2022) "Neural Sheaf Diffusion:
A Topological Perspective on Heterophily and Oversmoothing in GNNs".
"""

import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex


class SheafLaplacian(nn.Module):
    """Learnable sheaf Laplacian over a cell complex.

    Each edge e = (u, v) has two restriction maps:
    - F_{e,u}: R^d -> R^d (how node u's features project onto edge e)
    - F_{e,v}: R^d -> R^d (how node v's features project onto edge e)

    The sheaf Laplacian is a (N*d, N*d) block matrix:
    - L_sheaf[u, v] = -F_{e,u}^T @ F_{e,v}  (for edge e between u, v)
    - L_sheaf[u, u] = sum_{e containing u} F_{e,u}^T @ F_{e,u}

    When all restriction maps are identity, this reduces to the standard
    graph Laplacian (tensorized with I_d).
    """

    def __init__(self, n_edges: int, feature_dim: int, rank: int | None = None):
        """Initialize the sheaf Laplacian.

        Args:
            n_edges: Number of edges in the complex.
            feature_dim: Dimension of node features (d).
            rank: If set, use low-rank restriction maps F = A @ B where
                A is (d, r) and B is (r, d). Reduces parameters from
                2 * n_edges * d^2 to 2 * n_edges * d * r * 2.
        """
        super().__init__()
        self.n_edges = n_edges
        self.feature_dim = feature_dim
        self.rank = rank

        if rank is not None:
            # Low-rank: F_{e,node} = A_{e,node} @ B_{e,node}
            # A: (n_edges, 2, d, rank), B: (n_edges, 2, rank, d)
            # Index 0 = source restriction, Index 1 = target restriction
            self.A = nn.Parameter(torch.randn(n_edges, 2, feature_dim, rank) * 0.1)
            self.B = nn.Parameter(torch.randn(n_edges, 2, rank, feature_dim) * 0.1)
        else:
            # Full-rank restriction maps: (n_edges, 2, d, d)
            # Initialized near identity for stable training start
            maps = torch.zeros(n_edges, 2, feature_dim, feature_dim)
            for e in range(n_edges):
                maps[e, 0] = torch.eye(feature_dim) + torch.randn(feature_dim, feature_dim) * 0.01
                maps[e, 1] = torch.eye(feature_dim) + torch.randn(feature_dim, feature_dim) * 0.01
            self.restriction_maps = nn.Parameter(maps)

    def get_restriction_map(self, edge_idx: int, endpoint: int) -> torch.Tensor:
        """Get the restriction map for a given edge and endpoint.

        Args:
            edge_idx: Edge index.
            endpoint: 0 for source, 1 for target.

        Returns:
            Tensor of shape (d, d).
        """
        if self.rank is not None:
            return self.A[edge_idx, endpoint] @ self.B[edge_idx, endpoint]
        return self.restriction_maps[edge_idx, endpoint]

    def forward(self, cc: CellComplex) -> torch.Tensor:
        """Build the full (N*d, N*d) sheaf Laplacian matrix.

        Args:
            cc: Cell complex defining the graph structure.

        Returns:
            Block Laplacian tensor of shape (N*d, N*d).
        """
        n = cc.num_cells(0)
        d = self.feature_dim
        device = next(self.parameters()).device

        L = torch.zeros(n * d, n * d, device=device)

        n_edges = cc.num_cells(1)
        for e in range(n_edges):
            u = cc._1_cell_sources[e]
            v = cc._1_cell_targets[e]

            F_eu = self.get_restriction_map(e, 0)  # (d, d)
            F_ev = self.get_restriction_map(e, 1)  # (d, d)

            # Off-diagonal blocks: L[u_block, v_block] -= F_eu^T @ F_ev
            off_diag = F_eu.T @ F_ev  # (d, d)
            L[u * d:(u + 1) * d, v * d:(v + 1) * d] -= off_diag
            L[v * d:(v + 1) * d, u * d:(u + 1) * d] -= F_ev.T @ F_eu

            # Diagonal blocks: L[u_block, u_block] += F_eu^T @ F_eu
            L[u * d:(u + 1) * d, u * d:(u + 1) * d] += F_eu.T @ F_eu
            L[v * d:(v + 1) * d, v * d:(v + 1) * d] += F_ev.T @ F_ev

        return L


class SheafDiffusion(nn.Module):
    """Diffusion using the sheaf Laplacian.

    Applies the heat equation on the sheaf: f(t) = exp(-t * L_sheaf) @ f(0).
    Since L_sheaf can be large (N*d x N*d), we use Euler integration
    rather than full eigendecomposition.

    f_{k+1} = f_k - dt * L_sheaf @ f_k
    """

    def __init__(self, feature_dim: int, n_steps: int = 5):
        """Initialize sheaf diffusion.

        Args:
            feature_dim: Dimension of node features.
            n_steps: Number of Euler integration steps.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.n_steps = n_steps

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        sheaf_laplacian: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        """Diffuse node features using the sheaf Laplacian.

        Args:
            cc: Cell complex (used for node count).
            signal: Node features of shape (N, d).
            sheaf_laplacian: Pre-computed sheaf Laplacian of shape (N*d, N*d).
            diffusion_time: Positive scalar tensor controlling diffusion extent.

        Returns:
            Diffused signal of shape (N, d).
        """
        n = cc.num_cells(0)
        d = self.feature_dim

        # Flatten signal to (N*d,) for block-matrix multiplication
        f = signal.reshape(n * d)

        # Clamp dt for Euler stability: dt * λ_max must be < 2.
        # With normalized Laplacian (λ_max ≈ 1), dt < 1.0 is safe.
        dt = (diffusion_time / self.n_steps).clamp(max=0.9)

        for _ in range(self.n_steps):
            # Euler step: f_{k+1} = f_k - dt * L_sheaf @ f_k
            f = f - dt * (sheaf_laplacian @ f)

        return f.reshape(n, d)
