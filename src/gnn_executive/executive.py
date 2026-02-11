import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.spatial import SpatialGNN
from src.gnn_executive.spectral_filter import SpectralGNN


class GNNExecutive(nn.Module):
    """Dual-path GNN that fuses spatial and spectral representations.

    Runs a spatial message-passing GNN and a spectral filtering GNN in
    parallel on the same cell complex, then combines their outputs with
    a learned gating mechanism and residual connection.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, num_spatial_layers: int,
                 num_spectral_layers: int, max_freqs: int):
        super().__init__()
        self.spatial_gnn = SpatialGNN(
            in_dim=embedding_dim, hidden_dim=hidden_dim,
            out_dim=embedding_dim, num_layers=num_spatial_layers,
        )
        self.spectral_gnn = SpectralGNN(
            in_dim=embedding_dim, hidden_dim=hidden_dim,
            out_dim=embedding_dim, num_layers=num_spectral_layers,
            max_freqs=max_freqs,
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, cc: CellComplex) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run dual-path GNN on a cell complex.

        Args:
            cc: Cell complex with 0-cell and 1-cell embeddings.

        Returns:
            Tuple of (node_embeddings, edge_embeddings). Edge embeddings
            are currently None (reserved for future use).
        """
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()
        spatial_out = self.spatial_gnn(x, edge_index)
        spectral_out = self.spectral_gnn(cc)
        gate_input = torch.cat([spatial_out, spectral_out], dim=-1)
        g = self.gate(gate_input)
        fused = g * spatial_out + (1 - g) * spectral_out
        fused = self.norm(fused + x)
        return fused, None
