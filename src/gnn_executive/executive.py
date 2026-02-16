import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.spatial import SpatialGNN
from src.gnn_executive.spectral_filter import SpectralGNN
from src.gnn_executive.higher_order import HigherOrderGNN
from src.gnn_executive.control_head import ControlHead, ControlSignal


class GNNExecutive(nn.Module):
    """Dual-path GNN that fuses spatial and spectral representations.

    Runs a spatial message-passing GNN and a spectral filtering GNN in
    parallel on the same cell complex, then combines their outputs with
    a learned gating mechanism and residual connection. Optionally applies
    higher-order message passing when 2-cells are present.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, num_spatial_layers: int,
                 num_spectral_layers: int, max_freqs: int,
                 use_higher_order: bool = False,
                 produce_control_signals: bool = False,
                 num_filters: int = 0):
        super().__init__()
        self.use_higher_order = use_higher_order
        self.produce_control_signals = produce_control_signals
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

        if use_higher_order:
            self.higher_order = HigherOrderGNN(embedding_dim)
            self.ho_gate = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.Sigmoid(),
            )

        if produce_control_signals:
            self.control_head = ControlHead(
                embedding_dim=embedding_dim, num_freqs=max_freqs,
                num_filters=num_filters,
            )

    def forward(self, cc: CellComplex) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run dual-path GNN on a cell complex.

        Args:
            cc: Cell complex with 0-cell and 1-cell embeddings.

        Returns:
            Tuple of (node_embeddings, edge_embeddings). Edge embeddings
            are None unless higher-order message passing is enabled.
        """
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()
        spatial_out = self.spatial_gnn(x, edge_index)
        spectral_out = self.spectral_gnn(cc)
        gate_input = torch.cat([spatial_out, spectral_out], dim=-1)
        g = self.gate(gate_input)
        fused = g * spatial_out + (1 - g) * spectral_out
        fused = self.norm(fused + x)

        edge_out = None
        if self.use_higher_order:
            ho_nodes, ho_edges = self.higher_order(cc, fused)
            # Gate higher-order contribution
            ho_g = self.ho_gate(torch.cat([fused, ho_nodes], dim=-1))
            fused = self.norm(ho_g * fused + (1 - ho_g) * ho_nodes)
            edge_out = ho_edges

        return fused, edge_out

    def forward_with_control(
        self, cc: CellComplex, harmonic_energy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, ControlSignal]:
        """Run dual-path GNN and produce control signals for the TAT.

        Calls the standard forward pass, then feeds the fused node embeddings
        through the ControlHead to produce a ControlSignal.

        Args:
            cc: Cell complex with 0-cell and 1-cell embeddings.
            harmonic_energy: Optional scalar tensor with harmonic component energy.

        Returns:
            Tuple of (node_embeddings, edge_embeddings, control_signal).
            Edge embeddings are None unless higher-order message passing is enabled.

        Raises:
            RuntimeError: If produce_control_signals was not enabled at construction.
        """
        if not self.produce_control_signals:
            raise RuntimeError(
                "forward_with_control() requires produce_control_signals=True"
            )
        fused, edge_out = self.forward(cc)
        control = self.control_head(fused, harmonic_energy=harmonic_energy)
        return fused, edge_out, control
