"""Multi-scale Laplacian filtering across L0, L1, L2 Hodge Laplacians."""

import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex


class MultiScaleLaplacianFilter(nn.Module):
    """Parallel spectral filtering on L0, L1, L2 with gated fusion.

    Runs the same filter type on all three Hodge Laplacians, projects
    higher-order outputs back to node space, and fuses with learned gates.
    """

    def __init__(
        self,
        embedding_dim: int,
        filter_type: str = "wave_cosine",
        skip_l1: bool = False,
        skip_l2: bool = False,
        **filter_kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.skip_l1 = skip_l1
        self.skip_l2 = skip_l2

        from src.wave.spectral_filters import create_spectral_filter

        self.filter_L0 = create_spectral_filter(filter_type, dim=0, **filter_kwargs)

        if not skip_l1:
            self.filter_L1 = create_spectral_filter(filter_type, dim=1, **filter_kwargs)
            self.proj_L1 = nn.Linear(embedding_dim, embedding_dim)

        if not skip_l2:
            self.filter_L2 = create_spectral_filter(filter_type, dim=2, **filter_kwargs)
            self.proj_L2 = nn.Linear(embedding_dim, embedding_dim)

        # Learnable gates (sigmoid-activated)
        self.gate_L0 = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5
        self.gate_L1 = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5
        self.gate_L2 = nn.Parameter(torch.tensor(-1.0))  # start small

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        diffusion_time: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        device = signal.device
        filter_kwargs = {"diffusion_time": diffusion_time} if diffusion_time is not None else {}

        # L0 path (nodes) — always active
        try:
            out_L0 = self.filter_L0(cc, signal, **filter_kwargs)
        except (RuntimeError, ValueError):
            out_L0 = signal

        # L1 path (edges -> nodes)
        out_L1_node = torch.zeros_like(signal)
        if not self.skip_l1 and cc.num_cells(1) > 0:
            edge_signal = cc.get_embeddings(1).to(device)
            try:
                filtered_edges = self.filter_L1(cc, edge_signal, **filter_kwargs)
                B1 = cc.boundary_operator(1).to(device)
                projected = B1 @ filtered_edges  # (N, D)
                out_L1_node = self.proj_L1(projected)
            except (RuntimeError, ValueError):
                pass

        # L2 path (triangles -> nodes)
        out_L2_node = torch.zeros_like(signal)
        if not self.skip_l2 and cc.num_cells(2) > 0:
            tri_signal = cc.get_embeddings(2).to(device)
            try:
                filtered_tris = self.filter_L2(cc, tri_signal, **filter_kwargs)
                I_nt = cc.node_triangle_incidence().to(device)
                projected = I_nt @ filtered_tris  # (N, D)
                out_L2_node = self.proj_L2(projected)
            except (RuntimeError, ValueError):
                pass

        # Gated fusion
        g0 = torch.sigmoid(self.gate_L0)
        g1 = torch.sigmoid(self.gate_L1) if not self.skip_l1 else 0.0
        g2 = torch.sigmoid(self.gate_L2) if not self.skip_l2 else 0.0

        return g0 * out_L0 + g1 * out_L1_node + g2 * out_L2_node
