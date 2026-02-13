from __future__ import annotations

import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.tat.spatial_attention import TopologicalSpatialAttention
from src.tat.spectral_attention import TopologicalSpectralAttention
from src.spectral.decomposition import spectral_decomposition

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.gnn_executive.control_head import ControlSignal


class TATBlock(nn.Module):
    """Single Topology-Aware Transformer block with dual spatial/spectral attention.

    Runs spatial attention (adjacency-masked) and spectral attention (Fourier-domain)
    in parallel, fuses them via a learned gate, then applies a feed-forward network
    with residual connection and layer normalization.

    Args:
        embed_dim: Embedding dimension.
        num_spatial_heads: Number of heads for spatial attention.
        num_spectral_heads: Number of heads for spectral attention.
        ff_dim: Hidden dimension of the feed-forward network.
        num_freqs: Number of Laplacian eigenfrequencies for spectral attention.
        dropout: Dropout probability.
    """

    def __init__(self, embed_dim: int, num_spatial_heads: int, num_spectral_heads: int,
                 ff_dim: int, num_freqs: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = TopologicalSpatialAttention(
            embed_dim=embed_dim, num_heads=num_spatial_heads, dropout=dropout,
        )
        self.spectral_attn = TopologicalSpectralAttention(
            embed_dim=embed_dim, num_heads=num_spectral_heads,
            num_freqs=num_freqs, dropout=dropout,
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
                eigenvalues: torch.Tensor, eigenvectors: torch.Tensor,
                control_signal: ControlSignal | None = None,
                edge_weights: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the dual-attention block.

        Args:
            x: Node features of shape (N, embed_dim).
            adjacency: Binary adjacency matrix of shape (N, N).
            eigenvalues: Laplacian eigenvalues of shape (num_freqs,).
            eigenvectors: Laplacian eigenvectors of shape (N, num_freqs).
            control_signal: Optional :class:`ControlSignal` from the GNN
                executive.  When provided, ``spatial_focus`` biases spatial
                attention and ``frequency_gate`` gates spectral attention.
            edge_weights: Optional tensor of shape (N, N) with scalar edge
                weights passed to spatial attention.

        Returns:
            Output tensor of shape (N, embed_dim).
        """
        spatial_focus = control_signal.spatial_focus if control_signal is not None else None
        frequency_gate = control_signal.frequency_gate if control_signal is not None else None

        spatial_out = self.spatial_attn(x, adjacency=adjacency, spatial_focus=spatial_focus,
                                        edge_weights=edge_weights)
        spectral_out = self.spectral_attn(x, eigenvalues=eigenvalues, eigenvectors=eigenvectors,
                                          frequency_gate=frequency_gate)
        g = self.gate(torch.cat([spatial_out, spectral_out], dim=-1))
        x = g * spatial_out + (1 - g) * spectral_out
        x = x + self.ff(x)
        x = self.ff_norm(x)
        return x


class TopologyAwareTransformer(nn.Module):
    """Full Topology-Aware Transformer that stacks TATBlocks.

    Extracts embeddings and topology from a CellComplex, computes spectral
    decomposition, then passes through a stack of dual spatial/spectral
    attention blocks. Optionally adds topological positional encodings.

    Args:
        embedding_dim: Embedding dimension.
        num_layers: Number of TATBlock layers.
        num_spatial_heads: Number of heads for spatial attention in each block.
        num_spectral_heads: Number of heads for spectral attention in each block.
        ff_dim: Hidden dimension of the feed-forward network in each block.
        num_freqs: Number of Laplacian eigenfrequencies for spectral attention.
        dropout: Dropout probability.
        use_topological_pe: Whether to use topological positional encodings.
    """

    def __init__(self, embedding_dim: int, num_layers: int, num_spatial_heads: int,
                 num_spectral_heads: int, ff_dim: int, num_freqs: int, dropout: float = 0.1,
                 use_topological_pe: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.use_topological_pe = use_topological_pe
        self.blocks = nn.ModuleList([
            TATBlock(embed_dim=embedding_dim, num_spatial_heads=num_spatial_heads,
                     num_spectral_heads=num_spectral_heads, ff_dim=ff_dim,
                     num_freqs=num_freqs, dropout=dropout)
            for _ in range(num_layers)
        ])

        if use_topological_pe:
            from src.spectral.positional_encoding import TopologicalPositionalEncoding
            self.topo_pe = TopologicalPositionalEncoding(
                embedding_dim=embedding_dim,
                num_eigenvectors=min(num_freqs, 8),
                num_persistence_features=8,
                max_2_cells=32,
            )

    def forward(self, cc: CellComplex,
                control_signal: ControlSignal | None = None) -> torch.Tensor:
        """Forward pass through the full transformer.

        Extracts node embeddings and adjacency from the cell complex,
        computes spectral decomposition, then runs through all TATBlocks.

        Args:
            cc: Input CellComplex with 0-cells and 1-cells.
            control_signal: Optional :class:`ControlSignal` from the GNN
                executive.  Passed through to each :class:`TATBlock` to
                modulate spatial and spectral attention.

        Returns:
            Output node features of shape (N, embedding_dim).
        """
        x = cc.get_embeddings(0)
        adjacency = cc.adjacency_matrix(0)
        adjacency = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)
        adjacency = (adjacency > 0).float()
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0, k=self.num_freqs)

        # Build edge weight matrix from 1-cell embeddings (dim 0 = delay/weight)
        edge_weights = cc.edge_weight_matrix(feature_dim=0) if cc.num_cells(1) > 0 else None

        if self.use_topological_pe:
            topo_pe = self.topo_pe(cc)
            x = x + topo_pe

        for block in self.blocks:
            x = block(x, adjacency, eigenvalues, eigenvectors,
                      control_signal=control_signal, edge_weights=edge_weights)
        return x
