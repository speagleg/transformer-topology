import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition


class SpectralFilterLayer(nn.Module):
    """Graph spectral filter operating in the Fourier domain.

    Transforms node signals to the spectral domain via eigenvectors of
    the Hodge Laplacian, applies learnable per-frequency filters, then
    transforms back to the spatial domain.
    """

    def __init__(self, in_dim: int, out_dim: int, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        self.filter_weights = nn.Parameter(torch.ones(num_freqs, in_dim))  # diagonal scaling
        self.output_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
        """Apply spectral filter to node features.

        Args:
            x: Node features of shape (N, in_dim).
            eigenvalues: Eigenvalues of shape (k,).
            eigenvectors: Eigenvectors of shape (N, k).

        Returns:
            Filtered features of shape (N, out_dim).
        """
        k = min(self.num_freqs, eigenvectors.shape[1])
        U = eigenvectors[:, :k]  # (N, k)
        x_hat = U.T @ x  # (k, in_dim) — graph Fourier transform
        x_filtered = x_hat * self.filter_weights[:k]  # (k, in_dim) — diagonal scaling
        out = self.output_proj(U @ x_filtered)  # (N, out_dim) — inverse GFT + projection
        return self.norm(out)


class SpectralGNN(nn.Module):
    """GNN operating in the spectral domain of the cell complex.

    Projects node features, applies multiple spectral filter layers with
    residual connections, then projects to output dimension.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, max_freqs: int):
        super().__init__()
        self.max_freqs = max_freqs
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SpectralFilterLayer(hidden_dim, hidden_dim, max_freqs) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, cc: CellComplex) -> torch.Tensor:
        """Run spectral GNN on a cell complex.

        Args:
            cc: Cell complex with 0-cell embeddings.

        Returns:
            Output features of shape (N, out_dim).
        """
        x = cc.get_embeddings(0)
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0, k=self.max_freqs)
        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            x = layer(x, eigenvalues, eigenvectors)
            if residual.shape == x.shape:
                x = x + residual
        return self.output_proj(x)
