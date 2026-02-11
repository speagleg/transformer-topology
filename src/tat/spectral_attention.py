import torch
import torch.nn as nn
import math


class TopologicalSpectralAttention(nn.Module):
    """Multi-head attention in the graph Fourier domain.

    Projects tokens to the Laplacian eigenbasis, performs attention in
    frequency space with learnable spectral filters, then projects back
    to the spatial domain.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_freqs: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_freqs = num_freqs
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.spectral_filter = nn.Parameter(torch.ones(num_heads, num_freqs))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
        """Spectral attention forward pass.

        Args:
            x: Node features of shape (n, embed_dim).
            eigenvalues: Laplacian eigenvalues of shape (num_freqs,) or larger.
            eigenvectors: Laplacian eigenvectors of shape (n, num_freqs) or larger.

        Returns:
            Output tensor of shape (n, embed_dim).
        """
        n = x.shape[0]
        k = min(self.num_freqs, eigenvectors.shape[1])
        residual = x
        U = eigenvectors[:, :k]  # (n, k)

        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k_proj = self.k_proj(x).view(n, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_heads, self.head_dim)

        # Transform Q, K to spectral domain
        q_hat = torch.einsum("nk,nhd->khd", U, q)
        k_hat = torch.einsum("nk,nhd->khd", U, k_proj)

        # Attention in spectral domain per frequency
        scores = torch.einsum("khd,khd->kh", q_hat, k_hat) / self.scale
        filtered_scores = scores * self.spectral_filter[:, :k].T
        weights = torch.softmax(filtered_scores, dim=0)
        weights = self.dropout(weights)

        # Weighted combination in spectral domain
        v_hat = torch.einsum("nk,nhd->khd", U, v)
        out_hat = torch.einsum("kh,khd->hd", weights, v_hat)

        # Broadcast back to spatial domain
        out_spatial = torch.einsum("nk,kh->nh", U, weights)
        out = torch.einsum("nh,nhd->nhd", out_spatial, v)
        out = out.contiguous().view(n, self.embed_dim)
        out = self.out_proj(out)
        return self.norm(out + residual)
