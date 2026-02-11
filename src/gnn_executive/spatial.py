import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SpatialMessagePassingLayer(MessagePassing):
    """Single layer of spatial message passing on a cell complex graph.

    Uses concatenation-based messages with MLP transformations and
    residual connections with layer normalization.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="add")
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = self.propagate(edge_index, x=x)
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        out = (out + x) if x.shape[-1] == out.shape[-1] else out
        return self.norm(out)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.message_mlp(torch.cat([x_i, x_j], dim=-1))


class SpatialGNN(nn.Module):
    """Multi-layer spatial GNN operating on cell complex 0-cells.

    Stacks multiple SpatialMessagePassingLayer instances with optional
    input/output projections when dimensions differ.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [SpatialMessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_proj(x)
