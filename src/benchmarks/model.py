import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.loop import ReasoningLoop


class MultiHopReasoningModel(nn.Module):
    def __init__(self, embedding_dim: int, gnn_hidden: int, gnn_spatial_layers: int,
                 gnn_spectral_layers: int, max_freqs: int, tat_layers: int,
                 tat_spatial_heads: int, tat_spectral_heads: int, tat_ff_dim: int,
                 max_hops: int, max_iterations: int = 3):
        super().__init__()
        self.reasoning_loop = ReasoningLoop(
            embedding_dim=embedding_dim, gnn_hidden=gnn_hidden,
            gnn_spatial_layers=gnn_spatial_layers,
            gnn_spectral_layers=gnn_spectral_layers,
            max_freqs=max_freqs, tat_layers=tat_layers,
            tat_spatial_heads=tat_spatial_heads,
            tat_spectral_heads=tat_spectral_heads,
            tat_ff_dim=tat_ff_dim, max_iterations=max_iterations,
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, max_hops + 1),
        )

    def forward(self, cc: CellComplex, query_node: int, target_node: int) -> torch.Tensor:
        output, num_iters = self.reasoning_loop(cc)
        query_emb = output[query_node]
        target_emb = output[target_node]
        diff_emb = query_emb - target_emb
        combined = torch.cat([query_emb, target_emb, diff_emb])
        logits = self.classifier(combined)
        return logits
