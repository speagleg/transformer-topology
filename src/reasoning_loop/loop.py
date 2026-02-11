import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.executive import GNNExecutive
from src.tat.transformer import TopologyAwareTransformer


class ReasoningLoop(nn.Module):
    """Interleaved GNN-TAT co-processing loop with convergence detection.

    Alternates between the GNN Executive (spatial + spectral message passing)
    and the Topology-Aware Transformer (dual attention), blending their outputs
    at each iteration and updating the cell complex embeddings in place.  The
    loop terminates early when the embedding delta falls below a convergence
    threshold, or after a fixed number of iterations.

    Args:
        embedding_dim: Dimension of cell embeddings.
        gnn_hidden: Hidden dimension for the GNN Executive.
        gnn_spatial_layers: Number of spatial GNN layers.
        gnn_spectral_layers: Number of spectral GNN layers.
        max_freqs: Number of Laplacian eigenfrequencies.
        tat_layers: Number of Topology-Aware Transformer blocks.
        tat_spatial_heads: Number of spatial attention heads per TAT block.
        tat_spectral_heads: Number of spectral attention heads per TAT block.
        tat_ff_dim: Feed-forward hidden dimension in TAT blocks.
        max_iterations: Maximum number of reasoning iterations.
        convergence_threshold: Stop when embedding delta norm falls below this.
    """

    def __init__(self, embedding_dim: int, gnn_hidden: int, gnn_spatial_layers: int,
                 gnn_spectral_layers: int, max_freqs: int, tat_layers: int,
                 tat_spatial_heads: int, tat_spectral_heads: int, tat_ff_dim: int,
                 max_iterations: int = 5, convergence_threshold: float = 0.01):
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.gnn_executive = GNNExecutive(
            embedding_dim=embedding_dim, hidden_dim=gnn_hidden,
            num_spatial_layers=gnn_spatial_layers,
            num_spectral_layers=gnn_spectral_layers, max_freqs=max_freqs,
        )
        self.tat = TopologyAwareTransformer(
            embedding_dim=embedding_dim, num_layers=tat_layers,
            num_spatial_heads=tat_spatial_heads,
            num_spectral_heads=tat_spectral_heads,
            ff_dim=tat_ff_dim, num_freqs=max_freqs,
        )
        self.blend = nn.Linear(2 * embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, cc: CellComplex) -> tuple[torch.Tensor, int]:
        """Run the interleaved reasoning loop on a cell complex.

        Each iteration:
          1. GNN Executive processes the cell complex (spatial + spectral).
          2. Updated embeddings are written back to the cell complex.
          3. TAT processes the cell complex (dual attention).
          4. GNN and TAT outputs are blended and normalized with a residual.
          5. Blended embeddings are written back to the cell complex.
          6. Convergence is checked via L2 norm of the embedding delta.

        Args:
            cc: Input CellComplex with 0-cells and 1-cells.

        Returns:
            Tuple of (final_node_embeddings, num_iterations_run).
        """
        prev_embeddings = cc.get_embeddings(0)
        num_iters = 0

        for i in range(self.max_iterations):
            num_iters = i + 1

            # Step 1: GNN Executive
            gnn_out, _ = self.gnn_executive(cc)

            # Step 2: Update cell complex for TAT
            cc.set_embeddings(0, gnn_out.detach() if not self.training else gnn_out)

            # Step 3: Topology-Aware Transformer
            tat_out = self.tat(cc)

            # Step 4: Blend GNN and TAT outputs with residual
            blended = self.blend(torch.cat([gnn_out, tat_out], dim=-1))
            current_embeddings = self.norm(blended + prev_embeddings)

            # Step 5: Write blended embeddings back to cell complex
            cc.set_embeddings(0, current_embeddings.detach() if not self.training else current_embeddings)

            # Step 6: Convergence check
            delta = (current_embeddings - prev_embeddings).norm()
            if delta < self.convergence_threshold:
                break

            prev_embeddings = current_embeddings

        return current_embeddings, num_iters
