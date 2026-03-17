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
                 num_filters: int = 0,
                 use_topo_feedback: bool = False,
                 use_embedding_topo_feedback: bool = False,
                 use_metacog: bool = False,
                 num_tasks: int = 19,
                 fusion_weight_init: float = -1.0):
        super().__init__()
        self.use_higher_order = use_higher_order
        self.produce_control_signals = produce_control_signals
        self.use_metacog = use_metacog
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
            if use_metacog:
                from src.gnn_executive.metacognitive_controller import MetaCognitiveController
                self.control_head = MetaCognitiveController(
                    embedding_dim=embedding_dim,
                    num_freqs=max_freqs,
                    num_filters=num_filters,
                    num_tasks=num_tasks,
                    use_topo_feedback=use_topo_feedback,
                    use_embedding_topo_feedback=use_embedding_topo_feedback,
                )
            else:
                self.control_head = ControlHead(
                    embedding_dim=embedding_dim, num_freqs=max_freqs,
                    num_filters=num_filters,
                    use_topo_feedback=use_topo_feedback,
                    use_embedding_topo_feedback=use_embedding_topo_feedback,
                    fusion_weight_init=fusion_weight_init,
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
        topo_features: torch.Tensor | None = None,
        task_id: int | None = None,
        iteration_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, ControlSignal]:
        """Run dual-path GNN and produce control signals for the TAT.

        Calls the standard forward pass, then feeds the fused node embeddings
        through the ControlHead (or MetaCognitiveController) to produce a
        ControlSignal.

        Args:
            cc: Cell complex with 0-cell and 1-cell embeddings.
            harmonic_energy: Optional scalar tensor with harmonic component energy.
            topo_features: Optional topological feedback features.
            task_id: Integer task index for MetaCognitiveController (ignored when
                use_metacog is False).
            iteration_context: (3,) tensor [iter_ratio, prev_confidence, prev_delta]
                for MetaCognitiveController (ignored when use_metacog is False).

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
        if self.use_metacog:
            control = self.control_head(
                fused, task_id=task_id,
                harmonic_energy=harmonic_energy,
                topo_features=topo_features,
                iteration_context=iteration_context,
            )
        else:
            control = self.control_head(
                fused, harmonic_energy=harmonic_energy,
                topo_features=topo_features,
            )
        return fused, edge_out, control
