"""Phase 3 model: Temporal/causal reasoning with ExecutiveReasoningLoop and wave dynamics."""

import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop
from src.spectral.decomposition import hodge_decomposition
from src.spectral.persistence import vectorize_persistence, compute_persistence_diagram

# persistence: 2 dimensions (H0, H1) * 16 features each = 32
PERSISTENCE_FEATURES = 32


class TemporalReasoningModel(nn.Module):
    """Model for Phase 3 temporal/causal reasoning benchmarks.

    Extends the Phase 2 model with:
    - ExecutiveReasoningLoop (hierarchical GNN->TAT with control signals)
    - Higher-order GNN message passing over 2-cells
    - Wave dynamics for temporal propagation
    - Wave energy features derived from control signal diagnostics
    - Persistence homology features (H0 + H1 diagrams)
    - Deeper classifier head with LayerNorm/GELU/Dropout
    """

    def __init__(self, embedding_dim: int, gnn_hidden: int, gnn_spatial_layers: int,
                 gnn_spectral_layers: int, max_freqs: int, tat_layers: int,
                 tat_spatial_heads: int, tat_spectral_heads: int, tat_ff_dim: int,
                 max_classes: int, max_iterations: int = 7,
                 convergence_threshold: float = 0.1,
                 use_wave_dynamics: bool = True,
                 use_higher_order: bool = True,
                 use_topological_pe: bool = False,
                 use_structural_features: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.executive_loop = ExecutiveReasoningLoop(
            embedding_dim=embedding_dim, gnn_hidden=gnn_hidden,
            gnn_spatial_layers=gnn_spatial_layers,
            gnn_spectral_layers=gnn_spectral_layers,
            max_freqs=max_freqs, tat_layers=tat_layers,
            tat_spatial_heads=tat_spatial_heads,
            tat_spectral_heads=tat_spectral_heads,
            tat_ff_dim=tat_ff_dim, max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            use_wave_dynamics=use_wave_dynamics,
            use_higher_order=use_higher_order,
            use_topological_pe=use_topological_pe,
            use_structural_features=use_structural_features,
        )

        # Classifier input: query_emb + target_emb + diff_emb + hodge(3) + wave_energy(1) + persistence(32)
        classifier_input_dim = 3 * embedding_dim + 3 + 1 + PERSISTENCE_FEATURES
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 4 * embedding_dim),
            nn.LayerNorm(4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.LayerNorm(2 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, max_classes),
        )

    def _compute_hodge_features(self, cc: CellComplex) -> torch.Tensor:
        """Compute Hodge decomposition summary features.

        Returns 3 scalar features: mean of gradient, curl, and harmonic norms
        over all edge signals.
        """
        if cc.num_cells(1) == 0 or cc.num_cells(0) == 0:
            return torch.zeros(3)

        try:
            edge_embs = cc.get_embeddings(1)
            signal = edge_embs.mean(dim=1)
            gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
            return torch.tensor([
                gradient.abs().mean().item(),
                curl.abs().mean().item(),
                harmonic.abs().mean().item(),
            ])
        except (RuntimeError, ValueError):
            return torch.zeros(3)

    def _compute_wave_energy(self, diagnostics: dict) -> torch.Tensor:
        """Compute wave energy from the last control signal's diffusion_time and wave_damping.

        Wave energy = diffusion_time^2 + wave_damping^2 from the last iteration's
        control signal. Returns a 1-element tensor.
        """
        control_signals = diagnostics.get('control_signals', [])
        if not control_signals:
            return torch.zeros(1)

        last_control = control_signals[-1]
        dt = last_control.diffusion_time
        wd = last_control.wave_damping
        energy = dt.pow(2) + wd.pow(2)
        return energy.unsqueeze(0)

    def _compute_persistence_features(self, cc: CellComplex) -> torch.Tensor:
        """Compute persistence homology features (H0 and H1 diagrams).

        Returns a fixed-size vector of topological invariants capturing
        connected components (H0) and cycles (H1).
        """
        try:
            diagrams = compute_persistence_diagram(cc, max_dimension=1)
            return vectorize_persistence(diagrams, num_features=16)
        except (RuntimeError, ValueError):
            return torch.zeros(PERSISTENCE_FEATURES)

    def forward(self, cc: CellComplex, query_node: int, target_node: int) -> torch.Tensor:
        output, num_iters, diagnostics = self.executive_loop(cc)
        query_emb = output[query_node]
        target_emb = output[target_node]
        diff_emb = query_emb - target_emb

        dev = query_emb.device
        hodge_features = self._compute_hodge_features(cc).to(dev)
        wave_energy = self._compute_wave_energy(diagnostics).to(dev)
        persistence_features = self._compute_persistence_features(cc).to(dev)

        combined = torch.cat([query_emb, target_emb, diff_emb,
                              hodge_features, wave_energy, persistence_features])
        logits = self.classifier(combined)
        return logits
