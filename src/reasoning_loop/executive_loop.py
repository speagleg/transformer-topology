"""Hierarchical executive reasoning loop: GNN controls TAT via control signals.

The GNN Executive acts as the "conscious mind" that analyzes the cell complex,
produces control signals, and orchestrates the TAT. Wave dynamics provide
temporal propagation between GNN analysis and TAT execution.

Loop per iteration:
    1. GNN analyzes CC → (gnn_out, edge_out, control_signal)
    2. Wave dynamics (optional) → wave_out, added as residual to gnn_out
    3. TAT executes with control signals → tat_out
    4. Confidence-weighted integration: conf * tat_out + (1-conf) * gnn_out
    5. Harmonic convergence check
"""

import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.executive import GNNExecutive
from src.gnn_executive.control_head import ControlSignal
from src.tat.transformer import TopologyAwareTransformer
from src.wave.dynamics import WaveDynamics
from src.spectral.decomposition import hodge_decomposition


class ExecutiveReasoningLoop(nn.Module):
    """Hierarchical reasoning loop where the GNN executive controls the TAT.

    Unlike the symmetric ReasoningLoop, here the GNN produces control signals
    (frequency_gate, spatial_focus, confidence_weights, diffusion_time, wave_damping)
    that modulate TAT behavior and wave dynamics. Integration uses per-node
    confidence weighting rather than a simple blend.

    Convergence is tracked via the harmonic component energy of the edge signals
    rather than raw embedding delta.
    """

    def __init__(self, embedding_dim: int, gnn_hidden: int, gnn_spatial_layers: int,
                 gnn_spectral_layers: int, max_freqs: int, tat_layers: int,
                 tat_spatial_heads: int, tat_spectral_heads: int, tat_ff_dim: int,
                 max_iterations: int = 5, convergence_threshold: float = 0.1,
                 use_wave_dynamics: bool = True,
                 use_higher_order: bool = False):
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_wave_dynamics = use_wave_dynamics
        self.embedding_dim = embedding_dim

        self.gnn_executive = GNNExecutive(
            embedding_dim=embedding_dim, hidden_dim=gnn_hidden,
            num_spatial_layers=gnn_spatial_layers,
            num_spectral_layers=gnn_spectral_layers,
            max_freqs=max_freqs,
            produce_control_signals=True,
            use_higher_order=use_higher_order,
        )

        self.tat = TopologyAwareTransformer(
            embedding_dim=embedding_dim, num_layers=tat_layers,
            num_spatial_heads=tat_spatial_heads,
            num_spectral_heads=tat_spectral_heads,
            ff_dim=tat_ff_dim, num_freqs=max_freqs,
        )

        if use_wave_dynamics:
            self.wave_dynamics = WaveDynamics(embedding_dim)

        self.norm = nn.LayerNorm(embedding_dim)

    def _compute_harmonic_energy(self, cc: CellComplex) -> torch.Tensor:
        """Compute harmonic component energy of edge signals for convergence tracking.

        Returns scalar tensor (energy of harmonic component).
        """
        if cc.num_cells(1) == 0:
            return torch.tensor(0.0)

        try:
            edge_embs = cc.get_embeddings(1)
            signal = edge_embs.mean(dim=1)  # (E,)
            _, _, harmonic = hodge_decomposition(cc, signal, dim=1)
            return harmonic.pow(2).sum()
        except (RuntimeError, ValueError):
            return torch.tensor(0.0)

    def forward(self, cc: CellComplex) -> tuple[torch.Tensor, int, dict]:
        """Run the hierarchical executive reasoning loop.

        Each iteration:
          1. GNN Executive analyzes CC → (gnn_out, edge_out, control_signal)
          2. Wave dynamics applied to gnn_out using control.diffusion_time/damping
          3. TAT executes with control.frequency_gate and control.spatial_focus
          4. Confidence-weighted integration of GNN and TAT outputs
          5. Harmonic convergence check

        Args:
            cc: Input CellComplex with 0-cells and 1-cells.

        Returns:
            Tuple of (final_embeddings, num_iterations, diagnostics).
            diagnostics contains:
                - 'harmonic_energies': list of harmonic energy values per iteration
                - 'convergence_deltas': list of |harmonic_t - harmonic_{t-1}| values
                - 'control_signals': list of ControlSignal objects per iteration
        """
        prev_embeddings = cc.get_embeddings(0)
        prev_harmonic_energy = self._compute_harmonic_energy(cc)

        diagnostics = {
            'harmonic_energies': [prev_harmonic_energy.item()],
            'convergence_deltas': [],
            'control_signals': [],
        }
        num_iters = 0

        for i in range(self.max_iterations):
            num_iters = i + 1

            # 1. GNN Executive → embeddings + control signals
            harmonic_energy_input = prev_harmonic_energy.detach()
            gnn_out, edge_out, control = self.gnn_executive.forward_with_control(
                cc, harmonic_energy=harmonic_energy_input,
            )
            diagnostics['control_signals'].append(control)

            # 2. Wave dynamics (optional): temporal propagation
            if self.use_wave_dynamics:
                wave_out = self.wave_dynamics(
                    cc, gnn_out,
                    diffusion_time=control.diffusion_time,
                    wave_damping=control.wave_damping,
                )
                gnn_out = gnn_out + wave_out  # residual addition

            # Update cell complex for TAT (detach for graph safety)
            cc.set_embeddings(0, gnn_out.detach())

            # 3. TAT executes with control signals
            tat_out = self.tat(cc, control_signal=control)

            # 4. Confidence-weighted integration
            confidence = control.confidence_weights.unsqueeze(-1)  # (N, 1)
            integrated = confidence * tat_out + (1 - confidence) * gnn_out
            current_embeddings = self.norm(integrated + prev_embeddings)

            # Write integrated embeddings back to cell complex
            cc.set_embeddings(0, current_embeddings.detach())

            # Update edge embeddings if available
            if edge_out is not None:
                cc.set_embeddings(1, edge_out.detach())

            # 5. Harmonic convergence check
            current_harmonic_energy = self._compute_harmonic_energy(cc)
            diagnostics['harmonic_energies'].append(current_harmonic_energy.item())

            delta = abs(current_harmonic_energy.item() - prev_harmonic_energy.item())
            diagnostics['convergence_deltas'].append(delta)

            if delta < self.convergence_threshold:
                break

            prev_embeddings = current_embeddings
            prev_harmonic_energy = current_harmonic_energy

        return current_embeddings, num_iters, diagnostics
