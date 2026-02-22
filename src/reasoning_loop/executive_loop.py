"""Hierarchical executive reasoning loop: GNN controls TAT via control signals.

The GNN Executive acts as the "conscious mind" that analyzes the cell complex,
produces control signals, and orchestrates the TAT. Wave dynamics provide
temporal propagation between GNN analysis and TAT execution.

Loop per iteration:
    1. GNN analyzes CC → (gnn_out, edge_out, control_signal)
    2. Wave dynamics (optional) → wave_out, added as residual to gnn_out
    2.5. DSM (optional) → semantic_bias for TAT attention
    3. TAT executes with control signals + semantic bias → tat_out
    4. Confidence-weighted integration: conf * tat_out + (1-conf) * gnn_out
    5. Harmonic convergence check
"""

import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.cell_complex.structural_features import StructuralFeatureEncoder
from src.gnn_executive.executive import GNNExecutive
from src.gnn_executive.control_head import ControlSignal
from src.tat.transformer import TopologyAwareTransformer
from src.wave.dynamics import WaveDynamics, SheafWaveDynamics, MultiFilterDynamics
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
                 use_higher_order: bool = False,
                 use_topological_pe: bool = False,
                 use_structural_features: bool = False,
                 wave_filter_type: str = 'heat',
                 wave_laplacian_dim: int = 0,
                 wave_strength_gate: bool = False,
                 wave_use_neural_ode: bool = True,
                 wave_filter_kwargs: dict | None = None,
                 wave_mode: str = 'spectral',
                 use_dsm: bool = False,
                 dsm_config: dict | None = None):
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_wave_dynamics = use_wave_dynamics
        self.use_structural_features = use_structural_features
        self.embedding_dim = embedding_dim

        if use_structural_features:
            self.structural_encoder = StructuralFeatureEncoder(embedding_dim)

        # Create wave dynamics first to determine num_filters for ControlHead
        num_filters = 0
        if use_wave_dynamics:
            _wfk = wave_filter_kwargs or {}
            if wave_mode == 'sheaf':
                self.wave_dynamics = SheafWaveDynamics(
                    embedding_dim,
                    use_wave_strength_gate=wave_strength_gate,
                )
            elif wave_mode == 'ensemble':
                filter_types = _wfk.get('filter_types', ['chebyshev', 'wave_cosine', 'heat'])
                include_identity = _wfk.get('include_identity', True)
                include_sheaf = _wfk.get('include_sheaf', False)
                self.wave_dynamics = MultiFilterDynamics(
                    embedding_dim,
                    filter_types=filter_types,
                    laplacian_dim=wave_laplacian_dim,
                    include_identity=include_identity,
                    include_sheaf=include_sheaf,
                    use_wave_strength_gate=wave_strength_gate,
                    use_neural_ode=wave_use_neural_ode,
                )
                num_filters = self.wave_dynamics.num_filters
            else:
                self.wave_dynamics = WaveDynamics(
                    embedding_dim,
                    filter_type=wave_filter_type,
                    laplacian_dim=wave_laplacian_dim,
                    use_wave_strength_gate=wave_strength_gate,
                    use_neural_ode=wave_use_neural_ode,
                    **_wfk,
                )

        self.gnn_executive = GNNExecutive(
            embedding_dim=embedding_dim, hidden_dim=gnn_hidden,
            num_spatial_layers=gnn_spatial_layers,
            num_spectral_layers=gnn_spectral_layers,
            max_freqs=max_freqs,
            produce_control_signals=True,
            use_higher_order=use_higher_order,
            num_filters=num_filters,
        )

        self.tat = TopologyAwareTransformer(
            embedding_dim=embedding_dim, num_layers=tat_layers,
            num_spatial_heads=tat_spatial_heads,
            num_spectral_heads=tat_spectral_heads,
            ff_dim=tat_ff_dim, num_freqs=max_freqs,
            use_topological_pe=use_topological_pe,
        )

        self.norm = nn.LayerNorm(embedding_dim)

        # Edge residual ratio: fraction of old edge embeddings to keep.
        # Prevents total overwrite of initial edge signal (which destroys curl
        # content that the GNN's B1^T messages can't reconstruct from nodes).
        # Fixed at 0.5 because the blend happens inside detach() so a learned
        # gate would get zero gradients.
        self.edge_residual_ratio = 0.5

        # DSM integration (optional)
        self.use_dsm = use_dsm
        self.topo_bridge = None
        if use_dsm:
            from src.llm.dsm_backend import DSMBackend
            from src.llm.topo_bridge import TopoBridge
            dc = dsm_config or {}
            dsm_dim = dc.get('dsm_dim', 1024)
            backend = DSMBackend(
                dsm_dim=dsm_dim,
                num_heads=dc.get('num_heads', 16),
                ff_dim=dc.get('ff_dim', 4096),
                num_layers=dc.get('num_layers', 16),
                cross_attn_layer=dc.get('cross_attn_layer', 4),
            )
            self.topo_bridge = TopoBridge(
                backend=backend,
                topo_dim=embedding_dim,
                llm_dim=dsm_dim,
                num_prefix=dc.get('num_prefix', 8),
            )

    def _compute_harmonic_energy(self, cc: CellComplex) -> torch.Tensor:
        """Compute harmonic component energy of edge signals for convergence tracking.

        Returns scalar tensor (energy normalized by number of edges so that
        convergence thresholds are size-invariant).
        """
        num_edges = cc.num_cells(1)
        if num_edges == 0:
            return torch.tensor(0.0, device=cc.device)

        try:
            edge_embs = cc.get_embeddings(1)
            signal = edge_embs[:, 0]  # dim 0 carries the task signal
            _, _, harmonic = hodge_decomposition(cc, signal, dim=1)
            return harmonic.pow(2).sum() / max(1, num_edges)
        except (RuntimeError, ValueError):
            return torch.tensor(0.0, device=cc.device)

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
        # Add structural features to initial node embeddings
        if self.use_structural_features:
            struct_feat = self.structural_encoder(cc)
            cc.set_embeddings(0, (cc.get_embeddings(0) + struct_feat).detach())

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
            #    Force fp32 — spectral decomposition + neural ODE overflow in bf16.
            if self.use_wave_dynamics:
                with torch.amp.autocast('cuda', enabled=False):
                    wave_out = self.wave_dynamics(
                        cc, gnn_out.float(),
                        diffusion_time=control.diffusion_time.float(),
                        wave_damping=control.wave_damping.float(),
                        filter_weights=control.filter_weights.float() if control.filter_weights is not None else None,
                    )
                gnn_out = gnn_out + wave_out.to(gnn_out.dtype)  # residual addition

            # 2.5. DSM: encode → DSM forward → decode → semantic_bias
            semantic_bias = None
            semantic_weight = None
            if self.use_dsm and self.topo_bridge is not None:
                semantic_weight = control.semantic_weight
                _, semantic_bias = self.topo_bridge(gnn_out, semantic_weight)

            # Update cell complex for TAT (detach for graph safety)
            cc.set_embeddings(0, gnn_out.detach())

            # 3. TAT executes with control signals + semantic bias
            tat_out = self.tat(cc, control_signal=control,
                               semantic_bias=semantic_bias,
                               semantic_weight=semantic_weight)

            # 4. Confidence-weighted integration
            confidence = control.confidence_weights.unsqueeze(-1)  # (N, 1)
            integrated = confidence * tat_out + (1 - confidence) * gnn_out
            current_embeddings = self.norm(integrated + prev_embeddings)

            # Write integrated embeddings back to cell complex
            cc.set_embeddings(0, current_embeddings.detach())

            # Update edge embeddings with residual blending.
            # Pure replacement destroys curl signal because the GNN produces
            # edge_out primarily via B1^T @ node_features (gradient subspace).
            # Blending preserves curl content from previous edge embeddings.
            if edge_out is not None:
                old_edge = cc.get_embeddings(1)
                r = self.edge_residual_ratio
                blended_edge = (1 - r) * edge_out + r * old_edge
                cc.set_embeddings(1, blended_edge.detach())

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

    def forward_batched(
        self, ccs: list[CellComplex],
    ) -> list[tuple[torch.Tensor, int, dict]]:
        """Batched executive loop: process multiple graphs with batched DSM.

        Per iteration:
          1. Per-graph GNN + wave (cheap, sequential)
          2. Batched DSM call if enabled (expensive, NOW BATCHED)
          3. Per-graph TAT + integration (cheap, sequential)

        Uses fixed max_iterations (no per-graph convergence stopping) to keep
        graphs synchronized for batched DSM calls.

        Args:
            ccs: List of B CellComplexes.

        Returns:
            List of (embeddings, num_iters, diagnostics) per graph.
        """
        B = len(ccs)

        # Initialize structural features per graph
        if self.use_structural_features:
            for cc in ccs:
                struct_feat = self.structural_encoder(cc)
                cc.set_embeddings(0, (cc.get_embeddings(0) + struct_feat).detach())

        prev_embeddings = [cc.get_embeddings(0) for cc in ccs]
        all_diagnostics = [
            {'harmonic_energies': [], 'convergence_deltas': [], 'control_signals': []}
            for _ in range(B)
        ]
        current_embeddings = [None] * B

        for iteration in range(self.max_iterations):
            # Phase 1: Per-graph GNN + wave (cheap)
            gnn_outputs = []
            for g in range(B):
                harmonic_energy = self._compute_harmonic_energy(ccs[g]).detach()
                gnn_out, edge_out, control = self.gnn_executive.forward_with_control(
                    ccs[g], harmonic_energy=harmonic_energy,
                )
                all_diagnostics[g]['control_signals'].append(control)

                if self.use_wave_dynamics:
                    with torch.amp.autocast('cuda', enabled=False):
                        wave_out = self.wave_dynamics(
                            ccs[g], gnn_out.float(),
                            diffusion_time=control.diffusion_time.float(),
                            wave_damping=control.wave_damping.float(),
                            filter_weights=(control.filter_weights.float()
                                            if control.filter_weights is not None else None),
                        )
                    gnn_out = gnn_out + wave_out.to(gnn_out.dtype)

                gnn_outputs.append((gnn_out, edge_out, control))

            # Phase 2: Batched DSM call (expensive, NOW BATCHED)
            semantic_biases = [None] * B
            if self.use_dsm and self.topo_bridge is not None:
                node_embs_list = [go[0] for go in gnn_outputs]
                semantic_weights = [go[2].semantic_weight for go in gnn_outputs]
                dsm_results = self.topo_bridge.forward_batched(
                    node_embs_list, semantic_weights,
                )
                for g, (_, sb) in enumerate(dsm_results):
                    semantic_biases[g] = sb

            # Phase 3: Per-graph TAT + integration (cheap)
            for g in range(B):
                gnn_out, edge_out, control = gnn_outputs[g]
                ccs[g].set_embeddings(0, gnn_out.detach())

                tat_out = self.tat(
                    ccs[g], control_signal=control,
                    semantic_bias=semantic_biases[g],
                    semantic_weight=control.semantic_weight if self.use_dsm else None,
                )

                confidence = control.confidence_weights.unsqueeze(-1)
                integrated = confidence * tat_out + (1 - confidence) * gnn_out
                current_embeddings[g] = self.norm(integrated + prev_embeddings[g])

                ccs[g].set_embeddings(0, current_embeddings[g].detach())

                if edge_out is not None:
                    old_edge = ccs[g].get_embeddings(1)
                    r = self.edge_residual_ratio
                    blended = (1 - r) * edge_out + r * old_edge
                    ccs[g].set_embeddings(1, blended.detach())

            prev_embeddings = [ce.clone() for ce in current_embeddings]

        # Build return values
        results = []
        for g in range(B):
            results.append((current_embeddings[g], self.max_iterations, all_diagnostics[g]))
        return results
