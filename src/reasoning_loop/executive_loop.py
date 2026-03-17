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
from src.reasoning_loop.text_conditioned_gnn import TextConditionedGNN
from src.reasoning_loop.cross_attention import BidirectionalCrossAttention


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
                 dsm_config: dict | None = None,
                 use_topo_feedback: bool = False,
                 use_embedding_topo_feedback: bool = False,
                 use_metacog: bool = False,
                 num_tasks: int = 19,
                 use_dual_track: bool = False,
                 fusion_weight_init: float = -1.0):
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_wave_dynamics = use_wave_dynamics
        self.use_structural_features = use_structural_features
        self.embedding_dim = embedding_dim
        self.use_dual_track = use_dual_track

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
            use_topo_feedback=use_topo_feedback,
            use_embedding_topo_feedback=use_embedding_topo_feedback,
            use_metacog=use_metacog,
            num_tasks=num_tasks,
            fusion_weight_init=fusion_weight_init,
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

        # Dual-track modules (iteration 2: cross-modal fusion)
        if use_dual_track:
            self.text_gnn = TextConditionedGNN(
                embed_dim=embedding_dim, hidden_dim=embedding_dim, num_layers=2,
            )
            self.cross_attn = BidirectionalCrossAttention(
                embed_dim=embedding_dim, num_heads=4, dropout=0.1,
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

    def _extract_edge_index(self, cc: CellComplex) -> torch.Tensor:
        """Extract PyG-compatible edge_index tensor from CellComplex."""
        return cc.edge_index()

    def forward(self, cc: CellComplex,
                topo_features: torch.Tensor | None = None,
                task_id: int | None = None,
                text_embeddings: torch.Tensor | None = None,
                force_fusion_weight: float | None = None,
                precomputed_pe: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, int, dict]:
        """Run the hierarchical executive reasoning loop.

        Each iteration:
          1. GNN Executive analyzes CC → (gnn_out, edge_out, control_signal)
          2. Wave dynamics applied to gnn_out using control.diffusion_time/damping
          3. TAT executes with control.frequency_gate and control.spatial_focus
          4. Confidence-weighted integration of GNN and TAT outputs
          5. Harmonic convergence check

        Args:
            cc: Input CellComplex with 0-cells and 1-cells.
            topo_features: Optional 6-feature tensor from TopologyObserver
                for active topology feedback to ControlHead.
            task_id: Integer task index for MetaCognitiveController (ignored
                when use_metacog is False).

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

        # ---- Dual-track path: fixed 2-iteration asymmetric loop ----
        if self.use_dual_track:
            return self._forward_dual_track(cc, topo_features, task_id,
                                            text_embeddings, force_fusion_weight,
                                            precomputed_pe)

        # ---- Standard (non-dual-track) path ----
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

            # Build iteration context for metacog controller
            iteration_context = None
            if getattr(self.gnn_executive, 'use_metacog', False):
                prev_conf = 0.5
                prev_delta = 0.0
                if diagnostics['control_signals']:
                    last_cs = diagnostics['control_signals'][-1]
                    if last_cs.uncertainty is not None:
                        prev_conf = last_cs.uncertainty.item()
                if diagnostics['convergence_deltas']:
                    prev_delta = diagnostics['convergence_deltas'][-1]
                iteration_context = torch.tensor(
                    [i / max(self.max_iterations, 1), prev_conf, prev_delta],
                    device=cc.device, dtype=torch.float32,
                )

            gnn_out, edge_out, control = self.gnn_executive.forward_with_control(
                cc, harmonic_energy=harmonic_energy_input,
                topo_features=topo_features,
                task_id=task_id,
                iteration_context=iteration_context,
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
            sem_feat = None
            if self.use_dsm and self.topo_bridge is not None:
                semantic_weight = control.semantic_weight
                node_texts = getattr(cc, 'node_texts', None) or None
                _, semantic_bias, sem_feat, _ = self.topo_bridge(
                    gnn_out, semantic_weight, node_texts=node_texts,
                )

            # Update cell complex for TAT (detach for graph safety)
            cc.set_embeddings(0, gnn_out.detach())

            # 3. TAT executes with control signals + semantic bias
            tat_out = self.tat(cc, control_signal=control,
                               semantic_bias=semantic_bias,
                               semantic_weight=semantic_weight,
                               precomputed_pe=precomputed_pe)

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

            # Soft learned iteration budget (metacog)
            if (control.iteration_budget is not None
                    and control.uncertainty is not None
                    and i > 0):
                budget_ratio = float(i) / max(float(control.iteration_budget.item()), 1.0)
                if budget_ratio > 1.5 and control.uncertainty.item() < 0.3:
                    break

            prev_embeddings = current_embeddings
            prev_harmonic_energy = current_harmonic_energy

        # Expose last iteration's semantic features for contrastive loss
        if sem_feat is not None:
            diagnostics['semantic_features'] = sem_feat

        return current_embeddings, num_iters, diagnostics

    def _forward_dual_track(
        self,
        cc: CellComplex,
        topo_features: torch.Tensor | None,
        task_id: int | None,
        text_embeddings: torch.Tensor | None,
        force_fusion_weight: float | None,
        precomputed_pe: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, dict]:
        """Dual-track forward: iteration 1 (structural) + iteration 2 (cross-modal fusion).

        Iteration 1: standard GNN + wave + TAT (pure structural)
        Iteration 2: TextConditionedGNN + CrossAttention + TAT (cross-modal)
        Final output: (1 - fw) * h_struct + fw * h_fused

        No .detach() between iterations so gradients flow through both tracks.
        """
        prev_embeddings = cc.get_embeddings(0)
        harmonic_energy = self._compute_harmonic_energy(cc).detach()

        diagnostics = {
            'harmonic_energies': [harmonic_energy.item()],
            'convergence_deltas': [],
            'control_signals': [],
        }

        # === Iteration 1: Structural ===
        gnn_out, edge_out, control = self.gnn_executive.forward_with_control(
            cc, harmonic_energy=harmonic_energy,
            topo_features=topo_features,
            task_id=task_id,
        )
        diagnostics['control_signals'].append(control)

        # Wave dynamics (iteration 1 only)
        if self.use_wave_dynamics:
            with torch.amp.autocast('cuda', enabled=False):
                wave_out = self.wave_dynamics(
                    cc, gnn_out.float(),
                    diffusion_time=control.diffusion_time.float(),
                    wave_damping=control.wave_damping.float(),
                    filter_weights=(control.filter_weights.float()
                                    if control.filter_weights is not None else None),
                )
            gnn_out = gnn_out + wave_out.to(gnn_out.dtype)

        # NO .detach() here — gradients flow through to iteration 2
        cc.set_embeddings(0, gnn_out)

        # TAT iteration 1
        h_struct = self.tat(cc, control_signal=control, precomputed_pe=precomputed_pe)

        # Confidence-weighted integration (iteration 1)
        confidence = control.confidence_weights.unsqueeze(-1)
        h_struct = confidence * h_struct + (1 - confidence) * gnn_out
        h_struct = self.norm(h_struct + prev_embeddings)

        # Get fusion_weight
        if force_fusion_weight is not None:
            fw = torch.tensor(force_fusion_weight, device=h_struct.device,
                              dtype=h_struct.dtype)
        else:
            fw = control.fusion_weight
        diagnostics['fusion_weight'] = fw.item() if torch.is_tensor(fw) else fw

        # === Iteration 2: Cross-modal fusion ===
        if text_embeddings is not None and (not torch.is_tensor(fw) or fw.item() > 0):
            edge_index = self._extract_edge_index(cc)

            # TextConditionedGNN: structural + text similarity modulation
            h_text_gnn = self.text_gnn(h_struct, edge_index, text_embeddings)

            # BidirectionalCrossAttention: structural ↔ text
            h_cross, h_text_enriched = self.cross_attn(h_text_gnn, text_embeddings)
            # NOTE: h_text_enriched must not be detached — gradients flow through
            # this tensor to the t→s cross-attention parameters.
            diagnostics['text_embeddings'] = h_text_enriched

            # Update CC for TAT iteration 2 (no detach for gradient flow)
            cc.set_embeddings(0, h_cross)

            # TAT iteration 2
            h_fused = self.tat(cc, control_signal=control, precomputed_pe=precomputed_pe)

            # Blend: (1-fw)*structural + fw*fused
            h_out = (1 - fw) * h_struct + fw * h_fused
        else:
            h_out = h_struct
            diagnostics['fusion_weight'] = 0.0

        # Write final embeddings back
        cc.set_embeddings(0, h_out.detach())

        # Update edge embeddings with residual blending
        if edge_out is not None:
            old_edge = cc.get_embeddings(1)
            r = self.edge_residual_ratio
            blended_edge = (1 - r) * edge_out + r * old_edge
            cc.set_embeddings(1, blended_edge.detach())

        # Harmonic energy after both iterations
        final_harmonic = self._compute_harmonic_energy(cc)
        diagnostics['harmonic_energies'].append(final_harmonic.item())
        delta = abs(final_harmonic.item() - harmonic_energy.item())
        diagnostics['convergence_deltas'].append(delta)

        return h_out, 2, diagnostics

    def forward_dual_track_batched(
        self,
        ccs: list[CellComplex],
        topo_features_list: list[torch.Tensor] | None = None,
        task_id: int | None = None,
        text_embeddings_list: list[torch.Tensor | None] | None = None,
        force_fusion_weight: float | None = None,
        precomputed_pe_list: list[torch.Tensor | None] | None = None,
    ) -> list[tuple[torch.Tensor, int, dict]]:
        """Batched dual-track forward: process multiple CellComplexes.

        Per-graph operations (eigendecomp, harmonic energy) run sequentially.
        Provides the batched interface for the v10 training loop with reduced
        Python overhead vs calling forward() in a loop.

        Args:
            ccs: List of B CellComplexes (already on device).
            topo_features_list: Optional list of topo feature tensors per graph.
            task_id: Integer task index for ControlHead.
            text_embeddings_list: Optional list of per-graph text embeddings.
            force_fusion_weight: Optional forced fusion weight (overrides learned).
            precomputed_pe_list: Optional list of precomputed TopologicalPE features.

        Returns:
            List of (embeddings, num_iters, diagnostics) per graph.
        """
        B = len(ccs)
        enriched_text: list[torch.Tensor | None] = [None] * B
        device = ccs[0].device

        # Structural features (per-graph, cheap)
        if self.use_structural_features:
            for cc in ccs:
                struct_feat = self.structural_encoder(cc)
                cc.set_embeddings(0, (cc.get_embeddings(0) + struct_feat).detach())

        # Pre-iteration: collect per-graph data
        prev_embeddings = [cc.get_embeddings(0) for cc in ccs]
        harmonic_energies = [self._compute_harmonic_energy(cc).detach() for cc in ccs]

        all_diagnostics = [{
            'harmonic_energies': [he.item()],
            'convergence_deltas': [],
            'control_signals': [],
        } for he in harmonic_energies]

        # === Iteration 1: Structural (per-graph GNN + wave + TAT) ===
        gnn_outputs = []
        edge_outputs = []
        controls = []
        for g in range(B):
            topo_feat = topo_features_list[g] if topo_features_list else None
            gnn_out, edge_out, control = self.gnn_executive.forward_with_control(
                ccs[g], harmonic_energy=harmonic_energies[g],
                topo_features=topo_feat, task_id=task_id,
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

            ccs[g].set_embeddings(0, gnn_out)
            gnn_outputs.append(gnn_out)
            edge_outputs.append(edge_out)
            controls.append(control)

        # TAT iteration 1 (per-graph — eigendecomp is graph-specific)
        h_structs = []
        for g in range(B):
            pe = precomputed_pe_list[g] if precomputed_pe_list else None
            h_s = self.tat(ccs[g], control_signal=controls[g], precomputed_pe=pe)
            confidence = controls[g].confidence_weights.unsqueeze(-1)
            h_s = confidence * h_s + (1 - confidence) * gnn_outputs[g]
            h_s = self.norm(h_s + prev_embeddings[g])
            h_structs.append(h_s)

        # === Iteration 2: Cross-modal fusion (per-graph) ===
        h_outs = []
        for g in range(B):
            control = controls[g]
            if force_fusion_weight is not None:
                fw = torch.tensor(force_fusion_weight, device=device,
                                  dtype=h_structs[g].dtype)
            else:
                fw = control.fusion_weight
            all_diagnostics[g]['fusion_weight'] = fw.item() if torch.is_tensor(fw) else fw

            text_emb = text_embeddings_list[g] if text_embeddings_list else None
            if text_emb is not None and (not torch.is_tensor(fw) or fw.item() > 0):
                edge_index = self._extract_edge_index(ccs[g])
                h_text_gnn = self.text_gnn(h_structs[g], edge_index, text_emb)
                h_cross, h_text_enriched = self.cross_attn(h_text_gnn, text_emb)
                ccs[g].set_embeddings(0, h_cross)
                enriched_text[g] = h_text_enriched
                pe = precomputed_pe_list[g] if precomputed_pe_list else None
                h_fused = self.tat(ccs[g], control_signal=control, precomputed_pe=pe)
                h_out = (1 - fw) * h_structs[g] + fw * h_fused
            else:
                h_out = h_structs[g]
                all_diagnostics[g]['fusion_weight'] = 0.0

            # Write final embeddings back
            ccs[g].set_embeddings(0, h_out.detach())
            if edge_outputs[g] is not None:
                old_edge = ccs[g].get_embeddings(1)
                r = self.edge_residual_ratio
                blended = (1 - r) * edge_outputs[g] + r * old_edge
                ccs[g].set_embeddings(1, blended.detach())

            # Final harmonic energy
            final_he = self._compute_harmonic_energy(ccs[g])
            all_diagnostics[g]['harmonic_energies'].append(final_he.item())
            delta = abs(final_he.item() - harmonic_energies[g].item())
            all_diagnostics[g]['convergence_deltas'].append(delta)

            h_outs.append(h_out)

        results = []
        for g in range(B):
            # NOTE: enriched_text[g] must not be detached — gradients flow through
            # this tensor to the t→s cross-attention parameters.
            all_diagnostics[g]['text_embeddings'] = enriched_text[g]
            results.append((h_outs[g], 2, all_diagnostics[g]))
        return results

    def forward_batched(
        self, ccs: list[CellComplex],
        topo_features_list: list[torch.Tensor] | None = None,
        task_id: int | None = None,
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
            topo_features_list: Optional list of topo feature tensors per graph.
            task_id: Integer task index for MetaCognitiveController (ignored
                when use_metacog is False).

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
                topo_feat = topo_features_list[g] if topo_features_list else None

                # Build iteration context for metacog controller
                iteration_context = None
                if getattr(self.gnn_executive, 'use_metacog', False):
                    prev_conf = 0.5
                    prev_delta = 0.0
                    if all_diagnostics[g]['control_signals']:
                        last_cs = all_diagnostics[g]['control_signals'][-1]
                        if last_cs.uncertainty is not None:
                            prev_conf = last_cs.uncertainty.item()
                    if all_diagnostics[g]['convergence_deltas']:
                        prev_delta = all_diagnostics[g]['convergence_deltas'][-1]
                    iteration_context = torch.tensor(
                        [iteration / max(self.max_iterations, 1), prev_conf, prev_delta],
                        device=ccs[g].device, dtype=torch.float32,
                    )

                gnn_out, edge_out, control = self.gnn_executive.forward_with_control(
                    ccs[g], harmonic_energy=harmonic_energy,
                    topo_features=topo_feat,
                    task_id=task_id,
                    iteration_context=iteration_context,
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
            sem_feats = [None] * B
            if self.use_dsm and self.topo_bridge is not None:
                node_embs_list = [go[0] for go in gnn_outputs]
                semantic_weights = [go[2].semantic_weight for go in gnn_outputs]
                dsm_results = self.topo_bridge.forward_batched(
                    node_embs_list, semantic_weights,
                )
                for g, (_, sb, sf, _) in enumerate(dsm_results):
                    semantic_biases[g] = sb
                    sem_feats[g] = sf

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

        # Build return values — attach last iteration's semantic features
        results = []
        for g in range(B):
            if sem_feats[g] is not None:
                all_diagnostics[g]['semantic_features'] = sem_feats[g]
            results.append((current_embeddings[g], self.max_iterations, all_diagnostics[g]))
        return results
