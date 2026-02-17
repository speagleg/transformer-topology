"""Wave and diffusion dynamics on cell complexes.

Provides heat diffusion (analytic spectral solution) and wave propagation
(neural ODE) modules that are parameterized by control signals from the
GNN executive.  WaveDynamics blends both with a learned per-node gate.

The spectral filter used for the analytic path is configurable via
``filter_type`` (default ``'heat'``). A wave strength gate can be
enabled to let the model learn to bypass wave dynamics per-task.
"""

import torch
import torch.nn as nn

from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition
from src.spectral.laplacian import hodge_laplacian_0
from src.spectral.sheaf_diffusion import SheafDiffusion
from src.wave.spectral_filters import create_spectral_filter, SpectralFilter

try:
    from torchdiffeq import odeint

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


# ---------------------------------------------------------------------------
# Heat diffusion (no learnable parameters)
# ---------------------------------------------------------------------------

class HeatDiffusion(nn.Module):
    """Analytic heat equation on 0-cells via eigendecomposition.

    f(t) = U @ diag(exp(-lambda * t)) @ U^T @ f(0)

    Small t preserves locality; large t smooths toward the global mean.
    """

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        """Diffuse *signal* for duration *diffusion_time*.

        Args:
            cc: Cell complex whose L0 Hodge Laplacian is used.
            signal: Node signal, shape (N,) or (N, D).
            diffusion_time: Positive scalar tensor.

        Returns:
            Diffused signal, same shape as *signal*.
        """
        squeeze = signal.dim() == 1
        if squeeze:
            signal = signal.unsqueeze(-1)  # (N, 1)

        try:
            eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)
        except Exception:
            # If decomposition fails, return signal unchanged.
            return signal.squeeze(-1) if squeeze else signal

        # eigenvalues: (N,), eigenvectors: (N, N)
        # Heat kernel in eigenbasis: exp(-lambda_i * t)
        heat_coeffs = torch.exp(-eigenvalues * diffusion_time)  # (N,)

        # Project signal into eigenbasis, scale, project back.
        # (N, N)^T @ (N, D) -> (N, D)
        projected = eigenvectors.T @ signal
        scaled = heat_coeffs.unsqueeze(-1) * projected  # (N, D)
        result = eigenvectors @ scaled  # (N, D)

        return result.squeeze(-1) if squeeze else result


# ---------------------------------------------------------------------------
# Wave propagation (neural ODE)
# ---------------------------------------------------------------------------

class _WaveODEFunc(nn.Module):
    """ODE right-hand side: df/dt = -L @ f - damping * f + correction(f, t)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.correction = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        # Will be set before each integration
        self.L: torch.Tensor | None = None
        self.damping: torch.Tensor | None = None

    def forward(self, t: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Compute df/dt.

        Args:
            t: Current time (scalar, unused by linear terms but available to correction).
            f: State tensor of shape (N, D).

        Returns:
            df/dt of shape (N, D).
        """
        assert self.L is not None and self.damping is not None
        laplacian_term = -self.L @ f             # -L @ f
        damping_term = -self.damping * f          # -damping * f
        correction_term = self.correction(f)      # learned residual
        return laplacian_term + damping_term + correction_term


def _euler_fallback(
    func: _WaveODEFunc,
    f0: torch.Tensor,
    t_span: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    """Manual Euler integration when torchdiffeq is unavailable."""
    t_start, t_end = t_span[0], t_span[-1]
    dt = (t_end - t_start) / steps
    f = f0
    t = t_start
    for _ in range(steps):
        f = f + dt * func(t, f)
        t = t + dt
    return f


class WavePropagation(nn.Module):
    """Damped wave propagation with a learned correction, integrated via neural ODE.

    ODE: df/dt = -L0 @ f - damping * f + correction(f, t)
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.ode_func = _WaveODEFunc(embedding_dim)

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        diffusion_time: torch.Tensor,
        wave_damping: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate *signal* through damped wave dynamics.

        Args:
            cc: Cell complex.
            signal: Node signal of shape (N, D).
            diffusion_time: Integration horizon (positive scalar).
            wave_damping: Damping coefficient (positive scalar).

        Returns:
            Propagated signal of shape (N, D).
        """
        L = hodge_laplacian_0(cc)  # (N, N)
        self.ode_func.L = L
        self.ode_func.damping = wave_damping

        # Guard diffusion_time: NaN → fallback, zero → clamp.
        # (t_span must be strictly increasing, so dt > 0 and finite is required).
        if torch.isnan(diffusion_time) or torch.isinf(diffusion_time):
            diffusion_time = torch.tensor(0.1, device=signal.device)
        diffusion_time = diffusion_time.clamp(min=1e-4)

        t_span = torch.stack([
            torch.zeros_like(diffusion_time),
            diffusion_time,
        ])  # (2,)

        if HAS_TORCHDIFFEQ:
            # odeint returns (T, N, D); we want the final state.
            trajectory = odeint(
                self.ode_func,
                signal,
                t_span,
                method="euler",
                options={"step_size": diffusion_time.detach() / 5},
            )
            return trajectory[-1]  # (N, D)
        else:
            return _euler_fallback(self.ode_func, signal, t_span, steps=5)


# ---------------------------------------------------------------------------
# Combined dynamics with learned gate
# ---------------------------------------------------------------------------

class WaveDynamics(nn.Module):
    """Blended spectral filter + wave propagation with a learned per-node gate.

    The spectral filter (default: heat diffusion) is configurable via
    ``filter_type``. A wave strength gate can be enabled to let the model
    learn to zero-out wave contributions on tasks where they hurt.

    gate = sigmoid(Linear(concat(filter_out, wave_out)))
    output = gate * filter_out + (1 - gate) * wave_out
    """

    def __init__(
        self,
        embedding_dim: int,
        filter_type: str = 'heat',
        laplacian_dim: int = 0,
        use_wave_strength_gate: bool = False,
        use_neural_ode: bool = True,
        **filter_kwargs,
    ):
        super().__init__()
        self.use_wave_strength_gate = use_wave_strength_gate
        self.use_neural_ode = use_neural_ode

        self.spectral_filter = create_spectral_filter(
            filter_type=filter_type, dim=laplacian_dim, **filter_kwargs,
        )
        if use_neural_ode:
            self.wave = WavePropagation(embedding_dim)
            self.gate = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.Sigmoid(),
            )

        if use_wave_strength_gate:
            self.wave_strength = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        diffusion_time: torch.Tensor,
        wave_damping: torch.Tensor,
        filter_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run spectral filter and wave ODE in parallel, blend with learned gate.

        Args:
            cc: Cell complex.
            signal: Node signal of shape (N, D).
            diffusion_time: Positive scalar (controls both filter and wave).
            wave_damping: Positive scalar (wave only).
            filter_weights: Unused (accepted for API compatibility with MultiFilterDynamics).

        Returns:
            Blended signal of shape (N, D).
        """
        # Sanitize diffusion_time (NaN/Inf from ControlHead during early training).
        if torch.isnan(diffusion_time) or torch.isinf(diffusion_time):
            diffusion_time = torch.tensor(0.1, device=signal.device)
        diffusion_time = diffusion_time.clamp(min=1e-4)

        filter_out = self.spectral_filter(
            cc, signal, diffusion_time=diffusion_time,
        )  # (N, D)

        if self.use_neural_ode:
            wave_out = self.wave(
                cc, signal, diffusion_time, wave_damping,
            )  # (N, D)
            gate_input = torch.cat([filter_out, wave_out], dim=-1)  # (N, 2D)
            g = self.gate(gate_input)  # (N, D)
            blended = g * filter_out + (1 - g) * wave_out
        else:
            blended = filter_out

        if self.use_wave_strength_gate:
            strength = torch.sigmoid(self.wave_strength)
            return strength * blended + (1 - strength) * signal
        return blended


# ---------------------------------------------------------------------------
# Multi-filter ensemble dynamics
# ---------------------------------------------------------------------------


class MultiFilterDynamics(nn.Module):
    """Ensemble of spectral filters with executive-gated blending.

    Runs multiple spectral filters in parallel and blends their outputs
    using ``filter_weights`` from the GNN executive's ControlSignal.
    Includes an optional identity (nowave) path so the executive can
    learn to bypass spectral processing for structural tasks.

    When ``filter_weights`` is None (e.g. non-ensemble mode), outputs
    are blended with equal weights.
    """

    def __init__(
        self,
        embedding_dim: int,
        filter_types: tuple[str, ...] | list[str] = ('chebyshev', 'wave_cosine', 'heat'),
        laplacian_dim: int = 0,
        include_identity: bool = True,
        include_sheaf: bool = False,
        use_wave_strength_gate: bool = False,
        use_neural_ode: bool = False,
        **filter_kwargs,
    ):
        super().__init__()
        self.include_identity = include_identity
        self.include_sheaf = include_sheaf
        self.use_wave_strength_gate = use_wave_strength_gate
        self.use_neural_ode = use_neural_ode

        self.filters = nn.ModuleList([
            create_spectral_filter(ft, dim=laplacian_dim, **filter_kwargs)
            for ft in filter_types
        ])

        # Sheaf diffusion as a special ensemble path (learnable restriction maps,
        # different mechanism from SpectralFilter but same forward interface).
        if include_sheaf:
            self.sheaf = SheafWaveDynamics(
                embedding_dim,
                use_wave_strength_gate=False,  # ensemble handles gating
            )

        self.num_paths = (
            len(filter_types)
            + (1 if include_identity else 0)
            + (1 if include_sheaf else 0)
        )

        if use_neural_ode:
            self.wave = WavePropagation(embedding_dim)
            self.ode_gate = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.Sigmoid(),
            )

        if use_wave_strength_gate:
            self.wave_strength = nn.Parameter(torch.tensor(1.0))

    @property
    def num_filters(self) -> int:
        """Number of filter paths (for ControlHead sizing)."""
        return self.num_paths

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        diffusion_time: torch.Tensor,
        wave_damping: torch.Tensor,
        filter_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all filters and blend outputs with executive-provided weights.

        Args:
            cc: Cell complex.
            signal: Node signal of shape (N, D).
            diffusion_time: Positive scalar.
            wave_damping: Positive scalar (used by neural ODE if enabled).
            filter_weights: Optional (num_paths,) softmax weights from ControlHead.
                If None, equal weighting is used.

        Returns:
            Blended signal of shape (N, D).
        """
        if torch.isnan(diffusion_time) or torch.isinf(diffusion_time):
            diffusion_time = torch.tensor(0.1, device=signal.device)
        diffusion_time = diffusion_time.clamp(min=1e-4)

        # Run each spectral filter
        outputs = []
        for f in self.filters:
            out = f(cc, signal, diffusion_time=diffusion_time)
            outputs.append(out)

        # Sheaf diffusion path
        if self.include_sheaf:
            sheaf_out = self.sheaf(cc, signal, diffusion_time, wave_damping)
            outputs.append(sheaf_out)

        # Identity (nowave) path
        if self.include_identity:
            outputs.append(signal)

        # Stack: (num_paths, N, D)
        stacked = torch.stack(outputs, dim=0)

        # Blend with weights
        if filter_weights is not None:
            # filter_weights: (num_paths,) → (num_paths, 1, 1) for broadcast
            w = filter_weights.view(-1, 1, 1)
            blended = (w * stacked).sum(dim=0)  # (N, D)
        else:
            blended = stacked.mean(dim=0)  # (N, D)

        # Optional neural ODE blend
        if self.use_neural_ode:
            wave_out = self.wave(cc, signal, diffusion_time, wave_damping)
            gate_input = torch.cat([blended, wave_out], dim=-1)
            g = self.ode_gate(gate_input)
            blended = g * blended + (1 - g) * wave_out

        # Wave strength gate: learned bypass
        if self.use_wave_strength_gate:
            strength = torch.sigmoid(self.wave_strength)
            return strength * blended + (1 - strength) * signal
        return blended


# ---------------------------------------------------------------------------
# Sheaf wave dynamics (amortized restriction maps)
# ---------------------------------------------------------------------------


class SheafWaveDynamics(nn.Module):
    """Wave dynamics using sheaf diffusion with amortized restriction maps.

    Since graphs vary in size, we cannot use per-edge learnable parameters
    (as ``SheafLaplacian`` does). Instead a small MLP takes concatenated
    source/target embeddings and produces low-rank restriction map factors
    on the fly.  This allows the sheaf structure to generalize across
    graphs of different sizes and topologies.

    The resulting sheaf Laplacian is passed to ``SheafDiffusion`` which
    runs Euler-step heat diffusion in the block (N*d, N*d) space.
    """

    def __init__(
        self,
        embedding_dim: int,
        use_wave_strength_gate: bool = False,
        rank: int = 4,
        n_diffusion_steps: int = 5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_wave_strength_gate = use_wave_strength_gate
        self.rank = rank

        # MLP: (source_emb || target_emb) -> low-rank factors for both endpoints.
        # Output size: 2 * rank * embedding_dim (two (rank, d) matrices).
        map_size = 2 * rank * embedding_dim
        self.restriction_net = nn.Sequential(
            nn.Linear(2 * embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, map_size),
        )

        # Learnable scale for restriction maps (after Frobenius normalization)
        self.restriction_scale = nn.Parameter(torch.tensor(0.1))

        self.diffusion = SheafDiffusion(embedding_dim, n_steps=n_diffusion_steps)

        if use_wave_strength_gate:
            self.wave_strength = nn.Parameter(torch.tensor(1.0))

    def _build_sheaf_laplacian(
        self,
        cc: CellComplex,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Build a sheaf Laplacian from node embeddings via the restriction net.

        Args:
            cc: Cell complex defining graph topology.
            node_embeddings: (N, D) node feature matrix.

        Returns:
            Block sheaf Laplacian of shape (N*D, N*D).
        """
        n = cc.num_cells(0)
        d = self.embedding_dim
        device = node_embeddings.device

        L = torch.zeros(n * d, n * d, device=device)
        n_edges = cc.num_cells(1)

        if n_edges == 0:
            return L

        scale = torch.sigmoid(self.restriction_scale)  # bounded (0, 1)

        for e in range(n_edges):
            u = cc._1_cell_sources[e]
            v = cc._1_cell_targets[e]

            # Concatenate endpoint embeddings and predict restriction factors
            edge_input = torch.cat([node_embeddings[u], node_embeddings[v]])  # (2*d,)
            # Clamp input to prevent overflow in MLP (hub nodes can have large embeddings)
            edge_input = edge_input.clamp(-10, 10)
            maps_flat = self.restriction_net(edge_input)  # (2 * rank * d,)

            r = self.rank
            maps = maps_flat.view(2, r, d)  # (2, rank, d)

            # Spectral normalization — constrain max singular value ≤ scale
            for i in range(2):
                U, S, Vh = torch.linalg.svd(maps[i].float(), full_matrices=False)
                S = S.clamp(max=1.0)
                maps[i] = U @ torch.diag(S) @ Vh
            maps = maps * scale

            F_eu = maps[0]  # (rank, d)
            F_ev = maps[1]  # (rank, d)

            # Off-diagonal blocks: L[u_block, v_block] -= F_eu^T @ F_ev
            off_diag = F_eu.T @ F_ev  # (d, d)
            L[u * d:(u + 1) * d, v * d:(v + 1) * d] -= off_diag
            L[v * d:(v + 1) * d, u * d:(u + 1) * d] -= F_ev.T @ F_eu

            # Diagonal blocks
            L[u * d:(u + 1) * d, u * d:(u + 1) * d] += F_eu.T @ F_eu
            L[v * d:(v + 1) * d, v * d:(v + 1) * d] += F_ev.T @ F_ev

        # Symmetrize first
        L = (L + L.T) / 2

        # For small graphs, use exact eigendecomposition for stability
        nd = n * d
        if nd <= 512:
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(L.float())
                # Enforce positive semi-definiteness: clamp negative eigenvalues
                eigenvalues = eigenvalues.clamp(min=0)
                # Normalize by largest eigenvalue
                lam_max = eigenvalues.max().clamp(min=1e-8)
                eigenvalues = eigenvalues / lam_max
                L = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
            except Exception:
                # Fallback: just normalize by Frobenius norm
                L = L / L.norm().clamp(min=1e-8)
        else:
            # Large graphs: power iteration (existing approach)
            v = torch.randn(nd, device=device)
            v = v / v.norm()
            for _ in range(15):
                v = L @ v
                v_norm = v.norm()
                if v_norm < 1e-8:
                    break
                v = v / v_norm
            lam_max = (v @ L @ v).clamp(min=1e-8)
            L = L / lam_max
            # Symmetrize again after normalization
            L = (L + L.T) / 2

        # Early NaN guard
        if torch.isnan(L).any():
            L = torch.zeros_like(L)

        return L

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        diffusion_time: torch.Tensor,
        wave_damping: torch.Tensor,
        filter_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sheaf diffusion to signal.

        Args:
            cc: Cell complex.
            signal: (N, D) node features.
            diffusion_time: Positive scalar controlling diffusion extent.
            wave_damping: Unused (kept for API compatibility with WaveDynamics).
            filter_weights: Unused (accepted for API compatibility with MultiFilterDynamics).

        Returns:
            Diffused signal of shape (N, D).
        """
        L_sheaf = self._build_sheaf_laplacian(cc, signal)
        if torch.isnan(L_sheaf).any():
            # Bail out: return signal unchanged
            if self.use_wave_strength_gate:
                strength = torch.sigmoid(self.wave_strength)
                return strength * signal + (1 - strength) * signal
            return signal
        if torch.isnan(diffusion_time) or torch.isinf(diffusion_time):
            diffusion_time = torch.tensor(0.1, device=signal.device)
        diffusion_time = diffusion_time.clamp(min=1e-4)
        result = self.diffusion(cc, signal, L_sheaf, diffusion_time)

        # NaN guard: if sheaf diffusion produces NaN, fall back to input signal.
        if torch.isnan(result).any():
            result = signal

        if self.use_wave_strength_gate:
            strength = torch.sigmoid(self.wave_strength)
            return strength * result + (1 - strength) * signal
        return result
