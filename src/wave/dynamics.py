"""Wave and diffusion dynamics on cell complexes.

Provides heat diffusion (analytic spectral solution) and wave propagation
(neural ODE) modules that are parameterized by control signals from the
GNN executive.  WaveDynamics blends both with a learned per-node gate.
"""

import torch
import torch.nn as nn

from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition
from src.spectral.laplacian import hodge_laplacian_0

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
    """Blended heat diffusion + wave propagation with a learned per-node gate.

    gate = sigmoid(Linear(concat(heat_out, wave_out)))
    output = gate * heat_out + (1 - gate) * wave_out
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.heat = HeatDiffusion()
        self.wave = WavePropagation(embedding_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        diffusion_time: torch.Tensor,
        wave_damping: torch.Tensor,
    ) -> torch.Tensor:
        """Run heat and wave in parallel, blend with learned gate.

        Args:
            cc: Cell complex.
            signal: Node signal of shape (N, D).
            diffusion_time: Positive scalar (controls both heat and wave).
            wave_damping: Positive scalar (wave only).

        Returns:
            Blended signal of shape (N, D).
        """
        heat_out = self.heat(cc, signal, diffusion_time)   # (N, D)
        wave_out = self.wave(cc, signal, diffusion_time, wave_damping)  # (N, D)

        gate_input = torch.cat([heat_out, wave_out], dim=-1)  # (N, 2D)
        g = self.gate(gate_input)  # (N, D)

        return g * heat_out + (1 - g) * wave_out
