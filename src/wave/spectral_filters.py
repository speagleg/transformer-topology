"""Pluggable spectral filters for cell complexes.

Each filter implements h(λ) applied in the eigenbasis of a Hodge Laplacian.
Filters share the eigendecomposition step — only the filter response differs.

Usage:
    filter = FILTER_REGISTRY['bandpass'](dim=0)
    output = filter(cc, signal, diffusion_time=t)
"""

import math
import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition
from src.spectral.magnetic_laplacian import magnetic_spectral_decomposition


class SpectralFilter(nn.Module):
    """Base class for analytic spectral filters on cell complexes.

    Subclasses implement filter_response(eigenvalues, **params) → coefficients.
    The base class handles eigendecomposition and the forward transform.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def filter_response(self, eigenvalues: torch.Tensor, **params) -> torch.Tensor:
        """Compute filter coefficients h(λ) for each eigenvalue.

        Args:
            eigenvalues: Tensor of shape (K,), sorted ascending.
            **params: Filter-specific parameters (e.g. diffusion_time).

        Returns:
            Coefficients tensor of shape (K,).
        """
        raise NotImplementedError

    @property
    def _skip_normalize(self) -> bool:
        """Whether this filter already normalizes eigenvalues internally."""
        return False

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        **params,
    ) -> torch.Tensor:
        """Apply spectral filter to signal on the cell complex.

        Eigenvalues are normalized to [0, 1] for size-invariance unless
        the filter already handles normalization internally.

        Args:
            cc: Cell complex whose Hodge Laplacian (at self.dim) is used.
            signal: Signal tensor, shape (N,) or (N, D).
            **params: Passed to filter_response.

        Returns:
            Filtered signal, same shape as input.
        """
        squeeze = signal.dim() == 1
        if squeeze:
            signal = signal.unsqueeze(-1)

        try:
            eigenvalues, eigenvectors = spectral_decomposition(
                cc, dim=self.dim, normalize=not self._skip_normalize,
            )
        except Exception:
            return signal.squeeze(-1) if squeeze else signal

        coeffs = self.filter_response(eigenvalues, **params)

        projected = eigenvectors.T @ signal
        scaled = coeffs.unsqueeze(-1) * projected
        result = eigenvectors @ scaled

        return result.squeeze(-1) if squeeze else result


# ---------------------------------------------------------------------------
# Concrete filter implementations
# ---------------------------------------------------------------------------


class HeatFilter(SpectralFilter):
    """Heat diffusion: h(λ) = exp(-λt).

    Low-pass smoother. Small t preserves locality, large t → global mean.
    """

    def filter_response(self, eigenvalues, diffusion_time, **kwargs):
        return torch.exp(-eigenvalues * diffusion_time)


class BandPassFilter(SpectralFilter):
    """Band-pass filter: h(λ) = exp(-(λ-λ₀)²/2σ²).

    Learnable center frequency λ₀ and bandwidth σ. Selects a spectral band.
    """

    def __init__(self, dim: int = 0, center: float = 1.0, bandwidth: float = 0.5):
        super().__init__(dim)
        self.center = nn.Parameter(torch.tensor(center))
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth))

    def filter_response(self, eigenvalues, **kwargs):
        sigma = self.bandwidth.abs().clamp(min=1e-4)
        return torch.exp(-(eigenvalues - self.center) ** 2 / (2 * sigma ** 2))


class WaveCosineFilter(SpectralFilter):
    """True wave equation: h(λ) = cos(√λ · t).

    Oscillatory — preserves frequency content via interference patterns.
    Unlike heat diffusion, this does not smooth the signal.
    """

    def filter_response(self, eigenvalues, diffusion_time, **kwargs):
        return torch.cos(torch.sqrt(eigenvalues.clamp(min=0)) * diffusion_time)


class SchrodingerFilter(SpectralFilter):
    """Schrodinger equation (real part): h(λ) = cos(λt).

    Quantum walk filter. Norm-preserving, produces interference patterns.
    Returns real part only (for real-valued signals).
    """

    def filter_response(self, eigenvalues, diffusion_time, **kwargs):
        return torch.cos(eigenvalues * diffusion_time)


class ChebyshevFilter(SpectralFilter):
    """Chebyshev polynomial filter: h(λ) = Σ θ_k T_k(λ̃).

    Learnable polynomial over the normalized spectrum. More flexible than
    single-kernel filters. λ̃ = 2λ/λ_max - 1 maps eigenvalues to [-1, 1].
    """

    def __init__(self, dim: int = 0, order: int = 5):
        super().__init__(dim)
        self.coeffs = nn.Parameter(torch.zeros(order))
        # Initialize as approximate low-pass
        nn.init.constant_(self.coeffs[0], 1.0)

    @property
    def _skip_normalize(self) -> bool:
        """ChebyshevFilter normalizes eigenvalues internally via λ_max."""
        return True

    def filter_response(self, eigenvalues, **kwargs):
        lambda_max = eigenvalues.max().clamp(min=1e-8)
        x = 2 * eigenvalues / lambda_max - 1  # Normalize to [-1, 1]

        # Chebyshev recurrence: T_0=1, T_1=x, T_{k+1}=2x*T_k - T_{k-1}
        T_prev = torch.ones_like(x)
        T_curr = x
        result = self.coeffs[0] * T_prev

        if len(self.coeffs) > 1:
            result = result + self.coeffs[1] * T_curr

        for k in range(2, len(self.coeffs)):
            T_next = 2 * x * T_curr - T_prev
            result = result + self.coeffs[k] * T_next
            T_prev = T_curr
            T_curr = T_next

        return result


class SpectralWaveletFilter(SpectralFilter):
    """Multi-scale spectral graph wavelets.

    Uses Mexican hat wavelet g(sλ) = sλ·exp(-sλ) at multiple learned scales.
    Averages across scales to produce a single output.
    """

    def __init__(self, dim: int = 0, n_scales: int = 4):
        super().__init__(dim)
        self.log_scales = nn.Parameter(torch.linspace(-1, 2, n_scales))

    def filter_response(self, eigenvalues, **kwargs):
        # Average response across all scales
        total = torch.zeros_like(eigenvalues)
        for log_s in self.log_scales:
            s = torch.exp(log_s)
            sl = s * eigenvalues
            total = total + sl * torch.exp(-sl)
        return total / len(self.log_scales)


class MagneticFilter(SpectralFilter):
    """Magnetic Laplacian heat kernel: h(λ) = exp(-λt) in the magnetic eigenbasis.

    Uses complex-valued eigenvectors from the magnetic Laplacian L_q which
    encodes edge directionality via phase factors parameterized by charge q.
    At q=0 this reduces to standard heat diffusion on the graph Laplacian.

    The magnetic eigenvectors are complex (unitary basis), so the projection
    uses conjugate transpose: V^H @ signal, and we take the real part of
    the reconstructed signal.
    """

    def __init__(self, dim: int = 0, q: float = 0.25):
        super().__init__(dim)
        self.q = q

    def filter_response(self, eigenvalues: torch.Tensor, diffusion_time, **kwargs) -> torch.Tensor:
        return torch.exp(-eigenvalues * diffusion_time)

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        **params,
    ) -> torch.Tensor:
        """Apply magnetic spectral filter to signal on the cell complex.

        Overrides the base forward to use magnetic_spectral_decomposition
        with complex eigenvectors.

        Args:
            cc: Cell complex whose magnetic Laplacian is used.
            signal: Signal tensor, shape (N,) or (N, D).
            **params: Must include diffusion_time.

        Returns:
            Filtered signal (real-valued), same shape as input.
        """
        squeeze = signal.dim() == 1
        if squeeze:
            signal = signal.unsqueeze(-1)

        try:
            eigenvalues, eigenvectors = magnetic_spectral_decomposition(
                cc, q=self.q, k=None,
            )
        except Exception:
            return signal.squeeze(-1) if squeeze else signal

        # Normalize eigenvalues to [0, 1] for size-invariance
        eigenvalues = eigenvalues / eigenvalues.max().clamp(min=1e-8)

        coeffs = self.filter_response(eigenvalues, **params)

        # Complex projection: V^H @ signal (conjugate transpose)
        signal_c = signal.to(eigenvectors.dtype)
        projected = eigenvectors.conj().T @ signal_c
        scaled = coeffs.unsqueeze(-1) * projected
        result = eigenvectors @ scaled

        # Return real part only
        result = result.real

        return result.squeeze(-1) if squeeze else result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FILTER_REGISTRY: dict[str, type[SpectralFilter]] = {
    'heat': HeatFilter,
    'bandpass': BandPassFilter,
    'wave_cosine': WaveCosineFilter,
    'schrodinger': SchrodingerFilter,
    'chebyshev': ChebyshevFilter,
    'wavelet': SpectralWaveletFilter,
    'magnetic': MagneticFilter,
}


def create_spectral_filter(
    filter_type: str = 'heat',
    dim: int = 0,
    **kwargs,
) -> SpectralFilter:
    """Create a spectral filter from config.

    Args:
        filter_type: Key in FILTER_REGISTRY.
        dim: Hodge Laplacian dimension (0=node, 1=edge, 2=face).
        **kwargs: Filter-specific params (e.g. order=5 for chebyshev).

    Returns:
        Instantiated SpectralFilter.
    """
    if filter_type not in FILTER_REGISTRY:
        raise ValueError(
            f"Unknown filter type '{filter_type}'. "
            f"Available: {list(FILTER_REGISTRY.keys())}"
        )
    cls = FILTER_REGISTRY[filter_type]
    return cls(dim=dim, **kwargs)
