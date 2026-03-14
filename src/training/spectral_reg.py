"""Spectral gap regularization to encourage information flow."""
import torch
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0


def spectral_gap_loss(spectral_gap: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalize small spectral gaps: -log(gap + eps).

    Larger gaps → smaller loss. Encourages information to flow
    rather than cycle in the cell complex.
    """
    return -torch.log(spectral_gap + eps)


def compute_batch_spectral_gap(ccs: list[CellComplex]) -> torch.Tensor:
    """Compute mean spectral gap across a batch of cell complexes.

    Uses the L0 Hodge Laplacian eigenvalues. The spectral gap is the
    smallest nonzero eigenvalue (Fiedler value).
    """
    gaps = []
    for cc in ccs:
        if cc.num_cells(0) == 0 or cc.num_cells(1) == 0:
            continue
        try:
            L0 = hodge_laplacian_0(cc).float()
            if L0.numel() == 0:
                continue
            eigenvalues = torch.linalg.eigvalsh(L0)
            # Spectral gap = smallest eigenvalue > threshold
            nonzero = eigenvalues[eigenvalues > 1e-5]
            if len(nonzero) > 0:
                gaps.append(nonzero[0])
        except Exception:
            continue
    if not gaps:
        return torch.tensor(0.0)
    return torch.stack(gaps).mean()
