import torch
from typing import Optional
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0, hodge_laplacian_1, hodge_laplacian_2


def spectral_decomposition(
    cc: CellComplex, dim: int, k: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute eigendecomposition of the Hodge Laplacian.

    Args:
        cc: Cell complex to decompose.
        dim: Cell dimension (0 or 1) selecting which Hodge Laplacian to use.
        k: If provided, return only the k smallest eigenvalues/vectors.

    Returns:
        eigenvalues: Tensor of shape (n,) or (k,), sorted ascending.
        eigenvectors: Tensor of shape (N, n) or (N, k), columns are eigenvectors.
    """
    if dim == 0:
        L = hodge_laplacian_0(cc)
    elif dim == 1:
        L = hodge_laplacian_1(cc)
    elif dim == 2:
        L = hodge_laplacian_2(cc)
    else:
        raise ValueError(f"dim={dim} not supported")

    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    if k is not None:
        k = min(k, eigenvalues.shape[0])
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    return eigenvalues, eigenvectors


def hodge_decomposition(
    cc: CellComplex, signal: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose a k-chain signal into gradient, curl, and harmonic components.

    Uses the Hodge decomposition: signal = gradient + curl + harmonic
    where:
      - gradient = projection onto image(B_k^T)  (exact forms)
      - curl = projection onto image(B_{k+1})     (coexact forms)
      - harmonic = signal - gradient - curl        (harmonic forms, kernel of L_k)

    Args:
        cc: Cell complex with boundary operators.
        signal: Signal tensor of shape (num_k_cells,).
        dim: Cell dimension of the signal (e.g., 1 for edge signals).

    Returns:
        (gradient, curl, harmonic) each of shape (num_k_cells,).
    """
    # Gradient component: projection onto image(B_k^T)
    # P_grad = B^T @ (B @ B^T)^+ @ B
    if dim > 0:
        B = cc.boundary_operator(dim)  # shape: (num_{k-1}_cells, num_k_cells)
        BBT_pinv = torch.linalg.pinv(B @ B.T)  # (num_{k-1}, num_{k-1})
        gradient = B.T @ (BBT_pinv @ (B @ signal))
    else:
        gradient = torch.zeros_like(signal)

    # Curl component: projection onto image(B_{k+1})
    if cc.num_cells(dim + 1) > 0:
        B_up = cc.boundary_operator(dim + 1)  # shape: (num_k_cells, num_{k+1}_cells)
        # Project onto column space of B_{k+1}
        BupTBup_pinv = torch.linalg.pinv(B_up.T @ B_up)
        curl = B_up @ (BupTBup_pinv @ (B_up.T @ signal))
    else:
        curl = torch.zeros_like(signal)

    # Harmonic component
    harmonic = signal - gradient - curl

    return gradient, curl, harmonic
