import torch
from typing import Optional
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0, hodge_laplacian_1


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
    else:
        raise ValueError(f"dim={dim} not supported in Phase 1")

    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    if k is not None:
        k = min(k, eigenvalues.shape[0])
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    return eigenvalues, eigenvectors
