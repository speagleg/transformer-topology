"""Magnetic Laplacian for directed/flow graphs on cell complexes.

The magnetic Laplacian encodes edge directionality via complex-valued phases,
parameterized by a "charge" q. At q=0 it reduces to the standard graph
Laplacian L0. As q increases, eigenvalues split to reflect directional
asymmetry in the graph.

Reference: Fanuel, Suykens (2018) "Magnetic eigenmaps for the visualization
of directed networks".
"""

import torch
from typing import Optional
from src.cell_complex.cell_complex import CellComplex


def magnetic_laplacian(cc: CellComplex, q: float = 0.25) -> torch.Tensor:
    """Compute the magnetic Laplacian L_q (complex-valued).

    Uses B1 to determine edge orientations. For edge e connecting nodes u, v:
    - B1[u, e] = -1, B1[v, e] = +1 (boundary operator convention)
    - theta_uv = +1 (u -> v direction), theta_vu = -1 (v -> u)

    L_q[u, v] = -exp(i * 2*pi*q * theta_uv) if (u, v) is an edge
    L_q[u, u] = degree(u)

    The resulting matrix is Hermitian: L_q = L_q^H.

    Args:
        cc: Cell complex with boundary operator B1.
        q: Magnetic charge parameter (default 0.25). At q=0 this reduces
           to the standard graph Laplacian.

    Returns:
        Complex tensor of shape (N, N) where N = number of 0-cells.
    """
    n = cc.num_cells(0)
    device = cc.device
    L = torch.zeros(n, n, dtype=torch.complex64, device=device)

    n_edges = cc.num_cells(1)
    for e in range(n_edges):
        u = cc._1_cell_sources[e]
        v = cc._1_cell_targets[e]

        # theta_uv = +1 (u -> v), theta_vu = -1 (v -> u)
        # Off-diagonal: L_q[u, v] = -exp(i * 2*pi*q * theta_uv)
        phase_uv = torch.tensor(2.0 * torch.pi * q * 1.0)   # theta_uv = +1
        phase_vu = torch.tensor(2.0 * torch.pi * q * (-1.0))  # theta_vu = -1

        L[u, v] -= torch.exp(1j * phase_uv)
        L[v, u] -= torch.exp(1j * phase_vu)

        # Diagonal: degree contribution
        L[u, u] += 1.0
        L[v, v] += 1.0

    return L


def magnetic_spectral_decomposition(
    cc: CellComplex, q: float = 0.25, k: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of the magnetic Laplacian.

    Since the magnetic Laplacian is Hermitian (L_q = L_q^H), its eigenvalues
    are real and eigenvectors form a unitary basis (but are complex-valued).

    Args:
        cc: Cell complex with boundary operator B1.
        q: Magnetic charge parameter.
        k: If provided, return only the k smallest eigenvalues/vectors.

    Returns:
        eigenvalues: Real tensor of shape (N,) or (k,), sorted ascending.
        eigenvectors: Complex tensor of shape (N, N) or (N, k), columns
            are eigenvectors.
    """
    L_q = magnetic_laplacian(cc, q=q)

    # eigh works on Hermitian matrices, returns real eigenvalues (fp32 required)
    eigenvalues, eigenvectors = torch.linalg.eigh(L_q.to(torch.complex64))

    if k is not None:
        k = min(k, eigenvalues.shape[0])
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    return eigenvalues, eigenvectors
