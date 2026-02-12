import torch
from src.cell_complex.cell_complex import CellComplex


def hodge_laplacian_0(cc: CellComplex) -> torch.Tensor:
    """L_0 = B_1 @ B_1^T (standard graph Laplacian)."""
    B1 = cc.boundary_operator(1)
    L0 = B1 @ B1.T
    return L0


def hodge_laplacian_1(cc: CellComplex) -> torch.Tensor:
    """L_1 = B_1^T @ B_1 + B_2 @ B_2^T (full Hodge 1-Laplacian).

    When no 2-cells exist, reduces to B_1^T @ B_1.
    """
    B1 = cc.boundary_operator(1)
    L1 = B1.T @ B1
    if cc.num_cells(2) > 0:
        B2 = cc.boundary_operator(2)
        L1 = L1 + B2 @ B2.T
    return L1


def hodge_laplacian_2(cc: CellComplex) -> torch.Tensor:
    """L_2 = B_2^T @ B_2."""
    B2 = cc.boundary_operator(2)
    L2 = B2.T @ B2
    return L2
