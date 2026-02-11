import torch
from src.cell_complex.cell_complex import CellComplex


def hodge_laplacian_0(cc: CellComplex) -> torch.Tensor:
    """L_0 = B_1 @ B_1^T (standard graph Laplacian)."""
    B1 = cc.boundary_operator(1)
    L0 = B1 @ B1.T
    return L0


def hodge_laplacian_1(cc: CellComplex) -> torch.Tensor:
    """L_1 = B_1^T @ B_1 (Phase 1: no B_2 term)."""
    B1 = cc.boundary_operator(1)
    L1 = B1.T @ B1
    return L1
