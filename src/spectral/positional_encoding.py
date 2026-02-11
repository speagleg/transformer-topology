import torch
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition


def laplacian_pe(cc: CellComplex, dim: int, k: int) -> torch.Tensor:
    """Compute Laplacian positional encodings for cells of the given dimension.

    Returns the k smallest eigenvectors of the Hodge Laplacian as positional
    encodings. Each row corresponds to a cell, each column to an eigenvector.

    Args:
        cc: Cell complex.
        dim: Cell dimension (0 or 1).
        k: Number of eigenvectors to use as positional encoding dimensions.

    Returns:
        Tensor of shape (num_cells, k) -- the positional encodings.
    """
    eigenvalues, eigenvectors = spectral_decomposition(cc, dim=dim, k=k)
    return eigenvectors
