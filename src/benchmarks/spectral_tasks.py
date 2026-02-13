"""Spectral benchmark tasks that directly test the spectral/topological machinery.

Two tasks:
- spectral_gap: Predict which bucket the algebraic connectivity (lambda_2) falls into.
  Requires global spectral analysis - local message passing alone isn't enough.
- hodge_class: Predict whether edge flow is gradient-, curl-, or harmonic-dominated.
  Tests whether higher-order topological machinery (2-cells, Hodge Laplacians) contributes.
"""

import random

import numpy as np
import torch

from src.benchmarks.graph_generators import random_graph
from src.benchmarks.graph_convert import nx_to_cell_complex
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition, hodge_decomposition


def _calibrate_spectral_gap_buckets(
    n_nodes: int,
    embedding_dim: int,
    num_buckets: int = 8,
    calibration_size: int = 500,
    topologies: list[str] | None = None,
) -> list[float]:
    """Compute bucket boundaries for spectral gap discretization.

    Generates a calibration set of graphs, computes lambda_2 for each,
    and returns quantile-based bucket boundaries.

    Returns:
        List of (num_buckets - 1) boundary values.
    """
    lambda2_values = []
    all_topos = topologies or ['ba', 'ws', 'sbm', 'grid', 'tree', 'ladder', 'caveman', 'er']

    for _ in range(calibration_size):
        topo = random.choice(all_topos)
        G = random_graph(n_nodes, topology=topo)
        cc, _ = nx_to_cell_complex(G, embedding_dim, source_node=0, target_node=1)
        try:
            eigenvalues, _ = spectral_decomposition(cc, dim=0)
            if len(eigenvalues) > 1:
                lambda2_values.append(eigenvalues[1].item())
            else:
                lambda2_values.append(0.0)
        except (RuntimeError, ValueError):
            lambda2_values.append(0.0)

    quantiles = np.linspace(0, 1, num_buckets + 1)[1:-1]  # e.g. [0.125, 0.25, ..., 0.875]
    boundaries = list(np.quantile(lambda2_values, quantiles))
    return boundaries


def _discretize(value: float, boundaries: list[float]) -> int:
    """Map a continuous value to a bucket index given sorted boundaries."""
    for i, boundary in enumerate(boundaries):
        if value < boundary:
            return i
    return len(boundaries)


# Module-level cache for bucket boundaries
_BUCKET_CACHE: dict[tuple, list[float]] = {}


def _get_bucket_boundaries(
    n_nodes: int, embedding_dim: int, num_buckets: int,
    topologies: list[str] | None,
) -> list[float]:
    """Get or compute cached bucket boundaries."""
    topo_key = tuple(sorted(topologies)) if topologies else ()
    key = (n_nodes, embedding_dim, num_buckets, topo_key)
    if key not in _BUCKET_CACHE:
        _BUCKET_CACHE[key] = _calibrate_spectral_gap_buckets(
            n_nodes, embedding_dim, num_buckets,
            calibration_size=500, topologies=topologies,
        )
    return _BUCKET_CACHE[key]


def generate_spectral_gap_task(
    n_nodes: int,
    embedding_dim: int,
    num_buckets: int = 8,
    topologies: list[str] | None = None,
) -> tuple[CellComplex, int, int, int]:
    """Generate a spectral gap (algebraic connectivity) classification task.

    Predicts which bucket lambda_2 (second-smallest eigenvalue of L0) falls into.
    This CANNOT be solved without global spectral analysis.

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        num_buckets: Number of discretization buckets (output classes).
        topologies: List of allowed topology names, or None for all.

    Returns:
        (cell_complex, source, target, answer) tuple.
    """
    topo = random.choice(topologies) if topologies else None
    G = random_graph(n_nodes, topology=topo)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx,
        target_node=target_nx,
    )

    try:
        eigenvalues, _ = spectral_decomposition(cc, dim=0)
        lambda_2 = eigenvalues[1].item() if len(eigenvalues) > 1 else 0.0
    except (RuntimeError, ValueError):
        lambda_2 = 0.0

    boundaries = _get_bucket_boundaries(n_nodes, embedding_dim, num_buckets, topologies)
    answer = _discretize(lambda_2, boundaries)

    return cc, node_map[source_nx], node_map[target_nx], answer


def generate_hodge_class_task(
    n_nodes: int,
    embedding_dim: int,
    topologies: list[str] | None = None,
) -> tuple[CellComplex, int, int, int]:
    """Generate a Hodge decomposition dominance classification task.

    Constructs an explicit edge signal from one of the three Hodge components
    (gradient, curl, harmonic) plus noise, then encodes it in edge embeddings.
    The model must classify which component dominates.

    Class 0: gradient-dominated (signal in image of B1^T)
    Class 1: curl-dominated (signal in image of B2)
    Class 2: harmonic-dominated (signal in kernel of L1)

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        topologies: List of allowed topology names, or None for all.

    Returns:
        (cell_complex, source, target, answer) tuple.
    """
    topo = random.choice(topologies) if topologies else None
    G = random_graph(n_nodes, topology=topo)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx,
        target_node=target_nx,
    )

    n_edges = cc.num_cells(1)
    if n_edges == 0:
        return cc, node_map[source_nx], node_map[target_nx], 0

    # Choose target class (balanced 1/3 each)
    target_class = random.randint(0, 2)

    B1 = cc.boundary_operator(1)  # (n_nodes, n_edges)
    has_2cells = cc.num_cells(2) > 0
    B2 = cc.boundary_operator(2) if has_2cells else None

    signal = None

    if target_class == 0:  # gradient: signal = B1^T @ v
        v = torch.randn(cc.num_cells(0))
        signal = B1.T @ v

    elif target_class == 1:  # curl: signal = B2 @ w
        if B2 is not None and cc.num_cells(2) > 0:
            w = torch.randn(cc.num_cells(2))
            signal = B2 @ w
            if signal.norm() < 1e-8:
                # Degenerate B2, fall back to gradient
                target_class = 0
                v = torch.randn(cc.num_cells(0))
                signal = B1.T @ v
        else:
            # No 2-cells available, fall back to gradient
            target_class = 0
            v = torch.randn(cc.num_cells(0))
            signal = B1.T @ v

    elif target_class == 2:  # harmonic: signal in ker(L1)
        L1 = B1.T @ B1
        if B2 is not None:
            L1 = L1 + B2 @ B2.T
        eigenvalues, eigenvectors = torch.linalg.eigh(L1)
        harmonic_mask = eigenvalues.abs() < 1e-5
        if harmonic_mask.sum() > 0:
            harmonic_basis = eigenvectors[:, harmonic_mask]
            coeffs = torch.randn(harmonic_basis.shape[1])
            signal = harmonic_basis @ coeffs
        else:
            # No harmonic space, fall back to gradient
            target_class = 0
            v = torch.randn(cc.num_cells(0))
            signal = B1.T @ v

    # Normalize signal, add noise (small enough to preserve class)
    if signal.norm() > 1e-8:
        signal = signal / signal.norm()
    noise = torch.randn(n_edges) * 0.1
    signal = signal + noise

    # Encode signal in edge embedding dimension 0
    emb = cc.get_embeddings(1).clone()
    emb[:, 0] = signal
    cc.set_embeddings(1, emb)

    # Verify answer via actual Hodge decomposition
    try:
        gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
        norms = [gradient.norm().item(), curl.norm().item(), harmonic.norm().item()]
        answer = norms.index(max(norms))
    except (RuntimeError, ValueError):
        answer = target_class

    return cc, node_map[source_nx], node_map[target_nx], answer
