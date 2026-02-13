"""Persistent homology computation and vectorization.

Uses gudhi for Rips complex computation when available.
Falls back to zeros when gudhi is not installed.
"""

import torch
import numpy as np
from src.cell_complex.cell_complex import CellComplex

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def compute_persistence_diagram(
    cc: CellComplex, max_dimension: int = 1
) -> list[np.ndarray]:
    """Compute persistence diagrams from node embeddings via Rips complex.

    Args:
        cc: Cell complex with node embeddings.
        max_dimension: Maximum homology dimension to compute.

    Returns:
        List of arrays, one per dimension. Each array has shape (n_features, 2)
        with (birth, death) pairs. Returns empty arrays if gudhi unavailable.
    """
    if not GUDHI_AVAILABLE:
        return [np.empty((0, 2)) for _ in range(max_dimension + 1)]

    embeddings = cc.get_embeddings(0).detach().cpu().numpy()
    if embeddings.shape[0] < 2:
        return [np.empty((0, 2)) for _ in range(max_dimension + 1)]

    rips = gudhi.RipsComplex(points=embeddings, max_edge_length=float('inf'))
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
    simplex_tree.compute_persistence()

    diagrams = []
    for dim in range(max_dimension + 1):
        pairs = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            diagrams.append(np.empty((0, 2)))
        else:
            # Filter out infinite death times
            finite_pairs = pairs[np.isfinite(pairs[:, 1])]
            if len(finite_pairs) == 0:
                diagrams.append(np.empty((0, 2)))
            else:
                diagrams.append(finite_pairs)
    return diagrams


def vectorize_persistence(
    diagrams: list[np.ndarray], num_features: int = 16
) -> torch.Tensor:
    """Convert persistence diagrams to a fixed-size feature vector.

    Features per dimension: sorted lifetimes (top-k) + statistics
    (mean lifetime, max lifetime, total persistence, num features).

    Args:
        diagrams: List of persistence diagrams from compute_persistence_diagram.
        num_features: Number of features per homology dimension.

    Returns:
        Tensor of shape (len(diagrams) * num_features,).
    """
    all_features = []
    for diagram in diagrams:
        if diagram.shape[0] == 0:
            all_features.append(torch.zeros(num_features))
            continue

        lifetimes = torch.tensor(diagram[:, 1] - diagram[:, 0], dtype=torch.float32)
        lifetimes_sorted, _ = lifetimes.sort(descending=True)

        # Top-k lifetimes (padded with zeros)
        n_lifetime_feats = num_features - 4
        top_k = torch.zeros(n_lifetime_feats)
        k = min(n_lifetime_feats, len(lifetimes_sorted))
        top_k[:k] = lifetimes_sorted[:k]

        # Statistics
        stats = torch.tensor([
            lifetimes.mean().item(),
            lifetimes.max().item(),
            lifetimes.sum().item(),
            float(len(lifetimes)),
        ])

        all_features.append(torch.cat([top_k, stats]))

    return torch.cat(all_features)


def persistence_node_features(
    cc: CellComplex, num_features: int = 8
) -> torch.Tensor:
    """Compute per-node persistence features from ego-graph persistence.

    For each node, computes persistence of its 1-hop neighborhood.

    Args:
        cc: Cell complex with node embeddings.
        num_features: Number of persistence features per node.

    Returns:
        Tensor of shape (num_nodes, num_features).
    """
    if not GUDHI_AVAILABLE:
        return torch.zeros(cc.num_cells(0), num_features)

    num_nodes = cc.num_cells(0)
    embeddings = cc.get_embeddings(0).detach()
    adj = cc.adjacency_matrix(0)
    features = torch.zeros(num_nodes, num_features)

    for i in range(num_nodes):
        # Get 1-hop neighbors including self
        neighbors = [i] + [j for j in range(num_nodes) if adj[i, j] > 0]
        if len(neighbors) < 2:
            continue

        ego_embs = embeddings[neighbors].cpu().numpy()
        rips = gudhi.RipsComplex(points=ego_embs, max_edge_length=float('inf'))
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()

        pairs = st.persistence_intervals_in_dimension(0)
        finite = pairs[np.isfinite(pairs[:, 1])] if len(pairs) > 0 else np.empty((0, 2))

        if finite.shape[0] > 0:
            lifetimes = torch.tensor(finite[:, 1] - finite[:, 0], dtype=torch.float32)
            lifetimes_sorted, _ = lifetimes.sort(descending=True)
            n_lt = num_features - 2
            k = min(n_lt, len(lifetimes_sorted))
            features[i, :k] = lifetimes_sorted[:k]
            features[i, -2] = lifetimes.mean()
            features[i, -1] = lifetimes.max()

    return features
