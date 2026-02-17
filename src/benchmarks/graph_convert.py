"""Convert NetworkX graphs and PyG Data objects to CellComplex.

Key innovation: structural embeddings instead of random.
- emb[0] = normalized degree
- emb[1] = clustering coefficient
- emb[2] = normalized BFS distance from source
- emb[3:] = small random noise for symmetry breaking

This forces the model to learn from topology, not memorize random vectors.
"""

import torch
import networkx as nx

from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.topological_tasks import auto_fill_triangles


def nx_to_cell_complex(
    G: nx.Graph,
    embedding_dim: int,
    source_node: int | None = None,
    target_node: int | None = None,
    blocked_nodes: list[int] | None = None,
    source2_node: int | None = None,
    edge_delays: dict[tuple[int, int], int] | None = None,
    fill_triangles: bool = True,
) -> tuple[CellComplex, dict[int, int]]:
    """Convert a NetworkX graph to a CellComplex with structural embeddings.

    Args:
        G: A NetworkX graph.
        embedding_dim: Dimension of cell embeddings (must be >= 4).
        source_node: NX node to mark as "source".
        target_node: NX node to mark as "target".
        blocked_nodes: NX nodes to mark as "blocked".
        source2_node: NX node to mark as "source2" (for interference tasks).
        edge_delays: Dict mapping (u, v) -> delay for weighted edges.
                     If None, all edges get delay=1.
        fill_triangles: Whether to auto-detect triangles and add 2-cells.

    Returns:
        (CellComplex, node_map) where node_map maps NX node -> CC node index.
    """
    assert embedding_dim >= 4, "embedding_dim must be >= 4 for structural features"

    cc = CellComplex(embedding_dim=embedding_dim)
    nodes = list(G.nodes())
    node_map: dict[int, int] = {}

    # Precompute structural features
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    max_degree = max(max_degree, 1)  # guard against all-isolated-node graphs
    clustering = nx.clustering(G)

    # BFS distances from source (if provided)
    if source_node is not None and source_node in G:
        bfs_dist = nx.single_source_shortest_path_length(G, source_node)
        max_dist = max(bfs_dist.values()) if bfs_dist else 1
    else:
        bfs_dist = {}
        max_dist = 1

    blocked_set = set(blocked_nodes) if blocked_nodes else set()

    # Add 0-cells with structural embeddings
    for nx_node in nodes:
        emb = torch.randn(embedding_dim) * 0.1  # small noise
        emb[0] = degrees[nx_node] / max_degree
        emb[1] = clustering[nx_node]
        emb[2] = bfs_dist.get(nx_node, 0) / max_dist if max_dist > 0 else 0.0

        # Determine cell type
        if nx_node == source_node:
            cell_type = "source"
        elif nx_node == target_node:
            cell_type = "target"
        elif nx_node in blocked_set:
            cell_type = "blocked"
        elif nx_node == source2_node:
            cell_type = "source2"
        else:
            cell_type = "node"

        cc_idx = cc.add_0_cell(emb, cell_type)
        node_map[nx_node] = cc_idx

    # Add 1-cells (edges) with structural edge embeddings
    for u, v in G.edges():
        emb = torch.randn(embedding_dim) * 0.1  # small noise
        if edge_delays is not None:
            delay = edge_delays.get((u, v), edge_delays.get((v, u), 1))
        else:
            delay = 1
        emb[0] = float(delay)
        cc.add_1_cell(node_map[u], node_map[v], emb, "edge")

    # Auto-fill 2-cells from triangles
    if fill_triangles:
        auto_fill_triangles(cc)

    return cc, node_map


def pyg_to_cell_complex(
    data,
    embedding_dim: int,
) -> CellComplex:
    """Convert a PyG Data object to a CellComplex.

    Uses node features from data.x if available, otherwise structural features.
    Edge features from data.edge_attr if available.

    Args:
        data: PyG Data object with edge_index (and optionally x, edge_attr).
        embedding_dim: Dimension of cell embeddings.

    Returns:
        CellComplex with the graph structure.
    """
    cc = CellComplex(embedding_dim=embedding_dim)

    num_nodes = data.num_nodes
    has_features = hasattr(data, 'x') and data.x is not None

    # Add 0-cells
    for i in range(num_nodes):
        if has_features:
            feat = data.x[i]
            if feat.shape[0] < embedding_dim:
                emb = torch.zeros(embedding_dim)
                emb[:feat.shape[0]] = feat
            elif feat.shape[0] > embedding_dim:
                emb = feat[:embedding_dim]
            else:
                emb = feat.clone()
        else:
            emb = torch.randn(embedding_dim) * 0.1
        cc.add_0_cell(emb, "node")

    # Add 1-cells (deduplicate undirected edges)
    edge_index = data.edge_index
    has_edge_attr = hasattr(data, 'edge_attr') and data.edge_attr is not None
    seen_edges: set[tuple[int, int]] = set()

    for idx in range(edge_index.shape[1]):
        u = edge_index[0, idx].item()
        v = edge_index[1, idx].item()
        key = (min(u, v), max(u, v))
        if key in seen_edges:
            continue
        seen_edges.add(key)

        if has_edge_attr:
            feat = data.edge_attr[idx]
            if feat.shape[0] < embedding_dim:
                emb = torch.zeros(embedding_dim)
                emb[:feat.shape[0]] = feat
            elif feat.shape[0] > embedding_dim:
                emb = feat[:embedding_dim]
            else:
                emb = feat.clone()
        else:
            emb = torch.randn(embedding_dim) * 0.1
        cc.add_1_cell(u, v, emb, "edge")

    auto_fill_triangles(cc)
    return cc
