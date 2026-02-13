"""Graph generator library wrapping NetworkX for diverse graph topologies.

All generators return connected NetworkX graphs suitable for conversion
to CellComplex via graph_convert.nx_to_cell_complex().
"""

import random
import networkx as nx


def generate_ba_graph(n: int, m: int = 2) -> nx.Graph:
    """Barabási-Albert preferential attachment (scale-free, hubs).

    Args:
        n: Number of nodes (must be >= m+1).
        m: Number of edges to attach from a new node to existing nodes.
    """
    m = min(m, n - 1)
    G = nx.barabasi_albert_graph(n, m)
    return G


def generate_ws_graph(n: int, k: int = 4, p: float = 0.3) -> nx.Graph:
    """Watts-Strogatz small-world graph (high clustering, short paths).

    Args:
        n: Number of nodes (must be >= k+1).
        k: Each node is joined with its k nearest neighbors in a ring.
        p: Probability of rewiring each edge.
    """
    k = min(k, n - 1)
    if k % 2 == 1:
        k = max(k - 1, 2)
    G = nx.watts_strogatz_graph(n, k, p)
    return G


def generate_sbm_graph(sizes: list[int], p_within: float = 0.5,
                       p_between: float = 0.05) -> nx.Graph:
    """Stochastic block model (community structure).

    Args:
        sizes: List of community sizes.
        p_within: Edge probability within communities.
        p_between: Edge probability between communities.
    """
    k = len(sizes)
    probs = [[p_within if i == j else p_between for j in range(k)]
             for i in range(k)]
    G = nx.stochastic_block_model(sizes, probs)
    # Ensure connected by adding edges between components
    _ensure_connected(G)
    return G


def generate_grid_graph(rows: int, cols: int) -> nx.Graph:
    """2D grid graph (regular, predictable long paths).

    Args:
        rows: Number of rows.
        cols: Number of columns.
    """
    G = nx.grid_2d_graph(rows, cols)
    # Relabel from (i,j) tuples to integers
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G


def generate_tree_graph(n: int) -> nx.Graph:
    """Random labeled tree (connected, no cycles, long paths).

    Args:
        n: Number of nodes.
    """
    G = nx.random_labeled_tree(n)
    return G


def generate_ladder_graph(n: int) -> nx.Graph:
    """Ladder graph: two parallel paths of n nodes connected by rungs.

    Total nodes = 2*n.

    Args:
        n: Length of each side of the ladder.
    """
    G = nx.ladder_graph(n)
    return G


def generate_caveman_graph(num_cliques: int, clique_size: int) -> nx.Graph:
    """Connected caveman graph (dense cliques with inter-clique edges).

    Args:
        num_cliques: Number of cliques.
        clique_size: Size of each clique.
    """
    G = nx.connected_caveman_graph(num_cliques, clique_size)
    return G


def generate_er_graph(n: int, p: float = 0.3) -> nx.Graph:
    """Erdős-Rényi random graph.

    Args:
        n: Number of nodes.
        p: Edge probability.
    """
    G = nx.erdos_renyi_graph(n, p)
    _ensure_connected(G)
    return G


def clrs_er_graph(n: int) -> nx.Graph:
    """ER graph with CLRS-standard edge probability (avg degree ~4).

    Scales p so that expected degree = min(4, n-1), matching the CLRS
    Algorithmic Reasoning Benchmark distribution.
    """
    p = min(4.0 / (n - 1), 0.5)
    return generate_er_graph(n, p)


def _ensure_connected(G: nx.Graph) -> None:
    """Add edges to make a disconnected graph connected (in-place)."""
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return
    # Chain components together
    for i in range(len(components) - 1):
        u = next(iter(components[i]))
        v = next(iter(components[i + 1]))
        G.add_edge(u, v)


def random_graph(n_nodes: int, topology: str | None = None) -> nx.Graph:
    """Generate a random connected graph with the given topology.

    If topology is None, picks randomly from available generators.
    Adapts parameters to each generator's constraints.

    Args:
        n_nodes: Approximate number of nodes (actual may vary slightly
                 for grid/ladder/caveman).
        topology: One of 'ba', 'ws', 'sbm', 'grid', 'tree', 'ladder',
                  'caveman', 'er', or None for random.

    Returns:
        A connected NetworkX graph.
    """
    topologies = ['ba', 'ws', 'sbm', 'grid', 'tree', 'ladder', 'caveman', 'er']
    if topology is None:
        topology = random.choice(topologies)

    if topology == 'ba':
        m = random.randint(1, min(3, n_nodes - 1))
        G = generate_ba_graph(n_nodes, m)
    elif topology == 'ws':
        k = random.choice([4, 6])
        k = min(k, n_nodes - 1)
        if k % 2 == 1:
            k = max(k - 1, 2)
        p = random.uniform(0.1, 0.5)
        G = generate_ws_graph(n_nodes, k, p)
    elif topology == 'sbm':
        num_communities = random.randint(2, min(4, n_nodes // 3))
        base_size = n_nodes // num_communities
        sizes = [base_size] * num_communities
        sizes[-1] += n_nodes - sum(sizes)  # absorb remainder
        G = generate_sbm_graph(sizes,
                               p_within=random.uniform(0.3, 0.6),
                               p_between=random.uniform(0.02, 0.1))
    elif topology == 'grid':
        import math
        sqrt = int(math.sqrt(n_nodes))
        rows = max(sqrt, 2)
        cols = max(n_nodes // rows, 2)
        G = generate_grid_graph(rows, cols)
    elif topology == 'tree':
        G = generate_tree_graph(max(n_nodes, 3))
    elif topology == 'ladder':
        half = max(n_nodes // 2, 2)
        G = generate_ladder_graph(half)
    elif topology == 'caveman':
        clique_size = random.randint(3, min(6, n_nodes // 2))
        num_cliques = max(n_nodes // clique_size, 2)
        G = generate_caveman_graph(num_cliques, clique_size)
    elif topology == 'er':
        p = random.uniform(0.1, 0.4)
        G = generate_er_graph(n_nodes, p)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    _ensure_connected(G)
    assert nx.is_connected(G), f"Graph not connected for topology={topology}"
    return G
