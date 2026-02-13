"""Phase 2 topological reasoning benchmarks.

Three task types:
- Cycle detection: does the graph contain a cycle? (binary)
- Path counting: how many edge-disjoint paths exist between two nodes?
- Betti number prediction: what is beta_1 of the complex?

Each task supports two modes:
- Original (hand-crafted graphs): backward-compatible, controlled structure
- Diverse topology mode: uses random_graph() + nx_to_cell_complex() for
  structural embeddings and varied graph topologies
"""

import torch
import random
import networkx as nx
from src.cell_complex.cell_complex import CellComplex


def auto_fill_triangles(cc: CellComplex) -> int:
    """Detect triangles in the graph and add 2-cells for each.

    Returns the number of 2-cells added.
    """
    n = cc.num_cells(0)
    adj = cc.adjacency_matrix(0)
    # Build edge lookup: (min_node, max_node) -> edge_index
    edge_lookup: dict[tuple[int, int], int] = {}
    for e in range(cc.num_cells(1)):
        s, t = cc._1_cell_sources[e], cc._1_cell_targets[e]
        key = (min(s, t), max(s, t))
        if key not in edge_lookup:
            edge_lookup[key] = e

    added = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                continue
            for k in range(j + 1, n):
                if adj[i, k] > 0 and adj[j, k] > 0:
                    # Triangle (i, j, k)
                    e_ij = edge_lookup.get((i, j))
                    e_jk = edge_lookup.get((j, k))
                    e_ik = edge_lookup.get((i, k))
                    if e_ij is not None and e_jk is not None and e_ik is not None:
                        emb = torch.randn(cc.embedding_dim)
                        cc.add_2_cell([e_ij, e_jk, e_ik], emb, "triangle")
                        added += 1
    return added


def generate_cycle_detection_task(
    num_nodes: int, embedding_dim: int, has_cycle: bool
) -> tuple[CellComplex, int, int, int]:
    """Generate a cycle detection task.

    Args:
        num_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of embeddings.
        has_cycle: Whether to include a cycle.

    Returns:
        (cc, query_node, target_node, answer) where answer is 1 if cycle, 0 if not.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    nodes = [cc.add_0_cell(torch.randn(embedding_dim), "node") for _ in range(num_nodes)]

    if has_cycle and num_nodes >= 3:
        # Create a cycle of length 3-5
        cycle_len = min(random.randint(3, 5), num_nodes)
        cycle_nodes = random.sample(nodes, cycle_len)
        for i in range(cycle_len):
            cc.add_1_cell(cycle_nodes[i], cycle_nodes[(i + 1) % cycle_len],
                          torch.randn(embedding_dim), "cycle_edge")
        # Add some extra tree edges
        remaining = [n for n in nodes if n not in cycle_nodes]
        for n in remaining:
            parent = random.choice(cycle_nodes + [r for r in remaining if r < n] or cycle_nodes)
            cc.add_1_cell(parent, n, torch.randn(embedding_dim), "tree_edge")
    else:
        # Build a tree (no cycles)
        for i in range(1, num_nodes):
            parent = random.randint(0, i - 1)
            cc.add_1_cell(nodes[parent], nodes[i], torch.randn(embedding_dim), "tree_edge")

    auto_fill_triangles(cc)
    answer = 1 if has_cycle else 0
    return cc, nodes[0], nodes[-1], answer


def generate_path_counting_task(
    num_paths: int, path_length: int, embedding_dim: int
) -> tuple[CellComplex, int, int, int]:
    """Generate a path counting task with known number of edge-disjoint paths.

    Args:
        num_paths: Number of edge-disjoint paths to create.
        path_length: Length of each path.
        embedding_dim: Dimension of embeddings.

    Returns:
        (cc, source, target, answer) where answer is num_paths.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    source = cc.add_0_cell(torch.randn(embedding_dim), "source")
    target = cc.add_0_cell(torch.randn(embedding_dim), "target")

    for _ in range(num_paths):
        prev = source
        for step in range(path_length - 1):
            mid = cc.add_0_cell(torch.randn(embedding_dim), "intermediate")
            cc.add_1_cell(prev, mid, torch.randn(embedding_dim), "path_edge")
            prev = mid
        cc.add_1_cell(prev, target, torch.randn(embedding_dim), "path_edge")

    # Add distractor nodes
    all_nodes = list(range(cc.num_cells(0)))
    for _ in range(3):
        d = cc.add_0_cell(torch.randn(embedding_dim), "distractor")
        t = random.choice(all_nodes)
        cc.add_1_cell(d, t, torch.randn(embedding_dim), "distractor_edge")
        all_nodes.append(d)

    auto_fill_triangles(cc)
    return cc, source, target, num_paths


def generate_betti_task(
    target_beta1: int, embedding_dim: int
) -> tuple[CellComplex, int, int, int]:
    """Generate a complex with known first Betti number.

    beta_1 = num_independent_cycles = num_edges - num_nodes + num_connected_components

    Args:
        target_beta1: Target first Betti number.
        embedding_dim: Dimension of embeddings.

    Returns:
        (cc, node_0, node_1, answer) where answer is target_beta1.
    """
    cc = CellComplex(embedding_dim=embedding_dim)

    # Start with a spanning tree of (target_beta1 + 3) nodes
    num_nodes = target_beta1 + 3
    nodes = [cc.add_0_cell(torch.randn(embedding_dim), "node") for _ in range(num_nodes)]

    # Spanning tree: n-1 edges
    for i in range(1, num_nodes):
        parent = random.randint(0, i - 1)
        cc.add_1_cell(nodes[parent], nodes[i], torch.randn(embedding_dim), "tree_edge")

    # Add exactly target_beta1 extra edges to create independent cycles
    added = 0
    existing_edges = set()
    for e in range(cc.num_cells(1)):
        s, t = cc._1_cell_sources[e], cc._1_cell_targets[e]
        existing_edges.add((min(s, t), max(s, t)))

    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)
                      if (i, j) not in existing_edges]
    random.shuffle(possible_edges)

    for i, j in possible_edges:
        if added >= target_beta1:
            break
        cc.add_1_cell(nodes[i], nodes[j], torch.randn(embedding_dim), "cycle_edge")
        added += 1

    return cc, nodes[0], nodes[1], target_beta1


def generate_cycle_detection_task_diverse(
    n_nodes: int,
    embedding_dim: int,
    topologies: list[str] | None = None,
) -> tuple[CellComplex, int, int, int]:
    """Generate a cycle detection task using diverse graph topologies.

    Uses random_graph() + nx_to_cell_complex() for structural embeddings.
    Balances classes by mixing ~50% tree (no cycle) with ~50% other topologies.

    Args:
        n_nodes: Number of nodes.
        embedding_dim: Dimension of cell embeddings.
        topologies: List of allowed topology names, or None for all.

    Returns:
        (cc, source, target, answer) where answer is 1 if cycle, 0 if not.
    """
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex

    # Balance: 50% trees (no cycles), 50% other topologies (usually have cycles)
    force_tree = random.random() < 0.5

    if force_tree:
        topo = 'tree'
    elif topologies:
        non_tree = [t for t in topologies if t != 'tree']
        topo = random.choice(non_tree) if non_tree else random.choice(topologies)
    else:
        topo = random.choice(['ba', 'ws', 'sbm', 'grid', 'ladder', 'caveman', 'er'])

    G = random_graph(n_nodes, topology=topo)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx,
        target_node=target_nx,
    )

    has_cycle = len(nx.cycle_basis(G)) > 0
    answer = 1 if has_cycle else 0
    return cc, node_map[source_nx], node_map[target_nx], answer


def generate_betti_task_diverse(
    n_nodes: int,
    embedding_dim: int,
    max_beta: int = 5,
    topologies: list[str] | None = None,
) -> tuple[CellComplex, int, int, int]:
    """Generate a Betti number task using diverse graph topologies.

    Computes beta_1 = |E| - |V| + 1 (for connected graphs) from the
    actual graph structure.

    Args:
        n_nodes: Number of nodes.
        embedding_dim: Dimension of cell embeddings.
        max_beta: Maximum Betti number (answers capped here).
        topologies: List of allowed topology names, or None for all.

    Returns:
        (cc, source, target, answer) where answer is beta_1.
    """
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex

    topo = random.choice(topologies) if topologies else None
    G = random_graph(n_nodes, topology=topo)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx,
        target_node=target_nx,
    )

    # beta_1 = |E| - |V| + num_connected_components (for connected: +1)
    beta_1 = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)
    answer = min(beta_1, max_beta)
    return cc, node_map[source_nx], node_map[target_nx], answer


def generate_path_counting_task_diverse(
    n_nodes: int,
    embedding_dim: int,
    max_paths: int = 4,
    topologies: list[str] | None = None,
) -> tuple[CellComplex, int, int, int]:
    """Generate a path counting task using diverse graph topologies.

    Uses edge_connectivity as the answer (number of edge-disjoint paths).

    Args:
        n_nodes: Number of nodes.
        embedding_dim: Dimension of cell embeddings.
        max_paths: Maximum path count (answers capped here).
        topologies: List of allowed topology names, or None for all.

    Returns:
        (cc, source, target, answer) where answer is edge connectivity.
    """
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex

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
        connectivity = nx.edge_connectivity(G, source_nx, target_nx)
    except nx.NetworkXError:
        connectivity = 0
    answer = min(connectivity, max_paths)
    return cc, node_map[source_nx], node_map[target_nx], answer


class TopologicalDataset:
    """Dataset of topological reasoning benchmark samples.

    Supports three task types: cycle_detection, path_counting, betti_number.
    When use_diverse_topology=True, uses random graph generators with structural
    embeddings instead of hand-crafted graphs.
    """

    def __init__(self, num_samples: int, task_type: str, embedding_dim: int,
                 use_diverse_topology: bool = False,
                 topologies: list[str] | None = None,
                 n_nodes: int = 10,
                 **kwargs):
        self.samples: list[tuple[CellComplex, int, int, int]] = []
        self._indices: list[int] | None = None

        for _ in range(num_samples):
            if use_diverse_topology:
                sample = self._generate_diverse(
                    task_type, n_nodes, embedding_dim, topologies, **kwargs,
                )
            else:
                sample = self._generate_original(
                    task_type, embedding_dim, n_nodes, **kwargs,
                )
            self.samples.append(sample)

        self.shuffle()

    @staticmethod
    def _generate_diverse(task_type, n_nodes, embedding_dim, topologies, **kwargs):
        if task_type == "cycle_detection":
            return generate_cycle_detection_task_diverse(
                n_nodes, embedding_dim, topologies=topologies,
            )
        elif task_type == "path_counting":
            max_paths = kwargs.get("max_paths", 4)
            return generate_path_counting_task_diverse(
                n_nodes, embedding_dim, max_paths=max_paths, topologies=topologies,
            )
        elif task_type == "betti_number":
            max_beta = kwargs.get("max_beta", 5)
            return generate_betti_task_diverse(
                n_nodes, embedding_dim, max_beta=max_beta, topologies=topologies,
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @staticmethod
    def _generate_original(task_type, embedding_dim, n_nodes, **kwargs):
        if task_type == "cycle_detection":
            num_nodes = kwargs.get("num_nodes", random.randint(5, 15))
            has_cycle = random.choice([True, False])
            return generate_cycle_detection_task(num_nodes, embedding_dim, has_cycle)
        elif task_type == "path_counting":
            num_paths = kwargs.get("num_paths", random.randint(1, 4))
            path_length = kwargs.get("path_length", random.randint(2, 4))
            return generate_path_counting_task(num_paths, path_length, embedding_dim)
        elif task_type == "betti_number":
            max_beta = kwargs.get("max_beta", 5)
            target = random.randint(0, max_beta)
            return generate_betti_task(target, embedding_dim)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def shuffle(self):
        """Shuffle sample access order for the next epoch."""
        self._indices = list(range(len(self.samples)))
        random.shuffle(self._indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[CellComplex, int, int, int]:
        if self._indices is not None:
            return self.samples[self._indices[idx]]
        return self.samples[idx]
