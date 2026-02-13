"""Phase 3 temporal/causal reasoning benchmarks.

Three task types:
- Propagation delay: shortest weighted path from source to target
- Blocking propagation: shortest path avoiding a blocked node
- Interference detection: constructive vs destructive interference from two sources
"""

import heapq
import random

import torch

from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.topological_tasks import auto_fill_triangles


def _dijkstra(
    adj: dict[int, list[tuple[int, int]]],
    source: int,
    blocked: set[int] | None = None,
) -> dict[int, int]:
    """Run Dijkstra's algorithm on a weighted adjacency list.

    Args:
        adj: Adjacency list mapping node -> [(neighbor, weight), ...].
        source: Starting node.
        blocked: Set of node indices that cannot be traversed.

    Returns:
        Dict mapping reachable node -> shortest distance from source.
    """
    if blocked is None:
        blocked = set()

    dist: dict[int, int] = {source: 0}
    heap: list[tuple[int, int]] = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            if v in blocked:
                continue
            new_dist = d + w
            if new_dist < dist.get(v, float("inf")):
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist


def _build_adjacency(cc: CellComplex) -> dict[int, list[tuple[int, int]]]:
    """Build a weighted adjacency list from a CellComplex.

    Edge weight (delay) is stored as the integer value in dimension 0
    of the edge embedding.

    Returns:
        Adjacency list mapping node -> [(neighbor, delay), ...].
    """
    adj: dict[int, list[tuple[int, int]]] = {
        i: [] for i in range(cc.num_cells(0))
    }
    for e in range(cc.num_cells(1)):
        s = cc._1_cell_sources[e]
        t = cc._1_cell_targets[e]
        delay = int(cc._1_cell_embeddings[e][0].item())
        adj[s].append((t, delay))
        adj[t].append((s, delay))
    return adj


def generate_propagation_delay_task(
    n_nodes: int,
    embedding_dim: int,
    max_delay: int = 10,
    edge_prob: float = 0.3,
    max_edge_delay: int = 3,
    use_diverse_topology: bool = False,
) -> tuple[CellComplex, int, int, int]:
    """Generate a propagation delay task.

    Creates a graph where each edge has a delay. The answer is the shortest
    weighted path length from source to target.
    If unreachable, the answer is max_delay - 1 (sentinel class).

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        max_delay: Number of output classes (max answer = max_delay - 1).
        edge_prob: Probability of an edge (only used for legacy ER graphs).
        max_edge_delay: Maximum delay per edge (default 3, was hardcoded).
        use_diverse_topology: If True, use random_graph + nx_to_cell_complex.

    Returns:
        (cc, source, target, answer) tuple.
    """
    if use_diverse_topology:
        return _generate_diverse_propagation(
            n_nodes, embedding_dim, max_delay, max_edge_delay)

    cc = CellComplex(embedding_dim=embedding_dim)
    nodes = [cc.add_0_cell(torch.randn(embedding_dim), "node") for _ in range(n_nodes)]

    # Mark source and target with distinct cell types
    source, target = random.sample(nodes, 2)
    cc._0_cell_types[source] = "source"
    cc._0_cell_types[target] = "target"

    # Add edges with random delays
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                delay = random.randint(1, max_edge_delay)
                emb = torch.randn(embedding_dim)
                emb[0] = float(delay)
                cc.add_1_cell(nodes[i], nodes[j], emb, "temporal_edge")

    # Compute ground truth via Dijkstra
    adj = _build_adjacency(cc)
    dist = _dijkstra(adj, source)
    if target in dist and dist[target] < max_delay:
        answer = dist[target]
    else:
        answer = max_delay - 1

    # Fill 2-cells from detected triangles for higher-order processing
    auto_fill_triangles(cc)

    return cc, source, target, answer


def _generate_diverse_propagation(
    n_nodes: int,
    embedding_dim: int,
    max_delay: int,
    max_edge_delay: int,
) -> tuple[CellComplex, int, int, int]:
    """Generate propagation delay task with diverse topologies."""
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex

    G = random_graph(n_nodes)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    # Assign random delays to edges
    edge_delays = {}
    for u, v in G.edges():
        edge_delays[(u, v)] = random.randint(1, max_edge_delay)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx,
        target_node=target_nx,
        edge_delays=edge_delays,
    )

    # Compute ground truth via Dijkstra
    adj = _build_adjacency(cc)
    source_cc = node_map[source_nx]
    target_cc = node_map[target_nx]
    dist = _dijkstra(adj, source_cc)
    if target_cc in dist and dist[target_cc] < max_delay:
        answer = dist[target_cc]
    else:
        answer = max_delay - 1

    return cc, source_cc, target_cc, answer


def generate_blocking_task(
    n_nodes: int,
    embedding_dim: int,
    max_delay: int = 10,
    edge_prob: float = 0.3,
    max_edge_delay: int = 3,
    use_diverse_topology: bool = False,
) -> tuple[CellComplex, int, int, int]:
    """Generate a blocking propagation task.

    Same as propagation delay, but one intermediate node is blocked.
    Signal cannot pass through blocked nodes.

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        max_delay: Number of output classes (max answer = max_delay - 1).
        edge_prob: Probability of an edge (only used for legacy ER graphs).
        max_edge_delay: Maximum delay per edge.
        use_diverse_topology: If True, use random_graph + nx_to_cell_complex.

    Returns:
        (cc, source, target, answer) tuple.
    """
    if use_diverse_topology:
        return _generate_diverse_blocking(
            n_nodes, embedding_dim, max_delay, max_edge_delay)

    cc = CellComplex(embedding_dim=embedding_dim)
    nodes = [cc.add_0_cell(torch.randn(embedding_dim), "node") for _ in range(n_nodes)]

    source, target = random.sample(nodes, 2)
    cc._0_cell_types[source] = "source"
    cc._0_cell_types[target] = "target"

    # Pick a random intermediate node to block
    candidates = [n for n in nodes if n != source and n != target]
    blocked_node = random.choice(candidates) if candidates else None
    if blocked_node is not None:
        cc._0_cell_types[blocked_node] = "blocked"

    # Add edges with random delays
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                delay = random.randint(1, max_edge_delay)
                emb = torch.randn(embedding_dim)
                emb[0] = float(delay)
                cc.add_1_cell(nodes[i], nodes[j], emb, "temporal_edge")

    # Compute ground truth: shortest path avoiding blocked node
    adj = _build_adjacency(cc)
    blocked_set = {blocked_node} if blocked_node is not None else set()
    dist = _dijkstra(adj, source, blocked=blocked_set)
    if target in dist and dist[target] < max_delay:
        answer = dist[target]
    else:
        answer = max_delay - 1

    # Fill 2-cells from detected triangles for higher-order processing
    auto_fill_triangles(cc)

    return cc, source, target, answer


def _generate_diverse_blocking(
    n_nodes: int,
    embedding_dim: int,
    max_delay: int,
    max_edge_delay: int,
) -> tuple[CellComplex, int, int, int]:
    """Generate blocking task with diverse topologies."""
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex

    G = random_graph(n_nodes)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    # Pick a random intermediate node to block
    candidates = [n for n in nodes if n != source_nx and n != target_nx]
    blocked_nx = random.choice(candidates) if candidates else None

    # Assign random delays to edges
    edge_delays = {}
    for u, v in G.edges():
        edge_delays[(u, v)] = random.randint(1, max_edge_delay)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx,
        target_node=target_nx,
        blocked_nodes=[blocked_nx] if blocked_nx is not None else None,
        edge_delays=edge_delays,
    )

    # Compute ground truth: shortest path avoiding blocked node
    adj = _build_adjacency(cc)
    source_cc = node_map[source_nx]
    target_cc = node_map[target_nx]
    blocked_cc = {node_map[blocked_nx]} if blocked_nx is not None else set()
    dist = _dijkstra(adj, source_cc, blocked=blocked_cc)
    if target_cc in dist and dist[target_cc] < max_delay:
        answer = dist[target_cc]
    else:
        answer = max_delay - 1

    return cc, source_cc, target_cc, answer


def generate_interference_task(
    n_nodes: int,
    embedding_dim: int,
    edge_prob: float = 0.3,
    max_edge_delay: int = 3,
    use_diverse_topology: bool = False,
) -> tuple[CellComplex, int, int, int]:
    """Generate an interference detection task (binary).

    Two sources fire simultaneously toward a target node.
    - Answer = 1 (constructive) if the difference in shortest path lengths is even.
    - Answer = 0 (destructive) if the difference is odd.
    - If either source cannot reach the target, answer = 0.

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        edge_prob: Probability of an edge (only used for legacy ER graphs).
        max_edge_delay: Maximum delay per edge.
        use_diverse_topology: If True, use random_graph + nx_to_cell_complex.

    Returns:
        (cc, source1, target, answer) tuple.
        source1 is the first source node (query node).
    """
    if use_diverse_topology:
        return _generate_diverse_interference(
            n_nodes, embedding_dim, max_edge_delay)

    cc = CellComplex(embedding_dim=embedding_dim)
    nodes = [cc.add_0_cell(torch.randn(embedding_dim), "node") for _ in range(n_nodes)]

    # Pick 2 sources and 1 target (all distinct)
    chosen = random.sample(nodes, 3)
    source1, source2, target = chosen[0], chosen[1], chosen[2]
    cc._0_cell_types[source1] = "source1"
    cc._0_cell_types[source2] = "source2"
    cc._0_cell_types[target] = "target"

    # Add edges with random delays
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                delay = random.randint(1, max_edge_delay)
                emb = torch.randn(embedding_dim)
                emb[0] = float(delay)
                cc.add_1_cell(nodes[i], nodes[j], emb, "temporal_edge")

    # Compute ground truth
    adj = _build_adjacency(cc)
    dist1 = _dijkstra(adj, source1)
    dist2 = _dijkstra(adj, source2)

    if target not in dist1 or target not in dist2:
        answer = 0  # unreachable -> destructive
    else:
        diff = abs(dist1[target] - dist2[target])
        answer = 1 if diff % 2 == 0 else 0

    # Fill 2-cells from detected triangles for higher-order processing
    auto_fill_triangles(cc)

    return cc, source1, target, answer


def _generate_diverse_interference(
    n_nodes: int,
    embedding_dim: int,
    max_edge_delay: int,
) -> tuple[CellComplex, int, int, int]:
    """Generate interference task with diverse topologies."""
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex

    G = random_graph(n_nodes)
    nodes = list(G.nodes())
    source1_nx, source2_nx, target_nx = random.sample(nodes, 3)

    # Assign random delays to edges
    edge_delays = {}
    for u, v in G.edges():
        edge_delays[(u, v)] = random.randint(1, max_edge_delay)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source1_nx,
        target_node=target_nx,
        source2_node=source2_nx,
        edge_delays=edge_delays,
    )

    # Compute ground truth
    adj = _build_adjacency(cc)
    source1_cc = node_map[source1_nx]
    source2_cc = node_map[source2_nx]
    target_cc = node_map[target_nx]

    dist1 = _dijkstra(adj, source1_cc)
    dist2 = _dijkstra(adj, source2_cc)

    if target_cc not in dist1 or target_cc not in dist2:
        answer = 0
    else:
        diff = abs(dist1[target_cc] - dist2[target_cc])
        answer = 1 if diff % 2 == 0 else 0

    return cc, source1_cc, target_cc, answer


class TemporalDataset:
    """Dataset of temporal/causal reasoning benchmark samples.

    Supports three task types: propagation_delay, blocking, interference.
    """

    def __init__(
        self,
        num_samples: int,
        task_type: str,
        embedding_dim: int,
        n_nodes: int = 8,
        max_delay: int = 10,
        max_edge_delay: int = 3,
        use_diverse_topology: bool = False,
    ):
        self.task_type = task_type
        self.samples: list[tuple[CellComplex, int, int, int]] = []
        self._indices: list[int] | None = None

        for _ in range(num_samples):
            if task_type == "propagation_delay":
                sample = generate_propagation_delay_task(
                    n_nodes=n_nodes,
                    embedding_dim=embedding_dim,
                    max_delay=max_delay,
                    max_edge_delay=max_edge_delay,
                    use_diverse_topology=use_diverse_topology,
                )
            elif task_type == "blocking":
                sample = generate_blocking_task(
                    n_nodes=n_nodes,
                    embedding_dim=embedding_dim,
                    max_delay=max_delay,
                    max_edge_delay=max_edge_delay,
                    use_diverse_topology=use_diverse_topology,
                )
            elif task_type == "interference":
                sample = generate_interference_task(
                    n_nodes=n_nodes,
                    embedding_dim=embedding_dim,
                    max_edge_delay=max_edge_delay,
                    use_diverse_topology=use_diverse_topology,
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            self.samples.append(sample)

        # Initial shuffle
        self.shuffle()

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

    def save(self, path: str):
        """Save dataset samples to disk."""
        torch.save(self.samples, path)

    @classmethod
    def load(cls, path: str) -> 'TemporalDataset':
        """Load dataset samples from disk."""
        obj = cls.__new__(cls)
        obj.samples = torch.load(path, weights_only=False)
        obj._indices = None
        obj.task_type = "loaded"
        return obj
