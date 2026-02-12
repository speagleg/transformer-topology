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
) -> tuple[CellComplex, int, int, int]:
    """Generate a propagation delay task.

    Creates a random graph where each edge has a delay (1, 2, or 3).
    The answer is the shortest weighted path length from source to target.
    If unreachable, the answer is max_delay - 1 (sentinel class).

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        max_delay: Number of output classes (max answer = max_delay - 1).
        edge_prob: Probability of an edge between any two nodes.

    Returns:
        (cc, source, target, answer) tuple.
    """
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
                delay = random.randint(1, 3)
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


def generate_blocking_task(
    n_nodes: int,
    embedding_dim: int,
    max_delay: int = 10,
    edge_prob: float = 0.3,
) -> tuple[CellComplex, int, int, int]:
    """Generate a blocking propagation task.

    Same as propagation delay, but one intermediate node is blocked.
    Signal cannot pass through blocked nodes.

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        max_delay: Number of output classes (max answer = max_delay - 1).
        edge_prob: Probability of an edge between any two nodes.

    Returns:
        (cc, source, target, answer) tuple.
    """
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
                delay = random.randint(1, 3)
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


def generate_interference_task(
    n_nodes: int,
    embedding_dim: int,
    edge_prob: float = 0.3,
) -> tuple[CellComplex, int, int, int]:
    """Generate an interference detection task (binary).

    Two sources fire simultaneously toward a target node.
    - Answer = 1 (constructive) if the difference in shortest path lengths is even.
    - Answer = 0 (destructive) if the difference is odd.
    - If either source cannot reach the target, answer = 0.

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        edge_prob: Probability of an edge between any two nodes.

    Returns:
        (cc, source1, target, answer) tuple.
        source1 is the first source node (query node).
    """
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
                delay = random.randint(1, 3)
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
    ):
        self.task_type = task_type
        self.samples: list[tuple[CellComplex, int, int, int]] = []

        for _ in range(num_samples):
            if task_type == "propagation_delay":
                sample = generate_propagation_delay_task(
                    n_nodes=n_nodes,
                    embedding_dim=embedding_dim,
                    max_delay=max_delay,
                )
            elif task_type == "blocking":
                sample = generate_blocking_task(
                    n_nodes=n_nodes,
                    embedding_dim=embedding_dim,
                    max_delay=max_delay,
                )
            elif task_type == "interference":
                sample = generate_interference_task(
                    n_nodes=n_nodes,
                    embedding_dim=embedding_dim,
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[CellComplex, int, int, int]:
        return self.samples[idx]
