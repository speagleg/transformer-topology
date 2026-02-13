"""CLRS-comparable benchmark tasks: BFS and Dijkstra on ER graphs.

Generates tasks matching the CLRS Algorithmic Reasoning Benchmark
distribution (Erdős-Rényi graphs, avg degree ~4). Train on small graphs
(n=16), test on larger (n=32, n=64) to measure size generalization.
"""

import random

import networkx as nx
import torch

from src.benchmarks.graph_generators import clrs_er_graph
from src.benchmarks.graph_convert import nx_to_cell_complex
from src.benchmarks.temporal_tasks import _dijkstra, _build_adjacency
from src.cell_complex.cell_complex import CellComplex


def generate_bfs_task(
    n_nodes: int,
    embedding_dim: int,
    max_distance: int = 16,
) -> tuple[CellComplex, int, int, int]:
    """Generate a BFS shortest-path distance task on an ER graph.

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        max_distance: Number of output classes (answer capped at max_distance - 1).

    Returns:
        (cell_complex, source, target, answer) tuple.
    """
    G = clrs_er_graph(n_nodes)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx,
        target_node=target_nx,
    )

    # BFS shortest path
    dist = nx.shortest_path_length(G, source_nx, target_nx)
    answer = min(dist, max_distance - 1)

    return cc, node_map[source_nx], node_map[target_nx], answer


def generate_dijkstra_task(
    n_nodes: int,
    embedding_dim: int,
    max_distance: int = 16,
    max_edge_weight: int = 5,
) -> tuple[CellComplex, int, int, int]:
    """Generate a Dijkstra shortest-path distance task on a weighted ER graph.

    Args:
        n_nodes: Number of nodes in the graph.
        embedding_dim: Dimension of cell embeddings.
        max_distance: Number of output classes (answer capped at max_distance - 1).
        max_edge_weight: Maximum integer edge weight.

    Returns:
        (cell_complex, source, target, answer) tuple.
    """
    G = clrs_er_graph(n_nodes)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    # Assign random weights
    edge_delays = {}
    for u, v in G.edges():
        edge_delays[(u, v)] = random.randint(1, max_edge_weight)

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx,
        target_node=target_nx,
        edge_delays=edge_delays,
    )

    # Dijkstra shortest path via existing infrastructure
    adj = _build_adjacency(cc)
    source_cc = node_map[source_nx]
    target_cc = node_map[target_nx]
    dist = _dijkstra(adj, source_cc)

    if target_cc in dist and dist[target_cc] < max_distance:
        answer = dist[target_cc]
    else:
        answer = max_distance - 1

    return cc, source_cc, target_cc, answer


class CLRSDataset:
    """CLRS-comparable benchmark dataset.

    Generates ER graphs of a fixed size with BFS or Dijkstra tasks.
    Supports save/load for reproducibility.
    """

    def __init__(
        self,
        num_samples: int,
        task_type: str,
        n_nodes: int,
        embedding_dim: int,
        max_distance: int = 16,
        max_edge_weight: int = 5,
    ):
        self.task_type = task_type
        self.n_nodes = n_nodes
        self.samples: list[tuple[CellComplex, int, int, int]] = []
        self._indices: list[int] | None = None

        for i in range(num_samples):
            if task_type == "bfs":
                sample = generate_bfs_task(
                    n_nodes=n_nodes,
                    embedding_dim=embedding_dim,
                    max_distance=max_distance,
                )
            elif task_type == "dijkstra":
                sample = generate_dijkstra_task(
                    n_nodes=n_nodes,
                    embedding_dim=embedding_dim,
                    max_distance=max_distance,
                    max_edge_weight=max_edge_weight,
                )
            else:
                raise ValueError(f"Unknown CLRS task type: {task_type}")
            self.samples.append(sample)
            if (i + 1) % 100 == 0:
                print(f"    Generated {i + 1}/{num_samples} {task_type} samples (n={n_nodes})")

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
        torch.save({
            "samples": self.samples,
            "task_type": self.task_type,
            "n_nodes": self.n_nodes,
        }, path)

    @classmethod
    def load(cls, path: str) -> "CLRSDataset":
        """Load dataset samples from disk."""
        data = torch.load(path, weights_only=False)
        obj = cls.__new__(cls)
        obj.samples = data["samples"]
        obj.task_type = data["task_type"]
        obj.n_nodes = data["n_nodes"]
        obj._indices = None
        return obj
