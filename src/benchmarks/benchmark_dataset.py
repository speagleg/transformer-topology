"""Unified benchmark dataset with topology and size control.

Wraps any task generator with consistent topology/size control,
supporting the full benchmark suite evaluation axes:
- In-distribution (ID)
- Topology transfer (OOD-Topo)
- Size generalization (OOD-Size)
"""

import random

import torch

from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.multi_hop import generate_diverse_task
from src.benchmarks.temporal_tasks import (
    generate_propagation_delay_task,
    generate_blocking_task,
    generate_interference_task,
)
from src.benchmarks.topological_tasks import (
    generate_cycle_detection_task_diverse,
    generate_path_counting_task_diverse,
    generate_betti_task_diverse,
)
from src.benchmarks.clrs_tasks import generate_bfs_task, generate_dijkstra_task
from src.benchmarks.spectral_tasks import generate_spectral_gap_task, generate_hodge_class_task
from src.benchmarks.llm_tasks import (
    generate_graph_completion_task,
    generate_labeled_reasoning_task,
    generate_analogical_transfer_task,
)


def _wrap_diverse(n_nodes, embedding_dim, topologies, **kwargs):
    """Wrapper for generate_diverse_task with topology control."""
    max_hops = kwargs.get('max_hops', 10)
    target_hops = random.randint(kwargs.get('min_hops', 2), max_hops)
    topo = random.choice(topologies) if topologies else None
    return generate_diverse_task(
        target_hops=target_hops,
        embedding_dim=embedding_dim,
        n_nodes_range=(n_nodes, n_nodes),
        topology=topo,
    )


def _wrap_propagation_delay(n_nodes, embedding_dim, topologies, **kwargs):
    """Wrapper for propagation_delay with topology control."""
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex
    from src.benchmarks.temporal_tasks import _dijkstra, _build_adjacency

    max_delay = kwargs.get('max_delay', 10)
    max_edge_delay = kwargs.get('max_edge_delay', 5)

    topo = random.choice(topologies) if topologies else None
    G = random_graph(n_nodes, topology=topo)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)

    edge_delays = {(u, v): random.randint(1, max_edge_delay) for u, v in G.edges()}
    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx, target_node=target_nx,
        edge_delays=edge_delays,
    )

    adj = _build_adjacency(cc)
    src_cc = node_map[source_nx]
    tgt_cc = node_map[target_nx]
    dist = _dijkstra(adj, src_cc)
    answer = min(dist.get(tgt_cc, max_delay - 1), max_delay - 1)
    return cc, src_cc, tgt_cc, answer


def _wrap_blocking(n_nodes, embedding_dim, topologies, **kwargs):
    """Wrapper for blocking task with topology control."""
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex
    from src.benchmarks.temporal_tasks import _dijkstra, _build_adjacency

    max_delay = kwargs.get('max_delay', 10)
    max_edge_delay = kwargs.get('max_edge_delay', 5)

    topo = random.choice(topologies) if topologies else None
    G = random_graph(n_nodes, topology=topo)
    nodes = list(G.nodes())
    source_nx, target_nx = random.sample(nodes, 2)
    others = [n for n in nodes if n not in (source_nx, target_nx)]
    blocked_nx = random.choice(others) if others else None

    edge_delays = {(u, v): random.randint(1, max_edge_delay) for u, v in G.edges()}
    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx, target_node=target_nx,
        blocked_nodes=[blocked_nx] if blocked_nx is not None else None,
        edge_delays=edge_delays,
    )

    adj = _build_adjacency(cc)
    src_cc = node_map[source_nx]
    tgt_cc = node_map[target_nx]
    blocked_cc = {node_map[blocked_nx]} if blocked_nx is not None else set()
    dist = _dijkstra(adj, src_cc, blocked=blocked_cc)
    answer = min(dist.get(tgt_cc, max_delay - 1), max_delay - 1)
    return cc, src_cc, tgt_cc, answer


def _wrap_interference(n_nodes, embedding_dim, topologies, **kwargs):
    """Wrapper for interference task with topology control."""
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex
    from src.benchmarks.temporal_tasks import _dijkstra, _build_adjacency

    topo = random.choice(topologies) if topologies else None
    G = random_graph(n_nodes, topology=topo)
    nodes = list(G.nodes())
    if len(nodes) < 3:
        # Fallback
        return generate_interference_task(
            n_nodes=n_nodes, embedding_dim=embedding_dim, max_edge_delay=5,
        )
    source_nx, source2_nx, target_nx = random.sample(nodes, 3)

    max_edge_delay = kwargs.get('max_edge_delay', 5)
    edge_delays = {(u, v): random.randint(1, max_edge_delay) for u, v in G.edges()}
    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source_nx, target_node=target_nx,
        source2_node=source2_nx,
        edge_delays=edge_delays,
    )

    adj = _build_adjacency(cc)
    d1 = _dijkstra(adj, node_map[source_nx])
    d2 = _dijkstra(adj, node_map[source2_nx])
    tgt_cc = node_map[target_nx]
    dist1 = d1.get(tgt_cc, 0)
    dist2 = d2.get(tgt_cc, 0)
    diff = abs(dist1 - dist2)
    answer = 0 if diff % 2 == 0 else 1
    return cc, node_map[source_nx], tgt_cc, answer


def _wrap_cycle_detection(n_nodes, embedding_dim, topologies, **kwargs):
    return generate_cycle_detection_task_diverse(n_nodes, embedding_dim, topologies=topologies)


def _wrap_path_counting(n_nodes, embedding_dim, topologies, **kwargs):
    max_paths = kwargs.get('max_paths', 4)
    return generate_path_counting_task_diverse(
        n_nodes, embedding_dim, max_paths=max_paths, topologies=topologies,
    )


def _wrap_betti(n_nodes, embedding_dim, topologies, **kwargs):
    max_beta = kwargs.get('max_beta', 5)
    return generate_betti_task_diverse(
        n_nodes, embedding_dim, max_beta=max_beta, topologies=topologies,
    )


def _wrap_bfs(n_nodes, embedding_dim, topologies, **kwargs):
    max_distance = kwargs.get('max_distance', 16)
    return generate_bfs_task(n_nodes, embedding_dim, max_distance=max_distance)


def _wrap_dijkstra(n_nodes, embedding_dim, topologies, **kwargs):
    max_distance = kwargs.get('max_distance', 16)
    max_edge_weight = kwargs.get('max_edge_weight', 5)
    return generate_dijkstra_task(
        n_nodes, embedding_dim,
        max_distance=max_distance, max_edge_weight=max_edge_weight,
    )


def _wrap_spectral_gap(n_nodes, embedding_dim, topologies, **kwargs):
    num_buckets = kwargs.get('num_buckets', 8)
    n_nodes_range = kwargs.get('n_nodes_range')
    return generate_spectral_gap_task(
        n_nodes, embedding_dim, num_buckets=num_buckets, topologies=topologies,
        n_nodes_range=n_nodes_range,
    )


def _wrap_hodge_class(n_nodes, embedding_dim, topologies, **kwargs):
    return generate_hodge_class_task(n_nodes, embedding_dim, topologies=topologies)


def _wrap_graph_completion(n_nodes, embedding_dim, topologies, **kwargs):
    return generate_graph_completion_task(n_nodes, embedding_dim, topologies=topologies)


def _wrap_labeled_reasoning(n_nodes, embedding_dim, topologies, **kwargs):
    return generate_labeled_reasoning_task(n_nodes, embedding_dim, topologies=topologies)


def _wrap_analogical_transfer(n_nodes, embedding_dim, topologies, **kwargs):
    return generate_analogical_transfer_task(n_nodes, embedding_dim, topologies=topologies)


# Task registry: task_type -> (generator_wrapper, max_classes, default_kwargs)
TASK_REGISTRY: dict[str, tuple[callable, int, dict]] = {
    "diverse":            (_wrap_diverse, 11, {"min_hops": 2, "max_hops": 10}),
    "propagation_delay":  (_wrap_propagation_delay, 10, {"max_delay": 10, "max_edge_delay": 5}),
    "blocking":           (_wrap_blocking, 10, {"max_delay": 10, "max_edge_delay": 5}),
    "interference":       (_wrap_interference, 2, {"max_edge_delay": 5}),
    "cycle_detection":    (_wrap_cycle_detection, 2, {}),
    "path_counting":      (_wrap_path_counting, 5, {"max_paths": 4}),
    "betti_number":       (_wrap_betti, 6, {"max_beta": 5}),
    "bfs":                (_wrap_bfs, 16, {"max_distance": 16}),
    "dijkstra":           (_wrap_dijkstra, 16, {"max_distance": 16, "max_edge_weight": 5}),
    "spectral_gap":       (_wrap_spectral_gap, 8, {"num_buckets": 8}),
    "hodge_class":          (_wrap_hodge_class, 3, {}),
    "graph_completion":     (_wrap_graph_completion, 2, {}),
    "labeled_reasoning":    (_wrap_labeled_reasoning, 3, {}),
    "analogical_transfer":  (_wrap_analogical_transfer, 5, {}),
}


class BenchmarkDataset:
    """Unified dataset for the benchmark suite.

    Wraps any task generator with consistent topology/size control.
    Supports save/load and epoch shuffling.

    When ``n_nodes_range`` is provided each sample draws a random node count
    from ``[range[0], range[1]]`` (inclusive), enabling mixed-size training
    for better size generalization.
    """

    def __init__(
        self,
        num_samples: int,
        task_type: str,
        n_nodes: int,
        embedding_dim: int,
        topologies: list[str] | None = None,
        n_nodes_range: tuple[int, int] | None = None,
        **task_kwargs,
    ):
        if task_type not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )

        self.task_type = task_type
        self.n_nodes = n_nodes
        self.n_nodes_range = n_nodes_range
        self.samples: list[tuple] = []  # 4-tuple or 5-tuple (with metadata)
        self._indices: list[int] | None = None

        generator, _, defaults = TASK_REGISTRY[task_type]
        kwargs = {**defaults, **task_kwargs}
        if n_nodes_range is not None:
            kwargs['n_nodes_range'] = n_nodes_range

        for i in range(num_samples):
            if n_nodes_range is not None:
                sample_n = random.randint(n_nodes_range[0], n_nodes_range[1])
            else:
                sample_n = n_nodes
            sample = generator(sample_n, embedding_dim, topologies, **kwargs)
            self.samples.append(sample)
            if (i + 1) % 100 == 0:
                size_str = f"n={n_nodes_range[0]}-{n_nodes_range[1]}" if n_nodes_range else f"n={n_nodes}"
                print(f"    Generated {i + 1}/{num_samples} {task_type} samples ({size_str})")

        self.shuffle()

    def shuffle(self):
        """Shuffle sample access order for the next epoch."""
        self._indices = list(range(len(self.samples)))
        random.shuffle(self._indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        if self._indices is not None:
            return self.samples[self._indices[idx]]
        return self.samples[idx]

    def save(self, path: str):
        """Save dataset to disk."""
        torch.save({
            "samples": self.samples,
            "task_type": self.task_type,
            "n_nodes": self.n_nodes,
            "n_nodes_range": self.n_nodes_range,
        }, path)

    @classmethod
    def load(cls, path: str) -> "BenchmarkDataset":
        """Load dataset from disk."""
        data = torch.load(path, weights_only=False)
        obj = cls.__new__(cls)
        obj.samples = data["samples"]
        obj.task_type = data["task_type"]
        obj.n_nodes = data["n_nodes"]
        obj.n_nodes_range = data.get("n_nodes_range")
        obj._indices = None
        return obj


def get_max_classes(task_type: str) -> int:
    """Return the number of output classes for a task type."""
    if task_type not in TASK_REGISTRY:
        raise ValueError(f"Unknown task type: {task_type}")
    return TASK_REGISTRY[task_type][1]
