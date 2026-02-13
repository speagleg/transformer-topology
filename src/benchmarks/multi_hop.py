import torch
import random
import networkx as nx
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.topological_tasks import auto_fill_triangles


def generate_chain_task(num_hops: int, num_distractors: int, embedding_dim: int
) -> tuple[CellComplex, int, int, int]:
    """Generate a chain-traversal benchmark task.

    Creates a linear chain of (num_hops + 1) nodes connected by num_hops edges,
    plus distractor nodes with random connections to make the task non-trivial.

    Returns:
        (cell_complex, query_node, target_node, answer) where answer is num_hops.
    """
    cc = CellComplex(embedding_dim=embedding_dim)

    # Build the main chain
    chain_nodes = []
    for i in range(num_hops + 1):
        emb = torch.randn(embedding_dim)
        emb[0] = float(i)
        node = cc.add_0_cell(emb, "chain")
        chain_nodes.append(node)

    for i in range(num_hops):
        emb = torch.randn(embedding_dim)
        cc.add_1_cell(chain_nodes[i], chain_nodes[i + 1], emb, "chain_edge")

    # Add distractor nodes with random edges
    distractor_nodes = []
    for _ in range(num_distractors):
        emb = torch.randn(embedding_dim)
        node = cc.add_0_cell(emb, "distractor")
        distractor_nodes.append(node)

    all_nodes = chain_nodes + distractor_nodes
    for d in distractor_nodes:
        num_connections = random.randint(1, min(3, len(all_nodes) - 1))
        targets = random.sample([n for n in all_nodes if n != d], num_connections)
        for t in targets:
            emb = torch.randn(embedding_dim)
            cc.add_1_cell(d, t, emb, "distractor_edge")

    auto_fill_triangles(cc)
    return cc, chain_nodes[0], chain_nodes[-1], num_hops


def generate_tree_task(depth: int, branching: int, num_distractors: int, embedding_dim: int
) -> tuple[CellComplex, int, int, int]:
    """Generate a tree-traversal benchmark task.

    Creates a balanced tree of given depth and branching factor, then adds
    distractor nodes. The task is to find a path from root to a random leaf.

    Returns:
        (cell_complex, root_node, target_leaf, answer) where answer is depth.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    root = cc.add_0_cell(torch.randn(embedding_dim), "root")

    current_level = [root]
    for d in range(depth):
        next_level = []
        for parent in current_level:
            for _ in range(branching):
                child = cc.add_0_cell(torch.randn(embedding_dim), "node")
                cc.add_1_cell(parent, child, torch.randn(embedding_dim), "tree_edge")
                next_level.append(child)
        current_level = next_level

    leaves = current_level
    all_nodes = list(range(cc.num_cells(0)))

    # Add distractor nodes
    for _ in range(num_distractors):
        d = cc.add_0_cell(torch.randn(embedding_dim), "distractor")
        targets = random.sample(all_nodes, min(2, len(all_nodes)))
        for t in targets:
            cc.add_1_cell(d, t, torch.randn(embedding_dim), "distractor_edge")
        all_nodes.append(d)

    target = random.choice(leaves)
    auto_fill_triangles(cc)
    return cc, root, target, depth


def generate_diverse_task(
    target_hops: int,
    embedding_dim: int,
    n_nodes_range: tuple[int, int] = (30, 80),
    max_retries: int = 20,
    topology: str | None = None,
) -> tuple[CellComplex, int, int, int]:
    """Generate a diverse-topology graph traversal task.

    Uses random graph topologies with structural embeddings.
    Finds a source-target pair at exactly target_hops distance.

    Args:
        target_hops: Desired shortest path length (the answer).
        embedding_dim: Embedding dimension (must be >= 4).
        n_nodes_range: (min_nodes, max_nodes) for graph size.
        max_retries: Max attempts to find a pair at the target distance.
        topology: Optional topology name to use (e.g. 'ba', 'er', 'grid').

    Returns:
        (cell_complex, source_node, target_node, answer) tuple.
    """
    from src.benchmarks.graph_generators import random_graph
    from src.benchmarks.graph_convert import nx_to_cell_complex

    for _ in range(max_retries):
        n_nodes = random.randint(*n_nodes_range)
        G = random_graph(n_nodes, topology=topology)

        # Find all pairs at the target distance using BFS from random starts
        nodes = list(G.nodes())
        random.shuffle(nodes)

        for source_nx in nodes[:min(10, len(nodes))]:
            lengths = nx.single_source_shortest_path_length(G, source_nx)
            candidates = [n for n, d in lengths.items() if d == target_hops]
            if candidates:
                target_nx = random.choice(candidates)
                cc, node_map = nx_to_cell_complex(
                    G, embedding_dim,
                    source_node=source_nx,
                    target_node=target_nx,
                )
                return cc, node_map[source_nx], node_map[target_nx], target_hops

    # Fallback: build a chain of exact length (guaranteed to work)
    return generate_chain_task(target_hops, num_distractors=10, embedding_dim=embedding_dim)


class MultiHopDataset:
    """Dataset of multi-hop graph traversal benchmark samples.

    Each sample is a (CellComplex, query_node, target_node, answer) tuple.

    Supports three task types:
    - "chain": linear chain graphs (legacy, backward compat)
    - "tree": balanced tree graphs (legacy, backward compat)
    - "diverse": random topologies with structural embeddings (recommended)
    """

    def __init__(self, num_samples: int, min_hops: int, max_hops: int,
                 num_distractors: int, embedding_dim: int, task_type: str = "chain",
                 n_nodes_range: tuple[int, int] = (30, 80)):
        self.samples: list[tuple[CellComplex, int, int, int]] = []
        self._indices: list[int] | None = None

        if task_type == "diverse":
            # Class-balanced sampling: equal samples per hop count
            hop_range = list(range(min_hops, max_hops + 1))
            samples_per_hop = num_samples // len(hop_range)
            remainder = num_samples % len(hop_range)

            for i, hop_count in enumerate(hop_range):
                count = samples_per_hop + (1 if i < remainder else 0)
                for _ in range(count):
                    sample = generate_diverse_task(
                        target_hops=hop_count,
                        embedding_dim=embedding_dim,
                        n_nodes_range=n_nodes_range,
                    )
                    self.samples.append(sample)
        else:
            for _ in range(num_samples):
                num_hops = random.randint(min_hops, max_hops)
                if task_type == "chain":
                    sample = generate_chain_task(num_hops, num_distractors, embedding_dim)
                elif task_type == "tree":
                    sample = generate_tree_task(num_hops, 2, num_distractors, embedding_dim)
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
    def load(cls, path: str) -> 'MultiHopDataset':
        """Load dataset samples from disk."""
        obj = cls.__new__(cls)
        obj.samples = torch.load(path, weights_only=False)
        obj._indices = None
        return obj
