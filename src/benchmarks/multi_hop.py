import torch
import random
from src.cell_complex.cell_complex import CellComplex


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
    return cc, root, target, depth


class MultiHopDataset:
    """Dataset of multi-hop graph traversal benchmark samples.

    Each sample is a (CellComplex, query_node, target_node, answer) tuple.
    """

    def __init__(self, num_samples: int, min_hops: int, max_hops: int,
                 num_distractors: int, embedding_dim: int, task_type: str = "chain"):
        self.samples = []
        for _ in range(num_samples):
            num_hops = random.randint(min_hops, max_hops)
            if task_type == "chain":
                sample = generate_chain_task(num_hops, num_distractors, embedding_dim)
            elif task_type == "tree":
                sample = generate_tree_task(num_hops, 2, num_distractors, embedding_dim)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[CellComplex, int, int, int]:
        return self.samples[idx]
