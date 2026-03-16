"""Graph template factory for v12 metacognitive reasoning.

Each template returns a CellComplex whose topology reflects the structural
demands of a given problem type.  Placeholder zero embeddings are used; the
caller is expected to fill in meaningful embeddings before training.
"""

import torch

from src.cell_complex.cell_complex import CellComplex


def create_template(
    problem_type: str,
    embedding_dim: int = 128,
    num_steps: int = 6,
) -> CellComplex:
    """Create a CellComplex template for the given problem type.

    Args:
        problem_type: One of "SEQUENTIAL", "CONSTRAINT", "MULTI_HOP",
            "EXPLORATION", "VERIFICATION".  Unknown types fall back to
            SEQUENTIAL.
        embedding_dim: Dimensionality of all node/edge embeddings.
        num_steps: Controls the scale of the template (number of steps,
            depth, etc.; interpretation is per-template).

    Returns:
        A CellComplex with placeholder zero embeddings.
    """
    builders = {
        "SEQUENTIAL": _build_sequential,
        "CONSTRAINT": _build_constraint,
        "MULTI_HOP": _build_multi_hop,
        "EXPLORATION": _build_exploration,
        "VERIFICATION": _build_verification,
    }
    builder = builders.get(problem_type, _build_sequential)
    return builder(embedding_dim, num_steps)


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

def _zero(embedding_dim: int) -> torch.Tensor:
    return torch.zeros(embedding_dim)


def _build_sequential(embedding_dim: int, num_steps: int) -> CellComplex:
    """Linear chain: n nodes connected by n-1 directed edges.

    Represents problems where reasoning proceeds step-by-step.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    n = num_steps
    for i in range(n):
        cc.add_0_cell(_zero(embedding_dim), cell_type="step")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, _zero(embedding_dim), relation_type="next")
    return cc


def _build_constraint(embedding_dim: int, num_steps: int) -> CellComplex:
    """Star graph: one central goal node connected to num_steps leaf constraint nodes.

    Center = goal, leaves = constraints.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    center = cc.add_0_cell(_zero(embedding_dim), cell_type="goal")
    for _ in range(num_steps):
        leaf = cc.add_0_cell(_zero(embedding_dim), cell_type="constraint")
        cc.add_1_cell(center, leaf, _zero(embedding_dim), relation_type="constrains")
    return cc


def _build_multi_hop(embedding_dim: int, num_steps: int) -> CellComplex:
    """DAG with two evidence streams that merge at a conclusion node.

    Stream A: num_steps nodes feeding into conclusion.
    Stream B: num_steps nodes feeding into conclusion.
    The conclusion node is the final node.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    depth = max(num_steps, 2)

    # Stream A nodes (indices 0 .. depth-1)
    stream_a = []
    for _ in range(depth):
        idx = cc.add_0_cell(_zero(embedding_dim), cell_type="evidence_a")
        stream_a.append(idx)

    # Stream B nodes (indices depth .. 2*depth-1)
    stream_b = []
    for _ in range(depth):
        idx = cc.add_0_cell(_zero(embedding_dim), cell_type="evidence_b")
        stream_b.append(idx)

    # Conclusion node
    conclusion = cc.add_0_cell(_zero(embedding_dim), cell_type="conclusion")

    # Chain within each stream
    for i in range(depth - 1):
        cc.add_1_cell(stream_a[i], stream_a[i + 1], _zero(embedding_dim), relation_type="supports")
        cc.add_1_cell(stream_b[i], stream_b[i + 1], _zero(embedding_dim), relation_type="supports")

    # Final edge from each stream into conclusion
    cc.add_1_cell(stream_a[-1], conclusion, _zero(embedding_dim), relation_type="implies")
    cc.add_1_cell(stream_b[-1], conclusion, _zero(embedding_dim), relation_type="implies")

    return cc


def _build_exploration(embedding_dim: int, num_steps: int) -> CellComplex:
    """Complete binary tree of depth d where 2^d - 1 >= num_steps.

    Represents search/discovery problems.  The depth is chosen so the tree
    has at least num_steps nodes.
    """
    import math

    # Choose depth so total nodes >= num_steps
    depth = max(1, math.ceil(math.log2(num_steps + 1)))
    total_nodes = 2 ** depth - 1

    cc = CellComplex(embedding_dim=embedding_dim)
    for i in range(total_nodes):
        cc.add_0_cell(_zero(embedding_dim), cell_type="node")

    # Standard binary tree parent relationship: parent of node i is (i-1)//2
    for i in range(1, total_nodes):
        parent = (i - 1) // 2
        cc.add_1_cell(parent, i, _zero(embedding_dim), relation_type="branch")

    return cc


def _build_verification(embedding_dim: int, num_steps: int) -> CellComplex:
    """Bipartite graph: num_steps claim nodes ↔ num_steps evidence nodes.

    Each claim is connected to the corresponding evidence node.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    n = max(num_steps, 1)

    claims = []
    for _ in range(n):
        idx = cc.add_0_cell(_zero(embedding_dim), cell_type="claim")
        claims.append(idx)

    evidence = []
    for _ in range(n):
        idx = cc.add_0_cell(_zero(embedding_dim), cell_type="evidence")
        evidence.append(idx)

    for c, e in zip(claims, evidence):
        cc.add_1_cell(c, e, _zero(embedding_dim), relation_type="supported_by")

    return cc
