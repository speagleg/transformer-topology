"""Tests for ReasoningGraphConstructor."""

import torch
from src.metacog.graph_constructor import ReasoningGraphConstructor


def _emb(dim=128, seed=None):
    """Create a random embedding vector."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(dim)


def test_first_step_no_edges():
    """First step creates a node with no edges."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    step_id = gc.add_step("Observe X", _emb(128, seed=0), depends_on=[])
    assert step_id == 1
    assert gc.num_steps == 1
    cc = gc.get_snapshot()
    assert cc.num_cells(0) == 1
    assert cc.num_cells(1) == 0


def test_dependent_step():
    """Step depending on step 1 creates an edge."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    gc.add_step("Step 1", _emb(128, seed=0), depends_on=[])
    gc.add_step("Step 2", _emb(128, seed=1), depends_on=[1])
    cc = gc.get_snapshot()
    assert cc.num_cells(0) == 2
    assert cc.num_cells(1) == 1
    # Edge should be from node 0 to node 1
    assert cc._1_cell_sources[0] == 0
    assert cc._1_cell_targets[0] == 1


def test_multiple_deps():
    """Step with multiple dependencies gets multiple edges."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    gc.add_step("A", _emb(128, seed=0), depends_on=[])
    gc.add_step("B", _emb(128, seed=1), depends_on=[1])
    gc.add_step("C", _emb(128, seed=2), depends_on=[1, 2])
    cc = gc.get_snapshot()
    assert cc.num_cells(0) == 3
    # Step 3 depends on step 1 and step 2 => 2 new edges + 1 from step 2
    # Total: step2->step1 (1) + step3->step1 (1) + step3->step2 (1) = 3
    assert cc.num_cells(1) == 3


def test_invalid_deps_ignored():
    """Invalid dependency references are silently ignored."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    gc.add_step("A", _emb(128, seed=0), depends_on=[])
    gc.add_step("B", _emb(128, seed=1), depends_on=[99, 100])
    cc = gc.get_snapshot()
    assert cc.num_cells(0) == 2
    # Invalid deps ignored, fallback to previous step
    assert cc.num_cells(1) == 1


def test_empty_deps_fallback():
    """Empty depends_on on non-first step falls back to previous step."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    gc.add_step("A", _emb(128, seed=0), depends_on=[])
    gc.add_step("B", _emb(128, seed=1), depends_on=[])
    cc = gc.get_snapshot()
    assert cc.num_cells(1) == 1
    assert cc._1_cell_sources[0] == 0
    assert cc._1_cell_targets[0] == 1


def test_triangle_detection():
    """Three mutually connected nodes form a 2-cell."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    gc.add_step("A", _emb(128, seed=0), depends_on=[])
    gc.add_step("B", _emb(128, seed=1), depends_on=[1])       # edge 0-1
    gc.add_step("C", _emb(128, seed=2), depends_on=[1, 2])    # edges 0-2, 1-2
    cc = gc.get_snapshot()
    assert cc.num_cells(0) == 3
    assert cc.num_cells(1) == 3
    # Triangle (0,1,2) should produce a 2-cell
    assert cc.num_cells(2) == 1
    # Verify chain complex property
    assert cc.verify_chain_complex()


def test_snapshot_cloning():
    """Snapshot returns an independent clone."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    gc.add_step("A", _emb(128, seed=0), depends_on=[])
    snap1 = gc.get_snapshot()
    gc.add_step("B", _emb(128, seed=1), depends_on=[1])
    snap2 = gc.get_snapshot()
    # snap1 should not be affected by subsequent add_step
    assert snap1.num_cells(0) == 1
    assert snap2.num_cells(0) == 2


def test_edge_direction():
    """Edges from earlier to later steps are labeled 'forward'."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    gc.add_step("A", _emb(128, seed=0), depends_on=[])
    gc.add_step("B", _emb(128, seed=1), depends_on=[1])
    cc = gc.get_snapshot()
    assert cc._1_cell_types[0] == "forward"


def test_cosine_similarity_in_edge_embedding():
    """Edge embedding dim 0 stores cosine similarity between endpoints."""
    gc = ReasoningGraphConstructor(embedding_dim=128)
    e0 = _emb(128, seed=0)
    e1 = _emb(128, seed=1)
    gc.add_step("A", e0, depends_on=[])
    gc.add_step("B", e1, depends_on=[1])
    cc = gc.get_snapshot()
    expected_cos = torch.nn.functional.cosine_similarity(
        e0.unsqueeze(0), e1.unsqueeze(0)
    ).item()
    actual_cos = cc._1_cell_embeddings[0][0].item()
    assert abs(actual_cos - expected_cos) < 1e-5
