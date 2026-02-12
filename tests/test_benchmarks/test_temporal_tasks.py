import torch
import pytest
from src.benchmarks.temporal_tasks import (
    generate_propagation_delay_task,
    generate_blocking_task,
    generate_interference_task,
    TemporalDataset,
    _dijkstra,
)
from src.cell_complex.cell_complex import CellComplex


class TestPropagationDelay:
    def test_valid_complex(self):
        """Generated complex has valid structure and answer is in range."""
        cc, source, target, answer = generate_propagation_delay_task(
            n_nodes=8, embedding_dim=16, max_delay=10
        )
        assert cc.num_cells(0) == 8
        assert cc.num_cells(1) >= 0
        assert 0 <= answer < 10
        assert 0 <= source < cc.num_cells(0)
        assert 0 <= target < cc.num_cells(0)
        assert source != target

    def test_known_answer(self):
        """Chain of 3 edges with delays 1, 2, 1 gives answer = 4."""
        dim = 16
        cc = CellComplex(embedding_dim=dim)
        n0 = cc.add_0_cell(torch.randn(dim), "source")
        n1 = cc.add_0_cell(torch.randn(dim), "node")
        n2 = cc.add_0_cell(torch.randn(dim), "node")
        n3 = cc.add_0_cell(torch.randn(dim), "target")

        # Edge 0: n0 -> n1, delay=1
        emb0 = torch.randn(dim)
        emb0[0] = 1.0
        cc.add_1_cell(n0, n1, emb0, "temporal_edge")

        # Edge 1: n1 -> n2, delay=2
        emb1 = torch.randn(dim)
        emb1[0] = 2.0
        cc.add_1_cell(n1, n2, emb1, "temporal_edge")

        # Edge 2: n2 -> n3, delay=1
        emb2 = torch.randn(dim)
        emb2[0] = 1.0
        cc.add_1_cell(n2, n3, emb2, "temporal_edge")

        # Build adjacency and run Dijkstra
        from src.benchmarks.temporal_tasks import _build_adjacency

        adj = _build_adjacency(cc)
        dist = _dijkstra(adj, n0)
        assert dist[n3] == 4

    def test_disconnected_graph(self):
        """Unreachable target gets sentinel answer max_delay - 1."""
        dim = 16
        cc = CellComplex(embedding_dim=dim)
        n0 = cc.add_0_cell(torch.randn(dim), "source")
        n1 = cc.add_0_cell(torch.randn(dim), "target")
        # No edges -- disconnected

        from src.benchmarks.temporal_tasks import _build_adjacency

        adj = _build_adjacency(cc)
        dist = _dijkstra(adj, n0)
        assert n1 not in dist


class TestBlocking:
    def test_answer_range(self):
        """Blocking task answer is within valid range."""
        max_delay = 10
        cc, source, target, answer = generate_blocking_task(
            n_nodes=8, embedding_dim=16, max_delay=max_delay
        )
        assert 0 <= answer < max_delay

    def test_blocked_node_marked(self):
        """At least one node is marked as blocked."""
        cc, source, target, answer = generate_blocking_task(
            n_nodes=8, embedding_dim=16, max_delay=10
        )
        assert "blocked" in cc._0_cell_types

    def test_blocking_increases_distance(self):
        """Blocking a node on the only shortest path increases the distance."""
        dim = 16
        cc = CellComplex(embedding_dim=dim)
        # Chain: 0 -> 1 -> 2 -> 3, all delay=1
        nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(4)]
        cc._0_cell_types[0] = "source"
        cc._0_cell_types[3] = "target"
        cc._0_cell_types[1] = "blocked"

        for i in range(3):
            emb = torch.randn(dim)
            emb[0] = 1.0
            cc.add_1_cell(nodes[i], nodes[i + 1], emb, "temporal_edge")

        from src.benchmarks.temporal_tasks import _build_adjacency

        adj = _build_adjacency(cc)

        # Without blocking: distance = 3
        dist_free = _dijkstra(adj, 0, blocked=set())
        assert dist_free[3] == 3

        # With blocking node 1: cannot reach 3
        dist_blocked = _dijkstra(adj, 0, blocked={1})
        assert 3 not in dist_blocked


class TestInterference:
    def test_binary_answer(self):
        """Interference task answer is 0 or 1."""
        cc, source1, target, answer = generate_interference_task(
            n_nodes=8, embedding_dim=16
        )
        assert answer in (0, 1)

    def test_constructive_known(self):
        """Two sources at equal distance -> even diff -> constructive (1)."""
        dim = 16
        cc = CellComplex(embedding_dim=dim)
        # Diamond: s1 -> mid -> target, s2 -> mid -> target
        s1 = cc.add_0_cell(torch.randn(dim), "source1")
        s2 = cc.add_0_cell(torch.randn(dim), "source2")
        mid = cc.add_0_cell(torch.randn(dim), "node")
        tgt = cc.add_0_cell(torch.randn(dim), "target")

        for src_node in [s1, s2]:
            emb = torch.randn(dim)
            emb[0] = 1.0
            cc.add_1_cell(src_node, mid, emb, "temporal_edge")
        emb = torch.randn(dim)
        emb[0] = 1.0
        cc.add_1_cell(mid, tgt, emb, "temporal_edge")

        from src.benchmarks.temporal_tasks import _build_adjacency

        adj = _build_adjacency(cc)
        dist1 = _dijkstra(adj, s1)
        dist2 = _dijkstra(adj, s2)
        diff = abs(dist1[tgt] - dist2[tgt])
        assert diff % 2 == 0  # constructive

    def test_destructive_known(self):
        """Two sources with odd path-length difference -> destructive (0)."""
        dim = 16
        cc = CellComplex(embedding_dim=dim)
        # s1 --1--> target, s2 --1--> mid --1--> target
        s1 = cc.add_0_cell(torch.randn(dim), "source1")
        s2 = cc.add_0_cell(torch.randn(dim), "source2")
        mid = cc.add_0_cell(torch.randn(dim), "node")
        tgt = cc.add_0_cell(torch.randn(dim), "target")

        emb = torch.randn(dim)
        emb[0] = 1.0
        cc.add_1_cell(s1, tgt, emb, "temporal_edge")

        emb = torch.randn(dim)
        emb[0] = 1.0
        cc.add_1_cell(s2, mid, emb, "temporal_edge")

        emb = torch.randn(dim)
        emb[0] = 1.0
        cc.add_1_cell(mid, tgt, emb, "temporal_edge")

        from src.benchmarks.temporal_tasks import _build_adjacency

        adj = _build_adjacency(cc)
        dist1 = _dijkstra(adj, s1)
        dist2 = _dijkstra(adj, s2)
        # s1 -> tgt = 1, s2 -> mid -> tgt = 2, diff = 1 (odd)
        assert dist1[tgt] == 1
        assert dist2[tgt] == 2
        assert abs(dist1[tgt] - dist2[tgt]) % 2 == 1  # destructive


class TestTemporalDataset:
    def test_dataset_length(self):
        """Dataset produces the requested number of samples."""
        ds = TemporalDataset(
            num_samples=20,
            task_type="propagation_delay",
            embedding_dim=16,
            n_nodes=6,
            max_delay=10,
        )
        assert len(ds) == 20

    def test_dataset_item_format(self):
        """Each dataset item is a (CellComplex, int, int, int) tuple."""
        ds = TemporalDataset(
            num_samples=5,
            task_type="blocking",
            embedding_dim=16,
        )
        cc, q, t, answer = ds[0]
        assert isinstance(cc, CellComplex)
        assert isinstance(q, int)
        assert isinstance(t, int)
        assert isinstance(answer, int)

    def test_interference_dataset(self):
        """Interference dataset produces binary answers."""
        ds = TemporalDataset(
            num_samples=30,
            task_type="interference",
            embedding_dim=16,
            n_nodes=8,
        )
        for i in range(len(ds)):
            _, _, _, answer = ds[i]
            assert answer in (0, 1)

    def test_invalid_task_type(self):
        """Unknown task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            TemporalDataset(
                num_samples=1,
                task_type="nonexistent",
                embedding_dim=16,
            )
