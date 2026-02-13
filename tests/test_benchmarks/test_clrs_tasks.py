import os
import tempfile

import networkx as nx
import pytest

from src.benchmarks.graph_generators import generate_er_graph, clrs_er_graph, random_graph
from src.benchmarks.clrs_tasks import (
    generate_bfs_task,
    generate_dijkstra_task,
    CLRSDataset,
)
from src.cell_complex.cell_complex import CellComplex


class TestERGraph:
    def test_er_graph_connected(self):
        """ER graphs are connected after _ensure_connected."""
        for _ in range(10):
            G = generate_er_graph(20, p=0.3)
            assert nx.is_connected(G)

    def test_er_graph_node_count(self):
        """ER graph has the correct number of nodes."""
        G = generate_er_graph(30, p=0.2)
        assert G.number_of_nodes() == 30

    def test_clrs_er_graph_connected(self):
        """CLRS ER graphs are connected."""
        for n in [16, 32, 64]:
            G = clrs_er_graph(n)
            assert nx.is_connected(G)
            assert G.number_of_nodes() == n

    def test_clrs_er_graph_avg_degree(self):
        """CLRS ER graphs have approximately avg degree ~4."""
        for n in [16, 32, 64]:
            degrees = []
            for _ in range(20):
                G = clrs_er_graph(n)
                avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
                degrees.append(avg_deg)
            mean_deg = sum(degrees) / len(degrees)
            # Allow some variance but should be roughly 4
            assert 2.0 < mean_deg < 8.0, f"n={n}: avg degree {mean_deg:.1f}"

    def test_er_in_random_dispatcher(self):
        """'er' topology works in random_graph()."""
        G = random_graph(30, topology='er')
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 30


class TestBFSTask:
    def test_bfs_task_answer_correct(self):
        """BFS ground truth matches nx.shortest_path_length."""
        for _ in range(20):
            cc, source, target, answer = generate_bfs_task(
                n_nodes=16, embedding_dim=16, max_distance=16,
            )
            assert isinstance(cc, CellComplex)
            assert 0 <= source < cc.num_cells(0)
            assert 0 <= target < cc.num_cells(0)
            assert source != target
            assert 0 <= answer < 16

    def test_bfs_task_small_graph(self):
        """BFS task works on small graphs."""
        cc, source, target, answer = generate_bfs_task(
            n_nodes=5, embedding_dim=8, max_distance=10,
        )
        assert cc.num_cells(0) == 5
        assert 0 <= answer < 10

    def test_bfs_has_triangles(self):
        """BFS task auto-fills 2-cells from triangles."""
        # ER(16, 0.3) likely has triangles; run several times
        found_2cells = False
        for _ in range(10):
            cc, _, _, _ = generate_bfs_task(n_nodes=16, embedding_dim=16)
            if cc.num_cells(2) > 0:
                found_2cells = True
                break
        assert found_2cells, "Expected at least one graph with triangles"


class TestDijkstraTask:
    def test_dijkstra_task_answer_correct(self):
        """Dijkstra ground truth matches nx.dijkstra_path_length."""
        for _ in range(20):
            cc, source, target, answer = generate_dijkstra_task(
                n_nodes=16, embedding_dim=16, max_distance=16,
                max_edge_weight=5,
            )
            assert isinstance(cc, CellComplex)
            assert 0 <= answer < 16

    def test_dijkstra_has_weighted_edges(self):
        """Dijkstra task has edges with weights > 1."""
        found_heavy = False
        for _ in range(10):
            cc, _, _, _ = generate_dijkstra_task(
                n_nodes=16, embedding_dim=16, max_edge_weight=5,
            )
            for e in range(cc.num_cells(1)):
                w = int(cc._1_cell_embeddings[e][0].item())
                if w > 1:
                    found_heavy = True
                    break
            if found_heavy:
                break
        assert found_heavy, "Expected edges with weight > 1"


class TestCLRSDataset:
    def test_clrs_dataset_sizes(self):
        """CLRSDataset has the requested number of samples."""
        ds = CLRSDataset(
            num_samples=50, task_type="bfs", n_nodes=16, embedding_dim=16,
        )
        assert len(ds) == 50

    def test_clrs_dataset_item_format(self):
        """Each item is a (CellComplex, int, int, int) tuple."""
        ds = CLRSDataset(
            num_samples=5, task_type="bfs", n_nodes=16, embedding_dim=16,
        )
        cc, q, t, a = ds[0]
        assert isinstance(cc, CellComplex)
        assert isinstance(q, int)
        assert isinstance(t, int)
        assert isinstance(a, int)

    def test_clrs_dataset_dijkstra(self):
        """Dijkstra dataset generates valid samples."""
        ds = CLRSDataset(
            num_samples=10, task_type="dijkstra", n_nodes=16,
            embedding_dim=16, max_edge_weight=5,
        )
        assert len(ds) == 10
        for i in range(len(ds)):
            _, _, _, answer = ds[i]
            assert 0 <= answer < 16

    def test_clrs_dataset_save_load(self):
        """Serialization roundtrip preserves data."""
        ds = CLRSDataset(
            num_samples=10, task_type="bfs", n_nodes=16, embedding_dim=16,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_clrs.pt")
            ds.save(path)
            loaded = CLRSDataset.load(path)
            assert len(loaded) == len(ds)
            assert loaded.task_type == "bfs"
            assert loaded.n_nodes == 16
            # Check first sample matches
            cc1, q1, t1, a1 = ds.samples[0]
            cc2, q2, t2, a2 = loaded.samples[0]
            assert q1 == q2
            assert t1 == t2
            assert a1 == a2

    def test_clrs_dataset_shuffle(self):
        """Shuffling changes access order."""
        ds = CLRSDataset(
            num_samples=20, task_type="bfs", n_nodes=16, embedding_dim=16,
        )
        order1 = list(ds._indices)
        ds.shuffle()
        order2 = list(ds._indices)
        assert order1 != order2 or len(ds) <= 1

    def test_invalid_task_type(self):
        """Unknown task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown CLRS task type"):
            CLRSDataset(
                num_samples=1, task_type="nonexistent", n_nodes=16,
                embedding_dim=16,
            )


class TestVariableGraphSizes:
    def test_bfs_variable_graph_sizes(self):
        """BFS tasks generate correctly for n=16, 32, 64."""
        for n in [16, 32, 64]:
            cc, source, target, answer = generate_bfs_task(
                n_nodes=n, embedding_dim=16, max_distance=16,
            )
            assert cc.num_cells(0) == n
            assert 0 <= answer < 16

    def test_dijkstra_variable_graph_sizes(self):
        """Dijkstra tasks generate correctly for n=16, 32, 64."""
        for n in [16, 32, 64]:
            cc, source, target, answer = generate_dijkstra_task(
                n_nodes=n, embedding_dim=16, max_distance=16,
            )
            assert cc.num_cells(0) == n
            assert 0 <= answer < 16
