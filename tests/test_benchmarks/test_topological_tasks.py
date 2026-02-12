import torch
import pytest
from src.benchmarks.topological_tasks import (
    generate_cycle_detection_task,
    generate_path_counting_task,
    generate_betti_task,
    auto_fill_triangles,
    TopologicalDataset,
)
from src.cell_complex.cell_complex import CellComplex


class TestAutoFillTriangles:
    def test_fills_triangle(self):
        cc = CellComplex(embedding_dim=8)
        n0 = cc.add_0_cell(torch.randn(8), "node")
        n1 = cc.add_0_cell(torch.randn(8), "node")
        n2 = cc.add_0_cell(torch.randn(8), "node")
        cc.add_1_cell(n0, n1, torch.randn(8), "edge")
        cc.add_1_cell(n1, n2, torch.randn(8), "edge")
        cc.add_1_cell(n0, n2, torch.randn(8), "edge")
        assert cc.num_cells(2) == 0
        added = auto_fill_triangles(cc)
        assert added == 1
        assert cc.num_cells(2) == 1
        assert cc.verify_chain_complex()


class TestCycleDetection:
    def test_with_cycle(self):
        cc, q, t, answer = generate_cycle_detection_task(8, 16, has_cycle=True)
        assert answer == 1
        assert cc.num_cells(0) >= 3

    def test_without_cycle(self):
        cc, q, t, answer = generate_cycle_detection_task(8, 16, has_cycle=False)
        assert answer == 0


class TestPathCounting:
    def test_known_paths(self):
        cc, source, target, answer = generate_path_counting_task(
            num_paths=3, path_length=2, embedding_dim=16
        )
        assert answer == 3
        assert cc.num_cells(0) >= 2


class TestBettiPrediction:
    def test_known_betti(self):
        cc, n0, n1, answer = generate_betti_task(target_beta1=2, embedding_dim=16)
        assert answer == 2
        # Verify: beta_1 = edges - nodes + 1 (single component)
        n_edges = cc.num_cells(1)
        n_nodes = cc.num_cells(0)
        # beta_1 = n_edges - n_nodes + 1 for connected graph
        assert n_edges - n_nodes + 1 == 2


class TestTopologicalDataset:
    def test_dataset_length(self):
        ds = TopologicalDataset(num_samples=10, task_type="cycle_detection", embedding_dim=16)
        assert len(ds) == 10

    def test_dataset_item_format(self):
        ds = TopologicalDataset(num_samples=5, task_type="betti_number", embedding_dim=16, max_beta=3)
        cc, q, t, answer = ds[0]
        assert isinstance(cc, CellComplex)
        assert isinstance(answer, int)
