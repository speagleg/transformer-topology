import os
import tempfile

import pytest

from src.benchmarks.benchmark_dataset import (
    BenchmarkDataset,
    TASK_REGISTRY,
    get_max_classes,
)
from src.cell_complex.cell_complex import CellComplex


class TestTaskRegistry:
    def test_all_tasks_registered(self):
        """All expected task types are in the registry."""
        expected = {
            "diverse", "propagation_delay", "blocking", "interference",
            "cycle_detection", "path_counting", "betti_number",
            "bfs", "dijkstra", "spectral_gap", "hodge_class",
        }
        assert set(TASK_REGISTRY.keys()) == expected

    def test_get_max_classes(self):
        """get_max_classes returns correct values."""
        assert get_max_classes("diverse") == 11
        assert get_max_classes("hodge_class") == 3
        assert get_max_classes("spectral_gap") == 8
        assert get_max_classes("bfs") == 16

    def test_get_max_classes_invalid(self):
        """Unknown task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            get_max_classes("nonexistent")


class TestBenchmarkDataset:
    def test_dataset_length(self):
        """BenchmarkDataset has the requested number of samples."""
        ds = BenchmarkDataset(
            num_samples=10, task_type="hodge_class",
            n_nodes=10, embedding_dim=16,
        )
        assert len(ds) == 10

    def test_dataset_item_format(self):
        """Each item is a (CellComplex, int, int, int) tuple."""
        ds = BenchmarkDataset(
            num_samples=5, task_type="bfs",
            n_nodes=16, embedding_dim=16,
        )
        cc, q, t, a = ds[0]
        assert isinstance(cc, CellComplex)
        assert isinstance(q, int)
        assert isinstance(t, int)
        assert isinstance(a, int)

    def test_topology_filtering(self):
        """Dataset respects topology filtering."""
        ds = BenchmarkDataset(
            num_samples=10, task_type="cycle_detection",
            n_nodes=15, embedding_dim=16,
            topologies=["tree", "ba"],
        )
        assert len(ds) == 10
        for i in range(len(ds)):
            _, _, _, answer = ds[i]
            assert answer in (0, 1)

    def test_shuffle(self):
        """Shuffling changes access order."""
        ds = BenchmarkDataset(
            num_samples=20, task_type="hodge_class",
            n_nodes=10, embedding_dim=16,
        )
        order1 = list(ds._indices)
        ds.shuffle()
        order2 = list(ds._indices)
        # Extremely unlikely to be the same with 20 elements
        assert order1 != order2

    def test_save_load(self):
        """Serialization roundtrip preserves data."""
        ds = BenchmarkDataset(
            num_samples=10, task_type="bfs",
            n_nodes=16, embedding_dim=16,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_benchmark.pt")
            ds.save(path)
            loaded = BenchmarkDataset.load(path)
            assert len(loaded) == len(ds)
            assert loaded.task_type == "bfs"
            assert loaded.n_nodes == 16
            # Check first sample matches
            cc1, q1, t1, a1 = ds.samples[0]
            cc2, q2, t2, a2 = loaded.samples[0]
            assert q1 == q2 and t1 == t2 and a1 == a2

    def test_invalid_task_type(self):
        """Unknown task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            BenchmarkDataset(
                num_samples=1, task_type="nonexistent",
                n_nodes=10, embedding_dim=16,
            )

    def test_spectral_gap_dataset(self):
        """spectral_gap task generates valid samples."""
        ds = BenchmarkDataset(
            num_samples=10, task_type="spectral_gap",
            n_nodes=15, embedding_dim=16,
        )
        for i in range(len(ds)):
            cc, src, tgt, answer = ds[i]
            assert 0 <= answer < 8

    def test_diverse_dataset(self):
        """diverse task generates valid samples."""
        ds = BenchmarkDataset(
            num_samples=10, task_type="diverse",
            n_nodes=20, embedding_dim=16,
        )
        for i in range(len(ds)):
            cc, src, tgt, answer = ds[i]
            assert 0 <= answer <= 10

    def test_propagation_delay_dataset(self):
        """propagation_delay with topology control works."""
        ds = BenchmarkDataset(
            num_samples=10, task_type="propagation_delay",
            n_nodes=15, embedding_dim=16,
            topologies=["ba", "er"],
        )
        for i in range(len(ds)):
            _, _, _, answer = ds[i]
            assert 0 <= answer < 10
