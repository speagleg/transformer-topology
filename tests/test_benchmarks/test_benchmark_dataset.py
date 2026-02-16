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
            "graph_completion", "labeled_reasoning", "analogical_transfer",
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


class TestMixedSizeTraining:
    def test_mixed_size_generates_varied_sizes(self):
        """n_nodes_range produces samples with different node counts."""
        ds = BenchmarkDataset(
            num_samples=20, task_type="bfs",
            n_nodes=16, embedding_dim=16,
            n_nodes_range=(10, 25),
        )
        sizes = {ds.samples[i][0].num_cells(0) for i in range(len(ds))}
        # With 20 samples across range 10-25, we should see multiple distinct sizes
        assert len(sizes) > 1, f"Expected varied sizes, got {sizes}"
        for s in sizes:
            assert 10 <= s <= 25, f"Size {s} outside range [10, 25]"

    def test_mixed_size_within_range(self):
        """All samples have node counts within the specified range."""
        ds = BenchmarkDataset(
            num_samples=15, task_type="bfs",
            n_nodes=20, embedding_dim=16,
            n_nodes_range=(12, 18),
        )
        for i in range(len(ds)):
            cc = ds.samples[i][0]
            n = cc.num_cells(0)
            assert 12 <= n <= 18, f"Sample {i}: n={n} outside [12, 18]"

    def test_fixed_size_without_range(self):
        """Without n_nodes_range, all samples have the same node count."""
        ds = BenchmarkDataset(
            num_samples=10, task_type="bfs",
            n_nodes=16, embedding_dim=16,
        )
        for i in range(len(ds)):
            cc = ds.samples[i][0]
            assert cc.num_cells(0) == 16

    def test_save_load_with_range(self):
        """n_nodes_range is preserved through save/load."""
        ds = BenchmarkDataset(
            num_samples=5, task_type="hodge_class",
            n_nodes=20, embedding_dim=16,
            n_nodes_range=(15, 25),
        )
        assert ds.n_nodes_range == (15, 25)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mixed.pt")
            ds.save(path)
            loaded = BenchmarkDataset.load(path)
            assert loaded.n_nodes_range == (15, 25)
            assert len(loaded) == 5
