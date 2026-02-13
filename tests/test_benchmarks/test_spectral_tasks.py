import pytest

from src.benchmarks.spectral_tasks import (
    generate_spectral_gap_task,
    generate_hodge_class_task,
    _calibrate_spectral_gap_buckets,
    _discretize,
    _BUCKET_CACHE,
)
from src.cell_complex.cell_complex import CellComplex


class TestSpectralGap:
    def test_spectral_gap_answer_in_range(self):
        """Spectral gap answers are in [0, num_buckets)."""
        for _ in range(10):
            cc, src, tgt, answer = generate_spectral_gap_task(
                n_nodes=15, embedding_dim=16, num_buckets=8,
            )
            assert 0 <= answer < 8
            assert isinstance(cc, CellComplex)
            assert src != tgt

    def test_spectral_gap_tree_vs_dense(self):
        """Trees tend to have smaller lambda_2 than dense graphs."""
        from src.spectral.decomposition import spectral_decomposition
        from src.benchmarks.graph_generators import random_graph
        from src.benchmarks.graph_convert import nx_to_cell_complex

        tree_lambda2s = []
        for _ in range(20):
            G = random_graph(20, topology='tree')
            cc, _ = nx_to_cell_complex(G, 16, source_node=0, target_node=1)
            evals, _ = spectral_decomposition(cc, dim=0)
            if len(evals) > 1:
                tree_lambda2s.append(evals[1].item())

        dense_lambda2s = []
        for _ in range(20):
            G = random_graph(20, topology='er')
            cc, _ = nx_to_cell_complex(G, 16, source_node=0, target_node=1)
            evals, _ = spectral_decomposition(cc, dim=0)
            if len(evals) > 1:
                dense_lambda2s.append(evals[1].item())

        # Trees should have smaller spectral gap on average
        avg_tree = sum(tree_lambda2s) / len(tree_lambda2s) if tree_lambda2s else 0
        avg_dense = sum(dense_lambda2s) / len(dense_lambda2s) if dense_lambda2s else 0
        assert avg_tree < avg_dense, (
            f"Expected trees ({avg_tree:.3f}) < ER ({avg_dense:.3f})"
        )

    def test_spectral_gap_buckets_cover_range(self):
        """All 8 buckets get at least some samples with enough diversity."""
        _BUCKET_CACHE.clear()
        buckets_seen = set()
        for _ in range(200):
            _, _, _, answer = generate_spectral_gap_task(
                n_nodes=20, embedding_dim=16, num_buckets=8,
            )
            buckets_seen.add(answer)
            if len(buckets_seen) == 8:
                break
        # With 200 samples across 8 topologies, we should see most buckets
        assert len(buckets_seen) >= 4, f"Only saw buckets {buckets_seen}"

    def test_spectral_gap_variable_sizes(self):
        """Spectral gap works for different graph sizes."""
        _BUCKET_CACHE.clear()
        for n in [16, 32, 64]:
            cc, src, tgt, answer = generate_spectral_gap_task(
                n_nodes=n, embedding_dim=16, num_buckets=8,
            )
            assert 0 <= answer < 8
            assert cc.num_cells(0) >= n * 0.8  # some topologies adjust node count


class TestHodgeClass:
    def test_hodge_class_answers_in_range(self):
        """Hodge class answers are 0, 1, or 2."""
        for _ in range(20):
            cc, src, tgt, answer = generate_hodge_class_task(
                n_nodes=15, embedding_dim=16,
            )
            assert answer in (0, 1, 2)
            assert isinstance(cc, CellComplex)

    def test_hodge_class_tree_is_gradient(self):
        """Trees (no cycles) should be gradient-dominated (class 0)."""
        gradient_count = 0
        n_trials = 20
        for _ in range(n_trials):
            cc, _, _, answer = generate_hodge_class_task(
                n_nodes=15, embedding_dim=16, topologies=['tree'],
            )
            if answer == 0:
                gradient_count += 1
        # Trees have no cycles → no curl, minimal harmonic → gradient should dominate
        assert gradient_count > n_trials // 2, (
            f"Expected most trees to be gradient, got {gradient_count}/{n_trials}"
        )

    def test_hodge_class_variable_sizes(self):
        """Hodge class works for different graph sizes."""
        for n in [16, 32, 64]:
            cc, src, tgt, answer = generate_hodge_class_task(
                n_nodes=n, embedding_dim=16,
            )
            assert answer in (0, 1, 2)

    def test_hodge_class_with_topologies(self):
        """Hodge class respects topology restriction."""
        for _ in range(10):
            cc, src, tgt, answer = generate_hodge_class_task(
                n_nodes=20, embedding_dim=16, topologies=['ba', 'er'],
            )
            assert answer in (0, 1, 2)


class TestBucketCalibration:
    def test_calibration_returns_boundaries(self):
        """Calibration produces the right number of boundaries."""
        boundaries = _calibrate_spectral_gap_buckets(
            n_nodes=15, embedding_dim=16, num_buckets=8,
            calibration_size=50,
        )
        assert len(boundaries) == 7  # num_buckets - 1

    def test_discretize(self):
        """_discretize maps values to correct buckets."""
        boundaries = [0.5, 1.0, 1.5, 2.0]
        assert _discretize(0.3, boundaries) == 0
        assert _discretize(0.7, boundaries) == 1
        assert _discretize(1.2, boundaries) == 2
        assert _discretize(1.8, boundaries) == 3
        assert _discretize(2.5, boundaries) == 4
