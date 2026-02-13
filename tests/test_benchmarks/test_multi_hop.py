import torch
import pytest
from src.benchmarks.multi_hop import (
    generate_chain_task, generate_tree_task, generate_diverse_task, MultiHopDataset,
)


class TestChainTask:
    def test_basic_chain(self):
        cc, query_node, target_node, answer = generate_chain_task(
            num_hops=3, num_distractors=5, embedding_dim=16,
        )
        assert cc.num_cells(0) >= 4
        assert cc.num_cells(1) >= 3
        assert 0 <= query_node < cc.num_cells(0)
        assert 0 <= target_node < cc.num_cells(0)
        assert query_node != target_node

    def test_distractor_nodes(self):
        cc, _, _, _ = generate_chain_task(num_hops=3, num_distractors=10, embedding_dim=16)
        assert cc.num_cells(0) == 4 + 10

    def test_varying_hops(self):
        for hops in [2, 5, 8]:
            cc, q, t, a = generate_chain_task(num_hops=hops, num_distractors=5, embedding_dim=16)
            assert cc.num_cells(0) == hops + 1 + 5


class TestTreeTask:
    def test_basic_tree(self):
        cc, query_node, target_node, answer = generate_tree_task(
            depth=3, branching=2, num_distractors=5, embedding_dim=16,
        )
        assert cc.num_cells(0) > 0
        assert cc.num_cells(1) > 0


class TestMultiHopDataset:
    def test_dataset_length(self):
        ds = MultiHopDataset(
            num_samples=100, min_hops=2, max_hops=5,
            num_distractors=10, embedding_dim=16,
        )
        assert len(ds) == 100

    def test_dataset_item(self):
        ds = MultiHopDataset(
            num_samples=10, min_hops=2, max_hops=4,
            num_distractors=5, embedding_dim=16,
        )
        cc, query, target, answer = ds[0]
        assert cc.num_cells(0) > 0


class TestDiverseTask:
    def test_generates_valid_sample(self):
        cc, query, target, answer = generate_diverse_task(
            target_hops=4, embedding_dim=16, n_nodes_range=(20, 40))
        assert cc.num_cells(0) > 0
        assert cc.num_cells(1) > 0
        assert 0 <= query < cc.num_cells(0)
        assert 0 <= target < cc.num_cells(0)
        assert answer == 4

    def test_various_hop_counts(self):
        for hops in [2, 5, 8]:
            cc, q, t, a = generate_diverse_task(
                target_hops=hops, embedding_dim=16, n_nodes_range=(20, 50))
            assert a == hops


class TestDiverseDataset:
    def test_diverse_dataset_length(self):
        ds = MultiHopDataset(
            num_samples=27, min_hops=2, max_hops=4,
            num_distractors=10, embedding_dim=16,
            task_type="diverse", n_nodes_range=(15, 30),
        )
        assert len(ds) == 27

    def test_class_balance(self):
        """Diverse mode produces balanced classes."""
        ds = MultiHopDataset(
            num_samples=27, min_hops=2, max_hops=4,
            num_distractors=10, embedding_dim=16,
            task_type="diverse", n_nodes_range=(15, 30),
        )
        # 3 classes (2,3,4), 27 samples -> 9 per class
        counts = {}
        for i in range(len(ds)):
            _, _, _, answer = ds.samples[i]  # access raw samples for counting
            counts[answer] = counts.get(answer, 0) + 1
        assert counts[2] == 9
        assert counts[3] == 9
        assert counts[4] == 9

    def test_shuffle_changes_order(self):
        ds = MultiHopDataset(
            num_samples=20, min_hops=2, max_hops=4,
            num_distractors=10, embedding_dim=16,
        )
        order1 = [ds._indices[i] for i in range(len(ds))]
        ds.shuffle()
        order2 = [ds._indices[i] for i in range(len(ds))]
        # Very unlikely to be the same twice
        assert order1 != order2 or len(ds) <= 1
