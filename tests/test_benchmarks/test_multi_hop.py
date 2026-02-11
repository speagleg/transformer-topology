import torch
import pytest
from src.benchmarks.multi_hop import generate_chain_task, generate_tree_task, MultiHopDataset


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
