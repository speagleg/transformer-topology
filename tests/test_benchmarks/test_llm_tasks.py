"""Tests for LLM-dependent benchmark tasks."""

import random

import pytest
import torch

from collections import Counter

from src.benchmarks.llm_tasks import (
    generate_graph_completion_task,
    generate_labeled_reasoning_task,
    generate_analogical_transfer_task,
    CAUSAL_LABELS,
    NODE_LABEL_POOLS,
)
from src.benchmarks.benchmark_dataset import BenchmarkDataset, TASK_REGISTRY, get_max_classes
from src.cell_complex.cell_complex import CellComplex


class TestGraphCompletion:
    def test_returns_5_tuple(self):
        result = generate_graph_completion_task(15, 16)
        assert len(result) == 5

    def test_output_types(self):
        cc, src, tgt, answer, metadata = generate_graph_completion_task(15, 16)
        assert isinstance(cc, CellComplex)
        assert isinstance(src, int)
        assert isinstance(tgt, int)
        assert answer in (0, 1)
        assert isinstance(metadata, dict)

    def test_metadata_fields(self):
        _, _, _, _, metadata = generate_graph_completion_task(15, 16)
        assert 'task_type' in metadata
        assert metadata['task_type'] == 'graph_completion'
        assert 'task_prompt' in metadata
        assert 'removed_edges' in metadata

    def test_answer_range(self):
        for _ in range(20):
            _, _, _, answer, _ = generate_graph_completion_task(12, 16)
            assert answer in (0, 1)

    def test_class_balance_approximate(self):
        answers = []
        for _ in range(50):
            _, _, _, ans, _ = generate_graph_completion_task(15, 16)
            answers.append(ans)
        # Not perfectly balanced but should have both classes
        assert 0 in answers
        assert 1 in answers

    def test_topology_control(self):
        for _ in range(5):
            result = generate_graph_completion_task(15, 16, topologies=['ba'])
            assert len(result) == 5


class TestLabeledReasoning:
    def test_returns_5_tuple(self):
        result = generate_labeled_reasoning_task(15, 16)
        assert len(result) == 5

    def test_output_types(self):
        cc, src, tgt, answer, metadata = generate_labeled_reasoning_task(15, 16)
        assert isinstance(cc, CellComplex)
        assert isinstance(src, int)
        assert isinstance(tgt, int)
        assert answer in (0, 1, 2)
        assert isinstance(metadata, dict)

    def test_metadata_has_labels(self):
        _, _, _, _, metadata = generate_labeled_reasoning_task(15, 16)
        assert 'node_labels' in metadata
        assert 'edge_labels' in metadata
        assert 'domain' in metadata
        assert metadata['domain'] in NODE_LABEL_POOLS

    def test_answer_range(self):
        for _ in range(20):
            _, _, _, answer, _ = generate_labeled_reasoning_task(15, 16)
            assert answer in (0, 1, 2)

    def test_edge_labels_are_valid(self):
        _, _, _, _, metadata = generate_labeled_reasoning_task(15, 16)
        for label in metadata['edge_labels'].values():
            assert label in CAUSAL_LABELS


class TestLabeledReasoningBalance:
    def test_all_three_classes_present(self):
        """All 3 classes (causal_chain, blocked, independent) must appear."""
        classes_seen = set()
        for _ in range(200):
            _, _, _, answer, _ = generate_labeled_reasoning_task(20, 32)
            classes_seen.add(answer)
        assert classes_seen == {0, 1, 2}, f"Missing classes, only got: {classes_seen}"

    def test_roughly_balanced(self):
        """No class should be <15% or >55% of samples."""
        counts = Counter()
        n = 300
        for _ in range(n):
            _, _, _, answer, _ = generate_labeled_reasoning_task(20, 32)
            counts[answer] += 1
        for cls in [0, 1, 2]:
            pct = counts[cls] / n
            assert pct > 0.15, f"Class {cls} only {pct:.1%} ({counts[cls]}/{n})"
            assert pct < 0.55, f"Class {cls} too dominant at {pct:.1%} ({counts[cls]}/{n})"

    def test_independent_has_disconnected_components(self):
        """Class 2 (independent) samples should have no path between src/tgt."""
        for _ in range(30):
            cc, src, tgt, answer, metadata = generate_labeled_reasoning_task(20, 32)
            if answer == 2:
                # Verify the metadata reflects no path
                assert 'path_len' not in metadata['task_prompt']
                break
        else:
            pytest.fail("No class-2 sample seen in 30 attempts")

    def test_causal_chain_has_no_prevents(self):
        """Class 0 (causal_chain) path labels should not contain 'prevents'."""
        for _ in range(30):
            _, _, _, answer, metadata = generate_labeled_reasoning_task(20, 32)
            if answer == 0:
                prompt = metadata['task_prompt']
                # Extract path_labels from prompt
                if 'path_labels=' in prompt:
                    labels_str = prompt.split('path_labels=')[1].split(' |')[0]
                    labels = labels_str.split(',')
                    assert 'prevents' not in labels, f"Class 0 has 'prevents': {labels}"
                break
        else:
            pytest.fail("No class-0 sample seen in 30 attempts")

    def test_blocked_has_prevents(self):
        """Class 1 (blocked) path labels should contain 'prevents'."""
        for _ in range(30):
            _, _, _, answer, metadata = generate_labeled_reasoning_task(20, 32)
            if answer == 1:
                prompt = metadata['task_prompt']
                if 'path_labels=' in prompt:
                    labels_str = prompt.split('path_labels=')[1].split(' |')[0]
                    labels = labels_str.split(',')
                    assert 'prevents' in labels, f"Class 1 missing 'prevents': {labels}"
                break
        else:
            pytest.fail("No class-1 sample seen in 30 attempts")


class TestAnalogicalTransfer:
    def test_returns_5_tuple(self):
        result = generate_analogical_transfer_task(15, 16)
        assert len(result) == 5

    def test_output_types(self):
        cc, src, tgt, answer, metadata = generate_analogical_transfer_task(15, 16)
        assert isinstance(cc, CellComplex)
        assert isinstance(src, int)
        assert isinstance(tgt, int)
        assert 0 <= answer <= 2
        assert isinstance(metadata, dict)

    def test_metadata_has_domains(self):
        _, _, _, _, metadata = generate_analogical_transfer_task(15, 16)
        assert 'domain_a' in metadata
        assert 'domain_b' in metadata
        assert 'role_index' in metadata
        assert 'role_name' in metadata

    def test_answer_range(self):
        for _ in range(20):
            _, _, _, answer, _ = generate_analogical_transfer_task(15, 16)
            assert 0 <= answer <= 2


class TestRegistryIntegration:
    def test_tasks_registered(self):
        assert 'graph_completion' in TASK_REGISTRY
        assert 'labeled_reasoning' in TASK_REGISTRY
        assert 'analogical_transfer' in TASK_REGISTRY

    def test_max_classes(self):
        assert get_max_classes('graph_completion') == 2
        assert get_max_classes('labeled_reasoning') == 3
        assert get_max_classes('analogical_transfer') == 3

    def test_benchmark_dataset_graph_completion(self):
        ds = BenchmarkDataset(5, 'graph_completion', 12, 16)
        assert len(ds) == 5
        sample = ds[0]
        assert len(sample) == 5

    def test_benchmark_dataset_labeled_reasoning(self):
        ds = BenchmarkDataset(5, 'labeled_reasoning', 12, 16)
        assert len(ds) == 5
        sample = ds[0]
        assert len(sample) == 5

    def test_benchmark_dataset_analogical_transfer(self):
        ds = BenchmarkDataset(5, 'analogical_transfer', 12, 16)
        assert len(ds) == 5
        sample = ds[0]
        assert len(sample) == 5


class TestAnalogicalTransferFix:
    def test_roles_assigned_by_degree_centrality(self):
        """Role nodes should be the highest-degree nodes."""
        high_degree_count = 0
        for _ in range(20):
            cc, q, t, ans, m = generate_analogical_transfer_task(
                20, 32, topologies=['ba'],
            )
            assert 'role_degrees' in m
            assert 'avg_degree' in m
            avg_degree = m['avg_degree']
            above_avg = sum(1 for d in m['role_degrees'] if d > avg_degree)
            if above_avg >= 2:
                high_degree_count += 1
        assert high_degree_count >= 14, f"Only {high_degree_count}/20 had high-degree roles"

    def test_three_roles_only(self):
        """Reduced to 3 roles (from 5)."""
        for _ in range(50):
            _, _, _, answer, _ = generate_analogical_transfer_task(20, 32)
            assert 0 <= answer <= 2, f"Got answer={answer}, expected 0-2"

    def test_get_max_classes_analogical(self):
        assert get_max_classes('analogical_transfer') == 3

    def test_metadata_has_role_info(self):
        _, _, _, _, m = generate_analogical_transfer_task(20, 32)
        assert 'role_degrees' in m
        assert 'avg_degree' in m
        assert len(m['role_degrees']) == 3
        assert isinstance(m['avg_degree'], float)


class TestBackwardCompat:
    def test_existing_4_tuple_tasks_unchanged(self):
        ds = BenchmarkDataset(3, 'bfs', 12, 16)
        sample = ds[0]
        assert len(sample) == 4

    def test_diverse_still_4_tuple(self):
        ds = BenchmarkDataset(3, 'diverse', 12, 16)
        sample = ds[0]
        assert len(sample) == 4
