"""Tests for LLM-dependent benchmark tasks."""

import random

import pytest
import torch

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


class TestAnalogicalTransfer:
    def test_returns_5_tuple(self):
        result = generate_analogical_transfer_task(15, 16)
        assert len(result) == 5

    def test_output_types(self):
        cc, src, tgt, answer, metadata = generate_analogical_transfer_task(15, 16)
        assert isinstance(cc, CellComplex)
        assert isinstance(src, int)
        assert isinstance(tgt, int)
        assert 0 <= answer <= 4
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
            assert 0 <= answer <= 4


class TestRegistryIntegration:
    def test_tasks_registered(self):
        assert 'graph_completion' in TASK_REGISTRY
        assert 'labeled_reasoning' in TASK_REGISTRY
        assert 'analogical_transfer' in TASK_REGISTRY

    def test_max_classes(self):
        assert get_max_classes('graph_completion') == 2
        assert get_max_classes('labeled_reasoning') == 3
        assert get_max_classes('analogical_transfer') == 5

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


class TestBackwardCompat:
    def test_existing_4_tuple_tasks_unchanged(self):
        ds = BenchmarkDataset(3, 'bfs', 12, 16)
        sample = ds[0]
        assert len(sample) == 4

    def test_diverse_still_4_tuple(self):
        ds = BenchmarkDataset(3, 'diverse', 12, 16)
        sample = ds[0]
        assert len(sample) == 4
