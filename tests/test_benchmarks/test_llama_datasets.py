"""Tests for Llama dataset expansion: enriched domains, generation, mixed sizes, curriculum loading."""

import os
import random
import tempfile

import pytest
import torch

from src.benchmarks.llm_tasks import (
    NODE_LABEL_POOLS,
    ANALOGY_DOMAINS,
    CAUSAL_LABELS,
    generate_labeled_reasoning_task,
    generate_analogical_transfer_task,
    generate_graph_completion_task,
)
from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes


class TestEnrichedDomains:
    """Verify expanded domain pools have correct format and sufficient variety."""

    def test_node_label_pools_count(self):
        assert len(NODE_LABEL_POOLS) >= 8

    def test_node_label_pools_minimum_terms(self):
        for domain, terms in NODE_LABEL_POOLS.items():
            assert len(terms) >= 7, f"Domain '{domain}' has only {len(terms)} terms"

    def test_node_label_pools_no_duplicates(self):
        for domain, terms in NODE_LABEL_POOLS.items():
            assert len(terms) == len(set(terms)), f"Domain '{domain}' has duplicates"

    def test_analogy_domains_count(self):
        assert len(ANALOGY_DOMAINS) >= 2

    def test_analogy_domains_have_roles(self):
        for domain_a, domain_b in ANALOGY_DOMAINS:
            assert len(domain_a) >= 3, f"domain_a has only {len(domain_a)} roles"
            assert len(domain_b) >= 3, f"domain_b has only {len(domain_b)} roles"

    def test_analogy_role_indices_match(self):
        for domain_a, domain_b in ANALOGY_DOMAINS:
            n = len(domain_a)
            expected = set(range(n))
            assert set(domain_a.values()) == expected
            assert set(domain_b.values()) == expected

    def test_new_domains_generate_valid_samples(self):
        domains_seen = set()
        for _ in range(50):
            _, _, _, answer, metadata = generate_labeled_reasoning_task(15, 16)
            assert answer in (0, 1, 2)
            domains_seen.add(metadata['domain'])
        assert len(domains_seen) >= 5, f"Only saw domains: {domains_seen}"

    def test_new_analogy_pairs_generate_valid_samples(self):
        for _ in range(30):
            _, _, _, answer, metadata = generate_analogical_transfer_task(15, 16)
            assert 0 <= answer <= 2
            assert len(metadata['domain_a']) >= 3
            assert len(metadata['domain_b']) >= 3

    def test_enriched_task_prompt_labeled_reasoning(self):
        _, _, _, _, metadata = generate_labeled_reasoning_task(15, 16)
        prompt = metadata['task_prompt']
        assert 'domain=' in prompt
        assert 'task=labeled_reasoning' in prompt
        assert 'edges=' in prompt

    def test_enriched_task_prompt_graph_completion(self):
        _, _, _, _, metadata = generate_graph_completion_task(15, 16)
        prompt = metadata['task_prompt']
        assert 'task=graph_completion' in prompt
        assert 'original_edges=' in prompt

    def test_enriched_task_prompt_analogical_transfer(self):
        _, _, _, _, metadata = generate_analogical_transfer_task(15, 16)
        prompt = metadata['task_prompt']
        assert 'task=analogical_transfer' in prompt
        assert 'assigned_roles=' in prompt


class TestDatasetGeneration:
    """Test dataset save/load and mixed-size generation."""

    def test_save_load_roundtrip_5tuple(self):
        ds = BenchmarkDataset(10, 'labeled_reasoning', 12, 16)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_lr.pt")
            ds.save(path)
            loaded = BenchmarkDataset.load(path)
            assert len(loaded) == 10
            assert loaded.task_type == 'labeled_reasoning'
            sample = loaded.samples[0]
            assert len(sample) == 5
            assert isinstance(sample[4], dict)

    def test_mixed_size_llm_tasks(self):
        ds = BenchmarkDataset(
            20, 'graph_completion', 20, 16,
            n_nodes_range=(12, 30),
        )
        sizes = {ds.samples[i][0].num_cells(0) for i in range(len(ds))}
        assert len(sizes) > 1, f"Expected varied sizes, got {sizes}"

    def test_class_coverage_diverse(self):
        ds = BenchmarkDataset(200, 'diverse', 20, 16)
        answers = {ds.samples[i][3] for i in range(len(ds))}
        assert len(answers) >= 6, f"Only {len(answers)} classes seen"


class TestMixedSizeDatasets:
    """Verify mixed-size datasets have varied node counts."""

    def test_mixed_size_range_respected(self):
        ds = BenchmarkDataset(
            30, 'bfs', 20, 16,
            n_nodes_range=(12, 30),
        )
        for i in range(len(ds)):
            n = ds.samples[i][0].num_cells(0)
            assert 12 <= n <= 30, f"n={n} outside [12, 30]"

    def test_mixed_size_produces_variety(self):
        ds = BenchmarkDataset(
            50, 'hodge_class', 20, 16,
            n_nodes_range=(12, 30),
        )
        sizes = {ds.samples[i][0].num_cells(0) for i in range(len(ds))}
        assert len(sizes) >= 5, f"Only {len(sizes)} distinct sizes in 50 samples"
