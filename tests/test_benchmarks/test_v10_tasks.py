"""Tests for v10 KG task generators."""

import pytest
import networkx as nx
from src.benchmarks.conceptnet_tasks import (
    generate_kg_transitive_task,
    generate_kg_consistency_task,
    generate_kg_analogy_task_v10,
    generate_kg_causal_chain_task,
    TRANSITIVE_CLASSES,
    CAUSAL_CHAIN_CLASSES,
)


def _make_conceptnet_graph():
    """Create a small ConceptNet-like graph for testing."""
    G = nx.DiGraph()
    # IsA chain: dog -> mammal -> animal
    G.add_edge("dog", "mammal", relation="IsA", weight=2.0)
    G.add_edge("mammal", "animal", relation="IsA", weight=2.0)
    # AtLocation: dog -> park, park -> city
    G.add_edge("dog", "park", relation="AtLocation", weight=1.5)
    G.add_edge("park", "city", relation="PartOf", weight=1.5)
    # More edges for subgraph extraction
    G.add_edge("cat", "mammal", relation="IsA", weight=2.0)
    G.add_edge("bird", "animal", relation="IsA", weight=2.0)
    G.add_edge("robin", "bird", relation="IsA", weight=2.0)
    G.add_edge("fish", "animal", relation="IsA", weight=2.0)
    G.add_edge("dog", "bone", relation="UsedFor", weight=1.0)
    G.add_edge("cat", "house", relation="AtLocation", weight=1.0)
    G.add_edge("bird", "sky", relation="AtLocation", weight=1.0)
    G.add_edge("house", "city", relation="PartOf", weight=1.0)
    G.add_edge("sky", "nature", relation="PartOf", weight=1.0)
    G.add_edge("bone", "dog", relation="UsedFor", weight=1.0)
    G.add_edge("park", "nature", relation="PartOf", weight=1.0)
    # Causal relations
    G.add_edge("rain", "wet", relation="Causes", weight=1.5)
    G.add_edge("wet", "slip", relation="Causes", weight=1.5)
    G.add_edge("rain", "umbrella", relation="UsedFor", weight=1.0)
    G.add_edge("slip", "fall", relation="Causes", weight=1.0)
    return G


class TestKgTransitive:

    def test_transitive_classes_defined(self):
        assert len(TRANSITIVE_CLASSES) == 5
        assert "none" in TRANSITIVE_CLASSES

    def test_generate_returns_5_tuple(self):
        G = _make_conceptnet_graph()
        result = generate_kg_transitive_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        assert result is not None
        cc, query, target, answer, meta = result
        assert 0 <= answer < 5
        assert meta['task_type'] == 'kg_transitive'

    def test_answer_in_range(self):
        G = _make_conceptnet_graph()
        for _ in range(20):
            result = generate_kg_transitive_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
            if result is not None:
                _, _, _, answer, _ = result
                assert 0 <= answer < len(TRANSITIVE_CLASSES)

    def test_metadata_has_chain(self):
        G = _make_conceptnet_graph()
        result = generate_kg_transitive_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        if result is not None:
            _, _, _, _, meta = result
            assert 'chain' in meta


class TestKgConsistency:

    def test_generate_returns_5_tuple(self):
        G = _make_conceptnet_graph()
        result = generate_kg_consistency_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        assert result is not None
        cc, query, target, answer, meta = result
        assert answer in (0, 1)
        assert meta['task_type'] == 'kg_consistency'

    def test_balanced_classes(self):
        G = _make_conceptnet_graph()
        answers = []
        for _ in range(50):
            result = generate_kg_consistency_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
            if result is not None:
                answers.append(result[3])
        if len(answers) > 10:
            assert 0 in answers and 1 in answers

    def test_metadata_has_corruption_info(self):
        G = _make_conceptnet_graph()
        result = generate_kg_consistency_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        if result is not None:
            _, _, _, _, meta = result
            assert 'corrupted' in meta


class TestKgAnalogyV10:

    def test_generate_returns_5_tuple(self):
        G = _make_conceptnet_graph()
        result = generate_kg_analogy_task_v10(G, embedding_dim=32, min_nodes=3, max_nodes=8)
        assert result is not None
        cc, query, target, answer, meta = result
        assert 0 <= answer <= 2
        assert meta['task_type'] == 'kg_analogy'

    def test_metadata_has_subgraph_info(self):
        G = _make_conceptnet_graph()
        result = generate_kg_analogy_task_v10(G, embedding_dim=32, min_nodes=3, max_nodes=8)
        if result is not None:
            _, _, _, _, meta = result
            assert 'subgraph_a_size' in meta
            assert 'subgraph_b_size' in meta


class TestKgCausalChain:

    def test_causal_classes_defined(self):
        assert len(CAUSAL_CHAIN_CLASSES) == 3
        assert "coherent" in CAUSAL_CHAIN_CLASSES
        assert "broken" in CAUSAL_CHAIN_CLASSES
        assert "incoherent" in CAUSAL_CHAIN_CLASSES

    def test_generate_returns_5_tuple(self):
        G = _make_conceptnet_graph()
        result = generate_kg_causal_chain_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        assert result is not None
        cc, query, target, answer, meta = result
        assert 0 <= answer < 3
        assert meta['task_type'] == 'kg_causal_chain'

    def test_metadata_has_chain(self):
        G = _make_conceptnet_graph()
        result = generate_kg_causal_chain_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        if result is not None:
            _, _, _, _, meta = result
            assert 'chain' in meta
            assert 'class_name' in meta
