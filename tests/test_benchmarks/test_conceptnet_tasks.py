"""Tests for ConceptNet knowledge graph task generators."""

import random

import networkx as nx
import pytest

from src.benchmarks.conceptnet_tasks import (
    RELATION_CLASSES,
    CONCEPT_CATEGORIES,
    DOMAIN_CLASSES,
    classify_concept,
    classify_domain,
    generate_kg_relation_task,
    generate_kg_concept_task,
    generate_kg_pathvalid_task,
    generate_kg_analogy_task,
    generate_kg_cluster_task,
)
from src.cell_complex.cell_complex import CellComplex


# ---------------------------------------------------------------------------
# Test graph helper
# ---------------------------------------------------------------------------


def _make_test_graph():
    G = nx.Graph()
    concepts = [
        ("dog", "animal", "IsA"), ("cat", "animal", "IsA"),
        ("dog", "park", "AtLocation"), ("cat", "house", "AtLocation"),
        ("park", "tree", "HasA"), ("house", "roof", "HasA"),
        ("dog", "loyalty", "HasProperty"), ("cat", "independence", "HasProperty"),
        ("animal", "living_thing", "IsA"), ("park", "city", "PartOf"),
        ("tree", "leaf", "HasA"), ("roof", "tile", "HasA"),
        ("loyalty", "trust", "RelatedTo"), ("city", "street", "HasA"),
        ("trust", "friend", "RelatedTo"), ("friend", "help", "CapableOf"),
    ]
    for src, tgt, rel in concepts:
        G.add_edge(src, tgt, relation=rel, weight=1.5)
    return G


EMBEDDING_DIM = 16


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_relation_classes_count(self):
        assert len(RELATION_CLASSES) == 10

    def test_concept_categories_count(self):
        assert len(CONCEPT_CATEGORIES) == 9

    def test_domain_classes_count(self):
        assert len(DOMAIN_CLASSES) == 6


# ---------------------------------------------------------------------------
# Task 1: Relation classification
# ---------------------------------------------------------------------------


class TestKgRelation:
    def test_kg_relation_returns_5_tuple(self):
        G = _make_test_graph()
        result = generate_kg_relation_task(G, EMBEDDING_DIM, min_nodes=4, max_nodes=12)
        assert len(result) == 5

    def test_kg_relation_answer_valid_range(self):
        G = _make_test_graph()
        for _ in range(10):
            _, _, _, answer, _ = generate_kg_relation_task(
                G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
            )
            assert 0 <= answer <= 9, f"answer {answer} out of range 0-9"

    def test_kg_relation_types(self):
        G = _make_test_graph()
        cc, q, t, answer, metadata = generate_kg_relation_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert isinstance(cc, CellComplex)
        assert isinstance(q, int)
        assert isinstance(t, int)
        assert isinstance(answer, int)
        assert isinstance(metadata, dict)

    def test_kg_relation_metadata_fields(self):
        G = _make_test_graph()
        _, _, _, _, metadata = generate_kg_relation_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert metadata["task_type"] == "kg_relation"
        assert "node_texts" in metadata
        assert "task_prompt" in metadata
        assert "relation" in metadata


# ---------------------------------------------------------------------------
# Task 2: Concept classification
# ---------------------------------------------------------------------------


class TestKgConcept:
    def test_kg_concept_returns_5_tuple(self):
        G = _make_test_graph()
        result = generate_kg_concept_task(G, EMBEDDING_DIM, min_nodes=4, max_nodes=12)
        assert len(result) == 5

    def test_kg_concept_has_mask_in_node_texts(self):
        G = _make_test_graph()
        cc, q, t, answer, metadata = generate_kg_concept_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert "[MASK]" in cc.node_texts
        assert "[MASK]" in metadata["node_texts"].values()

    def test_kg_concept_answer_valid_range(self):
        G = _make_test_graph()
        for _ in range(10):
            _, _, _, answer, _ = generate_kg_concept_task(
                G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
            )
            assert 0 <= answer <= 8, f"answer {answer} out of range 0-8"

    def test_kg_concept_metadata_fields(self):
        G = _make_test_graph()
        _, _, _, _, metadata = generate_kg_concept_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert metadata["task_type"] == "kg_concept"
        assert "node_texts" in metadata
        assert "task_prompt" in metadata
        assert "masked_concept" in metadata


# ---------------------------------------------------------------------------
# Task 3: Path validity
# ---------------------------------------------------------------------------


class TestKgPathvalid:
    def test_kg_pathvalid_returns_5_tuple(self):
        G = _make_test_graph()
        result = generate_kg_pathvalid_task(G, EMBEDDING_DIM, min_nodes=4, max_nodes=12)
        assert len(result) == 5

    def test_kg_pathvalid_answer_binary(self):
        G = _make_test_graph()
        for _ in range(10):
            _, _, _, answer, _ = generate_kg_pathvalid_task(
                G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
            )
            assert answer in (0, 1), f"answer {answer} not binary"

    def test_kg_pathvalid_types(self):
        G = _make_test_graph()
        cc, q, t, answer, metadata = generate_kg_pathvalid_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert isinstance(cc, CellComplex)
        assert isinstance(q, int)
        assert isinstance(t, int)
        assert isinstance(answer, int)

    def test_kg_pathvalid_metadata_fields(self):
        G = _make_test_graph()
        _, _, _, _, metadata = generate_kg_pathvalid_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert metadata["task_type"] == "kg_pathvalid"
        assert "node_texts" in metadata
        assert "task_prompt" in metadata
        assert "path" in metadata


# ---------------------------------------------------------------------------
# Task 4: Analogy
# ---------------------------------------------------------------------------


class TestKgAnalogy:
    def test_kg_analogy_returns_5_tuple(self):
        G = _make_test_graph()
        result = generate_kg_analogy_task(G, EMBEDDING_DIM, min_nodes=4, max_nodes=12)
        assert len(result) == 5

    def test_kg_analogy_answer_valid_range(self):
        G = _make_test_graph()
        for _ in range(10):
            _, _, _, answer, _ = generate_kg_analogy_task(
                G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
            )
            assert 0 <= answer <= 2, f"answer {answer} out of range 0-2"

    def test_kg_analogy_types(self):
        G = _make_test_graph()
        cc, q, t, answer, metadata = generate_kg_analogy_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert isinstance(cc, CellComplex)
        assert isinstance(q, int)
        assert isinstance(t, int)
        assert isinstance(answer, int)

    def test_kg_analogy_metadata_fields(self):
        G = _make_test_graph()
        _, _, _, _, metadata = generate_kg_analogy_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert metadata["task_type"] == "kg_analogy"
        assert "node_texts" in metadata
        assert "task_prompt" in metadata
        assert "overlap" in metadata

    def test_kg_analogy_merged_node_texts_have_prefix(self):
        G = _make_test_graph()
        _, _, _, _, metadata = generate_kg_analogy_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        texts = metadata["node_texts"]
        has_a = any(k.startswith("a_") for k in texts)
        has_b = any(k.startswith("b_") for k in texts)
        assert has_a, "Expected a_ prefixed nodes in analogy"
        assert has_b, "Expected b_ prefixed nodes in analogy"


# ---------------------------------------------------------------------------
# Task 5: Cluster / domain
# ---------------------------------------------------------------------------


class TestKgCluster:
    def test_kg_cluster_returns_5_tuple(self):
        G = _make_test_graph()
        result = generate_kg_cluster_task(G, EMBEDDING_DIM, min_nodes=4, max_nodes=12)
        assert len(result) == 5

    def test_kg_cluster_answer_valid_range(self):
        G = _make_test_graph()
        for _ in range(10):
            _, _, _, answer, _ = generate_kg_cluster_task(
                G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
            )
            assert 0 <= answer <= 5, f"answer {answer} out of range 0-5"

    def test_kg_cluster_types(self):
        G = _make_test_graph()
        cc, q, t, answer, metadata = generate_kg_cluster_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert isinstance(cc, CellComplex)
        assert isinstance(q, int)
        assert isinstance(t, int)
        assert isinstance(answer, int)

    def test_kg_cluster_metadata_fields(self):
        G = _make_test_graph()
        _, _, _, _, metadata = generate_kg_cluster_task(
            G, EMBEDDING_DIM, min_nodes=4, max_nodes=12,
        )
        assert metadata["task_type"] == "kg_cluster"
        assert "node_texts" in metadata
        assert "task_prompt" in metadata
        assert "true_domain" in metadata


# ---------------------------------------------------------------------------
# Keyword classifiers
# ---------------------------------------------------------------------------


class TestClassifiers:
    def test_classify_concept_animal(self):
        assert classify_concept("dog") == "animal"

    def test_classify_concept_place(self):
        assert classify_concept("city") == "place"

    def test_classify_concept_fallback(self):
        # Unknown concept defaults to abstract
        result = classify_concept("xyzzy_gorp")
        assert result in CONCEPT_CATEGORIES

    def test_classify_domain_science(self):
        assert classify_domain("atom") == "science"

    def test_classify_domain_spatial(self):
        assert classify_domain("mountain") == "spatial"

    def test_classify_domain_fallback(self):
        result = classify_domain("xyzzy_gorp")
        assert result in DOMAIN_CLASSES
