"""Tests for the ConceptNet data pipeline."""

import tempfile
from pathlib import Path

import networkx as nx
import pytest
import torch

from src.data.conceptnet import (
    RELATION_CATEGORIES,
    categorize_relation,
    concept_to_text,
    conceptnet_subgraph_to_cc,
    extract_subgraph,
    load_cached_graph,
    load_conceptnet_graph,
    parse_conceptnet_line,
    save_conceptnet_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_line(rel: str, src: str, tgt: str, weight: float) -> str:
    """Build a fake ConceptNet TSV line."""
    assertion = f"/a/[/r/{rel}/,/c/en/{src}/,/c/en/{tgt}/]"
    metadata = f'{{"weight": {weight}}}'
    return f"{assertion}\t/r/{rel}\t/c/en/{src}\t/c/en/{tgt}\t{metadata}"


def _build_small_graph() -> nx.Graph:
    """Build a small hand-crafted ConceptNet-style graph for testing.

    Graph has 25 nodes and ~35 edges with at least 4 relation types.
    """
    G = nx.Graph()
    # Animals cluster
    animals = [
        ("dog", "animal", "IsA"),
        ("cat", "animal", "IsA"),
        ("bird", "animal", "IsA"),
        ("fish", "animal", "IsA"),
        ("dog", "pet", "IsA"),
        ("cat", "pet", "IsA"),
    ]
    # Properties
    props = [
        ("dog", "loyal", "HasProperty"),
        ("cat", "independent", "HasProperty"),
        ("bird", "feathers", "HasProperty"),
        ("fish", "scales", "HasProperty"),
    ]
    # Locations
    locs = [
        ("dog", "house", "AtLocation"),
        ("cat", "house", "AtLocation"),
        ("bird", "tree", "AtLocation"),
        ("fish", "water", "AtLocation"),
        ("pet", "house", "AtLocation"),
    ]
    # Capabilities
    caps = [
        ("dog", "bark", "CapableOf"),
        ("cat", "purr", "CapableOf"),
        ("bird", "fly", "CapableOf"),
        ("fish", "swim", "CapableOf"),
    ]
    # Causes
    causes = [
        ("bark", "noise", "Causes"),
        ("purr", "comfort", "Causes"),
        ("fly", "freedom", "Causes"),
        ("swim", "exercise", "Causes"),
    ]
    # UsedFor
    used = [
        ("house", "shelter", "UsedFor"),
        ("tree", "shade", "UsedFor"),
        ("water", "drinking", "UsedFor"),
    ]
    # RelatedTo
    related = [
        ("loyal", "trust", "RelatedTo"),
        ("independent", "freedom", "RelatedTo"),
        ("noise", "sound", "RelatedTo"),
        ("comfort", "relaxation", "RelatedTo"),
        ("exercise", "health", "RelatedTo"),
    ]

    for src, tgt, rel in (
        animals + props + locs + caps + causes + used + related
    ):
        G.add_edge(src, tgt, relation=rel, raw_relation=rel, weight=2.0)

    return G


# ---------------------------------------------------------------------------
# parse_conceptnet_line tests
# ---------------------------------------------------------------------------


class TestParseConceptNetLine:
    def test_valid_line(self):
        line = _make_line("IsA", "dog", "animal", 2.0)
        result = parse_conceptnet_line(line)
        assert result is not None
        rel, src, tgt, weight = result
        assert rel == "IsA"
        assert src == "dog"
        assert tgt == "animal"
        assert weight == 2.0

    def test_non_english_source(self):
        """Non-English source concept should be filtered out."""
        line = (
            "/a/[/r/IsA/,/c/fr/chien/,/c/en/animal/]\t"
            "/r/IsA\t/c/fr/chien\t/c/en/animal\t"
            '{"weight": 2.0}'
        )
        assert parse_conceptnet_line(line) is None

    def test_non_english_target(self):
        """Non-English target concept should be filtered out."""
        line = (
            "/a/[/r/IsA/,/c/en/dog/,/c/de/tier/]\t"
            "/r/IsA\t/c/en/dog\t/c/de/tier\t"
            '{"weight": 2.0}'
        )
        assert parse_conceptnet_line(line) is None

    def test_low_weight(self):
        """Edges with weight < 1.0 should be filtered out."""
        line = _make_line("IsA", "dog", "animal", 0.5)
        assert parse_conceptnet_line(line) is None

    def test_exact_weight_one(self):
        """Weight exactly 1.0 should pass the filter."""
        line = _make_line("RelatedTo", "cat", "kitten", 1.0)
        result = parse_conceptnet_line(line)
        assert result is not None
        assert result[3] == 1.0

    def test_empty_line(self):
        assert parse_conceptnet_line("") is None

    def test_malformed_line(self):
        assert parse_conceptnet_line("not\tenough\tfields") is None

    def test_bad_json_metadata(self):
        line = (
            "/a/[/r/IsA/,/c/en/a/,/c/en/b/]\t"
            "/r/IsA\t/c/en/a\t/c/en/b\t"
            "not-json"
        )
        assert parse_conceptnet_line(line) is None


# ---------------------------------------------------------------------------
# categorize_relation tests
# ---------------------------------------------------------------------------


class TestCategorizeRelation:
    def test_all_sixteen_categories_exist(self):
        assert len(RELATION_CATEGORIES) == 16

    def test_isa_variants(self):
        for rel in ["IsA", "DefinedAs", "MannerOf", "InstanceOf"]:
            assert categorize_relation(rel) == "IsA"

    def test_hasa(self):
        assert categorize_relation("HasA") == "HasA"

    def test_partof_variants(self):
        for rel in [
            "PartOf",
            "HasSubevent",
            "HasFirstSubevent",
            "HasLastSubevent",
            "MadeOf",
        ]:
            assert categorize_relation(rel) == "PartOf"

    def test_usedfor(self):
        assert categorize_relation("UsedFor") == "UsedFor"

    def test_capableof_variants(self):
        for rel in ["CapableOf", "ReceivesAction", "NotCapableOf"]:
            assert categorize_relation(rel) == "CapableOf"

    def test_atlocation_variants(self):
        for rel in ["AtLocation", "LocatedNear"]:
            assert categorize_relation(rel) == "AtLocation"

    def test_causes_variants(self):
        for rel in [
            "Causes",
            "HasPrerequisite",
            "MotivatedByGoal",
            "CausesDesire",
            "CreatedBy",
            "Desires",
            "NotDesires",
        ]:
            assert categorize_relation(rel) == "Causes"

    def test_hasproperty_variants(self):
        for rel in ["HasProperty", "NotHasProperty"]:
            assert categorize_relation(rel) == "HasProperty"

    def test_relatedto_variants(self):
        for rel in [
            "RelatedTo",
            "Synonym",
            "Antonym",
            "SimilarTo",
            "DerivedFrom",
            "EtymologicallyRelatedTo",
            "FormOf",
            "DistinctFrom",
            "HasContext",
        ]:
            assert categorize_relation(rel) == "RelatedTo"

    def test_unknown_relation(self):
        assert categorize_relation("CompletelyMadeUp") == "Other"


# ---------------------------------------------------------------------------
# concept_to_text tests
# ---------------------------------------------------------------------------


class TestConceptToText:
    def test_simple(self):
        assert concept_to_text("hot_dog") == "hot dog"

    def test_with_uri_prefix(self):
        assert concept_to_text("/c/en/hot_dog") == "hot dog"

    def test_no_underscores(self):
        assert concept_to_text("dog") == "dog"

    def test_multiple_underscores(self):
        assert concept_to_text("new_york_city") == "new york city"


# ---------------------------------------------------------------------------
# load_conceptnet_graph tests
# ---------------------------------------------------------------------------


class TestLoadConceptNetGraph:
    def test_load_from_tsv(self, tmp_path: Path):
        lines = [
            _make_line("IsA", "dog", "animal", 2.0),
            _make_line("HasProperty", "dog", "loyal", 1.5),
            _make_line("IsA", "cat", "animal", 3.0),
            # Should be filtered (non-English):
            "/a/x\t/r/IsA\t/c/fr/chien\t/c/en/animal\t" + '{"weight": 2.0}',
            # Should be filtered (low weight):
            _make_line("RelatedTo", "dog", "wolf", 0.3),
        ]
        tsv_file = tmp_path / "conceptnet.tsv"
        tsv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        G = load_conceptnet_graph(tsv_file)

        assert G.number_of_nodes() == 4  # dog, animal, loyal, cat
        assert G.number_of_edges() == 3
        assert G.has_edge("dog", "animal")
        assert G["dog"]["animal"]["relation"] == "IsA"

    def test_load_gzip(self, tmp_path: Path):
        import gzip as gz

        lines = [
            _make_line("IsA", "apple", "fruit", 2.0),
            _make_line("AtLocation", "apple", "tree", 1.0),
        ]
        gz_file = tmp_path / "conceptnet.csv.gz"
        with gz.open(gz_file, "wt", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        G = load_conceptnet_graph(gz_file)
        assert G.number_of_nodes() == 3  # apple, fruit, tree
        assert G.number_of_edges() == 2

    def test_self_loops_skipped(self, tmp_path: Path):
        line = _make_line("RelatedTo", "thing", "thing", 5.0)
        tsv_file = tmp_path / "self_loop.tsv"
        tsv_file.write_text(line + "\n", encoding="utf-8")

        G = load_conceptnet_graph(tsv_file)
        assert G.number_of_edges() == 0

    def test_duplicate_edges_keep_highest_weight(self, tmp_path: Path):
        lines = [
            _make_line("IsA", "dog", "animal", 1.0),
            _make_line("RelatedTo", "dog", "animal", 5.0),
        ]
        tsv_file = tmp_path / "dup.tsv"
        tsv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        G = load_conceptnet_graph(tsv_file)
        assert G.number_of_edges() == 1
        assert G["dog"]["animal"]["weight"] == 5.0
        # Higher weight line wins, so relation should be RelatedTo
        assert G["dog"]["animal"]["relation"] == "RelatedTo"


# ---------------------------------------------------------------------------
# extract_subgraph tests
# ---------------------------------------------------------------------------


class TestExtractSubgraph:
    def test_basic_extraction(self):
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="dog", min_nodes=5, max_nodes=15
        )
        assert 5 <= len(node_list) <= 15
        assert "dog" in node_list
        assert sub.number_of_nodes() == len(node_list)
        # Should be connected (BFS from seed)
        assert nx.is_connected(sub)

    def test_max_nodes_respected(self):
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="animal", min_nodes=3, max_nodes=8
        )
        assert len(node_list) <= 8

    def test_seed_not_in_graph(self):
        G = _build_small_graph()
        with pytest.raises(ValueError, match="not in graph"):
            extract_subgraph(G, seed="nonexistent_concept")

    def test_empty_graph(self):
        G = nx.Graph()
        with pytest.raises(ValueError, match="empty"):
            extract_subgraph(G)

    def test_random_seed_selection(self):
        G = _build_small_graph()
        # With no seed, should pick a random high-degree node
        sub, node_list = extract_subgraph(G, min_nodes=3, max_nodes=10)
        assert len(node_list) >= 1
        assert nx.is_connected(sub)

    def test_relation_diversity(self):
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="dog", min_nodes=5, max_nodes=25
        )
        relation_types = set()
        for _, _, data in sub.edges(data=True):
            relation_types.add(data.get("relation"))
        # Our hand-built graph has many relation types near "dog"
        assert len(relation_types) >= 3


# ---------------------------------------------------------------------------
# conceptnet_subgraph_to_cc tests
# ---------------------------------------------------------------------------


class TestConceptNetToCC:
    def test_basic_conversion(self):
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="dog", min_nodes=5, max_nodes=15
        )
        cc, node_map, edge_relations = conceptnet_subgraph_to_cc(
            sub, embedding_dim=16, node_list=node_list
        )

        assert cc.num_cells(0) == len(node_list)
        assert cc.num_cells(1) == sub.number_of_edges()
        assert len(node_map) == len(node_list)
        assert len(edge_relations) == sub.number_of_edges()

    def test_node_texts_set(self):
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="dog", min_nodes=5, max_nodes=15
        )
        cc, node_map, _ = conceptnet_subgraph_to_cc(
            sub, embedding_dim=16, node_list=node_list
        )

        assert len(cc.node_texts) == cc.num_cells(0)
        # "dog" should be in the text list (not "dog" with underscores)
        assert "dog" in cc.node_texts

    def test_structural_only_embeddings(self):
        """Verify that no text leaks into the embedding vectors.

        Embeddings should only contain:
          [0] = normalized degree
          [1] = clustering coefficient
          [2] = 0.0
          [3:] = small noise
        """
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="dog", min_nodes=5, max_nodes=15
        )
        cc, node_map, _ = conceptnet_subgraph_to_cc(
            sub, embedding_dim=16, node_list=node_list
        )

        embeddings = cc.get_embeddings(0)  # (N, 16)
        assert embeddings.shape == (cc.num_cells(0), 16)

        # emb[0] should be normalized degree in [0, 1]
        assert (embeddings[:, 0] >= 0.0).all()
        assert (embeddings[:, 0] <= 1.0).all()

        # emb[1] should be clustering coefficient in [0, 1]
        assert (embeddings[:, 1] >= 0.0).all()
        assert (embeddings[:, 1] <= 1.0).all()

        # emb[2] should be exactly 0.0
        assert (embeddings[:, 2] == 0.0).all()

        # emb[3:] should be small noise (abs < 0.1 with high probability)
        noise = embeddings[:, 3:]
        assert noise.abs().max() < 0.5  # generous bound for small noise

    def test_edge_relations_populated(self):
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="dog", min_nodes=5, max_nodes=15
        )
        _, _, edge_relations = conceptnet_subgraph_to_cc(
            sub, embedding_dim=16, node_list=node_list
        )

        # Every edge relation should be a valid category
        for rel in edge_relations:
            assert rel in RELATION_CATEGORIES

    def test_node_map_keys_are_concepts(self):
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="dog", min_nodes=5, max_nodes=15
        )
        _, node_map, _ = conceptnet_subgraph_to_cc(
            sub, embedding_dim=16, node_list=node_list
        )

        for concept, idx in node_map.items():
            assert isinstance(concept, str)
            assert isinstance(idx, int)
            assert 0 <= idx < len(node_map)

    def test_triangles_filled(self):
        """If the subgraph has triangles, 2-cells should be added."""
        # Build a graph with a guaranteed triangle
        G = nx.Graph()
        G.add_edge("a", "b", relation="IsA", raw_relation="IsA", weight=2.0)
        G.add_edge("b", "c", relation="IsA", raw_relation="IsA", weight=2.0)
        G.add_edge("a", "c", relation="IsA", raw_relation="IsA", weight=2.0)
        # Add extra nodes so it looks more realistic
        G.add_edge("a", "d", relation="HasA", raw_relation="HasA", weight=1.5)
        G.add_edge("b", "e", relation="HasA", raw_relation="HasA", weight=1.5)

        cc, _, _ = conceptnet_subgraph_to_cc(
            G, embedding_dim=16, node_list=["a", "b", "c", "d", "e"]
        )
        assert cc.num_cells(2) >= 1  # at least the a-b-c triangle

    def test_embedding_dim_too_small(self):
        G = _build_small_graph()
        sub, node_list = extract_subgraph(
            G, seed="dog", min_nodes=5, max_nodes=10
        )
        with pytest.raises(AssertionError):
            conceptnet_subgraph_to_cc(sub, embedding_dim=2, node_list=node_list)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_and_load(self, tmp_path: Path):
        G = _build_small_graph()
        pkl_path = tmp_path / "graph.pkl"
        save_conceptnet_graph(G, pkl_path)
        assert pkl_path.exists()

        loaded = load_cached_graph(pkl_path)
        assert loaded.number_of_nodes() == G.number_of_nodes()
        assert loaded.number_of_edges() == G.number_of_edges()

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_cached_graph(tmp_path / "nonexistent.pkl")
