"""Tests for GraphParser: NL query -> GraphSpec."""
import json
import pytest
from src.nl_pipeline.data_types import GraphSpec
from src.nl_pipeline.graph_parser import MockGraphParser, parse_graph_json, GRAPH_EXTRACTION_PROMPT


class TestParseGraphJson:
    def test_valid_json(self):
        raw = json.dumps({
            "nodes": [{"name": "a", "type": "node"}, {"name": "b", "type": "node"}],
            "edges": [{"source": "a", "target": "b", "relation": "causes"}],
            "query_node": "a", "target_node": "b", "domain": "technology",
        })
        spec = parse_graph_json(raw)
        assert len(spec.nodes) == 2
        assert len(spec.edges) == 1
        assert spec.query_node == "a"
        assert spec.domain == "technology"

    def test_missing_target_defaults_none(self):
        raw = json.dumps({
            "nodes": [{"name": "x", "type": "entity"}],
            "edges": [], "query_node": "x", "domain": "general",
        })
        spec = parse_graph_json(raw)
        assert spec.target_node is None

    def test_invalid_json_returns_none(self):
        assert parse_graph_json("not valid json {{{") is None

    def test_missing_required_field_returns_none(self):
        raw = json.dumps({"nodes": [], "edges": []})
        assert parse_graph_json(raw) is None


class TestMockGraphParser:
    def test_returns_graph_spec(self):
        parser = MockGraphParser()
        spec = parser.parse("Why does the server crash?")
        assert isinstance(spec, GraphSpec)
        assert len(spec.nodes) >= 2
        assert spec.query_node is not None

    def test_extracts_keywords_as_nodes(self):
        parser = MockGraphParser()
        spec = parser.parse("How does rain cause flooding?")
        names = {n.name for n in spec.nodes}
        assert len(names) >= 2

    def test_always_has_query_node(self):
        parser = MockGraphParser()
        spec = parser.parse("Tell me something.")
        assert spec.query_node in {n.name for n in spec.nodes}


class TestGraphExtractionPrompt:
    def test_prompt_contains_examples(self):
        assert "nodes" in GRAPH_EXTRACTION_PROMPT
        assert "edges" in GRAPH_EXTRACTION_PROMPT
        assert "query_node" in GRAPH_EXTRACTION_PROMPT

    def test_prompt_contains_json(self):
        assert "JSON" in GRAPH_EXTRACTION_PROMPT or "json" in GRAPH_EXTRACTION_PROMPT
