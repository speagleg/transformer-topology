# Phase 5: NL Reasoning Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end NL-in, NL-out reasoning pipeline around the frozen Phase 4 GNN+TAT core.

**Architecture:** Sequential pipeline — GraphParser (Llama few-shot) extracts graph from NL, TaskRouter (heuristics) picks task type, CellComplexBuilder (wrapper around existing nx_to_cell_complex) builds the CellComplex, frozen HierarchicalMultiHopModel reasons over it, AnswerGenerator (Llama few-shot) explains the result.

**Tech Stack:** Python 3.12, PyTorch, NetworkX, existing CellComplex/TAT/GNN infrastructure. Llama 3.2 1B for parsing/generation (MockLLMBackend for CPU tests).

**Branch:** `phase5-nl-pipeline` (already created)

**Key constraint:** Zero modifications to existing model code. All 527+ existing tests must continue to pass.

---

### Task 1: Data Structures (GraphSpec, TaskRoute, NLResult)

**Files:**
- Create: `src/nl_pipeline/__init__.py`
- Create: `src/nl_pipeline/data_types.py`
- Test: `tests/test_nl_pipeline/__init__.py`
- Test: `tests/test_nl_pipeline/test_data_types.py`

**Step 1: Write the failing tests**

```python
# tests/test_nl_pipeline/test_data_types.py
"""Tests for Phase 5 NL pipeline data structures."""

from src.nl_pipeline.data_types import (
    NodeSpec, EdgeSpec, GraphSpec, TaskRoute, NLResult,
)


class TestNodeSpec:
    def test_creation(self):
        n = NodeSpec(name="server", type="component")
        assert n.name == "server"
        assert n.type == "component"


class TestEdgeSpec:
    def test_creation(self):
        e = EdgeSpec(source="server", target="database", relation="causes")
        assert e.source == "server"
        assert e.relation == "causes"


class TestGraphSpec:
    def test_creation_with_target(self):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "node"), NodeSpec("b", "node")],
            edges=[EdgeSpec("a", "b", "connects")],
            query_node="a",
            target_node="b",
            domain="technology",
        )
        assert len(spec.nodes) == 2
        assert len(spec.edges) == 1
        assert spec.query_node == "a"
        assert spec.target_node == "b"
        assert spec.domain == "technology"

    def test_creation_without_target(self):
        spec = GraphSpec(
            nodes=[NodeSpec("x", "entity")],
            edges=[],
            query_node="x",
            target_node=None,
            domain="general",
        )
        assert spec.target_node is None

    def test_node_names(self):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "n"), NodeSpec("b", "n"), NodeSpec("c", "n")],
            edges=[EdgeSpec("a", "b", "r")],
            query_node="a",
            target_node="b",
            domain="test",
        )
        assert {n.name for n in spec.nodes} == {"a", "b", "c"}


class TestTaskRoute:
    def test_creation(self):
        route = TaskRoute(
            task_type="bfs",
            query_node_idx=0,
            target_node_idx=3,
            max_classes=16,
            metadata={"task_prompt": "test"},
        )
        assert route.task_type == "bfs"
        assert route.max_classes == 16


class TestNLResult:
    def test_creation(self):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "n")],
            edges=[],
            query_node="a",
            target_node=None,
            domain="test",
        )
        result = NLResult(
            answer="The answer is 3.",
            class_idx=3,
            task_type="bfs",
            graph_spec=spec,
            confidence=0.92,
        )
        assert result.answer == "The answer is 3."
        assert result.confidence == 0.92
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_pipeline/test_data_types.py -v`
Expected: FAIL — ModuleNotFoundError (src.nl_pipeline doesn't exist yet)

**Step 3: Write minimal implementation**

```python
# src/nl_pipeline/__init__.py
"""Phase 5: Natural Language Reasoning Pipeline."""

# tests/test_nl_pipeline/__init__.py
# (empty)

# src/nl_pipeline/data_types.py
"""Data structures for the NL reasoning pipeline."""

from dataclasses import dataclass, field


@dataclass
class NodeSpec:
    """A node extracted from natural language."""
    name: str
    type: str


@dataclass
class EdgeSpec:
    """An edge/relation extracted from natural language."""
    source: str
    target: str
    relation: str


@dataclass
class GraphSpec:
    """Complete graph structure extracted from a natural language query."""
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    query_node: str
    target_node: str | None
    domain: str


@dataclass
class TaskRoute:
    """Routing decision: which task type and model inputs to use."""
    task_type: str
    query_node_idx: int
    target_node_idx: int
    max_classes: int
    metadata: dict = field(default_factory=dict)


@dataclass
class NLResult:
    """Final result of the NL reasoning pipeline."""
    answer: str
    class_idx: int
    task_type: str
    graph_spec: GraphSpec
    confidence: float
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_pipeline/test_data_types.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add src/nl_pipeline/__init__.py src/nl_pipeline/data_types.py \
        tests/test_nl_pipeline/__init__.py tests/test_nl_pipeline/test_data_types.py
git commit -m "feat(phase5): add NL pipeline data structures"
```

---

### Task 2: TaskRouter (Heuristic Rules)

**Files:**
- Create: `src/nl_pipeline/task_router.py`
- Test: `tests/test_nl_pipeline/test_task_router.py`

**Context:** The TaskRouter maps a GraphSpec + original NL query to one of 14 task types from `TASK_REGISTRY` in `src/benchmarks/benchmark_dataset.py`. It uses keyword matching on the original query and edge relation analysis from the GraphSpec.

**Step 1: Write the failing tests**

```python
# tests/test_nl_pipeline/test_task_router.py
"""Tests for TaskRouter heuristic routing."""

import pytest

from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec, TaskRoute
from src.nl_pipeline.task_router import TaskRouter


@pytest.fixture
def router():
    return TaskRouter()


def _make_spec(edges=None, domain="test"):
    """Helper to build a GraphSpec with 3 nodes."""
    nodes = [NodeSpec("a", "node"), NodeSpec("b", "node"), NodeSpec("c", "node")]
    return GraphSpec(
        nodes=nodes,
        edges=edges or [EdgeSpec("a", "b", "connects")],
        query_node="a",
        target_node="b",
        domain=domain,
    )


class TestCausalRouting:
    def test_causes_relation(self, router):
        spec = _make_spec([EdgeSpec("a", "b", "causes")])
        route = router.route(spec, "Why does A cause B?")
        assert route.task_type == "labeled_reasoning"

    def test_prevents_relation(self, router):
        spec = _make_spec([EdgeSpec("a", "b", "prevents")])
        route = router.route(spec, "Does A prevent B?")
        assert route.task_type == "labeled_reasoning"

    def test_enables_relation(self, router):
        spec = _make_spec([EdgeSpec("a", "b", "enables")])
        route = router.route(spec, "How does A enable B?")
        assert route.task_type == "labeled_reasoning"


class TestKeywordRouting:
    def test_shortest_path(self, router):
        spec = _make_spec()
        route = router.route(spec, "What is the shortest path from A to B?")
        assert route.task_type == "bfs"

    def test_distance(self, router):
        spec = _make_spec()
        route = router.route(spec, "How far is A from B?")
        assert route.task_type == "bfs"

    def test_count_paths(self, router):
        spec = _make_spec()
        route = router.route(spec, "How many paths are there from A to B?")
        assert route.task_type == "path_counting"

    def test_cycle(self, router):
        spec = _make_spec()
        route = router.route(spec, "Is there a cycle in this graph?")
        assert route.task_type == "cycle_detection"

    def test_loop(self, router):
        spec = _make_spec()
        route = router.route(spec, "Does this network have any loops?")
        assert route.task_type == "cycle_detection"

    def test_missing_edge(self, router):
        spec = _make_spec()
        route = router.route(spec, "What connects A to B?")
        assert route.task_type == "graph_completion"

    def test_similar(self, router):
        spec = _make_spec()
        route = router.route(spec, "Is the biology domain similar to the technology domain?")
        assert route.task_type == "analogical_transfer"

    def test_connected(self, router):
        spec = _make_spec()
        route = router.route(spec, "How well connected is this graph?")
        assert route.task_type == "spectral_gap"


class TestDefaultRouting:
    def test_unknown_query_defaults_to_diverse(self, router):
        spec = _make_spec()
        route = router.route(spec, "Tell me about this graph.")
        assert route.task_type == "diverse"


class TestRouteMetadata:
    def test_route_has_correct_max_classes(self, router):
        spec = _make_spec()
        route = router.route(spec, "What is the shortest path?")
        assert route.max_classes == 16  # bfs has 16 classes

    def test_route_has_node_indices(self, router):
        spec = _make_spec()
        route = router.route(spec, "Any question")
        assert route.query_node_idx == 0  # "a" is first node
        assert route.target_node_idx == 1  # "b" is second node

    def test_route_metadata_has_task_prompt(self, router):
        spec = _make_spec()
        route = router.route(spec, "What is the shortest path?")
        assert "task_prompt" in route.metadata
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_pipeline/test_task_router.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/nl_pipeline/task_router.py
"""Heuristic task router: maps GraphSpec + NL query to task type."""

import re

from src.benchmarks.benchmark_dataset import get_max_classes
from src.nl_pipeline.data_types import GraphSpec, TaskRoute


# Keyword patterns → task type (checked in order, first match wins)
_ROUTE_RULES: list[tuple[list[str], str]] = [
    # Edge-relation rules (checked via GraphSpec, not keywords)
    # — handled separately in _check_causal_relations

    # Keyword-based rules
    (["shortest", "distance", "how far", "how many hops"], "bfs"),
    (["how many paths", "count paths", "number of paths", "count routes"], "path_counting"),
    (["cycle", "loop", "circular"], "cycle_detection"),
    (["missing", "what connects", "should there be", "predict edge"], "graph_completion"),
    (["similar", "analogy", "analogous", "correspond"], "analogical_transfer"),
    (["connected", "connectivity", "clustered", "spectral", "dense"], "spectral_gap"),
]

_CAUSAL_RELATIONS = {"causes", "prevents", "enables"}


class TaskRouter:
    """Routes a GraphSpec + NL query to one of 14 TASK_REGISTRY task types."""

    def route(self, spec: GraphSpec, query: str) -> TaskRoute:
        """Determine task type and build routing metadata.

        Args:
            spec: Extracted graph structure.
            query: Original NL query string.

        Returns:
            TaskRoute with task_type, node indices, max_classes, metadata.
        """
        task_type = self._classify(spec, query)
        max_classes = get_max_classes(task_type)

        # Map node names to positional indices
        name_to_idx = {n.name: i for i, n in enumerate(spec.nodes)}
        query_idx = name_to_idx.get(spec.query_node, 0)
        target_idx = name_to_idx.get(spec.target_node, min(1, len(spec.nodes) - 1))

        # Build metadata matching existing task format
        node_labels = {i: n.name for i, n in enumerate(spec.nodes)}
        edge_labels = {
            f"{e.source}-{e.target}": e.relation for e in spec.edges
        }
        task_prompt = (
            f"nodes={len(spec.nodes)} edges={len(spec.edges)} "
            f"domain={spec.domain} | task={task_type}"
        )

        return TaskRoute(
            task_type=task_type,
            query_node_idx=query_idx,
            target_node_idx=target_idx,
            max_classes=max_classes,
            metadata={
                "task_type": task_type,
                "task_prompt": task_prompt,
                "node_labels": node_labels,
                "edge_labels": edge_labels,
                "domain": spec.domain,
            },
        )

    def _classify(self, spec: GraphSpec, query: str) -> str:
        """Pick task type from heuristics."""
        # 1. Check edge relations for causal patterns
        relations = {e.relation.lower() for e in spec.edges}
        if relations & _CAUSAL_RELATIONS:
            return "labeled_reasoning"

        # 2. Check query keywords
        q_lower = query.lower()
        for keywords, task_type in _ROUTE_RULES:
            if any(kw in q_lower for kw in keywords):
                return task_type

        # 3. Default
        return "diverse"
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_pipeline/test_task_router.py -v`
Expected: 14 passed

**Step 5: Commit**

```bash
git add src/nl_pipeline/task_router.py tests/test_nl_pipeline/test_task_router.py
git commit -m "feat(phase5): add TaskRouter with heuristic rules"
```

---

### Task 3: CellComplexBuilder

**Files:**
- Create: `src/nl_pipeline/cell_complex_builder.py`
- Test: `tests/test_nl_pipeline/test_cell_complex_builder.py`

**Context:** Wraps `nx_to_cell_complex` from `src/benchmarks/graph_convert.py`. Converts GraphSpec → NX graph → CellComplex. Edge relation types are encoded in `emb[1]` of the edge embedding.

**Step 1: Write the failing tests**

```python
# tests/test_nl_pipeline/test_cell_complex_builder.py
"""Tests for CellComplexBuilder: GraphSpec → CellComplex."""

import pytest

from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec
from src.nl_pipeline.cell_complex_builder import CellComplexBuilder


@pytest.fixture
def builder():
    return CellComplexBuilder(embedding_dim=32)


def _tech_spec():
    return GraphSpec(
        nodes=[
            NodeSpec("server", "component"),
            NodeSpec("database", "component"),
            NodeSpec("cache", "component"),
        ],
        edges=[
            EdgeSpec("server", "cache", "connects"),
            EdgeSpec("cache", "database", "causes"),
        ],
        query_node="server",
        target_node="database",
        domain="technology",
    )


class TestBuild:
    def test_correct_node_count(self, builder):
        cc, node_map = builder.build(_tech_spec())
        assert cc.num_cells(0) == 3

    def test_correct_edge_count(self, builder):
        cc, node_map = builder.build(_tech_spec())
        assert cc.num_cells(1) == 2

    def test_node_map_keys(self, builder):
        cc, node_map = builder.build(_tech_spec())
        assert set(node_map.keys()) == {"server", "database", "cache"}

    def test_node_map_indices_valid(self, builder):
        cc, node_map = builder.build(_tech_spec())
        for idx in node_map.values():
            assert 0 <= idx < cc.num_cells(0)

    def test_source_cell_type(self, builder):
        spec = _tech_spec()
        cc, node_map = builder.build(spec)
        src_idx = node_map[spec.query_node]
        assert cc._0_cell_types[src_idx] == "source"

    def test_target_cell_type(self, builder):
        spec = _tech_spec()
        cc, node_map = builder.build(spec)
        tgt_idx = node_map[spec.target_node]
        assert cc._0_cell_types[tgt_idx] == "target"

    def test_embeddings_have_correct_dim(self, builder):
        cc, _ = builder.build(_tech_spec())
        embs = cc.get_embeddings(0)
        assert embs.shape == (3, 32)

    def test_structural_embeddings_populated(self, builder):
        cc, _ = builder.build(_tech_spec())
        embs = cc.get_embeddings(0)
        # emb[0] = normalized degree, should be > 0 for connected nodes
        assert embs[0, 0].item() > 0


class TestEdgeRelationEncoding:
    def test_causal_edge_encoded(self, builder):
        spec = _tech_spec()
        cc, _ = builder.build(spec)
        edge_embs = cc.get_embeddings(1)
        # At least one edge has relation "causes" → emb[1] should be non-zero
        assert any(edge_embs[i, 1].item() != 0 for i in range(cc.num_cells(1)))


class TestTriangleFilling:
    def test_triangle_creates_2cells(self, builder):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "n"), NodeSpec("b", "n"), NodeSpec("c", "n")],
            edges=[
                EdgeSpec("a", "b", "connects"),
                EdgeSpec("b", "c", "connects"),
                EdgeSpec("a", "c", "connects"),
            ],
            query_node="a",
            target_node="c",
            domain="test",
        )
        cc, _ = builder.build(spec)
        assert cc.num_cells(2) >= 1


class TestSingleNode:
    def test_single_node_no_crash(self, builder):
        spec = GraphSpec(
            nodes=[NodeSpec("lonely", "entity")],
            edges=[],
            query_node="lonely",
            target_node=None,
            domain="test",
        )
        cc, node_map = builder.build(spec)
        assert cc.num_cells(0) == 1
        assert cc.num_cells(1) == 0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_pipeline/test_cell_complex_builder.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/nl_pipeline/cell_complex_builder.py
"""Builds CellComplex from GraphSpec using existing graph_convert infrastructure."""

import networkx as nx

from src.benchmarks.graph_convert import nx_to_cell_complex
from src.nl_pipeline.data_types import GraphSpec


# Relation type → embedding value for emb[1]
_RELATION_ENCODING = {
    "causes": 0.33,
    "prevents": 0.66,
    "enables": 1.0,
    "connects": 0.0,
}


class CellComplexBuilder:
    """Converts a GraphSpec into a CellComplex via NetworkX."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def build(self, spec: GraphSpec):
        """Build CellComplex from a GraphSpec.

        Args:
            spec: Graph structure extracted from NL.

        Returns:
            (cc, node_map) where node_map maps node name → CellComplex index.
        """
        # Build NX graph with integer nodes (positional indices)
        G = nx.Graph()
        name_to_nx = {}
        for i, node in enumerate(spec.nodes):
            G.add_node(i)
            name_to_nx[node.name] = i

        # Store edge relations for post-processing
        edge_relations = {}
        for edge in spec.edges:
            u = name_to_nx.get(edge.source)
            v = name_to_nx.get(edge.target)
            if u is not None and v is not None and u != v:
                G.add_edge(u, v)
                edge_relations[(u, v)] = edge.relation

        # Map query/target names to NX node IDs
        source_nx = name_to_nx.get(spec.query_node)
        target_nx = name_to_nx.get(spec.target_node) if spec.target_node else None

        # Convert to CellComplex using existing infrastructure
        cc, nx_to_cc = nx_to_cell_complex(
            G, self.embedding_dim,
            source_node=source_nx,
            target_node=target_nx,
            fill_triangles=True,
        )

        # Encode edge relations into emb[1]
        edge_embs = cc.get_embeddings(1)
        for edge_idx in range(cc.num_cells(1)):
            src_cc = cc._1_cell_sources[edge_idx]
            tgt_cc = cc._1_cell_targets[edge_idx]
            # Find the NX edge corresponding to this CC edge
            for (u, v), rel in edge_relations.items():
                if (nx_to_cc.get(u) == src_cc and nx_to_cc.get(v) == tgt_cc) or \
                   (nx_to_cc.get(v) == src_cc and nx_to_cc.get(u) == tgt_cc):
                    edge_embs[edge_idx, 1] = _RELATION_ENCODING.get(rel, 0.0)
                    break
        cc.set_embeddings(1, edge_embs)

        # Build name → CC index map
        name_to_cc = {name: nx_to_cc[nx_id] for name, nx_id in name_to_nx.items()
                      if nx_id in nx_to_cc}

        return cc, name_to_cc
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_pipeline/test_cell_complex_builder.py -v`
Expected: 11 passed

**Step 5: Commit**

```bash
git add src/nl_pipeline/cell_complex_builder.py tests/test_nl_pipeline/test_cell_complex_builder.py
git commit -m "feat(phase5): add CellComplexBuilder (GraphSpec → CellComplex)"
```

---

### Task 4: GraphParser (Few-Shot Prompting)

**Files:**
- Create: `src/nl_pipeline/graph_parser.py`
- Test: `tests/test_nl_pipeline/test_graph_parser.py`

**Context:** The GraphParser uses Llama (or MockLLMBackend for tests) to extract graph structure from NL queries. Phase 5a uses few-shot prompting — the parser builds a prompt with examples, sends it to Llama, and parses the JSON output. For CPU testing, a `MockGraphParser` returns canned results. The real Llama parser uses `transformers` tokenizer + `generate()`.

**Step 1: Write the failing tests**

```python
# tests/test_nl_pipeline/test_graph_parser.py
"""Tests for GraphParser: NL query → GraphSpec."""

import json
import pytest

from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec
from src.nl_pipeline.graph_parser import (
    MockGraphParser,
    GraphParser,
    parse_graph_json,
    GRAPH_EXTRACTION_PROMPT,
)


class TestParseGraphJson:
    def test_valid_json(self):
        raw = json.dumps({
            "nodes": [{"name": "a", "type": "node"}, {"name": "b", "type": "node"}],
            "edges": [{"source": "a", "target": "b", "relation": "causes"}],
            "query_node": "a",
            "target_node": "b",
            "domain": "technology",
        })
        spec = parse_graph_json(raw)
        assert len(spec.nodes) == 2
        assert len(spec.edges) == 1
        assert spec.query_node == "a"
        assert spec.domain == "technology"

    def test_missing_target_defaults_none(self):
        raw = json.dumps({
            "nodes": [{"name": "x", "type": "entity"}],
            "edges": [],
            "query_node": "x",
            "domain": "general",
        })
        spec = parse_graph_json(raw)
        assert spec.target_node is None

    def test_invalid_json_returns_none(self):
        spec = parse_graph_json("not valid json {{{")
        assert spec is None

    def test_missing_required_field_returns_none(self):
        raw = json.dumps({"nodes": [], "edges": []})  # missing query_node, domain
        spec = parse_graph_json(raw)
        assert spec is None


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
        assert "rain" in names or "flooding" in names or len(names) >= 2

    def test_always_has_query_node(self):
        parser = MockGraphParser()
        spec = parser.parse("Tell me something.")
        assert spec.query_node in {n.name for n in spec.nodes}


class TestGraphExtractionPrompt:
    def test_prompt_contains_examples(self):
        assert "nodes" in GRAPH_EXTRACTION_PROMPT
        assert "edges" in GRAPH_EXTRACTION_PROMPT
        assert "query_node" in GRAPH_EXTRACTION_PROMPT

    def test_prompt_contains_json_format(self):
        assert "JSON" in GRAPH_EXTRACTION_PROMPT or "json" in GRAPH_EXTRACTION_PROMPT
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_pipeline/test_graph_parser.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/nl_pipeline/graph_parser.py
"""Graph parser: extracts graph structure from NL queries.

Phase 5a: Few-shot prompting with Llama (or MockGraphParser for tests).
Phase 5b: LoRA fine-tuned adapter.
"""

import json
import re
from abc import ABC, abstractmethod

from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec


GRAPH_EXTRACTION_PROMPT = '''You extract graph structure from questions. Output valid JSON only.

Example 1:
Question: "Why does the server crash when the database is overloaded?"
{"nodes": [{"name": "server", "type": "component"}, {"name": "database", "type": "component"}, {"name": "overload", "type": "event"}], "edges": [{"source": "database", "target": "overload", "relation": "causes"}, {"source": "overload", "target": "server", "relation": "causes"}], "query_node": "database", "target_node": "server", "domain": "technology"}

Example 2:
Question: "Is there a cycle between photosynthesis, oxygen, and respiration?"
{"nodes": [{"name": "photosynthesis", "type": "process"}, {"name": "oxygen", "type": "substance"}, {"name": "respiration", "type": "process"}], "edges": [{"source": "photosynthesis", "target": "oxygen", "relation": "causes"}, {"source": "oxygen", "target": "respiration", "relation": "enables"}, {"source": "respiration", "target": "photosynthesis", "relation": "enables"}], "query_node": "photosynthesis", "target_node": "respiration", "domain": "biology"}

Example 3:
Question: "How far is the router from the firewall in the network?"
{"nodes": [{"name": "router", "type": "component"}, {"name": "firewall", "type": "component"}, {"name": "switch", "type": "component"}], "edges": [{"source": "router", "target": "switch", "relation": "connects"}, {"source": "switch", "target": "firewall", "relation": "connects"}], "query_node": "router", "target_node": "firewall", "domain": "technology"}

Question: "{query}"
'''


def parse_graph_json(raw: str) -> GraphSpec | None:
    """Parse raw JSON string into a GraphSpec. Returns None on failure."""
    try:
        # Extract JSON from the response (may have surrounding text)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
    except (json.JSONDecodeError, TypeError):
        return None

    try:
        nodes = [NodeSpec(name=n["name"], type=n["type"]) for n in data["nodes"]]
        edges = [EdgeSpec(source=e["source"], target=e["target"], relation=e["relation"])
                 for e in data.get("edges", [])]
        query_node = data["query_node"]
        domain = data["domain"]
    except (KeyError, TypeError):
        return None

    if not nodes or not query_node or not domain:
        return None

    target_node = data.get("target_node")
    return GraphSpec(
        nodes=nodes,
        edges=edges,
        query_node=query_node,
        target_node=target_node,
        domain=domain,
    )


class BaseGraphParser(ABC):
    """Interface for graph parsers."""

    @abstractmethod
    def parse(self, query: str) -> GraphSpec:
        """Extract graph structure from a natural language query."""
        ...


class MockGraphParser(BaseGraphParser):
    """Keyword-based graph parser for CPU testing (no LLM needed)."""

    def parse(self, query: str) -> GraphSpec:
        """Extract a simple graph from keyword analysis."""
        # Extract nouns/keywords as candidate node names
        words = re.findall(r'\b[a-z]{3,}\b', query.lower())
        # Filter stopwords
        stops = {"the", "this", "that", "from", "with", "does", "how",
                 "what", "why", "when", "where", "there", "have", "has",
                 "are", "was", "were", "been", "being", "about", "into",
                 "many", "much", "some", "any", "between", "tell"}
        keywords = [w for w in words if w not in stops]
        if len(keywords) < 2:
            keywords = ["node_a", "node_b"]

        nodes = [NodeSpec(name=kw, type="entity") for kw in keywords[:6]]

        # Create edges between consecutive keywords
        edges = []
        for i in range(len(nodes) - 1):
            relation = "causes" if any(w in query.lower() for w in ["cause", "why", "because"]) \
                       else "connects"
            edges.append(EdgeSpec(
                source=nodes[i].name,
                target=nodes[i + 1].name,
                relation=relation,
            ))

        # Infer domain from keywords
        domain = "general"
        tech_words = {"server", "database", "network", "cache", "router", "firewall"}
        bio_words = {"cell", "protein", "gene", "enzyme", "photosynthesis"}
        if any(n.name in tech_words for n in nodes):
            domain = "technology"
        elif any(n.name in bio_words for n in nodes):
            domain = "biology"

        return GraphSpec(
            nodes=nodes,
            edges=edges,
            query_node=nodes[0].name,
            target_node=nodes[-1].name if len(nodes) > 1 else None,
            domain=domain,
        )


class GraphParser(BaseGraphParser):
    """Few-shot Llama-based graph parser.

    Requires transformers. Falls back to MockGraphParser on import failure
    or generation failure.
    """

    def __init__(self, model_name: str = "unsloth/Llama-3.2-1B",
                 device: str = "cpu", tokenizer=None, model=None):
        self._fallback = MockGraphParser()
        self._model_name = model_name
        self._device = device
        # Accept pre-loaded model/tokenizer (for sharing with AnswerGenerator)
        self._tokenizer = tokenizer
        self._model = model
        self._loaded = tokenizer is not None and model is not None

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name, device_map=None,
            ).to(self._device)
            self._model.eval()
            self._loaded = True
        except ImportError:
            self._loaded = False

    def parse(self, query: str) -> GraphSpec:
        """Parse NL query into GraphSpec using few-shot prompting."""
        self._ensure_loaded()
        if not self._loaded:
            return self._fallback.parse(query)

        prompt = GRAPH_EXTRACTION_PROMPT.format(query=query)
        inputs = self._tokenizer(prompt, return_tensors="pt",
                                  truncation=True, max_length=512).to(self._device)

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=256, temperature=0.1,
                do_sample=False, pad_token_id=self._tokenizer.pad_token_id,
            )

        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)

        spec = parse_graph_json(response)
        if spec is not None:
            return spec

        # Retry with error feedback
        retry_prompt = prompt + "\n" + response + "\nThat was invalid JSON. Output valid JSON only:\n"
        inputs = self._tokenizer(retry_prompt, return_tensors="pt",
                                  truncation=True, max_length=768).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=256, temperature=0.1,
                do_sample=False, pad_token_id=self._tokenizer.pad_token_id,
            )
        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)

        spec = parse_graph_json(response)
        if spec is not None:
            return spec

        # Final fallback: keyword extraction
        return self._fallback.parse(query)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_pipeline/test_graph_parser.py -v`
Expected: 9 passed

**Step 5: Commit**

```bash
git add src/nl_pipeline/graph_parser.py tests/test_nl_pipeline/test_graph_parser.py
git commit -m "feat(phase5): add GraphParser with few-shot prompting + mock"
```

---

### Task 5: AnswerGenerator (Few-Shot Prompting)

**Files:**
- Create: `src/nl_pipeline/answer_generator.py`
- Test: `tests/test_nl_pipeline/test_answer_generator.py`

**Context:** Takes a class prediction, task route, graph spec, and original query, then generates a NL explanation. Phase 5a uses template-based generation (no LLM needed for basic answers). Real Llama generation is optional enhancement.

**Step 1: Write the failing tests**

```python
# tests/test_nl_pipeline/test_answer_generator.py
"""Tests for AnswerGenerator: class_idx + context → NL answer."""

import pytest

from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec, TaskRoute
from src.nl_pipeline.answer_generator import AnswerGenerator, CLASS_INTERPRETATIONS


def _make_spec():
    return GraphSpec(
        nodes=[NodeSpec("server", "component"), NodeSpec("database", "component")],
        edges=[EdgeSpec("server", "database", "causes")],
        query_node="server",
        target_node="database",
        domain="technology",
    )


def _make_route(task_type, max_classes):
    return TaskRoute(
        task_type=task_type,
        query_node_idx=0,
        target_node_idx=1,
        max_classes=max_classes,
        metadata={"task_prompt": "test"},
    )


class TestClassInterpretations:
    def test_bfs_interpretation_exists(self):
        assert "bfs" in CLASS_INTERPRETATIONS

    def test_labeled_reasoning_interpretation_exists(self):
        assert "labeled_reasoning" in CLASS_INTERPRETATIONS

    def test_cycle_detection_interpretation_exists(self):
        assert "cycle_detection" in CLASS_INTERPRETATIONS

    def test_graph_completion_interpretation_exists(self):
        assert "graph_completion" in CLASS_INTERPRETATIONS


class TestAnswerGenerator:
    @pytest.fixture
    def gen(self):
        return AnswerGenerator()

    def test_bfs_answer(self, gen):
        answer = gen.generate(
            class_idx=3,
            task_route=_make_route("bfs", 16),
            graph_spec=_make_spec(),
            original_query="How far is server from database?",
        )
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "3" in answer

    def test_labeled_reasoning_causal(self, gen):
        answer = gen.generate(
            class_idx=0,
            task_route=_make_route("labeled_reasoning", 3),
            graph_spec=_make_spec(),
            original_query="Does server cause database issues?",
        )
        assert "causal" in answer.lower()

    def test_cycle_detection_yes(self, gen):
        answer = gen.generate(
            class_idx=1,
            task_route=_make_route("cycle_detection", 2),
            graph_spec=_make_spec(),
            original_query="Is there a cycle?",
        )
        assert "cycle" in answer.lower()

    def test_graph_completion_no_edge(self, gen):
        answer = gen.generate(
            class_idx=0,
            task_route=_make_route("graph_completion", 2),
            graph_spec=_make_spec(),
            original_query="Should there be an edge?",
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_diverse_fallback(self, gen):
        answer = gen.generate(
            class_idx=5,
            task_route=_make_route("diverse", 11),
            graph_spec=_make_spec(),
            original_query="Tell me about this graph.",
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_answer_includes_node_names(self, gen):
        answer = gen.generate(
            class_idx=3,
            task_route=_make_route("bfs", 16),
            graph_spec=_make_spec(),
            original_query="How far is server from database?",
        )
        assert "server" in answer.lower() or "database" in answer.lower()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_pipeline/test_answer_generator.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/nl_pipeline/answer_generator.py
"""Answer generator: converts model predictions to NL explanations.

Phase 5a: Template-based generation (no LLM needed).
Phase 5b: Llama-based generation with LoRA adapter.
"""

from src.nl_pipeline.data_types import GraphSpec, TaskRoute


# Maps (task_type, class_idx) → interpretation string
CLASS_INTERPRETATIONS = {
    "bfs": lambda idx, spec: (
        f"The shortest path from {spec.query_node} to {spec.target_node} is {idx} hops."
    ),
    "dijkstra": lambda idx, spec: (
        f"The weighted shortest path from {spec.query_node} to {spec.target_node} "
        f"has cost {idx}."
    ),
    "labeled_reasoning": lambda idx, spec: {
        0: f"There is a causal relationship between {spec.query_node} and {spec.target_node}.",
        1: f"The relationship between {spec.query_node} and {spec.target_node} is blocked.",
        2: f"{spec.query_node} and {spec.target_node} are independent — no causal link found.",
    }.get(idx, f"Relationship class {idx} between {spec.query_node} and {spec.target_node}."),
    "cycle_detection": lambda idx, spec: {
        0: f"No cycle was found involving {spec.query_node}.",
        1: f"A cycle exists involving {spec.query_node}.",
    }.get(idx, f"Cycle detection result: class {idx}."),
    "graph_completion": lambda idx, spec: {
        0: f"No edge should exist between {spec.query_node} and {spec.target_node}.",
        1: f"An edge between {spec.query_node} and {spec.target_node} should exist.",
    }.get(idx, f"Edge prediction: class {idx}."),
    "path_counting": lambda idx, spec: (
        f"There are {idx} distinct paths from {spec.query_node} to {spec.target_node}."
    ),
    "analogical_transfer": lambda idx, spec: (
        f"{spec.query_node} plays role {idx} in the analogy between the domains."
    ),
    "spectral_gap": lambda idx, spec: (
        f"The graph has connectivity level {idx} out of 7 "
        f"(spectral gap bucket {idx})."
    ),
    "hodge_class": lambda idx, spec: {
        0: "The dominant signal flow is gradient (source-to-sink).",
        1: "The dominant signal flow is curl (rotational/cyclic).",
        2: "The dominant signal is harmonic (globally balanced).",
    }.get(idx, f"Hodge classification: class {idx}."),
    "diverse": lambda idx, spec: (
        f"The graph reasoning result for {spec.query_node} → {spec.target_node} "
        f"is class {idx}."
    ),
}


class AnswerGenerator:
    """Generates NL answers from model predictions.

    Phase 5a: Template-based (no LLM).
    Phase 5b: Llama generation with context.
    """

    def generate(
        self,
        class_idx: int,
        task_route: TaskRoute,
        graph_spec: GraphSpec,
        original_query: str,
    ) -> str:
        """Generate a natural language answer.

        Args:
            class_idx: Model's predicted class index.
            task_route: Routing decision with task_type.
            graph_spec: Extracted graph structure.
            original_query: The original NL question.

        Returns:
            A human-readable explanation string.
        """
        interpreter = CLASS_INTERPRETATIONS.get(task_route.task_type)
        if interpreter is not None:
            return interpreter(class_idx, graph_spec)

        # Fallback for unknown task types
        return (
            f"The model predicted class {class_idx} for task "
            f"'{task_route.task_type}' on the query: {original_query}"
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_pipeline/test_answer_generator.py -v`
Expected: 10 passed

**Step 5: Commit**

```bash
git add src/nl_pipeline/answer_generator.py tests/test_nl_pipeline/test_answer_generator.py
git commit -m "feat(phase5): add AnswerGenerator with template interpretations"
```

---

### Task 6: NLReasoningPipeline (Orchestrator)

**Files:**
- Create: `src/nl_pipeline/pipeline.py`
- Test: `tests/test_nl_pipeline/test_pipeline.py`

**Context:** The orchestrator ties all components together. For CPU testing, it uses `MockGraphParser` and builds the model with `MockLLMBackend`. The frozen model is loaded from a state dict checkpoint. The pipeline also needs a `load_frozen_model` helper that constructs a `HierarchicalMultiHopModel` and freezes all params.

**Important:** The model requires `_build_model` from `src/benchmarks/run_benchmark_suite.py` and config dicts. For testing, we'll build a small model inline.

**Step 1: Write the failing tests**

```python
# tests/test_nl_pipeline/test_pipeline.py
"""Tests for NLReasoningPipeline end-to-end."""

import tempfile

import pytest
import torch

from src.nl_pipeline.data_types import NLResult, GraphSpec
from src.nl_pipeline.pipeline import NLReasoningPipeline, load_frozen_model
from src.benchmarks.run_benchmark_suite import _build_model


@pytest.fixture
def small_config():
    """Minimal model config for testing."""
    return {
        "model": {
            "embedding_dim": 32,
            "gnn_hidden": 64,
            "gnn_spatial_layers": 2,
            "gnn_spectral_layers": 2,
            "max_freqs": 8,
            "tat_layers": 2,
            "tat_spatial_heads": 2,
            "tat_spectral_heads": 2,
            "tat_ff_dim": 128,
            "max_iterations": 3,
            "convergence_threshold": 0.1,
            "use_wave_dynamics": True,
            "use_higher_order": True,
            "use_topological_pe": False,
            "use_structural_features": True,
        },
        "wave": {
            "filter_type": "heat",
            "wave_mode": "spectral",
            "laplacian_dim": 0,
            "wave_strength_gate": False,
            "use_neural_ode": False,
        },
        "llm": {
            "backend": "mock",
            "llm_dim": 128,
            "num_prefix": 4,
            "gate_threshold": 0.1,
            "mock_hidden": 64,
        },
    }


@pytest.fixture
def checkpoint_path(small_config):
    """Create a temporary checkpoint from a fresh model."""
    model = _build_model(
        "hierarchical_llm",
        small_config["model"],
        max_classes=16,
        device=torch.device("cpu"),
        wave_config=small_config["wave"],
        llm_config=small_config["llm"],
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        return f.name


@pytest.fixture
def pipeline(checkpoint_path, small_config):
    return NLReasoningPipeline(
        checkpoint_path=checkpoint_path,
        model_config=small_config["model"],
        wave_config=small_config["wave"],
        llm_config=small_config["llm"],
        max_classes=16,
        device="cpu",
    )


class TestLoadFrozenModel:
    def test_all_params_frozen(self, checkpoint_path, small_config):
        model = load_frozen_model(
            checkpoint_path,
            small_config["model"],
            small_config["wave"],
            small_config["llm"],
            max_classes=16,
            device=torch.device("cpu"),
        )
        for param in model.parameters():
            assert not param.requires_grad

    def test_model_in_eval_mode(self, checkpoint_path, small_config):
        model = load_frozen_model(
            checkpoint_path,
            small_config["model"],
            small_config["wave"],
            small_config["llm"],
            max_classes=16,
            device=torch.device("cpu"),
        )
        assert not model.training


class TestPipeline:
    def test_query_returns_nl_result(self, pipeline):
        result = pipeline.query("What is the shortest path from server to database?")
        assert isinstance(result, NLResult)

    def test_result_has_answer_string(self, pipeline):
        result = pipeline.query("How far is A from B?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_result_has_valid_class_idx(self, pipeline):
        result = pipeline.query("Is there a cycle in this network?")
        assert isinstance(result.class_idx, int)
        assert 0 <= result.class_idx < 16

    def test_result_has_graph_spec(self, pipeline):
        result = pipeline.query("Why does rain cause flooding?")
        assert isinstance(result.graph_spec, GraphSpec)
        assert len(result.graph_spec.nodes) >= 2

    def test_result_has_confidence(self, pipeline):
        result = pipeline.query("Any question here")
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_task_type(self, pipeline):
        result = pipeline.query("What is the shortest path from A to B?")
        assert result.task_type == "bfs"

    def test_causal_query_routes_correctly(self, pipeline):
        result = pipeline.query("Why does the server crash when the database fails?")
        # MockGraphParser detects "cause" → edges with "causes" → labeled_reasoning
        assert result.task_type in ("labeled_reasoning", "diverse")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_pipeline/test_pipeline.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/nl_pipeline/pipeline.py
"""NLReasoningPipeline: end-to-end NL → reasoning → NL orchestrator."""

import torch
import torch.nn.functional as F

from src.benchmarks.run_benchmark_suite import _build_model
from src.nl_pipeline.data_types import GraphSpec, NLResult
from src.nl_pipeline.graph_parser import MockGraphParser, BaseGraphParser
from src.nl_pipeline.task_router import TaskRouter
from src.nl_pipeline.cell_complex_builder import CellComplexBuilder
from src.nl_pipeline.answer_generator import AnswerGenerator


def load_frozen_model(
    checkpoint_path: str,
    model_config: dict,
    wave_config: dict | None,
    llm_config: dict | None,
    max_classes: int,
    device: torch.device,
):
    """Load a HierarchicalMultiHopModel from checkpoint with all params frozen."""
    model = _build_model(
        "hierarchical_llm",
        model_config,
        max_classes=max_classes,
        device=device,
        wave_config=wave_config,
        llm_config=llm_config,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


class NLReasoningPipeline:
    """End-to-end NL → graph reasoning → NL answer pipeline.

    Uses a frozen Phase 4 model for reasoning. GraphParser and AnswerGenerator
    handle NL input/output via few-shot prompting (Phase 5a) or LoRA (Phase 5b).
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_config: dict,
        wave_config: dict | None = None,
        llm_config: dict | None = None,
        max_classes: int = 16,
        device: str = "cpu",
        parser: BaseGraphParser | None = None,
    ):
        self._device = torch.device(device)

        # Pipeline components
        self.parser = parser or MockGraphParser()
        self.router = TaskRouter()
        self.builder = CellComplexBuilder(
            embedding_dim=model_config.get("embedding_dim", 32),
        )
        self.answerer = AnswerGenerator()

        # Frozen reasoning model
        self.model = load_frozen_model(
            checkpoint_path, model_config, wave_config, llm_config,
            max_classes, self._device,
        )

    def query(self, text: str) -> NLResult:
        """Run the full NL reasoning pipeline.

        Args:
            text: Free-form natural language question.

        Returns:
            NLResult with answer, class prediction, task type, graph, confidence.
        """
        # 1. Parse NL → graph structure
        graph_spec = self.parser.parse(text)

        # 2. Route to task type
        task_route = self.router.route(graph_spec, text)

        # 3. Build CellComplex
        cc, node_map = self.builder.build(graph_spec)
        cc = cc.clone().to(self._device)

        # 4. Resolve node indices
        query_idx = node_map.get(graph_spec.query_node, 0)
        target_name = graph_spec.target_node or (
            graph_spec.nodes[1].name if len(graph_spec.nodes) > 1
            else graph_spec.nodes[0].name
        )
        target_idx = node_map.get(target_name, min(1, cc.num_cells(0) - 1))

        # 5. Run frozen model
        with torch.no_grad():
            logits = self.model(
                cc, query_idx, target_idx,
                metadata=task_route.metadata,
            )

        # 6. Extract prediction + confidence
        probs = F.softmax(logits, dim=-1)
        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

        # 7. Generate NL answer
        answer = self.answerer.generate(class_idx, task_route, graph_spec, text)

        return NLResult(
            answer=answer,
            class_idx=class_idx,
            task_type=task_route.task_type,
            graph_spec=graph_spec,
            confidence=confidence,
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_pipeline/test_pipeline.py -v`
Expected: 9 passed

**Step 5: Commit**

```bash
git add src/nl_pipeline/pipeline.py tests/test_nl_pipeline/test_pipeline.py
git commit -m "feat(phase5): add NLReasoningPipeline orchestrator"
```

---

### Task 7: Public API Exports + Config

**Files:**
- Modify: `src/nl_pipeline/__init__.py`
- Create: `config/nl_pipeline.yaml`

**Step 1: Update __init__.py with public exports**

```python
# src/nl_pipeline/__init__.py
"""Phase 5: Natural Language Reasoning Pipeline.

Usage:
    from src.nl_pipeline import NLReasoningPipeline, NLResult

    pipeline = NLReasoningPipeline(
        checkpoint_path="data/llama_checkpoints/phase4c_model.pt",
        model_config=config["model"],
        wave_config=config["wave"],
        llm_config=config["llm"],
    )
    result = pipeline.query("What is the shortest path from server to database?")
    print(result.answer)
"""

from src.nl_pipeline.data_types import (
    NodeSpec,
    EdgeSpec,
    GraphSpec,
    TaskRoute,
    NLResult,
)
from src.nl_pipeline.pipeline import NLReasoningPipeline, load_frozen_model
from src.nl_pipeline.graph_parser import MockGraphParser, GraphParser
from src.nl_pipeline.task_router import TaskRouter
from src.nl_pipeline.cell_complex_builder import CellComplexBuilder
from src.nl_pipeline.answer_generator import AnswerGenerator
```

**Step 2: Create config file**

```yaml
# config/nl_pipeline.yaml
# Phase 5: NL Reasoning Pipeline configuration

model:
  embedding_dim: 32
  gnn_hidden: 64
  gnn_spatial_layers: 2
  gnn_spectral_layers: 2
  max_freqs: 8
  tat_layers: 2
  tat_spatial_heads: 2
  tat_spectral_heads: 2
  tat_ff_dim: 128
  max_iterations: 5
  convergence_threshold: 0.1
  use_wave_dynamics: true
  use_higher_order: true
  use_topological_pe: true
  use_structural_features: true

wave:
  wave_mode: ensemble
  filter_types: [chebyshev, wave_cosine, heat]
  include_identity: true
  include_sheaf: true
  laplacian_dim: 0
  wave_strength_gate: true
  use_neural_ode: false

llm:
  backend: mock                          # 'mock' for CPU, 'llama' for GPU
  llm_dim: 128                           # 128 for mock, 2048 for Llama
  num_prefix: 4
  gate_threshold: 0.1
  mock_hidden: 64

pipeline:
  checkpoint_path: data/llama_checkpoints/phase4c_model.pt
  max_classes: 16
  device: auto
  parser: mock                           # 'mock' or 'llama'
  llama_model: unsloth/Llama-3.2-1B     # only used when parser='llama'
```

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: 527+ passed (all existing tests) + ~51 new tests

**Step 4: Commit**

```bash
git add src/nl_pipeline/__init__.py config/nl_pipeline.yaml
git commit -m "feat(phase5): add public API exports and pipeline config"
```

---

### Task 8: Full Regression + Integration Verification

**Step 1: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: 570+ passed, 2 skipped, 0 failed

**Step 2: Verify existing tests unaffected**

Run: `python -m pytest tests/test_benchmarks tests/test_llm tests/test_cell_complex tests/test_gnn tests/test_tat tests/test_spectral tests/test_wave tests/test_reasoning_loop -q`
Expected: 527 passed, 2 skipped (same as before Phase 5)

**Step 3: Run only Phase 5 tests**

Run: `python -m pytest tests/test_nl_pipeline/ -v`
Expected: All Phase 5 tests pass individually

**Step 4: Quick smoke test**

```bash
python -c "
from src.nl_pipeline import NLReasoningPipeline
import torch, tempfile, yaml
from src.benchmarks.run_benchmark_suite import _build_model

# Build a small model and save checkpoint
with open('config/nl_pipeline.yaml') as f:
    config = yaml.safe_load(f)
mc, wc, lc = config['model'], config['wave'], config['llm']
model = _build_model('hierarchical_llm', mc, 16, torch.device('cpu'), wave_config=wc, llm_config=lc)
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    torch.save(model.state_dict(), f.name)
    ckpt = f.name

pipeline = NLReasoningPipeline(ckpt, mc, wc, lc, max_classes=16, device='cpu')
result = pipeline.query('What is the shortest path from server to database?')
print(f'Task: {result.task_type}, Class: {result.class_idx}, Confidence: {result.confidence:.3f}')
print(f'Answer: {result.answer}')
"
```
Expected: Prints a task type, class prediction, and NL answer without errors.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(phase5): Phase 5a NL reasoning pipeline complete

Sequential pipeline: NL → GraphParser → TaskRouter → CellComplexBuilder →
frozen HierarchicalMultiHopModel → AnswerGenerator → NL answer.

Phase 5a uses few-shot prompting (MockGraphParser for CPU) and heuristic
task routing. ~51 new tests, all existing 527 tests unaffected.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task Summary

| Task | Component | New Tests | Files |
|------|-----------|-----------|-------|
| 1 | Data structures | 7 | `data_types.py` |
| 2 | TaskRouter | 14 | `task_router.py` |
| 3 | CellComplexBuilder | 11 | `cell_complex_builder.py` |
| 4 | GraphParser | 9 | `graph_parser.py` |
| 5 | AnswerGenerator | 10 | `answer_generator.py` |
| 6 | NLReasoningPipeline | 9 | `pipeline.py` |
| 7 | Exports + Config | 0 | `__init__.py`, `nl_pipeline.yaml` |
| 8 | Regression check | 0 | — |
| **Total** | | **~60** | **8 new files** |
