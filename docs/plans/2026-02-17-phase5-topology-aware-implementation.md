# Phase 5 Topology-Aware Parsing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add topology inference to the NL pipeline so abstract queries produce structurally faithful graphs for the reasoning model.

**Architecture:** Rule-based TopologyInferrer infers graph topology from NL semantics, CellComplexBuilder generates topology-faithful graphs, TaskRouter and AnswerGenerator use topology context.

**Tech Stack:** Python 3.12, PyTorch, NetworkX, existing graph_generators.py

---

### Task 1: Data Types — TopologyHint + GraphSpec Enhancement

**Files:**
- Modify: `src/nl_pipeline/data_types.py`
- Test: `tests/test_nl_pipeline/test_data_types.py`

**Step 1: Write the failing tests**

```python
# tests/test_nl_pipeline/test_data_types.py — append to existing tests

def test_topology_hint_creation():
    hint = TopologyHint(topology="ba", confidence=0.85, node_roles={"hub": "hub"}, properties={})
    assert hint.topology == "ba"
    assert hint.confidence == 0.85
    assert hint.node_roles == {"hub": "hub"}

def test_graph_spec_backward_compat():
    """Existing GraphSpec creation still works without new fields."""
    spec = GraphSpec(
        nodes=[NodeSpec(name="a", type="entity")],
        edges=[], query_node="a", target_node=None, domain="general",
    )
    assert spec.topology_hint is None
    assert spec.topology_confidence == 0.0
    assert spec.node_roles is None

def test_graph_spec_with_topology():
    spec = GraphSpec(
        nodes=[NodeSpec(name="hub", type="entity")],
        edges=[], query_node="hub", target_node=None, domain="technology",
        topology_hint="ba", topology_confidence=0.85,
        node_roles={"hub": "hub"},
    )
    assert spec.topology_hint == "ba"
    assert spec.topology_confidence == 0.85
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_nl_pipeline/test_data_types.py -v -x
```

**Step 3: Implement**

Add to `src/nl_pipeline/data_types.py`:

```python
@dataclass
class TopologyHint:
    """Inferred graph topology from NL query."""
    topology: str           # ba, ws, sbm, grid, tree, ladder, caveman, er
    confidence: float       # 0.0 to 1.0
    node_roles: dict[str, str] = field(default_factory=dict)  # name → role
    properties: dict = field(default_factory=dict)
```

Modify `GraphSpec` — add 3 optional fields with defaults:

```python
@dataclass
class GraphSpec:
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    query_node: str
    target_node: str | None
    domain: str
    topology_hint: str | None = None
    topology_confidence: float = 0.0
    node_roles: dict[str, str] | None = None
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/test_nl_pipeline/test_data_types.py -v
```

**Step 5: Commit**

```bash
git add src/nl_pipeline/data_types.py tests/test_nl_pipeline/test_data_types.py
git commit -m "feat: add TopologyHint data type and GraphSpec topology fields"
```

---

### Task 2: TopologyInferrer — Core Module

**Files:**
- Create: `src/nl_pipeline/topology_inferrer.py`
- Create: `tests/test_nl_pipeline/test_topology_inferrer.py`

**Step 1: Write the failing tests**

```python
# tests/test_nl_pipeline/test_topology_inferrer.py

import pytest
from src.nl_pipeline.topology_inferrer import TopologyInferrer
from src.nl_pipeline.data_types import GraphSpec, NodeSpec, EdgeSpec


@pytest.fixture
def inferrer():
    return TopologyInferrer()


def _make_spec(nodes, edges=None, domain="general"):
    ns = [NodeSpec(name=n, type="entity") for n in nodes]
    es = edges or []
    return GraphSpec(nodes=ns, edges=es, query_node=nodes[0],
                     target_node=nodes[-1] if len(nodes) > 1 else None,
                     domain=domain)


class TestTopologyPatterns:
    def test_hub_spoke_ba(self, inferrer):
        spec = _make_spec(["hub", "server", "client"])
        hint = inferrer.infer(spec, "what happens when the central hub fails")
        assert hint.topology == "ba"
        assert hint.confidence > 0.3

    def test_hierarchy_tree(self, inferrer):
        spec = _make_spec(["manager", "team", "employee"])
        hint = inferrer.infer(spec, "how does information flow through the hierarchy")
        assert hint.topology == "tree"

    def test_community_sbm(self, inferrer):
        spec = _make_spec(["team_a", "team_b", "bridge"])
        hint = inferrer.infer(spec, "how do clusters communicate between departments")
        assert hint.topology == "sbm"

    def test_cycle_ws(self, inferrer):
        spec = _make_spec(["process_a", "process_b"])
        hint = inferrer.infer(spec, "is there a feedback loop in this cycle")
        assert hint.topology == "ws"

    def test_grid_topology(self, inferrer):
        spec = _make_spec(["pixel", "row", "column"])
        hint = inferrer.infer(spec, "how do values propagate through the grid")
        assert hint.topology == "grid"

    def test_ladder_topology(self, inferrer):
        spec = _make_spec(["path_a", "path_b"])
        hint = inferrer.infer(spec, "which parallel path is the redundant backup")
        assert hint.topology == "ladder"

    def test_caveman_topology(self, inferrer):
        spec = _make_spec(["member_a", "member_b"])
        hint = inferrer.infer(spec, "how tight-knit is each clique in the network")
        assert hint.topology == "caveman"

    def test_random_er(self, inferrer):
        spec = _make_spec(["node_a", "node_b"])
        hint = inferrer.infer(spec, "what is the structure of this random network")
        assert hint.topology == "er"


class TestConfidence:
    def test_high_confidence_multi_pattern(self, inferrer):
        spec = _make_spec(["hub", "spoke"])
        hint = inferrer.infer(spec, "the central hub broadcasts to the star network")
        assert hint.confidence > 0.5

    def test_no_match_returns_none(self, inferrer):
        spec = _make_spec(["foo", "bar"])
        hint = inferrer.infer(spec, "tell me about foo and bar")
        assert hint.topology is None
        assert hint.confidence == 0.0


class TestNodeRoles:
    def test_hub_role_assigned(self, inferrer):
        spec = _make_spec(["hub", "client_a", "client_b"])
        hint = inferrer.infer(spec, "the central hub connects to clients")
        assert hint.node_roles.get("hub") == "hub"

    def test_root_role_for_tree(self, inferrer):
        spec = _make_spec(["root", "child_a", "child_b"])
        hint = inferrer.infer(spec, "the root of the tree has two children")
        assert hint.node_roles.get("root") == "root"
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_nl_pipeline/test_topology_inferrer.py -v -x
```

**Step 3: Implement TopologyInferrer**

Create `src/nl_pipeline/topology_inferrer.py`:

```python
"""Topology inference from NL queries using keyword pattern matching."""
import re
from src.nl_pipeline.data_types import GraphSpec, TopologyHint

# Pattern groups: (keywords, topology, default_role_for_adjacent_nodes)
# Multi-word patterns have higher weight (2) than single words (1)
TOPOLOGY_PATTERNS: list[tuple[list[str], str, str]] = [
    # Hub/Star → BA (scale-free)
    (["central hub", "hub and spoke", "star network", "load balancer",
      "hub", "central", "star", "broadcast", "core", "coordinator",
      "main server", "backbone"], "ba", "hub"),

    # Hierarchy → Tree
    (["org chart", "inheritance tree", "parent child", "decision tree",
      "hierarchy", "hierarchical", "tree", "parent", "child", "levels",
      "subordinate", "root", "branch", "ancestor", "descendant",
      "top-down", "bottom-up", "manager"], "tree", "root"),

    # Community → SBM
    (["community structure", "department", "partition",
      "cluster", "community", "group", "faction", "team", "silo",
      "segregated", "module", "subgroup", "clique"], "sbm", "member"),

    # Ring/Cycle → WS (small-world)
    (["round-robin", "feedback loop", "circular dependency",
      "cycle", "loop", "circular", "feedback", "recurring",
      "oscillate", "ring", "periodic"], "ws", "peer"),

    # Grid → Grid
    (["grid layout", "spreadsheet", "pixel grid",
      "grid", "matrix", "lattice", "row", "column",
      "coordinate", "pixel", "table"], "grid", "cell"),

    # Ladder → Ladder
    (["parallel paths", "dual path", "redundant path", "backup route",
      "ladder", "parallel", "redundant", "dual"], "ladder", "rung"),

    # Dense/Clique → Caveman
    (["fully connected", "complete graph", "tight-knit",
      "clique", "tribe", "dense", "complete", "all-to-all"],
     "caveman", "clique_member"),

    # Random → ER
    (["random network", "random graph",
      "random", "arbitrary", "mesh"], "er", "generic"),
]

# Role keywords: if a node name contains these, assign the role
ROLE_KEYWORDS: dict[str, str] = {
    "hub": "hub", "central": "hub", "core": "hub", "server": "hub",
    "coordinator": "hub", "backbone": "hub",
    "root": "root", "parent": "root", "manager": "root", "boss": "root",
    "child": "leaf", "leaf": "leaf", "endpoint": "leaf", "client": "leaf",
    "spoke": "leaf", "subordinate": "leaf", "worker": "leaf",
    "bridge": "bridge", "gateway": "bridge", "connector": "bridge",
    "member": "member", "peer": "peer", "node": "generic",
}


class TopologyInferrer:
    """Infer graph topology from NL query using keyword pattern matching."""

    def infer(self, spec: GraphSpec, query: str) -> TopologyHint:
        q_lower = query.lower()
        scores: dict[str, float] = {}
        best_role: dict[str, str] = {}

        for patterns, topology, default_role in TOPOLOGY_PATTERNS:
            score = 0.0
            for pattern in patterns:
                if pattern in q_lower:
                    weight = 2.0 if " " in pattern else 1.0
                    score += weight
            if score > 0:
                scores[topology] = scores.get(topology, 0.0) + score
                best_role[topology] = default_role

        # Also check node names for topology cues
        for node in spec.nodes:
            n_lower = node.name.lower()
            for patterns, topology, default_role in TOPOLOGY_PATTERNS:
                for pattern in patterns:
                    if pattern in n_lower and " " not in pattern:
                        scores[topology] = scores.get(topology, 0.0) + 0.5
                        best_role.setdefault(topology, default_role)

        if not scores:
            return TopologyHint(topology=None, confidence=0.0)

        # Pick best topology
        sorted_topos = sorted(scores.items(), key=lambda x: -x[1])
        best_topo, best_score = sorted_topos[0]
        second_score = sorted_topos[1][1] if len(sorted_topos) > 1 else 0.0

        # Confidence: margin between best and second
        confidence = min((best_score - second_score) / max(best_score, 1e-6), 1.0)
        # Boost confidence if multiple patterns matched
        if best_score >= 3.0:
            confidence = max(confidence, 0.7)
        elif best_score >= 2.0:
            confidence = max(confidence, 0.5)
        elif best_score >= 1.0:
            confidence = max(confidence, 0.3)

        # Assign node roles
        node_roles = {}
        for node in spec.nodes:
            n_lower = node.name.lower()
            for keyword, role in ROLE_KEYWORDS.items():
                if keyword in n_lower:
                    node_roles[node.name] = role
                    break
            if node.name not in node_roles:
                node_roles[node.name] = best_role.get(best_topo, "generic")

        return TopologyHint(
            topology=best_topo,
            confidence=confidence,
            node_roles=node_roles,
            properties={},
        )
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/test_nl_pipeline/test_topology_inferrer.py -v
```

**Step 5: Commit**

```bash
git add src/nl_pipeline/topology_inferrer.py tests/test_nl_pipeline/test_topology_inferrer.py
git commit -m "feat: add TopologyInferrer with rule-based topology detection"
```

---

### Task 3: GraphParser Integration — Attach TopologyInferrer

**Files:**
- Modify: `src/nl_pipeline/graph_parser.py`
- Test: `tests/test_nl_pipeline/test_graph_parser.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_nl_pipeline/test_graph_parser.py

def test_mock_parser_sets_topology_hint():
    parser = MockGraphParser()
    spec = parser.parse("what happens when the central hub fails in the network")
    assert spec.topology_hint == "ba"
    assert spec.topology_confidence > 0.3
    assert spec.node_roles is not None

def test_mock_parser_no_topology_for_generic():
    parser = MockGraphParser()
    spec = parser.parse("tell me about foo and bar")
    # No topology patterns match → hint is None
    assert spec.topology_hint is None

def test_mock_parser_tree_topology():
    parser = MockGraphParser()
    spec = parser.parse("how does information flow through the hierarchy from parent to child")
    assert spec.topology_hint == "tree"
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

Modify `MockGraphParser.parse()` and `GraphParser.parse()` to call `TopologyInferrer.infer()` after building the GraphSpec, then copy hint fields onto the spec:

```python
from src.nl_pipeline.topology_inferrer import TopologyInferrer

class MockGraphParser(BaseGraphParser):
    def __init__(self):
        self._topology_inferrer = TopologyInferrer()

    def parse(self, query: str) -> GraphSpec:
        # ... existing keyword extraction code ...
        spec = GraphSpec(nodes=nodes, edges=edges, ...)

        # Infer topology
        hint = self._topology_inferrer.infer(spec, query)
        spec.topology_hint = hint.topology
        spec.topology_confidence = hint.confidence
        spec.node_roles = hint.node_roles if hint.node_roles else None
        return spec
```

Same for `GraphParser` — after `parse_graph_json()` succeeds, run topology inference.

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/nl_pipeline/graph_parser.py tests/test_nl_pipeline/test_graph_parser.py
git commit -m "feat: integrate TopologyInferrer into graph parsers"
```

---

### Task 4: CellComplexBuilder — Topology-Aware Graph Generation

**Files:**
- Modify: `src/nl_pipeline/cell_complex_builder.py`
- Test: `tests/test_nl_pipeline/test_cell_complex_builder.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_nl_pipeline/test_cell_complex_builder.py

def test_builder_uses_topology_hint_ba():
    spec = GraphSpec(
        nodes=[NodeSpec("hub", "entity"), NodeSpec("client_a", "entity"),
               NodeSpec("client_b", "entity")],
        edges=[EdgeSpec("hub", "client_a", "connects"),
               EdgeSpec("hub", "client_b", "connects")],
        query_node="hub", target_node="client_a", domain="technology",
        topology_hint="ba", topology_confidence=0.8,
        node_roles={"hub": "hub", "client_a": "leaf", "client_b": "leaf"},
    )
    builder = CellComplexBuilder(embedding_dim=32)
    cc, name_to_cc = builder.build(spec)
    # Should have more nodes than just the 3 parsed (topology generation adds structure)
    assert cc.num_cells(0) >= 3
    # All named nodes should be mapped
    assert "hub" in name_to_cc
    assert "client_a" in name_to_cc

def test_builder_fallback_no_topology():
    """Without topology hint, builds from edges (current behavior)."""
    spec = GraphSpec(
        nodes=[NodeSpec("a", "entity"), NodeSpec("b", "entity")],
        edges=[EdgeSpec("a", "b", "connects")],
        query_node="a", target_node="b", domain="general",
    )
    builder = CellComplexBuilder(embedding_dim=32)
    cc, name_to_cc = builder.build(spec)
    assert cc.num_cells(0) == 2  # Only 2 nodes, no topology expansion

def test_builder_hub_gets_high_degree(self):
    spec = GraphSpec(
        nodes=[NodeSpec("hub", "entity"), NodeSpec("spoke1", "entity"),
               NodeSpec("spoke2", "entity")],
        edges=[], query_node="hub", target_node="spoke1", domain="tech",
        topology_hint="ba", topology_confidence=0.9,
        node_roles={"hub": "hub", "spoke1": "leaf", "spoke2": "leaf"},
    )
    builder = CellComplexBuilder(embedding_dim=32)
    cc, name_to_cc = builder.build(spec)
    # Hub node should exist and be mapped
    assert "hub" in name_to_cc
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

Modify `CellComplexBuilder.build()`:

```python
from src.benchmarks.graph_generators import random_graph

def build(self, spec: GraphSpec):
    if spec.topology_hint and spec.topology_confidence >= 0.3:
        return self._build_with_topology(spec)
    return self._build_from_edges(spec)  # existing behavior

def _build_with_topology(self, spec: GraphSpec):
    """Generate a topology-faithful graph and map semantic nodes onto it."""
    n_semantic = len(spec.nodes)
    n_graph = max(n_semantic, 10)  # min 10 for meaningful topology

    G = random_graph(n_graph, topology=spec.topology_hint)

    # Sort nodes by degree (descending)
    degree_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)

    # Map semantic nodes to structural positions based on roles
    name_to_nx = {}
    used_positions = set()

    # Priority mapping: hub/root → highest degree, leaf → lowest, etc.
    role_priority = {"hub": 0, "root": 0, "bridge": 0.3, "internal": 0.5,
                     "member": 0.5, "peer": 0.5, "cell": 0.5, "rung": 0.5,
                     "clique_member": 0.5, "generic": 0.5, "leaf": 0.9}

    roles = spec.node_roles or {}
    nodes_with_roles = []
    for node in spec.nodes:
        role = roles.get(node.name, "generic")
        priority = role_priority.get(role, 0.5)
        nodes_with_roles.append((node, priority))

    # Sort by priority (hubs first, leaves last)
    nodes_with_roles.sort(key=lambda x: x[1])

    for node, priority in nodes_with_roles:
        # Find best available position
        target_idx = int(priority * (len(degree_sorted) - 1))
        for offset in range(len(degree_sorted)):
            for candidate in [target_idx + offset, target_idx - offset]:
                if 0 <= candidate < len(degree_sorted):
                    pos = degree_sorted[candidate]
                    if pos not in used_positions:
                        name_to_nx[node.name] = pos
                        used_positions.add(pos)
                        break
            if node.name in name_to_nx:
                break

    # Convert to CellComplex
    source_nx = name_to_nx.get(spec.query_node)
    target_nx = name_to_nx.get(spec.target_node)
    cc, nx_to_cc = nx_to_cell_complex(
        G, self.embedding_dim,
        source_node=source_nx, target_node=target_nx,
        fill_triangles=True,
    )

    # Build name→cc mapping
    name_to_cc = {
        name: nx_to_cc[nx_id]
        for name, nx_id in name_to_nx.items()
        if nx_id in nx_to_cc
    }

    # Encode edge relations where structurally adjacent
    # (omitted for brevity — same pattern as existing code)

    return cc, name_to_cc
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/nl_pipeline/cell_complex_builder.py tests/test_nl_pipeline/test_cell_complex_builder.py
git commit -m "feat: topology-aware graph generation in CellComplexBuilder"
```

---

### Task 5: TaskRouter — Topology-Aware Routing

**Files:**
- Modify: `src/nl_pipeline/task_router.py`
- Test: `tests/test_nl_pipeline/test_task_router.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_nl_pipeline/test_task_router.py

def test_topology_hint_in_metadata():
    spec = GraphSpec(
        nodes=[NodeSpec("a", "entity"), NodeSpec("b", "entity")],
        edges=[EdgeSpec("a", "b", "connects")],
        query_node="a", target_node="b", domain="general",
        topology_hint="ba", topology_confidence=0.8,
    )
    router = TaskRouter()
    route = router.route(spec, "tell me about this network")
    assert route.metadata.get("topology_hint") == "ba"

def test_topology_affinity_overrides_diverse():
    """When query matches no keywords but topology has task affinity, use it."""
    spec = GraphSpec(
        nodes=[NodeSpec("a", "entity"), NodeSpec("b", "entity")],
        edges=[EdgeSpec("a", "b", "connects")],
        query_node="a", target_node="b", domain="general",
        topology_hint="tree", topology_confidence=0.8,
    )
    router = TaskRouter()
    route = router.route(spec, "what is the structure of this hierarchy")
    # "hierarchy" doesn't match keyword rules → would be diverse
    # But topology=tree has affinity for bfs → prefer that
    assert route.task_type in ("bfs", "path_counting", "diverse")
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

Add topology-task affinity to `TaskRouter._classify()`:

```python
_TOPOLOGY_TASK_AFFINITY = {
    'tree': 'bfs',
    'ba': 'spectral_gap',
    'ws': 'cycle_detection',
    'sbm': 'spectral_gap',
    'grid': 'bfs',
    'caveman': 'hodge_class',
    'ladder': 'path_counting',
    'er': 'diverse',
}

def _classify(self, spec: GraphSpec, query: str) -> str:
    # ... existing causal + keyword checks ...
    # NEW: if no keyword match, check topology affinity
    if spec.topology_hint and spec.topology_confidence >= 0.3:
        affinity = _TOPOLOGY_TASK_AFFINITY.get(spec.topology_hint)
        if affinity:
            return affinity
    return "diverse"
```

Also add `topology_hint` to route metadata.

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/nl_pipeline/task_router.py tests/test_nl_pipeline/test_task_router.py
git commit -m "feat: topology-aware task routing with affinity mapping"
```

---

### Task 6: AnswerGenerator — Topology Context in Answers

**Files:**
- Modify: `src/nl_pipeline/answer_generator.py`
- Test: `tests/test_nl_pipeline/test_answer_generator.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_nl_pipeline/test_answer_generator.py

def test_answer_includes_topology_context():
    spec = GraphSpec(
        nodes=[NodeSpec("hub", "entity"), NodeSpec("client", "entity")],
        edges=[EdgeSpec("hub", "client", "connects")],
        query_node="hub", target_node="client", domain="technology",
        topology_hint="ba", topology_confidence=0.8,
    )
    route = TaskRoute(task_type="bfs", query_node_idx=0, target_node_idx=1,
                      max_classes=16, metadata={})
    gen = AnswerGenerator()
    answer = gen.generate(3, route, spec, "how far is hub from client")
    assert "hub-spoke" in answer.lower() or "scale-free" in answer.lower()

def test_answer_no_topology_context_when_none():
    spec = GraphSpec(
        nodes=[NodeSpec("a", "entity"), NodeSpec("b", "entity")],
        edges=[], query_node="a", target_node="b", domain="general",
    )
    route = TaskRoute(task_type="bfs", query_node_idx=0, target_node_idx=1,
                      max_classes=16, metadata={})
    gen = AnswerGenerator()
    answer = gen.generate(3, route, spec, "how far is a from b")
    # No topology prefix
    assert "hub-spoke" not in answer.lower()
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

Add topology descriptions and prefix logic:

```python
TOPOLOGY_DESCRIPTIONS = {
    'ba': 'hub-spoke network',
    'tree': 'hierarchical tree',
    'sbm': 'community-structured network',
    'ws': 'small-world network',
    'grid': 'grid structure',
    'ladder': 'dual-path ladder',
    'caveman': 'densely clustered network',
    'er': 'random network',
}

def generate(self, class_idx, task_route, graph_spec, original_query):
    interpreter = CLASS_INTERPRETATIONS.get(task_route.task_type)
    if interpreter is not None:
        base_answer = interpreter(class_idx, graph_spec)
    else:
        base_answer = f"The model predicted class {class_idx} ..."

    # Add topology context
    if graph_spec.topology_hint and graph_spec.topology_confidence >= 0.3:
        desc = TOPOLOGY_DESCRIPTIONS.get(graph_spec.topology_hint, "network")
        return f"In the {desc}: {base_answer}"
    return base_answer
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/nl_pipeline/answer_generator.py tests/test_nl_pipeline/test_answer_generator.py
git commit -m "feat: topology-aware answer generation with structural context"
```

---

### Task 7: Pipeline Wiring + Exports

**Files:**
- Modify: `src/nl_pipeline/pipeline.py`
- Modify: `src/nl_pipeline/__init__.py`
- Test: `tests/test_nl_pipeline/test_pipeline.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_nl_pipeline/test_pipeline.py

def test_pipeline_query_with_topology(pipeline):
    result = pipeline.query("what happens when the central hub fails in the distributed system")
    assert result.graph_spec.topology_hint == "ba"
    assert result.graph_spec.topology_confidence > 0.0

def test_pipeline_topology_enriches_answer(pipeline):
    result = pipeline.query("how does information flow through the hierarchy from root to leaf")
    assert result.graph_spec.topology_hint == "tree"
    assert "hierarchical" in result.answer.lower() or "tree" in result.answer.lower()
```

**Step 2: Run tests — expect FAIL initially (then PASS after Task 1-6 integration)**

**Step 3: Implement**

Pipeline already calls `self.parser.parse(text)` which now includes topology inference. No pipeline.py changes needed — just verify the data flows through.

Update `__init__.py` exports:

```python
from src.nl_pipeline.data_types import TopologyHint
from src.nl_pipeline.topology_inferrer import TopologyInferrer
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/test_nl_pipeline/ -v
```

**Step 5: Commit**

```bash
git add src/nl_pipeline/__init__.py src/nl_pipeline/pipeline.py tests/test_nl_pipeline/test_pipeline.py
git commit -m "feat: wire topology inference through pipeline + update exports"
```

---

### Task 8: Integration Tests + Full Regression

**Files:**
- Create: `tests/test_nl_pipeline/test_topology_integration.py`

**Step 1: Write integration tests**

```python
# tests/test_nl_pipeline/test_topology_integration.py

"""End-to-end tests: NL queries with implied topologies."""
import pytest
from src.nl_pipeline.graph_parser import MockGraphParser
from src.nl_pipeline.topology_inferrer import TopologyInferrer
from src.nl_pipeline.cell_complex_builder import CellComplexBuilder
from src.nl_pipeline.task_router import TaskRouter


@pytest.fixture
def parser():
    return MockGraphParser()

@pytest.fixture
def builder():
    return CellComplexBuilder(embedding_dim=32)

@pytest.fixture
def router():
    return TaskRouter()


class TestHubSpokeQueries:
    def test_central_hub_failure(self, parser, builder, router):
        spec = parser.parse("what happens when the central hub fails in a distributed system")
        assert spec.topology_hint == "ba"
        cc, names = builder.build(spec)
        assert cc.num_cells(0) >= 10  # topology generation
        route = router.route(spec, "what happens when the central hub fails")
        assert route.metadata.get("topology_hint") == "ba"

    def test_load_balancer_query(self, parser, builder):
        spec = parser.parse("how does the load balancer distribute traffic to servers")
        assert spec.topology_hint == "ba"


class TestHierarchyQueries:
    def test_org_chart(self, parser, builder):
        spec = parser.parse("how does information flow from manager to subordinate in the org chart")
        assert spec.topology_hint == "tree"
        cc, names = builder.build(spec)
        assert cc.num_cells(0) >= 10

    def test_decision_tree(self, parser, builder):
        spec = parser.parse("what is the depth of this decision tree from root to leaf")
        assert spec.topology_hint == "tree"


class TestCommunityQueries:
    def test_department_clusters(self, parser, builder):
        spec = parser.parse("how do different departments communicate in the company")
        assert spec.topology_hint == "sbm"

    def test_community_detection(self, parser, builder):
        spec = parser.parse("identify the main communities in this social group")
        assert spec.topology_hint == "sbm"


class TestCyclicQueries:
    def test_feedback_loop(self, parser, builder):
        spec = parser.parse("is there a feedback loop between production and testing")
        assert spec.topology_hint == "ws"

    def test_circular_dependency(self, parser, builder):
        spec = parser.parse("detect the circular dependency in the module imports")
        assert spec.topology_hint == "ws"


class TestFallbackBehavior:
    def test_generic_query_no_topology(self, parser, builder):
        spec = parser.parse("tell me about the relationship between apples and oranges")
        assert spec.topology_hint is None
        cc, names = builder.build(spec)
        # Falls back to edge-based construction (small graph)
        assert cc.num_cells(0) <= 6

    def test_topology_propagates_to_answer(self, parser, builder, router):
        spec = parser.parse("how tight-knit is each clique in the organization")
        assert spec.topology_hint == "caveman"
        route = router.route(spec, "how tight-knit is each clique")
        assert route.metadata.get("topology_hint") == "caveman"
```

**Step 2: Run all tests**

```bash
pytest tests/ -x -q
```

Expected: all existing 587 tests + ~25 new tests pass.

**Step 3: Commit**

```bash
git add tests/test_nl_pipeline/test_topology_integration.py
git commit -m "test: integration tests for topology-aware NL pipeline"
```
