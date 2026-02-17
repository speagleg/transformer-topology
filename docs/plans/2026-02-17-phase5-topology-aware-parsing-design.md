# Phase 5 Enhancement: Topology-Aware NL Parsing

## Problem

The current Phase 5 NL pipeline has a critical gap: **the parser creates linear keyword chains regardless of what the query describes**. A query like "what happens when the central hub fails in a distributed system" produces `hub → fails → central → distributed → system` — a chain graph. It should produce a hub-spoke (BA) graph with the hub node at the center.

This matters because the reasoning model was trained on 8 distinct topologies (BA, WS, SBM, grid, tree, ladder, caveman, ER). If we feed it the wrong topology, the spectral and topological reasoning machinery operates on a meaningless structure. The metacognition model requires the graph structure to faithfully represent the relationships implied by the natural language query.

## Architecture

**Approach: Rule-based TopologyInferrer** with pluggable interface for future LLM-based inference.

The pipeline becomes:

```
NL Query
    ↓
GraphParser (extract concepts + relations)
    ↓
TopologyInferrer (NEW: infer graph topology from NL semantics)
    ↓
TaskRouter (route task, now topology-aware)
    ↓
CellComplexBuilder (UPGRADED: generate topology-faithful graph)
    ↓
Frozen Model → AnswerGenerator (UPGRADED: topology-aware answers)
```

## Components

### 1. TopologyHint (data type)

```python
@dataclass
class TopologyHint:
    topology: str           # ba, ws, sbm, grid, tree, ladder, caveman, er
    confidence: float       # 0.0 to 1.0
    node_roles: dict[str, str]  # node_name → structural role
    properties: dict        # topology-specific params (e.g., num_communities)
```

**Structural roles:** `hub`, `leaf`, `root`, `internal`, `bridge`, `member`, `peer`, `cell`, `generic`

### 2. TopologyInferrer

New module `src/nl_pipeline/topology_inferrer.py`. Rule-based with ~30 patterns.

**Keyword → Topology mapping:**

| Pattern Group | Keywords | Topology | Default Roles |
|---|---|---|---|
| Hub/Star | hub, central, star, broadcast, core, coordinator, main server, load balancer | `ba` | hub, spoke |
| Hierarchy | hierarchy, tree, parent, child, levels, org chart, manager, subordinate, root, branch, inheritance | `tree` | root, internal, leaf |
| Community | cluster, community, group, department, faction, team, partition, segregated, silo | `sbm` | member, bridge |
| Ring/Cycle | ring, cycle, loop, circular, round-robin, feedback, recurring, oscillate | `ws` | peer |
| Grid/Matrix | grid, matrix, row, column, lattice, table, spreadsheet, pixel, coordinate | `grid` | cell |
| Ladder/Dual | ladder, parallel paths, dual, redundant path, backup route | `ladder` | rung |
| Clique/Dense | clique, tribe, dense, tight-knit, fully connected, complete graph | `caveman` | clique_member |
| Random/General | random, general, complex, arbitrary, network, mesh | `er` | generic |

**Inference algorithm:**
1. Tokenize query, match against pattern groups
2. Count matches per topology, weighted by specificity (multi-word patterns > single words)
3. Check edge relations for structural cues (cycles → ws, hierarchy → tree)
4. Assign node roles based on adjacency to pattern keywords
5. Return TopologyHint with confidence = (best_score - second_best) / best_score

**Fallback:** If no patterns match (confidence=0), return `topology=None` → CellComplexBuilder falls back to building from parsed edges (current behavior).

### 3. GraphSpec Enhancement

Add optional fields (backward-compatible):

```python
@dataclass
class GraphSpec:
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    query_node: str
    target_node: str | None
    domain: str
    # NEW
    topology_hint: str | None = None          # inferred topology type
    topology_confidence: float = 0.0          # inference confidence
    node_roles: dict[str, str] | None = None  # name → structural role
```

### 4. CellComplexBuilder Upgrade

When `topology_hint` is present with sufficient confidence (>0.3):

1. **Generate topology graph** via `random_graph(n, topology=hint)` where `n = max(len(spec.nodes), min_size_for_topology)`
2. **Sort generated nodes by structural importance** (degree centrality)
3. **Map semantic nodes to structural positions:**
   - `hub` role → highest-degree node
   - `root` role → node 0 or highest-degree
   - `leaf` role → lowest-degree nodes
   - `bridge` role → nodes connecting communities (for SBM)
   - `member`/`peer` role → distributed evenly
   - `generic` → remaining positions
4. **Copy semantic labels** to mapped positions
5. **Encode edge relations** from parsed edges where structurally adjacent

When `topology_hint` is None or confidence < 0.3: current behavior (build from parsed edges).

### 5. TaskRouter Upgrade

Use topology_hint to refine routing:

```python
_TOPOLOGY_TASK_AFFINITY = {
    'tree': ['bfs', 'path_counting'],
    'ba': ['bfs', 'spectral_gap', 'graph_completion'],
    'ws': ['cycle_detection', 'spectral_gap'],
    'sbm': ['spectral_gap', 'hodge_class'],
    'grid': ['bfs', 'path_counting'],
    'caveman': ['cycle_detection', 'hodge_class'],
}
```

When keyword routing produces "diverse" but topology_hint suggests a specific task affinity, prefer the affinity task. This helps queries like "how connected are the clusters" (topology=sbm) route to `spectral_gap` instead of `diverse`.

### 6. AnswerGenerator Upgrade

Include topology context when available:

```python
TOPOLOGY_DESCRIPTIONS = {
    'ba': 'hub-spoke network',
    'tree': 'hierarchical tree structure',
    'sbm': 'community-structured graph',
    'ws': 'small-world network',
    'grid': 'grid/lattice structure',
    'ladder': 'dual-path ladder structure',
    'caveman': 'densely connected cluster structure',
    'er': 'random network',
}
```

Prefix answers with: "In the {topology_description}, ..." when topology_hint is present.

## Example: End-to-End

**Query:** "What happens when the central hub fails in a distributed system?"

1. **MockGraphParser** extracts: nodes=[hub, fails, central, distributed, system], edges=chain, domain=technology
2. **TopologyInferrer** detects: "central hub" → ba (conf=0.85), node_roles={hub: "hub", system: "spoke"}
3. **TaskRouter** routes: causal "fails" → labeled_reasoning, topology=ba confirms structural reasoning
4. **CellComplexBuilder**: generates BA graph (20 nodes), maps "hub" to highest-degree node, "system" to spoke
5. **Model** reasons over proper hub-spoke structure with Hodge/spectral machinery
6. **AnswerGenerator**: "In the hub-spoke network, there is a causal relationship between hub and system."

## Testing Strategy

- TopologyInferrer: 12+ tests covering all 8 topologies, multi-pattern queries, confidence scoring, no-match fallback
- GraphSpec: backward compat (existing tests still pass with new optional fields)
- CellComplexBuilder: topology-hint path produces correct graph type (verify degree distribution)
- TaskRouter: topology-aware routing overrides diverse fallback correctly
- AnswerGenerator: topology descriptions appear in answers when hint present
- Integration: end-to-end query with topology inference produces different (better) graph than without

## Files

| File | Action |
|---|---|
| `src/nl_pipeline/data_types.py` | Modify: add GraphSpec fields |
| `src/nl_pipeline/topology_inferrer.py` | Create: TopologyInferrer + TopologyHint |
| `src/nl_pipeline/graph_parser.py` | Modify: call TopologyInferrer after parsing |
| `src/nl_pipeline/cell_complex_builder.py` | Modify: topology-aware graph generation |
| `src/nl_pipeline/task_router.py` | Modify: topology-aware routing |
| `src/nl_pipeline/answer_generator.py` | Modify: topology-aware answer prefix |
| `src/nl_pipeline/pipeline.py` | Modify: wire TopologyInferrer |
| `src/nl_pipeline/__init__.py` | Modify: export new types |
| `tests/test_nl_pipeline/test_topology_inferrer.py` | Create: 12+ tests |
| `tests/test_nl_pipeline/test_topology_integration.py` | Create: integration tests |
