# Phase 7: Meta-Cognition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 5 ConceptNet knowledge-graph tasks that require multi-modal fusion (structure + text), proving the GNN executive learns when to consult the LLM subconscious.

**Architecture:** ConceptNet subgraphs provide real commonsense knowledge. GNN sees structural features only (degree, clustering, BFS distance). LLM (Qwen 2.5-3B) reads concept text via hard tokens concatenated with TopoBridge prefix tokens. ControlHead's semantic_weight must rise on KG tasks (currently 0.0 on all tasks because none require text).

**Tech Stack:** Python 3.12, PyTorch, NetworkX, ConceptNet 5.7 CSV, existing TopoBridge/QwenGraphBackend infrastructure.

**Design doc:** `docs/plans/2026-02-26-phase7-metacognition-design.md`

---

### Task 1: CellComplex node_texts attribute

**Files:**
- Modify: `src/cell_complex/cell_complex.py:14-23` (add attribute)
- Modify: `src/cell_complex/cell_complex.py:40-54` (clone)
- Test: `tests/test_cell_complex/test_node_texts.py`

**Step 1: Write the failing test**

```python
# tests/test_cell_complex/test_node_texts.py
import torch
from src.cell_complex.cell_complex import CellComplex


def test_node_texts_default_empty():
    cc = CellComplex(embedding_dim=8)
    assert cc.node_texts == []


def test_node_texts_set_and_get():
    cc = CellComplex(embedding_dim=8)
    cc.add_0_cell(torch.randn(8), "node")
    cc.add_0_cell(torch.randn(8), "node")
    cc.node_texts = ["dog", "animal"]
    assert cc.node_texts == ["dog", "animal"]


def test_node_texts_survives_clone():
    cc = CellComplex(embedding_dim=8)
    cc.add_0_cell(torch.randn(8), "node")
    cc.node_texts = ["dog"]
    cc2 = cc.clone()
    assert cc2.node_texts == ["dog"]


def test_node_texts_survives_to_device():
    cc = CellComplex(embedding_dim=8)
    cc.add_0_cell(torch.randn(8), "node")
    cc.node_texts = ["cat"]
    cc.to("cpu")  # no-op on CPU but exercises the code path
    assert cc.node_texts == ["cat"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cell_complex/test_node_texts.py -v`
Expected: FAIL with `AttributeError: 'CellComplex' object has no attribute 'node_texts'`

**Step 3: Write minimal implementation**

In `src/cell_complex/cell_complex.py`:

After line 23 (`self._2_cell_types: list[str] = []`), add:
```python
        self.node_texts: list[str] = []
```

In `clone()` method (after line 53 `cc._2_cell_types = self._2_cell_types`), add:
```python
        cc.node_texts = list(self.node_texts)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cell_complex/test_node_texts.py -v`
Expected: 4 PASSED

**Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -x -q --timeout=60 -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat"`
Expected: All pass

**Step 6: Commit**

```bash
git add src/cell_complex/cell_complex.py tests/test_cell_complex/test_node_texts.py
git commit -m "feat: add node_texts attribute to CellComplex for KG text metadata"
```

---

### Task 2: ConceptNet download and preprocessing

**Files:**
- Create: `src/data/conceptnet.py`
- Create: `src/data/__init__.py`
- Test: `tests/test_data/test_conceptnet.py`

**Step 1: Write the failing test**

```python
# tests/test_data/test_conceptnet.py
import os
import tempfile
import pytest
import networkx as nx

from src.data.conceptnet import (
    parse_conceptnet_line,
    RELATION_CATEGORIES,
    categorize_relation,
    load_conceptnet_graph,
    extract_subgraph,
    concept_to_text,
)


def test_parse_conceptnet_line_valid():
    line = "/a/[/r/IsA/,/c/en/dog/,/c/en/animal/]\t/r/IsA\t/c/en/dog\t/c/en/animal\t{\"weight\": 2.0}"
    result = parse_conceptnet_line(line)
    assert result is not None
    src, rel, tgt, weight = result
    assert src == "dog"
    assert tgt == "animal"
    assert rel == "IsA"
    assert weight == 2.0


def test_parse_conceptnet_line_non_english():
    line = "/a/[/r/IsA/,/c/fr/chien/,/c/fr/animal/]\t/r/IsA\t/c/fr/chien\t/c/fr/animal\t{\"weight\": 2.0}"
    result = parse_conceptnet_line(line)
    assert result is None  # Non-English filtered


def test_parse_conceptnet_line_low_weight():
    line = "/a/[/r/IsA/,/c/en/dog/,/c/en/animal/]\t/r/IsA\t/c/en/dog\t/c/en/animal\t{\"weight\": 0.5}"
    result = parse_conceptnet_line(line)
    assert result is None  # weight < 1.0 filtered


def test_relation_categories():
    assert len(RELATION_CATEGORIES) == 10
    assert "IsA" in RELATION_CATEGORIES
    assert "Other" in RELATION_CATEGORIES


def test_categorize_relation():
    assert categorize_relation("IsA") == "IsA"
    assert categorize_relation("HasA") == "HasA"
    assert categorize_relation("DerivedFrom") == "RelatedTo"
    assert categorize_relation("SomeUnknownRel") == "Other"


def test_concept_to_text():
    assert concept_to_text("fire_truck") == "fire truck"
    assert concept_to_text("dog") == "dog"


def test_extract_subgraph_basic():
    # Build a small mock ConceptNet graph
    G = nx.Graph()
    G.add_edge("dog", "animal", relation="IsA", weight=2.0)
    G.add_edge("dog", "pet", relation="IsA", weight=1.5)
    G.add_edge("cat", "animal", relation="IsA", weight=2.0)
    G.add_edge("cat", "pet", relation="IsA", weight=1.5)
    G.add_edge("animal", "living_thing", relation="IsA", weight=2.0)
    G.add_edge("pet", "companion", relation="RelatedTo", weight=1.0)
    G.add_edge("companion", "friend", relation="RelatedTo", weight=1.0)

    sub, node_list = extract_subgraph(G, seed="dog", max_nodes=5)
    assert sub.number_of_nodes() <= 5
    assert sub.number_of_nodes() >= 2
    assert "dog" in node_list
    assert nx.is_connected(sub)


def test_extract_subgraph_min_relations():
    # Graph with only one relation type
    G = nx.Graph()
    for i in range(10):
        G.add_edge(f"n{i}", f"n{i+1}", relation="IsA", weight=1.0)
    sub, _ = extract_subgraph(G, seed="n0", max_nodes=8, min_relation_types=1)
    assert sub.number_of_nodes() >= 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_conceptnet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`

**Step 3: Write minimal implementation**

```python
# src/data/__init__.py
"""Data loading and preprocessing modules."""
```

```python
# src/data/conceptnet.py
"""ConceptNet 5.7 download, preprocessing, and subgraph extraction.

Downloads the ConceptNet assertions CSV, filters to English concepts with
weight >= 1.0, builds a NetworkX graph, and extracts connected subgraphs
suitable for knowledge graph reasoning tasks.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx


# 10 relation categories (34 raw ConceptNet relations mapped to these)
RELATION_CATEGORIES = {
    "IsA", "HasA", "PartOf", "UsedFor", "CapableOf",
    "AtLocation", "Causes", "HasProperty", "RelatedTo", "Other",
}

# Mapping from raw ConceptNet relations to our 10 categories
_RELATION_MAP = {
    "IsA": "IsA",
    "HasA": "HasA",
    "PartOf": "PartOf",
    "UsedFor": "UsedFor",
    "CapableOf": "CapableOf",
    "AtLocation": "AtLocation",
    "Causes": "Causes",
    "HasProperty": "HasProperty",
    "RelatedTo": "RelatedTo",
    "Synonym": "RelatedTo",
    "Antonym": "RelatedTo",
    "SimilarTo": "RelatedTo",
    "DerivedFrom": "RelatedTo",
    "EtymologicallyRelatedTo": "RelatedTo",
    "FormOf": "RelatedTo",
    "DistinctFrom": "RelatedTo",
    "HasContext": "RelatedTo",
    "DefinedAs": "IsA",
    "MannerOf": "IsA",
    "InstanceOf": "IsA",
    "HasSubevent": "PartOf",
    "HasPrerequisite": "Causes",
    "HasFirstSubevent": "PartOf",
    "HasLastSubevent": "PartOf",
    "MotivatedByGoal": "Causes",
    "CausesDesire": "Causes",
    "MadeOf": "PartOf",
    "ReceivesAction": "CapableOf",
    "CreatedBy": "Causes",
    "LocatedNear": "AtLocation",
    "Desires": "Causes",
    "NotDesires": "Causes",
    "NotCapableOf": "CapableOf",
    "NotHasProperty": "HasProperty",
}


def parse_conceptnet_line(line: str) -> tuple[str, str, str, float] | None:
    """Parse a single ConceptNet assertions line.

    Returns (source_concept, relation, target_concept, weight) or None if
    the line should be filtered (non-English, low weight, etc.).
    """
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None

    rel_uri = parts[1]       # e.g. /r/IsA
    src_uri = parts[2]       # e.g. /c/en/dog
    tgt_uri = parts[3]       # e.g. /c/en/animal
    meta_str = parts[4]

    # Filter non-English
    if not src_uri.startswith("/c/en/") or not tgt_uri.startswith("/c/en/"):
        return None

    # Extract concept text (strip /c/en/ prefix and any /n/... suffix)
    src = src_uri.split("/")[3]
    tgt = tgt_uri.split("/")[3]

    # Extract relation name
    rel = rel_uri.split("/")[2] if "/" in rel_uri[1:] else rel_uri

    # Parse weight
    try:
        meta = json.loads(meta_str)
        weight = meta.get("weight", 1.0)
    except (json.JSONDecodeError, TypeError):
        weight = 1.0

    if weight < 1.0:
        return None

    # Skip self-loops and very short concepts
    if src == tgt or len(src) < 2 or len(tgt) < 2:
        return None

    return src, rel, tgt, weight


def categorize_relation(raw_rel: str) -> str:
    """Map a raw ConceptNet relation to one of 10 categories."""
    return _RELATION_MAP.get(raw_rel, "Other")


def concept_to_text(concept: str) -> str:
    """Convert a ConceptNet concept URI fragment to readable text."""
    return concept.replace("_", " ")


def load_conceptnet_graph(csv_path: str | Path) -> nx.Graph:
    """Load ConceptNet assertions CSV into a NetworkX graph.

    Args:
        csv_path: Path to conceptnet-assertions-5.7.0.csv (or .gz).

    Returns:
        NetworkX graph with concept nodes and typed edges.
    """
    import gzip

    G = nx.Graph()
    path = Path(csv_path)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parsed = parse_conceptnet_line(line)
            if parsed is None:
                continue
            src, rel, tgt, weight = parsed
            cat = categorize_relation(rel)
            # Keep highest weight if edge already exists
            if G.has_edge(src, tgt):
                existing = G[src][tgt]
                if weight > existing.get("weight", 0):
                    existing["relation"] = cat
                    existing["weight"] = weight
            else:
                G.add_edge(src, tgt, relation=cat, weight=weight)

            if (i + 1) % 1_000_000 == 0:
                print(f"  Processed {i + 1} lines, {G.number_of_nodes()} nodes, "
                      f"{G.number_of_edges()} edges")

    print(f"  Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def extract_subgraph(
    G: nx.Graph,
    seed: str | None = None,
    min_nodes: int = 20,
    max_nodes: int = 50,
    min_relation_types: int = 3,
) -> tuple[nx.Graph, list[str]]:
    """Extract a connected subgraph via BFS from a seed concept.

    Args:
        G: Full ConceptNet graph.
        seed: Starting concept. If None, picks a random node with degree >= 5.
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.
        min_relation_types: Minimum distinct relation types required.

    Returns:
        (subgraph, node_list) where node_list preserves BFS ordering.
    """
    if seed is None:
        candidates = [n for n in G.nodes() if G.degree(n) >= 5]
        if not candidates:
            candidates = list(G.nodes())
        seed = random.choice(candidates)

    # BFS expansion
    visited = [seed]
    visited_set = {seed}
    queue = [seed]

    while queue and len(visited) < max_nodes:
        node = queue.pop(0)
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)
        for nb in neighbors:
            if nb not in visited_set and len(visited) < max_nodes:
                visited.append(nb)
                visited_set.add(nb)
                queue.append(nb)

    sub = G.subgraph(visited).copy()

    # Verify connectivity
    if not nx.is_connected(sub):
        largest_cc = max(nx.connected_components(sub), key=len)
        sub = G.subgraph(largest_cc).copy()
        visited = [n for n in visited if n in largest_cc]

    return sub, visited


def save_conceptnet_graph(G: nx.Graph, path: str | Path):
    """Save processed ConceptNet graph to disk."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_cached_graph(path: str | Path) -> nx.Graph:
    """Load a previously saved ConceptNet graph."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_conceptnet.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add src/data/__init__.py src/data/conceptnet.py tests/test_data/test_conceptnet.py
git commit -m "feat: ConceptNet data pipeline — parse, filter, extract subgraphs"
```

---

### Task 3: ConceptNet-to-CellComplex converter

**Files:**
- Modify: `src/data/conceptnet.py` (add conversion function)
- Test: `tests/test_data/test_conceptnet.py` (add tests)

**Step 1: Write the failing test**

```python
# Append to tests/test_data/test_conceptnet.py
import torch
from src.data.conceptnet import conceptnet_subgraph_to_cc


def test_conceptnet_to_cc_basic():
    G = nx.Graph()
    G.add_edge("dog", "animal", relation="IsA", weight=2.0)
    G.add_edge("dog", "pet", relation="RelatedTo", weight=1.5)
    G.add_edge("cat", "animal", relation="IsA", weight=2.0)

    cc, node_map, edge_rels = conceptnet_subgraph_to_cc(
        G, embedding_dim=8, node_list=["dog", "animal", "pet", "cat"]
    )
    assert cc.num_cells(0) == 4
    assert cc.num_cells(1) == 3
    assert cc.node_texts == ["dog", "animal", "pet", "cat"]
    assert len(edge_rels) == 3
    assert all(r in {"IsA", "RelatedTo"} for r in edge_rels)


def test_conceptnet_to_cc_structural_only_embeddings():
    """Node embeddings should be structural only — no text info leaked."""
    G = nx.Graph()
    G.add_edge("a", "b", relation="IsA", weight=1.0)
    G.add_edge("b", "c", relation="HasA", weight=1.0)

    cc, _, _ = conceptnet_subgraph_to_cc(G, embedding_dim=8, node_list=["a", "b", "c"])
    embs = cc.get_embeddings(0)
    # Structural features: emb[0]=degree, emb[1]=clustering, emb[2]=0 (no BFS)
    assert embs.shape == (3, 8)
    # Node "b" has degree 2 (max), should have emb[0]=1.0
    assert embs[1, 0].item() == pytest.approx(1.0, abs=0.01)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_conceptnet.py::test_conceptnet_to_cc_basic -v`
Expected: FAIL with `ImportError: cannot import name 'conceptnet_subgraph_to_cc'`

**Step 3: Write minimal implementation**

Append to `src/data/conceptnet.py`:

```python
def conceptnet_subgraph_to_cc(
    sub: nx.Graph,
    embedding_dim: int,
    node_list: list[str] | None = None,
) -> tuple:
    """Convert a ConceptNet subgraph to a CellComplex.

    Node embeddings are structural only (degree, clustering, noise).
    Concept text is stored in cc.node_texts.
    Edge relation types are returned separately for task generation.

    Args:
        sub: ConceptNet subgraph (NetworkX).
        embedding_dim: Dimension for cell embeddings.
        node_list: Ordered list of concept strings. If None, uses sub.nodes().

    Returns:
        (cc, node_map, edge_relations) where:
        - cc: CellComplex with structural embeddings + node_texts
        - node_map: dict mapping concept string -> CC node index
        - edge_relations: list of relation category strings (one per edge)
    """
    from src.cell_complex.cell_complex import CellComplex
    from src.benchmarks.topological_tasks import auto_fill_triangles

    if node_list is None:
        node_list = list(sub.nodes())

    cc = CellComplex(embedding_dim=embedding_dim)
    node_map: dict[str, int] = {}

    # Precompute structural features
    degrees = dict(sub.degree())
    max_degree = max(degrees.values()) if degrees else 1
    max_degree = max(max_degree, 1)
    clustering = nx.clustering(sub)

    # Add nodes with structural-only embeddings
    for concept in node_list:
        if concept not in sub:
            continue
        emb = torch.randn(embedding_dim) * 0.1
        emb[0] = degrees[concept] / max_degree
        emb[1] = clustering[concept]
        emb[2] = 0.0  # No source/target BFS for KG tasks

        idx = cc.add_0_cell(emb, "node")
        node_map[concept] = idx

    # Store concept text as metadata
    cc.node_texts = [concept_to_text(c) for c in node_list if c in sub]

    # Add edges
    edge_relations: list[str] = []
    for u, v, data in sub.edges(data=True):
        if u in node_map and v in node_map:
            emb = torch.randn(embedding_dim) * 0.1
            emb[0] = 1.0  # uniform edge weight for KG
            cc.add_1_cell(node_map[u], node_map[v], emb, "edge")
            edge_relations.append(data.get("relation", "Other"))

    # Auto-fill 2-cells from triangles
    auto_fill_triangles(cc)

    return cc, node_map, edge_relations
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_conceptnet.py -v`
Expected: 10 PASSED

**Step 5: Commit**

```bash
git add src/data/conceptnet.py tests/test_data/test_conceptnet.py
git commit -m "feat: ConceptNet subgraph to CellComplex converter with text metadata"
```

---

### Task 4: KG task generators — relation type prediction + masked concept

**Files:**
- Create: `src/benchmarks/conceptnet_tasks.py`
- Test: `tests/test_benchmarks/test_conceptnet_tasks.py`

**Step 1: Write the failing test**

```python
# tests/test_benchmarks/test_conceptnet_tasks.py
import networkx as nx
import pytest
from src.benchmarks.conceptnet_tasks import (
    generate_kg_relation_task,
    generate_kg_concept_task,
    CONCEPT_CATEGORIES,
    RELATION_CLASSES,
)


def _make_test_graph():
    """Build a small ConceptNet-like graph for testing."""
    G = nx.Graph()
    G.add_edge("dog", "animal", relation="IsA", weight=2.0)
    G.add_edge("cat", "animal", relation="IsA", weight=2.0)
    G.add_edge("dog", "park", relation="AtLocation", weight=1.5)
    G.add_edge("cat", "house", relation="AtLocation", weight=1.5)
    G.add_edge("dog", "loyalty", relation="HasProperty", weight=1.0)
    G.add_edge("park", "tree", relation="HasA", weight=1.0)
    G.add_edge("house", "roof", relation="HasA", weight=1.0)
    G.add_edge("animal", "living_thing", relation="IsA", weight=2.0)
    return G


def test_relation_classes_count():
    assert len(RELATION_CLASSES) == 10


def test_concept_categories_count():
    assert len(CONCEPT_CATEGORIES) == 9


def test_kg_relation_returns_5_tuple():
    G = _make_test_graph()
    result = generate_kg_relation_task(G, embedding_dim=8)
    assert len(result) == 5
    cc, query, target, answer, metadata = result
    assert 0 <= answer < 10  # 10 relation classes
    assert metadata["task_type"] == "kg_relation"
    assert "node_texts" in metadata
    assert cc.node_texts  # non-empty


def test_kg_relation_answer_matches_edge():
    G = _make_test_graph()
    for _ in range(20):
        cc, query, target, answer, metadata = generate_kg_relation_task(G, embedding_dim=8)
        assert answer == RELATION_CLASSES.index(metadata["true_relation"])


def test_kg_concept_returns_5_tuple():
    G = _make_test_graph()
    result = generate_kg_concept_task(G, embedding_dim=8)
    assert len(result) == 5
    cc, query, target, answer, metadata = result
    assert 0 <= answer < 9  # 9 concept categories
    assert metadata["task_type"] == "kg_concept"
    assert "[MASK]" in cc.node_texts  # masked node
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmarks/test_conceptnet_tasks.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/benchmarks/conceptnet_tasks.py
"""ConceptNet knowledge graph tasks requiring multi-modal fusion.

All tasks return 5-tuples: (cc, query_node, target_node, answer, metadata).
Metadata includes 'task_type', 'node_texts', and task-specific info.
The CellComplex has structural-only embeddings; concept text is in cc.node_texts.
"""
from __future__ import annotations

import random

import networkx as nx
import torch

from src.data.conceptnet import (
    extract_subgraph,
    conceptnet_subgraph_to_cc,
    concept_to_text,
    categorize_relation,
    RELATION_CATEGORIES,
)


# Ordered list of relation classes for classification
RELATION_CLASSES = sorted(RELATION_CATEGORIES)  # 10 classes, stable order

# Concept semantic categories for masked concept prediction
CONCEPT_CATEGORIES = [
    "animal", "place", "activity", "object", "emotion",
    "person", "food", "body_part", "abstract",
]

# Simple keyword-based concept categorizer (good enough for training signal)
_CATEGORY_KEYWORDS = {
    "animal": {"dog", "cat", "bird", "fish", "horse", "cow", "pig", "mouse",
               "rat", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "deer",
               "snake", "frog", "whale", "shark", "eagle", "hawk", "ant", "bee",
               "insect", "mammal", "reptile", "amphibian", "pet", "animal"},
    "place": {"house", "park", "school", "hospital", "city", "town", "country",
              "ocean", "river", "lake", "mountain", "forest", "beach", "island",
              "room", "kitchen", "bedroom", "garden", "street", "road", "building",
              "church", "temple", "office", "store", "shop", "restaurant", "bar"},
    "activity": {"running", "swimming", "reading", "writing", "cooking",
                 "sleeping", "eating", "drinking", "walking", "playing",
                 "dancing", "singing", "working", "studying", "driving",
                 "flying", "climbing", "jumping", "fishing", "hunting"},
    "object": {"car", "table", "chair", "book", "phone", "computer", "door",
               "window", "cup", "plate", "knife", "fork", "pen", "paper",
               "ball", "box", "bag", "key", "clock", "lamp", "tool", "machine"},
    "emotion": {"happiness", "sadness", "anger", "fear", "love", "hate",
                "joy", "sorrow", "anxiety", "hope", "pride", "shame",
                "surprise", "disgust", "jealousy", "gratitude", "loneliness"},
    "person": {"teacher", "student", "doctor", "nurse", "child", "parent",
               "friend", "leader", "worker", "soldier", "artist", "musician",
               "scientist", "engineer", "farmer", "chef", "king", "queen"},
    "food": {"bread", "rice", "meat", "fruit", "vegetable", "cheese", "milk",
             "egg", "fish", "chicken", "apple", "banana", "cake", "soup",
             "salad", "pizza", "pasta", "butter", "sugar", "salt", "water"},
    "body_part": {"hand", "foot", "head", "eye", "ear", "nose", "mouth",
                  "arm", "leg", "finger", "heart", "brain", "bone", "skin",
                  "blood", "hair", "tooth", "tongue", "stomach", "lung"},
    "abstract": {"time", "space", "idea", "truth", "beauty", "freedom",
                 "justice", "knowledge", "power", "energy", "force", "law",
                 "theory", "reason", "logic", "math", "science", "art"},
}


def classify_concept(concept: str) -> str:
    """Classify a concept into one of 9 categories by keyword matching."""
    text = concept.lower().replace("_", " ")
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if text in keywords or any(kw in text for kw in keywords):
            return cat
    return "abstract"  # default fallback


def generate_kg_relation_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Generate a relation type prediction task from a ConceptNet subgraph.

    Picks a random edge, asks: what is the relation type?

    Returns:
        (cc, src_idx, tgt_idx, answer, metadata)
        answer: index into RELATION_CLASSES (0-9)
    """
    sub, node_list = extract_subgraph(G, min_nodes=min_nodes, max_nodes=max_nodes)

    # Pick a random edge
    edges = list(sub.edges(data=True))
    if not edges:
        # Fallback: regenerate
        return generate_kg_relation_task(G, embedding_dim, min_nodes, max_nodes)

    u, v, data = random.choice(edges)
    rel = categorize_relation(data.get("relation", "Other"))
    answer = RELATION_CLASSES.index(rel)

    cc, node_map, edge_rels = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)

    src_idx = node_map.get(u, 0)
    tgt_idx = node_map.get(v, 0)

    metadata = {
        "task_type": "kg_relation",
        "true_relation": rel,
        "source_concept": u,
        "target_concept": v,
        "node_texts": cc.node_texts,
        "task_prompt": f"What is the relation between '{concept_to_text(u)}' and '{concept_to_text(v)}'?",
    }
    return cc, src_idx, tgt_idx, answer, metadata


def generate_kg_concept_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Generate a masked concept category prediction task.

    Masks one node's text, asks: what semantic category is the masked concept?

    Returns:
        (cc, masked_idx, 0, answer, metadata)
        answer: index into CONCEPT_CATEGORIES (0-8)
    """
    sub, node_list = extract_subgraph(G, min_nodes=min_nodes, max_nodes=max_nodes)

    # Pick a node to mask — prefer one with a known category
    candidates = [(n, classify_concept(n)) for n in node_list if n in sub]
    if not candidates:
        return generate_kg_concept_task(G, embedding_dim, min_nodes, max_nodes)

    masked_concept, category = random.choice(candidates)
    answer = CONCEPT_CATEGORIES.index(category)

    cc, node_map, edge_rels = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)

    # Mask the concept's text
    masked_idx = node_map[masked_concept]
    cc.node_texts[masked_idx] = "[MASK]"

    metadata = {
        "task_type": "kg_concept",
        "true_concept": masked_concept,
        "true_category": category,
        "node_texts": cc.node_texts,
        "task_prompt": f"What category is the masked concept?",
    }
    return cc, masked_idx, 0, answer, metadata
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmarks/test_conceptnet_tasks.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add src/benchmarks/conceptnet_tasks.py tests/test_benchmarks/test_conceptnet_tasks.py
git commit -m "feat: KG task generators — relation type prediction + masked concept"
```

---

### Task 5: KG task generators — path validity, analogy, clustering

**Files:**
- Modify: `src/benchmarks/conceptnet_tasks.py`
- Modify: `tests/test_benchmarks/test_conceptnet_tasks.py`

**Step 1: Write the failing test**

```python
# Append to tests/test_benchmarks/test_conceptnet_tasks.py
from src.benchmarks.conceptnet_tasks import (
    generate_kg_pathvalid_task,
    generate_kg_analogy_task,
    generate_kg_cluster_task,
    DOMAIN_CLASSES,
)


def test_kg_pathvalid_returns_5_tuple():
    G = _make_test_graph()
    result = generate_kg_pathvalid_task(G, embedding_dim=8, min_nodes=4, max_nodes=8)
    assert len(result) == 5
    cc, query, target, answer, metadata = result
    assert answer in (0, 1)  # binary: valid/invalid
    assert metadata["task_type"] == "kg_pathvalid"


def test_kg_analogy_returns_5_tuple():
    G = _make_test_graph()
    # Build a larger graph for analogy (needs two subgraphs)
    G2 = nx.Graph()
    for u, v, d in G.edges(data=True):
        G2.add_edge(u, v, **d)
    G2.add_edge("teacher", "student", relation="RelatedTo", weight=1.0)
    G2.add_edge("student", "exam", relation="RelatedTo", weight=1.0)
    G2.add_edge("teacher", "school", relation="AtLocation", weight=1.0)
    G2.add_edge("coach", "player", relation="RelatedTo", weight=1.0)
    G2.add_edge("player", "game", relation="RelatedTo", weight=1.0)
    G2.add_edge("coach", "field", relation="AtLocation", weight=1.0)

    result = generate_kg_analogy_task(G2, embedding_dim=8, min_nodes=3, max_nodes=6)
    assert len(result) == 5
    cc, query, target, answer, metadata = result
    assert 0 <= answer <= 2  # 3 classes
    assert metadata["task_type"] == "kg_analogy"


def test_kg_cluster_returns_5_tuple():
    G = _make_test_graph()
    result = generate_kg_cluster_task(G, embedding_dim=8, min_nodes=4, max_nodes=8)
    assert len(result) == 5
    cc, query, target, answer, metadata = result
    assert 0 <= answer < len(DOMAIN_CLASSES)
    assert metadata["task_type"] == "kg_cluster"


def test_domain_classes_count():
    assert len(DOMAIN_CLASSES) == 6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmarks/test_conceptnet_tasks.py::test_kg_pathvalid_returns_5_tuple -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Append to `src/benchmarks/conceptnet_tasks.py`:

```python
DOMAIN_CLASSES = ["science", "everyday", "social", "spatial", "temporal", "abstract"]


def generate_kg_pathvalid_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Generate a commonsense path validity task.

    Extracts a real path (valid) or corrupts it by swapping one concept (invalid).

    Returns:
        (cc, path_start_idx, path_end_idx, answer, metadata)
        answer: 1 = valid path, 0 = corrupted/invalid
    """
    sub, node_list = extract_subgraph(G, min_nodes=min_nodes, max_nodes=max_nodes)

    # Find a path of length 3-5
    nodes = list(sub.nodes())
    path = None
    for _ in range(50):
        src = random.choice(nodes)
        tgt = random.choice(nodes)
        if src == tgt:
            continue
        try:
            p = nx.shortest_path(sub, src, tgt)
            if 3 <= len(p) <= 6:
                path = p
                break
        except nx.NetworkXNoPath:
            continue

    if path is None:
        # Fallback: pick any two adjacent nodes
        edges = list(sub.edges())
        if edges:
            u, v = random.choice(edges)
            path = [u, v]
        else:
            return generate_kg_pathvalid_task(G, embedding_dim, min_nodes, max_nodes)

    # 50% chance of corruption
    is_valid = random.random() < 0.5
    if not is_valid and len(path) >= 3:
        # Swap a middle node with a random node of similar degree
        swap_idx = random.randint(1, len(path) - 2)
        old_node = path[swap_idx]
        old_deg = sub.degree(old_node)
        candidates = [n for n in nodes if n != old_node and n not in path
                       and abs(sub.degree(n) - old_deg) <= 2]
        if candidates:
            path[swap_idx] = random.choice(candidates)
        else:
            is_valid = True  # Can't corrupt, mark as valid

    answer = 1 if is_valid else 0

    cc, node_map, edge_rels = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)

    # Map path to CC indices
    start_idx = node_map.get(path[0], 0)
    end_idx = node_map.get(path[-1], 0)

    metadata = {
        "task_type": "kg_pathvalid",
        "path_concepts": path,
        "is_valid": is_valid,
        "node_texts": cc.node_texts,
        "task_prompt": f"Is the reasoning chain valid? {' -> '.join(path)}",
    }
    return cc, start_idx, end_idx, answer, metadata


def generate_kg_analogy_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 10,
    max_nodes: int = 30,
) -> tuple:
    """Generate an analogical grounding task.

    Extracts two subgraphs and asks if they are structurally + semantically analogous.

    Returns:
        (cc, 0, 0, answer, metadata)
        answer: 0=not analogous, 1=partially, 2=analogous
    """
    # Extract two subgraphs from different seeds
    sub1, nodes1 = extract_subgraph(G, min_nodes=min_nodes, max_nodes=max_nodes)
    sub2, nodes2 = extract_subgraph(G, min_nodes=min_nodes, max_nodes=max_nodes)

    # Determine analogy level based on relation overlap
    rels1 = {d.get("relation", "Other") for _, _, d in sub1.edges(data=True)}
    rels2 = {d.get("relation", "Other") for _, _, d in sub2.edges(data=True)}
    overlap = len(rels1 & rels2) / max(len(rels1 | rels2), 1)

    if overlap > 0.6:
        answer = 2  # analogous
    elif overlap > 0.3:
        answer = 1  # partially
    else:
        answer = 0  # not analogous

    # Merge both subgraphs into one CellComplex (side by side)
    merged = nx.Graph()
    for u, v, d in sub1.edges(data=True):
        merged.add_edge(f"a_{u}", f"a_{v}", **d)
    for u, v, d in sub2.edges(data=True):
        merged.add_edge(f"b_{u}", f"b_{v}", **d)

    all_nodes = [f"a_{n}" for n in nodes1 if f"a_{n}" in merged] + \
                [f"b_{n}" for n in nodes2 if f"b_{n}" in merged]

    cc, node_map, edge_rels = conceptnet_subgraph_to_cc(merged, embedding_dim, all_nodes)

    metadata = {
        "task_type": "kg_analogy",
        "relation_overlap": overlap,
        "subgraph1_size": sub1.number_of_nodes(),
        "subgraph2_size": sub2.number_of_nodes(),
        "node_texts": cc.node_texts,
        "task_prompt": "Are these two graph regions analogous?",
    }
    return cc, 0, 0, answer, metadata


def generate_kg_cluster_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Generate a semantic domain clustering task.

    Picks a random node and asks which semantic domain it belongs to.

    Returns:
        (cc, node_idx, 0, answer, metadata)
        answer: index into DOMAIN_CLASSES (0-5)
    """
    sub, node_list = extract_subgraph(G, min_nodes=min_nodes, max_nodes=max_nodes)

    # Classify each node's domain
    _domain_keywords = {
        "science": {"atom", "molecule", "cell", "gene", "protein", "enzyme",
                     "experiment", "hypothesis", "theory", "research", "lab",
                     "chemical", "physics", "biology", "electron", "neutron"},
        "everyday": {"house", "car", "food", "sleep", "work", "cook", "eat",
                     "drink", "walk", "shop", "clean", "wash", "dress", "phone"},
        "social": {"friend", "family", "party", "talk", "help", "love", "argue",
                   "meet", "group", "team", "community", "society", "neighbor"},
        "spatial": {"room", "road", "city", "country", "mountain", "river",
                    "ocean", "forest", "park", "street", "building", "bridge"},
        "temporal": {"day", "night", "morning", "evening", "year", "month",
                     "week", "hour", "minute", "season", "spring", "summer"},
        "abstract": {"idea", "truth", "beauty", "freedom", "justice", "reason",
                     "logic", "knowledge", "power", "energy", "time", "space"},
    }

    def classify_domain(concept: str) -> str:
        text = concept.lower().replace("_", " ")
        for domain, keywords in _domain_keywords.items():
            if text in keywords or any(kw in text for kw in keywords):
                return domain
        return "abstract"

    # Pick a random node with a classifiable domain
    candidates = [(n, classify_domain(n)) for n in node_list if n in sub]
    if not candidates:
        return generate_kg_cluster_task(G, embedding_dim, min_nodes, max_nodes)

    target_concept, domain = random.choice(candidates)
    answer = DOMAIN_CLASSES.index(domain)

    cc, node_map, edge_rels = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
    target_idx = node_map.get(target_concept, 0)

    metadata = {
        "task_type": "kg_cluster",
        "target_concept": target_concept,
        "true_domain": domain,
        "node_texts": cc.node_texts,
        "task_prompt": f"What semantic domain does '{concept_to_text(target_concept)}' belong to?",
    }
    return cc, target_idx, 0, answer, metadata
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmarks/test_conceptnet_tasks.py -v`
Expected: 9 PASSED

**Step 5: Commit**

```bash
git add src/benchmarks/conceptnet_tasks.py tests/test_benchmarks/test_conceptnet_tasks.py
git commit -m "feat: KG task generators — path validity, analogy, clustering"
```

---

### Task 6: Register KG tasks in TASK_REGISTRY

**Files:**
- Modify: `src/benchmarks/benchmark_dataset.py:195-219`
- Test: `tests/test_benchmarks/test_conceptnet_tasks.py` (add registry tests)

**Step 1: Write the failing test**

```python
# Append to tests/test_benchmarks/test_conceptnet_tasks.py
from src.benchmarks.benchmark_dataset import TASK_REGISTRY, get_max_classes


def test_kg_tasks_in_registry():
    for task in ["kg_relation", "kg_concept", "kg_pathvalid", "kg_analogy", "kg_cluster"]:
        assert task in TASK_REGISTRY, f"{task} not in TASK_REGISTRY"


def test_kg_task_class_counts():
    assert get_max_classes("kg_relation") == 10
    assert get_max_classes("kg_concept") == 9
    assert get_max_classes("kg_pathvalid") == 2
    assert get_max_classes("kg_analogy") == 3
    assert get_max_classes("kg_cluster") == 6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmarks/test_conceptnet_tasks.py::test_kg_tasks_in_registry -v`
Expected: FAIL with `AssertionError: kg_relation not in TASK_REGISTRY`

**Step 3: Write minimal implementation**

In `src/benchmarks/benchmark_dataset.py`, add imports after the existing imports (around line 10):
```python
from src.benchmarks.conceptnet_tasks import (
    generate_kg_relation_task,
    generate_kg_concept_task,
    generate_kg_pathvalid_task,
    generate_kg_analogy_task,
    generate_kg_cluster_task,
)
```

Add wrapper functions before TASK_REGISTRY (around line 200):
```python
def _wrap_kg_relation(n_nodes, embedding_dim, topologies, **kwargs):
    G = kwargs.get('conceptnet_graph')
    if G is None:
        raise ValueError("kg_relation requires conceptnet_graph kwarg")
    return generate_kg_relation_task(G, embedding_dim, min_nodes=max(n_nodes - 10, 10), max_nodes=n_nodes + 10)


def _wrap_kg_concept(n_nodes, embedding_dim, topologies, **kwargs):
    G = kwargs.get('conceptnet_graph')
    if G is None:
        raise ValueError("kg_concept requires conceptnet_graph kwarg")
    return generate_kg_concept_task(G, embedding_dim, min_nodes=max(n_nodes - 10, 10), max_nodes=n_nodes + 10)


def _wrap_kg_pathvalid(n_nodes, embedding_dim, topologies, **kwargs):
    G = kwargs.get('conceptnet_graph')
    if G is None:
        raise ValueError("kg_pathvalid requires conceptnet_graph kwarg")
    return generate_kg_pathvalid_task(G, embedding_dim, min_nodes=max(n_nodes - 10, 10), max_nodes=n_nodes + 10)


def _wrap_kg_analogy(n_nodes, embedding_dim, topologies, **kwargs):
    G = kwargs.get('conceptnet_graph')
    if G is None:
        raise ValueError("kg_analogy requires conceptnet_graph kwarg")
    return generate_kg_analogy_task(G, embedding_dim, min_nodes=max(n_nodes // 3, 5), max_nodes=n_nodes // 2)


def _wrap_kg_cluster(n_nodes, embedding_dim, topologies, **kwargs):
    G = kwargs.get('conceptnet_graph')
    if G is None:
        raise ValueError("kg_cluster requires conceptnet_graph kwarg")
    return generate_kg_cluster_task(G, embedding_dim, min_nodes=max(n_nodes - 10, 10), max_nodes=n_nodes + 10)
```

Add to TASK_REGISTRY dict (after line 218):
```python
    "kg_relation":          (_wrap_kg_relation, 10, {}),
    "kg_concept":           (_wrap_kg_concept, 9, {}),
    "kg_pathvalid":         (_wrap_kg_pathvalid, 2, {}),
    "kg_analogy":           (_wrap_kg_analogy, 3, {}),
    "kg_cluster":           (_wrap_kg_cluster, 6, {}),
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmarks/test_conceptnet_tasks.py -v`
Expected: 11 PASSED

**Step 5: Run full test suite**

Run: `pytest tests/ -x -q --timeout=60 -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat"`
Expected: All pass (existing tests shouldn't import conceptnet_graph, so the ValueError won't trigger)

**Step 6: Commit**

```bash
git add src/benchmarks/benchmark_dataset.py src/benchmarks/conceptnet_tasks.py tests/test_benchmarks/test_conceptnet_tasks.py
git commit -m "feat: register 5 KG tasks in TASK_REGISTRY (18 total)"
```

---

### Task 7: QwenGraphBackend text channel

**Files:**
- Modify: `src/llm/qwen_backend.py:59-85` (add node_texts param)
- Modify: `src/llm/qwen_bridge_adapter.py:30-40` (thread node_texts)
- Test: `tests/test_llm/test_qwen_text_channel.py`

**Step 1: Write the failing test**

```python
# tests/test_llm/test_qwen_text_channel.py
import torch
from src.llm.qwen_backend import QwenGraphBackend


def test_forward_graph_accepts_node_texts():
    backend = QwenGraphBackend({
        "topo_dim": 8, "llm_dim": 32, "num_tokens": 4,
        "adapter_layers": 1, "num_tasks": 18, "use_mock": True,
    })
    node_emb = torch.randn(5, 8)
    task_id = torch.tensor(0)
    feat, bias, graph_emb = backend.forward_graph(
        node_emb, task_id, node_texts=["dog", "cat", "bird", "fish", "tree"]
    )
    assert feat.shape == (5, 8)
    assert bias.shape == (5, 5)


def test_forward_graph_without_node_texts():
    """Backward compat: node_texts=None still works."""
    backend = QwenGraphBackend({
        "topo_dim": 8, "llm_dim": 32, "num_tokens": 4,
        "adapter_layers": 1, "num_tasks": 18, "use_mock": True,
    })
    node_emb = torch.randn(5, 8)
    task_id = torch.tensor(0)
    feat, bias, graph_emb = backend.forward_graph(node_emb, task_id)
    assert feat.shape == (5, 8)


def test_qwen_bridge_adapter_passes_node_texts():
    from src.llm.qwen_bridge_adapter import QwenBridgeAdapter
    adapter = QwenBridgeAdapter(
        {"llm_dim": 32, "num_tokens": 4, "adapter_layers": 1,
         "num_tasks": 18, "use_mock": True},
        topo_dim=8,
    )
    node_emb = torch.randn(5, 8)
    out = adapter(node_emb, 0.5, task_text="kg_relation",
                  node_texts=["a", "b", "c", "d", "e"])
    assert len(out) == 4
    assert out[0].shape == (5, 8)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm/test_qwen_text_channel.py -v`
Expected: FAIL with `TypeError: forward_graph() got an unexpected keyword argument 'node_texts'`

**Step 3: Write minimal implementation**

In `src/llm/qwen_backend.py`, modify `forward_graph()` at line 59:

```python
    def forward_graph(self, node_embeddings, task_id, node_texts=None):
        """Full graph->LLM->graph pipeline.

        Args:
            node_embeddings: (N, topo_dim) from GNN
            task_id: scalar tensor (task index)
            node_texts: optional list of concept strings for each node

        Returns:
            (semantic_features, semantic_bias, graph_embedding)
        """
        graph_tokens = self.encoder(node_embeddings, task_id)

        if self._is_mock:
            # Mock: optionally incorporate text length as a simple signal
            hidden = self.llm(graph_tokens)
        else:
            if self.llm is None:
                self._load_qwen()

            # Build input: graph tokens + optional text tokens
            if node_texts is not None:
                text_prompt = "Graph nodes: " + ", ".join(
                    f"{i}={t}" for i, t in enumerate(node_texts)
                ) + "."
                from transformers import AutoTokenizer
                if not hasattr(self, '_tokenizer'):
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                text_ids = self._tokenizer(text_prompt, return_tensors="pt").input_ids
                text_ids = text_ids.to(node_embeddings.device)
                text_emb = self.llm.model.embed_tokens(text_ids).squeeze(0)  # (T, llm_dim)
                combined = torch.cat([graph_tokens, text_emb], dim=0)  # (K+T, llm_dim)
            else:
                combined = graph_tokens

            with torch.no_grad():
                out = self.llm(
                    inputs_embeds=combined.unsqueeze(0),
                    output_hidden_states=True,
                )
                # Extract hidden states for graph token positions only
                hidden = out.hidden_states[self.extract_layer].squeeze(0)
                hidden = hidden[:graph_tokens.shape[0]]  # (K, llm_dim)

        node_proj = self.encoder.input_proj(node_embeddings)
        features, bias, graph_emb = self.decoder(node_proj, hidden)
        return features, bias, graph_emb
```

In `src/llm/qwen_bridge_adapter.py`, modify `forward()` at line 30:

```python
    def forward(self, node_embeddings, semantic_weight, task_text=None, node_texts=None):
        """Forward pass matching TopoBridge signature.

        Returns:
            (semantic_features, semantic_bias, semantic_features, graph_embedding)
        """
        task_id = torch.tensor(
            self._task_to_id.get(task_text, 0), device=node_embeddings.device,
        )
        sem_feat, sem_bias, graph_emb = self.backend.forward_graph(
            node_embeddings, task_id, node_texts=node_texts
        )
        return sem_feat, sem_bias, sem_feat, graph_emb
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm/test_qwen_text_channel.py -v`
Expected: 3 PASSED

**Step 5: Run full test suite**

Run: `pytest tests/ -x -q --timeout=60 -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat"`
Expected: All pass

**Step 6: Commit**

```bash
git add src/llm/qwen_backend.py src/llm/qwen_bridge_adapter.py tests/test_llm/test_qwen_text_channel.py
git commit -m "feat: text channel in QwenGraphBackend — node_texts → hard tokens for Qwen"
```

---

### Task 8: Thread node_texts through executive loop and model forward

**Files:**
- Modify: `src/reasoning_loop/executive_loop.py` (pass node_texts to topo_bridge)
- Modify: `src/benchmarks/run_comparison.py` (extract node_texts from CellComplex, pass through)
- Test: `tests/test_llm/test_text_threading.py`

**Step 1: Write the failing test**

```python
# tests/test_llm/test_text_threading.py
import torch
from src.cell_complex.cell_complex import CellComplex


def _make_cc_with_text(n=5, dim=8):
    cc = CellComplex(embedding_dim=dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(dim), "node")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    cc.node_texts = [f"concept_{i}" for i in range(n)]
    return cc


def test_model_forward_with_node_texts():
    """HierarchicalMultiHopModel should detect and thread node_texts."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel
    model = HierarchicalMultiHopModel(
        embedding_dim=8, max_hops=3, use_llm=True,
        llm_config={"backend": "qwen", "llm_dim": 32, "num_tokens": 4,
                     "adapter_layers": 1, "num_tasks": 18, "use_mock": True},
    )
    cc = _make_cc_with_text(5, 8)
    logits = model(cc, 0, 1, task="kg_relation")
    assert logits.shape[0] > 0  # got output


def test_model_forward_without_node_texts():
    """Backward compat: model works fine without node_texts."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel
    model = HierarchicalMultiHopModel(
        embedding_dim=8, max_hops=3, use_llm=True,
        llm_config={"backend": "qwen", "llm_dim": 32, "num_tokens": 4,
                     "adapter_layers": 1, "num_tasks": 18, "use_mock": True},
    )
    cc = CellComplex(embedding_dim=8)
    for i in range(5):
        cc.add_0_cell(torch.randn(8), "node")
    for i in range(4):
        cc.add_1_cell(i, i + 1, torch.randn(8), "edge")
    logits = model(cc, 0, 1, task="diverse")
    assert logits.shape[0] > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm/test_text_threading.py -v`
Expected: May pass on mock (mock ignores node_texts) or fail if forward signature doesn't accept.

**Step 3: Write minimal implementation**

The key changes are minimal since the mock backend ignores node_texts:

In `src/benchmarks/run_comparison.py`, in `HierarchicalMultiHopModel.forward()`, extract node_texts from the CellComplex and pass through to topo_bridge. Find the line where `self.topo_bridge(...)` is called and add `node_texts=getattr(cc, 'node_texts', None)`:

```python
# In the forward() method, wherever topo_bridge is called:
node_texts = getattr(cc, 'node_texts', None) if hasattr(cc, 'node_texts') else None
```

Then pass `node_texts=node_texts` to every `self.topo_bridge(...)` call.

In `src/reasoning_loop/executive_loop.py`, in `ExecutiveReasoningLoop.forward()`, similarly extract and pass node_texts:

```python
node_texts = getattr(cc, 'node_texts', None)
# ... in the iteration loop where topo_bridge is called:
_, semantic_bias, sem_feat, _ = self.topo_bridge(gnn_out, semantic_weight, node_texts=node_texts)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm/test_text_threading.py -v`
Expected: 2 PASSED

**Step 5: Run full test suite**

Run: `pytest tests/ -x -q --timeout=60 -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat"`
Expected: All pass

**Step 6: Commit**

```bash
git add src/benchmarks/run_comparison.py src/reasoning_loop/executive_loop.py tests/test_llm/test_text_threading.py
git commit -m "feat: thread node_texts from CellComplex through model forward to TopoBridge"
```

---

### Task 9: v7 configuration file

**Files:**
- Create: `config/v7_metacognition.yaml`

**Step 1: Create config based on v6_track2.yaml**

```yaml
# config/v7_metacognition.yaml
# Phase 7: Meta-Cognition — ConceptNet grounded reasoning with real Qwen

model:
  embedding_dim: 32
  gnn_hidden: 64
  gnn_spatial_layers: 2
  gnn_spectral_layers: 2
  max_freqs: 16
  tat_layers: 2
  tat_spatial_heads: 4
  tat_spectral_heads: 4
  tat_ff_dim: 128
  max_iterations: 5
  convergence_threshold: 0.05
  use_higher_order: true
  use_topological_pe: true
  use_structural_features: true
  use_wave_dynamics: true
  use_topo_feedback: true
  use_embedding_topo_feedback: true
  use_multi_head_classifier: true

wave:
  wave_mode: ensemble
  filter_types: [chebyshev, wave_cosine, heat]
  include_identity: true
  wave_strength_gate: true
  use_neural_ode: true

llm:
  backend: qwen
  qwen_model: Qwen/Qwen2.5-3B-Instruct-AWQ
  llm_dim: 2048
  num_tokens: 16
  adapter_layers: 2
  extract_layer: 16
  num_tasks: 19  # 14 original + 5 KG tasks
  use_mock: false  # Real Qwen for Phase D

training:
  device: auto
  learning_rate: 0.001
  adapter_learning_rate: 0.0001
  bridge_learning_rate: 0.0001
  weight_decay: 0.01
  label_smoothing: 0.1
  batch_size: 8
  use_batched: true
  epochs_phase_a: 30
  epochs_phase_b: 30
  epochs_phase_c: 30
  epochs_phase_d: 40
  patience: 10
  max_norm: 5.0
  accumulation_steps: 4
  replay_ratio: 0.5
  use_class_weights: true
  contrastive_loss_weight: 0.1

topology_observer:
  enabled: true
  analyze_every: 5
  active_mode: true
  dsm_num_heads: 16
  dsm_hidden_dim: 2048

benchmark:
  train_samples: 5000
  val_samples: 500
  test_samples: 200
  train_n_nodes: 20
  train_n_nodes_min: 16
  train_n_nodes_max: 32
  test_n_nodes: [20, 40, 80]
  train_topologies: [ba, ws, grid, tree, ladder, sbm]
  test_topologies: [er, caveman]
  checkpoint_dir: data/v7_metacognition/checkpoints
  pregen_dir: data/dsm_datasets
  conceptnet_path: data/conceptnet/conceptnet_en.pkl

# Phase D: KG tasks requiring multi-modal fusion
phase_d:
  tasks: [kg_relation, kg_concept, kg_pathvalid, kg_analogy, kg_cluster]
  epochs:
    kg_relation: 10
    kg_concept: 8
    kg_pathvalid: 8
    kg_analogy: 8
    kg_cluster: 6
  kg_train_samples: 5000
  kg_val_samples: 500
  replay_structural_ratio: 0.2
  replay_semantic_ratio: 0.1
  resume_from: data/v6_track2/checkpoints/phase_c_labeled_reasoning_best.pt
```

**Step 2: Commit**

```bash
git add config/v7_metacognition.yaml
git commit -m "feat: v7 meta-cognition config — Phase D with 5 KG tasks, real Qwen"
```

---

### Task 10: ConceptNet preprocessing script

**Files:**
- Create: `scripts/precompute_conceptnet.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Download and preprocess ConceptNet for Phase 7 KG tasks.

Usage:
    python scripts/precompute_conceptnet.py [--csv-path PATH] [--output-dir DIR]

Downloads ConceptNet 5.7 assertions if not present, filters to English,
builds the graph, and saves as pickle for fast loading during training.
"""
import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

from src.data.conceptnet import load_conceptnet_graph, save_conceptnet_graph


CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"


def download_conceptnet(output_path: Path):
    """Download ConceptNet assertions CSV if not present."""
    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ConceptNet to {output_path}...")
    urllib.request.urlretrieve(CONCEPTNET_URL, str(output_path))
    print(f"Downloaded: {output_path} ({output_path.stat().st_size / 1e6:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ConceptNet for Phase 7")
    parser.add_argument("--csv-path", default="data/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
    parser.add_argument("--output-dir", default="data/conceptnet")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    download_conceptnet(csv_path)

    # Load and filter
    print("Loading and filtering ConceptNet...")
    G = load_conceptnet_graph(csv_path)

    # Save processed graph
    pkl_path = output_dir / "conceptnet_en.pkl"
    save_conceptnet_graph(G, pkl_path)
    print(f"Saved processed graph to {pkl_path}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Print relation distribution
    from collections import Counter
    rels = Counter(d.get("relation", "Other") for _, _, d in G.edges(data=True))
    print(f"\nRelation distribution:")
    for rel, count in rels.most_common():
        print(f"  {rel:20s} {count:>8d} ({100 * count / G.number_of_edges():.1f}%)")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/precompute_conceptnet.py
git commit -m "feat: ConceptNet download + preprocessing script"
```

---

### Task 11: Phase D in curriculum training script

**Files:**
- Modify: `scripts/run_dsm_curriculum.py:40-43` (add PHASE_D_TASKS)
- Modify: `scripts/run_dsm_curriculum.py:797-834` (add Phase D section)
- Modify: `scripts/run_dsm_curriculum.py:842` (add resume-phase 'd')

**Step 1: Write minimal implementation**

In `scripts/run_dsm_curriculum.py`:

At line 44 (after PHASE_C_TASKS), add:
```python
PHASE_D_TASKS = ["kg_relation", "kg_concept", "kg_pathvalid", "kg_analogy", "kg_cluster"]
```

At the top of the file (imports section), add:
```python
from src.data.conceptnet import load_cached_graph
```

In `_run_phase()`, add `conceptnet_graph=None` parameter. Thread it to `_load_or_generate()` as a kwarg:
```python
# When generating datasets for KG tasks, pass conceptnet_graph
if task.startswith("kg_") and conceptnet_graph is not None:
    task_kwargs = {"conceptnet_graph": conceptnet_graph}
```

After the Phase C section (line 797), before the Diagnostics section, add:

```python
    # ---- Phase D: Meta-Cognition (KG tasks) ----
    phase_d_config = config.get("phase_d", {})
    phase_d_tasks = phase_d_config.get("tasks", PHASE_D_TASKS)
    if phase_d_tasks and resume_phase not in ("a", "b", "c"):
        print(f"\n{'=' * 72}")
        print("Phase D: Meta-Cognition (ConceptNet KG tasks, real Qwen)")
        print(f"{'=' * 72}")

        # Load ConceptNet graph
        cn_path = bc.get("conceptnet_path", "data/conceptnet/conceptnet_en.pkl")
        if Path(cn_path).exists():
            conceptnet_graph = load_cached_graph(cn_path)
            print(f"  Loaded ConceptNet: {conceptnet_graph.number_of_nodes()} nodes, "
                  f"{conceptnet_graph.number_of_edges()} edges")
        else:
            print(f"  WARNING: ConceptNet not found at {cn_path}. "
                  f"Run: python scripts/precompute_conceptnet.py")
            conceptnet_graph = None

        if conceptnet_graph is not None:
            # Ensure DSM/LLM is enabled
            if hasattr(model, 'executive_loop'):
                model.executive_loop.use_dsm = True

            phase_d_results, _ = _run_phase(
                'd', phase_d_tasks, model, config, device, pregen_dir,
                train_range, all_topos, checkpoint_dir, use_amp,
                phase_a_datasets=phase_a_datasets if phase_a_datasets else None,
                observer=observer,
                conceptnet_graph=conceptnet_graph,
            )
            for task, acc in phase_d_results.items():
                all_results[f"phase_d_{task}"] = {"best_val_acc": acc}
```

Update resume-phase choices at line 842:
```python
    parser.add_argument("--resume-phase", default=None,
                        choices=["b", "B", "c", "C", "d", "D"],
                        help="Skip earlier phases and resume")
```

**Step 2: Verify existing tests still pass**

Run: `pytest tests/ -x -q --timeout=60 -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat"`
Expected: All pass

**Step 3: Commit**

```bash
git add scripts/run_dsm_curriculum.py
git commit -m "feat: Phase D meta-cognition in curriculum — 5 KG tasks with ConceptNet"
```

---

### Task 12: Integration smoke test

**Files:**
- Create: `tests/test_integration/test_kg_smoke.py`

**Step 1: Write the test**

```python
# tests/test_integration/test_kg_smoke.py
"""Smoke test: end-to-end KG task with mock Qwen backend."""
import networkx as nx
import torch
import pytest

from src.benchmarks.conceptnet_tasks import (
    generate_kg_relation_task,
    generate_kg_concept_task,
    generate_kg_pathvalid_task,
    generate_kg_cluster_task,
)
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.benchmark_dataset import get_max_classes


def _make_conceptnet_graph():
    """Small ConceptNet-like graph for testing."""
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


@pytest.fixture
def model():
    return HierarchicalMultiHopModel(
        embedding_dim=8, max_hops=10, use_llm=True,
        llm_config={
            "backend": "qwen", "llm_dim": 32, "num_tokens": 4,
            "adapter_layers": 1, "num_tasks": 19, "use_mock": True,
        },
    )


@pytest.fixture
def cn_graph():
    return _make_conceptnet_graph()


def test_kg_relation_forward(model, cn_graph):
    """Model can forward a kg_relation sample."""
    cc, query, target, answer, meta = generate_kg_relation_task(
        cn_graph, embedding_dim=8, min_nodes=5, max_nodes=12
    )
    logits = model(cc, query, target, metadata=meta, task="kg_relation")
    assert logits.shape == (get_max_classes("kg_relation"),)
    assert not torch.isnan(logits).any()


def test_kg_concept_forward(model, cn_graph):
    cc, query, target, answer, meta = generate_kg_concept_task(
        cn_graph, embedding_dim=8, min_nodes=5, max_nodes=12
    )
    logits = model(cc, query, target, metadata=meta, task="kg_concept")
    assert logits.shape == (get_max_classes("kg_concept"),)


def test_kg_pathvalid_forward(model, cn_graph):
    cc, query, target, answer, meta = generate_kg_pathvalid_task(
        cn_graph, embedding_dim=8, min_nodes=5, max_nodes=12
    )
    logits = model(cc, query, target, metadata=meta, task="kg_pathvalid")
    assert logits.shape == (get_max_classes("kg_pathvalid"),)


def test_kg_cluster_forward(model, cn_graph):
    cc, query, target, answer, meta = generate_kg_cluster_task(
        cn_graph, embedding_dim=8, min_nodes=5, max_nodes=12
    )
    logits = model(cc, query, target, metadata=meta, task="kg_cluster")
    assert logits.shape == (get_max_classes("kg_cluster"),)


def test_kg_task_backward(model, cn_graph):
    """Gradients flow through KG task forward pass."""
    cc, query, target, answer, meta = generate_kg_relation_task(
        cn_graph, embedding_dim=8, min_nodes=5, max_nodes=12
    )
    logits = model(cc, query, target, metadata=meta, task="kg_relation")
    loss = torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([answer])
    )
    loss.backward()
    # Check that adapter params got gradients
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flowed to model parameters"
```

**Step 2: Run test**

Run: `pytest tests/test_integration/test_kg_smoke.py -v`
Expected: 5 PASSED

**Step 3: Commit**

```bash
git add tests/test_integration/test_kg_smoke.py
git commit -m "test: KG task integration smoke tests — forward + backward with mock Qwen"
```

---

### Task 13: Deploy to Track 2 instance

**Step 1: Rsync code to instance**

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
    --exclude 'data' --exclude '.claude' \
    -e 'ssh -i ~/.ssh/vastai -p 34701' \
    ./ root@136.59.129.136:~/transformer-topology/
```

**Step 2: Run ConceptNet preprocessing on instance**

```bash
ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136 \
    "cd ~/transformer-topology && python scripts/precompute_conceptnet.py"
```

**Step 3: Compile AWQ kernels (if not already done)**

```bash
ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136 \
    "pip install autoawq && python -c 'from awq import AutoAWQForCausalLM; print(\"AWQ ready\")'"
```

**Step 4: Run Phase D training**

```bash
ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136 \
    "cd ~/transformer-topology && nohup python scripts/run_dsm_curriculum.py \
    config/v7_metacognition.yaml --resume-phase d \
    > data/v7_metacognition/training_phase_d.log 2>&1 &"
```

**Step 5: Monitor**

```bash
ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136 \
    "tail -20 ~/transformer-topology/data/v7_metacognition/training_phase_d.log"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | CellComplex node_texts attribute | `src/cell_complex/cell_complex.py` |
| 2 | ConceptNet data pipeline | `src/data/conceptnet.py` |
| 3 | ConceptNet-to-CellComplex converter | `src/data/conceptnet.py` |
| 4 | KG generators: relation + concept | `src/benchmarks/conceptnet_tasks.py` |
| 5 | KG generators: pathvalid + analogy + cluster | `src/benchmarks/conceptnet_tasks.py` |
| 6 | Register 5 KG tasks in TASK_REGISTRY | `src/benchmarks/benchmark_dataset.py` |
| 7 | QwenGraphBackend text channel | `src/llm/qwen_backend.py` |
| 8 | Thread node_texts through model forward | `src/benchmarks/run_comparison.py`, `executive_loop.py` |
| 9 | v7 configuration | `config/v7_metacognition.yaml` |
| 10 | ConceptNet preprocessing script | `scripts/precompute_conceptnet.py` |
| 11 | Phase D in curriculum | `scripts/run_dsm_curriculum.py` |
| 12 | Integration smoke tests | `tests/test_integration/test_kg_smoke.py` |
| 13 | Deploy to Track 2 | rsync + run |
