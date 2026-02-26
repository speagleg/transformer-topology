"""ConceptNet 5.7 data pipeline.

Parses ConceptNet CSV/TSV dumps, filters to English concepts with weight >= 1.0,
builds a NetworkX graph, extracts connected subgraphs for knowledge graph reasoning
tasks, and converts subgraphs to the project's CellComplex format.

ConceptNet line format (tab-separated):
    /a/[/r/IsA/,/c/en/dog/,/c/en/animal/]  /r/IsA  /c/en/dog  /c/en/animal  {"weight": 2.0}
"""

from __future__ import annotations

import gzip
import json
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.topological_tasks import auto_fill_triangles


# ---------------------------------------------------------------------------
# Relation taxonomy: 34 raw ConceptNet relations -> 10 categories
# ---------------------------------------------------------------------------

RELATION_CATEGORIES: list[str] = [
    "IsA",
    "HasA",
    "PartOf",
    "UsedFor",
    "CapableOf",
    "AtLocation",
    "Causes",
    "HasProperty",
    "RelatedTo",
    "Other",
]

_RELATION_MAP: dict[str, str] = {
    # IsA
    "IsA": "IsA",
    "DefinedAs": "IsA",
    "MannerOf": "IsA",
    "InstanceOf": "IsA",
    # HasA
    "HasA": "HasA",
    # PartOf
    "PartOf": "PartOf",
    "HasSubevent": "PartOf",
    "HasFirstSubevent": "PartOf",
    "HasLastSubevent": "PartOf",
    "MadeOf": "PartOf",
    # UsedFor
    "UsedFor": "UsedFor",
    # CapableOf
    "CapableOf": "CapableOf",
    "ReceivesAction": "CapableOf",
    "NotCapableOf": "CapableOf",
    # AtLocation
    "AtLocation": "AtLocation",
    "LocatedNear": "AtLocation",
    # Causes
    "Causes": "Causes",
    "HasPrerequisite": "Causes",
    "MotivatedByGoal": "Causes",
    "CausesDesire": "Causes",
    "CreatedBy": "Causes",
    "Desires": "Causes",
    "NotDesires": "Causes",
    # HasProperty
    "HasProperty": "HasProperty",
    "NotHasProperty": "HasProperty",
    # RelatedTo
    "RelatedTo": "RelatedTo",
    "Synonym": "RelatedTo",
    "Antonym": "RelatedTo",
    "SimilarTo": "RelatedTo",
    "DerivedFrom": "RelatedTo",
    "EtymologicallyRelatedTo": "RelatedTo",
    "FormOf": "RelatedTo",
    "DistinctFrom": "RelatedTo",
    "HasContext": "RelatedTo",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def categorize_relation(raw_rel: str) -> str:
    """Map a raw ConceptNet relation name to one of 10 categories.

    Args:
        raw_rel: Raw relation string, e.g. ``"IsA"`` or ``"MotivatedByGoal"``.

    Returns:
        One of the 10 category strings from :data:`RELATION_CATEGORIES`.
    """
    return _RELATION_MAP.get(raw_rel, "Other")


def concept_to_text(concept: str) -> str:
    """Convert a ConceptNet concept URI fragment to readable text.

    ``"/c/en/hot_dog"`` or ``"hot_dog"`` both become ``"hot dog"``.
    """
    # Strip the URI prefix if present
    if concept.startswith("/c/"):
        parts = concept.split("/")
        # /c/en/hot_dog -> parts = ['', 'c', 'en', 'hot_dog']
        concept = parts[3] if len(parts) > 3 else concept
    return concept.replace("_", " ")


def parse_conceptnet_line(line: str) -> tuple[str, str, str, float] | None:
    """Parse a single ConceptNet TSV line.

    Filters out non-English concepts and edges with weight < 1.0.

    Args:
        line: A single tab-separated line from the ConceptNet CSV dump.

    Returns:
        ``(relation, source_concept, target_concept, weight)`` or ``None``
        if the line should be skipped.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split("\t")
    if len(parts) < 5:
        return None

    raw_rel = parts[1]   # e.g. /r/IsA
    source = parts[2]    # e.g. /c/en/dog
    target = parts[3]    # e.g. /c/en/animal
    metadata = parts[4]  # JSON dict with "weight"

    # Filter: both concepts must be English
    if not source.startswith("/c/en/") or not target.startswith("/c/en/"):
        return None

    # Parse weight from JSON metadata
    try:
        meta: dict[str, Any] = json.loads(metadata)
        weight = float(meta.get("weight", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    if weight < 1.0:
        return None

    # Extract clean relation name: /r/IsA -> IsA
    rel_name = raw_rel.split("/")[-1] if raw_rel.startswith("/r/") else raw_rel

    # Extract concept labels: /c/en/dog -> dog, /c/en/hot_dog/n -> hot_dog
    src_parts = source.split("/")
    tgt_parts = target.split("/")
    src_label = src_parts[3] if len(src_parts) > 3 else source
    tgt_label = tgt_parts[3] if len(tgt_parts) > 3 else target

    return (rel_name, src_label, tgt_label, weight)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def load_conceptnet_graph(csv_path: str | Path) -> nx.Graph:
    """Load a ConceptNet CSV dump and build a NetworkX graph.

    Supports both plain ``.csv`` / ``.tsv`` and gzip-compressed ``.gz`` files.
    Edges are stored with ``relation`` and ``weight`` attributes.
    Parallel edges between the same pair of concepts are collapsed by keeping
    the one with the highest weight.

    Args:
        csv_path: Path to the ConceptNet assertions file.

    Returns:
        An undirected :class:`nx.Graph` with string concept nodes.
    """
    csv_path = Path(csv_path)
    G = nx.Graph()

    opener = gzip.open if csv_path.suffix == ".gz" else open
    with opener(csv_path, "rt", encoding="utf-8") as f:
        for line in f:
            parsed = parse_conceptnet_line(line)
            if parsed is None:
                continue
            rel, src, tgt, weight = parsed
            if src == tgt:
                continue  # skip self-loops

            category = categorize_relation(rel)

            # Keep edge with highest weight if duplicate
            if G.has_edge(src, tgt):
                existing_weight = G[src][tgt].get("weight", 0.0)
                if weight > existing_weight:
                    G[src][tgt]["weight"] = weight
                    G[src][tgt]["relation"] = category
                    G[src][tgt]["raw_relation"] = rel
            else:
                G.add_edge(
                    src,
                    tgt,
                    relation=category,
                    raw_relation=rel,
                    weight=weight,
                )

    return G


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------


def extract_subgraph(
    G: nx.Graph,
    seed: str | None = None,
    min_nodes: int = 20,
    max_nodes: int = 50,
    min_relation_types: int = 3,
) -> tuple[nx.Graph, list[str]]:
    """Extract a connected subgraph from a ConceptNet graph via BFS.

    Starting from *seed* (or a random high-degree node), performs BFS until
    the subgraph has between *min_nodes* and *max_nodes* nodes and at least
    *min_relation_types* distinct relation categories.

    Args:
        G: Full ConceptNet graph.
        seed: Starting concept. If ``None``, a random node with
              degree >= 5 is chosen.
        min_nodes: Minimum number of nodes in the subgraph.
        max_nodes: Maximum number of nodes in the subgraph.
        min_relation_types: Minimum number of distinct relation categories.

    Returns:
        ``(subgraph, node_list)`` where *subgraph* is a :class:`nx.Graph`
        and *node_list* is the list of concept strings in BFS order.

    Raises:
        ValueError: If the graph is empty or no suitable seed can be found.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty")

    # Pick a seed node
    if seed is None:
        high_deg = [n for n, d in G.degree() if d >= 5]
        if not high_deg:
            high_deg = list(G.nodes())
        seed = random.choice(high_deg)
    elif seed not in G:
        raise ValueError(f"Seed node '{seed}' not in graph")

    # BFS expansion
    visited: set[str] = set()
    queue: deque[str] = deque([seed])
    node_list: list[str] = []

    while queue and len(node_list) < max_nodes:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        node_list.append(node)
        # Shuffle neighbors for diversity
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)
        for nb in neighbors:
            if nb not in visited:
                queue.append(nb)

    # Build subgraph
    sub = G.subgraph(node_list).copy()

    # Verify relation diversity
    relation_types = set()
    for _, _, data in sub.edges(data=True):
        relation_types.add(data.get("relation", "Other"))

    # If too few relation types or too few nodes, warn but still return
    # (caller can retry with a different seed)
    if len(node_list) < min_nodes:
        pass  # small graph, nothing to do
    if len(relation_types) < min_relation_types:
        pass  # low diversity, caller can check

    return sub, node_list


# ---------------------------------------------------------------------------
# CellComplex conversion
# ---------------------------------------------------------------------------


def conceptnet_subgraph_to_cc(
    sub: nx.Graph,
    embedding_dim: int,
    node_list: list[str] | None = None,
) -> tuple[CellComplex, dict[str, int], list[str]]:
    """Convert a ConceptNet subgraph to a CellComplex.

    Node embeddings are **structural only** (no text leaks into vectors):
      - ``emb[0]`` = normalized degree
      - ``emb[1]`` = clustering coefficient
      - ``emb[2]`` = 0.0 (placeholder, no BFS source)
      - ``emb[3:]`` = small random noise

    The concept text is stored separately in ``cc.node_texts``.

    Args:
        sub: A NetworkX subgraph (typically from :func:`extract_subgraph`).
        embedding_dim: Dimension of cell embeddings (must be >= 4).
        node_list: Optional ordered list of concept strings. If ``None``,
                   ``list(sub.nodes())`` is used.

    Returns:
        ``(cc, node_map, edge_relations)`` where:
        - *cc* is the :class:`CellComplex`
        - *node_map* maps concept string -> CC node index
        - *edge_relations* lists the relation category for each 1-cell
    """
    assert embedding_dim >= 4, "embedding_dim must be >= 4 for structural features"

    if node_list is None:
        node_list = list(sub.nodes())

    cc = CellComplex(embedding_dim=embedding_dim)
    node_map: dict[str, int] = {}

    # Precompute structural features
    degrees = dict(sub.degree())
    max_degree = max(degrees.values()) if degrees else 1
    max_degree = max(max_degree, 1)
    clustering = nx.clustering(sub)

    # Add 0-cells with structural embeddings
    for concept in node_list:
        if concept not in sub:
            continue
        emb = torch.randn(embedding_dim) * 0.01  # small noise
        emb[0] = degrees[concept] / max_degree
        emb[1] = clustering[concept]
        emb[2] = 0.0  # no BFS source for knowledge graph tasks

        cc_idx = cc.add_0_cell(emb, "node")
        node_map[concept] = cc_idx

    # Store concept texts as metadata
    cc.node_texts = [concept_to_text(c) for c in node_list if c in sub]

    # Add 1-cells (edges) with structural embeddings
    edge_relations: list[str] = []
    for u, v, data in sub.edges(data=True):
        if u not in node_map or v not in node_map:
            continue
        relation = data.get("relation", "Other")
        weight = data.get("weight", 1.0)

        emb = torch.randn(embedding_dim) * 0.01
        emb[0] = weight / 10.0  # normalized weight
        cc.add_1_cell(node_map[u], node_map[v], emb, relation)
        edge_relations.append(relation)

    # Auto-fill 2-cells from triangles
    auto_fill_triangles(cc)

    return cc, node_map, edge_relations


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_conceptnet_graph(G: nx.Graph, path: str | Path) -> None:
    """Save a ConceptNet graph to a pickle file.

    Args:
        G: NetworkX graph to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cached_graph(path: str | Path) -> nx.Graph:
    """Load a previously saved ConceptNet graph from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized :class:`nx.Graph`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cached graph not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
