"""ConceptNet knowledge graph reasoning tasks.

Five generators that accept a NetworkX ConceptNet (sub)graph and produce
5-tuples ``(cc, query_node, target_node, answer, metadata)`` where the GNN
sees graph structure and the LLM reads concept text via ``cc.node_texts``.

Tasks
-----
1. **kg_relation** -- Predict the relation category of a random edge (10 cls).
2. **kg_concept** -- Predict the semantic category of a masked node (9 cls).
3. **kg_pathvalid** -- Decide if a multi-hop path is valid or corrupted (2 cls).
4. **kg_analogy** -- Decide if two subgraphs are structurally analogous (3 cls).
5. **kg_cluster** -- Predict the semantic domain of a random node (6 cls).
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import torch

from src.data.conceptnet import (
    extract_subgraph,
    conceptnet_subgraph_to_cc,
    concept_to_text,
    categorize_relation,
    RELATION_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELATION_CLASSES: list[str] = sorted(
    {
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
    }
)
assert len(RELATION_CLASSES) == 10

CONCEPT_CATEGORIES: list[str] = [
    "animal",
    "place",
    "activity",
    "object",
    "emotion",
    "person",
    "food",
    "body_part",
    "abstract",
]
assert len(CONCEPT_CATEGORIES) == 9

DOMAIN_CLASSES: list[str] = [
    "science",
    "everyday",
    "social",
    "spatial",
    "temporal",
    "abstract",
]
assert len(DOMAIN_CLASSES) == 6

# ---------------------------------------------------------------------------
# Keyword-based concept classifier  (task 2)
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "animal": [
        "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep",
        "chicken", "duck", "mouse", "rat", "bear", "lion", "tiger",
        "snake", "frog", "whale", "shark", "insect",
    ],
    "place": [
        "city", "town", "country", "park", "house", "building", "school",
        "hospital", "church", "store", "restaurant", "hotel", "beach",
        "mountain", "forest", "river", "lake", "ocean", "street", "room",
    ],
    "activity": [
        "run", "walk", "swim", "play", "dance", "sing", "cook", "drive",
        "read", "write", "climb", "jump", "throw", "catch", "fight",
        "travel", "exercise", "paint", "draw", "sleep",
    ],
    "object": [
        "table", "chair", "car", "phone", "computer", "book", "pen",
        "door", "window", "cup", "plate", "knife", "bottle", "key",
        "clock", "lamp", "bag", "box", "tool", "wheel",
    ],
    "emotion": [
        "love", "hate", "fear", "anger", "joy", "sadness", "happiness",
        "surprise", "disgust", "anxiety", "pride", "shame", "guilt",
        "trust", "loyalty", "envy", "jealousy", "hope", "grief",
        "excitement",
    ],
    "person": [
        "man", "woman", "child", "baby", "friend", "teacher", "doctor",
        "father", "mother", "brother", "sister", "king", "queen",
        "soldier", "worker", "student", "artist", "leader", "hero",
        "stranger",
    ],
    "food": [
        "bread", "meat", "fruit", "vegetable", "rice", "cake", "cheese",
        "milk", "egg", "soup", "salad", "pizza", "pasta", "candy",
        "chocolate", "sugar", "salt", "butter", "apple", "banana",
    ],
    "body_part": [
        "head", "hand", "foot", "eye", "ear", "nose", "mouth", "arm",
        "leg", "finger", "toe", "heart", "brain", "bone", "skin",
        "hair", "tooth", "tongue", "knee", "shoulder",
    ],
    "abstract": [
        "time", "space", "idea", "thought", "mind", "soul", "truth",
        "knowledge", "science", "math", "philosophy", "freedom",
        "justice", "power", "energy", "force", "law", "theory",
        "concept", "reason",
    ],
}


def classify_concept(concept: str) -> str:
    """Classify a concept string into one of 9 categories using keywords.

    Args:
        concept: A concept slug like ``"dog"`` or ``"hot_dog"``.

    Returns:
        Category string from :data:`CONCEPT_CATEGORIES`.
    """
    text = concept_to_text(concept).lower()
    tokens = set(text.split())

    best_cat = "abstract"  # default fallback
    best_score = 0

    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in tokens or kw in text:
                score += 1
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat


# ---------------------------------------------------------------------------
# Keyword-based domain classifier  (task 5)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "science": [
        "atom", "molecule", "electron", "proton", "neutron", "cell",
        "gene", "protein", "enzyme", "photosynthesis", "gravity",
        "energy", "radiation", "spectrum", "experiment",
    ],
    "everyday": [
        "house", "car", "food", "cook", "clean", "shop", "work",
        "sleep", "eat", "drink", "walk", "phone", "clothes", "money",
        "home",
    ],
    "social": [
        "friend", "family", "love", "trust", "help", "community",
        "leader", "team", "party", "wedding", "meeting", "argue",
        "agree", "share", "teach",
    ],
    "spatial": [
        "city", "park", "mountain", "river", "ocean", "forest",
        "street", "room", "building", "bridge", "road", "map",
        "north", "south", "location",
    ],
    "temporal": [
        "morning", "night", "year", "season", "winter", "summer",
        "spring", "autumn", "hour", "minute", "yesterday", "tomorrow",
        "clock", "calendar", "history",
    ],
    "abstract": [
        "idea", "thought", "concept", "theory", "truth", "freedom",
        "justice", "knowledge", "mind", "reason", "logic", "philosophy",
        "meaning", "purpose", "belief",
    ],
}


def classify_domain(concept: str) -> str:
    """Classify a concept string into one of 6 semantic domains using keywords.

    Args:
        concept: A concept slug like ``"atom"`` or ``"hot_dog"``.

    Returns:
        Domain string from :data:`DOMAIN_CLASSES`.
    """
    text = concept_to_text(concept).lower()
    tokens = set(text.split())

    best_domain = "abstract"  # default fallback
    best_score = 0

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in tokens or kw in text:
                score += 1
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES = 10


def _safe_extract(
    G: nx.Graph,
    min_nodes: int,
    max_nodes: int,
) -> tuple[nx.Graph, list[str]]:
    """Try to extract a valid subgraph, retrying on failure."""
    last_err: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            sub, node_list = extract_subgraph(
                G, seed=None, min_nodes=min_nodes, max_nodes=max_nodes,
            )
            if sub.number_of_nodes() >= min_nodes:
                return sub, node_list
        except (ValueError, IndexError) as exc:
            last_err = exc
            continue

    # Final fallback: just use whatever the graph gives us
    try:
        sub, node_list = extract_subgraph(
            G, seed=None, min_nodes=1, max_nodes=max_nodes,
        )
        return sub, node_list
    except (ValueError, IndexError):
        pass

    if last_err:
        raise last_err
    raise ValueError(f"Cannot extract subgraph from graph with {G.number_of_nodes()} nodes")


def _build_node_texts(sub: nx.Graph, node_list: list[str]) -> dict[str, str]:
    """Map each concept in *node_list* to readable text."""
    return {c: concept_to_text(c) for c in node_list if c in sub}


# ---------------------------------------------------------------------------
# Task 1: Relation classification
# ---------------------------------------------------------------------------


def generate_kg_relation_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Predict the relation category of a random edge.

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is an index into :data:`RELATION_CLASSES` (0--9).
    """
    sub, node_list = _safe_extract(G, min_nodes, max_nodes)

    # Pick a random edge
    edges = list(sub.edges(data=True))
    if not edges:
        raise ValueError("Subgraph has no edges")
    u, v, data = random.choice(edges)

    raw_rel = data.get("relation", data.get("raw_relation", "RelatedTo"))
    category = categorize_relation(raw_rel)
    answer = RELATION_CLASSES.index(category) if category in RELATION_CLASSES else RELATION_CLASSES.index("Other")

    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
    node_texts = _build_node_texts(sub, node_list)

    query_idx = node_map[u]
    target_idx = node_map[v]

    metadata = {
        "task_type": "kg_relation",
        "node_texts": node_texts,
        "task_prompt": (
            f"Predict the relation between '{concept_to_text(u)}' and "
            f"'{concept_to_text(v)}' | "
            f"nodes={sub.number_of_nodes()} edges={sub.number_of_edges()} | "
            f"task=kg_relation"
        ),
        "relation": category,
        "num_classes": len(RELATION_CLASSES),
    }

    return cc, query_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# Task 2: Concept category classification (masked)
# ---------------------------------------------------------------------------


def generate_kg_concept_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Predict the category of a masked concept node.

    One node's text is replaced with ``"[MASK]"`` in ``cc.node_texts``.
    The model must infer the semantic category from the graph neighbourhood.

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is an index into :data:`CONCEPT_CATEGORIES` (0--8).
        *target_node_idx* is set to a random neighbor of the masked node.
    """
    sub, node_list = _safe_extract(G, min_nodes, max_nodes)

    # Pick a node to mask
    masked_concept = random.choice(node_list)
    category = classify_concept(masked_concept)
    answer = CONCEPT_CATEGORIES.index(category)

    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)

    # Replace the masked node's text in cc.node_texts
    masked_idx = node_map[masked_concept]
    cc.node_texts[masked_idx] = "[MASK]"

    # Pick a neighbor as target node (gives the model spatial context)
    neighbors = list(sub.neighbors(masked_concept))
    if neighbors:
        target_concept = random.choice(neighbors)
    else:
        # Pick any other node
        others = [c for c in node_list if c != masked_concept and c in node_map]
        target_concept = random.choice(others) if others else masked_concept

    target_idx = node_map[target_concept]

    node_texts = _build_node_texts(sub, node_list)
    node_texts[masked_concept] = "[MASK]"

    metadata = {
        "task_type": "kg_concept",
        "node_texts": node_texts,
        "task_prompt": (
            f"Predict the category of the masked concept at node {masked_idx} | "
            f"neighbors={len(neighbors)} | "
            f"nodes={sub.number_of_nodes()} edges={sub.number_of_edges()} | "
            f"task=kg_concept"
        ),
        "masked_concept": masked_concept,
        "true_category": category,
        "num_classes": len(CONCEPT_CATEGORIES),
    }

    return cc, masked_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# Task 3: Path validity
# ---------------------------------------------------------------------------


def generate_kg_pathvalid_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Decide if a multi-hop path in the knowledge graph is valid or corrupted.

    Finds a shortest path of 3--5 hops. With 50 % probability the path is
    kept valid (answer = 1); otherwise the middle node is swapped with a
    random same-degree node (answer = 0).

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is 0 (corrupted) or 1 (valid).
    """
    sub, node_list = _safe_extract(G, min_nodes, max_nodes)
    nodes = list(sub.nodes())

    # Find a path of 3-5 hops
    path = None
    for _ in range(50):
        src, tgt = random.sample(nodes, 2)
        try:
            candidate = nx.shortest_path(sub, src, tgt)
        except nx.NetworkXNoPath:
            continue
        if 3 <= len(candidate) - 1 <= 5:
            path = candidate
            break

    if path is None:
        # Fallback: take any shortest path >= 2 hops
        for _ in range(50):
            src, tgt = random.sample(nodes, 2)
            try:
                candidate = nx.shortest_path(sub, src, tgt)
            except nx.NetworkXNoPath:
                continue
            if len(candidate) >= 3:
                path = candidate
                break

    if path is None:
        # Ultimate fallback: pick two connected nodes + a neighbor
        src = random.choice(nodes)
        neighbors = list(sub.neighbors(src))
        if neighbors:
            tgt = random.choice(neighbors)
            path = [src, tgt]
        else:
            path = [src, nodes[(nodes.index(src) + 1) % len(nodes)]]

    src_concept = path[0]
    tgt_concept = path[-1]

    # 50% corrupt, 50% valid
    if random.random() < 0.5 and len(path) >= 3:
        # Corrupt: swap a middle node with a random node of similar degree
        mid_idx = len(path) // 2
        mid_node = path[mid_idx]
        mid_degree = sub.degree(mid_node)

        # Find candidates with similar degree (within +/- 2)
        candidates = [
            n for n in nodes
            if n not in path and abs(sub.degree(n) - mid_degree) <= 2
        ]
        if not candidates:
            candidates = [n for n in nodes if n not in path]

        if candidates:
            swap_node = random.choice(candidates)
            path[mid_idx] = swap_node
            answer = 0
        else:
            answer = 1  # can't corrupt, keep valid
    else:
        answer = 1

    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
    node_texts = _build_node_texts(sub, node_list)

    query_idx = node_map[src_concept]
    target_idx = node_map[tgt_concept]

    path_text = " -> ".join(concept_to_text(c) for c in path)

    metadata = {
        "task_type": "kg_pathvalid",
        "node_texts": node_texts,
        "task_prompt": (
            f"Is the path valid? {path_text} | "
            f"hops={len(path) - 1} | "
            f"nodes={sub.number_of_nodes()} edges={sub.number_of_edges()} | "
            f"task=kg_pathvalid"
        ),
        "path": [concept_to_text(c) for c in path],
        "path_length": len(path) - 1,
        "num_classes": 2,
    }

    return cc, query_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# Task 4: Structural analogy
# ---------------------------------------------------------------------------


def generate_kg_analogy_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 10,
    max_nodes: int = 30,
) -> tuple:
    """Decide if two knowledge subgraphs are structurally analogous.

    Extracts two subgraphs from different seed nodes, computes the
    relation-type overlap (Jaccard), and classifies:
      - overlap > 0.6 -> analogous (2)
      - overlap > 0.3 -> partial (1)
      - else           -> not analogous (0)

    Both subgraphs are merged into one CellComplex with prefixed node names.

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum per-subgraph size.
        max_nodes: Maximum per-subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is 0 (not analogous), 1 (partial), or 2 (analogous).
    """
    from src.cell_complex.cell_complex import CellComplex as CC
    from src.benchmarks.topological_tasks import auto_fill_triangles

    # Extract two subgraphs from different seeds
    sub_a, nlist_a = _safe_extract(G, min_nodes, max_nodes)
    sub_b, nlist_b = _safe_extract(G, min_nodes, max_nodes)

    # Compute relation overlap (Jaccard on the multiset of relation types)
    rels_a = set(
        data.get("relation", "Other") for _, _, data in sub_a.edges(data=True)
    )
    rels_b = set(
        data.get("relation", "Other") for _, _, data in sub_b.edges(data=True)
    )

    if rels_a or rels_b:
        overlap = len(rels_a & rels_b) / len(rels_a | rels_b)
    else:
        overlap = 0.0

    if overlap > 0.6:
        answer = 2  # analogous
    elif overlap > 0.3:
        answer = 1  # partial
    else:
        answer = 0  # not analogous

    # Build a merged graph with prefixed node names
    merged = nx.Graph()

    # Precompute structural features for sub_a
    degrees_a = dict(sub_a.degree())
    max_deg_a = max(degrees_a.values()) if degrees_a else 1
    max_deg_a = max(max_deg_a, 1)
    clust_a = nx.clustering(sub_a)

    # Precompute structural features for sub_b
    degrees_b = dict(sub_b.degree())
    max_deg_b = max(degrees_b.values()) if degrees_b else 1
    max_deg_b = max(max_deg_b, 1)
    clust_b = nx.clustering(sub_b)

    for node in sub_a.nodes():
        merged.add_node(f"a_{node}", prefix="a", original=node)
    for u, v, data in sub_a.edges(data=True):
        merged.add_edge(f"a_{u}", f"a_{v}", **data)

    for node in sub_b.nodes():
        merged.add_node(f"b_{node}", prefix="b", original=node)
    for u, v, data in sub_b.edges(data=True):
        merged.add_edge(f"b_{u}", f"b_{v}", **data)

    # Build CellComplex from the merged graph
    cc = CC(embedding_dim=embedding_dim)
    node_map: dict[str, int] = {}
    merged_nodes = list(merged.nodes())
    merged_degrees = dict(merged.degree())
    merged_max_deg = max(merged_degrees.values()) if merged_degrees else 1
    merged_max_deg = max(merged_max_deg, 1)
    merged_clust = nx.clustering(merged)

    for mnode in merged_nodes:
        emb = torch.randn(embedding_dim) * 0.01
        emb[0] = merged_degrees[mnode] / merged_max_deg
        emb[1] = merged_clust[mnode]
        emb[2] = 0.0  # no BFS for merged graph

        cc_idx = cc.add_0_cell(emb, "node")
        node_map[mnode] = cc_idx

    # Populate node_texts using the original concept names
    cc.node_texts = []
    for mnode in merged_nodes:
        original = merged.nodes[mnode].get("original", mnode)
        prefix = merged.nodes[mnode].get("prefix", "")
        cc.node_texts.append(f"{prefix}: {concept_to_text(original)}")

    # Add edges
    for u, v, data in merged.edges(data=True):
        emb = torch.randn(embedding_dim) * 0.01
        weight = data.get("weight", 1.0)
        emb[0] = weight / 10.0
        cc.add_1_cell(node_map[u], node_map[v], emb, "edge")

    auto_fill_triangles(cc)

    # Pick representative nodes from each subgraph
    query_node = f"a_{nlist_a[0]}"
    target_node = f"b_{nlist_b[0]}"
    query_idx = node_map[query_node]
    target_idx = node_map[target_node]

    node_texts_a = {c: concept_to_text(c) for c in nlist_a if c in sub_a}
    node_texts_b = {c: concept_to_text(c) for c in nlist_b if c in sub_b}
    combined_texts = {}
    for k, v_ in node_texts_a.items():
        combined_texts[f"a_{k}"] = v_
    for k, v_ in node_texts_b.items():
        combined_texts[f"b_{k}"] = v_

    metadata = {
        "task_type": "kg_analogy",
        "node_texts": combined_texts,
        "task_prompt": (
            f"Are these subgraphs analogous? overlap={overlap:.3f} | "
            f"rels_a={sorted(rels_a)} rels_b={sorted(rels_b)} | "
            f"nodes_a={sub_a.number_of_nodes()} nodes_b={sub_b.number_of_nodes()} | "
            f"task=kg_analogy"
        ),
        "overlap": overlap,
        "rels_a": sorted(rels_a),
        "rels_b": sorted(rels_b),
        "num_classes": 3,
    }

    return cc, query_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# Task 5: Semantic domain clustering
# ---------------------------------------------------------------------------


def generate_kg_cluster_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Predict the semantic domain of a random node.

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is an index into :data:`DOMAIN_CLASSES` (0--5).
    """
    sub, node_list = _safe_extract(G, min_nodes, max_nodes)

    # Pick a random node
    query_concept = random.choice(node_list)
    domain = classify_domain(query_concept)
    answer = DOMAIN_CLASSES.index(domain)

    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
    node_texts = _build_node_texts(sub, node_list)

    query_idx = node_map[query_concept]

    # Pick a neighbor as target for spatial context
    neighbors = list(sub.neighbors(query_concept))
    if neighbors:
        target_concept = random.choice(neighbors)
    else:
        others = [c for c in node_list if c != query_concept and c in node_map]
        target_concept = random.choice(others) if others else query_concept
    target_idx = node_map[target_concept]

    metadata = {
        "task_type": "kg_cluster",
        "node_texts": node_texts,
        "task_prompt": (
            f"Predict the semantic domain of '{concept_to_text(query_concept)}' | "
            f"neighbors={len(neighbors)} | "
            f"nodes={sub.number_of_nodes()} edges={sub.number_of_edges()} | "
            f"task=kg_cluster"
        ),
        "query_concept": query_concept,
        "true_domain": domain,
        "num_classes": len(DOMAIN_CLASSES),
    }

    return cc, query_idx, target_idx, answer, metadata
