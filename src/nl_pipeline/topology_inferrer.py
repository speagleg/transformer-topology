"""Infer graph topology from NL query text and parsed GraphSpec."""
from __future__ import annotations

import re

from src.nl_pipeline.data_types import GraphSpec, TopologyHint


# (keywords, topology, default_role_for_unmatched_nodes)
TOPOLOGY_PATTERNS: list[tuple[list[str], str, str]] = [
    # ba — hub/star/scale-free
    (
        [
            "central hub", "hub and spoke", "star network", "load balancer",
            "main server", "hub", "central", "star", "broadcast", "core",
            "coordinator", "backbone",
        ],
        "ba",
        "spoke",
    ),
    # tree — hierarchy
    (
        [
            "org chart", "inheritance tree", "parent child", "decision tree",
            "hierarchy", "hierarchical", "tree", "parent", "child", "levels",
            "subordinate", "root", "branch", "ancestor", "descendant",
            "top-down", "bottom-up", "manager",
        ],
        "tree",
        "internal",
    ),
    # sbm — community/partition
    (
        [
            "community structure", "department", "partition", "cluster",
            "community", "group", "faction", "team", "silo", "segregated",
            "module", "subgroup",
        ],
        "sbm",
        "member",
    ),
    # ws — ring/cycle/small-world
    (
        [
            "round-robin", "feedback loop", "circular dependency",
            "cycle", "loop", "circular", "feedback", "recurring",
            "oscillate", "ring", "periodic",
        ],
        "ws",
        "peer",
    ),
    # grid
    (
        [
            "grid layout", "spreadsheet", "pixel grid",
            "grid", "matrix", "lattice", "row", "column", "coordinate",
            "pixel", "table",
        ],
        "grid",
        "cell",
    ),
    # ladder — parallel/dual paths
    (
        [
            "parallel paths", "dual path", "redundant path", "backup route",
            "ladder", "parallel", "redundant", "dual",
        ],
        "ladder",
        "rung",
    ),
    # caveman — clique/dense
    (
        [
            "fully connected", "complete graph", "tight-knit",
            "clique", "tribe", "dense", "complete", "all-to-all",
        ],
        "caveman",
        "member",
    ),
    # er — random
    (
        [
            "random network", "random graph",
            "random", "arbitrary", "mesh",
        ],
        "er",
        "generic",
    ),
]


# node-name keyword -> structural role
ROLE_KEYWORDS: dict[str, str] = {
    "hub": "hub",
    "central": "hub",
    "core": "hub",
    "server": "hub",
    "coordinator": "hub",
    "backbone": "hub",
    "root": "root",
    "parent": "root",
    "manager": "root",
    "boss": "root",
    "child": "leaf",
    "leaf": "leaf",
    "endpoint": "leaf",
    "client": "leaf",
    "spoke": "leaf",
    "subordinate": "leaf",
    "worker": "leaf",
    "bridge": "bridge",
    "gateway": "bridge",
    "connector": "bridge",
    "member": "member",
    "peer": "peer",
    "node": "generic",
}


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Check if *keyword* appears in *text* as a whole-word or word-prefix match.

    Multi-word phrases use simple substring matching (false positives are
    negligible).  Single words require a leading word boundary so that e.g.
    "row" does not match inside "brown", but "cluster" still matches
    "clusters" (common English plural/inflection).
    """
    if " " in keyword:
        return keyword in text
    # \b before keyword ensures we start at a word boundary.
    # After the keyword we allow optional common suffixes (s, es, ed, ing, ...)
    # by simply not requiring a trailing \b -- instead we require the keyword
    # to be preceded by a word boundary and to not be preceded by more word chars.
    return bool(re.search(r"\b" + re.escape(keyword) + r"(?:s|es|ed|ing|er)?\b", text))


class TopologyInferrer:
    """Infer the most likely graph topology from query text and node names."""

    def infer(self, spec: GraphSpec, query: str) -> TopologyHint:
        """Score each topology against *query* and node names, return best match."""
        query_lower = query.lower()

        # --- score each topology group ---
        scores: dict[str, float] = {}
        default_roles: dict[str, str] = {}

        for keywords, topology, default_role in TOPOLOGY_PATTERNS:
            score = 0.0
            for kw in keywords:
                if _keyword_in_text(kw, query_lower):
                    weight = 2.0 if " " in kw else 1.0
                    score += weight
            # also check node names (weight 0.5)
            for node in spec.nodes:
                node_lower = node.name.lower()
                for kw in keywords:
                    if _keyword_in_text(kw, node_lower):
                        score += 0.5
            if score > 0:
                scores[topology] = score
                default_roles[topology] = default_role

        # --- no match ---
        if not scores:
            return TopologyHint(topology=None, confidence=0.0)

        # --- pick best topology ---
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_topo, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        default_role = default_roles[best_topo]

        # --- confidence from margin + minimum thresholds ---
        margin = (best_score - second_score) / best_score if best_score > 0 else 0.0
        confidence = margin

        if best_score >= 3.0:
            confidence = max(confidence, 0.7)
        elif best_score >= 2.0:
            confidence = max(confidence, 0.5)
        elif best_score >= 1.0:
            confidence = max(confidence, 0.3)

        confidence = min(confidence, 1.0)

        # --- assign node roles ---
        node_roles: dict[str, str] = {}
        for node in spec.nodes:
            role = self._role_for_name(node.name)
            node_roles[node.name] = role if role else default_role

        return TopologyHint(
            topology=best_topo,
            confidence=confidence,
            node_roles=node_roles,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _role_for_name(name: str) -> str | None:
        """Return a structural role if *name* matches a role keyword."""
        name_lower = name.lower()
        for keyword, role in ROLE_KEYWORDS.items():
            if keyword in name_lower:
                return role
        return None
