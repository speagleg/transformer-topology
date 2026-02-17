"""Builds CellComplex from GraphSpec using existing graph_convert infrastructure."""
import networkx as nx
from src.benchmarks.graph_convert import nx_to_cell_complex
from src.benchmarks.graph_generators import random_graph
from src.nl_pipeline.data_types import GraphSpec

_RELATION_ENCODING = {"causes": 0.33, "prevents": 0.66, "enables": 1.0, "connects": 0.0}

# Priority for mapping semantic nodes to structural positions.
# Lower value = higher priority for high-degree positions.
_ROLE_PRIORITY = {
    "hub": 0.0, "root": 0.0,
    "bridge": 0.3, "gateway": 0.3,
    "internal": 0.5, "member": 0.5, "peer": 0.5,
    "cell": 0.5, "rung": 0.5, "clique_member": 0.5, "generic": 0.5,
    "leaf": 0.9, "spoke": 0.9, "endpoint": 0.9,
}

# Minimum graph size per topology for meaningful structure.
_MIN_TOPOLOGY_SIZE = {
    "ba": 8, "ws": 8, "sbm": 9, "grid": 9, "tree": 8,
    "ladder": 8, "caveman": 6, "er": 8,
}


class CellComplexBuilder:
    """Converts a GraphSpec (string node names) into a CellComplex with structural embeddings."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def build(self, spec: GraphSpec):
        """Build a CellComplex from a GraphSpec.

        When topology_hint is present with sufficient confidence, generates a
        topology-faithful graph and maps semantic nodes to structurally
        appropriate positions. Otherwise falls back to edge-based construction.

        Returns:
            (cc, name_to_cc) where name_to_cc maps string node names to CC indices.
        """
        if spec.topology_hint and spec.topology_confidence >= 0.3:
            return self._build_with_topology(spec)
        return self._build_from_edges(spec)

    def _build_with_topology(self, spec: GraphSpec):
        """Generate topology-faithful graph and map semantic nodes onto it."""
        n_semantic = len(spec.nodes)
        min_size = _MIN_TOPOLOGY_SIZE.get(spec.topology_hint, 8)
        n_graph = max(n_semantic, min_size)

        G = random_graph(n_graph, topology=spec.topology_hint)

        # Sort generated nodes by degree (descending) for role-based mapping
        degree_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)

        # Map semantic nodes to structural positions based on roles
        roles = spec.node_roles or {}
        name_to_nx = {}
        used_positions = set()

        # Sort semantic nodes by role priority (hubs first, leaves last)
        nodes_with_priority = []
        for node in spec.nodes:
            role = roles.get(node.name, "generic")
            priority = _ROLE_PRIORITY.get(role, 0.5)
            nodes_with_priority.append((node, priority))
        nodes_with_priority.sort(key=lambda x: x[1])

        for node, priority in nodes_with_priority:
            target_idx = int(priority * (len(degree_sorted) - 1))
            # Search outward from target position for available slot
            assigned = False
            for offset in range(len(degree_sorted)):
                for candidate_idx in [target_idx + offset, target_idx - offset]:
                    if 0 <= candidate_idx < len(degree_sorted):
                        pos = degree_sorted[candidate_idx]
                        if pos not in used_positions:
                            name_to_nx[node.name] = pos
                            used_positions.add(pos)
                            assigned = True
                            break
                if assigned:
                    break

        source_nx = name_to_nx.get(spec.query_node)
        target_nx = name_to_nx.get(spec.target_node) if spec.target_node else None

        cc, nx_to_cc = nx_to_cell_complex(
            G, self.embedding_dim,
            source_node=source_nx, target_node=target_nx,
            fill_triangles=True,
        )

        # Encode edge relations where structurally adjacent
        edge_relations = {}
        for edge in spec.edges:
            u = name_to_nx.get(edge.source)
            v = name_to_nx.get(edge.target)
            if u is not None and v is not None and G.has_edge(u, v):
                edge_relations[(u, v)] = edge.relation

        self._encode_edge_relations(cc, nx_to_cc, edge_relations)

        name_to_cc = {
            name: nx_to_cc[nx_id]
            for name, nx_id in name_to_nx.items()
            if nx_id in nx_to_cc
        }
        return cc, name_to_cc

    def _build_from_edges(self, spec: GraphSpec):
        """Build from parsed edges (original behavior)."""
        G = nx.Graph()
        name_to_nx = {}
        for i, node in enumerate(spec.nodes):
            G.add_node(i)
            name_to_nx[node.name] = i

        edge_relations = {}
        for edge in spec.edges:
            u = name_to_nx.get(edge.source)
            v = name_to_nx.get(edge.target)
            if u is not None and v is not None and u != v:
                G.add_edge(u, v)
                edge_relations[(u, v)] = edge.relation

        source_nx = name_to_nx.get(spec.query_node)
        target_nx = name_to_nx.get(spec.target_node) if spec.target_node else None

        cc, nx_to_cc = nx_to_cell_complex(
            G, self.embedding_dim,
            source_node=source_nx, target_node=target_nx,
            fill_triangles=True,
        )

        self._encode_edge_relations(cc, nx_to_cc, edge_relations)

        name_to_cc = {
            name: nx_to_cc[nx_id]
            for name, nx_id in name_to_nx.items()
            if nx_id in nx_to_cc
        }
        return cc, name_to_cc

    @staticmethod
    def _encode_edge_relations(cc, nx_to_cc, edge_relations):
        """Encode edge relation types into edge embedding dimension 1."""
        if not edge_relations:
            return
        edge_embs = cc.get_embeddings(1)
        for edge_idx in range(cc.num_cells(1)):
            src_cc = cc._1_cell_sources[edge_idx]
            tgt_cc = cc._1_cell_targets[edge_idx]
            for (u, v), rel in edge_relations.items():
                if (nx_to_cc.get(u) == src_cc and nx_to_cc.get(v) == tgt_cc) or \
                   (nx_to_cc.get(v) == src_cc and nx_to_cc.get(u) == tgt_cc):
                    edge_embs[edge_idx, 1] = _RELATION_ENCODING.get(rel, 0.0)
                    break
        cc.set_embeddings(1, edge_embs)
