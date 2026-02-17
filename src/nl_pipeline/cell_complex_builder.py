"""Builds CellComplex from GraphSpec using existing graph_convert infrastructure."""
import networkx as nx
from src.benchmarks.graph_convert import nx_to_cell_complex
from src.nl_pipeline.data_types import GraphSpec

_RELATION_ENCODING = {"causes": 0.33, "prevents": 0.66, "enables": 1.0, "connects": 0.0}


class CellComplexBuilder:
    """Converts a GraphSpec (string node names) into a CellComplex with structural embeddings."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def build(self, spec: GraphSpec):
        """Build a CellComplex from a GraphSpec.

        Returns:
            (cc, name_to_cc) where name_to_cc maps string node names to CC indices.
        """
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

        # Encode edge relations into emb[1]
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

        name_to_cc = {
            name: nx_to_cc[nx_id]
            for name, nx_id in name_to_nx.items()
            if nx_id in nx_to_cc
        }
        return cc, name_to_cc
