import torch
import networkx as nx
from src.benchmarks.graph_convert import nx_to_cell_complex
from src.cell_complex.cell_complex import CellComplex


class TestNxToCellComplex:
    def test_preserves_node_count(self):
        G = nx.path_graph(5)
        cc, node_map = nx_to_cell_complex(G, embedding_dim=16)
        assert cc.num_cells(0) == 5

    def test_preserves_edge_count(self):
        G = nx.path_graph(5)
        cc, node_map = nx_to_cell_complex(G, embedding_dim=16, fill_triangles=False)
        assert cc.num_cells(1) == 4  # path has n-1 edges

    def test_triangle_fills_2cells(self):
        G = nx.complete_graph(3)
        cc, node_map = nx_to_cell_complex(G, embedding_dim=16, fill_triangles=True)
        assert cc.num_cells(0) == 3
        assert cc.num_cells(1) == 3
        assert cc.num_cells(2) == 1  # one triangle

    def test_node_map_complete(self):
        G = nx.path_graph(10)
        cc, node_map = nx_to_cell_complex(G, embedding_dim=16)
        assert len(node_map) == 10
        for nx_node in G.nodes():
            assert nx_node in node_map
            assert 0 <= node_map[nx_node] < cc.num_cells(0)

    def test_source_target_types(self):
        G = nx.path_graph(5)
        cc, node_map = nx_to_cell_complex(
            G, embedding_dim=16, source_node=0, target_node=4)
        assert cc._0_cell_types[node_map[0]] == "source"
        assert cc._0_cell_types[node_map[4]] == "target"

    def test_blocked_node_types(self):
        G = nx.path_graph(5)
        cc, node_map = nx_to_cell_complex(
            G, embedding_dim=16, source_node=0, target_node=4,
            blocked_nodes=[2])
        assert cc._0_cell_types[node_map[2]] == "blocked"

    def test_source2_type(self):
        G = nx.path_graph(5)
        cc, node_map = nx_to_cell_complex(
            G, embedding_dim=16, source_node=0, target_node=4,
            source2_node=1)
        assert cc._0_cell_types[node_map[1]] == "source2"

    def test_edge_delays_propagate(self):
        G = nx.path_graph(3)
        delays = {(0, 1): 5, (1, 2): 3}
        cc, node_map = nx_to_cell_complex(
            G, embedding_dim=16, edge_delays=delays, fill_triangles=False)
        # Check delay is stored in emb[0]
        for e in range(cc.num_cells(1)):
            delay_val = int(cc._1_cell_embeddings[e][0].item())
            assert delay_val in (3, 5)

    def test_structural_features_deterministic(self):
        """Same graph structure produces same structural features (emb[0:3])."""
        G = nx.path_graph(5)
        cc1, _ = nx_to_cell_complex(G, embedding_dim=16, source_node=0, target_node=4)
        cc2, _ = nx_to_cell_complex(G, embedding_dim=16, source_node=0, target_node=4)
        # Structural features (first 3 dims) should match
        for i in range(5):
            e1 = cc1._0_cell_embeddings[i][:3]
            e2 = cc2._0_cell_embeddings[i][:3]
            assert torch.allclose(e1, e2)

    def test_structural_embedding_degree(self):
        """Hub node in a star graph should have highest normalized degree (1.0)."""
        G = nx.star_graph(5)  # node 0 is hub with degree 5
        cc, node_map = nx_to_cell_complex(G, embedding_dim=16)
        hub_emb = cc._0_cell_embeddings[node_map[0]]
        assert hub_emb[0].item() == 1.0  # normalized degree = 5/5

    def test_structural_embedding_bfs_distance(self):
        """Source node should have BFS distance 0."""
        G = nx.path_graph(5)
        cc, node_map = nx_to_cell_complex(
            G, embedding_dim=16, source_node=0, target_node=4)
        source_emb = cc._0_cell_embeddings[node_map[0]]
        assert source_emb[2].item() == 0.0  # BFS distance from source = 0

    def test_chain_complex_valid(self):
        """Generated CellComplex satisfies B1@B2=0."""
        G = nx.complete_graph(4)
        cc, _ = nx_to_cell_complex(G, embedding_dim=16, fill_triangles=True)
        if cc.num_cells(2) > 0:
            assert cc.verify_chain_complex()
