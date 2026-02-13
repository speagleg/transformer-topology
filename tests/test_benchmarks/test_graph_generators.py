import pytest
import networkx as nx
from src.benchmarks.graph_generators import (
    generate_ba_graph,
    generate_ws_graph,
    generate_sbm_graph,
    generate_grid_graph,
    generate_tree_graph,
    generate_ladder_graph,
    generate_caveman_graph,
    random_graph,
)


class TestIndividualGenerators:
    def test_ba_graph(self):
        G = generate_ba_graph(30, m=2)
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 30

    def test_ws_graph(self):
        G = generate_ws_graph(30, k=4, p=0.3)
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 30

    def test_sbm_graph(self):
        G = generate_sbm_graph([10, 10, 10], p_within=0.5, p_between=0.05)
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 30

    def test_grid_graph(self):
        G = generate_grid_graph(5, 6)
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 30

    def test_tree_graph(self):
        G = generate_tree_graph(30)
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 30
        # Tree has n-1 edges
        assert G.number_of_edges() == 29

    def test_ladder_graph(self):
        G = generate_ladder_graph(15)
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 30

    def test_caveman_graph(self):
        G = generate_caveman_graph(5, 6)
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 30


class TestRandomGraph:
    def test_all_topologies_produce_connected_graphs(self):
        for topo in ['ba', 'ws', 'sbm', 'grid', 'tree', 'ladder', 'caveman']:
            G = random_graph(30, topology=topo)
            assert nx.is_connected(G), f"Not connected for topology={topo}"
            assert G.number_of_nodes() >= 4, f"Too few nodes for topology={topo}"

    def test_random_topology_selection(self):
        """Dispatcher with topology=None picks randomly and produces connected graph."""
        for _ in range(20):
            G = random_graph(30)
            assert nx.is_connected(G)

    def test_small_graphs(self):
        """Small n_nodes doesn't crash."""
        for topo in ['ba', 'ws', 'tree', 'ladder']:
            G = random_graph(5, topology=topo)
            assert nx.is_connected(G)

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            random_graph(30, topology="nonexistent")

    def test_size_range(self):
        """Graph sizes are within reasonable bounds of requested size."""
        for _ in range(10):
            G = random_graph(50)
            # Some generators produce different sizes (grid, ladder, caveman)
            assert 4 <= G.number_of_nodes() <= 100
