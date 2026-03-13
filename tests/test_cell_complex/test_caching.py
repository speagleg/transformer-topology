"""Tests for CellComplex caching infrastructure (Task 1: GPU optimization).

Tests verify that boundary/adjacency/edge_index results are cached and
reused, that topology mutations invalidate caches, that set_embeddings
does NOT invalidate structural caches, and that spectral_decomposition
is also cached through the CC._spectral_cache dict.
"""
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition


def _make_triangle_cc(embedding_dim: int = 4) -> CellComplex:
    """Return a CC with 3 nodes and 3 edges forming a triangle."""
    cc = CellComplex(embedding_dim=embedding_dim)
    a = cc.add_0_cell(torch.randn(embedding_dim), "concept")
    b = cc.add_0_cell(torch.randn(embedding_dim), "concept")
    c = cc.add_0_cell(torch.randn(embedding_dim), "concept")
    cc.add_1_cell(a, b, torch.randn(embedding_dim), "r")
    cc.add_1_cell(b, c, torch.randn(embedding_dim), "r")
    cc.add_1_cell(a, c, torch.randn(embedding_dim), "r")
    return cc


class TestBoundaryOperatorCaching:
    def test_boundary_operator_returns_same_object_on_second_call(self):
        cc = _make_triangle_cc()
        B1_first = cc.boundary_operator(1)
        B1_second = cc.boundary_operator(1)
        assert B1_first is B1_second, (
            "Second call to boundary_operator(1) should return the cached tensor"
        )

    def test_boundary_operator_cache_invalidated_by_add_1_cell(self):
        cc = _make_triangle_cc()
        B1_before = cc.boundary_operator(1)
        # Add a new node and a new edge — topology changes
        d = cc.add_0_cell(torch.randn(4), "concept")
        cc.add_1_cell(0, d, torch.randn(4), "r")
        B1_after = cc.boundary_operator(1)
        assert B1_before is not B1_after, (
            "boundary_operator cache must be invalidated after add_1_cell"
        )

    def test_boundary_operator_cache_invalidated_by_add_0_cell(self):
        cc = _make_triangle_cc()
        B1_before = cc.boundary_operator(1)
        cc.add_0_cell(torch.randn(4), "concept")
        # The cached value should be stale; a new object should be produced
        B1_after = cc.boundary_operator(1)
        assert B1_before is not B1_after, (
            "boundary_operator cache must be invalidated after add_0_cell"
        )

    def test_boundary_operator_dim2_cached(self):
        cc = _make_triangle_cc()
        # Add a 2-cell
        cc.add_2_cell([0, 1, 2], torch.randn(4), "face")
        B2_first = cc.boundary_operator(2)
        B2_second = cc.boundary_operator(2)
        assert B2_first is B2_second

    def test_boundary_operator_cache_invalidated_by_add_2_cell(self):
        cc = _make_triangle_cc()
        B2_before = cc.boundary_operator(2)
        cc.add_2_cell([0, 1, 2], torch.randn(4), "face")
        B2_after = cc.boundary_operator(2)
        assert B2_before is not B2_after


class TestSetEmbeddingsDoesNotInvalidateCache:
    def test_set_embeddings_does_not_invalidate_boundary_cache(self):
        cc = _make_triangle_cc()
        B1_before = cc.boundary_operator(1)
        # Replace embeddings — topology (connectivity) is unchanged
        new_embs = torch.randn(cc.num_cells(0), 4)
        cc.set_embeddings(0, new_embs)
        B1_after = cc.boundary_operator(1)
        assert B1_before is B1_after, (
            "set_embeddings should NOT invalidate boundary/adjacency caches "
            "because topology (connectivity) is unchanged"
        )

    def test_set_embeddings_does_not_invalidate_adjacency_cache(self):
        cc = _make_triangle_cc()
        A_before = cc.adjacency_matrix(0)
        cc.set_embeddings(1, torch.randn(cc.num_cells(1), 4))
        A_after = cc.adjacency_matrix(0)
        assert A_before is A_after


class TestAdjacencyMatrixCaching:
    def test_adjacency_matrix_cached(self):
        cc = _make_triangle_cc()
        A1 = cc.adjacency_matrix(0)
        A2 = cc.adjacency_matrix(0)
        assert A1 is A2

    def test_adjacency_matrix_dim1_cached(self):
        cc = _make_triangle_cc()
        A1 = cc.adjacency_matrix(1)
        A2 = cc.adjacency_matrix(1)
        assert A1 is A2

    def test_adjacency_matrix_invalidated_by_topology_change(self):
        cc = _make_triangle_cc()
        A_before = cc.adjacency_matrix(0)
        d = cc.add_0_cell(torch.randn(4), "concept")
        cc.add_1_cell(0, d, torch.randn(4), "r")
        A_after = cc.adjacency_matrix(0)
        assert A_before is not A_after


class TestEdgeIndexCaching:
    def test_edge_index_cached(self):
        cc = _make_triangle_cc()
        ei1 = cc.edge_index()
        ei2 = cc.edge_index()
        assert ei1 is ei2

    def test_edge_index_invalidated_by_topology_change(self):
        cc = _make_triangle_cc()
        ei_before = cc.edge_index()
        d = cc.add_0_cell(torch.randn(4), "concept")
        cc.add_1_cell(0, d, torch.randn(4), "r")
        ei_after = cc.edge_index()
        assert ei_before is not ei_after

    def test_edge_index_not_invalidated_by_set_embeddings(self):
        cc = _make_triangle_cc()
        ei_before = cc.edge_index()
        cc.set_embeddings(0, torch.randn(cc.num_cells(0), 4))
        ei_after = cc.edge_index()
        assert ei_before is ei_after


class TestCloneHasFreshCache:
    def test_clone_has_empty_cache(self):
        cc = _make_triangle_cc()
        # Warm the cache
        cc.boundary_operator(1)
        cc.adjacency_matrix(0)
        cc.edge_index()

        clone = cc.clone()
        # The clone's caches should be empty, so calling these returns new objects
        B1_clone = clone.boundary_operator(1)
        B1_original = cc.boundary_operator(1)
        assert B1_clone is not B1_original, (
            "clone() should start with fresh (empty) caches"
        )

    def test_clone_cache_independent_of_original(self):
        cc = _make_triangle_cc()
        clone = cc.clone()

        # Prime original cache
        B1_orig = cc.boundary_operator(1)
        # Prime clone cache
        B1_clone = clone.boundary_operator(1)

        # Invalidate clone's cache by adding a cell
        d = clone.add_0_cell(torch.randn(4), "concept")
        clone.add_1_cell(0, d, torch.randn(4), "r")

        # Original cache must be unaffected
        B1_orig_after = cc.boundary_operator(1)
        assert B1_orig is B1_orig_after, (
            "Mutating the clone must not affect the original's cache"
        )


class TestToDeviceClearsCache:
    def test_to_device_clears_boundary_cache(self):
        cc = _make_triangle_cc()
        B1_before = cc.boundary_operator(1)
        # Moving to same device (cpu) still clears cache
        cc.to("cpu")
        B1_after = cc.boundary_operator(1)
        assert B1_before is not B1_after, (
            "to(device) must clear caches so cached tensors are not stale"
        )


class TestSpectralDecompositionCaching:
    def test_spectral_decomposition_cached_on_second_call(self):
        cc = _make_triangle_cc()
        eigenvalues1, eigenvectors1 = spectral_decomposition(cc, dim=0)
        eigenvalues2, eigenvectors2 = spectral_decomposition(cc, dim=0)
        assert eigenvalues1 is eigenvalues2, (
            "spectral_decomposition should return cached eigenvalues on second call"
        )
        assert eigenvectors1 is eigenvectors2

    def test_spectral_cache_keyed_by_dim_k_normalize(self):
        cc = _make_triangle_cc()
        ev1, _ = spectral_decomposition(cc, dim=0, k=None, normalize=False)
        ev2, _ = spectral_decomposition(cc, dim=0, k=2, normalize=False)
        # Different k — different cache entries, different objects
        assert ev1 is not ev2

    def test_spectral_cache_invalidated_by_topology_change(self):
        cc = _make_triangle_cc()
        ev_before, _ = spectral_decomposition(cc, dim=0)
        # Change topology
        d = cc.add_0_cell(torch.randn(4), "concept")
        cc.add_1_cell(0, d, torch.randn(4), "r")
        ev_after, _ = spectral_decomposition(cc, dim=0)
        assert ev_before is not ev_after, (
            "spectral cache must be invalidated after topology change"
        )

    def test_spectral_cache_not_invalidated_by_set_embeddings(self):
        cc = _make_triangle_cc()
        ev_before, _ = spectral_decomposition(cc, dim=0)
        cc.set_embeddings(0, torch.randn(cc.num_cells(0), 4))
        ev_after, _ = spectral_decomposition(cc, dim=0)
        assert ev_before is ev_after, (
            "spectral cache must NOT be invalidated by set_embeddings"
        )

    def test_spectral_cache_attribute_exists(self):
        cc = _make_triangle_cc()
        assert hasattr(cc, "_spectral_cache"), (
            "CellComplex must have _spectral_cache attribute"
        )

    def test_topology_version_increments_on_mutation(self):
        cc = _make_triangle_cc()
        v0 = cc._topology_version
        d = cc.add_0_cell(torch.randn(4), "concept")
        v1 = cc._topology_version
        cc.add_1_cell(0, d, torch.randn(4), "r")
        v2 = cc._topology_version
        assert v1 > v0
        assert v2 > v1
