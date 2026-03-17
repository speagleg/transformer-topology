"""Tests for multi-scale Laplacian filtering — node-triangle incidence matrix."""
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex


def _make_cc_with_triangles(embedding_dim: int = 32) -> CellComplex:
    """Create a CellComplex with 5 nodes and two triangles.

    Nodes: 0, 1, 2, 3, 4
    Edges (indices):
        e0: (0, 1)
        e1: (1, 2)
        e2: (0, 2)
        e3: (2, 3)
        e4: (1, 3)
    Triangle 0 (t0): edges [e0, e1, e2] — nodes {0, 1, 2}
    Triangle 1 (t1): edges [e1, e3, e4] — nodes {1, 2, 3}
    Node 4 is in neither triangle.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    dim = embedding_dim

    # 5 nodes
    for _ in range(5):
        cc.add_0_cell(torch.randn(dim), "node")

    # 5 edges
    e0 = cc.add_1_cell(0, 1, torch.randn(dim), "edge")
    e1 = cc.add_1_cell(1, 2, torch.randn(dim), "edge")
    e2 = cc.add_1_cell(0, 2, torch.randn(dim), "edge")
    e3 = cc.add_1_cell(2, 3, torch.randn(dim), "edge")
    e4 = cc.add_1_cell(1, 3, torch.randn(dim), "edge")

    # Triangle 0: 0-1-2 via edges e0, e1, e2
    cc.add_2_cell([e0, e1, e2], torch.randn(dim), "triangle")
    # Triangle 1: 1-2-3 via edges e1, e3, e4
    cc.add_2_cell([e1, e3, e4], torch.randn(dim), "triangle")

    return cc


class TestNodeTriangleIncidenceShape:
    def test_shape_with_triangles(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        assert I.shape == (5, 2), f"Expected (5, 2), got {I.shape}"

    def test_shape_no_triangles(self):
        cc = CellComplex(embedding_dim=16)
        for _ in range(4):
            cc.add_0_cell(torch.randn(16), "node")
        cc.add_1_cell(0, 1, torch.randn(16), "edge")
        cc.add_1_cell(1, 2, torch.randn(16), "edge")
        cc.add_1_cell(2, 3, torch.randn(16), "edge")
        I = cc.node_triangle_incidence()
        assert I.shape == (4, 0), f"Expected (4, 0), got {I.shape}"

    def test_shape_no_nodes(self):
        cc = CellComplex(embedding_dim=8)
        I = cc.node_triangle_incidence()
        assert I.shape == (0, 0), f"Expected (0, 0), got {I.shape}"


class TestNodeTriangleIncidenceValues:
    def test_triangle0_contains_nodes_0_1_2(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        # Triangle 0 (column 0) must contain nodes 0, 1, 2
        assert I[0, 0].item() == 1.0, "Node 0 should be in triangle 0"
        assert I[1, 0].item() == 1.0, "Node 1 should be in triangle 0"
        assert I[2, 0].item() == 1.0, "Node 2 should be in triangle 0"

    def test_triangle0_excludes_nodes_3_4(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        assert I[3, 0].item() == 0.0, "Node 3 should NOT be in triangle 0"
        assert I[4, 0].item() == 0.0, "Node 4 should NOT be in triangle 0"

    def test_triangle1_contains_nodes_1_2_3(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        # Triangle 1 (column 1) must contain nodes 1, 2, 3
        assert I[1, 1].item() == 1.0, "Node 1 should be in triangle 1"
        assert I[2, 1].item() == 1.0, "Node 2 should be in triangle 1"
        assert I[3, 1].item() == 1.0, "Node 3 should be in triangle 1"

    def test_triangle1_excludes_nodes_0_4(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        assert I[0, 1].item() == 0.0, "Node 0 should NOT be in triangle 1"
        assert I[4, 1].item() == 0.0, "Node 4 should NOT be in triangle 1"

    def test_values_are_binary(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        unique_vals = I.unique().tolist()
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Non-binary value in incidence matrix: {v}"

    def test_each_triangle_has_exactly_3_nodes(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        for t in range(2):
            count = I[:, t].sum().item()
            assert count == 3.0, f"Triangle {t} should have exactly 3 nodes, got {count}"

    def test_node1_shared_by_both_triangles(self):
        """Node 1 is a vertex of both triangles (shared edge e1)."""
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        assert I[1, 0].item() == 1.0 and I[1, 1].item() == 1.0, (
            "Node 1 should appear in both triangles"
        )

    def test_node4_in_no_triangle(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        assert I[4, :].sum().item() == 0.0, "Node 4 should be in no triangle"


class TestNodeTriangleIncidenceDevice:
    def test_output_dtype_float(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        assert I.dtype == torch.float32, f"Expected float32, got {I.dtype}"

    def test_output_on_cpu(self):
        cc = _make_cc_with_triangles()
        I = cc.node_triangle_incidence()
        assert I.device.type == "cpu"


# ---------------------------------------------------------------------------
# MultiScaleLaplacianFilter tests
# ---------------------------------------------------------------------------

from src.spectral.multiscale_filter import MultiScaleLaplacianFilter


def test_multiscale_filter_forward_shape():
    cc = _make_cc_with_triangles(embedding_dim=32)
    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    signal = cc.get_embeddings(0)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    assert out.shape == signal.shape


def test_multiscale_filter_no_triangles():
    cc = CellComplex(embedding_dim=32)
    for i in range(5):
        cc.add_0_cell(torch.randn(32), cell_type="node")
    cc.add_1_cell(0, 1, torch.randn(32), relation_type="edge")
    cc.add_1_cell(1, 2, torch.randn(32), relation_type="edge")
    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    signal = cc.get_embeddings(0)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    assert out.shape == signal.shape


def test_multiscale_filter_no_edges():
    cc = CellComplex(embedding_dim=32)
    for i in range(3):
        cc.add_0_cell(torch.randn(32), cell_type="node")
    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    signal = cc.get_embeddings(0)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    assert out.shape == signal.shape


def test_multiscale_gates_are_learnable():
    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    param_names = [n for n, _ in msf.named_parameters()]
    assert any("gate" in n for n in param_names)


def test_multiscale_gradients_flow():
    cc = _make_cc_with_triangles(embedding_dim=32)
    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    signal = cc.get_embeddings(0).clone().requires_grad_(True)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    out.sum().backward()
    assert signal.grad is not None


def test_multiscale_skip_l1():
    cc = _make_cc_with_triangles(embedding_dim=32)
    msf = MultiScaleLaplacianFilter(embedding_dim=32, skip_l1=True)
    assert not hasattr(msf, 'filter_L1')
    signal = cc.get_embeddings(0)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    assert out.shape == signal.shape


def test_multiscale_skip_l2():
    cc = _make_cc_with_triangles(embedding_dim=32)
    msf = MultiScaleLaplacianFilter(embedding_dim=32, skip_l2=True)
    assert not hasattr(msf, 'filter_L2')
    signal = cc.get_embeddings(0)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    assert out.shape == signal.shape


# ---------------------------------------------------------------------------
# MultiFilterDynamics + use_multiscale tests
# ---------------------------------------------------------------------------

from src.wave.dynamics import MultiFilterDynamics


def test_multifilter_with_multiscale():
    cc = _make_cc_with_triangles(embedding_dim=32)
    mfd = MultiFilterDynamics(
        embedding_dim=32,
        filter_types=["wave_cosine"],
        use_multiscale=True,
        include_identity=True,
    )
    signal = cc.get_embeddings(0)
    dt = torch.tensor(0.5)
    wd = torch.tensor(0.1)
    out = mfd(cc, signal, dt, wd)
    assert out.shape == signal.shape


def test_multifilter_multiscale_num_paths():
    mfd = MultiFilterDynamics(
        embedding_dim=32,
        filter_types=["chebyshev", "wave_cosine"],
        use_multiscale=True,
        include_identity=True,
    )
    assert mfd.num_filters == 3  # 2 multiscale filters + identity


def test_multifilter_without_multiscale_unchanged():
    """Verify default behavior is unchanged."""
    mfd = MultiFilterDynamics(
        embedding_dim=32,
        filter_types=["wave_cosine"],
        include_identity=True,
    )
    assert mfd.num_filters == 2  # 1 filter + identity
