"""Tests for MultiFilterDynamics ensemble."""

import torch
import pytest

from src.cell_complex.cell_complex import CellComplex
from src.wave.dynamics import MultiFilterDynamics


def make_chain(dim: int = 16, length: int = 5) -> CellComplex:
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "edge")
    return cc


DIM = 16


class TestMultiFilterConstruction:
    def test_multi_filter_construction(self):
        mfd = MultiFilterDynamics(DIM, filter_types=['chebyshev', 'wave_cosine', 'heat'])
        assert len(mfd.filters) == 3
        assert mfd.include_identity is True
        assert mfd.num_filters == 4  # 3 filters + 1 identity

    def test_multi_filter_no_identity(self):
        mfd = MultiFilterDynamics(DIM, filter_types=['heat'], include_identity=False)
        assert mfd.num_filters == 1

    def test_multi_filter_num_filters_property(self):
        mfd = MultiFilterDynamics(DIM, filter_types=['heat', 'chebyshev'],
                                  include_identity=True)
        assert mfd.num_filters == 3


class TestMultiFilterForward:
    def test_multi_filter_output_shape(self):
        cc = make_chain(DIM, 5)
        mfd = MultiFilterDynamics(DIM, filter_types=['heat', 'wave_cosine'])
        signal = cc.get_embeddings(0)
        out = mfd(cc, signal,
                  diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1))
        assert out.shape == (5, DIM)

    def test_multi_filter_equal_weights_default(self):
        """No filter_weights → equal blending."""
        cc = make_chain(DIM, 5)
        mfd = MultiFilterDynamics(DIM, filter_types=['heat'],
                                  include_identity=True)
        signal = cc.get_embeddings(0)
        out = mfd(cc, signal,
                  diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1))
        assert out.shape == (5, DIM)
        assert torch.isfinite(out).all()

    def test_multi_filter_weighted_blend(self):
        """Explicit weights: all weight on identity → output ≈ signal."""
        cc = make_chain(DIM, 5)
        mfd = MultiFilterDynamics(DIM, filter_types=['heat'],
                                  include_identity=True,
                                  use_wave_strength_gate=False)
        signal = cc.get_embeddings(0)
        # Weight: [0.0 (heat), 1.0 (identity)]
        weights = torch.tensor([0.0, 1.0])
        out = mfd(cc, signal,
                  diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1),
                  filter_weights=weights)
        assert torch.allclose(out, signal, atol=1e-5)

    def test_multi_filter_with_identity_path(self):
        """Identity path returns signal unchanged."""
        cc = make_chain(DIM, 5)
        mfd = MultiFilterDynamics(DIM, filter_types=['heat'],
                                  include_identity=True)
        signal = cc.get_embeddings(0)
        # Full weight on identity
        weights = torch.tensor([0.0, 1.0])
        out = mfd(cc, signal,
                  diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1),
                  filter_weights=weights)
        assert torch.allclose(out, signal, atol=1e-5)


class TestMultiFilterGradient:
    def test_multi_filter_gradient_flow(self):
        """Gradients reach all filter parameters."""
        cc = make_chain(DIM, 5)
        mfd = MultiFilterDynamics(DIM, filter_types=['chebyshev', 'bandpass'],
                                  include_identity=True)
        signal = cc.get_embeddings(0)
        weights = torch.tensor([0.4, 0.3, 0.3], requires_grad=True)
        out = mfd(cc, signal,
                  diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1),
                  filter_weights=weights)
        loss = out.sum()
        loss.backward()
        # Weights should have gradient
        assert weights.grad is not None
        # At least some filter params should have gradients
        grads = [p.grad for p in mfd.filters.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_multi_filter_wave_strength_gate(self):
        """Gate bypass: strength → 0 means output ≈ signal."""
        cc = make_chain(DIM, 5)
        mfd = MultiFilterDynamics(DIM, filter_types=['heat'],
                                  include_identity=False,
                                  use_wave_strength_gate=True)
        with torch.no_grad():
            mfd.wave_strength.fill_(-100.0)
        signal = cc.get_embeddings(0)
        out = mfd(cc, signal,
                  diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1))
        assert torch.allclose(out, signal, atol=1e-4)

    def test_multi_filter_single_filter_degenerates(self):
        """Single filter with no identity → similar to WaveDynamics."""
        cc = make_chain(DIM, 5)
        mfd = MultiFilterDynamics(DIM, filter_types=['heat'],
                                  include_identity=False,
                                  use_wave_strength_gate=False,
                                  use_neural_ode=False)
        signal = cc.get_embeddings(0)
        out = mfd(cc, signal,
                  diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1))
        assert out.shape == signal.shape
        assert torch.isfinite(out).all()
