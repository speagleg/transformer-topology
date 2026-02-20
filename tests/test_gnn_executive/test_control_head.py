import torch
import pytest
from src.gnn_executive.control_head import ControlSignal, ControlHead


class TestControlSignal:
    def test_dataclass_fields(self):
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
        )
        assert cs.frequency_gate.shape == (8,)
        assert cs.spatial_focus.shape == (5,)
        assert cs.confidence_weights.shape == (5,)
        assert cs.diffusion_time.shape == ()
        assert cs.wave_damping.shape == ()


class TestControlHead:
    def test_output_shapes(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32)
        cs = head(node_emb)
        assert cs.frequency_gate.shape == (8,)
        assert cs.spatial_focus.shape == (5,)
        assert cs.confidence_weights.shape == (5,)
        assert cs.diffusion_time.shape == ()
        assert cs.wave_damping.shape == ()

    def test_value_ranges(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(10, 32)
        cs = head(node_emb)
        # Sigmoid outputs in [0, 1]
        assert (cs.frequency_gate >= 0).all() and (cs.frequency_gate <= 1).all()
        assert (cs.spatial_focus >= 0).all() and (cs.spatial_focus <= 1).all()
        assert (cs.confidence_weights >= 0).all() and (cs.confidence_weights <= 1).all()
        # Softplus outputs are positive
        assert cs.diffusion_time > 0
        assert cs.wave_damping > 0

    def test_with_harmonic_energy(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32)
        cs_without = head(node_emb)
        cs_with = head(node_emb, harmonic_energy=torch.tensor(2.5))
        # Outputs should differ when harmonic energy changes
        assert not torch.allclose(cs_without.frequency_gate, cs_with.frequency_gate)

    def test_gradient_flow(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32, requires_grad=True)
        cs = head(node_emb)
        loss = (cs.frequency_gate.sum() + cs.spatial_focus.sum() +
                cs.confidence_weights.sum() + cs.diffusion_time + cs.wave_damping +
                cs.semantic_weight)
        loss.backward()
        assert node_emb.grad is not None
        assert node_emb.grad.abs().sum() > 0
        for name, p in head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_different_graph_sizes(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        for n_nodes in [3, 7, 15]:
            node_emb = torch.randn(n_nodes, 32)
            cs = head(node_emb)
            assert cs.spatial_focus.shape == (n_nodes,)
            assert cs.confidence_weights.shape == (n_nodes,)
            assert cs.frequency_gate.shape == (8,)

    def test_finite_for_wide_size_range(self):
        """Control head outputs are finite for n=5 through n=100."""
        head = ControlHead(embedding_dim=32, num_freqs=8)
        for n_nodes in [5, 10, 20, 50, 100]:
            node_emb = torch.randn(n_nodes, 32)
            cs = head(node_emb)
            assert torch.isfinite(cs.frequency_gate).all(), f"Non-finite freq_gate at n={n_nodes}"
            assert torch.isfinite(cs.spatial_focus).all(), f"Non-finite spatial_focus at n={n_nodes}"
            assert torch.isfinite(cs.confidence_weights).all(), f"Non-finite confidence at n={n_nodes}"
            assert torch.isfinite(cs.diffusion_time), f"Non-finite diffusion_time at n={n_nodes}"
            assert torch.isfinite(cs.wave_damping), f"Non-finite wave_damping at n={n_nodes}"
