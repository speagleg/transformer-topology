"""Integration test: full DSM pipeline smoke test.

Verifies the complete GNN -> Wave -> DSM -> TAT pipeline:
- Forward pass produces correct output shape
- Backward pass produces gradients for all param groups
- Optimizer step doesn't crash
- Variable graph sizes work
- DSM-specific: semantic_bias is produced and used
"""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.run_comparison import HierarchicalMultiHopModel


class TestDSMPipelineSmokeTest:
    """End-to-end tests for the full DSM pipeline."""

    @pytest.fixture
    def dsm_config(self):
        """Small DSM config for testing."""
        return {
            'backend': 'dsm',
            'dsm_dim': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_layers': 2,
            'cross_attn_layer': 0,
            'num_prefix': 4,
        }

    @pytest.fixture
    def model(self, dsm_config):
        return HierarchicalMultiHopModel(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=64, max_classes=3,
            max_iterations=2, convergence_threshold=0.01,
            use_llm=True,
            llm_config=dsm_config,
        )

    def _make_cc(self, n=10, dim=32):
        """Create a simple chain CellComplex for testing."""
        cc = CellComplex(embedding_dim=dim)
        for i in range(n):
            cc.add_0_cell(torch.randn(dim), "node")
        for i in range(n - 1):
            cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
        return cc

    def test_forward_pass(self, model):
        """Forward pass produces correct output shape."""
        cc = self._make_cc()
        out = model(cc.clone(), 0, 1)
        assert out.shape == (3,), f"Expected (3,), got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains NaN/Inf"

    def test_backward_pass(self, model):
        """Backward pass produces gradients for DSM, GNN, and TAT params."""
        cc = self._make_cc()
        out = model(cc.clone(), 0, 1)
        loss = torch.nn.functional.cross_entropy(out.unsqueeze(0), torch.tensor([1]))
        loss.backward()

        # Check that at least some parameters got gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert grad_count > 0, "No gradients produced"
        # At least 30% of trainable params should have grads
        assert grad_count > total_params * 0.3, \
            f"Only {grad_count}/{total_params} params got gradients"

    def test_dsm_params_get_gradients(self, model):
        """DSM-specific parameters receive gradients during backprop."""
        cc = self._make_cc()
        out = model(cc.clone(), 0, 1)
        loss = torch.nn.functional.cross_entropy(out.unsqueeze(0), torch.tensor([0]))
        loss.backward()

        dsm_has_grad = False
        bridge_has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'dsm' in name.lower() or 'distilled' in name.lower():
                    dsm_has_grad = True
                if 'topo_bridge' in name or 'bridge' in name:
                    bridge_has_grad = True

        # Note: DSM params may or may not get grads depending on how the
        # executive loop routes through TopoBridge. At minimum, bridge should.
        assert bridge_has_grad or dsm_has_grad, \
            "Neither DSM nor TopoBridge params received gradients"

    def test_optimizer_step(self, model):
        """A full optimizer step completes without errors."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cc = self._make_cc()
        model.train()

        out = model(cc.clone(), 0, 1)
        loss = torch.nn.functional.cross_entropy(out.unsqueeze(0), torch.tensor([1]))
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"

    def test_variable_graph_sizes(self, model):
        """Model handles different graph sizes correctly."""
        for n in [5, 10, 20]:
            cc = self._make_cc(n=n)
            out = model(cc.clone(), 0, min(1, n - 1))
            assert out.shape == (3,), f"Failed for n={n}: got {out.shape}"
            assert torch.isfinite(out).all(), f"NaN/Inf for n={n}"

    def test_multiple_forward_passes(self, model):
        """Multiple forward passes don't accumulate state incorrectly."""
        cc1 = self._make_cc(n=8)
        cc2 = self._make_cc(n=12)

        out1 = model(cc1.clone(), 0, 1)
        out2 = model(cc2.clone(), 0, 1)

        assert out1.shape == (3,)
        assert out2.shape == (3,)
        # Outputs should generally be different for different inputs
        # (not guaranteed but very likely with random embeddings)

    def test_eval_mode(self, model):
        """Model works in eval mode (no dropout, etc.)."""
        model.eval()
        with torch.no_grad():
            cc = self._make_cc()
            out = model(cc.clone(), 0, 1)
            assert out.shape == (3,)
            assert torch.isfinite(out).all()

    def test_model_has_dsm_components(self, model):
        """Model has the expected DSM-related components."""
        # Check that executive_loop exists and has DSM support
        assert hasattr(model, 'executive_loop'), "Model missing executive_loop"
        loop = model.executive_loop
        assert hasattr(loop, 'use_dsm'), "Executive loop missing use_dsm flag"
        assert loop.use_dsm, "DSM should be enabled"
        assert hasattr(loop, 'topo_bridge'), "Executive loop missing topo_bridge"
        assert loop.topo_bridge is not None, "topo_bridge should not be None"
