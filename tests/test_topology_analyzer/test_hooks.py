"""Tests for TransformerHookManager."""

import torch
import torch.nn as nn


def _make_simple_transformer(num_layers=3, hidden_dim=32, num_heads=4):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=num_heads, dim_feedforward=64,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


class TestTransformerHookManager:
    def test_captures_hidden_states(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=3, hidden_dim=32)
        x = torch.randn(1, 8, 32)
        with TransformerHookManager(model) as manager:
            _ = model(x)
            hidden = manager.get_hidden_states()
        assert len(hidden) == 3
        for layer_idx, h in hidden.items():
            assert h.shape == (8, 32)

    def test_captures_attention_maps(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        x = torch.randn(1, 8, 32)
        with TransformerHookManager(model, capture_attention=True) as manager:
            _ = model(x)
            attn = manager.get_attention_maps()
        assert len(attn) == 2 * 4
        for (li, hi), a in attn.items():
            assert a.shape == (8, 8)

    def test_hooks_removed_on_exit(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=2, hidden_dim=32)
        with TransformerHookManager(model) as manager:
            pass
        assert len(manager._handles) == 0

    def test_clear_resets_state(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=2, hidden_dim=32)
        x = torch.randn(1, 8, 32)
        with TransformerHookManager(model) as manager:
            _ = model(x)
            assert len(manager.get_hidden_states()) == 2
            manager.clear()
            assert len(manager.get_hidden_states()) == 0

    def test_works_with_dsm_structure(self):
        from src.topology_analyzer.hooks import TransformerHookManager

        class FakeDSM(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=16, nhead=2, dim_feedforward=32, batch_first=True,
                    )
                    for _ in range(2)
                ])
                self.final_norm = nn.LayerNorm(16)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.final_norm(x)

        model = FakeDSM()
        x = torch.randn(1, 4, 16)
        with TransformerHookManager(model) as manager:
            _ = model(x)
            hidden = manager.get_hidden_states()
        assert len(hidden) == 2

    def test_batch_dim_handling(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=2, hidden_dim=32)
        x = torch.randn(4, 8, 32)
        with TransformerHookManager(model) as manager:
            _ = model(x)
            hidden = manager.get_hidden_states()
        for layer_idx, h in hidden.items():
            assert h.shape == (8, 32)
