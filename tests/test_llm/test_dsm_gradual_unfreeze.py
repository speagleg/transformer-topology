"""Tests for DSM gradual unfreezing."""
import torch
from src.llm.dsm_backend import DSMBackend


class TestDSMGradualUnfreeze:
    def _make_backend(self):
        config = {
            "dsm_dim": 64, "num_heads": 4, "ff_dim": 256,
            "num_layers": 4, "cross_attn_layer": 1,
        }
        return DSMBackend(config)

    def test_freeze_all(self):
        backend = self._make_backend()
        backend.freeze_all()
        for p in backend.dsm.parameters():
            assert not p.requires_grad

    def test_unfreeze_top_n(self):
        backend = self._make_backend()
        backend.freeze_all()
        backend.unfreeze_top_n(2)
        frozen = sum(1 for p in backend.dsm.layers[:2].parameters() if not p.requires_grad)
        unfrozen = sum(1 for p in backend.dsm.layers[2:].parameters() if p.requires_grad)
        assert frozen > 0
        assert unfrozen > 0

    def test_unfreeze_all(self):
        backend = self._make_backend()
        backend.freeze_all()
        backend.unfreeze_all()
        for p in backend.dsm.parameters():
            assert p.requires_grad

    def test_load_pretrained_weights(self):
        backend = self._make_backend()
        pretrained = backend.dsm.state_dict()
        key = list(pretrained.keys())[0]
        pretrained[key] = torch.randn_like(pretrained[key])
        backend.load_pretrained(pretrained)
        assert torch.equal(backend.dsm.state_dict()[key], pretrained[key])
