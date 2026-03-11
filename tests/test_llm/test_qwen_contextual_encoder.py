import pytest
import torch
from src.llm.qwen_contextual_encoder import QwenContextualEncoder


class TestQwenContextualEncoder:
    """Tests for QwenContextualEncoder with mock mode."""

    def test_init_mock(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        assert enc.output_dim == 32
        assert enc.llm_dim == 64

    def test_forward_returns_correct_shape(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        texts = ["dog", "cat", "animal"]
        device = torch.device("cpu")
        result = enc(texts, device)
        assert result.shape == (3, 32)

    def test_forward_none_texts_returns_none(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        result = enc(None, torch.device("cpu"))
        assert result is None

    def test_forward_empty_texts_returns_none(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        result = enc([], torch.device("cpu"))
        assert result is None

    def test_projection_has_gradients(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        texts = ["dog", "cat"]
        result = enc(texts, torch.device("cpu"))
        loss = result.sum()
        loss.backward()
        assert enc.proj.weight.grad is not None

    def test_cache_precompute_and_lookup(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        device = torch.device("cpu")
        r1 = enc(["dog", "cat"], device)
        r2 = enc(["dog", "cat"], device)
        assert "dog" in enc._cpu_cache
        assert "cat" in enc._cpu_cache

    def test_cache_stores_raw_llm_dim(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        enc(["dog"], torch.device("cpu"))
        assert enc._cpu_cache["dog"].shape == (64,)

    def test_save_load_cache(self, tmp_path):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        enc(["dog", "cat"], torch.device("cpu"))
        path = tmp_path / "cache.pt"
        enc.save_cache(str(path))

        enc2 = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        loaded = enc2.load_cache(str(path))
        assert loaded == 2
        assert "dog" in enc2._cpu_cache

    def test_build_gpu_cache(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        enc(["dog", "cat", "fish"], torch.device("cpu"))
        enc.build_gpu_cache(torch.device("cpu"))
        assert enc._gpu_cache is not None
        assert enc._gpu_cache.shape[0] == 3

    def test_different_texts_different_embeddings(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        r1 = enc(["dog"], torch.device("cpu"))
        r2 = enc(["democracy"], torch.device("cpu"))
        assert not torch.allclose(r1, r2)
