"""Forward-hook manager for extracting hidden states and attention maps.

Registers hooks on auto-detected transformer layers to capture intermediate
representations. Works as a context manager: hooks are registered on __enter__
and removed (with original forwards restored) on __exit__.

Supports both batch-first (batch, seq_len, hidden_dim) and non-batch-first
(seq_len, hidden_dim) outputs -- the DSM model uses the latter convention.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn as nn


_LAYER_ATTR_PATHS: list[tuple[str, ...]] = [
    ("layers",),                       # DSM, many custom models
    ("encoder", "layers"),             # nn.TransformerEncoder
    ("decoder", "layers"),             # nn.TransformerDecoder
    ("transformer", "h"),              # GPT-2 style
    ("model", "layers"),               # Llama
    ("model", "model", "layers"),      # PEFT-wrapped Llama
    ("encoder", "layer"),              # BERT-style
    ("h",),
]


def _find_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Auto-detect the list of transformer layers in a model.

    Walks well-known attribute paths and returns the first that resolves
    to an iterable of nn.Module instances. Returns empty list if none found.
    """
    for attr_path in _LAYER_ATTR_PATHS:
        obj = model
        try:
            for attr in attr_path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue

        if isinstance(obj, (nn.ModuleList, nn.Sequential)):
            return list(obj)
        if isinstance(obj, (list, tuple)) and all(isinstance(m, nn.Module) for m in obj):
            return list(obj)

    return []


_SELF_ATTN_ATTRS = ["self_attn", "attn", "attention", "self_attention", "mha"]


def _find_self_attn(layer: nn.Module) -> nn.Module | None:
    """Locate the self-attention sub-module inside a transformer layer."""
    for attr in _SELF_ATTN_ATTRS:
        attn = getattr(layer, attr, None)
        if attn is not None and isinstance(attn, nn.Module):
            return attn
    return None


class TransformerHookManager:
    """Context manager that hooks transformer layers to capture activations.

    On __enter__, forward hooks are registered on every detected layer.
    Optionally, self_attn modules are monkey-patched to return per-head
    attention weights.

    On __exit__, all hooks are removed and original forwards are restored.

    Args:
        model: Any nn.Module with detectable transformer layers.
        capture_attention: If True, also capture per-head attention maps.
    """

    def __init__(self, model: nn.Module, capture_attention: bool = False) -> None:
        self._model = model
        self._capture_attention = capture_attention
        self._layers = _find_transformer_layers(model)
        self._hidden_states: dict[int, torch.Tensor] = {}
        self._attention_maps: dict[tuple[int, int], torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._original_attn_forwards: dict[int, Any] = {}

    def __enter__(self) -> TransformerHookManager:
        self._register_hooks()
        return self

    def __exit__(self, *args: Any) -> None:
        self._remove_hooks()

    def get_hidden_states(self) -> dict[int, torch.Tensor]:
        """Return captured hidden states: layer_idx -> (seq_len, hidden_dim)."""
        return dict(self._hidden_states)

    def get_attention_maps(self) -> dict[tuple[int, int], torch.Tensor]:
        """Return captured attention maps: (layer_idx, head_idx) -> (seq_len, seq_len)."""
        return dict(self._attention_maps)

    def clear(self) -> None:
        """Reset captured data without removing hooks."""
        self._hidden_states.clear()
        self._attention_maps.clear()

    def _register_hooks(self) -> None:
        for layer_idx, layer in enumerate(self._layers):
            handle = layer.register_forward_hook(self._make_hidden_hook(layer_idx))
            self._handles.append(handle)

            if self._capture_attention:
                self._patch_attention(layer_idx, layer)

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        for layer_idx, original_forward in self._original_attn_forwards.items():
            attn_module = _find_self_attn(self._layers[layer_idx])
            if attn_module is not None:
                attn_module.forward = original_forward
        self._original_attn_forwards.clear()

    def _make_hidden_hook(self, layer_idx: int):
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            hidden = hidden.detach()
            # 3D -> take first batch item; 2D -> use directly (DSM)
            if hidden.dim() == 3:
                hidden = hidden[0]
            self._hidden_states[layer_idx] = hidden
        return hook

    def _patch_attention(self, layer_idx: int, layer: nn.Module) -> None:
        attn_module = _find_self_attn(layer)
        if attn_module is None or not isinstance(attn_module, nn.MultiheadAttention):
            return

        self._original_attn_forwards[layer_idx] = attn_module.forward
        manager = self

        @functools.wraps(attn_module.forward)
        def patched_forward(*args: Any, **kwargs: Any) -> Any:
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            original_fn = manager._original_attn_forwards[layer_idx]
            output = original_fn(*args, **kwargs)

            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    attn_weights = attn_weights.detach()
                    # 4D: (batch, heads, tgt, src) -> take [0]
                    if attn_weights.dim() == 4:
                        attn_weights = attn_weights[0]
                    if attn_weights.dim() == 3:
                        for head_idx in range(attn_weights.size(0)):
                            manager._attention_maps[(layer_idx, head_idx)] = (
                                attn_weights[head_idx]
                            )
            return output

        attn_module.forward = patched_forward
