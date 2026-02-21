from __future__ import annotations

import torch
import torch.nn as nn


class ComputationGraphCapture:
    """Context manager that hooks into PyTorch modules to capture
    computation graph structure as forward/backward records."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.forward_records: list[dict] = []
        self.backward_records: list[dict] = []
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self) -> 'ComputationGraphCapture':
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            h_fwd = module.register_forward_hook(self._make_forward_hook(name))
            h_bwd = module.register_full_backward_hook(
                self._make_backward_hook(name)
            )
            self._handles.append(h_fwd)
            self._handles.append(h_bwd)

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_forward_hook(self, module_name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                norm = output.detach().norm().item()
                shape = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0:
                norm = output[0].detach().norm().item() if isinstance(
                    output[0], torch.Tensor
                ) else 0.0
                shape = list(output[0].shape) if isinstance(
                    output[0], torch.Tensor
                ) else []
            else:
                norm = 0.0
                shape = []
            self.forward_records.append({
                'module_name': module_name,
                'module_type': type(module).__name__,
                'activation_norm': norm,
                'output_shape': shape,
            })
        return hook

    def _make_backward_hook(self, module_name: str):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                g = grad_output[0]
                norm = g.detach().norm().item() if g is not None else 0.0
            elif isinstance(grad_output, torch.Tensor):
                norm = grad_output.detach().norm().item()
            else:
                norm = 0.0
            self.backward_records.append({
                'module_name': module_name,
                'module_type': type(module).__name__,
                'gradient_norm': norm,
            })
        return hook
