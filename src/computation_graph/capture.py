from __future__ import annotations

import torch
import torch.nn as nn

from src.cell_complex.cell_complex import CellComplex


class ComputationGraphCapture:
    """Context manager that hooks into PyTorch modules to capture
    computation graph structure as forward/backward records."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.forward_records: list[dict] = []
        self.backward_records: list[dict] = []
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        # Use data_ptr() not id() because register_full_backward_hook wraps
        # output tensors, changing id() but preserving data_ptr().
        self._tensor_producers: dict[int, str] = {}  # data_ptr -> module_name
        self._tensor_consumers: list[tuple[str, str]] = []  # (producer, consumer)
        self._output_ptrs: dict[str, int] = {}  # module_name -> output data_ptr

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
            # Track input tensor consumers (skip connection detection).
            # Use data_ptr() because register_full_backward_hook wraps
            # tensors, changing id() but preserving data_ptr().
            if isinstance(input, tuple):
                for t in input:
                    if isinstance(t, torch.Tensor):
                        tid = t.data_ptr()
                        if tid in self._tensor_producers:
                            producer = self._tensor_producers[tid]
                            if producer != module_name:
                                self._tensor_consumers.append(
                                    (producer, module_name)
                                )
            elif isinstance(input, torch.Tensor):
                tid = input.data_ptr()
                if tid in self._tensor_producers:
                    producer = self._tensor_producers[tid]
                    if producer != module_name:
                        self._tensor_consumers.append(
                            (producer, module_name)
                        )

            # Track output tensor producer
            if isinstance(output, torch.Tensor):
                norm = output.detach().norm().item()
                shape = list(output.shape)
                self._tensor_producers[output.data_ptr()] = module_name
                self._output_ptrs[module_name] = output.data_ptr()
            elif isinstance(output, tuple) and len(output) > 0:
                norm = output[0].detach().norm().item() if isinstance(
                    output[0], torch.Tensor
                ) else 0.0
                shape = list(output[0].shape) if isinstance(
                    output[0], torch.Tensor
                ) else []
                if isinstance(output[0], torch.Tensor):
                    self._tensor_producers[output[0].data_ptr()] = module_name
                    self._output_ptrs[module_name] = output[0].data_ptr()
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

    def to_cell_complex(self, embedding_dim: int = 8) -> CellComplex:
        """Convert captured computation graph into a CellComplex.

        Mapping:
        - Each leaf module (no children) becomes a 0-cell, ordered by forward execution.
        - Each consecutive pair of leaf modules becomes a 1-cell (data flow edge).
        - Each composite module (>=2 leaf children) becomes a 2-cell.
          A closure edge (last child -> first child) is added to form a valid
          boundary cycle so that the chain complex property B1 @ B2 = 0 holds.

        Args:
            embedding_dim: Dimension of cell embeddings.

        Returns:
            A CellComplex representing the computation graph topology.
        """
        cc = CellComplex(embedding_dim=embedding_dim)

        # Identify leaf modules (no children)
        leaf_names = set()
        for name, mod in self.model.named_modules():
            if len(list(mod.children())) == 0:
                leaf_names.add(name)

        # Order leaves by forward execution (first occurrence in forward_records)
        seen: set[str] = set()
        ordered_leaves: list[str] = []
        fwd_norms: dict[str, float] = {}
        for rec in self.forward_records:
            name = rec['module_name']
            if name in leaf_names and name not in seen:
                seen.add(name)
                ordered_leaves.append(name)
                fwd_norms[name] = rec['activation_norm']

        # Collect backward norms (last occurrence wins, which is fine)
        bwd_norms: dict[str, float] = {}
        for rec in self.backward_records:
            name = rec['module_name']
            if name in leaf_names:
                bwd_norms[name] = rec['gradient_norm']

        # Add 0-cells (one per leaf module)
        name_to_idx: dict[str, int] = {}
        for name in ordered_leaves:
            emb = torch.zeros(embedding_dim)
            emb[0] = fwd_norms.get(name, 0.0)
            emb[1] = bwd_norms.get(name, 0.0)
            idx = cc.add_0_cell(emb, name)
            name_to_idx[name] = idx

        # Add 1-cells for consecutive leaf pairs (data flow edges)
        edge_indices: list[int] = []
        for i in range(len(ordered_leaves) - 1):
            src_name = ordered_leaves[i]
            tgt_name = ordered_leaves[i + 1]
            emb = torch.zeros(embedding_dim)
            emb[0] = fwd_norms.get(src_name, 0.0)
            emb[1] = fwd_norms.get(tgt_name, 0.0)
            emb[2] = bwd_norms.get(tgt_name, 0.0)
            eidx = cc.add_1_cell(
                name_to_idx[src_name], name_to_idx[tgt_name],
                emb, "data_flow",
            )
            edge_indices.append(eidx)

        # Add skip edges from tensor tracking
        existing_pairs = set()
        for ei in range(len(ordered_leaves) - 1):
            existing_pairs.add((
                name_to_idx[ordered_leaves[ei]],
                name_to_idx[ordered_leaves[ei + 1]],
            ))
        for producer, consumer in self._tensor_consumers:
            if producer in name_to_idx and consumer in name_to_idx:
                src_idx = name_to_idx[producer]
                tgt_idx = name_to_idx[consumer]
                if (src_idx, tgt_idx) not in existing_pairs:
                    existing_pairs.add((src_idx, tgt_idx))
                    emb = torch.zeros(embedding_dim)
                    emb[0] = fwd_norms.get(producer, 0.0)
                    emb[1] = fwd_norms.get(consumer, 0.0)
                    emb[2] = bwd_norms.get(consumer, 0.0)
                    eidx = cc.add_1_cell(src_idx, tgt_idx, emb, "skip")
                    edge_indices.append(eidx)

        # Detect residual connections: if a composite module's output differs
        # from its last child's output, it indicates a skip/residual path.
        for name, mod in self.model.named_modules():
            if len(list(mod.children())) == 0:
                continue  # leaf, skip
            parent_ptr = self._output_ptrs.get(name)
            if parent_ptr is None:
                continue
            # Find leaf descendants in execution order
            child_leaves: list[str] = []
            for leaf in ordered_leaves:
                if leaf != name and (
                    (name and leaf.startswith(name + ".")) or
                    (not name and leaf in leaf_names)
                ):
                    child_leaves.append(leaf)
            if len(child_leaves) < 2:
                continue
            last_child_ptr = self._output_ptrs.get(child_leaves[-1])
            if last_child_ptr is not None and parent_ptr != last_child_ptr:
                # Parent transforms last child's output → residual/skip path
                first_idx = name_to_idx[child_leaves[0]]
                last_idx = name_to_idx[child_leaves[-1]]
                if (last_idx, first_idx) not in existing_pairs:
                    existing_pairs.add((last_idx, first_idx))
                    emb = torch.zeros(embedding_dim)
                    emb[0] = fwd_norms.get(child_leaves[-1], 0.0)
                    emb[1] = fwd_norms.get(child_leaves[0], 0.0)
                    emb[2] = bwd_norms.get(child_leaves[0], 0.0)
                    eidx = cc.add_1_cell(last_idx, first_idx, emb, "skip")
                    edge_indices.append(eidx)

        # Add 2-cells for composite modules (those with >=2 leaf descendants)
        for name, mod in self.model.named_modules():
            children = list(mod.children())
            if len(children) < 2:
                continue

            # Find leaf descendants of this module
            child_leaf_names: list[str] = []
            for cname, _cmod in mod.named_modules():
                full = f"{name}.{cname}" if (name and cname) else (name or cname)
                if full in name_to_idx and full != name:
                    child_leaf_names.append(full)

            if len(child_leaf_names) < 3:
                # Need at least 3 nodes to form a cycle with 3 edges
                continue

            # Find the data flow edges that connect consecutive children
            child_idxs = {name_to_idx[n] for n in child_leaf_names}
            boundary: list[int] = []
            for ei in range(len(ordered_leaves) - 1):
                src_idx = name_to_idx[ordered_leaves[ei]]
                tgt_idx = name_to_idx[ordered_leaves[ei + 1]]
                if src_idx in child_idxs and tgt_idx in child_idxs:
                    boundary.append(edge_indices[ei])

            if len(boundary) < 2:
                continue

            # Add a closure edge (last child -> first child) to form a cycle
            # This is needed for a valid chain complex (B1 @ B2 = 0)
            first_child_name = child_leaf_names[0]
            last_child_name = child_leaf_names[-1]
            closure_emb = torch.zeros(embedding_dim)
            closure_emb[0] = fwd_norms.get(last_child_name, 0.0)
            closure_emb[1] = fwd_norms.get(first_child_name, 0.0)
            closure_idx = cc.add_1_cell(
                name_to_idx[last_child_name], name_to_idx[first_child_name],
                closure_emb, "closure",
            )
            boundary.append(closure_idx)

            # Build 2-cell embedding
            emb = torch.zeros(embedding_dim)
            child_fwd = [fwd_norms.get(n, 0.0) for n in child_leaf_names]
            child_bwd = [bwd_norms.get(n, 0.0) for n in child_leaf_names]
            if child_fwd:
                emb[0] = sum(child_fwd) / len(child_fwd)
            if child_bwd:
                emb[1] = sum(child_bwd) / len(child_bwd)
            emb[2] = float(len(child_leaf_names))
            cc.add_2_cell(boundary, emb)

        return cc
