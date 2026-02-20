"""Build text prompts from graph statistics and control signals for LLM input."""

import torch

from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.control_head import ControlSignal


def build_topo_prompt(
    cc: CellComplex,
    control_signal: ControlSignal | None = None,
    task_type: str = "unknown",
    metadata: dict | None = None,
    iteration: int = 0,
    max_iterations: int = 5,
) -> str:
    """Build a text prompt summarizing the graph and reasoning state.

    Format:
    [TOPO] nodes=N edges=E topology=T beta1=B | task=T |
     iter=I/M conf=C [/TOPO]

    Args:
        cc: The cell complex being reasoned over.
        control_signal: Current control signal from the GNN executive.
        task_type: Name of the current task.
        metadata: Optional task metadata dict (may contain 'task_prompt').
        iteration: Current reasoning iteration.
        max_iterations: Maximum reasoning iterations.

    Returns:
        A short text string for LLM context.
    """
    # If metadata has a prebuilt prompt, use it
    if metadata and 'task_prompt' in metadata:
        return f"[TOPO] {metadata['task_prompt']} | iter={iteration}/{max_iterations} [/TOPO]"

    # Build from graph statistics
    n_nodes = cc.num_cells(0)
    n_edges = cc.num_cells(1)
    n_faces = cc.num_cells(2)

    parts = [f"nodes={n_nodes}", f"edges={n_edges}"]
    if n_faces > 0:
        parts.append(f"faces={n_faces}")

    parts.append(f"task={task_type}")
    parts.append(f"iter={iteration}/{max_iterations}")

    if control_signal is not None:
        conf = control_signal.confidence_weights.mean().item()
        parts.append(f"conf={conf:.2f}")
        if control_signal.semantic_weight is not None:
            parts.append(f"gate={control_signal.semantic_weight.item():.2f}")

    return "[TOPO] " + " | ".join(parts) + " [/TOPO]"
