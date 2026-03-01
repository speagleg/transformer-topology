"""Tests for MLP-based TaskHead in MultiHeadClassifier."""
import torch
from src.training.multi_head_classifier import MultiHeadClassifier


def test_task_head_is_mlp():
    """Each task head should be an MLP (Sequential), not a single Linear."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16, 'kg_relation': 10})
    head = mc.heads['bfs']
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) >= 2, f"Expected MLP with >=2 Linear layers, got {len(linears)}"


def test_task_head_hidden_dim():
    """MLP hidden dim should default to 128."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16})
    head = mc.heads['bfs']
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert linears[0].out_features == 128


def test_task_head_output_shape():
    """MLP output should match n_classes."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16, 'kg_relation': 10})
    x = torch.randn(1, 233)
    out_bfs = mc(x, 'bfs')
    out_kg = mc(x, 'kg_relation')
    assert out_bfs.shape == (1, 16)
    assert out_kg.shape == (1, 10)


def test_add_task_creates_mlp():
    """Dynamically added tasks should also use MLP heads."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16})
    mc.add_task('kg_concept', 9)
    head = mc.heads['kg_concept']
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) >= 2


def test_task_head_has_dropout():
    """MLP head should include dropout for regularization."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16})
    head = mc.heads['bfs']
    dropouts = [m for m in head.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) >= 1


def test_gradient_flows_through_mlp():
    """Gradients should flow through MLP hidden layers."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16})
    x = torch.randn(1, 233, requires_grad=True)
    out = mc(x, 'bfs')
    out.sum().backward()
    assert x.grad is not None
    linears = [m for m in mc.heads['bfs'].modules() if isinstance(m, torch.nn.Linear)]
    for lin in linears:
        assert lin.weight.grad is not None
