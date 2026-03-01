"""Tests for MultiHeadClassifier."""
import torch
from src.training.multi_head_classifier import MultiHeadClassifier


class TestMultiHeadClassifier:
    def test_forward_returns_correct_shape(self):
        task_classes = {"diverse": 11, "bfs": 16, "graph_completion": 2}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        out = clf(x, "diverse")
        assert out.shape == (1, 11)

    def test_different_tasks_different_shapes(self):
        task_classes = {"diverse": 11, "bfs": 16, "graph_completion": 2}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        assert clf(x, "diverse").shape == (1, 11)
        assert clf(x, "bfs").shape == (1, 16)
        assert clf(x, "graph_completion").shape == (1, 2)

    def test_heads_persist_across_calls(self):
        task_classes = {"diverse": 11, "bfs": 16}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        sd_before = {k: v.clone() for k, v in clf.heads["diverse"].state_dict().items()}
        _ = clf(x, "bfs")
        sd_after = clf.heads["diverse"].state_dict()
        for k in sd_before:
            assert torch.equal(sd_before[k], sd_after[k])

    def test_add_task_creates_new_head(self):
        task_classes = {"diverse": 11}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        clf.add_task("new_task", 5)
        x = torch.randn(1, 64)
        assert clf(x, "new_task").shape == (1, 5)

    def test_unknown_task_raises(self):
        task_classes = {"diverse": 11}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        try:
            clf(x, "nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_from_task_registry(self):
        from src.benchmarks.benchmark_dataset import TASK_REGISTRY, get_max_classes
        task_classes = {t: get_max_classes(t) for t in TASK_REGISTRY}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        for task, n_cls in task_classes.items():
            assert clf(x, task).shape == (1, n_cls)
