"""Tests for feature replay buffer."""
import torch
from src.training.feature_replay import FeatureReplayBuffer


class TestFeatureReplayBuffer:
    def test_save_and_sample(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        features = torch.randn(64)
        buf.save("diverse", features, label=3)
        samples = buf.sample("diverse", n=1)
        assert len(samples) == 1
        feat, label = samples[0]
        assert feat.shape == (64,)
        assert label == 3

    def test_sample_multiple(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        for i in range(20):
            buf.save("bfs", torch.randn(64), label=i % 16)
        samples = buf.sample("bfs", n=10)
        assert len(samples) == 10

    def test_max_capacity(self):
        buf = FeatureReplayBuffer(max_per_task=5)
        for i in range(10):
            buf.save("diverse", torch.randn(64), label=0)
        assert len(buf.buffers["diverse"]) == 5

    def test_sample_empty_task(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        samples = buf.sample("nonexistent", n=5)
        assert len(samples) == 0

    def test_sample_ratio(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        for i in range(50):
            buf.save("diverse", torch.randn(64), label=i % 11)
        samples = buf.sample_ratio("diverse", ratio=0.5)
        assert len(samples) == 25

    def test_all_tasks(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        buf.save("diverse", torch.randn(64), label=0)
        buf.save("bfs", torch.randn(64), label=1)
        assert set(buf.tasks()) == {"diverse", "bfs"}
