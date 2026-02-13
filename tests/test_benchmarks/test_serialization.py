"""Tests for dataset save/load serialization."""

import os
import tempfile
import torch
import pytest
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.temporal_tasks import TemporalDataset


class TestMultiHopSerialization:
    def test_save_load_roundtrip(self):
        ds = MultiHopDataset(
            num_samples=5, min_hops=2, max_hops=4,
            num_distractors=3, embedding_dim=8,
        )
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            ds.save(path)
            loaded = MultiHopDataset.load(path)
            assert len(loaded) == len(ds)
            # Check first sample
            cc1, q1, t1, a1 = ds.samples[0]
            cc2, q2, t2, a2 = loaded.samples[0]
            assert q1 == q2
            assert t1 == t2
            assert a1 == a2
            assert torch.allclose(cc1.get_embeddings(0), cc2.get_embeddings(0))
        finally:
            os.unlink(path)

    def test_loaded_has_shuffle(self):
        ds = MultiHopDataset(
            num_samples=3, min_hops=2, max_hops=3,
            num_distractors=2, embedding_dim=8,
        )
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            ds.save(path)
            loaded = MultiHopDataset.load(path)
            # Should be able to call shuffle without error
            loaded.shuffle()
            assert loaded._indices is not None
        finally:
            os.unlink(path)


class TestTemporalSerialization:
    def test_save_load_roundtrip(self):
        ds = TemporalDataset(
            num_samples=5, task_type="propagation_delay",
            embedding_dim=8, n_nodes=6, max_delay=5,
        )
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            ds.save(path)
            loaded = TemporalDataset.load(path)
            assert len(loaded) == len(ds)
            cc1, q1, t1, a1 = ds.samples[0]
            cc2, q2, t2, a2 = loaded.samples[0]
            assert q1 == q2
            assert t1 == t2
            assert a1 == a2
        finally:
            os.unlink(path)
