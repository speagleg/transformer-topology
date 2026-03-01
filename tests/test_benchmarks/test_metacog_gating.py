"""Tests for gated feature composition in HierarchicalMultiHopModel."""
import torch
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.cell_complex.cell_complex import CellComplex


def _make_model(use_metacog=True, use_llm=True, backend='qwen'):
    llm_config = {'backend': backend, 'llm_dim': 64, 'use_mock': True} if use_llm else None
    return HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=16, max_iterations=2, convergence_threshold=0.05,
        use_wave_dynamics=False, use_higher_order=False,
        use_llm=use_llm,
        llm_config=llm_config,
        use_multi_head_classifier=True,
        use_metacog=use_metacog,
    )


def _make_cc(n=10, dim=32, with_texts=True):
    cc = CellComplex(dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(dim), f"concept_{i}")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    if with_texts:
        cc.node_texts = [f"concept_{i}" for i in range(n)]
    return cc


def test_classifier_input_dim_with_metacog():
    """233 = 132 (base) + 96 (text) + 5 (metacog)."""
    model = _make_model(use_metacog=True, use_llm=True, backend='qwen')
    assert model.classifier_input_dim == 233


def test_classifier_input_dim_without_metacog():
    """228 = 132 (base) + 96 (text) + 0 (no metacog)."""
    model = _make_model(use_metacog=False, use_llm=True, backend='qwen')
    assert model.classifier_input_dim == 228


def test_classifier_input_dim_no_text():
    """132 = base only, no text, no metacog."""
    model = _make_model(use_metacog=False, use_llm=False)
    assert model.classifier_input_dim == 132


def test_classifier_input_dim_metacog_no_text():
    """137 = 132 (base) + 0 (no text) + 5 (metacog)."""
    model = _make_model(use_metacog=True, use_llm=False)
    assert model.classifier_input_dim == 137


def test_forward_with_metacog():
    """Full forward pass with metacog produces correct output shape."""
    model = _make_model(use_metacog=True, use_llm=True, backend='qwen')
    cc = _make_cc()
    logits = model(cc, 0, 5, task='bfs')
    assert logits.shape == (16,)  # bfs has 16 classes


def test_batched_forward_with_metacog():
    """_forward_batch batched DSM path should handle metacog gated composition."""
    from src.training.batch_utils import _forward_batch
    # Must use backend='dsm' to trigger the batched path (use_dsm=True)
    model = _make_model(use_metacog=True, use_llm=True, backend='dsm')
    cc1 = _make_cc(n=10, with_texts=True)
    cc2 = _make_cc(n=12, with_texts=True)
    batch = [
        (cc1, 0, 5, 3, None),
        (cc2, 1, 6, 7, None),
    ]
    results = _forward_batch(model, batch, torch.device('cpu'), task='bfs')
    assert len(results) == 2
    for logits, answer in results:
        assert logits.shape == (16,)  # bfs has 16 classes
