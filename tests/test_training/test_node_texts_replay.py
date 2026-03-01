"""Test that structural samples without node_texts don't crash during replay."""
import torch
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.run_comparison import HierarchicalMultiHopModel


def test_structural_sample_no_crash():
    """Forward pass on a structural sample (empty node_texts) should not crash."""
    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=16, max_iterations=2, convergence_threshold=0.05,
        use_wave_dynamics=False, use_higher_order=False,
        use_llm=True,
        llm_config={'backend': 'qwen', 'llm_dim': 64, 'use_mock': True},
        use_multi_head_classifier=True,
        use_metacog=True,
    )
    # Structural sample: node_texts is empty list (not None, not missing)
    cc = CellComplex(32)
    for i in range(10):
        cc.add_0_cell(torch.randn(32), f"node_{i}")
    for i in range(9):
        cc.add_1_cell(i, i + 1, torch.randn(32), "edge")
    # node_texts initialized as [] by CellComplex.__init__

    # Should NOT crash -- text_features will be None, zero-padded
    logits = model(cc, 0, 5, task='bfs')
    assert logits.shape == (16,)


def test_structural_sample_batched_no_crash():
    """Batched forward on structural samples (empty node_texts) should not crash."""
    from src.training.batch_utils import _forward_batch
    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=16, max_iterations=2, convergence_threshold=0.05,
        use_wave_dynamics=False, use_higher_order=False,
        use_llm=True,
        llm_config={'backend': 'dsm', 'use_mock': True},
        use_multi_head_classifier=True,
        use_metacog=True,
    )
    cc = CellComplex(32)
    for i in range(10):
        cc.add_0_cell(torch.randn(32), f"node_{i}")
    for i in range(9):
        cc.add_1_cell(i, i + 1, torch.randn(32), "edge")
    # No node_texts set -- empty list from CellComplex.__init__

    batch = [(cc, 0, 5, 3, None)]
    results = _forward_batch(model, batch, torch.device('cpu'), task='bfs')
    assert len(results) == 1
    logits, answer = results[0]
    assert logits.shape == (16,)
