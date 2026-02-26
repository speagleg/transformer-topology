import torch
from src.cell_complex.cell_complex import CellComplex


def test_node_texts_default_empty():
    cc = CellComplex(embedding_dim=8)
    assert cc.node_texts == []


def test_node_texts_set_and_get():
    cc = CellComplex(embedding_dim=8)
    cc.add_0_cell(torch.randn(8), "node")
    cc.add_0_cell(torch.randn(8), "node")
    cc.node_texts = ["dog", "animal"]
    assert cc.node_texts == ["dog", "animal"]


def test_node_texts_survives_clone():
    cc = CellComplex(embedding_dim=8)
    cc.add_0_cell(torch.randn(8), "node")
    cc.node_texts = ["dog"]
    cc2 = cc.clone()
    assert cc2.node_texts == ["dog"]
    # Verify it's a copy, not shared reference
    cc2.node_texts.append("cat")
    assert len(cc.node_texts) == 1


def test_node_texts_survives_to_device():
    cc = CellComplex(embedding_dim=8)
    cc.add_0_cell(torch.randn(8), "node")
    cc.node_texts = ["cat"]
    cc.to("cpu")
    assert cc.node_texts == ["cat"]
