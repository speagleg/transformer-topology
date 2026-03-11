"""End-to-end test: build v10 model, run forward pass, verify shapes and gradients."""

import pytest
import torch
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop
from src.reasoning_loop.attention_readout import AttentionReadout
from src.llm.qwen_contextual_encoder import QwenContextualEncoder


DIM = 32
LOOP_KWARGS = dict(
    embedding_dim=DIM, gnn_hidden=64, gnn_spatial_layers=2,
    gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
    tat_spatial_heads=4, tat_spectral_heads=4, tat_ff_dim=64,
    max_iterations=2, use_dual_track=True,
    use_wave_dynamics=False,
)


def _make_cc(n=10, embed_dim=32):
    cc = CellComplex(embedding_dim=embed_dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(embed_dim), "node")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, torch.randn(embed_dim), "edge")
    cc.node_texts = [f"concept_{i}" for i in range(n)]
    return cc


class TestV10EndToEnd:

    def test_full_forward_pass(self):
        """Build all v10 components, run one sample, verify output."""
        encoder = QwenContextualEncoder(llm_dim=64, output_dim=DIM, use_mock=True)
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        readout = AttentionReadout(embed_dim=DIM, num_tasks=23)
        classifier = torch.nn.Sequential(
            torch.nn.Linear(101, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 10),
        )

        cc = _make_cc()
        device = torch.device("cpu")

        text_embs = encoder(cc.node_texts, device)
        assert text_embs.shape == (10, DIM)

        h_out, iters, diag = loop(cc, text_embeddings=text_embs)
        assert h_out.shape == (10, DIM)
        assert 'fusion_weight' in diag

        topo = torch.randn(4)
        fw_val = diag['fusion_weight']
        fw = torch.tensor(fw_val) if not isinstance(fw_val, torch.Tensor) else fw_val
        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=0, topo_features=topo, fusion_weight=fw,
        )
        assert combined.shape == (101,)

        logits = classifier(combined)
        assert logits.shape == (10,)

    def test_gradient_flows_end_to_end(self):
        encoder = QwenContextualEncoder(llm_dim=64, output_dim=DIM, use_mock=True)
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        readout = AttentionReadout(embed_dim=DIM, num_tasks=23)
        classifier = torch.nn.Sequential(
            torch.nn.Linear(101, 64), torch.nn.GELU(), torch.nn.Linear(64, 5),
        )

        cc = _make_cc()
        text_embs = encoder(cc.node_texts, torch.device("cpu"))
        h_out, _, diag = loop(cc, text_embeddings=text_embs)

        topo = torch.randn(4)
        fw_val = diag.get('fusion_weight', 0.5)
        fw = torch.tensor(fw_val) if not isinstance(fw_val, torch.Tensor) else fw_val
        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=0, topo_features=topo, fusion_weight=fw,
        )
        logits = classifier(combined)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([2]))
        loss.backward()

        # Verify gradients reach all components
        assert encoder.proj.weight.grad is not None, "No gradient to Qwen projection"
        gnn_grads = sum(1 for p in loop.parameters() if p.grad is not None)
        assert gnn_grads > 0, "No gradients to executive loop"
        readout_grads = sum(1 for p in readout.parameters() if p.grad is not None)
        assert readout_grads > 0, "No gradients to readout"
