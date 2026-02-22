"""Layer 2: Attention Flow Topology Analyzer.

Builds CellComplex from attention matrices, applies Hodge decomposition,
classifies attention heads by dominant component (gradient/curl/harmonic).
"""

from __future__ import annotations

import torch

from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import hodge_decomposition
from src.topology_analyzer.profile import LayerCellComplex

_EMBEDDING_DIM = 4


class AttentionFlowAnalyzer:
    """Build CellComplex from attention, Hodge decompose, classify heads."""

    def __init__(
        self,
        threshold_mode: str = "adaptive",
        threshold: float = 0.0,
    ):
        self.threshold_mode = threshold_mode
        self.threshold = threshold

    def build_cell_complex(
        self, attn_map: torch.Tensor, layer_idx: int, head_idx: int,
    ) -> LayerCellComplex:
        """Build CellComplex from an attention matrix.

        Args:
            attn_map: (seq_len, seq_len) attention weights.
            layer_idx: Transformer layer index.
            head_idx: Attention head index.
        """
        seq_len = attn_map.shape[0]
        cc = CellComplex(embedding_dim=_EMBEDDING_DIM)

        for i in range(seq_len):
            emb = torch.zeros(_EMBEDDING_DIM)
            emb[0] = attn_map[i].sum().item()
            cc.add_0_cell(emb, "token")

        if self.threshold_mode == "adaptive":
            mean_val = attn_map.mean().item()
            std_val = attn_map.std().item()
            thresh = mean_val + 0.5 * std_val
        else:
            thresh = self.threshold

        # Undirected edges for above-threshold attention
        edge_candidates: dict[tuple[int, int], tuple[float, float]] = {}
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                if attn_map[i, j].item() > thresh:
                    key = (min(i, j), max(i, j))
                    if key not in edge_candidates:
                        edge_candidates[key] = (
                            attn_map[key[0], key[1]].item(),
                            attn_map[key[1], key[0]].item(),
                        )

        edge_key_to_idx: dict[tuple[int, int], int] = {}
        for (lo, hi), (attn_lo_hi, attn_hi_lo) in edge_candidates.items():
            emb = torch.zeros(_EMBEDDING_DIM)
            emb[0] = (attn_lo_hi + attn_hi_lo) / 2.0
            emb[1] = attn_lo_hi
            emb[2] = attn_hi_lo
            idx = cc.add_1_cell(lo, hi, emb, "attention")
            edge_key_to_idx[(lo, hi)] = idx

        # Triangles
        adjacency: dict[int, set[int]] = {}
        for lo, hi in edge_key_to_idx:
            adjacency.setdefault(lo, set()).add(hi)
            adjacency.setdefault(hi, set()).add(lo)

        for a in sorted(adjacency):
            for b in sorted(adjacency.get(a, set())):
                if b <= a:
                    continue
                for c in sorted(adjacency.get(a, set()) & adjacency.get(b, set())):
                    if c <= b:
                        continue
                    e_ab = edge_key_to_idx[(a, b)]
                    e_ac = edge_key_to_idx[(a, c)]
                    e_bc = edge_key_to_idx[(b, c)]
                    cc.add_2_cell(
                        [e_ab, e_bc, e_ac],
                        torch.zeros(_EMBEDDING_DIM),
                        "attn_triangle",
                    )

        return LayerCellComplex(
            cc=cc, layer_idx=layer_idx, source="attention",
            head_idx=head_idx, neighborhood="threshold",
            k_or_epsilon=thresh,
        )

    def analyze_head(self, lcc: LayerCellComplex) -> dict:
        """Compute Hodge decomposition of attention flow."""
        cc = lcc.cc

        if cc.num_cells(1) < 2:
            return {"hodge_ratios": (0.0, 0.0, 0.0), "classification": "harmonic"}

        edge_embs = cc.get_embeddings(1)
        signal = edge_embs[:, 0]

        try:
            gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
            g_e = (gradient ** 2).sum().item()
            c_e = (curl ** 2).sum().item()
            h_e = (harmonic ** 2).sum().item()
            total = g_e + c_e + h_e
            if total < 1e-12:
                ratios = (0.0, 0.0, 0.0)
            else:
                ratios = (g_e / total, c_e / total, h_e / total)
        except Exception:
            ratios = (0.0, 0.0, 0.0)

        return {
            "hodge_ratios": ratios,
            "classification": self.classify_head(ratios),
        }

    @staticmethod
    def classify_head(hodge_ratios: tuple[float, float, float]) -> str:
        """Classify head by dominant Hodge component."""
        g, c, h = hodge_ratios
        if g >= c and g >= h and g > 0:
            return "gradient"
        if c >= g and c >= h and c > 0:
            return "curl"
        return "harmonic"
