"""Layer 1: Embedding Manifold Analyzer.

Builds CellComplex from transformer hidden states using k-NN neighborhoods,
then computes topological invariants (persistence, Betti, spectral gap, Hodge).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.cell_complex.cell_complex import CellComplex
from src.spectral.persistence import compute_persistence_diagram
from src.spectral.decomposition import spectral_decomposition, hodge_decomposition
from src.topology_analyzer.profile import LayerCellComplex


class EmbeddingManifoldAnalyzer:
    """Build CellComplex from hidden states and compute topological invariants."""

    def __init__(self, neighborhood: str = "knn", k: int = 5, epsilon: float = 1.0):
        self.neighborhood = neighborhood
        self.k = k
        self.epsilon = epsilon

    def build_cell_complex(
        self, hidden_states: torch.Tensor, layer_idx: int,
    ) -> LayerCellComplex:
        """Build CellComplex from hidden states at one layer.

        Args:
            hidden_states: (seq_len, d_model) tensor.
            layer_idx: Which transformer layer this came from.
        """
        seq_len, d_model = hidden_states.shape
        cc = CellComplex(embedding_dim=d_model)

        for i in range(seq_len):
            cc.add_0_cell(hidden_states[i].detach(), "token")

        dists = torch.cdist(hidden_states.unsqueeze(0), hidden_states.unsqueeze(0)).squeeze(0)
        edges = self._find_edges(dists, seq_len)

        norms = F.normalize(hidden_states.detach(), dim=1)
        cosine_sim = norms @ norms.T

        edge_set: set[tuple[int, int]] = set()
        for i, j in edges:
            key = (min(i, j), max(i, j))
            if key not in edge_set:
                edge_set.add(key)
                emb = torch.zeros(d_model)
                emb[0] = cosine_sim[key[0], key[1]].item()
                emb[1] = dists[key[0], key[1]].item()
                cc.add_1_cell(key[0], key[1], emb, "neighbor")

        # Build adjacency for triangle detection
        adjacency: dict[int, set[int]] = {}
        for lo, hi in edge_set:
            adjacency.setdefault(lo, set()).add(hi)
            adjacency.setdefault(hi, set()).add(lo)

        # Find triangles (a < b < c)
        for a in sorted(adjacency):
            for b in sorted(adjacency.get(a, set())):
                if b <= a:
                    continue
                for c in sorted(adjacency.get(a, set()) & adjacency.get(b, set())):
                    if c <= b:
                        continue
                    e_ab = self._edge_idx(cc, a, b)
                    e_ac = self._edge_idx(cc, a, c)
                    e_bc = self._edge_idx(cc, b, c)
                    if e_ab is not None and e_ac is not None and e_bc is not None:
                        boundary = [e_ab, e_bc, e_ac]
                        cc.add_2_cell(boundary, torch.zeros(d_model), "triangle")

        k_or_eps = float(self.k) if self.neighborhood != "epsilon" else self.epsilon
        return LayerCellComplex(
            cc=cc, layer_idx=layer_idx, source="embedding",
            head_idx=None, neighborhood=self.neighborhood,
            k_or_epsilon=k_or_eps,
        )

    def _find_edges(self, dists: torch.Tensor, n: int) -> list[tuple[int, int]]:
        edges = []
        k = min(self.k, n - 1)
        if k < 1:
            return edges

        if self.neighborhood == "knn":
            _, indices = dists.topk(k + 1, largest=False, dim=1)
            for i in range(n):
                for j_idx in range(1, k + 1):
                    j = indices[i, j_idx].item()
                    edges.append((i, j))

        elif self.neighborhood == "mutual_knn":
            _, indices = dists.topk(k + 1, largest=False, dim=1)
            knn_sets = [set(indices[i, 1:k + 1].tolist()) for i in range(n)]
            for i in range(n):
                for j in knn_sets[i]:
                    if i in knn_sets[j]:
                        edges.append((i, j))

        elif self.neighborhood == "epsilon":
            mask = (dists < self.epsilon) & (dists > 0)
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i, j]:
                        edges.append((i, j))
        return edges

    @staticmethod
    def _edge_idx(cc: CellComplex, src: int, tgt: int) -> int | None:
        for e in range(cc.num_cells(1)):
            s, t = cc._1_cell_sources[e], cc._1_cell_targets[e]
            if (s == src and t == tgt) or (s == tgt and t == src):
                return e
        return None

    def analyze_layer(self, lcc: LayerCellComplex) -> dict:
        """Compute all topological invariants for a layer's CellComplex."""
        cc = lcc.cc
        result = {}

        diagrams = compute_persistence_diagram(cc, max_dimension=1)
        result["persistence"] = diagrams

        beta_0 = diagrams[0].shape[0] if diagrams[0].shape[0] > 0 else 0
        beta_1 = diagrams[1].shape[0] if len(diagrams) > 1 and diagrams[1].shape[0] > 0 else 0
        result["betti"] = (beta_0, beta_1)

        if cc.num_cells(0) >= 2:
            try:
                eigenvalues, _ = spectral_decomposition(cc, dim=0)
                nonzero = eigenvalues[eigenvalues > 1e-6]
                result["spectral_gap"] = nonzero[0].item() if len(nonzero) > 0 else 0.0
            except Exception:
                result["spectral_gap"] = 0.0
        else:
            result["spectral_gap"] = 0.0

        if cc.num_cells(1) >= 2:
            try:
                edge_embs = cc.get_embeddings(1)
                signal = edge_embs[:, 0]
                gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
                total = (signal ** 2).sum().item()
                if total > 1e-12:
                    g = (gradient ** 2).sum().item() / total
                    c = (curl ** 2).sum().item() / total
                    h = (harmonic ** 2).sum().item() / total
                    result["hodge_ratios"] = (g, c, h)
                else:
                    result["hodge_ratios"] = (0.0, 0.0, 0.0)
            except Exception:
                result["hodge_ratios"] = (0.0, 0.0, 0.0)
        else:
            result["hodge_ratios"] = (0.0, 0.0, 0.0)

        return result
