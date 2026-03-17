"""Build a CellComplex from LLM reasoning steps."""

import torch
import torch.nn.functional as F
from src.cell_complex.cell_complex import CellComplex


class ReasoningGraphConstructor:
    """Incrementally constructs a CellComplex from sequential reasoning steps.

    Each step becomes a 0-cell. Dependencies become 1-cells (edges).
    Triangles among mutually connected nodes become 2-cells.

    Args:
        embedding_dim: Dimension of node/edge/face embeddings (default: 128).
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._cc = CellComplex(embedding_dim)
        self._step_texts: list[str] = []
        # Track adjacency for triangle detection: set of (min, max) node pairs
        self._adjacency: set[tuple[int, int]] = set()
        # Map (min_node, max_node) -> edge_index for triangle 2-cell construction
        self._edge_map: dict[tuple[int, int], int] = {}

    def add_step(
        self,
        step_text: str,
        embedding: torch.Tensor,
        depends_on: list[int],
        step_type: str = "deduction",
        contradicts: list[int] | None = None,
    ) -> int:
        """Add a reasoning step as a 0-cell and connect to dependencies.

        Args:
            step_text: Natural language description of the step.
            embedding: Embedding tensor of shape (embedding_dim,).
            depends_on: 1-indexed step numbers this step depends on.
            step_type: Type label for the 0-cell (e.g., "deduction", "observation").
            contradicts: 1-indexed step numbers this step contradicts (optional).

        Returns:
            1-indexed step number of the newly added step.
        """
        node_idx = self._cc.add_0_cell(embedding, cell_type=step_type)
        self._step_texts.append(step_text)

        # Convert 1-indexed depends_on to 0-indexed, filter invalid
        dep_indices = []
        for d in depends_on:
            try:
                d_int = int(d)
            except (TypeError, ValueError):
                continue
            zero_idx = d_int - 1
            if 0 <= zero_idx < node_idx:  # must reference earlier step
                dep_indices.append(zero_idx)

        # Fallback: if no valid deps and not the first step, connect to previous
        if not dep_indices and node_idx > 0:
            dep_indices = [node_idx - 1]

        # Add dependency edges
        for dep_idx in dep_indices:
            direction = "forward" if dep_idx < node_idx else "backward"
            edge_emb = self._make_edge_embedding(dep_idx, node_idx, embedding)
            self._cc.add_1_cell(dep_idx, node_idx, edge_emb, relation_type=direction)
            pair = (min(dep_idx, node_idx), max(dep_idx, node_idx))
            self._adjacency.add(pair)
            self._edge_map[pair] = self._cc.num_cells(1) - 1

        # Add contradiction edges
        if contradicts:
            for c in contradicts:
                zero_idx = c - 1
                if 0 <= zero_idx < node_idx:
                    edge_emb = self._make_edge_embedding(zero_idx, node_idx, embedding)
                    self._cc.add_1_cell(zero_idx, node_idx, edge_emb, relation_type="contradiction")
                    pair = (min(zero_idx, node_idx), max(zero_idx, node_idx))
                    self._adjacency.add(pair)
                    self._edge_map[pair] = self._cc.num_cells(1) - 1

        # Scan for new triangles involving node_idx
        self._detect_triangles(node_idx)

        return node_idx + 1  # return 1-indexed

    def _make_edge_embedding(self, src_idx: int, tgt_idx: int, new_emb: torch.Tensor) -> torch.Tensor:
        """Create edge embedding with cosine similarity in dim 0."""
        src_emb = self._cc._0_cell_embeddings[src_idx]
        cos_sim = F.cosine_similarity(src_emb.unsqueeze(0), new_emb.unsqueeze(0)).item()
        edge_emb = torch.zeros(self.embedding_dim)
        edge_emb[0] = cos_sim
        return edge_emb

    def _detect_triangles(self, new_node: int):
        """Find triangles formed by new_node with existing edges."""
        # Get all neighbors of new_node
        neighbors = []
        for a, b in self._adjacency:
            if a == new_node:
                neighbors.append(b)
            elif b == new_node:
                neighbors.append(a)

        # Check all pairs of neighbors for mutual edge
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                ni, nj = neighbors[i], neighbors[j]
                pair = (min(ni, nj), max(ni, nj))
                if pair in self._adjacency:
                    # Triangle: new_node, ni, nj — add 2-cell
                    self._add_triangle(new_node, ni, nj)

    def _add_triangle(self, a: int, b: int, c: int):
        """Add a 2-cell for triangle (a, b, c) if the 3 edges exist."""
        pairs = [
            (min(a, b), max(a, b)),
            (min(b, c), max(b, c)),
            (min(a, c), max(a, c)),
        ]
        edge_indices = []
        for p in pairs:
            if p not in self._edge_map:
                return  # edge missing, cannot form 2-cell
            edge_indices.append(self._edge_map[p])

        # Average the edge embeddings for the face embedding
        face_emb = torch.zeros(self.embedding_dim)
        for eidx in edge_indices:
            face_emb += self._cc._1_cell_embeddings[eidx]
        face_emb /= len(edge_indices)

        self._cc.add_2_cell(edge_indices, face_emb, cell_type="triangle")

    def get_snapshot(self) -> CellComplex:
        """Return a cloned CellComplex for read-only analysis."""
        return self._cc.clone()

    @property
    def num_steps(self) -> int:
        """Number of reasoning steps added so far."""
        return self._cc.num_cells(0)
