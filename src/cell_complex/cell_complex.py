import torch
from typing import Optional


class CellComplex:
    """CW-complex data structure with 0-cells, 1-cells, and 2-cells.

    Stores cell embeddings and boundary operators as sparse tensors.
    Phase 1: 0-cells and 1-cells only. 2-cells added in Phase 2.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self._0_cell_embeddings: list[torch.Tensor] = []
        self._0_cell_types: list[str] = []
        self._1_cell_embeddings: list[torch.Tensor] = []
        self._1_cell_types: list[str] = []
        self._1_cell_sources: list[int] = []
        self._1_cell_targets: list[int] = []
        self._2_cell_embeddings: list[torch.Tensor] = []
        self._2_cell_boundaries: list[list[int]] = []

    def num_cells(self, dim: int) -> int:
        """Return the number of cells of the given dimension."""
        if dim == 0:
            return len(self._0_cell_embeddings)
        elif dim == 1:
            return len(self._1_cell_embeddings)
        elif dim == 2:
            return len(self._2_cell_embeddings)
        raise ValueError(f"Unsupported cell dimension: {dim}")

    def add_0_cell(self, embedding: torch.Tensor, cell_type: str) -> int:
        """Add a 0-cell (node) and return its index."""
        assert embedding.shape == (self.embedding_dim,)
        idx = len(self._0_cell_embeddings)
        self._0_cell_embeddings.append(embedding.clone())
        self._0_cell_types.append(cell_type)
        return idx

    def add_1_cell(self, source: int, target: int, embedding: torch.Tensor, relation_type: str) -> int:
        """Add a 1-cell (edge) between two 0-cells and return its index."""
        assert embedding.shape == (self.embedding_dim,)
        assert 0 <= source < self.num_cells(0)
        assert 0 <= target < self.num_cells(0)
        idx = len(self._1_cell_embeddings)
        self._1_cell_embeddings.append(embedding.clone())
        self._1_cell_types.append(relation_type)
        self._1_cell_sources.append(source)
        self._1_cell_targets.append(target)
        return idx

    def get_embeddings(self, dim: int) -> torch.Tensor:
        """Return stacked embeddings for all cells of the given dimension."""
        if dim == 0:
            if not self._0_cell_embeddings:
                return torch.empty(0, self.embedding_dim)
            return torch.stack(self._0_cell_embeddings)
        elif dim == 1:
            if not self._1_cell_embeddings:
                return torch.empty(0, self.embedding_dim)
            return torch.stack(self._1_cell_embeddings)
        elif dim == 2:
            if not self._2_cell_embeddings:
                return torch.empty(0, self.embedding_dim)
            return torch.stack(self._2_cell_embeddings)
        raise ValueError(f"Unsupported cell dimension: {dim}")

    def set_embeddings(self, dim: int, embeddings: torch.Tensor):
        """Replace all embeddings for cells of the given dimension."""
        if dim == 0:
            assert embeddings.shape == (self.num_cells(0), self.embedding_dim)
            self._0_cell_embeddings = list(embeddings)
        elif dim == 1:
            assert embeddings.shape == (self.num_cells(1), self.embedding_dim)
            self._1_cell_embeddings = list(embeddings)
        else:
            raise ValueError(f"Unsupported cell dimension: {dim}")

    def boundary_operator(self, dim: int) -> torch.Tensor:
        """Compute the boundary operator matrix B_dim.

        B_1 is shape (num_0_cells, num_1_cells) where:
          B_1[source, edge] = -1  (edge leaves source)
          B_1[target, edge] = +1  (edge arrives at target)
        """
        if dim == 1:
            n0 = self.num_cells(0)
            n1 = self.num_cells(1)
            if n0 == 0 or n1 == 0:
                return torch.zeros(max(n0, 1), max(n1, 1))
            B = torch.zeros(n0, n1)
            for j, (s, t) in enumerate(zip(self._1_cell_sources, self._1_cell_targets)):
                B[s, j] = -1.0
                B[t, j] = 1.0
            return B
        raise ValueError(f"Boundary operator for dim={dim} not implemented (Phase 1)")

    def adjacency_matrix(self, dim: int) -> torch.Tensor:
        """Compute the adjacency matrix for cells of the given dimension.

        For dim=0, two 0-cells are adjacent if connected by a 1-cell.
        """
        if dim == 0:
            n = self.num_cells(0)
            A = torch.zeros(n, n)
            for s, t in zip(self._1_cell_sources, self._1_cell_targets):
                A[s, t] = 1.0
                A[t, s] = 1.0
            return A
        raise ValueError(f"Adjacency for dim={dim} not implemented")

    def edge_index(self) -> torch.Tensor:
        """Return PyG-compatible edge_index tensor (2, 2*num_edges) for undirected graph."""
        sources = self._1_cell_sources
        targets = self._1_cell_targets
        row = torch.tensor(sources + targets, dtype=torch.long)
        col = torch.tensor(targets + sources, dtype=torch.long)
        return torch.stack([row, col], dim=0)
