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
        self._2_cell_signs: list[list[float]] = []  # orientation signs for B2
        self._2_cell_types: list[str] = []

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
        elif dim == 2:
            assert embeddings.shape == (self.num_cells(2), self.embedding_dim)
            self._2_cell_embeddings = list(embeddings)
        else:
            raise ValueError(f"Unsupported cell dimension: {dim}")

    def add_2_cell(self, boundary_edges: list[int], embedding: torch.Tensor,
                   cell_type: str = "face") -> int:
        """Add a 2-cell (face) bounded by a cycle of edges and return its index.

        Args:
            boundary_edges: List of 1-cell indices forming the boundary cycle.
            embedding: Embedding tensor of shape (embedding_dim,).
            cell_type: Type label for the 2-cell.

        Returns:
            Index of the new 2-cell.
        """
        assert embedding.shape == (self.embedding_dim,)
        n1 = self.num_cells(1)
        for e in boundary_edges:
            assert 0 <= e < n1, f"Edge index {e} out of range [0, {n1})"
        # Validate that boundary edges form a cycle
        edges_as_pairs = []
        for e in boundary_edges:
            edges_as_pairs.append((self._1_cell_sources[e], self._1_cell_targets[e]))
        # Check connectivity: each node must appear exactly twice (once as endpoint of two edges)
        node_counts: dict[int, int] = {}
        for s, t in edges_as_pairs:
            node_counts[s] = node_counts.get(s, 0) + 1
            node_counts[t] = node_counts.get(t, 0) + 1
        for node, count in node_counts.items():
            assert count == 2, (
                f"Node {node} appears {count} times in boundary edges; expected 2 for a cycle"
            )

        # Compute orientation signs for B2 (chain complex property: B1 @ B2 = 0)
        # Order edges into a cycle and determine if each edge is traversed
        # in its stored direction (+1) or opposite (-1)
        signs = self._compute_boundary_signs(boundary_edges)

        idx = len(self._2_cell_embeddings)
        self._2_cell_embeddings.append(embedding.clone())
        self._2_cell_boundaries.append(list(boundary_edges))
        self._2_cell_signs.append(signs)
        self._2_cell_types.append(cell_type)
        return idx

    def _compute_boundary_signs(self, boundary_edges: list[int]) -> list[float]:
        """Compute orientation signs for boundary edges so B1 @ B2 = 0.

        Walks the cycle and assigns +1 if the edge direction matches
        the walk direction, -1 otherwise.
        """
        # Build the cycle walk
        edge_endpoints = {}
        for e in boundary_edges:
            edge_endpoints[e] = (self._1_cell_sources[e], self._1_cell_targets[e])

        # Start with first edge
        signs = [0.0] * len(boundary_edges)
        first_edge = boundary_edges[0]
        s, t = edge_endpoints[first_edge]
        signs[0] = 1.0  # First edge goes in its natural direction
        current_node = t
        used = {first_edge}

        for step in range(1, len(boundary_edges)):
            for idx, e in enumerate(boundary_edges):
                if e in used:
                    continue
                es, et = edge_endpoints[e]
                if es == current_node:
                    signs[idx] = 1.0
                    current_node = et
                    used.add(e)
                    break
                elif et == current_node:
                    signs[idx] = -1.0
                    current_node = es
                    used.add(e)
                    break

        return signs

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
        elif dim == 2:
            n1 = self.num_cells(1)
            n2 = self.num_cells(2)
            if n1 == 0 or n2 == 0:
                return torch.zeros(max(n1, 1), max(n2, 1))
            B = torch.zeros(n1, n2)
            for j, (boundary_edges, signs) in enumerate(
                zip(self._2_cell_boundaries, self._2_cell_signs)
            ):
                for e, sign in zip(boundary_edges, signs):
                    B[e, j] = sign
            return B
        raise ValueError(f"Boundary operator for dim={dim} not supported")

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
        elif dim == 1:
            # Edge-edge adjacency: two 1-cells are adjacent if they share a 0-cell endpoint
            n = self.num_cells(1)
            A = torch.zeros(n, n)
            for i in range(n):
                for j in range(i + 1, n):
                    endpoints_i = {self._1_cell_sources[i], self._1_cell_targets[i]}
                    endpoints_j = {self._1_cell_sources[j], self._1_cell_targets[j]}
                    if endpoints_i & endpoints_j:
                        A[i, j] = 1.0
                        A[j, i] = 1.0
            return A
        raise ValueError(f"Adjacency for dim={dim} not supported")

    def verify_chain_complex(self) -> bool:
        """Verify the chain complex property: B1 @ B2 = 0.

        Returns True if the property holds (within numerical tolerance).
        """
        if self.num_cells(2) == 0:
            return True
        B1 = self.boundary_operator(1)
        B2 = self.boundary_operator(2)
        product = B1 @ B2
        return torch.allclose(product, torch.zeros_like(product), atol=1e-6)

    def edge_index(self) -> torch.Tensor:
        """Return PyG-compatible edge_index tensor (2, 2*num_edges) for undirected graph."""
        sources = self._1_cell_sources
        targets = self._1_cell_targets
        row = torch.tensor(sources + targets, dtype=torch.long)
        col = torch.tensor(targets + sources, dtype=torch.long)
        return torch.stack([row, col], dim=0)
