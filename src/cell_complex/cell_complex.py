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
        self.node_texts: list[str] = []
        # Caching infrastructure (Task 1: GPU optimization)
        self._boundary_cache: dict[int, torch.Tensor] = {}
        self._adjacency_cache: dict[int, torch.Tensor] = {}
        self._edge_index_cache: torch.Tensor | None = None
        self._topology_version: int = 0
        self._spectral_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _ensure_caches(self):
        """Initialize cache attributes for objects deserialized from old format."""
        if not hasattr(self, '_boundary_cache'):
            self._boundary_cache = {}
            self._adjacency_cache = {}
            self._edge_index_cache = None
            self._topology_version = 0
            self._spectral_cache = {}

    def _invalidate_caches(self):
        """Clear all topology-dependent caches. Called after any structural mutation."""
        self._ensure_caches()
        self._boundary_cache.clear()
        self._adjacency_cache.clear()
        self._edge_index_cache = None
        self._topology_version += 1
        self._spectral_cache.clear()

    @property
    def device(self) -> torch.device:
        """Return the device of the cell embeddings (defaults to CPU)."""
        if self._0_cell_embeddings:
            return self._0_cell_embeddings[0].device
        return torch.device('cpu')

    def to(self, device: torch.device | str) -> 'CellComplex':
        """Move all cell embeddings to the given device. Returns self."""
        device = torch.device(device)
        self._0_cell_embeddings = [e.to(device) for e in self._0_cell_embeddings]
        self._1_cell_embeddings = [e.to(device) for e in self._1_cell_embeddings]
        self._2_cell_embeddings = [e.to(device) for e in self._2_cell_embeddings]
        # Cached tensors are device-specific; clear them so they are rebuilt on new device
        self._ensure_caches()
        self._boundary_cache.clear()
        self._adjacency_cache.clear()
        self._edge_index_cache = None
        self._spectral_cache.clear()
        return self

    def clone(self) -> 'CellComplex':
        """Create a copy with cloned embeddings. Structural data is shared."""
        cc = CellComplex.__new__(CellComplex)
        cc.embedding_dim = self.embedding_dim
        cc._0_cell_embeddings = [e.clone() for e in self._0_cell_embeddings]
        cc._1_cell_embeddings = [e.clone() for e in self._1_cell_embeddings]
        cc._2_cell_embeddings = [e.clone() for e in self._2_cell_embeddings]
        cc._0_cell_types = self._0_cell_types
        cc._1_cell_types = self._1_cell_types
        cc._1_cell_sources = self._1_cell_sources
        cc._1_cell_targets = self._1_cell_targets
        cc._2_cell_boundaries = self._2_cell_boundaries
        cc._2_cell_signs = self._2_cell_signs
        cc._2_cell_types = self._2_cell_types
        cc.node_texts = list(self.node_texts)
        # Fresh caches for the clone — do not share with the original
        cc._boundary_cache = {}
        cc._adjacency_cache = {}
        cc._edge_index_cache = None
        self._ensure_caches()
        cc._topology_version = self._topology_version
        cc._spectral_cache = {}
        return cc

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
        self._invalidate_caches()
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
        self._invalidate_caches()
        return idx

    def get_embeddings(self, dim: int) -> torch.Tensor:
        """Return stacked embeddings for all cells of the given dimension."""
        if dim == 0:
            if not self._0_cell_embeddings:
                return torch.empty(0, self.embedding_dim, device=self.device)
            return torch.stack(self._0_cell_embeddings)
        elif dim == 1:
            if not self._1_cell_embeddings:
                return torch.empty(0, self.embedding_dim, device=self.device)
            return torch.stack(self._1_cell_embeddings)
        elif dim == 2:
            if not self._2_cell_embeddings:
                return torch.empty(0, self.embedding_dim, device=self.device)
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
        self._invalidate_caches()
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
        self._ensure_caches()
        if dim in self._boundary_cache:
            cached = self._boundary_cache[dim]
            if cached.device == self.device:
                return cached

        if dim == 1:
            n0 = self.num_cells(0)
            n1 = self.num_cells(1)
            if n0 == 0 or n1 == 0:
                result = torch.zeros(max(n0, 1), max(n1, 1), device=self.device)
            else:
                result = torch.zeros(n0, n1, device=self.device)
                for j, (s, t) in enumerate(zip(self._1_cell_sources, self._1_cell_targets)):
                    result[s, j] = -1.0
                    result[t, j] = 1.0
            self._boundary_cache[dim] = result
            return result
        elif dim == 2:
            n1 = self.num_cells(1)
            n2 = self.num_cells(2)
            if n1 == 0 or n2 == 0:
                result = torch.zeros(max(n1, 1), max(n2, 1), device=self.device)
            else:
                result = torch.zeros(n1, n2, device=self.device)
                for j, (boundary_edges, signs) in enumerate(
                    zip(self._2_cell_boundaries, self._2_cell_signs)
                ):
                    for e, sign in zip(boundary_edges, signs):
                        result[e, j] = sign
            self._boundary_cache[dim] = result
            return result
        raise ValueError(f"Boundary operator for dim={dim} not supported")

    def adjacency_matrix(self, dim: int) -> torch.Tensor:
        """Compute the adjacency matrix for cells of the given dimension.

        For dim=0, two 0-cells are adjacent if connected by a 1-cell.
        """
        self._ensure_caches()
        if dim in self._adjacency_cache:
            cached = self._adjacency_cache[dim]
            if cached.device == self.device:
                return cached

        if dim == 0:
            n = self.num_cells(0)
            result = torch.zeros(n, n, device=self.device)
            for s, t in zip(self._1_cell_sources, self._1_cell_targets):
                result[s, t] = 1.0
                result[t, s] = 1.0
            self._adjacency_cache[dim] = result
            return result
        elif dim == 1:
            # Edge-edge adjacency: two 1-cells are adjacent if they share a 0-cell endpoint
            n = self.num_cells(1)
            result = torch.zeros(n, n, device=self.device)
            for i in range(n):
                for j in range(i + 1, n):
                    endpoints_i = {self._1_cell_sources[i], self._1_cell_targets[i]}
                    endpoints_j = {self._1_cell_sources[j], self._1_cell_targets[j]}
                    if endpoints_i & endpoints_j:
                        result[i, j] = 1.0
                        result[j, i] = 1.0
            self._adjacency_cache[dim] = result
            return result
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

    def compute_structural_features(self) -> torch.Tensor:
        """Compute structural features for each 0-cell.

        Returns (N, 7) tensor with:
            0: Normalized degree (degree / max_degree, or 0 if no edges)
            1: is_source indicator (cell type == "source")
            2: is_target indicator (cell type == "target")
            3: is_blocked indicator (cell type == "blocked")
            4: is_source2 indicator (cell type == "source2")
            5: degree / N (degree normalized by total node count)
            6: log(N) / log(100) (graph scale indicator, same for all nodes)
        """
        import math

        n = self.num_cells(0)
        if n == 0:
            return torch.zeros(0, 7, device=self.device)

        features = torch.zeros(n, 7, device=self.device)

        # Degree
        degrees = torch.zeros(n, device=self.device)
        for s, t in zip(self._1_cell_sources, self._1_cell_targets):
            degrees[s] += 1
            degrees[t] += 1
        max_deg = degrees.max().item()
        if max_deg > 0:
            features[:, 0] = degrees / max_deg

        # Type indicators
        type_map = {"source": 1, "target": 2, "blocked": 3, "source2": 4}
        for i, ctype in enumerate(self._0_cell_types):
            col = type_map.get(ctype)
            if col is not None:
                features[i, col] = 1.0

        # Size-aware features
        features[:, 5] = degrees / n  # degree normalized by total node count
        features[:, 6] = math.log(max(n, 1)) / math.log(100)  # graph scale

        return features

    def edge_weight_matrix(self, feature_dim: int = 0) -> torch.Tensor:
        """Build an (N, N) matrix of edge weights from 1-cell embeddings.

        For each edge (s, t), the weight is taken from the edge embedding
        at the given feature dimension. The matrix is symmetric (undirected).

        Args:
            feature_dim: Which dimension of the edge embedding to use as weight.

        Returns:
            Tensor of shape (N, N) where N = num_0_cells. Zero for non-edges.
        """
        n = self.num_cells(0)
        W = torch.zeros(n, n, device=self.device)
        for i, (s, t) in enumerate(zip(self._1_cell_sources, self._1_cell_targets)):
            w = self._1_cell_embeddings[i][feature_dim].item()
            W[s, t] = w
            W[t, s] = w
        return W

    def edge_index(self) -> torch.Tensor:
        """Return PyG-compatible edge_index tensor (2, 2*num_edges) for undirected graph."""
        self._ensure_caches()
        if self._edge_index_cache is not None:
            if self._edge_index_cache.device == self.device:
                return self._edge_index_cache

        sources = self._1_cell_sources
        targets = self._1_cell_targets
        row = torch.tensor(sources + targets, dtype=torch.long, device=self.device)
        col = torch.tensor(targets + sources, dtype=torch.long, device=self.device)
        result = torch.stack([row, col], dim=0)
        self._edge_index_cache = result
        return result
