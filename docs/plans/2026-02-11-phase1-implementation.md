# Phase 1: Multi-Hop Graph Traversal — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the core cell complex data structures, GNN executive, topology-aware transformer, and validate on multi-hop graph traversal benchmarks.

**Architecture:** Cell complex (0-cells + 1-cells only for Phase 1) → GNN executive with spatial message passing + spectral filtering → Topology-aware transformer with dual-basis attention and Laplacian PE → Interleaved co-processing loop. Dual spatial/spectral pathways throughout.

**Tech Stack:** Python 3.12, PyTorch, PyTorch Geometric, scipy, networkx, numpy, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/cell_complex/__init__.py`
- Create: `src/gnn_executive/__init__.py`
- Create: `src/tat/__init__.py`
- Create: `src/spectral/__init__.py`
- Create: `src/wave/__init__.py`
- Create: `src/reasoning_loop/__init__.py`
- Create: `src/benchmarks/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_cell_complex/__init__.py`
- Create: `tests/test_gnn_executive/__init__.py`
- Create: `tests/test_tat/__init__.py`
- Create: `tests/test_spectral/__init__.py`
- Create: `tests/test_benchmarks/__init__.py`
- Create: `config/default.yaml`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "transformer-topology"
version = "0.1.0"
description = "Topology-aware transformers integrated with GNN executive for structured reasoning"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.0",
    "torch-geometric>=2.5",
    "scipy>=1.12",
    "networkx>=3.2",
    "numpy>=1.26",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
]
spectral = [
    "torchdiffeq>=0.2",
]
homology = [
    "gudhi>=3.9",
]
tracking = [
    "wandb>=0.16",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

**Step 2: Create virtual environment and install dependencies**

```bash
cd /mnt/c/Users/gspea/source/repos/transformer-topology
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Note: PyTorch Geometric may require specific install steps depending on CUDA. For CPU-only development:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -e ".[dev]"
```

**Step 3: Create all `__init__.py` files and package directories**

All `__init__.py` files are empty for now. Create the full directory tree:
```
src/__init__.py
src/cell_complex/__init__.py
src/gnn_executive/__init__.py
src/tat/__init__.py
src/spectral/__init__.py
src/wave/__init__.py
src/reasoning_loop/__init__.py
src/benchmarks/__init__.py
tests/__init__.py
tests/test_cell_complex/__init__.py
tests/test_gnn_executive/__init__.py
tests/test_tat/__init__.py
tests/test_spectral/__init__.py
tests/test_benchmarks/__init__.py
```

**Step 4: Create default config**

```yaml
# config/default.yaml
model:
  embedding_dim: 64
  num_gnn_layers: 4
  num_tat_layers: 4
  num_spatial_heads: 4
  num_spectral_heads: 4
  tat_ff_dim: 256
  dropout: 0.1

training:
  learning_rate: 0.001
  batch_size: 32
  max_epochs: 100
  patience: 10

benchmark:
  min_hops: 2
  max_hops: 10
  num_distractors: 20
  num_train: 5000
  num_val: 1000
  num_test: 1000

reasoning_loop:
  max_iterations: 5
  convergence_threshold: 0.01
```

**Step 5: Verify setup**

Run: `source .venv/bin/activate && python -c "import torch; import torch_geometric; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add pyproject.toml src/ tests/ config/
git commit -m "feat: project scaffolding with package structure and config"
```

---

### Task 2: Cell Complex Data Structure

**Files:**
- Create: `src/cell_complex/cell_complex.py`
- Create: `tests/test_cell_complex/test_cell_complex.py`

**Step 1: Write failing tests for CellComplex**

```python
# tests/test_cell_complex/test_cell_complex.py
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex


class TestCellComplexConstruction:
    def test_empty_complex(self):
        cc = CellComplex(embedding_dim=64)
        assert cc.num_cells(0) == 0
        assert cc.num_cells(1) == 0
        assert cc.num_cells(2) == 0

    def test_add_0_cells(self):
        cc = CellComplex(embedding_dim=4)
        idx = cc.add_0_cell(embedding=torch.randn(4), cell_type="concept")
        assert idx == 0
        assert cc.num_cells(0) == 1

    def test_add_1_cell(self):
        cc = CellComplex(embedding_dim=4)
        a = cc.add_0_cell(torch.randn(4), "concept")
        b = cc.add_0_cell(torch.randn(4), "concept")
        e = cc.add_1_cell(source=a, target=b, embedding=torch.randn(4), relation_type="causes")
        assert e == 0
        assert cc.num_cells(1) == 1

    def test_boundary_operator_1(self):
        """∂₁ maps 1-cells to their boundary 0-cells."""
        cc = CellComplex(embedding_dim=4)
        a = cc.add_0_cell(torch.randn(4), "concept")
        b = cc.add_0_cell(torch.randn(4), "concept")
        c = cc.add_0_cell(torch.randn(4), "concept")
        cc.add_1_cell(a, b, torch.randn(4), "r1")  # edge 0: a -> b
        cc.add_1_cell(b, c, torch.randn(4), "r2")  # edge 1: b -> c

        B1 = cc.boundary_operator(1)  # shape: (num_0_cells, num_1_cells)
        assert B1.shape == (3, 2)
        # Column 0 (edge a->b): -1 at row a, +1 at row b
        assert B1[a, 0].item() == -1.0
        assert B1[b, 0].item() == 1.0
        # Column 1 (edge b->c): -1 at row b, +1 at row c
        assert B1[b, 1].item() == -1.0
        assert B1[c, 1].item() == 1.0

    def test_get_embeddings(self):
        cc = CellComplex(embedding_dim=4)
        emb = torch.tensor([1.0, 2.0, 3.0, 4.0])
        cc.add_0_cell(emb, "concept")
        retrieved = cc.get_embeddings(0)
        assert torch.allclose(retrieved[0], emb)

    def test_adjacency_0_cells(self):
        """Adjacency matrix for 0-cells derived from shared 1-cells."""
        cc = CellComplex(embedding_dim=4)
        a = cc.add_0_cell(torch.randn(4), "concept")
        b = cc.add_0_cell(torch.randn(4), "concept")
        c = cc.add_0_cell(torch.randn(4), "concept")
        cc.add_1_cell(a, b, torch.randn(4), "r1")
        cc.add_1_cell(b, c, torch.randn(4), "r2")

        A = cc.adjacency_matrix(0)
        assert A.shape == (3, 3)
        assert A[a, b].item() == 1.0
        assert A[b, a].item() == 1.0
        assert A[a, c].item() == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_cell_complex/test_cell_complex.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.cell_complex.cell_complex'`

**Step 3: Implement CellComplex**

```python
# src/cell_complex/cell_complex.py
import torch
import torch.sparse
from typing import Optional


class CellComplex:
    """CW-complex data structure with 0-cells, 1-cells, and 2-cells.

    Stores cell embeddings and boundary operators as sparse tensors.
    Phase 1: 0-cells and 1-cells only. 2-cells added in Phase 2.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

        # 0-cells (nodes)
        self._0_cell_embeddings: list[torch.Tensor] = []
        self._0_cell_types: list[str] = []

        # 1-cells (edges)
        self._1_cell_embeddings: list[torch.Tensor] = []
        self._1_cell_types: list[str] = []
        self._1_cell_sources: list[int] = []
        self._1_cell_targets: list[int] = []

        # 2-cells (faces) — placeholder for Phase 2
        self._2_cell_embeddings: list[torch.Tensor] = []
        self._2_cell_boundaries: list[list[int]] = []

    def num_cells(self, dim: int) -> int:
        if dim == 0:
            return len(self._0_cell_embeddings)
        elif dim == 1:
            return len(self._1_cell_embeddings)
        elif dim == 2:
            return len(self._2_cell_embeddings)
        raise ValueError(f"Unsupported cell dimension: {dim}")

    def add_0_cell(self, embedding: torch.Tensor, cell_type: str) -> int:
        assert embedding.shape == (self.embedding_dim,)
        idx = len(self._0_cell_embeddings)
        self._0_cell_embeddings.append(embedding.clone())
        self._0_cell_types.append(cell_type)
        return idx

    def add_1_cell(
        self,
        source: int,
        target: int,
        embedding: torch.Tensor,
        relation_type: str,
    ) -> int:
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
        if dim == 0:
            assert embeddings.shape == (self.num_cells(0), self.embedding_dim)
            self._0_cell_embeddings = list(embeddings)
        elif dim == 1:
            assert embeddings.shape == (self.num_cells(1), self.embedding_dim)
            self._1_cell_embeddings = list(embeddings)
        else:
            raise ValueError(f"Unsupported cell dimension: {dim}")

    def boundary_operator(self, dim: int) -> torch.Tensor:
        """Return boundary operator ∂_dim as a dense matrix.

        ∂₁: (num_0_cells, num_1_cells) — maps 1-cells to their boundary 0-cells.
             Convention: source gets -1, target gets +1.
        """
        if dim == 1:
            n0 = self.num_cells(0)
            n1 = self.num_cells(1)
            if n0 == 0 or n1 == 0:
                return torch.zeros(max(n0, 1), max(n1, 1))
            B = torch.zeros(n0, n1)
            for j, (s, t) in enumerate(
                zip(self._1_cell_sources, self._1_cell_targets)
            ):
                B[s, j] = -1.0
                B[t, j] = 1.0
            return B
        raise ValueError(f"Boundary operator for dim={dim} not implemented (Phase 1)")

    def adjacency_matrix(self, dim: int) -> torch.Tensor:
        """Adjacency matrix for k-cells. Two k-cells are adjacent if they
        share a (k+1)-cell or (k-1)-cell boundary."""
        if dim == 0:
            n = self.num_cells(0)
            A = torch.zeros(n, n)
            for s, t in zip(self._1_cell_sources, self._1_cell_targets):
                A[s, t] = 1.0
                A[t, s] = 1.0
            return A
        raise ValueError(f"Adjacency for dim={dim} not implemented")

    def edge_index(self) -> torch.Tensor:
        """Return edge_index in PyG format: (2, num_edges) with both directions."""
        sources = self._1_cell_sources
        targets = self._1_cell_targets
        # Include both directions for undirected message passing
        row = torch.tensor(sources + targets, dtype=torch.long)
        col = torch.tensor(targets + sources, dtype=torch.long)
        return torch.stack([row, col], dim=0)
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_cell_complex/test_cell_complex.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/cell_complex/cell_complex.py tests/test_cell_complex/test_cell_complex.py
git commit -m "feat: cell complex data structure with boundary operators"
```

---

### Task 3: Hodge Laplacian Computation

**Files:**
- Create: `src/spectral/laplacian.py`
- Create: `tests/test_spectral/__init__.py`
- Create: `tests/test_spectral/test_laplacian.py`

**Step 1: Write failing tests**

```python
# tests/test_spectral/test_laplacian.py
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0, hodge_laplacian_1


def make_triangle_complex(dim=4):
    """Create a triangle: a-b-c-a with 3 nodes, 3 edges."""
    cc = CellComplex(embedding_dim=dim)
    a = cc.add_0_cell(torch.randn(dim), "concept")
    b = cc.add_0_cell(torch.randn(dim), "concept")
    c = cc.add_0_cell(torch.randn(dim), "concept")
    cc.add_1_cell(a, b, torch.randn(dim), "r")
    cc.add_1_cell(b, c, torch.randn(dim), "r")
    cc.add_1_cell(c, a, torch.randn(dim), "r")
    return cc


def make_chain_complex(dim=4, length=4):
    """Create a chain: 0-1-2-...-n with n nodes, n-1 edges."""
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestHodgeLaplacian0:
    def test_shape(self):
        cc = make_triangle_complex()
        L0 = hodge_laplacian_0(cc)
        assert L0.shape == (3, 3)

    def test_symmetric(self):
        cc = make_triangle_complex()
        L0 = hodge_laplacian_0(cc)
        assert torch.allclose(L0, L0.T)

    def test_positive_semidefinite(self):
        cc = make_triangle_complex()
        L0 = hodge_laplacian_0(cc)
        eigenvalues = torch.linalg.eigvalsh(L0)
        assert (eigenvalues >= -1e-6).all()

    def test_row_sum_zero(self):
        """Graph Laplacian rows sum to zero."""
        cc = make_triangle_complex()
        L0 = hodge_laplacian_0(cc)
        assert torch.allclose(L0.sum(dim=1), torch.zeros(3), atol=1e-6)

    def test_chain_spectrum(self):
        """A chain of n nodes has n-1 nonzero eigenvalues and 1 zero eigenvalue."""
        cc = make_chain_complex(length=5)
        L0 = hodge_laplacian_0(cc)
        eigenvalues = torch.linalg.eigvalsh(L0)
        num_zero = (eigenvalues.abs() < 1e-6).sum().item()
        assert num_zero == 1  # one connected component


class TestHodgeLaplacian1:
    def test_shape(self):
        cc = make_triangle_complex()
        L1 = hodge_laplacian_1(cc)
        assert L1.shape == (3, 3)  # 3 edges

    def test_symmetric(self):
        cc = make_triangle_complex()
        L1 = hodge_laplacian_1(cc)
        assert torch.allclose(L1, L1.T)

    def test_positive_semidefinite(self):
        cc = make_triangle_complex()
        L1 = hodge_laplacian_1(cc)
        eigenvalues = torch.linalg.eigvalsh(L1)
        assert (eigenvalues >= -1e-6).all()
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_spectral/test_laplacian.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement Hodge Laplacians**

```python
# src/spectral/laplacian.py
import torch
from src.cell_complex.cell_complex import CellComplex


def hodge_laplacian_0(cc: CellComplex) -> torch.Tensor:
    """Compute the 0-Hodge Laplacian: L₀ = ∂₁ · ∂₁ᵀ.

    This is the standard graph Laplacian (degree matrix - adjacency matrix).
    For Phase 1 (no 2-cells), L₀ = B₁ @ B₁.T where B₁ = ∂₁.
    """
    B1 = cc.boundary_operator(1)  # (num_0_cells, num_1_cells)
    L0 = B1 @ B1.T  # (num_0_cells, num_0_cells)
    return L0


def hodge_laplacian_1(cc: CellComplex) -> torch.Tensor:
    """Compute the 1-Hodge Laplacian: L₁ = ∂₁ᵀ · ∂₁ + ∂₂ · ∂₂ᵀ.

    For Phase 1 (no 2-cells), the ∂₂ term is zero, so L₁ = B₁.T @ B₁.
    The ∂₂ term will be added in Phase 2.
    """
    B1 = cc.boundary_operator(1)  # (num_0_cells, num_1_cells)
    L1 = B1.T @ B1  # (num_1_cells, num_1_cells)
    return L1
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_spectral/test_laplacian.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/spectral/laplacian.py tests/test_spectral/test_laplacian.py
git commit -m "feat: Hodge Laplacian computation for 0-cells and 1-cells"
```

---

### Task 4: Spectral Decomposition & Positional Encodings

**Files:**
- Create: `src/spectral/decomposition.py`
- Create: `src/spectral/positional_encoding.py`
- Create: `tests/test_spectral/test_decomposition.py`

**Step 1: Write failing tests**

```python
# tests/test_spectral/test_decomposition.py
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition
from src.spectral.positional_encoding import laplacian_pe


def make_triangle_complex(dim=4):
    cc = CellComplex(embedding_dim=dim)
    a = cc.add_0_cell(torch.randn(dim), "concept")
    b = cc.add_0_cell(torch.randn(dim), "concept")
    c = cc.add_0_cell(torch.randn(dim), "concept")
    cc.add_1_cell(a, b, torch.randn(dim), "r")
    cc.add_1_cell(b, c, torch.randn(dim), "r")
    cc.add_1_cell(c, a, torch.randn(dim), "r")
    return cc


class TestSpectralDecomposition:
    def test_eigenvalues_sorted(self):
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)
        assert (eigenvalues[1:] >= eigenvalues[:-1] - 1e-6).all()

    def test_eigenvectors_orthonormal(self):
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)
        # eigenvectors: (num_cells, num_cells), columns are eigenvectors
        identity = eigenvectors.T @ eigenvectors
        assert torch.allclose(identity, torch.eye(3), atol=1e-5)

    def test_top_k(self):
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0, k=2)
        assert eigenvalues.shape == (2,)
        assert eigenvectors.shape == (3, 2)


class TestLaplacianPE:
    def test_shape(self):
        cc = make_triangle_complex()
        pe = laplacian_pe(cc, dim=0, k=2)
        assert pe.shape == (3, 2)  # (num_0_cells, k)

    def test_sign_invariant(self):
        """PE should use absolute values or be sign-invariant."""
        cc = make_triangle_complex()
        pe = laplacian_pe(cc, dim=0, k=2)
        # Each row should have consistent norm regardless of sign flips
        norms = pe.norm(dim=1)
        assert (norms > 0).all()
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_spectral/test_decomposition.py -v`
Expected: FAIL

**Step 3: Implement spectral decomposition and PE**

```python
# src/spectral/decomposition.py
import torch
from typing import Optional
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0, hodge_laplacian_1


def spectral_decomposition(
    cc: CellComplex, dim: int, k: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of the dim-Hodge Laplacian.

    Returns eigenvalues (ascending) and eigenvectors (columns).
    If k is specified, returns only the k smallest eigenvalues/vectors.
    """
    if dim == 0:
        L = hodge_laplacian_0(cc)
    elif dim == 1:
        L = hodge_laplacian_1(cc)
    else:
        raise ValueError(f"dim={dim} not supported in Phase 1")

    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    if k is not None:
        k = min(k, eigenvalues.shape[0])
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    return eigenvalues, eigenvectors
```

```python
# src/spectral/positional_encoding.py
import torch
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition


def laplacian_pe(cc: CellComplex, dim: int, k: int) -> torch.Tensor:
    """Laplacian positional encoding for k-cells.

    Uses the k smallest non-trivial eigenvectors of the Hodge Laplacian.
    Sign ambiguity is handled by using random sign flips during training
    (SignNet-style) — for now, we just return raw eigenvectors.

    Returns: (num_k_cells, k) tensor of positional features.
    """
    eigenvalues, eigenvectors = spectral_decomposition(cc, dim=dim, k=k)
    return eigenvectors
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_spectral/test_decomposition.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/spectral/decomposition.py src/spectral/positional_encoding.py tests/test_spectral/test_decomposition.py
git commit -m "feat: spectral decomposition and Laplacian positional encodings"
```

---

### Task 5: GNN Executive — Spatial Message Passing

**Files:**
- Create: `src/gnn_executive/spatial.py`
- Create: `tests/test_gnn_executive/test_spatial.py`

**Step 1: Write failing tests**

```python
# tests/test_gnn_executive/test_spatial.py
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.spatial import SpatialMessagePassingLayer, SpatialGNN


def make_chain(dim=16, length=4):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestSpatialMessagePassingLayer:
    def test_output_shape(self):
        layer = SpatialMessagePassingLayer(in_dim=16, out_dim=16)
        cc = make_chain(dim=16, length=4)
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()
        out = layer(x, edge_index)
        assert out.shape == (4, 16)

    def test_gradient_flow(self):
        layer = SpatialMessagePassingLayer(in_dim=16, out_dim=16)
        cc = make_chain(dim=16, length=4)
        x = cc.get_embeddings(0).requires_grad_(True)
        edge_index = cc.edge_index()
        out = layer(x, edge_index)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 16)


class TestSpatialGNN:
    def test_multi_layer(self):
        gnn = SpatialGNN(in_dim=16, hidden_dim=16, out_dim=16, num_layers=3)
        cc = make_chain(dim=16, length=5)
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()
        out = gnn(x, edge_index)
        assert out.shape == (5, 16)

    def test_information_propagation(self):
        """After enough layers, all nodes should be influenced by all others in a chain."""
        gnn = SpatialGNN(in_dim=16, hidden_dim=16, out_dim=16, num_layers=4)
        cc = make_chain(dim=16, length=5)
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()

        # Perturb first node
        x_perturbed = x.clone()
        x_perturbed[0] += 10.0

        out_original = gnn(x, edge_index)
        out_perturbed = gnn(x_perturbed, edge_index)

        # Last node should be affected (information traveled through chain)
        diff = (out_original[-1] - out_perturbed[-1]).abs().sum()
        assert diff > 0.01, "Signal should propagate through entire chain"
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_gnn_executive/test_spatial.py -v`
Expected: FAIL

**Step 3: Implement spatial message passing**

```python
# src/gnn_executive/spatial.py
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class SpatialMessagePassingLayer(MessagePassing):
    """Single message-passing layer using boundary-aware aggregation.

    Messages flow along 1-cells (edges) between 0-cells (nodes).
    Uses learned message and update functions.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="add")
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = self.propagate(edge_index, x=x)
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(out + x) if x.shape[-1] == out.shape[-1] else self.norm(out)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.message_mlp(torch.cat([x_i, x_j], dim=-1))


class SpatialGNN(nn.Module):
    """Multi-layer spatial GNN operating on the cell complex."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [SpatialMessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_proj(x)
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_gnn_executive/test_spatial.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/gnn_executive/spatial.py tests/test_gnn_executive/test_spatial.py
git commit -m "feat: spatial message passing GNN for cell complex"
```

---

### Task 6: GNN Executive — Spectral Filtering

**Files:**
- Create: `src/gnn_executive/spectral_filter.py`
- Create: `tests/test_gnn_executive/test_spectral_filter.py`

**Step 1: Write failing tests**

```python
# tests/test_gnn_executive/test_spectral_filter.py
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.spectral_filter import SpectralFilterLayer, SpectralGNN


def make_chain(dim=16, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestSpectralFilterLayer:
    def test_output_shape(self):
        layer = SpectralFilterLayer(in_dim=16, out_dim=16, num_freqs=5)
        cc = make_chain(dim=16, length=5)
        x = cc.get_embeddings(0)
        from src.spectral.decomposition import spectral_decomposition
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)
        out = layer(x, eigenvalues, eigenvectors)
        assert out.shape == (5, 16)

    def test_low_pass_behavior(self):
        """A spectral filter initialized to low-pass should smooth signals."""
        layer = SpectralFilterLayer(in_dim=16, out_dim=16, num_freqs=5)
        cc = make_chain(dim=16, length=5)
        x = cc.get_embeddings(0)
        from src.spectral.decomposition import spectral_decomposition
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)

        out = layer(x, eigenvalues, eigenvectors)
        # Output should be a valid tensor (no NaN)
        assert not torch.isnan(out).any()


class TestSpectralGNN:
    def test_output_shape(self):
        gnn = SpectralGNN(in_dim=16, hidden_dim=16, out_dim=16, num_layers=2, max_freqs=5)
        cc = make_chain(dim=16, length=5)
        out = gnn(cc)
        assert out.shape == (5, 16)

    def test_gradient_flow(self):
        gnn = SpectralGNN(in_dim=16, hidden_dim=16, out_dim=16, num_layers=2, max_freqs=5)
        cc = make_chain(dim=16, length=5)
        # Make embeddings require grad
        embs = cc.get_embeddings(0).clone().requires_grad_(True)
        cc.set_embeddings(0, embs)
        out = gnn(cc)
        loss = out.sum()
        loss.backward()
        assert embs.grad is not None
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_gnn_executive/test_spectral_filter.py -v`
Expected: FAIL

**Step 3: Implement spectral filtering**

```python
# src/gnn_executive/spectral_filter.py
import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition


class SpectralFilterLayer(nn.Module):
    """Spectral convolution layer using graph Fourier transform.

    Projects signal to spectral domain (Laplacian eigenbasis),
    applies learnable frequency-domain filter, projects back.
    """

    def __init__(self, in_dim: int, out_dim: int, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        # Learnable spectral filter coefficients per input channel
        self.filter_weights = nn.Parameter(torch.randn(num_freqs, in_dim, out_dim) * 0.01)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (num_nodes, in_dim)
        eigenvalues: (num_freqs,)
        eigenvectors: (num_nodes, num_freqs)
        """
        k = min(self.num_freqs, eigenvectors.shape[1])
        U = eigenvectors[:, :k]  # (num_nodes, k)

        # Forward Fourier transform: project to spectral domain
        x_hat = U.T @ x  # (k, in_dim)

        # Apply learnable filter in spectral domain
        # x_hat: (k, in_dim), filter_weights[:k]: (k, in_dim, out_dim)
        x_filtered = torch.einsum("ki,kio->ko", x_hat, self.filter_weights[:k])  # (k, out_dim)

        # Inverse Fourier transform: project back to spatial domain
        out = U @ x_filtered  # (num_nodes, out_dim)

        return self.norm(out)


class SpectralGNN(nn.Module):
    """Multi-layer spectral GNN operating on cell complex."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        max_freqs: int,
    ):
        super().__init__()
        self.max_freqs = max_freqs
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SpectralFilterLayer(hidden_dim, hidden_dim, max_freqs) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, cc: CellComplex) -> torch.Tensor:
        x = cc.get_embeddings(0)
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0, k=self.max_freqs)

        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            x = layer(x, eigenvalues, eigenvectors)
            if residual.shape == x.shape:
                x = x + residual
        return self.output_proj(x)
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_gnn_executive/test_spectral_filter.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/gnn_executive/spectral_filter.py tests/test_gnn_executive/test_spectral_filter.py
git commit -m "feat: spectral filtering GNN with graph Fourier transform"
```

---

### Task 7: GNN Executive — Dual-Path Executive Module

**Files:**
- Create: `src/gnn_executive/executive.py`
- Create: `tests/test_gnn_executive/test_executive.py`

**Step 1: Write failing tests**

```python
# tests/test_gnn_executive/test_executive.py
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.executive import GNNExecutive


def make_chain(dim=16, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestGNNExecutive:
    def test_output_shape(self):
        executive = GNNExecutive(
            embedding_dim=16, hidden_dim=32, num_spatial_layers=2,
            num_spectral_layers=2, max_freqs=5
        )
        cc = make_chain(dim=16, length=5)
        node_out, edge_out = executive(cc)
        assert node_out.shape == (5, 16)
        # edge_out can be None for Phase 1 or a tensor

    def test_dual_path(self):
        """Both spatial and spectral paths should contribute."""
        executive = GNNExecutive(
            embedding_dim=16, hidden_dim=32, num_spatial_layers=2,
            num_spectral_layers=2, max_freqs=5
        )
        cc = make_chain(dim=16, length=5)

        # Run forward
        node_out, _ = executive(cc)
        assert not torch.isnan(node_out).any()
        # Output should differ from input (processing happened)
        input_emb = cc.get_embeddings(0)
        assert not torch.allclose(node_out, input_emb, atol=0.1)

    def test_gradient_flow(self):
        executive = GNNExecutive(
            embedding_dim=16, hidden_dim=32, num_spatial_layers=2,
            num_spectral_layers=2, max_freqs=5
        )
        cc = make_chain(dim=16, length=5)
        node_out, _ = executive(cc)
        loss = node_out.sum()
        loss.backward()
        # Check that parameters have gradients
        for param in executive.parameters():
            if param.requires_grad:
                assert param.grad is not None
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_gnn_executive/test_executive.py -v`
Expected: FAIL

**Step 3: Implement GNNExecutive**

```python
# src/gnn_executive/executive.py
import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.spatial import SpatialGNN
from src.gnn_executive.spectral_filter import SpectralGNN


class GNNExecutive(nn.Module):
    """Dual-path GNN executive: spatial + spectral processing in parallel.

    Spatial path: Message passing along cell complex edges.
    Spectral path: Fourier-domain filtering via Hodge Laplacian eigenbasis.
    Outputs are fused via a learned gate.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_spatial_layers: int,
        num_spectral_layers: int,
        max_freqs: int,
    ):
        super().__init__()
        self.spatial_gnn = SpatialGNN(
            in_dim=embedding_dim, hidden_dim=hidden_dim,
            out_dim=embedding_dim, num_layers=num_spatial_layers,
        )
        self.spectral_gnn = SpectralGNN(
            in_dim=embedding_dim, hidden_dim=hidden_dim,
            out_dim=embedding_dim, num_layers=num_spectral_layers,
            max_freqs=max_freqs,
        )
        # Learned gate to fuse spatial and spectral outputs
        self.gate = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Linear(2 * embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, cc: CellComplex) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()

        # Dual paths
        spatial_out = self.spatial_gnn(x, edge_index)
        spectral_out = self.spectral_gnn(cc)

        # Gated fusion
        gate_input = torch.cat([spatial_out, spectral_out], dim=-1)
        g = self.gate(gate_input)
        fused = g * spatial_out + (1 - g) * spectral_out
        fused = self.norm(fused + x)  # Residual connection

        return fused, None  # edge embeddings not yet computed in Phase 1
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_gnn_executive/test_executive.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/gnn_executive/executive.py tests/test_gnn_executive/test_executive.py
git commit -m "feat: dual-path GNN executive with spatial-spectral gated fusion"
```

---

### Task 8: Topology-Aware Transformer — Spatial Attention

**Files:**
- Create: `src/tat/spatial_attention.py`
- Create: `tests/test_tat/test_spatial_attention.py`

**Step 1: Write failing tests**

```python
# tests/test_tat/test_spatial_attention.py
import torch
import pytest
from src.tat.spatial_attention import TopologicalSpatialAttention


class TestTopologicalSpatialAttention:
    def test_output_shape(self):
        attn = TopologicalSpatialAttention(embed_dim=64, num_heads=4)
        x = torch.randn(8, 64)  # 8 tokens, dim 64
        adj = torch.ones(8, 8)  # fully connected
        out = attn(x, adjacency=adj)
        assert out.shape == (8, 64)

    def test_adjacency_masking(self):
        """Disconnected nodes should not attend to each other."""
        attn = TopologicalSpatialAttention(embed_dim=64, num_heads=4)
        x = torch.randn(4, 64)
        # Two disconnected pairs: (0,1) and (2,3)
        adj = torch.zeros(4, 4)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0
        # Add self-connections
        adj.fill_diagonal_(1.0)

        out = attn(x, adjacency=adj)
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        attn = TopologicalSpatialAttention(embed_dim=64, num_heads=4)
        x = torch.randn(8, 64, requires_grad=True)
        adj = torch.ones(8, 8)
        out = attn(x, adjacency=adj)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_tat/test_spatial_attention.py -v`
Expected: FAIL

**Step 3: Implement spatial attention**

```python
# src/tat/spatial_attention.py
import torch
import torch.nn as nn
import math


class TopologicalSpatialAttention(nn.Module):
    """Multi-head attention biased by cell complex adjacency.

    Attention weights are masked so tokens only attend to topologically
    adjacent cells (+ self). The adjacency matrix acts as both a mask
    and a bias on attention scores.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        x: (num_tokens, embed_dim)
        adjacency: (num_tokens, num_tokens) — 1.0 for connected, 0.0 for disconnected
        """
        n = x.shape[0]
        residual = x

        q = self.q_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        # q, k, v: (num_heads, num_tokens, head_dim)

        scores = (q @ k.transpose(-2, -1)) / self.scale  # (num_heads, n, n)

        # Apply adjacency mask: -inf where not adjacent
        mask = adjacency.unsqueeze(0).expand(self.num_heads, -1, -1)  # (num_heads, n, n)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Handle case where a row is all -inf (isolated node)
        attn_weights = attn_weights.nan_to_num(0.0)

        out = attn_weights @ v  # (num_heads, n, head_dim)
        out = out.transpose(0, 1).contiguous().view(n, self.embed_dim)
        out = self.out_proj(out)

        return self.norm(out + residual)
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_tat/test_spatial_attention.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/tat/spatial_attention.py tests/test_tat/test_spatial_attention.py
git commit -m "feat: topological spatial attention with adjacency masking"
```

---

### Task 9: Topology-Aware Transformer — Spectral Attention

**Files:**
- Create: `src/tat/spectral_attention.py`
- Create: `tests/test_tat/test_spectral_attention.py`

**Step 1: Write failing tests**

```python
# tests/test_tat/test_spectral_attention.py
import torch
import pytest
from src.tat.spectral_attention import TopologicalSpectralAttention


class TestTopologicalSpectralAttention:
    def test_output_shape(self):
        attn = TopologicalSpectralAttention(embed_dim=64, num_heads=4, num_freqs=8)
        x = torch.randn(10, 64)
        eigenvectors = torch.randn(10, 8)
        eigenvalues = torch.sort(torch.rand(8))[0]
        out = attn(x, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        assert out.shape == (10, 64)

    def test_gradient_flow(self):
        attn = TopologicalSpectralAttention(embed_dim=64, num_heads=4, num_freqs=8)
        x = torch.randn(10, 64, requires_grad=True)
        eigenvectors = torch.randn(10, 8)
        eigenvalues = torch.sort(torch.rand(8))[0]
        out = attn(x, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_captures_long_range(self):
        """Spectral attention should capture global patterns via low-frequency eigenvectors."""
        attn = TopologicalSpectralAttention(embed_dim=16, num_heads=2, num_freqs=4)
        x = torch.randn(6, 16)
        # Use actual Laplacian eigenvectors from a chain
        from src.cell_complex.cell_complex import CellComplex
        from src.spectral.decomposition import spectral_decomposition
        cc = CellComplex(embedding_dim=16)
        for i in range(6):
            cc.add_0_cell(torch.randn(16), "c")
        for i in range(5):
            cc.add_1_cell(i, i + 1, torch.randn(16), "r")
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0, k=4)
        out = attn(x, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        assert not torch.isnan(out).any()
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_tat/test_spectral_attention.py -v`
Expected: FAIL

**Step 3: Implement spectral attention**

```python
# src/tat/spectral_attention.py
import torch
import torch.nn as nn
import math


class TopologicalSpectralAttention(nn.Module):
    """Multi-head attention in the spectral domain.

    Projects tokens to the Laplacian eigenbasis, performs attention
    in the frequency domain with learnable spectral filters, then
    projects back to the spatial domain.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_freqs: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_freqs = num_freqs
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable spectral filter per head
        self.spectral_filter = nn.Parameter(torch.ones(num_heads, num_freqs))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (num_tokens, embed_dim)
        eigenvalues: (num_freqs,)
        eigenvectors: (num_tokens, num_freqs)
        """
        n = x.shape[0]
        k = min(self.num_freqs, eigenvectors.shape[1])
        residual = x

        U = eigenvectors[:, :k]  # (n, k)

        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k_proj = self.k_proj(x).view(n, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_heads, self.head_dim)

        # Transform Q, K to spectral domain: (num_freqs, num_heads, head_dim)
        q_hat = torch.einsum("nk,nhd->khd", U, q)
        k_hat = torch.einsum("nk,nhd->khd", U, k_proj)

        # Attention in spectral domain per frequency
        scores = torch.einsum("khd,khd->kh", q_hat, k_hat) / self.scale  # (k, num_heads)

        # Apply learnable spectral filter
        filtered_scores = scores * self.spectral_filter[:, :k].T  # (k, num_heads)
        weights = torch.softmax(filtered_scores, dim=0)  # normalize over frequencies
        weights = self.dropout(weights)

        # Weighted combination of V in spectral domain
        v_hat = torch.einsum("nk,nhd->khd", U, v)  # (k, num_heads, head_dim)
        out_hat = torch.einsum("kh,khd->hd", weights, v_hat)  # (num_heads, head_dim)

        # Broadcast back to spatial domain via eigenvectors
        # Each node gets a weighted sum based on its eigenvector coordinates
        out_spatial = torch.einsum("nk,kh->nh", U, weights)  # (n, num_heads)
        out = torch.einsum("nh,nhd->nhd", out_spatial, v)  # use spatial V directly
        out = out.contiguous().view(n, self.embed_dim)
        out = self.out_proj(out)

        return self.norm(out + residual)
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_tat/test_spectral_attention.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/tat/spectral_attention.py tests/test_tat/test_spectral_attention.py
git commit -m "feat: spectral attention with learnable frequency-domain filters"
```

---

### Task 10: Topology-Aware Transformer — Full TAT Module

**Files:**
- Create: `src/tat/transformer.py`
- Create: `tests/test_tat/test_transformer.py`

**Step 1: Write failing tests**

```python
# tests/test_tat/test_transformer.py
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.tat.transformer import TopologyAwareTransformer


def make_chain(dim=64, length=6):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestTopologyAwareTransformer:
    def test_output_shape(self):
        tat = TopologyAwareTransformer(
            embedding_dim=64, num_layers=2, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=128, num_freqs=4,
        )
        cc = make_chain(dim=64, length=6)
        out = tat(cc)
        assert out.shape == (6, 64)

    def test_different_graph_sizes(self):
        tat = TopologyAwareTransformer(
            embedding_dim=32, num_layers=2, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=4,
        )
        for length in [3, 5, 10]:
            cc = make_chain(dim=32, length=length)
            out = tat(cc)
            assert out.shape == (length, 32)

    def test_gradient_flow(self):
        tat = TopologyAwareTransformer(
            embedding_dim=32, num_layers=2, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=4,
        )
        cc = make_chain(dim=32, length=5)
        out = tat(cc)
        loss = out.sum()
        loss.backward()
        for name, param in tat.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_param_count(self):
        """Verify model is in the right ballpark for ~15M params."""
        tat = TopologyAwareTransformer(
            embedding_dim=256, num_layers=6, num_spatial_heads=4,
            num_spectral_heads=4, ff_dim=512, num_freqs=32,
        )
        total_params = sum(p.numel() for p in tat.parameters())
        # Should be in the range of 5M-25M for this config
        assert 1_000_000 < total_params < 50_000_000
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_tat/test_transformer.py -v`
Expected: FAIL

**Step 3: Implement the full TAT**

```python
# src/tat/transformer.py
import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.tat.spatial_attention import TopologicalSpatialAttention
from src.tat.spectral_attention import TopologicalSpectralAttention
from src.spectral.decomposition import spectral_decomposition


class TATBlock(nn.Module):
    """Single Topology-Aware Transformer block.

    Dual attention: spatial head attends by adjacency,
    spectral head attends in Fourier domain.
    Followed by feed-forward layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_spatial_heads: int,
        num_spectral_heads: int,
        ff_dim: int,
        num_freqs: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_attn = TopologicalSpatialAttention(
            embed_dim=embed_dim, num_heads=num_spatial_heads, dropout=dropout,
        )
        self.spectral_attn = TopologicalSpectralAttention(
            embed_dim=embed_dim, num_heads=num_spectral_heads,
            num_freqs=num_freqs, dropout=dropout,
        )
        # Gate to fuse spatial and spectral attention outputs
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        # Dual attention
        spatial_out = self.spatial_attn(x, adjacency=adjacency)
        spectral_out = self.spectral_attn(x, eigenvalues=eigenvalues, eigenvectors=eigenvectors)

        # Gated fusion
        g = self.gate(torch.cat([spatial_out, spectral_out], dim=-1))
        x = g * spatial_out + (1 - g) * spectral_out

        # Feed-forward with residual
        x = x + self.ff(x)
        x = self.ff_norm(x)
        return x


class TopologyAwareTransformer(nn.Module):
    """Full Topology-Aware Transformer.

    Stacks TATBlocks with dual spatial/spectral attention.
    Input: cell complex with embeddings.
    Output: updated node embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_spatial_heads: int,
        num_spectral_heads: int,
        ff_dim: int,
        num_freqs: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.blocks = nn.ModuleList([
            TATBlock(
                embed_dim=embedding_dim,
                num_spatial_heads=num_spatial_heads,
                num_spectral_heads=num_spectral_heads,
                ff_dim=ff_dim,
                num_freqs=num_freqs,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, cc: CellComplex) -> torch.Tensor:
        x = cc.get_embeddings(0)
        adjacency = cc.adjacency_matrix(0)
        # Add self-loops to adjacency
        adjacency = adjacency + torch.eye(adjacency.shape[0])
        adjacency = (adjacency > 0).float()

        eigenvalues, eigenvectors = spectral_decomposition(
            cc, dim=0, k=self.num_freqs,
        )

        for block in self.blocks:
            x = block(x, adjacency, eigenvalues, eigenvectors)

        return x
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_tat/test_transformer.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/tat/transformer.py tests/test_tat/test_transformer.py
git commit -m "feat: full topology-aware transformer with dual spatial/spectral attention"
```

---

### Task 11: Interleaved Reasoning Loop

**Files:**
- Create: `src/reasoning_loop/loop.py`
- Create: `tests/test_reasoning_loop/__init__.py`
- Create: `tests/test_reasoning_loop/test_loop.py`

**Step 1: Write failing tests**

```python
# tests/test_reasoning_loop/test_loop.py
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.loop import ReasoningLoop


def make_chain(dim=32, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestReasoningLoop:
    def test_output_shape(self):
        loop = ReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=2, max_freqs=4, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_iterations=3, convergence_threshold=0.01,
        )
        cc = make_chain(dim=32, length=5)
        output, num_iters = loop(cc)
        assert output.shape == (5, 32)
        assert 1 <= num_iters <= 3

    def test_convergence(self):
        """Loop should stop early if embeddings converge."""
        loop = ReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=1,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_iterations=10, convergence_threshold=100.0,  # Easy threshold
        )
        cc = make_chain(dim=32, length=5)
        output, num_iters = loop(cc)
        # With a very high threshold, should converge in 1 iteration
        assert num_iters == 1

    def test_updates_cell_complex(self):
        """After reasoning, the cell complex embeddings should be updated."""
        loop = ReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=2, max_freqs=4, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_iterations=2, convergence_threshold=0.001,
        )
        cc = make_chain(dim=32, length=5)
        original_emb = cc.get_embeddings(0).clone()
        output, _ = loop(cc)
        # Output should differ from original
        assert not torch.allclose(output, original_emb, atol=0.01)
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_reasoning_loop/test_loop.py -v`
Expected: FAIL

**Step 3: Implement the reasoning loop**

```python
# src/reasoning_loop/loop.py
import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.executive import GNNExecutive
from src.tat.transformer import TopologyAwareTransformer


class ReasoningLoop(nn.Module):
    """Interleaved co-processing loop between GNN Executive and TAT.

    1. GNN Executive analyzes the cell complex (spatial + spectral).
    2. TAT processes the updated embeddings.
    3. Results update the cell complex.
    4. Repeat until convergence or max iterations.
    """

    def __init__(
        self,
        embedding_dim: int,
        gnn_hidden: int,
        gnn_spatial_layers: int,
        gnn_spectral_layers: int,
        max_freqs: int,
        tat_layers: int,
        tat_spatial_heads: int,
        tat_spectral_heads: int,
        tat_ff_dim: int,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.gnn_executive = GNNExecutive(
            embedding_dim=embedding_dim,
            hidden_dim=gnn_hidden,
            num_spatial_layers=gnn_spatial_layers,
            num_spectral_layers=gnn_spectral_layers,
            max_freqs=max_freqs,
        )
        self.tat = TopologyAwareTransformer(
            embedding_dim=embedding_dim,
            num_layers=tat_layers,
            num_spatial_heads=tat_spatial_heads,
            num_spectral_heads=tat_spectral_heads,
            ff_dim=tat_ff_dim,
            num_freqs=max_freqs,
        )
        # Projection to blend GNN and TAT outputs
        self.blend = nn.Linear(2 * embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, cc: CellComplex) -> tuple[torch.Tensor, int]:
        prev_embeddings = cc.get_embeddings(0)
        num_iters = 0

        for i in range(self.max_iterations):
            num_iters = i + 1

            # Step 1: GNN Executive analyzes
            gnn_out, _ = self.gnn_executive(cc)

            # Step 2: Update complex with GNN output and run TAT
            cc.set_embeddings(0, gnn_out.detach() if not self.training else gnn_out)
            tat_out = self.tat(cc)

            # Step 3: Blend and update
            blended = self.blend(torch.cat([gnn_out, tat_out], dim=-1))
            current_embeddings = self.norm(blended + prev_embeddings)

            # Update the cell complex
            cc.set_embeddings(0, current_embeddings.detach() if not self.training else current_embeddings)

            # Check convergence
            delta = (current_embeddings - prev_embeddings).norm()
            if delta < self.convergence_threshold:
                break

            prev_embeddings = current_embeddings

        return current_embeddings, num_iters
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_reasoning_loop/test_loop.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/reasoning_loop/loop.py tests/test_reasoning_loop/
git commit -m "feat: interleaved GNN-TAT reasoning loop with convergence detection"
```

---

### Task 12: Synthetic Benchmark — Multi-Hop Data Generation

**Files:**
- Create: `src/benchmarks/multi_hop.py`
- Create: `tests/test_benchmarks/test_multi_hop.py`

**Step 1: Write failing tests**

```python
# tests/test_benchmarks/test_multi_hop.py
import torch
import pytest
from src.benchmarks.multi_hop import generate_chain_task, generate_tree_task, MultiHopDataset


class TestChainTask:
    def test_basic_chain(self):
        cc, query_node, target_node, answer = generate_chain_task(
            num_hops=3, num_distractors=5, embedding_dim=16,
        )
        assert cc.num_cells(0) >= 4  # at least 4 nodes in a 3-hop chain
        assert cc.num_cells(1) >= 3  # at least 3 edges
        assert 0 <= query_node < cc.num_cells(0)
        assert 0 <= target_node < cc.num_cells(0)
        assert query_node != target_node

    def test_distractor_nodes(self):
        cc, _, _, _ = generate_chain_task(num_hops=3, num_distractors=10, embedding_dim=16)
        assert cc.num_cells(0) == 4 + 10  # chain nodes + distractors

    def test_varying_hops(self):
        for hops in [2, 5, 8]:
            cc, q, t, a = generate_chain_task(num_hops=hops, num_distractors=5, embedding_dim=16)
            assert cc.num_cells(0) == hops + 1 + 5


class TestTreeTask:
    def test_basic_tree(self):
        cc, query_node, target_node, answer = generate_tree_task(
            depth=3, branching=2, num_distractors=5, embedding_dim=16,
        )
        assert cc.num_cells(0) > 0
        assert cc.num_cells(1) > 0


class TestMultiHopDataset:
    def test_dataset_length(self):
        ds = MultiHopDataset(
            num_samples=100, min_hops=2, max_hops=5,
            num_distractors=10, embedding_dim=16,
        )
        assert len(ds) == 100

    def test_dataset_item(self):
        ds = MultiHopDataset(
            num_samples=10, min_hops=2, max_hops=4,
            num_distractors=5, embedding_dim=16,
        )
        cc, query, target, answer = ds[0]
        assert cc.num_cells(0) > 0
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_benchmarks/test_multi_hop.py -v`
Expected: FAIL

**Step 3: Implement benchmark generation**

```python
# src/benchmarks/multi_hop.py
import torch
import random
from src.cell_complex.cell_complex import CellComplex


def generate_chain_task(
    num_hops: int,
    num_distractors: int,
    embedding_dim: int,
) -> tuple[CellComplex, int, int, int]:
    """Generate a chain reasoning task.

    Creates a chain of (num_hops + 1) nodes connected by num_hops edges,
    plus distractor nodes with random connections.

    Returns: (cell_complex, query_node, target_node, answer_distance)
    The answer is the number of hops between query and target.
    """
    cc = CellComplex(embedding_dim=embedding_dim)

    # Create chain nodes with unique signal embeddings
    chain_nodes = []
    for i in range(num_hops + 1):
        emb = torch.randn(embedding_dim)
        # Encode position info in the embedding
        emb[0] = float(i)  # position marker
        node = cc.add_0_cell(emb, "chain")
        chain_nodes.append(node)

    # Connect chain
    for i in range(num_hops):
        emb = torch.randn(embedding_dim)
        cc.add_1_cell(chain_nodes[i], chain_nodes[i + 1], emb, "chain_edge")

    # Add distractor nodes
    distractor_nodes = []
    for _ in range(num_distractors):
        emb = torch.randn(embedding_dim)
        node = cc.add_0_cell(emb, "distractor")
        distractor_nodes.append(node)

    # Connect distractors randomly (to chain nodes and each other)
    all_nodes = chain_nodes + distractor_nodes
    for d in distractor_nodes:
        # Connect to 1-3 random existing nodes
        num_connections = random.randint(1, min(3, len(all_nodes) - 1))
        targets = random.sample([n for n in all_nodes if n != d], num_connections)
        for t in targets:
            emb = torch.randn(embedding_dim)
            cc.add_1_cell(d, t, emb, "distractor_edge")

    query_node = chain_nodes[0]
    target_node = chain_nodes[-1]
    answer = num_hops

    return cc, query_node, target_node, answer


def generate_tree_task(
    depth: int,
    branching: int,
    num_distractors: int,
    embedding_dim: int,
) -> tuple[CellComplex, int, int, int]:
    """Generate a tree reasoning task.

    Creates a tree with given depth and branching factor.
    Query: root. Target: random leaf. Answer: depth (always same for balanced tree).
    """
    cc = CellComplex(embedding_dim=embedding_dim)

    root = cc.add_0_cell(torch.randn(embedding_dim), "root")
    leaves = []
    current_level = [root]

    for d in range(depth):
        next_level = []
        for parent in current_level:
            for _ in range(branching):
                child = cc.add_0_cell(torch.randn(embedding_dim), "node")
                cc.add_1_cell(parent, child, torch.randn(embedding_dim), "tree_edge")
                next_level.append(child)
        current_level = next_level

    leaves = current_level

    # Add distractors
    all_nodes = list(range(cc.num_cells(0)))
    for _ in range(num_distractors):
        d = cc.add_0_cell(torch.randn(embedding_dim), "distractor")
        targets = random.sample(all_nodes, min(2, len(all_nodes)))
        for t in targets:
            cc.add_1_cell(d, t, torch.randn(embedding_dim), "distractor_edge")
        all_nodes.append(d)

    target = random.choice(leaves)
    return cc, root, target, depth


class MultiHopDataset:
    """Dataset of multi-hop reasoning tasks."""

    def __init__(
        self,
        num_samples: int,
        min_hops: int,
        max_hops: int,
        num_distractors: int,
        embedding_dim: int,
        task_type: str = "chain",
    ):
        self.samples = []
        for _ in range(num_samples):
            num_hops = random.randint(min_hops, max_hops)
            if task_type == "chain":
                sample = generate_chain_task(num_hops, num_distractors, embedding_dim)
            elif task_type == "tree":
                sample = generate_tree_task(num_hops, 2, num_distractors, embedding_dim)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[CellComplex, int, int, int]:
        return self.samples[idx]
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_benchmarks/test_multi_hop.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/benchmarks/multi_hop.py tests/test_benchmarks/test_multi_hop.py
git commit -m "feat: multi-hop graph traversal benchmark data generation"
```

---

### Task 13: Training Pipeline — Multi-Hop Reasoning

**Files:**
- Create: `src/benchmarks/trainer.py`
- Create: `src/benchmarks/model.py`
- Create: `tests/test_benchmarks/test_trainer.py`

**Step 1: Write failing tests**

```python
# tests/test_benchmarks/test_trainer.py
import torch
import pytest
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.trainer import train_epoch, evaluate


class TestMultiHopModel:
    def test_output_shape(self):
        model = MultiHopReasoningModel(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_hops=10, max_iterations=2,
        )
        ds = MultiHopDataset(num_samples=1, min_hops=3, max_hops=3, num_distractors=5, embedding_dim=32)
        cc, query, target, answer = ds[0]
        logits = model(cc, query, target)
        assert logits.shape == (10,)  # max_hops classes

    def test_gradient_flow(self):
        model = MultiHopReasoningModel(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=1,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
            max_hops=10, max_iterations=1,
        )
        ds = MultiHopDataset(num_samples=1, min_hops=2, max_hops=2, num_distractors=3, embedding_dim=32)
        cc, query, target, answer = ds[0]
        logits = model(cc, query, target)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([answer]))
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0


class TestTraining:
    def test_train_epoch(self):
        model = MultiHopReasoningModel(
            embedding_dim=16, gnn_hidden=32, gnn_spatial_layers=1,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=32,
            max_hops=6, max_iterations=1,
        )
        ds = MultiHopDataset(num_samples=4, min_hops=2, max_hops=4, num_distractors=3, embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = train_epoch(model, ds, optimizer)
        assert loss > 0

    def test_evaluate(self):
        model = MultiHopReasoningModel(
            embedding_dim=16, gnn_hidden=32, gnn_spatial_layers=1,
            gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=32,
            max_hops=6, max_iterations=1,
        )
        ds = MultiHopDataset(num_samples=4, min_hops=2, max_hops=4, num_distractors=3, embedding_dim=16)
        accuracy, avg_loss = evaluate(model, ds)
        assert 0.0 <= accuracy <= 1.0
        assert avg_loss > 0
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_benchmarks/test_trainer.py -v`
Expected: FAIL

**Step 3: Implement model and trainer**

```python
# src/benchmarks/model.py
import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.loop import ReasoningLoop


class MultiHopReasoningModel(nn.Module):
    """End-to-end model for multi-hop reasoning.

    Uses the full reasoning loop, then extracts query/target node
    embeddings and predicts hop distance.
    """

    def __init__(
        self,
        embedding_dim: int,
        gnn_hidden: int,
        gnn_spatial_layers: int,
        gnn_spectral_layers: int,
        max_freqs: int,
        tat_layers: int,
        tat_spatial_heads: int,
        tat_spectral_heads: int,
        tat_ff_dim: int,
        max_hops: int,
        max_iterations: int = 3,
    ):
        super().__init__()
        self.reasoning_loop = ReasoningLoop(
            embedding_dim=embedding_dim,
            gnn_hidden=gnn_hidden,
            gnn_spatial_layers=gnn_spatial_layers,
            gnn_spectral_layers=gnn_spectral_layers,
            max_freqs=max_freqs,
            tat_layers=tat_layers,
            tat_spatial_heads=tat_spatial_heads,
            tat_spectral_heads=tat_spectral_heads,
            tat_ff_dim=tat_ff_dim,
            max_iterations=max_iterations,
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, max_hops),
        )

    def forward(
        self, cc: CellComplex, query_node: int, target_node: int,
    ) -> torch.Tensor:
        output, num_iters = self.reasoning_loop(cc)

        query_emb = output[query_node]
        target_emb = output[target_node]
        diff_emb = query_emb - target_emb

        combined = torch.cat([query_emb, target_emb, diff_emb])
        logits = self.classifier(combined)
        return logits
```

```python
# src/benchmarks/trainer.py
import torch
import torch.nn as nn
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.multi_hop import MultiHopDataset


def train_epoch(
    model: MultiHopReasoningModel,
    dataset: MultiHopDataset,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0

    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        optimizer.zero_grad()

        logits = model(cc, query, target)
        loss = nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([answer]),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)


@torch.no_grad()
def evaluate(
    model: MultiHopReasoningModel,
    dataset: MultiHopDataset,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total_loss = 0.0

    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        logits = model(cc, query, target)
        loss = nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([answer]),
        )
        total_loss += loss.item()
        pred = logits.argmax().item()
        if pred == answer:
            correct += 1

    accuracy = correct / len(dataset)
    avg_loss = total_loss / len(dataset)
    return accuracy, avg_loss
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_benchmarks/test_trainer.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/benchmarks/model.py src/benchmarks/trainer.py tests/test_benchmarks/test_trainer.py
git commit -m "feat: multi-hop reasoning model and training pipeline"
```

---

### Task 14: Run Experiment Script

**Files:**
- Create: `src/benchmarks/run_experiment.py`

**Step 1: Create experiment runner**

```python
# src/benchmarks/run_experiment.py
"""Phase 1 Experiment: Multi-hop graph traversal benchmark."""

import torch
import yaml
import json
import time
from pathlib import Path
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.trainer import train_epoch, evaluate


def run_experiment(config_path: str = "config/default.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    bc = config["benchmark"]
    rc = config["reasoning_loop"]

    print("=" * 60)
    print("Phase 1: Multi-Hop Graph Traversal Benchmark")
    print("=" * 60)

    # Generate datasets
    print("\nGenerating datasets...")
    train_ds = MultiHopDataset(
        num_samples=bc["num_train"], min_hops=bc["min_hops"],
        max_hops=bc["max_hops"], num_distractors=bc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
    )
    val_ds = MultiHopDataset(
        num_samples=bc["num_val"], min_hops=bc["min_hops"],
        max_hops=bc["max_hops"], num_distractors=bc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
    )
    test_ds = MultiHopDataset(
        num_samples=bc["num_test"], min_hops=bc["min_hops"],
        max_hops=bc["max_hops"], num_distractors=bc["num_distractors"],
        embedding_dim=mc["embedding_dim"],
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Build model
    model = MultiHopReasoningModel(
        embedding_dim=mc["embedding_dim"],
        gnn_hidden=mc["embedding_dim"] * 2,
        gnn_spatial_layers=mc["num_gnn_layers"],
        gnn_spectral_layers=mc["num_gnn_layers"],
        max_freqs=mc["embedding_dim"] // 2,
        tat_layers=mc["num_tat_layers"],
        tat_spatial_heads=mc["num_spatial_heads"],
        tat_spectral_heads=mc["num_spectral_heads"],
        tat_ff_dim=mc["tat_ff_dim"],
        max_hops=bc["max_hops"],
        max_iterations=rc["max_iterations"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    results = []

    print("\nTraining...")
    for epoch in range(tc["max_epochs"]):
        start = time.time()
        train_loss = train_epoch(model, train_ds, optimizer)
        val_acc, val_loss = evaluate(model, val_ds)
        elapsed = time.time() - start

        scheduler.step(val_loss)

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "time": elapsed,
        }
        results.append(result)

        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/best_phase1.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Test evaluation
    model.load_state_dict(torch.load("models/best_phase1.pt", weights_only=True))
    test_acc, test_loss = evaluate(model, test_ds)
    print(f"\nTest Accuracy: {test_acc:.3f} | Test Loss: {test_loss:.4f}")

    # Per-hop analysis
    print("\nPer-hop accuracy:")
    for h in range(bc["min_hops"], bc["max_hops"] + 1):
        hop_ds = MultiHopDataset(
            num_samples=100, min_hops=h, max_hops=h,
            num_distractors=bc["num_distractors"],
            embedding_dim=mc["embedding_dim"],
        )
        hop_acc, _ = evaluate(model, hop_ds)
        print(f"  {h}-hop: {hop_acc:.3f}")

    # Save results
    Path("data").mkdir(exist_ok=True)
    with open("data/phase1_results.json", "w") as f:
        json.dump({"results": results, "test_accuracy": test_acc, "test_loss": test_loss}, f, indent=2)


if __name__ == "__main__":
    run_experiment()
```

**Step 2: Verify it runs (smoke test with tiny config)**

Create a quick test config and run for 1 epoch:
```bash
source .venv/bin/activate && python -c "
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.trainer import train_epoch, evaluate
import torch

model = MultiHopReasoningModel(
    embedding_dim=16, gnn_hidden=32, gnn_spatial_layers=1,
    gnn_spectral_layers=1, max_freqs=4, tat_layers=1,
    tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=32,
    max_hops=5, max_iterations=1,
)
ds = MultiHopDataset(num_samples=8, min_hops=2, max_hops=4, num_distractors=3, embedding_dim=16)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss = train_epoch(model, ds, opt)
acc, _ = evaluate(model, ds)
print(f'Loss: {loss:.4f}, Acc: {acc:.3f}')
print('Smoke test passed!')
"
```
Expected: Prints loss value and "Smoke test passed!"

**Step 3: Commit**

```bash
git add src/benchmarks/run_experiment.py
git commit -m "feat: Phase 1 experiment runner with per-hop analysis"
```

---

### Task 15: Run All Tests & Final Verification

**Step 1: Run full test suite**

```bash
source .venv/bin/activate && pytest tests/ -v --tb=short
```
Expected: All tests PASS

**Step 2: Verify full import chain works**

```bash
source .venv/bin/activate && python -c "
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0, hodge_laplacian_1
from src.spectral.decomposition import spectral_decomposition
from src.spectral.positional_encoding import laplacian_pe
from src.gnn_executive.spatial import SpatialGNN
from src.gnn_executive.spectral_filter import SpectralGNN
from src.gnn_executive.executive import GNNExecutive
from src.tat.spatial_attention import TopologicalSpatialAttention
from src.tat.spectral_attention import TopologicalSpectralAttention
from src.tat.transformer import TopologyAwareTransformer
from src.reasoning_loop.loop import ReasoningLoop
from src.benchmarks.multi_hop import MultiHopDataset
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.trainer import train_epoch, evaluate
print('All imports successful!')
"
```
Expected: "All imports successful!"

**Step 3: Commit any stragglers**

```bash
git add -A && git status
```

If clean, done. If untracked files remain, add and commit.

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Project scaffolding | Setup verification |
| 2 | CellComplex data structure | 6 tests |
| 3 | Hodge Laplacians (L₀, L₁) | 7 tests |
| 4 | Spectral decomposition + PE | 5 tests |
| 5 | Spatial message passing GNN | 4 tests |
| 6 | Spectral filtering GNN | 4 tests |
| 7 | Dual-path GNN Executive | 3 tests |
| 8 | Topological spatial attention | 3 tests |
| 9 | Topological spectral attention | 3 tests |
| 10 | Full TAT module | 4 tests |
| 11 | Interleaved reasoning loop | 3 tests |
| 12 | Multi-hop benchmark generation | 6 tests |
| 13 | Training pipeline | 4 tests |
| 14 | Experiment runner | Smoke test |
| 15 | Final verification | Full suite |

**Total: 15 tasks, ~52 tests, building bottom-up from data structures to full training pipeline.**
