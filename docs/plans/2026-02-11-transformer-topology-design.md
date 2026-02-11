# Transformer-Topology (TT) — Architecture Design

## Problem Statement

Current LLM architectures have two fundamental limitations:

1. **Structured reasoning** — LLMs are poor at multi-hop relational reasoning (e.g., "if A causes B and B blocks C, what happens to C?"). They simulate reasoning through token prediction rather than explicit graph traversal.
2. **Persistent world model** — LLMs are stateless per-call. There is no persistent, evolving knowledge representation that survives across interactions.

Transformer-Topology addresses both by building a hierarchical system where a GNN executive maintains a persistent cell complex (the "conscious mind") and orchestrates a topology-aware transformer (the "subconscious") through an interleaved co-processing loop.

---

## Architecture Overview

A two-tier reasoning system with three layers:

### 1. Cell Complex Knowledge Store

The persistent world model. Not a flat graph — a CW-complex with:
- **0-cells** (entities)
- **1-cells** (relations)
- **2-cells** (higher-order groupings: co-occurring triples, causal loops, analogies)

The topology itself (holes, cycles, connected components) carries semantic meaning. Seeded from an initial ontology, evolves as the system processes tasks.

### 2. GNN Executive

Operates directly on the cell complex. Computes Hodge Laplacians at each order, decomposes signals via Hodge decomposition (gradient + curl + harmonic), and runs both spatial message-passing and spectral filtering in parallel. The harmonic component — the topologically irreducible signal — drives high-level reasoning decisions. This layer decides *what to think about* and *how reasoning should flow*.

### 3. Topology-Aware Transformer (TAT)

A small custom transformer where attention patterns are biased by the local topology of the cell complex. Instead of full-sequence attention, attention weights incorporate topological positional encodings derived from persistent homology features and Laplacian eigenvectors. This layer executes the sub-tasks the GNN dispatches and returns results that update the complex.

---

## The Interleaved Co-Processing Loop

The reasoning cycle runs as a tight loop between the GNN executive and the TAT:

### Step 1 — Grounding

The input (a query or task) is parsed into a local subgraph and attached to the persistent cell complex. New entities become 0-cells, relations become 1-cells. If triples form closed loops, 2-cells are created automatically. The seed graph provides context — the new input "lands" on existing topology.

### Step 2 — Topological Analysis

The GNN executive computes:

- **Spatial path:** Message passing over the cell complex using boundary operators (∂₁, ∂₂) to propagate signals up and down the cell hierarchy. Nodes inform edges, edges inform faces, and back.
- **Spectral path:** Eigendecomposition of the Hodge Laplacians (L₀, L₁, L₂). Low-frequency eigenvectors capture global structure; high-frequency captures local detail. Signals are filtered in the spectral domain.
- **Hodge decomposition** separates each signal into gradient (locally explainable), curl (cyclic flow), and harmonic (topological) components.

### Step 3 — Task Dispatch

The GNN identifies reasoning sub-tasks based on the analysis — e.g., "resolve the relationship between nodes A and D" or "check if cycle X implies contradiction." These sub-tasks, along with their topological context (relevant subgraph embeddings, spectral features), are dispatched to the TAT.

### Step 4 — Execution & Update

The TAT processes each sub-task, generating outputs conditioned on the topological context. Results flow back to the GNN, which updates the cell complex — new cells, modified embeddings, potentially new higher-order cells if new relationships emerged.

**Steps 2-4 repeat** until the GNN's harmonic signal stabilizes (convergence) or a max iteration limit is hit.

---

## Topology-Aware Transformer Internals

The TAT is not a standard transformer with graph features bolted on. Topology is woven into every layer.

### Attention Mechanism — Dual-Basis Attention

Each attention head operates in one of two modes:

- **Spatial heads:** Attention weights are masked and biased by adjacency in the cell complex. A token attending to another token is weighted by their topological proximity — not just positional distance. The boundary operators (∂) define what "adjacent" means across cell dimensions.
- **Spectral heads:** Attention is computed in the Fourier domain. Token representations are projected onto the Laplacian eigenbasis, convolved with learnable spectral filters, then projected back. This captures long-range topological patterns that spatial heads would need many hops to reach.

### Positional Encoding — Topological PE

Instead of sinusoidal or learned position embeddings, each token receives:

- **Laplacian eigenvector features** — coordinates in the spectral embedding of the cell complex (a la Graphormer/SignNet).
- **Persistent homology signature** — a vectorized persistence diagram capturing local topological features (birth/death of holes) around the token's position.
- **Cell membership indicators** — which 1-cells and 2-cells this token participates in, giving awareness of higher-order structure.

### Feed-Forward Layers — Sheaf-Valued

Rather than standard MLPs, the feed-forward blocks operate on sheaf stalks. Each node's representation lives in its own vector space (stalk), and the linear maps between layers respect the sheaf restriction maps between adjacent cells. Information transfer is geometrically principled — it follows the topology rather than ignoring it.

---

## Cell Complex Knowledge Store — Details

### Structure

Implemented as three sparse tensors plus metadata:

- **C₀** (0-cells): Entity nodes. Each carries an embedding vector and a type label (concept, event, agent, property).
- **C₁** (1-cells): Relation edges. Each carries an embedding plus a relation type. The boundary operator ∂₁ maps each 1-cell to its two endpoint 0-cells.
- **C₂** (2-cells): Higher-order groupings — a triangle means "these three relations form a closed semantic unit" (e.g., a causal loop, a syllogism, a contradiction). ∂₂ maps each 2-cell to its boundary 1-cells.

### Seeding

The initial complex is built from a small structured knowledge base — ConceptNet subgraphs, a domain ontology, or hand-crafted reasoning primitives for synthetic benchmarks. The seed defines starting topology: which cycles exist, which components are connected, initial Betti numbers.

### Evolution

After each reasoning cycle:

- **Growth:** New cells added when the TAT identifies new entities or relations. If three new 1-cells form a closed boundary, a 2-cell is automatically created (triangle closure).
- **Embedding update:** Cell embeddings updated via GNN message passing. Frequently activated cells strengthen; rarely used cells decay.
- **Topological surgery:** Periodically, the system checks for topological changes — new holes appearing (increasing β₁), components merging, 2-cells collapsing. These structural events are logged as "topological events" and can themselves trigger reasoning.

### Persistence

The complex serializes to disk as sparse matrices (boundary operators) plus embedding tables. Between sessions, topology is preserved exactly; embeddings can be checkpointed.

---

## Wave Dynamics & Spectral Reasoning

The wave equation on cell complexes gives the system a natural notion of propagation over time.

### Diffusion Process

Given a signal f on k-cells, the heat equation ∂f/∂t = -Lₖf governs how information diffuses. Short diffusion time = local neighborhood activation. Long diffusion time = global spread. The GNN learns to control diffusion time as a parameter — fast for local lookups, slow for global reasoning.

### Wave Equation

The wave equation ∂²f/∂t² = -Lₖf gives oscillatory propagation. Signals bounce off topological boundaries (holes, disconnected components), creating interference patterns. **Interference patterns encode relational structure**: constructive interference indicates compatible reasoning paths; destructive interference flags contradictions.

### Spectral Filtering

In the Fourier domain (Laplacian eigenbasis), diffusion is low-pass filtering (smooth out high-frequency noise), wave propagation preserves all frequencies but shifts phases. The model learns filter banks that selectively amplify or dampen frequency bands — equivalent to choosing what "scale" of topological structure to reason about.

### Neural ODE Integration

Rather than discretizing the wave/diffusion equations into fixed steps, the system uses a neural ODE solver (torchdiffeq) to integrate continuously. The GNN learns initial conditions and ODE parameters; the solver handles the dynamics. This gives adaptive computation — simple problems converge quickly, complex ones take more integration time.

---

## Prototype Plan & Synthetic Benchmarks

### Phase 1 — Multi-hop Graph Traversal

- Synthetic dataset: chain/tree/DAG reasoning problems with increasing depth (2-hop to 10-hop) and distractors.
- Cell complex with 0-cells and 1-cells only (no 2-cells yet). Seed from task graph.
- TAT at ~15M params with spatial attention heads and Laplacian PE.
- Baseline: Compare against vanilla transformer and standard GNN (GAT/GCN).
- **Success metric:** Accuracy on deep chains (6+ hops) where vanilla transformers degrade.

### Phase 2 — Topological Reasoning

- Introduce 2-cells. Tasks require detecting cycles, alternate paths, connectivity changes.
- Add persistent homology features to positional encodings.
- Activate spectral heads alongside spatial heads.
- Add Hodge decomposition — harmonic component should directly encode topological answers.
- **Success metric:** Correct answers to topological questions requiring global structure awareness.

### Phase 3 — Temporal/Causal Chains

- Causal propagation with timing: "A fires at t=0, signal reaches B at t=2, C at t=5. If B is blocked, when does C activate?"
- Activate wave dynamics via neural ODE.
- Wave equation naturally handles propagation delays and interference.
- **Success metric:** Correct temporal predictions requiring propagation modeling.

Each phase builds on the previous — no throwaway code between phases.

---

## Project Structure

```
transformer-topology/
├── src/
│   ├── cell_complex/       # CW-complex data structures, boundary operators
│   ├── gnn_executive/      # GNN layer, Hodge Laplacians, message passing
│   ├── tat/                # Topology-Aware Transformer
│   ├── spectral/           # Eigendecomposition, Fourier transforms, filters
│   ├── wave/               # Wave/diffusion dynamics, neural ODE
│   ├── reasoning_loop/     # The interleaved co-processing loop
│   └── benchmarks/         # Synthetic data generation & evaluation
├── tests/                  # Unit + integration tests per module
├── data/                   # Seed graphs, generated benchmarks
├── models/                 # Saved checkpoints
├── config/                 # Hyperparameters, experiment configs
└── docs/                   # Design docs, research notes, paper drafts
```

## Dependencies

- `torch` + `torch_geometric` — core framework
- `torchdiffeq` — neural ODE solver for wave dynamics
- `gudhi` or `ripser` — persistent homology computation
- `scipy.sparse` — sparse Laplacian operations and eigendecomposition
- `networkx` — utility graph operations, seed graph construction
- `wandb` or `tensorboard` — experiment tracking

## Hardware Targets

- **Local:** Development and small experiments
- **Lambda Labs:** GPU bursts for full benchmark runs and hyperparameter sweeps

---

## LLM Scaling Path

1. **Phase A (prototype):** Custom tiny transformer (~15M params), topology-aware from the ground up. Full control for experimentation.
2. **Phase B (paper):** Same architecture, validated on synthetic benchmarks, controlled experiments for publication.
3. **Phase C (scale):** Swap in pretrained backbone (Llama 3 8B, Phi-3, etc.) with LoRA adaptation. Topology-aware attention layers wrap or replace standard attention. Interface between GNN executive and LLM adapts to the larger model's representation space.
