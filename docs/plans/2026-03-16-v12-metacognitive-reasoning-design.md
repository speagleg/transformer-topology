# v12: Topology-Guided Metacognitive Reasoning

## Vision

A metacognitive reasoning system where a GNN "conscious mind" plans reasoning structure, monitors reasoning quality via topological signal processing, and intervenes to correct a local LLM "subconscious" — all on a single RTX 4090.

The graph IS the reasoning. Nodes are thoughts, edges are logical dependencies, triangles are circular arguments. The topology of the reasoning graph directly encodes its quality — and the GNN can read that topology to diagnose and fix problems in real time.

## Background

### What We've Proven (v3-v11)

**Topology works for structural reasoning:**
- BFS 99.4%, diverse 98.7%, hodge_class 95.8% (with explicit features)
- Hodge decomposition reliably separates gradient/curl/harmonic components
- Spectral gaps detect information bottlenecks
- Curl energy detects information cycling (78% curl in v9 = reasoning going nowhere)
- Persistence diagrams capture multi-scale topological features

**Deep text-topology fusion works:**
- v11 achieved 69.9% balanced accuracy on 16-class KG relation classification
- Bidirectional cross-attention + text in classifier was the key
- fusion_weight learns task-appropriate blending

**What failed and why:**
- Inverse scaling was a data artifact (curl class imbalance), not a real property
- Shallow text integration (v6-v9) never contributed — text must be deeply integrated
- ConceptNet KG classification is not the right task for this architecture — topology is overhead when the signal is semantic

**The core insight:** Topology excels when the signal IS topological structure. For reasoning, the signal is the structure of the reasoning process itself — not the structure of external data.

### Related Work

**Graph of Thoughts (GoT)** — ETH Zurich, AAAI 2024. Models LLM thoughts as a graph. Prompt engineering only — no learned graph structure, no GNN, no topology analysis.

**L2T: Learn to Think** — IJCAI 2025. GNN processes reasoning graph, generates parameter vectors controlling LLM sampling (temperature, top-p, branch count). Closest existing work. Uses vanilla GNN embeddings — no topological analysis.

**GraphMind** — November 2025. Dynamic GNN for theorem selection in multi-step reasoning. Heterogeneous graph with typed nodes/edges. InfoNCE training. No topological features.

**Think2** — 2026. Metacognitive framework based on Brown's regulatory cycle (Planning → Monitoring → Evaluation). Prompt-based, no graphs or topology. Validates the metacognition framing.

### Our Differentiation

| Feature | L2T | GraphMind | Think2 | v12 (ours) |
|---------|-----|-----------|--------|------------|
| Graph structure | Flat thought graph | Heterogeneous reasoning graph | None (prompts only) | **Cell complex** (0/1/2-cells) |
| GNN role | Embedding + param generation | State encoding + theorem selection | N/A | **Topological signal processing** |
| Feedback mechanism | Sampling params (temp, top-p) | Theorem selection | NL prompt structure | **Topology-diagnosed NL corrections** |
| Monitoring signal | Node labels (stop/continue/backtrack) | Theorem matching score | Self-evaluation prompts | **Hodge decomposition, spectral gap, curl** |
| Planning | Implicit (branch count) | None (reactive) | Prompt template | **Explicit graph template from problem type** |
| Circular reasoning detection | No | No | No | **Yes (curl energy on 2-cells)** |
| Bottleneck detection | No | No | No | **Yes (spectral gap analysis)** |
| Dead-end detection | No | No | No | **Yes (harmonic energy)** |

**Novel contribution:** Using topological signal processing (Hodge decomposition, spectral gaps, curl energy, persistence) on the computation graph of reasoning itself to diagnose and correct reasoning quality. No prior work analyzes the topology of reasoning — they use graphs as data structures, not as objects of topological analysis.

## Architecture

### The Metacognitive Loop

```
NL Prompt
    │
    ▼
┌─────────────────┐
│ Problem Classifier │ ← LLM reads prompt, classifies reasoning type
└────────┬────────┘
         │ problem_type
         ▼
┌─────────────────┐
│ Graph Template   │ ← GNN Executive selects initial topology
│ Generator        │   (chain / tree / DAG / star / bipartite)
└────────┬────────┘
         │ initial_graph
         ▼
┌─────────────────────────────────────────────┐
│                REASONING LOOP                │
│                                              │
│  ┌──────────────┐    ┌───────────────────┐  │
│  │ LLM Step     │───▶│ Graph Constructor │  │
│  │ Generator    │    │ (step → cell      │  │
│  │              │    │  complex update)   │  │
│  └──────┬───────┘    └────────┬──────────┘  │
│         │                     │              │
│         │              ┌──────▼──────────┐   │
│         │              │ Topology        │   │
│         │              │ Analyzer        │   │
│         │              │ (Hodge, spectral│   │
│         │              │  gap, curl, β₁) │   │
│         │              └──────┬──────────┘   │
│         │                     │              │
│         │              ┌──────▼──────────┐   │
│         │              │ Reasoning       │   │
│  ┌──────┴───────┐      │ Health Report   │   │
│  │ Intervention │◀─────│ (diagnose +     │   │
│  │ Generator    │      │  decide action) │   │
│  │ (topology →  │      └─────────────────┘   │
│  │  NL prompt)  │                            │
│  └──────────────┘                            │
│                                              │
│  Repeat until: complete / max_steps / stuck  │
└──────────────────────────────────────────────┘

**Loop termination policy:**
- **Complete:** All template nodes filled, topology healthy (all signals within thresholds).
- **Max steps:** Hard cap of 15 reasoning steps. Emit best answer assembled from completed subgraph.
- **Stuck:** If two consecutive interventions for the same condition (e.g., high curl) fail to improve that metric by >10% relative, escalate: BACKTRACK (discard problematic subgraph, re-attempt from prior node) or STOP (emit best answer so far).
- **Diminishing returns:** If overall balanced accuracy of topology signals hasn't improved after 3 consecutive steps, STOP.
         │
         ▼
┌─────────────────┐
│ Answer Assembler │ ← Extract final answer from completed graph
└─────────────────┘
         │
         ▼
    Final Answer
```

### Component Specifications

#### 1. Problem Classifier

**Input:** NL prompt (raw text)
**Output:** Problem type enum + confidence

**Problem types:**
- `SEQUENTIAL` — steps build linearly (math, arithmetic)
- `CONSTRAINT` — multiple constraints must be satisfied simultaneously
- `MULTI_HOP` — evidence from multiple sources converges
- `EXPLORATION` — need to search alternatives (puzzles, planning)
- `VERIFICATION` — claims need evidence checking

**Implementation:** Single LLM forward pass with structured output. The LLM is prompted:
```
Classify the reasoning type needed for this problem.
Types: SEQUENTIAL, CONSTRAINT, MULTI_HOP, EXPLORATION, VERIFICATION
Problem: {prompt}
Output: {"type": "...", "confidence": 0.0-1.0, "explanation": "..."}
```

No training needed — this is prompt-based classification using the LLM's existing capabilities.

#### 2. Graph Template Generator

**Input:** Problem type
**Output:** Initial CellComplex with template structure

**Templates:**

| Type | Graph Structure | Nodes | Edges | Rationale |
|------|----------------|-------|-------|-----------|
| SEQUENTIAL | Linear chain (n=5-10) | Numbered steps | Sequential dependencies | Math builds step-by-step |
| CONSTRAINT | Star graph | Center = goal, leaves = constraints | Constraint → goal | All constraints feed into solution |
| MULTI_HOP | DAG with merge | Evidence sources → intermediate → conclusion | Evidential support | Multiple evidence streams converge |
| EXPLORATION | Binary tree (depth 3-4) | Decision points | Branch alternatives | Search space exploration |
| VERIFICATION | Bipartite | Claims on one side, evidence on other | Claim ↔ evidence | Match claims to support |

**Implementation:** Deterministic mapping from problem type to CellComplex construction. No learning — this is a lookup table of graph construction functions. The number of initial nodes is configurable (default: 5-8 steps).

Template nodes start with placeholder embeddings. As the LLM fills them, embeddings are replaced with actual step text embeddings.

#### 3. LLM Step Generator

**Input:** Current graph state + next node to fill + optional intervention
**Output:** Reasoning step text + declared dependencies

**Structured output format:**
```json
{
  "step": "Since x = 3 and y = x + 2, we get y = 5",
  "depends_on": [1, 3],
  "confidence": 0.85,
  "type": "deduction"
}
```

**Implementation:** LLM receives a prompt containing:
1. Original problem
2. All completed steps so far (with step numbers)
3. The graph template's expectation for this node (e.g., "this step should combine evidence from steps 2 and 4")
4. Any metacognitive intervention (if topology flagged an issue)

The `depends_on` field is critical — it tells the Graph Constructor which edges to create.

**LLM choice:** Qwen 2.5-3B-Instruct (same as v11, already deployed). Frozen, inference only. All 32 layers used for reasoning quality (v11 used only 4 layers for embedding).

**Node embedding strategy:** Reuse the LLM's hidden state from the generation pass — when Qwen generates a reasoning step, extract the last hidden state of the final token as the step embedding. This costs zero additional forward passes. Project from Qwen's hidden dim (2048) to the reasoning graph's embedding_dim (128) via a learned linear projection. The 128D dimension balances expressiveness with tractable spectral operations on small graphs.

**Reasoning CellComplex embedding_dim: 128.**

#### 4. Graph Constructor

**Input:** LLM step output + current CellComplex
**Output:** Updated CellComplex

**Node creation:**
- New 0-cell for each reasoning step
- Embedding = Qwen encoding of step text (full model, not truncated)
- Store step text, confidence, step type as node metadata

**Edge creation (1-cells):**
- For each `depends_on` reference: create edge from dependency → current step
- Edge embedding = concatenation of endpoint similarities + dependency type encoding
**Contradiction detection:** Cosine similarity < -0.5 is too rare with LLM embeddings. Instead, add a `contradicts` field to the structured output:
```json
{
  "step": "...",
  "depends_on": [1, 3],
  "contradicts": [],
  "confidence": 0.85,
  "type": "deduction"
}
```
If the LLM reports contradictions, add contradiction edges. As a secondary signal, if cosine similarity with a prior step is high (>0.8) but the LLM's step negates it (detected via keyword: "however", "but", "not", "wrong", "incorrect"), flag as potential contradiction.

**Triangle detection (2-cells):**
- After adding edges, scan for new triangles (3 mutually connected nodes)
- Each triangle becomes a 2-cell
- Triangles in reasoning = potential circular arguments (A→B→C→A)
**Edge direction semantics:** Track "forward" vs "backward" edges using the `relation_type` parameter of `add_1_cell`:
- Forward edge (A→B where A is an earlier step): valid logical progression
- Backward edge (B→A where B is later): potential circular dependency

Only flag 2-cells containing backward edges as potential circular reasoning. Forward-only triangles (A→B, B→C, A→C) are valid evidence triangulation — the argument is reinforced, not circular.

**Structured output fallback:** Qwen 2.5-3B may produce malformed JSON or invalid `depends_on` references. Fallback chain:
1. If JSON parsing fails: extract step text via regex, connect to previous step (linear chain)
2. If `depends_on` references non-existent steps: ignore invalid refs, keep valid ones
3. If `depends_on` is empty: connect to previous step + compute cosine similarity with all prior steps. If any similarity > 0.7, add a "latent dependency" edge
4. Secondary signal: always compute cosine similarity between new step embedding and all prior steps. If similarity > 0.7 with step N and N is not in `depends_on`, add a latent dependency edge (weaker edge type)

**Implementation:** Extends existing CellComplex with incremental updates. Uses `add_0_cell`, `add_1_cell`, `add_2_cell` from the existing API. Edge creation is O(n) per step (check all prior nodes).

#### 5. Topology Analyzer

**Input:** Current CellComplex (reasoning graph)
**Output:** ReasoningHealthReport

**Computed features:**

| Feature | Computation | Source |
|---------|-------------|--------|
| Curl energy | `hodge_decomposition(cc, signal, dim=1)` → curl component norm (high = circular patterns (necessary but not sufficient for circular reasoning)) | `src/spectral/decomposition.py` |
| Gradient energy | Same decomposition → gradient component norm | Same |
| Harmonic energy | Same decomposition → harmonic component norm | Same |
| Spectral gap | Fiedler value of `hodge_laplacian_0(cc)` | `src/spectral/laplacian.py` |
| Connected components | Count of zero eigenvalues in L0 | Same |
| Betti numbers (β₁) | `num_edges - num_nodes + num_components` (graph-theoretic) | Direct computation from boundary operators |
| Graph density | edges / possible edges | Direct computation |

**Signal for Hodge decomposition:** The edge signal is constructed from the cosine similarity between connected step embeddings. High similarity = strong dependency, low similarity = weak logical connection.

**Implementation:** Direct reuse of v3-v11 spectral pipeline. No modifications needed to the topology computation — only the interpretation layer is new.

**ReasoningHealthReport dataclass:**
```python
@dataclass
class ReasoningHealthReport:
    curl_energy: float          # 0-1, high = circular reasoning
    gradient_energy: float      # 0-1, high = forward progress
    harmonic_energy: float      # 0-1, high = trapped/disconnected info
    spectral_gap: float         # 0+, low = bottleneck
    num_components: int         # >1 = fragmented reasoning
    betti_1: int               # independent cycles count
    graph_density: float        # edge density
    action: str                # CONTINUE, INTERVENE, BACKTRACK, STOP
    diagnosis: str             # human-readable explanation
```

**Action decision thresholds (initial, tunable):**
- `curl_energy > 0.4` → INTERVENE (break cycle)
- `spectral_gap < 0.05` → INTERVENE (expand paths)
- `harmonic_energy > 0.3` → INTERVENE (connect orphans)
- `num_components > 1` → INTERVENE (bridge fragments)
- `betti_1 > 2` → INTERVENE (resolve loops)
- All healthy → CONTINUE

**Interpretation caveat:** High curl on the cosine similarity signal is a necessary condition for circular reasoning (circular arguments create 2-cells with circulating similarity) but not sufficient (topically coherent steps also produce similar patterns). Interventions should be exploratory ("ensure each step adds new information") rather than accusatory ("you're repeating yourself").

**Implementation note:** The Topology Analyzer must receive a cloned CellComplex snapshot (`cc.clone()`) to avoid the known mutation bug documented in the project. The Graph Constructor incrementally mutates the live CellComplex; the Topology Analyzer reads a frozen snapshot.

#### 6. Intervention Generator

**Input:** ReasoningHealthReport
**Output:** NL correction string to inject into LLM's next prompt

**Intervention templates:**

| Condition | Template |
|-----------|----------|
| High curl | "You appear to be repeating reasoning from steps {cycle_nodes}. State a new fact or approach not yet considered." |
| Low spectral gap | "Your reasoning has a bottleneck at step {bottleneck_node}. Consider an alternative path to the conclusion." |
| High harmonic | "Steps {orphan_nodes} are disconnected from your main argument. Either connect them or discard them." |
| Multiple components | "You have {n} separate reasoning threads. Explain how they relate to each other." |
| High β₁ | "You have {n} unresolved circular dependencies. Commit to the strongest chain of reasoning." |

**Cycle/bottleneck identification:** The topology analyzer identifies specific nodes involved in problems:
- Curl: nodes participating in the highest-energy 2-cells
- Bottleneck: node with highest betweenness centrality near the Fiedler cut
- Orphans: nodes in small connected components

**Implementation:** Template-based string formatting. The intervention is appended to the LLM's next prompt as a `[METACOGNITIVE NOTE]` section.

#### 7. Answer Assembler

**Input:** Completed reasoning graph
**Output:** Final answer text

**Strategy:** Extract the terminal node(s) of the reasoning DAG — the nodes with no outgoing dependency edges. If multiple terminal nodes, the LLM is prompted to synthesize them into a single answer.

For SEQUENTIAL problems: the last step IS the answer.
For CONSTRAINT problems: the center node (goal) contains the answer.
For MULTI_HOP: the merge point(s) contain the synthesized answer.

## Evaluation

### Primary Benchmark: GSM8K

Multi-step math reasoning. 8.5K grade school math problems requiring 2-8 reasoning steps.

**Why GSM8K:**
- Well-studied failure modes for small LLMs
- Clear ground truth (numerical answer)
- Multi-step = reasoning graph is meaningful
- Existing baselines for Qwen 3B class models

### Secondary Benchmarks

- **FOLIO** — first-order logic (tests CONSTRAINT template)
- **ARC-Challenge** — abstract reasoning (tests EXPLORATION template)

### Baselines

1. **Qwen 3B raw** — no prompting structure
2. **Qwen 3B + CoT** — chain-of-thought prompting
3. **Qwen 3B + ToT** — tree of thoughts (if feasible on 3B)
4. **Qwen 3B + v12 metacognitive** — our system

### Ablations

1. **v12 without monitoring** — GNN plans graph template, LLM fills it, no topology analysis. Isolates planning value.
2. **v12 without planning** — No template. LLM reasons freely, GNN builds graph from output and intervenes when topology is unhealthy. Isolates monitoring value.
3. **v12 without topological features** — Replace Hodge/spectral analysis with vanilla GNN node embeddings for intervention decisions (like L2T). Isolates topology's value vs standard GNN.
4. **v12 without intervention** — Full topology analysis but no corrective prompts injected. Isolates intervention value.

### Success Criteria

**Primary target:** 5% absolute improvement on GSM8K over CoT baseline with Qwen 3B.
**Stretch target:** 10% absolute improvement.
**Qualitative targets:** Does the system detect and correct specific failure modes? Does it produce more interpretable reasoning traces? These are valuable even without quantitative gains.

**Meaningful result even if target not met:** If ablation 3 (vanilla GNN) matches v12, topology is overhead. If ablation 1 (no monitoring) matches v12, planning alone suffices. Either finding redirects the project productively.

**Latency tracking:** Report accuracy-per-second alongside raw accuracy. If v12 achieves 5% accuracy gain at 10× latency over CoT, document the tradeoff explicitly.

**Error analysis:** Categorize GSM8K problems by type (arithmetic chain, multi-variable, word problem) and report per-type improvement. Understanding which problem types benefit from topology is more valuable than aggregate numbers.

**Inference time constraint:** <30 seconds per problem on RTX 4090.

## What We Reuse vs Build

### Reuse Directly (v3-v11)

| Module | Location | Used For |
|--------|----------|----------|
| CellComplex | `src/cell_complex/cell_complex.py` | Reasoning graph data structure |
| Hodge decomposition | `src/spectral/decomposition.py` | Curl/gradient/harmonic analysis |
| Spectral decomposition | `src/spectral/decomposition.py` | Eigenvalues for spectral gap |
| Hodge Laplacians (L0/L1/L2) | `src/spectral/laplacian.py` | Laplacian computation |
| Persistence diagrams | `src/spectral/persistence.py` | Betti numbers |
| Boundary operators | `src/cell_complex/cell_complex.py` | Chain complex structure |
| Node-triangle incidence | `src/cell_complex/cell_complex.py` | 2-cell analysis |

### Build New

| Module | Estimated Complexity | Dependencies |
|--------|---------------------|-------------|
| Problem Classifier | Low — prompt template | LLM |
| Graph Template Generator | Low — lookup table | CellComplex |
| LLM Step Generator | Medium — structured prompting + parsing | LLM, prompt engineering |
| Graph Constructor | High — NL to graph, edge detection, triangle scan | CellComplex, LLM embeddings |
| ReasoningHealthReport | Low — dataclass + thresholds | Topology Analyzer |
| Intervention Generator | Medium — template selection + node identification | ReasoningHealthReport |
| Answer Assembler | Low — graph traversal | CellComplex |
| Signal Constructor | Low — cosine similarity between step embeddings for Hodge signal | Node embeddings |
| Embedding Projection | Low — Linear(2048, 128) | Qwen hidden states |
| Evaluation harness | Medium — benchmark loading, baseline comparison | External datasets |

### Don't Need (from v3-v11)

| Module | Why Not |
|--------|---------|
| QwenContextualEncoder (4 layers) | Use full Qwen 3B for reasoning, not truncated encoder |
| AttentionReadout | No classification task — answer is assembled from graph |
| TextConditionedGNN | Edges are logical dependencies, not ConceptNet relations |
| BidirectionalCrossAttention | Text-structure fusion happens through the graph construction, not attention |
| Multi-head classifier | No classification — open-ended reasoning |
| TopologicalPE | Not needed for reasoning graphs (structure is dynamic) |
| Wave dynamics / neural ODE | Overkill for reasoning graph analysis |
| MultiScaleLaplacianFilter | May revisit later, but start with L0 analysis |

## Implementation Order

0. **Phase 0: Structured output validation** — Run 50 GSM8K problems through Qwen 3B with the structured step-by-step prompt (no topology, no intervention). Manually inspect: Does the LLM produce valid JSON? Are `depends_on` references coherent? Do the resulting graphs make sense? This is a 1-day experiment that validates the most critical assumption. If graphs are incoherent, redesign the LLM interface before building the rest.
1. **LLM Step Generator + structured output** — Get Qwen 3B generating structured reasoning steps locally
2. **Graph Constructor** — Parse steps into CellComplex with edges and 2-cells
3. **Topology Analyzer integration** — Run existing spectral pipeline on reasoning graphs
4. **ReasoningHealthReport** — Thresholds and action decisions
5. **Intervention Generator** — Topology signals to NL corrections
6. **Problem Classifier + Graph Templates** — Problem type detection and initial graph selection
7. **The metacognitive loop** — Wire everything together
8. **Evaluation harness** — GSM8K loading, baseline comparison, ablations
9. **Tuning** — Threshold calibration, template refinement, prompt engineering

## Compute Requirements

- **LLM:** Qwen 2.5-3B-Instruct, full model (~6GB VRAM)
- **Topology:** CPU-based spectral decomposition on small graphs (5-20 nodes). Fast — <10ms per analysis.
- **Total VRAM:** ~8GB (LLM + small GNN overhead). Fits RTX 4090 (24.6GB) easily.
- **Inference time:** ~5-15 forward passes through Qwen per problem (one per reasoning step) + topology analysis. Estimated 10-25s per problem.
