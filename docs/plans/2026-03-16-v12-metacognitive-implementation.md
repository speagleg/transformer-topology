# v12: Topology-Guided Metacognitive Reasoning — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a metacognitive reasoning system where a GNN plans reasoning structure, monitors via topological signal processing (Hodge decomposition, spectral gap, curl), and intervenes to correct a local LLM on multi-step reasoning tasks.

**Architecture:** NL prompt → Problem Classifier → Graph Template Generator → LLM fills reasoning steps → Graph Constructor builds CellComplex → Topology Analyzer computes Hodge/spectral features → Intervention Generator injects NL corrections → repeat until healthy or max steps.

**Tech Stack:** Python 3.12, PyTorch, Qwen 2.5-3B-Instruct (local, full model), existing CellComplex + spectral pipeline from v3-v11.

**Spec:** `docs/plans/2026-03-16-v12-metacognitive-reasoning-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/metacog/step_generator.py` | Create | LLM Step Generator — structured step-by-step reasoning with Qwen |
| `src/metacog/graph_constructor.py` | Create | Graph Constructor — parse LLM steps into CellComplex |
| `src/metacog/topology_analyzer.py` | Create | Topology Analyzer — Hodge/spectral analysis on reasoning graph |
| `src/metacog/health_report.py` | Create | ReasoningHealthReport dataclass + action decision logic |
| `src/metacog/intervention.py` | Create | Intervention Generator — topology signals to NL corrections |
| `src/metacog/problem_classifier.py` | Create | Problem type classification via LLM |
| `src/metacog/graph_templates.py` | Create | Graph template generation by problem type |
| `src/metacog/reasoning_loop.py` | Create | The full metacognitive loop — wire everything together |
| `src/metacog/embedding_projection.py` | Create | Linear(2048, 128) projection for node embeddings |
| `src/metacog/__init__.py` | Create | Package init |
| `scripts/v12_phase0_validate.py` | Create | Phase 0: structured output validation on GSM8K |
| `scripts/v12_run_gsm8k.py` | Create | Evaluation harness for GSM8K benchmark |
| `tests/test_metacog/test_graph_constructor.py` | Create | Graph construction tests |
| `tests/test_metacog/test_topology_analyzer.py` | Create | Topology analysis on reasoning graphs |
| `tests/test_metacog/test_health_report.py` | Create | Health report thresholds and actions |
| `tests/test_metacog/test_intervention.py` | Create | Intervention template tests |
| `tests/test_metacog/test_reasoning_loop.py` | Create | End-to-end loop tests |

---

## Chunk 1: Phase 0 + LLM Step Generator + Graph Constructor

### Task 0: Phase 0 — Structured Output Validation

**Files:**
- Create: `scripts/v12_phase0_validate.py`

This is the critical gate. Before building anything, validate that Qwen 2.5-3B can reliably produce structured JSON reasoning steps with `depends_on` fields.

- [ ] **Step 1: Create the Phase 0 validation script**

```python
#!/usr/bin/env python
"""Phase 0: Validate Qwen 2.5-3B structured reasoning output.

Runs 50 GSM8K problems through the LLM with structured step-by-step
prompting. Measures: JSON parse rate, depends_on validity, graph coherence.

If JSON parse rate < 80% or depends_on validity < 60%, the LLM interface
needs redesign before building the rest of v12.

Usage:
    PYTHONPATH=. python scripts/v12_phase0_validate.py [--num-problems 50]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- GSM8K loader ---

def load_gsm8k(split="test", max_problems=50):
    """Load GSM8K problems from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split)
        problems = []
        for item in ds:
            problems.append({
                "question": item["question"],
                "answer": item["answer"],
            })
            if len(problems) >= max_problems:
                break
        return problems
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)


# --- LLM interface ---

STEP_PROMPT = """You are solving a math problem step by step.
For each step, output a JSON object with these fields:
- "step": your reasoning for this step (string)
- "depends_on": list of step numbers this step builds on (list of ints, 1-indexed)
- "confidence": how confident you are in this step (float 0-1)
- "type": one of "setup", "computation", "deduction", "verification", "answer"

Problem: {problem}

Steps completed so far:
{steps_so_far}

Output ONLY the next step as a JSON object. No other text."""


def generate_step(model, tokenizer, problem, steps_so_far, device, max_tokens=256):
    """Generate one structured reasoning step."""
    steps_text = ""
    for i, s in enumerate(steps_so_far, 1):
        steps_text += f"Step {i}: {s.get('step', s) if isinstance(s, dict) else s}\n"
    if not steps_text:
        steps_text = "(none yet)\n"

    prompt = STEP_PROMPT.format(problem=problem, steps_so_far=steps_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_step(response):
    """Parse JSON from LLM response. Returns dict or None."""
    # Try direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting first {...} block
    match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# --- Validation ---

def validate_problem(model, tokenizer, problem, device, max_steps=8):
    """Run structured reasoning on one problem, collect metrics."""
    steps = []
    parse_successes = 0
    depends_on_valid = 0
    total_steps = 0

    for step_num in range(max_steps):
        raw = generate_step(model, tokenizer, problem["question"], steps, device)
        parsed = parse_step(raw)
        total_steps += 1

        if parsed is None:
            steps.append({"step": raw, "depends_on": [step_num] if step_num > 0 else [],
                          "confidence": 0.5, "type": "unknown", "_parse_failed": True})
            continue

        parse_successes += 1

        # Validate depends_on
        deps = parsed.get("depends_on", [])
        if isinstance(deps, list) and all(isinstance(d, int) and 1 <= d <= len(steps) for d in deps):
            depends_on_valid += 1

        # Check if this is the answer step
        step_type = parsed.get("type", "")
        parsed["_parse_failed"] = False
        steps.append(parsed)

        if step_type == "answer":
            break

    return {
        "total_steps": total_steps,
        "parse_successes": parse_successes,
        "parse_rate": parse_successes / max(total_steps, 1),
        "depends_on_valid": depends_on_valid,
        "depends_on_rate": depends_on_valid / max(parse_successes, 1),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )

    problems = load_gsm8k(max_problems=args.num_problems)
    print(f"Loaded {len(problems)} GSM8K problems")

    results = []
    total_parse = 0
    total_deps_valid = 0
    total_steps = 0
    total_parsed_steps = 0

    for i, problem in enumerate(problems):
        result = validate_problem(model, tokenizer, problem, device, args.max_steps)
        results.append(result)

        total_steps += result["total_steps"]
        total_parse += result["parse_successes"]
        total_parsed_steps += result["parse_successes"]
        total_deps_valid += result["depends_on_valid"]

        if (i + 1) % 10 == 0:
            pr = total_parse / max(total_steps, 1)
            dr = total_deps_valid / max(total_parsed_steps, 1)
            print(f"  [{i+1}/{len(problems)}] parse_rate={pr:.1%} depends_on_rate={dr:.1%}")

    # Summary
    parse_rate = total_parse / max(total_steps, 1)
    deps_rate = total_deps_valid / max(total_parsed_steps, 1)

    print("\n" + "=" * 60)
    print("PHASE 0 VALIDATION RESULTS")
    print("=" * 60)
    print(f"Problems:          {len(problems)}")
    print(f"Total steps:       {total_steps}")
    print(f"JSON parse rate:   {parse_rate:.1%} ({total_parse}/{total_steps})")
    print(f"depends_on valid:  {deps_rate:.1%} ({total_deps_valid}/{total_parsed_steps})")
    print()

    if parse_rate >= 0.8 and deps_rate >= 0.6:
        print("PASS — Structured output is reliable enough for v12")
    elif parse_rate >= 0.6:
        print("MARGINAL — Need stronger fallback handling, but proceed with caution")
    else:
        print("FAIL — Redesign LLM interface before building v12")

    # Save results
    out_path = Path("data/v12_phase0_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "parse_rate": parse_rate,
            "depends_on_rate": deps_rate,
            "total_steps": total_steps,
            "results": results,
        }, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run Phase 0 on GPU (vast.ai or local)**

```bash
pip install datasets  # if needed
PYTHONPATH=. python scripts/v12_phase0_validate.py --num-problems 50
```

Expected: JSON parse rate report. If ≥80% parse rate and ≥60% depends_on validity, proceed. If not, redesign the prompt or add stronger fallback before continuing.

- [ ] **Step 3: Commit**

```bash
git add scripts/v12_phase0_validate.py
git commit -m "feat: Phase 0 structured output validation for v12 metacognition"
```

---

### Task 1: Embedding Projection Module

**Files:**
- Create: `src/metacog/__init__.py`
- Create: `src/metacog/embedding_projection.py`
- Test: `tests/test_metacog/test_embedding_projection.py`

Small utility: project Qwen hidden states (2048D) to reasoning graph embedding_dim (128D).

- [ ] **Step 1: Write failing test**

```python
# tests/test_metacog/test_embedding_projection.py
"""Tests for embedding projection (Qwen 2048D → 128D)."""
import torch
import pytest


def test_projection_shape():
    from src.metacog.embedding_projection import EmbeddingProjection
    proj = EmbeddingProjection(input_dim=2048, output_dim=128)
    x = torch.randn(5, 2048)
    out = proj(x)
    assert out.shape == (5, 128)


def test_projection_single():
    from src.metacog.embedding_projection import EmbeddingProjection
    proj = EmbeddingProjection(input_dim=2048, output_dim=128)
    x = torch.randn(2048)
    out = proj(x)
    assert out.shape == (128,)


def test_projection_gradients():
    from src.metacog.embedding_projection import EmbeddingProjection
    proj = EmbeddingProjection(input_dim=2048, output_dim=128)
    x = torch.randn(3, 2048, requires_grad=True)
    out = proj(x)
    out.sum().backward()
    assert x.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_embedding_projection.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement**

```python
# src/metacog/__init__.py
"""v12: Topology-Guided Metacognitive Reasoning."""

# src/metacog/embedding_projection.py
"""Project LLM hidden states to reasoning graph embedding space."""
import torch
import torch.nn as nn


class EmbeddingProjection(nn.Module):
    """Linear projection from LLM hidden dim to CellComplex embedding_dim."""

    def __init__(self, input_dim: int = 2048, output_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state(s) to embedding space.

        Args:
            hidden_state: (D,) or (N, D) tensor from LLM last hidden state.

        Returns:
            (output_dim,) or (N, output_dim) projected embedding.
        """
        return self.proj(hidden_state)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_embedding_projection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
mkdir -p tests/test_metacog
git add src/metacog/ tests/test_metacog/
git commit -m "feat: embedding projection module for v12 metacognition"
```

---

### Task 2: Graph Constructor

**Files:**
- Create: `src/metacog/graph_constructor.py`
- Test: `tests/test_metacog/test_graph_constructor.py`

The core new component: parses LLM structured output into an evolving CellComplex.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metacog/test_graph_constructor.py
"""Tests for reasoning graph construction from LLM steps."""
import torch
import pytest
from src.metacog.graph_constructor import ReasoningGraphConstructor


EMBEDDING_DIM = 128


def _make_constructor():
    return ReasoningGraphConstructor(embedding_dim=EMBEDDING_DIM)


def test_add_first_step():
    gc = _make_constructor()
    embedding = torch.randn(EMBEDDING_DIM)
    gc.add_step(
        step_text="Let x = 3",
        embedding=embedding,
        depends_on=[],
        step_type="setup",
    )
    assert gc.num_steps == 1
    assert gc.cell_complex.num_cells(0) == 1
    assert gc.cell_complex.num_cells(1) == 0


def test_add_dependent_step():
    gc = _make_constructor()
    gc.add_step("Let x = 3", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    gc.add_step("Then x + 2 = 5", torch.randn(EMBEDDING_DIM), depends_on=[1], step_type="computation")
    assert gc.num_steps == 2
    assert gc.cell_complex.num_cells(0) == 2
    assert gc.cell_complex.num_cells(1) == 1  # one dependency edge


def test_multiple_dependencies():
    gc = _make_constructor()
    gc.add_step("x = 3", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    gc.add_step("y = 4", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    gc.add_step("x + y = 7", torch.randn(EMBEDDING_DIM), depends_on=[1, 2], step_type="computation")
    assert gc.cell_complex.num_cells(1) == 2  # two dependency edges


def test_invalid_depends_on_ignored():
    gc = _make_constructor()
    gc.add_step("x = 3", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    gc.add_step("y = 5", torch.randn(EMBEDDING_DIM), depends_on=[1, 99], step_type="computation")
    # Step 99 doesn't exist — should be ignored, step 1 kept
    assert gc.cell_complex.num_cells(1) == 1


def test_empty_depends_on_fallback():
    """Empty depends_on on step 2+ should connect to previous step."""
    gc = _make_constructor()
    gc.add_step("x = 3", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    gc.add_step("y = 5", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="computation")
    # Fallback: step 2 should connect to step 1
    assert gc.cell_complex.num_cells(1) == 1


def test_triangle_detection():
    gc = _make_constructor()
    gc.add_step("A", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    gc.add_step("B", torch.randn(EMBEDDING_DIM), depends_on=[1], step_type="deduction")
    gc.add_step("C", torch.randn(EMBEDDING_DIM), depends_on=[1, 2], step_type="deduction")
    # Edges: 1→2, 1→3, 2→3 = triangle (0,1,2)
    assert gc.cell_complex.num_cells(2) >= 1


def test_get_snapshot():
    gc = _make_constructor()
    gc.add_step("x = 3", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    snapshot = gc.get_snapshot()
    # Snapshot should be a cloned CellComplex
    assert snapshot.num_cells(0) == 1
    # Modifying snapshot should not affect original
    snapshot.add_0_cell(torch.randn(EMBEDDING_DIM), cell_type="test")
    assert gc.cell_complex.num_cells(0) == 1


def test_edge_direction_tracking():
    gc = _make_constructor()
    gc.add_step("A", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    gc.add_step("B", torch.randn(EMBEDDING_DIM), depends_on=[1], step_type="deduction")
    # Edge from step 1 → step 2 should be "forward"
    assert gc.edge_directions[0] == "forward"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_graph_constructor.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement ReasoningGraphConstructor**

```python
# src/metacog/graph_constructor.py
"""Build a CellComplex from LLM reasoning steps.

Each reasoning step becomes a node. Dependencies become edges.
Triangles (3 mutually connected nodes) become 2-cells for curl analysis.
"""

import torch
from src.cell_complex.cell_complex import CellComplex


class ReasoningGraphConstructor:
    """Incrementally builds a reasoning CellComplex from LLM steps."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.cell_complex = CellComplex(embedding_dim=embedding_dim)
        self.steps: list[dict] = []
        self.edge_directions: list[str] = []  # "forward" or "backward"
        self._edge_pairs: list[tuple[int, int]] = []  # (source, target) for triangle scan

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def add_step(
        self,
        step_text: str,
        embedding: torch.Tensor,
        depends_on: list[int],
        step_type: str = "deduction",
        contradicts: list[int] | None = None,
    ) -> int:
        """Add a reasoning step to the graph.

        Args:
            step_text: The reasoning step text.
            embedding: Projected embedding (embedding_dim,).
            depends_on: 1-indexed step numbers this step depends on.
            step_type: Type of step (setup, computation, deduction, etc.).
            contradicts: 1-indexed step numbers this step contradicts.

        Returns:
            0-indexed node index in the CellComplex.
        """
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim {embedding.shape[-1]} != {self.embedding_dim}"
            )
        if embedding.dim() > 1:
            embedding = embedding.squeeze(0)

        # Add node
        node_idx = self.cell_complex.add_0_cell(embedding, cell_type=step_type)

        # Store step metadata
        self.steps.append({
            "text": step_text,
            "type": step_type,
            "depends_on": depends_on,
            "node_idx": node_idx,
        })

        # Filter valid dependencies (1-indexed → 0-indexed)
        valid_deps = [d - 1 for d in depends_on if 1 <= d <= len(self.steps) - 1]

        # Fallback: if no valid deps and this isn't the first step, connect to previous
        if not valid_deps and node_idx > 0:
            valid_deps = [node_idx - 1]

        # Add dependency edges
        for dep_idx in valid_deps:
            edge_emb = torch.zeros(self.embedding_dim)
            # Encode dependency strength as cosine similarity in dim 0
            dep_emb = self.cell_complex.get_embeddings(0)[dep_idx]
            cos_sim = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0), dep_emb.unsqueeze(0),
            ).item()
            edge_emb[0] = cos_sim

            direction = "forward" if dep_idx < node_idx else "backward"
            self.cell_complex.add_1_cell(
                dep_idx, node_idx, edge_emb, relation_type=direction,
            )
            self.edge_directions.append(direction)
            self._edge_pairs.append((dep_idx, node_idx))

        # Add contradiction edges
        if contradicts:
            valid_contras = [c - 1 for c in contradicts if 1 <= c <= len(self.steps) - 1]
            for contra_idx in valid_contras:
                edge_emb = torch.zeros(self.embedding_dim)
                edge_emb[0] = -1.0  # negative similarity = contradiction
                self.cell_complex.add_1_cell(
                    contra_idx, node_idx, edge_emb, relation_type="contradiction",
                )
                self.edge_directions.append("contradiction")
                self._edge_pairs.append((contra_idx, node_idx))

        # Scan for new triangles
        self._detect_triangles(node_idx)

        return node_idx

    def _detect_triangles(self, new_node: int):
        """Check if adding new_node created any triangles."""
        # Get neighbors of new_node
        neighbors = set()
        for src, tgt in self._edge_pairs:
            if src == new_node:
                neighbors.add(tgt)
            elif tgt == new_node:
                neighbors.add(src)

        # Check if any pair of neighbors are also connected
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                a, b = neighbor_list[i], neighbor_list[j]
                if self._are_connected(a, b):
                    # Triangle found: (a, b, new_node)
                    self._add_triangle(a, b, new_node)

    def _are_connected(self, a: int, b: int) -> bool:
        for src, tgt in self._edge_pairs:
            if (src == a and tgt == b) or (src == b and tgt == a):
                return True
        return False

    def _find_edge_index(self, a: int, b: int) -> int | None:
        """Find the 1-cell index for edge (a,b) or (b,a)."""
        for idx, (src, tgt) in enumerate(self._edge_pairs):
            if (src == a and tgt == b) or (src == b and tgt == a):
                return idx
        return None

    def _add_triangle(self, a: int, b: int, c: int):
        """Add a 2-cell for the triangle (a, b, c)."""
        e_ab = self._find_edge_index(a, b)
        e_bc = self._find_edge_index(b, c)
        e_ac = self._find_edge_index(a, c)

        if e_ab is not None and e_bc is not None and e_ac is not None:
            try:
                tri_emb = torch.zeros(self.embedding_dim)
                # Check if any edge is backward (potential circular reasoning)
                has_backward = any(
                    self.edge_directions[e] == "backward"
                    for e in [e_ab, e_bc, e_ac]
                )
                tri_emb[0] = -1.0 if has_backward else 1.0
                self.cell_complex.add_2_cell(
                    [e_ab, e_bc, e_ac], tri_emb,
                    cell_type="circular" if has_backward else "triangulation",
                )
            except (ValueError, AssertionError):
                pass  # Invalid triangle boundary — skip

    def get_snapshot(self) -> CellComplex:
        """Return a cloned CellComplex for read-only topology analysis."""
        return self.cell_complex.clone()

    def has_backward_edges(self) -> bool:
        """Check if the graph contains any backward (circular) edges."""
        return "backward" in self.edge_directions
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_graph_constructor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/metacog/graph_constructor.py tests/test_metacog/test_graph_constructor.py
git commit -m "feat: ReasoningGraphConstructor — LLM steps to CellComplex"
```

---

### Task 3: Topology Analyzer

**Files:**
- Create: `src/metacog/topology_analyzer.py`
- Create: `src/metacog/health_report.py`
- Test: `tests/test_metacog/test_topology_analyzer.py`

Wraps existing spectral pipeline to analyze reasoning graph health.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metacog/test_topology_analyzer.py
"""Tests for topology analysis on reasoning graphs."""
import torch
import pytest
from src.metacog.graph_constructor import ReasoningGraphConstructor
from src.metacog.topology_analyzer import TopologyAnalyzer
from src.metacog.health_report import ReasoningHealthReport


EMBEDDING_DIM = 128


def _make_linear_graph(n_steps=5):
    """Build a simple linear reasoning chain."""
    gc = ReasoningGraphConstructor(embedding_dim=EMBEDDING_DIM)
    for i in range(n_steps):
        deps = [i] if i > 0 else []
        gc.add_step(f"Step {i+1}", torch.randn(EMBEDDING_DIM), depends_on=deps, step_type="deduction")
    return gc


def _make_circular_graph():
    """Build a graph with circular dependencies."""
    gc = ReasoningGraphConstructor(embedding_dim=EMBEDDING_DIM)
    gc.add_step("A", torch.randn(EMBEDDING_DIM), depends_on=[], step_type="setup")
    gc.add_step("B", torch.randn(EMBEDDING_DIM), depends_on=[1], step_type="deduction")
    gc.add_step("C", torch.randn(EMBEDDING_DIM), depends_on=[1, 2], step_type="deduction")
    return gc


def test_analyzer_on_linear_graph():
    gc = _make_linear_graph()
    analyzer = TopologyAnalyzer()
    report = analyzer.analyze(gc.get_snapshot())
    assert isinstance(report, ReasoningHealthReport)
    assert report.num_components == 1
    assert report.gradient_energy >= 0


def test_health_report_action_continue():
    report = ReasoningHealthReport(
        curl_energy=0.1,
        gradient_energy=0.7,
        harmonic_energy=0.1,
        spectral_gap=0.2,
        num_components=1,
        betti_1=0,
        graph_density=0.5,
        action="CONTINUE",
        diagnosis="Reasoning is healthy",
    )
    assert report.action == "CONTINUE"


def test_health_report_action_intervene_curl():
    report = ReasoningHealthReport(
        curl_energy=0.6,
        gradient_energy=0.2,
        harmonic_energy=0.1,
        spectral_gap=0.2,
        num_components=1,
        betti_1=1,
        graph_density=0.5,
        action="INTERVENE",
        diagnosis="High curl energy",
    )
    assert report.action == "INTERVENE"


def test_analyzer_returns_valid_report():
    gc = _make_linear_graph(3)
    analyzer = TopologyAnalyzer()
    report = analyzer.analyze(gc.get_snapshot())
    assert 0 <= report.curl_energy <= 1 or report.curl_energy == 0.0
    assert 0 <= report.gradient_energy
    assert report.spectral_gap >= 0
    assert report.num_components >= 1
    assert report.action in ("CONTINUE", "INTERVENE", "BACKTRACK", "STOP")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_topology_analyzer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ReasoningHealthReport**

```python
# src/metacog/health_report.py
"""Reasoning health report — topology signals mapped to reasoning quality."""
from dataclasses import dataclass


@dataclass
class ReasoningHealthReport:
    """Topological diagnosis of reasoning graph health."""

    curl_energy: float          # 0-1, high = circular patterns
    gradient_energy: float      # 0-1, high = forward progress
    harmonic_energy: float      # 0-1, high = trapped/disconnected info
    spectral_gap: float         # 0+, low = bottleneck
    num_components: int         # >1 = fragmented reasoning
    betti_1: int               # independent cycles count
    graph_density: float        # edge density
    action: str                # CONTINUE, INTERVENE, BACKTRACK, STOP
    diagnosis: str             # human-readable explanation


def decide_action(
    curl_energy: float,
    gradient_energy: float,
    harmonic_energy: float,
    spectral_gap: float,
    num_components: int,
    betti_1: int,
    graph_density: float,
    curl_threshold: float = 0.4,
    gap_threshold: float = 0.05,
    harmonic_threshold: float = 0.3,
    max_betti: int = 2,
) -> ReasoningHealthReport:
    """Evaluate topology and decide action.

    Returns a ReasoningHealthReport with action and diagnosis.
    """
    problems = []

    if curl_energy > curl_threshold:
        problems.append(f"high curl energy ({curl_energy:.2f}): circular patterns detected")
    if spectral_gap < gap_threshold and spectral_gap > 0:
        problems.append(f"low spectral gap ({spectral_gap:.4f}): reasoning bottleneck")
    if harmonic_energy > harmonic_threshold:
        problems.append(f"high harmonic energy ({harmonic_energy:.2f}): disconnected steps")
    if num_components > 1:
        problems.append(f"{num_components} disconnected components: fragmented reasoning")
    if betti_1 > max_betti:
        problems.append(f"betti_1={betti_1}: too many unresolved loops")

    if problems:
        action = "INTERVENE"
        diagnosis = "; ".join(problems)
    else:
        action = "CONTINUE"
        diagnosis = "Reasoning is healthy"

    return ReasoningHealthReport(
        curl_energy=curl_energy,
        gradient_energy=gradient_energy,
        harmonic_energy=harmonic_energy,
        spectral_gap=spectral_gap,
        num_components=num_components,
        betti_1=betti_1,
        graph_density=graph_density,
        action=action,
        diagnosis=diagnosis,
    )
```

- [ ] **Step 4: Implement TopologyAnalyzer**

```python
# src/metacog/topology_analyzer.py
"""Analyze reasoning graph topology using existing spectral pipeline."""
import torch
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0
from src.metacog.health_report import ReasoningHealthReport, decide_action


class TopologyAnalyzer:
    """Compute topological features of a reasoning graph."""

    def analyze(self, cc: CellComplex) -> ReasoningHealthReport:
        """Analyze a reasoning CellComplex and return health report.

        Args:
            cc: Cloned CellComplex snapshot from ReasoningGraphConstructor.

        Returns:
            ReasoningHealthReport with topology signals and action decision.
        """
        n_nodes = cc.num_cells(0)
        n_edges = cc.num_cells(1)
        n_triangles = cc.num_cells(2)

        # Degenerate cases
        if n_nodes < 2 or n_edges == 0:
            return decide_action(
                curl_energy=0.0, gradient_energy=0.0, harmonic_energy=0.0,
                spectral_gap=0.0, num_components=1 if n_nodes > 0 else 0,
                betti_1=0, graph_density=0.0,
            )

        # Hodge decomposition on edge signal
        curl_energy, gradient_energy, harmonic_energy = self._hodge_analysis(cc)

        # Spectral gap
        spectral_gap = self._spectral_gap(cc)

        # Connected components (from L0 eigenvalues)
        num_components = self._count_components(cc)

        # Betti-1 (graph-theoretic: edges - nodes + components)
        betti_1 = max(0, n_edges - n_nodes + num_components)

        # Graph density
        max_edges = n_nodes * (n_nodes - 1) / 2
        graph_density = n_edges / max_edges if max_edges > 0 else 0.0

        return decide_action(
            curl_energy=curl_energy,
            gradient_energy=gradient_energy,
            harmonic_energy=harmonic_energy,
            spectral_gap=spectral_gap,
            num_components=num_components,
            betti_1=betti_1,
            graph_density=graph_density,
        )

    def _hodge_analysis(self, cc: CellComplex) -> tuple[float, float, float]:
        """Compute Hodge decomposition energy ratios."""
        try:
            from src.spectral.decomposition import hodge_decomposition

            # Edge signal = cosine similarity stored in dim 0
            edge_embs = cc.get_embeddings(1)
            signal = edge_embs[:, 0]

            gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)

            g_norm = gradient.norm().item()
            c_norm = curl.norm().item()
            h_norm = harmonic.norm().item()
            total = g_norm + c_norm + h_norm + 1e-8

            return c_norm / total, g_norm / total, h_norm / total
        except (RuntimeError, ValueError):
            return 0.0, 0.0, 0.0

    def _spectral_gap(self, cc: CellComplex) -> float:
        """Compute Fiedler value (smallest nonzero eigenvalue of L0)."""
        try:
            L0 = hodge_laplacian_0(cc).float()
            eigenvalues = torch.linalg.eigvalsh(L0)
            nonzero = eigenvalues[eigenvalues > 1e-5]
            if len(nonzero) > 0:
                return nonzero[0].item()
        except (RuntimeError, ValueError):
            pass
        return 0.0

    def _count_components(self, cc: CellComplex) -> int:
        """Count connected components from L0 zero eigenvalues."""
        try:
            L0 = hodge_laplacian_0(cc).float()
            eigenvalues = torch.linalg.eigvalsh(L0)
            return int((eigenvalues.abs() < 1e-5).sum().item())
        except (RuntimeError, ValueError):
            return 1
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_topology_analyzer.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/metacog/health_report.py src/metacog/topology_analyzer.py tests/test_metacog/test_topology_analyzer.py
git commit -m "feat: TopologyAnalyzer + ReasoningHealthReport for v12 metacognition"
```

---

### Task 4: Intervention Generator

**Files:**
- Create: `src/metacog/intervention.py`
- Test: `tests/test_metacog/test_intervention.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metacog/test_intervention.py
"""Tests for intervention generation from topology signals."""
import pytest
from src.metacog.health_report import ReasoningHealthReport
from src.metacog.intervention import generate_intervention


def test_no_intervention_when_healthy():
    report = ReasoningHealthReport(
        curl_energy=0.1, gradient_energy=0.7, harmonic_energy=0.1,
        spectral_gap=0.2, num_components=1, betti_1=0,
        graph_density=0.5, action="CONTINUE", diagnosis="healthy",
    )
    result = generate_intervention(report, steps=[])
    assert result is None


def test_intervention_on_high_curl():
    report = ReasoningHealthReport(
        curl_energy=0.6, gradient_energy=0.2, harmonic_energy=0.1,
        spectral_gap=0.2, num_components=1, betti_1=1,
        graph_density=0.5, action="INTERVENE",
        diagnosis="high curl energy (0.60): circular patterns detected",
    )
    result = generate_intervention(report, steps=[
        {"text": "A", "node_idx": 0},
        {"text": "B", "node_idx": 1},
        {"text": "C", "node_idx": 2},
    ])
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_intervention_on_fragmented():
    report = ReasoningHealthReport(
        curl_energy=0.1, gradient_energy=0.3, harmonic_energy=0.1,
        spectral_gap=0.2, num_components=3, betti_1=0,
        graph_density=0.2, action="INTERVENE",
        diagnosis="3 disconnected components: fragmented reasoning",
    )
    result = generate_intervention(report, steps=[])
    assert result is not None
    assert "reasoning" in result.lower() or "thread" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_intervention.py -v`

- [ ] **Step 3: Implement**

```python
# src/metacog/intervention.py
"""Generate natural language interventions from topology signals."""
from src.metacog.health_report import ReasoningHealthReport


def generate_intervention(
    report: ReasoningHealthReport,
    steps: list[dict],
) -> str | None:
    """Generate a corrective prompt from a ReasoningHealthReport.

    Args:
        report: Topology analysis results with action decision.
        steps: List of step dicts with 'text' and 'node_idx'.

    Returns:
        Intervention string to inject, or None if no intervention needed.
    """
    if report.action == "CONTINUE":
        return None

    parts = []

    if report.curl_energy > 0.4:
        parts.append(
            "Your recent reasoning steps are closely related to each other. "
            "Ensure each step adds genuinely new information or a new approach "
            "beyond what you've already established."
        )

    if report.spectral_gap < 0.05 and report.spectral_gap > 0:
        parts.append(
            "Your reasoning appears to have a bottleneck. "
            "Consider an alternative path to the conclusion, or break "
            "the current step into smaller sub-steps."
        )

    if report.harmonic_energy > 0.3:
        parts.append(
            "Some of your reasoning steps appear disconnected from the "
            "main argument. Either connect them to your chain of reasoning "
            "or set them aside."
        )

    if report.num_components > 1:
        parts.append(
            f"You have {report.num_components} separate lines of reasoning. "
            "Explain how they connect to each other before proceeding."
        )

    if report.betti_1 > 2:
        parts.append(
            f"Your reasoning has {report.betti_1} unresolved circular "
            "dependencies. Commit to the strongest chain and move forward."
        )

    if not parts:
        # Generic intervention for unspecified INTERVENE action
        parts.append(
            "Review your reasoning so far and ensure each step logically "
            "follows from the previous ones."
        )

    return " ".join(parts)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_intervention.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/metacog/intervention.py tests/test_metacog/test_intervention.py
git commit -m "feat: intervention generator — topology signals to NL corrections"
```

---

## Chunk 2: Problem Classifier + Graph Templates + The Loop

### Task 5: Problem Classifier + Graph Templates

**Files:**
- Create: `src/metacog/problem_classifier.py`
- Create: `src/metacog/graph_templates.py`
- Test: `tests/test_metacog/test_graph_templates.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metacog/test_graph_templates.py
"""Tests for problem classification and graph template generation."""
import torch
import pytest
from src.metacog.graph_templates import create_template, PROBLEM_TYPES
from src.cell_complex.cell_complex import CellComplex


def test_all_problem_types_exist():
    expected = {"SEQUENTIAL", "CONSTRAINT", "MULTI_HOP", "EXPLORATION", "VERIFICATION"}
    assert expected == set(PROBLEM_TYPES)


def test_sequential_template():
    cc = create_template("SEQUENTIAL", embedding_dim=128, num_steps=5)
    assert isinstance(cc, CellComplex)
    assert cc.num_cells(0) == 5
    assert cc.num_cells(1) == 4  # linear chain: 4 edges for 5 nodes


def test_exploration_template():
    cc = create_template("EXPLORATION", embedding_dim=128, num_steps=7)
    assert isinstance(cc, CellComplex)
    assert cc.num_cells(0) == 7


def test_unknown_type_defaults_to_sequential():
    cc = create_template("UNKNOWN", embedding_dim=128, num_steps=5)
    assert cc.num_cells(0) == 5
```

- [ ] **Step 2: Implement**

```python
# src/metacog/graph_templates.py
"""Reasoning graph templates by problem type."""
import torch
from src.cell_complex.cell_complex import CellComplex

PROBLEM_TYPES = {"SEQUENTIAL", "CONSTRAINT", "MULTI_HOP", "EXPLORATION", "VERIFICATION"}


def create_template(
    problem_type: str,
    embedding_dim: int = 128,
    num_steps: int = 6,
) -> CellComplex:
    """Create an initial reasoning graph template.

    Nodes have placeholder embeddings (zeros). The LLM fills them
    as reasoning progresses. The template defines the expected
    connectivity pattern.
    """
    cc = CellComplex(embedding_dim=embedding_dim)
    placeholder = torch.zeros(embedding_dim)

    if problem_type == "CONSTRAINT":
        # Star graph: center = goal, leaves = constraints
        for i in range(num_steps):
            cc.add_0_cell(placeholder.clone(), cell_type="constraint" if i > 0 else "goal")
        for i in range(1, num_steps):
            cc.add_1_cell(i, 0, placeholder.clone(), relation_type="constrains")

    elif problem_type == "MULTI_HOP":
        # DAG: two evidence streams merge at a conclusion
        for i in range(num_steps):
            cc.add_0_cell(placeholder.clone(), cell_type="evidence" if i < num_steps - 1 else "conclusion")
        mid = num_steps // 2
        # Stream 1: 0 → 1 → ... → mid
        for i in range(mid):
            cc.add_1_cell(i, i + 1, placeholder.clone(), relation_type="forward")
        # Stream 2: mid+1 → mid+2 → ... → last-1
        for i in range(mid + 1, num_steps - 1):
            cc.add_1_cell(i - 1 if i == mid + 1 else i, i + 1 if i < num_steps - 2 else num_steps - 1, placeholder.clone(), relation_type="forward")
        # Both merge at conclusion
        cc.add_1_cell(mid, num_steps - 1, placeholder.clone(), relation_type="forward")

    elif problem_type == "EXPLORATION":
        # Binary tree: root branches into alternatives
        for i in range(num_steps):
            cc.add_0_cell(placeholder.clone(), cell_type="branch" if i > 0 else "root")
        for i in range(1, num_steps):
            parent = (i - 1) // 2
            cc.add_1_cell(parent, i, placeholder.clone(), relation_type="forward")

    elif problem_type == "VERIFICATION":
        # Bipartite: claims on one side, evidence on other
        n_claims = num_steps // 2
        n_evidence = num_steps - n_claims
        for i in range(n_claims):
            cc.add_0_cell(placeholder.clone(), cell_type="claim")
        for i in range(n_evidence):
            cc.add_0_cell(placeholder.clone(), cell_type="evidence")
        # Each claim links to its evidence
        for i in range(min(n_claims, n_evidence)):
            cc.add_1_cell(i, n_claims + i, placeholder.clone(), relation_type="supports")

    else:
        # SEQUENTIAL (default): linear chain
        for i in range(num_steps):
            cc.add_0_cell(placeholder.clone(), cell_type="step")
        for i in range(num_steps - 1):
            cc.add_1_cell(i, i + 1, placeholder.clone(), relation_type="forward")

    return cc
```

```python
# src/metacog/problem_classifier.py
"""Classify reasoning problem type from NL prompt."""

CLASSIFICATION_PROMPT = """Classify the reasoning type needed for this problem.

Types:
- SEQUENTIAL: Steps build linearly (math, arithmetic, step-by-step logic)
- CONSTRAINT: Multiple constraints must be satisfied simultaneously
- MULTI_HOP: Evidence from multiple sources must be combined
- EXPLORATION: Need to search alternatives or try different approaches
- VERIFICATION: Claims need to be checked against evidence

Problem: {problem}

Respond with ONLY the type name (e.g., SEQUENTIAL). Nothing else."""


def classify_problem(problem: str, generate_fn=None) -> str:
    """Classify a problem into a reasoning type.

    Args:
        problem: The NL problem text.
        generate_fn: Optional callable(prompt) → response for LLM classification.
            If None, defaults to SEQUENTIAL.

    Returns:
        Problem type string.
    """
    if generate_fn is None:
        return "SEQUENTIAL"

    prompt = CLASSIFICATION_PROMPT.format(problem=problem)
    response = generate_fn(prompt).strip().upper()

    # Extract valid type from response
    from src.metacog.graph_templates import PROBLEM_TYPES
    for pt in PROBLEM_TYPES:
        if pt in response:
            return pt

    return "SEQUENTIAL"  # default fallback
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_graph_templates.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/metacog/problem_classifier.py src/metacog/graph_templates.py tests/test_metacog/test_graph_templates.py
git commit -m "feat: problem classifier + graph templates for v12 metacognition"
```

---

### Task 6: The Metacognitive Loop

**Files:**
- Create: `src/metacog/reasoning_loop.py`
- Test: `tests/test_metacog/test_reasoning_loop.py`

Wires everything together: classify → template → step → analyze → intervene → repeat.

- [ ] **Step 1: Write failing test**

```python
# tests/test_metacog/test_reasoning_loop.py
"""Tests for the metacognitive reasoning loop."""
import torch
import pytest
from src.metacog.reasoning_loop import MetacognitiveReasoner


def test_reasoner_init():
    """Test that the reasoner initializes without LLM (mock mode)."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    assert reasoner is not None


def test_mock_reasoning():
    """Test full loop with mock LLM (returns canned steps)."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert "answer" in result
    assert "steps" in result
    assert "health_reports" in result
    assert len(result["steps"]) > 0


def test_mock_reasoning_with_topology():
    """Test that topology analysis runs on the mock reasoning graph."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert len(result["health_reports"]) > 0
    for report in result["health_reports"]:
        assert hasattr(report, "action")
```

- [ ] **Step 2: Implement MetacognitiveReasoner**

```python
# src/metacog/reasoning_loop.py
"""The metacognitive reasoning loop — plan, execute, monitor, intervene."""

import json
import re
import torch

from src.metacog.graph_constructor import ReasoningGraphConstructor
from src.metacog.topology_analyzer import TopologyAnalyzer
from src.metacog.intervention import generate_intervention
from src.metacog.problem_classifier import classify_problem
from src.metacog.graph_templates import create_template
from src.metacog.health_report import ReasoningHealthReport


class MetacognitiveReasoner:
    """Topology-guided metacognitive reasoning system.

    Orchestrates: classify → template → step → analyze → intervene → repeat.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        max_steps: int = 15,
        use_mock: bool = False,
        model=None,
        tokenizer=None,
        projection=None,
        device=None,
    ):
        self.embedding_dim = embedding_dim
        self.max_steps = max_steps
        self.use_mock = use_mock
        self.model = model
        self.tokenizer = tokenizer
        self.projection = projection
        self.device = device or torch.device("cpu")
        self.analyzer = TopologyAnalyzer()

        # Track consecutive failed interventions for diminishing returns
        self._intervention_history: list[str] = []

    def solve(self, problem: str) -> dict:
        """Solve a problem using topology-guided metacognitive reasoning.

        Args:
            problem: Natural language problem text.

        Returns:
            Dict with 'answer', 'steps', 'health_reports', 'interventions',
            'graph' (final CellComplex), 'problem_type'.
        """
        # 1. Classify problem type
        generate_fn = self._generate if not self.use_mock else None
        problem_type = classify_problem(problem, generate_fn)

        # 2. Create graph template
        template = create_template(problem_type, self.embedding_dim)

        # 3. Initialize graph constructor
        gc = ReasoningGraphConstructor(embedding_dim=self.embedding_dim)

        # 4. Reasoning loop
        health_reports = []
        interventions = []
        current_intervention = None
        consecutive_same_issue = 0
        last_issue = None

        for step_num in range(self.max_steps):
            # Generate step
            if self.use_mock:
                step_data = self._mock_step(step_num, gc.num_steps)
            else:
                step_data = self._generate_step(
                    problem, gc.steps, current_intervention,
                )

            # Parse and add to graph
            embedding = self._get_embedding(step_data.get("step", ""))
            gc.add_step(
                step_text=step_data.get("step", f"Step {step_num + 1}"),
                embedding=embedding,
                depends_on=step_data.get("depends_on", []),
                step_type=step_data.get("type", "deduction"),
                contradicts=step_data.get("contradicts", None),
            )

            # Analyze topology
            snapshot = gc.get_snapshot()
            report = self.analyzer.analyze(snapshot)
            health_reports.append(report)

            # Check for answer step
            if step_data.get("type") == "answer":
                break

            # Decide intervention
            if report.action == "INTERVENE":
                intervention = generate_intervention(report, gc.steps)

                # Diminishing returns check
                current_issue = report.diagnosis.split(":")[0] if report.diagnosis else ""
                if current_issue == last_issue:
                    consecutive_same_issue += 1
                else:
                    consecutive_same_issue = 0
                    last_issue = current_issue

                if consecutive_same_issue >= 2:
                    # Two consecutive failed interventions for same issue → stop
                    break

                current_intervention = intervention
                interventions.append(intervention)
            else:
                current_intervention = None
                consecutive_same_issue = 0
                last_issue = None

        # Assemble answer
        answer = self._assemble_answer(gc.steps)

        return {
            "answer": answer,
            "steps": gc.steps,
            "health_reports": health_reports,
            "interventions": interventions,
            "graph": gc.cell_complex,
            "problem_type": problem_type,
        }

    def _mock_step(self, step_num: int, current_steps: int) -> dict:
        """Generate a mock reasoning step for testing."""
        if step_num == 0:
            return {"step": "Identify the values", "depends_on": [], "type": "setup", "confidence": 0.9}
        elif step_num < 3:
            return {"step": f"Compute intermediate result {step_num}", "depends_on": [step_num], "type": "computation", "confidence": 0.8}
        else:
            return {"step": "The answer is 5", "depends_on": list(range(1, step_num + 1)), "type": "answer", "confidence": 0.95}

    def _generate(self, prompt: str) -> str:
        """Generate text from the LLM."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM not loaded. Use use_mock=True for testing.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.7,
                top_p=0.9, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()

    def _generate_step(self, problem: str, prior_steps: list, intervention: str | None) -> dict:
        """Generate one structured reasoning step from the LLM."""
        steps_text = ""
        for i, s in enumerate(prior_steps, 1):
            steps_text += f"Step {i}: {s.get('text', '')}\n"
        if not steps_text:
            steps_text = "(none yet)\n"

        prompt = f"""You are solving a problem step by step.
Output a JSON object with: "step" (string), "depends_on" (list of step numbers), "confidence" (float 0-1), "type" (setup/computation/deduction/verification/answer).

Problem: {problem}

Steps so far:
{steps_text}"""

        if intervention:
            prompt += f"\n[METACOGNITIVE NOTE: {intervention}]\n"

        prompt += "\nOutput ONLY the JSON object:"

        response = self._generate(prompt)
        return self._parse_step(response)

    def _parse_step(self, response: str) -> dict:
        """Parse structured step from LLM response with fallbacks."""
        # Try JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: treat entire response as step text
        return {
            "step": response[:500],
            "depends_on": [],
            "confidence": 0.5,
            "type": "deduction",
        }

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a reasoning step."""
        if self.use_mock or self.model is None:
            return torch.randn(self.embedding_dim)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, -1, :]  # last token, last layer

        if self.projection is not None:
            return self.projection(hidden.squeeze(0)).cpu()
        return hidden.squeeze(0)[:self.embedding_dim].cpu()

    def _assemble_answer(self, steps: list) -> str:
        """Extract final answer from completed steps."""
        # Look for answer-type step
        for s in reversed(steps):
            if s.get("type") == "answer":
                return s.get("text", "")

        # Fallback: return last step
        if steps:
            return steps[-1].get("text", "")
        return "Unable to determine answer."
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_metacog/test_reasoning_loop.py -v`
Expected: PASS

- [ ] **Step 4: Run full metacog test suite**

Run: `.venv/bin/python -m pytest tests/test_metacog/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/metacog/reasoning_loop.py tests/test_metacog/test_reasoning_loop.py
git commit -m "feat: MetacognitiveReasoner — the full reasoning loop"
```

---

## Chunk 3: Evaluation Harness + Integration

### Task 7: GSM8K Evaluation Harness

**Files:**
- Create: `scripts/v12_run_gsm8k.py`

- [ ] **Step 1: Create evaluation script**

```python
#!/usr/bin/env python
"""Evaluate v12 metacognitive reasoning on GSM8K.

Compares: raw Qwen, CoT, and v12 metacognitive loop.

Usage:
    PYTHONPATH=. python scripts/v12_run_gsm8k.py --mode metacog --num-problems 100
    PYTHONPATH=. python scripts/v12_run_gsm8k.py --mode cot --num-problems 100
    PYTHONPATH=. python scripts/v12_run_gsm8k.py --mode raw --num-problems 100
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metacog.reasoning_loop import MetacognitiveReasoner
from src.metacog.embedding_projection import EmbeddingProjection


def load_gsm8k(max_problems=100):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for item in ds:
        # Extract numeric answer from GSM8K format
        answer_text = item["answer"]
        # GSM8K answers end with "#### <number>"
        match = re.search(r'####\s*(.+)', answer_text)
        numeric_answer = match.group(1).strip().replace(",", "") if match else ""
        problems.append({
            "question": item["question"],
            "answer_text": answer_text,
            "numeric_answer": numeric_answer,
        })
        if len(problems) >= max_problems:
            break
    return problems


def extract_numeric(text: str) -> str:
    """Extract the final numeric answer from model output."""
    # Look for explicit "answer is X" patterns
    patterns = [
        r'(?:answer|result|total)\s*(?:is|=|:)\s*\$?([\d,]+\.?\d*)',
        r'####\s*([\d,]+\.?\d*)',
        r'\$?([\d,]+\.?\d*)\s*$',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")
    # Last number in text
    numbers = re.findall(r'[\d,]+\.?\d*', text)
    return numbers[-1].replace(",", "") if numbers else ""


def evaluate_raw(model, tokenizer, problems, device):
    """Baseline: raw LLM, no prompting structure."""
    correct = 0
    for p in problems:
        prompt = f"Solve this math problem. Give only the final numeric answer.\n\n{p['question']}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, temperature=0.1,
                                pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted = extract_numeric(response)
        if predicted == p["numeric_answer"]:
            correct += 1
    return correct / len(problems)


def evaluate_cot(model, tokenizer, problems, device):
    """CoT baseline: chain-of-thought prompting."""
    correct = 0
    for p in problems:
        prompt = f"Solve this math problem step by step. Show your work, then give the final answer.\n\n{p['question']}\n\nLet me think step by step:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, temperature=0.7,
                                top_p=0.9, do_sample=True,
                                pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted = extract_numeric(response)
        if predicted == p["numeric_answer"]:
            correct += 1
    return correct / len(problems)


def evaluate_metacog(model, tokenizer, problems, device):
    """v12 metacognitive reasoning."""
    projection = EmbeddingProjection(input_dim=model.config.hidden_size, output_dim=128).to(device)

    reasoner = MetacognitiveReasoner(
        embedding_dim=128,
        max_steps=10,
        use_mock=False,
        model=model,
        tokenizer=tokenizer,
        projection=projection,
        device=device,
    )

    correct = 0
    total_interventions = 0
    total_steps = 0

    for i, p in enumerate(problems):
        t0 = time.time()
        result = reasoner.solve(p["question"])
        elapsed = time.time() - t0

        predicted = extract_numeric(result["answer"])
        is_correct = predicted == p["numeric_answer"]
        if is_correct:
            correct += 1

        n_steps = len(result["steps"])
        n_interventions = len(result["interventions"])
        total_steps += n_steps
        total_interventions += n_interventions

        if (i + 1) % 10 == 0:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(problems)}] acc={acc:.1%} "
                  f"avg_steps={total_steps/(i+1):.1f} "
                  f"avg_interventions={total_interventions/(i+1):.1f} "
                  f"last_time={elapsed:.1f}s")

    return {
        "accuracy": correct / len(problems),
        "avg_steps": total_steps / len(problems),
        "avg_interventions": total_interventions / len(problems),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw", "cot", "metacog", "all"], default="all")
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
    )

    problems = load_gsm8k(args.num_problems)
    print(f"Loaded {len(problems)} GSM8K problems")

    results = {}

    if args.mode in ("raw", "all"):
        print("\n=== RAW (no structure) ===")
        acc = evaluate_raw(model, tokenizer, problems, device)
        results["raw"] = acc
        print(f"Accuracy: {acc:.1%}")

    if args.mode in ("cot", "all"):
        print("\n=== CHAIN-OF-THOUGHT ===")
        acc = evaluate_cot(model, tokenizer, problems, device)
        results["cot"] = acc
        print(f"Accuracy: {acc:.1%}")

    if args.mode in ("metacog", "all"):
        print("\n=== METACOGNITIVE (v12) ===")
        mc_results = evaluate_metacog(model, tokenizer, problems, device)
        results["metacog"] = mc_results
        print(f"Accuracy: {mc_results['accuracy']:.1%}")
        print(f"Avg steps: {mc_results['avg_steps']:.1f}")
        print(f"Avg interventions: {mc_results['avg_interventions']:.1f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for mode, result in results.items():
        if isinstance(result, dict):
            print(f"  {mode}: {result['accuracy']:.1%} (steps={result['avg_steps']:.1f}, interventions={result['avg_interventions']:.1f})")
        else:
            print(f"  {mode}: {result:.1%}")

    # Save
    out_path = Path("data/v12_gsm8k_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/v12_run_gsm8k.py
git commit -m "feat: GSM8K evaluation harness for v12 metacognitive reasoning"
```

---

## Summary

| Task | What | Files | Dependency |
|------|------|-------|------------|
| 0 | Phase 0: Structured output validation | `scripts/v12_phase0_validate.py` | None — **GATE** |
| 1 | Embedding projection (2048→128) | `src/metacog/embedding_projection.py` | None |
| 2 | Graph Constructor | `src/metacog/graph_constructor.py` | Task 1 |
| 3 | Topology Analyzer + Health Report | `src/metacog/topology_analyzer.py`, `health_report.py` | Task 2 |
| 4 | Intervention Generator | `src/metacog/intervention.py` | Task 3 |
| 5 | Problem Classifier + Graph Templates | `src/metacog/problem_classifier.py`, `graph_templates.py` | None |
| 6 | The Metacognitive Loop | `src/metacog/reasoning_loop.py` | Tasks 1-5 |
| 7 | GSM8K Evaluation Harness | `scripts/v12_run_gsm8k.py` | Task 6 |

**Critical path:** Task 0 (Phase 0) is the gate. If Qwen 3B can't produce structured output reliably, redesign the LLM interface before proceeding. Tasks 1-5 can be implemented in parallel locally (no GPU needed). Task 6 wires everything. Task 7 is the evaluation.

**Estimated implementation time:** Tasks 1-5: ~1 day. Task 6: ~half day. Task 7: ~half day. Phase 0 + GPU evaluation: ~1 day on vast.ai.
