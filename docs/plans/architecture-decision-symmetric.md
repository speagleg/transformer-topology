# Architecture Decision: Hierarchical over Symmetric

**Date:** 2026-02-13
**Status:** Decided

## Decision

Pursuing the **hierarchical executive** architecture (ExecutiveReasoningLoop) as the primary model. The symmetric co-processing architecture (ReasoningLoop) is archived as a baseline reference.

## Context

Two reasoning loop architectures were evaluated:

- **Symmetric (ReasoningLoop):** GNN and TAT alternate as equal partners, blending outputs at each iteration. No control hierarchy.
- **Hierarchical (ExecutiveReasoningLoop):** GNN acts as executive, producing ControlSignal (frequency_gate, spatial_focus, confidence_weights, diffusion_time, wave_damping) that directs TAT behavior. Includes optional wave dynamics (heat diffusion + neural ODE wave propagation).

## Evidence

Benchmark suite results on Lambda Labs 8x A100 (2026-02-13). Symmetric was consistently outperformed by hierarchical across all tasks and was ~3x slower per epoch. Training was killed early for symmetric variants to save GPU budget.

Hierarchical also enables wave dynamics control, spectral gating, and other extensions not possible with the symmetric pattern.

## What's Archived

The following code is retained but marked as archived:

- `src/reasoning_loop/loop.py` — ReasoningLoop class (symmetric)
- `src/benchmarks/run_comparison.py` — SymmetricMultiHopModel class
- `src/benchmarks/model.py` — Phase 1 MultiHopReasoningModel (uses symmetric loop)
- `src/benchmarks/phase2_model.py` — Phase 2 model (uses symmetric loop)

These are still importable and testable. Configs no longer include "symmetric" as a default variant.

## Going Forward

- Default model_variants in configs: `[hierarchical, hierarchical_nowave]`
- New features (wave gating, band-pass filters, L1 dynamics) build on the hierarchical pattern
- The `hierarchical_nowave` variant serves as the ablation baseline instead of symmetric
