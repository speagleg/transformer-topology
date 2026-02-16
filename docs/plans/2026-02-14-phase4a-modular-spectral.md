# Phase 4a: Modular Spectral Dynamics

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the monolithic wave dynamics module with a pluggable spectral filter system. Any analytic filter h(λ) can be swapped via config. Add wave strength gate for per-task bypass.

**Architecture:** SpectralFilter base class with shared eigendecomposition. Filter registry maps config strings to filter classes. WaveDynamics updated to accept filter config. Executive loop unchanged — it still produces ControlSignal with diffusion_time/wave_damping.

**Tech Stack:** PyTorch, existing cell complex / spectral infrastructure

---

## Design

### SpectralFilter Base Class

```python
class SpectralFilter(nn.Module):
    """Base class for analytic spectral filters on cell complexes.

    All filters share the pattern:
    1. Eigendecompose the Hodge Laplacian (dim=0,1,2)
    2. Apply filter h(λ) to eigenvalues
    3. Inverse transform back to spatial domain
    """
    def __init__(self, dim: int = 0):
        self.dim = dim  # Which Hodge Laplacian

    def filter_response(self, eigenvalues, **params) -> Tensor:
        """Compute h(λ). Override in subclasses."""
        raise NotImplementedError

    def forward(self, cc, signal, **params) -> Tensor:
        # Shared eigendecomp + filter + inverse transform
```

### Filter Types

| Filter | h(λ) | Learnable Params | Config key |
|--------|-------|------------------|------------|
| Heat | exp(-λt) | 0 | `heat` |
| Band-pass | exp(-(λ-λ₀)²/2σ²) | 2 (center, bandwidth) | `bandpass` |
| True wave | cos(√λ·t) | 0 | `wave_cosine` |
| Chebyshev | Σ θ_k T_k(λ̃) | K (polynomial order) | `chebyshev` |
| Schrodinger | cos(λt) (real part) | 0 | `schrodinger` |
| Spectral wavelet | mean of g(s_k·λ) at multiple scales | n_scales | `wavelet` |

### Wave Strength Gate

Learned scalar that controls how much spectral dynamics contribute:

```python
self.wave_strength = nn.Parameter(torch.tensor(1.0))
# In forward:
wave_contribution = torch.sigmoid(self.wave_strength) * spectral_output
gnn_out = gnn_out + wave_contribution  # residual
```

This lets the model learn to zero out waves on tasks where they hurt.

### Config Changes

```yaml
wave:
  filter_type: heat          # heat, bandpass, wave_cosine, chebyshev, schrodinger, wavelet
  laplacian_dim: 0           # 0, 1, 2
  wave_strength_gate: true   # learnable bypass gate
  chebyshev_order: 5         # only for chebyshev
  wavelet_scales: 4          # only for wavelet
  use_neural_ode: true       # keep the WavePropagation ODE alongside spectral filter
```

### File Changes

| Action | File | Description |
|--------|------|-------------|
| NEW | `src/wave/spectral_filters.py` | SpectralFilter base + 6 filter implementations |
| MODIFY | `src/wave/dynamics.py` | WaveDynamics accepts filter_type config, adds strength gate |
| MODIFY | `src/reasoning_loop/executive_loop.py` | Pass filter config through |
| MODIFY | `config/*.yaml` | Add wave config section |
| NEW | `tests/test_wave/test_spectral_filters.py` | Tests for all 6 filters |
| MODIFY | `tests/test_wave/test_dynamics.py` | Update for new WaveDynamics interface |

---

## Tasks

### Task 1: SpectralFilter base class + Heat filter

**Files:**
- Create: `src/wave/spectral_filters.py`
- Test: `tests/test_wave/test_spectral_filters.py`

Implement SpectralFilter base with shared eigendecomp logic. Implement HeatFilter as the first concrete filter (matches existing HeatDiffusion behavior exactly).

### Task 2: Band-pass + True wave + Schrodinger filters

**Files:**
- Modify: `src/wave/spectral_filters.py`
- Modify: `tests/test_wave/test_spectral_filters.py`

Three analytic filters that take the same (eigenvalues, diffusion_time) signature.

### Task 3: Chebyshev + Spectral wavelet filters

**Files:**
- Modify: `src/wave/spectral_filters.py`
- Modify: `tests/test_wave/test_spectral_filters.py`

These have learnable parameters (polynomial coefficients, scale parameters).

### Task 4: Filter registry + L1/L2 support

**Files:**
- Modify: `src/wave/spectral_filters.py`

Add FILTER_REGISTRY dict mapping config strings to classes. Test all filters work on dim=0,1,2.

### Task 5: Wave strength gate + WaveDynamics refactor

**Files:**
- Modify: `src/wave/dynamics.py`
- Modify: `tests/test_wave/test_dynamics.py`

Update WaveDynamics to use SpectralFilter instead of HeatDiffusion. Add wave_strength_gate parameter.

### Task 6: Executive loop + config integration

**Files:**
- Modify: `src/reasoning_loop/executive_loop.py`
- Modify: `config/benchmark_suite.yaml`
- Modify: `config/small.yaml`

Wire filter_type and wave config through ExecutiveReasoningLoop constructor. Update configs.

### Task 7: Full test suite + backward compatibility

Run all ~270 existing tests. Ensure default config (filter_type='heat', no wave_strength_gate) produces identical behavior to current code.
