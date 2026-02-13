"""Executive diagnostics collector for analyzing control signal behavior.

Captures and aggregates statistics about what the hierarchical executive
actually learns: frequency gating, spatial focus, confidence weighting,
diffusion time, wave damping, and iteration counts.
"""

import math

import torch
import torch.nn as nn


class DiagnosticCollector:
    """Captures and aggregates executive control signal statistics.

    Designed to work with HierarchicalMultiHopModel which returns diagnostics
    from the ExecutiveReasoningLoop. Collects per-sample data during evaluation,
    then summarizes into aggregate statistics.
    """

    def __init__(self):
        self._records: list[dict] = []

    def collect(self, model: nn.Module, dataset, device: torch.device) -> list[dict]:
        """Run model on dataset, capture diagnostics from each sample.

        The model must have an `executive_loop` attribute that returns
        (output, num_iters, diagnostics) with control_signals in diagnostics.

        Args:
            model: A HierarchicalMultiHopModel (or similar with executive_loop).
            dataset: Dataset of (CellComplex, query, target, answer) tuples.
            device: Device to run on.

        Returns:
            List of per-sample diagnostic records.
        """
        model.eval()
        self._records = []

        with torch.no_grad():
            for i in range(len(dataset)):
                cc, query, target, answer = dataset[i]
                cc = cc.clone().to(device)

                # Run executive loop directly to get diagnostics
                output, num_iters, diagnostics = model.executive_loop(cc)

                record = self._extract_record(diagnostics, num_iters, answer)
                self._records.append(record)

        return self._records

    @staticmethod
    def _extract_record(diagnostics: dict, num_iters: int, answer: int) -> dict:
        """Extract a single diagnostic record from loop output."""
        control_signals = diagnostics.get('control_signals', [])

        record = {
            'num_iterations': num_iters,
            'harmonic_energies': diagnostics.get('harmonic_energies', []),
            'convergence_deltas': diagnostics.get('convergence_deltas', []),
            'answer': answer,
        }

        if control_signals:
            last_cs = control_signals[-1]
            # Frequency gate statistics
            fg = last_cs.frequency_gate.detach().cpu()
            record['frequency_gate_mean'] = fg.mean().item()
            record['frequency_gate_std'] = fg.std().item() if fg.numel() > 1 else 0.0
            record['frequency_gate_entropy'] = _entropy(fg).item()

            # Spatial focus statistics
            sf = last_cs.spatial_focus.detach().cpu()
            record['spatial_focus_mean'] = sf.mean().item()
            record['spatial_focus_std'] = sf.std().item() if sf.numel() > 1 else 0.0
            record['spatial_focus_gini'] = _gini(sf).item()

            # Confidence weights
            cw = last_cs.confidence_weights.detach().cpu()
            record['confidence_mean'] = cw.mean().item()
            record['confidence_std'] = cw.std().item() if cw.numel() > 1 else 0.0

            # Diffusion time and wave damping
            record['diffusion_time'] = last_cs.diffusion_time.detach().cpu().item()
            record['wave_damping'] = last_cs.wave_damping.detach().cpu().item()

        return record

    def summarize(self) -> dict:
        """Aggregate diagnostic records into summary statistics.

        Returns:
            Dict with aggregate statistics across all samples:
                - frequency_gate: {mean, std, entropy}
                - spatial_focus: {mean, std, gini}
                - confidence: {mean, std}
                - diffusion_time: {mean, std}
                - wave_damping: {mean, std}
                - num_iterations: {mean, std, histogram}
                - harmonic_energy: {initial_mean, final_mean, delta_mean}
        """
        if not self._records:
            return {}

        summary = {}

        # Iteration statistics
        iters = [r['num_iterations'] for r in self._records]
        summary['num_iterations'] = {
            'mean': _mean(iters),
            'std': _std(iters),
            'histogram': _histogram(iters),
        }

        # Harmonic energy trajectories
        he_initial = [r['harmonic_energies'][0] for r in self._records
                      if r['harmonic_energies']]
        he_final = [r['harmonic_energies'][-1] for r in self._records
                    if r['harmonic_energies']]
        summary['harmonic_energy'] = {
            'initial_mean': _mean(he_initial),
            'final_mean': _mean(he_final),
            'delta_mean': _mean([f - i for i, f in zip(he_initial, he_final)]),
        }

        # Control signal aggregates (only for records that have them)
        _agg_field(summary, self._records, 'frequency_gate_mean', 'frequency_gate', 'mean')
        _agg_field(summary, self._records, 'frequency_gate_std', 'frequency_gate', 'std')
        _agg_field(summary, self._records, 'frequency_gate_entropy', 'frequency_gate', 'entropy')

        _agg_field(summary, self._records, 'spatial_focus_mean', 'spatial_focus', 'mean')
        _agg_field(summary, self._records, 'spatial_focus_std', 'spatial_focus', 'std')
        _agg_field(summary, self._records, 'spatial_focus_gini', 'spatial_focus', 'gini')

        _agg_field(summary, self._records, 'confidence_mean', 'confidence', 'mean')
        _agg_field(summary, self._records, 'confidence_std', 'confidence', 'std')

        _agg_field(summary, self._records, 'diffusion_time', 'diffusion_time', 'mean')
        _agg_field(summary, self._records, 'wave_damping', 'wave_damping', 'mean')

        # Also compute std for diffusion_time and wave_damping
        dt_vals = [r['diffusion_time'] for r in self._records if 'diffusion_time' in r]
        wd_vals = [r['wave_damping'] for r in self._records if 'wave_damping' in r]
        if dt_vals:
            summary.setdefault('diffusion_time', {})['std'] = _std(dt_vals)
        if wd_vals:
            summary.setdefault('wave_damping', {})['std'] = _std(wd_vals)

        return summary


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of a probability-like vector (values in [0,1])."""
    p = probs.clamp(1e-8, 1.0 - 1e-8)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def _gini(values: torch.Tensor) -> torch.Tensor:
    """Compute Gini coefficient of a non-negative vector."""
    if values.numel() <= 1:
        return torch.tensor(0.0)
    sorted_vals = values.sort().values
    n = sorted_vals.numel()
    indices = torch.arange(1, n + 1, dtype=torch.float32)
    return (2.0 * (indices * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _histogram(values: list[int]) -> dict[int, int]:
    """Count occurrences of each integer value."""
    hist: dict[int, int] = {}
    for v in values:
        hist[v] = hist.get(v, 0) + 1
    return dict(sorted(hist.items()))


def _agg_field(summary: dict, records: list[dict],
               field_name: str, group_name: str, stat_name: str):
    """Aggregate a single field from records into summary."""
    vals = [r[field_name] for r in records if field_name in r]
    if vals:
        summary.setdefault(group_name, {})[stat_name] = _mean(vals)
