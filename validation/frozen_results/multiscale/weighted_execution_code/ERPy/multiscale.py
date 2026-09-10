"""Optional fixed-window-bank projection–energy inference.

This module does not change the default detector or its facade. Each declared
window uses the existing fixed-window projection sign-flip and separately
demeaned paired log-RMS sign-flip conjunction. Bonferroni combines the windows:
``p = min(1, K * min_w max(p_R,w, p_E,w))``. Its global null is the intersection,
over all declared windows, of the corresponding component-union nulls. Valid
component p-values suffice; independence between windows is unnecessary.
Optional predeclared positive weights summing to one instead combine as
``p = min(1, min_w p_joint,w / weight_w)``. Missing-window weights are never
redistributed. ``weights=None`` retains the equal-weight K-window rule.

The bank, clean-trial rule and contact family must be declared before examining
outcomes. Missing windows contribute one and never reduce K. All windows use
the same complete-case trials across the available bank support. Conditional
invariance after selection remains an assumption. Across-contact BH retains
its own independence/positive-dependence requirements. This is a candidate
operating point, not a general calibration or clinical-utility guarantee.

Prior contribution: CRP supplies the semi-normalized cross-trial projection
representation (Miller et al., 2023, doi:10.1371/journal.pcbi.1011105). ER-detect
also offers an intertrial-CRP fixed-window t-threshold mode (van den Boom et al.,
2025, doi:10.1016/j.jneumeth.2025.110389). This candidate combines ERPy's existing
whole-trial randomization/energy conjunctions with a standard Bonferroni bank;
neither CRP projection nor fixed-window CRP detection is claimed as new here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Sequence

import numpy as np

from .crp_energy import CRPEnergyConfig, CRPEnergyResult, run_crp_energy_array
from .inference import fdr_bh


MULTISCALE_VERSION = "experimental-fixed-bank-2"
DEFAULT_WINDOWS = ((0.010, 0.090), (0.090, 0.300), (0.010, 0.300))


@dataclass(frozen=True)
class MultiScaleConfig:
    """A predeclared window bank and common component settings.

    ``component_config.response_window`` is replaced by each bank entry.
    ``component_config.correction`` must be ``fdr_bh`` or ``none`` and is used
    only for the final family operation. No window is chosen from the signal.
    """

    windows: tuple[tuple[float, float], ...] = DEFAULT_WINDOWS
    weights: tuple[float, ...] | None = None
    component_config: CRPEnergyConfig = field(default_factory=lambda: CRPEnergyConfig(
        response_window=(0.010, 0.300), baseline_window=(-0.5, -0.02),
        artifact_interval=(0.0, 0.009), canonical_energy_cv=False,
    ))

    def validate(self) -> None:
        if not self.windows:
            raise ValueError("windows must contain a predeclared nonempty bank")
        normalized = []
        for window in self.windows:
            config = replace(self.component_config, response_window=window)
            config.validate()
            normalized.append(tuple(float(x) for x in window))
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate windows are not allowed")
        if self.weights is not None:
            _validate_weights(self.weights, len(self.windows))


@dataclass
class WindowEvidence:
    declared_window: tuple[float, float]
    p_joint: float
    p_for_combination: float
    available: bool
    status: str
    result: CRPEnergyResult | None
    seed: int


@dataclass
class MultiScaleResult:
    channel: str
    p_multiscale: float
    q_multiscale: float
    significant: bool
    available: bool
    n_declared_windows: int
    n_available_windows: int
    common_clean_trial_indices: np.ndarray
    windows: tuple[WindowEvidence, ...]
    parameters: str
    detector_version: str = MULTISCALE_VERSION

    def to_record(self) -> dict:
        """Return compact provenance without descriptive waveform arrays."""
        return {
            "channel": self.channel, "p_multiscale": self.p_multiscale,
            "q_multiscale": self.q_multiscale, "significant": self.significant,
            "available": self.available, "n_declared_windows": self.n_declared_windows,
            "n_available_windows": self.n_available_windows,
            "common_clean_trial_indices": self.common_clean_trial_indices.tolist(),
            "parameters": self.parameters, "detector_version": self.detector_version,
            "windows": [{
                "declared_window": list(w.declared_window), "p_joint": w.p_joint,
                "p_for_combination": w.p_for_combination, "available": w.available,
                "status": w.status, "seed": w.seed,
                "component_result": w.result.to_record(include_arrays=False) if w.result else None,
            } for w in self.windows],
        }


def window_seed(root_seed: int | None, channel: str, window: tuple[float, float]) -> int:
    """Stable contact/window seed, invariant to bank and contact ordering."""
    if root_seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    identity = json.dumps([int(root_seed), str(channel), list(map(float, window))], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")


def conservative_bonferroni(p_values: Sequence[float]) -> float:
    """Combine the complete declared bank; nonfinite values contribute one."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("p_values must be a nonempty one-dimensional declared bank")
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError("finite p-values must lie in [0, 1]")
    return float(min(1.0, len(values) * np.min(np.where(finite, values, 1.0))))


def _validate_weights(weights: Sequence[float], count: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if (values.ndim != 1 or len(values) != count or
            not np.all(np.isfinite(values)) or np.any(values <= 0) or
            not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("weights must be finite, positive, match the full bank, and sum to one")
    return values


def conservative_weighted_bonferroni(
    p_values: Sequence[float], weights: Sequence[float],
) -> float:
    """Use the complete predeclared allocation, without missing-data reweighting.

    The weights must be fixed independently of the tested outcomes. The union
    bound gives validity under the same per-window component assumptions as
    equal-weight Bonferroni; no cross-window independence is required.
    """
    # Reuse the p-value validation; nonfinite entries remain conservative ones.
    conservative_bonferroni(p_values)
    values = np.asarray(p_values, dtype=float)
    allocation = _validate_weights(weights, len(values))
    return float(min(1.0, np.min(np.where(np.isfinite(values), values, 1.0) / allocation)))


def _window_support(times: np.ndarray, config: CRPEnergyConfig) -> np.ndarray | None:
    start = max(config.response_window[0], config.artifact_interval[1])
    stop = config.response_window[1]
    step = float(np.median(np.diff(times)))
    # A short recording must not silently turn a fixed bank entry into a
    # different response window. Normal sample-grid rounding is permitted.
    if times[0] > start + step / 2 or times[-1] < stop - step / 2:
        return None
    response = np.flatnonzero((times >= start) & (times <= stop))
    baseline = np.flatnonzero((times >= config.baseline_window[0]) & (times <= config.baseline_window[1]))
    if len(response) < 2 or len(baseline) < len(response):
        return None
    return np.r_[response, baseline[-len(response):]]


def run_multiscale_array(
    x: np.ndarray, times: np.ndarray, *, channel: str = "",
    config: MultiScaleConfig | None = None,
) -> MultiScaleResult:
    """Evaluate one contact using a fixed bank and common complete-case trials.

    Unavailable entries remain in the bank with p=1, and the raw unavailable
    value/reason is retained. Family correction is performed separately by
    ``run_multiscale_family``. This function's ``significant`` is unadjusted.
    """
    config = config or MultiScaleConfig()
    config.validate()
    values = np.asarray(x, dtype=float)
    time_values = np.asarray(times, dtype=float)
    if values.ndim != 2 or time_values.ndim != 1 or values.shape[1] != len(time_values):
        raise ValueError("x must be trial by time with a matching one-dimensional time vector")
    if len(time_values) < 2 or not np.all(np.isfinite(time_values)):
        raise ValueError("times must contain at least two finite samples")
    steps = np.diff(time_values)
    median = float(np.median(steps))
    if median <= 0 or not np.all(steps > 0) or not np.allclose(steps, median, rtol=1e-6, atol=max(1e-12, abs(median)*1e-9)):
        raise ValueError("times must be uniformly sampled and strictly increasing")
    components = [replace(config.component_config, response_window=tuple(window),
                          random_state=window_seed(config.component_config.random_state, channel, window))
                  for window in config.windows]
    supports = [_window_support(time_values, c) for c in components]
    present = [s for s in supports if s is not None]
    union = np.unique(np.concatenate(present)) if present else np.array([], dtype=int)
    clean = np.flatnonzero(np.all(np.isfinite(values[:, union]), axis=1)) if len(union) else np.array([], dtype=int)
    # Keep original trial indexing in component records while imposing the
    # same complete-case set even where a narrower window was itself finite.
    common_values = values.copy()
    common_values[np.setdiff1d(np.arange(len(values)), clean), :] = np.nan
    windows = []
    for window, component, support in zip(config.windows, components, supports):
        result = None
        if support is None:
            status = "declared_window_unavailable"
            p = np.nan
        else:
            result = run_crp_energy_array(common_values, time_values, channel=channel, config=component)
            status = result.qc_status
            p = float(result.p_joint)
        available = bool(np.isfinite(p) and status == "pass")
        windows.append(WindowEvidence(tuple(window), p, p if available else 1.0,
                                      available, status, result, int(component.random_state)))
    window_ps = [w.p_for_combination for w in windows]
    p = (conservative_bonferroni(window_ps) if config.weights is None else
         conservative_weighted_bonferroni(window_ps, config.weights))
    available = any(w.available for w in windows)
    return MultiScaleResult(
        str(channel), p, np.nan,
        bool(available and p <= config.component_config.alpha), available,
        len(windows), sum(w.available for w in windows), clean, tuple(windows),
        json.dumps(asdict(config), sort_keys=True),
    )


def run_multiscale_family(
    x: np.ndarray, times: np.ndarray, *, channels: Sequence[str] | None = None,
    config: MultiScaleConfig | None = None,
) -> list[MultiScaleResult]:
    """Evaluate trial × time × contact data; adjust all declared contacts.

    Supply the predeclared non-stimulation contact family. An entirely
    unavailable contact still contributes p=1 to this conservative family.
    Its availability flag remains false; this does not invent a measured
    negative response or certify response-derived quality-control selection.
    """
    config = config or MultiScaleConfig()
    config.validate()
    values = np.asarray(x, dtype=float)
    if values.ndim != 3 or values.shape[2] == 0:
        raise ValueError("x must contain a nonempty trial by time by contact family")
    labels = list(channels) if channels is not None else [str(i) for i in range(values.shape[2])]
    if len(labels) != values.shape[2] or len(set(labels)) != len(labels):
        raise ValueError("channels must be unique and match the contact dimension")
    results = [run_multiscale_array(values[:, :, c], times, channel=str(label), config=config)
               for c, label in enumerate(labels)]
    ps = np.array([r.p_multiscale for r in results])
    if str(config.component_config.correction).strip().lower() == "fdr_bh":
        qs, calls = fdr_bh(ps, alpha=config.component_config.alpha)
    else:
        qs, calls = ps, ps <= config.component_config.alpha
    for r, q, call in zip(results, qs, calls):
        r.q_multiscale = float(q)
        r.significant = bool(r.available and call)
    return results
