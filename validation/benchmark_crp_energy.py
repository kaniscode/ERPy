#!/usr/bin/env python3
"""Deterministic synthetic operating-characteristic benchmark for CRP-energy.

The benchmark contains no patient data.  It creates stationary colored-noise
epochs with known null or injected, trial-reproducible biphasic responses and
then runs ERPy's public detector API.  A balanced factorial design varies total
trial count, injected response-to-noise ratio, and the fraction of target
trials randomly masked after upstream QC (represented by nonfinite analysis
windows).

The CRP-energy result is compared with ERPy's operational CRP-only, paired-RMS,
Kundu-Rolston, and N1-z decisions under the same simulated acquisitions.  Those
comparators are exploratory implementation checks; their definitions remain
distinct from exact reproductions of the original publications' cohorts,
preprocessing, and validation protocols.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats

import ERPy as ep
from ERPy.crp_energy import CRP_ENERGY_DETECTOR_VERSION


BENCHMARK_SCHEMA_VERSION = "1.0.0"
ERPY_COMPATIBILITY_ID = ep.CHECKPOINT_COMPATIBILITY_ID
NULL_TRUTH = "null_noise"
INJECTED_TRUTH = "injected"
DETECTORS = (
    "crp_energy",
    "crp_only_fdr",
    "paired_rms_fdr",
    "kundu_rolston",
    "n1_z",
)
MULTIPLICITY_ADJUSTMENT = {
    "crp_energy": (
        "Benjamini-Hochberg adjustment across simulated channel family"
    ),
    "crp_only_fdr": (
        "Benjamini-Hochberg adjustment across simulated channel family"
    ),
    "paired_rms_fdr": (
        "Benjamini-Hochberg adjustment across simulated channel family"
    ),
    "kundu_rolston": "unadjusted operational threshold",
    "n1_z": "unadjusted operational threshold",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved synthetic design and detector settings."""

    master_seed: int = 20_260_830
    n_replicates: int = 40
    trial_counts: tuple[int, ...] = (8, 12, 24, 36)
    snr_values: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    artifact_fractions: tuple[float, ...] = (0.0, 0.25, 0.5)
    n_family_channels: int = 4
    sfreq: float = 250.0
    tmin: float = -0.36
    tmax: float = 0.34
    baseline_window: tuple[float, float] = (-0.32, -0.016)
    response_window: tuple[float, float] = (0.016, 0.32)
    artifact_interval: tuple[float, float] = (0.0, 0.016)
    comparator_response_window: tuple[float, float] = (0.016, 0.12)
    noise_sd_uv: float = 10.0
    ar1_phi: float = 0.85
    trial_scale_log_sd: float = 0.18
    amplitude_cv: float = 0.15
    latency_jitter_sd_s: float = 0.003
    alpha: float = 0.05
    min_clean_trials: int = 8
    n_permutations: int = 5_000
    max_exact_trials: int = 16
    max_exact_reproducibility_trials: int = 12
    projection_start_s: float = 0.010
    projection_step_s: float = 0.005
    eps: float = 1e-12
    confidence: float = 0.95
    bootstrap_resamples: int = 5_000

    def validate(self) -> None:
        """Raise ``ValueError`` when the requested design is invalid."""

        if self.n_replicates < 1:
            raise ValueError("n_replicates must be at least one")
        if not self.trial_counts or min(self.trial_counts) < 2:
            raise ValueError("trial_counts must contain values of at least two")
        if not self.snr_values or min(self.snr_values) < 0:
            raise ValueError("snr_values must be nonempty and nonnegative")
        if 0.0 not in self.snr_values:
            raise ValueError("snr_values must include 0.0 as the known null")
        if not self.artifact_fractions or not all(
            0.0 <= value < 1.0 for value in self.artifact_fractions
        ):
            raise ValueError("artifact_fractions must lie in [0, 1)")
        if self.n_family_channels < 3:
            raise ValueError("n_family_channels must be at least three")
        if self.sfreq <= 20.0:
            raise ValueError("sfreq must exceed 20 Hz for all comparators")
        if not 0.0 <= self.ar1_phi < 1.0:
            raise ValueError("ar1_phi must lie in [0, 1)")
        if self.noise_sd_uv <= 0:
            raise ValueError("noise_sd_uv must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0 <= int(self.max_exact_trials) <= 18:
            raise ValueError("max_exact_trials must lie in [0, 18]")
        if not 0 <= int(self.max_exact_reproducibility_trials) <= 18:
            raise ValueError(
                "max_exact_reproducibility_trials must lie in [0, 18]"
            )
        if int(self.n_permutations) < 1:
            raise ValueError("n_permutations must be positive")
        if (
            not np.isfinite(self.projection_start_s)
            or self.projection_start_s < 0
        ):
            raise ValueError("projection_start_s must be nonnegative and finite")
        if (
            not np.isfinite(self.projection_step_s)
            or self.projection_step_s <= 0
        ):
            raise ValueError("projection_step_s must be positive and finite")
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be positive and finite")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("bootstrap_resamples must be at least 1,000")

    def to_json_record(self) -> dict[str, Any]:
        """Return a JSON-compatible, explicitly versioned configuration."""

        record = asdict(self)
        record["benchmark_schema_version"] = BENCHMARK_SCHEMA_VERSION
        record["detectors"] = list(DETECTORS)
        record["truth_definition"] = {
            NULL_TRUTH: "stationary colored noise; no poststimulus injection",
            "injected": (
                "RMS-normalized, biphasic response with positive trial amplitudes "
                "and small latency jitter"
            ),
        }
        record["artifact_definition"] = (
            "Randomly selected target-channel trials are marked nonfinite in the "
            "matched baseline and response windows, emulating upstream QC rejection"
        )
        record["primary_decision"] = (
            "BH-adjusted q_joint <= alpha, where p_joint=max(p_crp,p_energy)"
        )
        record["primary_random_stream_derivation"] = (
            "sha256-v1(root-seed, exact channel label; first 63 bits)"
        )
        return record


@dataclass(frozen=True)
class ScenarioSpec:
    """One independent four-channel acquisition replicate in the design."""

    replicate: int
    n_trials: int
    snr: float
    artifact_fraction: float


def parse_number_list(text: str, *, cast: type) -> tuple[Any, ...]:
    """Parse a comma-delimited CLI grid while rejecting empty values."""

    values = [part.strip() for part in str(text).split(",")]
    if not values or any(not value for value in values):
        raise argparse.ArgumentTypeError("grid values must be comma-delimited numbers")
    try:
        return tuple(cast(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def scenario_specs(config: BenchmarkConfig) -> list[ScenarioSpec]:
    """Return the complete design in stable, publication-table order."""

    config.validate()
    return [
        ScenarioSpec(replicate, n_trials, snr, artifact_fraction)
        for n_trials in config.trial_counts
        for artifact_fraction in config.artifact_fractions
        for snr in config.snr_values
        for replicate in range(config.n_replicates)
    ]


def scenario_id(spec: ScenarioSpec) -> str:
    """Return a stable, human-auditable identifier for a design replicate."""

    snr_code = int(round(float(spec.snr) * 1_000))
    artifact_code = int(round(float(spec.artifact_fraction) * 1_000))
    return (
        f"n{int(spec.n_trials):03d}_a{artifact_code:04d}_"
        f"s{snr_code:05d}_r{int(spec.replicate):04d}"
    )


def _seed_sequences(
    config: BenchmarkConfig,
    spec: ScenarioSpec,
) -> tuple[np.random.SeedSequence, np.random.SeedSequence, int]:
    coordinates = [
        int(config.master_seed),
        int(spec.replicate),
        int(spec.n_trials),
        int(round(float(spec.snr) * 1_000)),
        int(round(float(spec.artifact_fraction) * 1_000)),
    ]
    root = np.random.SeedSequence(coordinates)
    simulation_seed, detector_seed_sequence = root.spawn(2)
    detector_seed = int(detector_seed_sequence.generate_state(1, dtype=np.uint32)[0])
    return simulation_seed, detector_seed_sequence, detector_seed


def _time_axis(config: BenchmarkConfig) -> np.ndarray:
    step = 1.0 / float(config.sfreq)
    n_samples = int(round((float(config.tmax) - float(config.tmin)) / step)) + 1
    return float(config.tmin) + np.arange(n_samples, dtype=float) * step


def _ar1_noise(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    *,
    phi: float,
    scale: float,
) -> np.ndarray:
    """Generate burn-in-corrected stationary Gaussian AR(1) noise."""

    n_trials, n_times, n_channels = shape
    burn_in = max(200, int(round(2.0 * n_times)))
    innovations = rng.normal(
        0.0,
        float(scale) * math.sqrt(max(1.0 - float(phi) ** 2, 1e-12)),
        size=(n_trials, n_times + burn_in, n_channels),
    )
    colored = signal.lfilter(
        [1.0],
        [1.0, -float(phi)],
        innovations,
        axis=1,
    )
    return np.asarray(colored[:, burn_in:, :], dtype=float)


def response_waveform(times: np.ndarray, latency_shift_s: float = 0.0) -> np.ndarray:
    """Return the RMS-normalized injected biphasic response."""

    response_times = np.asarray(times, dtype=float) - float(latency_shift_s)
    waveform = (
        -np.exp(-0.5 * ((response_times - 0.042) / 0.010) ** 2)
        + 0.72 * np.exp(-0.5 * ((response_times - 0.105) / 0.022) ** 2)
        - 0.24 * np.exp(-0.5 * ((response_times - 0.205) / 0.040) ** 2)
    )
    rms = float(np.sqrt(np.mean(waveform * waveform)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("injected response waveform is degenerate")
    return waveform / rms


def simulate_scenario(
    config: BenchmarkConfig,
    spec: ScenarioSpec,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Create one synthetic acquisition and its exact truth provenance."""

    config.validate()
    simulation_seed, _, detector_seed = _seed_sequences(config, spec)
    rng = np.random.default_rng(simulation_seed)
    times = _time_axis(config)
    channels = ["target"] + [
        f"reference_{index:02d}" for index in range(1, config.n_family_channels)
    ]
    values = _ar1_noise(
        rng,
        (int(spec.n_trials), len(times), int(config.n_family_channels)),
        phi=float(config.ar1_phi),
        scale=float(config.noise_sd_uv),
    )
    scales = rng.lognormal(
        mean=-0.5 * float(config.trial_scale_log_sd) ** 2,
        sigma=float(config.trial_scale_log_sd),
        size=(int(spec.n_trials), 1, int(config.n_family_channels)),
    )
    values *= scales

    response_mask = (
        (times >= float(config.response_window[0]))
        & (times <= float(config.response_window[1]))
    )
    baseline_mask = (
        (times >= float(config.baseline_window[0]))
        & (times <= float(config.baseline_window[1]))
    )
    if not response_mask.any() or int(baseline_mask.sum()) < int(response_mask.sum()):
        raise ValueError("configured epochs do not contain matched analysis windows")

    target_noise = values[:, response_mask, 0].copy()
    injected = np.zeros_like(target_noise)
    if float(spec.snr) > 0:
        latency_jitter = rng.normal(
            0.0,
            float(config.latency_jitter_sd_s),
            size=int(spec.n_trials),
        )
        amplitude_weights = np.clip(
            rng.normal(1.0, float(config.amplitude_cv), size=int(spec.n_trials)),
            0.40,
            1.60,
        )
        response_times = times[response_mask]
        for trial_index in range(int(spec.n_trials)):
            injected[trial_index] = (
                float(spec.snr)
                * float(config.noise_sd_uv)
                * float(amplitude_weights[trial_index])
                * response_waveform(
                    response_times,
                    latency_shift_s=float(latency_jitter[trial_index]),
                )
            )
        values[:, response_mask, 0] += injected

    n_artifact_trials = min(
        int(spec.n_trials),
        int(round(float(spec.artifact_fraction) * int(spec.n_trials))),
    )
    if n_artifact_trials:
        artifact_trials = np.sort(
            rng.choice(int(spec.n_trials), size=n_artifact_trials, replace=False)
        )
        analysis_mask = baseline_mask | response_mask
        values[np.ix_(artifact_trials, np.flatnonzero(analysis_mask), [0])] = np.nan
    else:
        artifact_trials = np.array([], dtype=int)
    clean_trials = np.setdiff1d(
        np.arange(int(spec.n_trials), dtype=int),
        artifact_trials,
        assume_unique=True,
    )
    noise_rms = (
        float(np.sqrt(np.mean(target_noise[clean_trials] ** 2)))
        if clean_trials.size
        else np.nan
    )
    signal_rms = (
        float(np.sqrt(np.mean(injected[clean_trials] ** 2)))
        if clean_trials.size
        else np.nan
    )
    realized_snr = (
        float(signal_rms / noise_rms)
        if np.isfinite(noise_rms) and noise_rms > 0
        else np.nan
    )
    provenance = {
        "scenario_id": scenario_id(spec),
        "replicate": int(spec.replicate),
        "scenario_seed_entropy": [int(value) for value in simulation_seed.entropy],
        "detector_random_state": int(detector_seed),
        "truth": NULL_TRUTH if float(spec.snr) == 0.0 else INJECTED_TRUTH,
        "signal_present": bool(float(spec.snr) > 0.0),
        "n_trials_total": int(spec.n_trials),
        "artifact_fraction": float(spec.artifact_fraction),
        "n_artifact_trials": int(n_artifact_trials),
        "n_clean_trials_expected": int(len(clean_trials)),
        "artifact_trial_indices": ";".join(str(int(value)) for value in artifact_trials),
        "snr_nominal": float(spec.snr),
        "noise_rms_uv_realized": noise_rms,
        "signal_rms_uv_realized": signal_rms,
        "snr_realized": realized_snr,
        "null_reference_injection_rms_uv": 0.0,
    }
    return values, times, channels, provenance


def _epochs_from_array(
    values: np.ndarray,
    times: np.ndarray,
    channels: Sequence[str],
    config: BenchmarkConfig,
) -> ep.Epochs:
    index = pd.MultiIndex.from_product(
        [range(values.shape[0]), np.asarray(times, dtype=float)],
        names=["epoch", "time"],
    )
    return ep.Epochs(
        pd.DataFrame(
            np.asarray(values, dtype=float).reshape(-1, len(channels)),
            index=index,
            columns=list(channels),
        ),
        sfreq=float(config.sfreq),
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=tuple(config.baseline_window),
        metadata={"source": "synthetic_benchmark"},
    )


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _safe_bool(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    try:
        return bool(value) if not pd.isna(value) else False
    except (TypeError, ValueError):
        return False


def _method_row(
    detections: pd.DataFrame,
    *,
    channel: str,
    method: str,
) -> pd.Series:
    selected = detections[
        detections["channel"].astype(str).eq(str(channel))
        & detections["method"].astype(str).eq(str(method))
    ]
    if selected.empty:
        return pd.Series(dtype=object)
    return selected.iloc[0]


def _crp_energy_seed_provenance(
    primary: pd.Series,
    *,
    expected_root: int,
) -> tuple[int, int, str]:
    """Return exact CRP-energy seed fields and reject lossy provenance."""

    try:
        parameters = json.loads(str(primary.get("parameters", "")))
        parameter_effective = int(parameters["random_state"])
        row_root = int(primary.get("random_state_root"))
        row_effective = int(primary.get("random_state_effective"))
    except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError("CRP-energy random-stream provenance is unavailable") from exc
    expected_root = int(expected_root)
    if row_root != expected_root:
        raise ValueError(
            "CRP-energy random_state_root does not match the scenario root"
        )
    if row_effective != parameter_effective:
        raise ValueError(
            "CRP-energy random_state_effective was altered during table assembly"
        )
    derivation = str(primary.get("random_stream_derivation", "")).strip()
    if derivation != "sha256-v1(root-seed,channel-label;63-bit)":
        raise ValueError("unexpected CRP-energy random-stream derivation")
    return expected_root, parameter_effective, derivation


def detector_records(
    detections: pd.DataFrame,
    *,
    channels: Sequence[str],
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize all operational calls without relabeling their criteria."""

    records: list[dict[str, Any]] = []
    for channel in channels:
        is_target = str(channel) == "target"
        truth = scenario["truth"] if is_target else "null_reference"
        primary = _method_row(detections, channel=channel, method="crp_energy")
        crp = _method_row(detections, channel=channel, method="crp_significance")
        rms = _method_row(detections, channel=channel, method="rms_response")
        kundu = _method_row(detections, channel=channel, method="kundu_rolston")
        peak = _method_row(detections, channel=channel, method="peak_amplitude")
        seed_root, seed_effective, seed_derivation = (
            _crp_energy_seed_provenance(
                primary,
                expected_root=int(scenario["detector_random_state"]),
            )
        )

        primary_q = _finite_float(primary.get("q_joint"))
        crp_q = _finite_float(crp.get("crp_q_value"))
        rms_q = _finite_float(rms.get("rms_paired_q_value"))
        kundu_trials = _finite_float(kundu.get("kundu_n_valid_trials"))
        peak_stat = _finite_float(peak.get("hays_n1_z"))
        definitions = (
            (
                "crp_energy",
                _safe_bool(primary.get("significant")),
                str(primary.get("qc_status", "")) == "pass"
                and np.isfinite(primary_q),
                _finite_float(primary.get("p_joint")),
                primary_q,
                _finite_float(primary.get("rms_ratio_db")),
                _finite_float(primary.get("n_trials_clean")),
            ),
            (
                "crp_only_fdr",
                _safe_bool(crp.get("crp_fdr_significant")),
                np.isfinite(crp_q),
                _finite_float(crp.get("p_value")),
                crp_q,
                _finite_float(crp.get("crp_t_value")),
                _finite_float(crp.get("n_trials")),
            ),
            (
                "paired_rms_fdr",
                _safe_bool(rms.get("rms_paired_fdr_significant")),
                np.isfinite(rms_q),
                _finite_float(rms.get("rms_wilcoxon_p_value")),
                rms_q,
                _finite_float(rms.get("rms_ratio")),
                _finite_float(rms.get("rms_n_paired_trials")),
            ),
            (
                "kundu_rolston",
                _safe_bool(kundu.get("significant")),
                np.isfinite(kundu_trials) and kundu_trials >= 1,
                np.nan,
                np.nan,
                _finite_float(kundu.get("kundu_post_median_uv")),
                kundu_trials,
            ),
            (
                "n1_z",
                _safe_bool(peak.get("significant")),
                np.isfinite(peak_stat),
                np.nan,
                np.nan,
                peak_stat,
                _finite_float(scenario.get("n_clean_trials_expected")),
            ),
        )
        for detector, call, evaluable, raw_p, adjusted_p, statistic, n_valid in definitions:
            records.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "replicate": int(scenario["replicate"]),
                    "n_trials_total": int(scenario["n_trials_total"]),
                    "artifact_fraction": float(scenario["artifact_fraction"]),
                    "n_clean_trials_expected": int(
                        scenario["n_clean_trials_expected"]
                    ),
                    "snr_nominal": float(scenario["snr_nominal"]),
                    "truth": str(truth),
                    "channel": str(channel),
                    "is_target": bool(is_target),
                    "detector": detector,
                    "multiplicity_adjustment": MULTIPLICITY_ADJUSTMENT[detector],
                    "call": bool(call) if evaluable else False,
                    "evaluable": bool(evaluable),
                    "raw_p_value": raw_p,
                    "adjusted_p_value": adjusted_p,
                    "statistic": statistic,
                    "n_valid_trials": n_valid,
                    "crp_energy_random_state_root": seed_root,
                    "crp_energy_random_state_effective": seed_effective,
                    "crp_energy_random_stream_derivation": seed_derivation,
                }
            )
    return records


def run_scenario(
    config: BenchmarkConfig,
    spec: ScenarioSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Simulate and evaluate one four-channel testing group."""

    values, times, channels, scenario = simulate_scenario(config, spec)
    epochs = _epochs_from_array(values, times, channels, config)
    detector_seed = int(scenario["detector_random_state"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        detections = epochs.detect_erp_all(
            methods=[
                "crp_energy",
                "crp_significance",
                "kundu_rolston",
                "peak_amplitude",
                "rms_response",
            ],
            min_consensus=1,
            baseline_window=tuple(config.baseline_window),
            response_window=tuple(config.response_window),
            crp_energy={
                "alpha": float(config.alpha),
                "correction": "fdr_bh",
                "min_clean_trials": int(config.min_clean_trials),
                "n_permutations": int(config.n_permutations),
                "max_exact_trials": int(config.max_exact_trials),
                "max_exact_reproducibility_trials": int(
                    config.max_exact_reproducibility_trials
                ),
                # This changes only the diagnostic canonical-energy estimate,
                # not p_crp, p_energy, p_joint, q_joint, or the detector call.
                "canonical_energy_cv": False,
                "random_state": detector_seed,
                "artifact_interval": tuple(config.artifact_interval),
                "projection_start_s": float(config.projection_start_s),
                "projection_step_s": float(config.projection_step_s),
                "eps": float(config.eps),
                "return_arrays": False,
            },
            crp_significance={
                "alpha": float(config.alpha),
                "min_trials": int(config.min_clean_trials),
                "random_state": detector_seed,
            },
            kundu_rolston={
                "baseline_window": (-0.100, -0.016),
                "response_window": tuple(config.comparator_response_window),
            },
            peak_amplitude={"z_threshold": 6.0},
            rms_response={
                "alpha": float(config.alpha),
                "response_window": tuple(config.comparator_response_window),
                "n1_z_threshold": 6.0,
            },
        )
    normalized = detector_records(
        detections,
        channels=channels,
        scenario=scenario,
    )
    target_records = {
        record["detector"]: record
        for record in normalized
        if record["is_target"]
    }
    for detector in DETECTORS:
        record = target_records[detector]
        prefix = detector.replace("_", "__")
        scenario[f"{prefix}__call"] = bool(record["call"])
        scenario[f"{prefix}__evaluable"] = bool(record["evaluable"])
        scenario[f"{prefix}__raw_p_value"] = record["raw_p_value"]
        scenario[f"{prefix}__adjusted_p_value"] = record["adjusted_p_value"]
        scenario[f"{prefix}__statistic"] = record["statistic"]

    primary = _method_row(detections, channel="target", method="crp_energy")
    primary_record = target_records["crp_energy"]
    scenario.update(
        {
            "primary_p_crp": _finite_float(primary.get("p_crp")),
            "primary_p_energy": _finite_float(primary.get("p_energy")),
            "primary_p_joint": _finite_float(primary.get("p_joint")),
            "primary_q_joint": _finite_float(primary.get("q_joint")),
            "primary_classification": str(primary.get("classification", "")),
            "primary_qc_status": str(primary.get("qc_status", "")),
            "primary_n_trials_clean": int(
                _finite_float(primary.get("n_trials_clean"))
            )
            if np.isfinite(_finite_float(primary.get("n_trials_clean")))
            else 0,
            "primary_crp_significant": _safe_bool(
                primary.get("crp_significant")
            ),
            "primary_energy_significant": _safe_bool(
                primary.get("energy_significant")
            ),
            "primary_random_state_root": int(
                primary_record["crp_energy_random_state_root"]
            ),
            "primary_random_state_effective": int(
                primary_record["crp_energy_random_state_effective"]
            ),
            "primary_random_stream_derivation": str(
                primary_record["crp_energy_random_stream_derivation"]
            ),
            "primary_energy_test_exact": _safe_bool(
                primary.get("energy_test_exact")
            ),
            "primary_energy_n_permutations": int(
                _finite_float(primary.get("energy_n_permutations"))
            )
            if np.isfinite(
                _finite_float(primary.get("energy_n_permutations"))
            )
            else 0,
            "primary_testing_family_size": int(
                _finite_float(primary.get("testing_family_size"))
            )
            if np.isfinite(_finite_float(primary.get("testing_family_size")))
            else 0,
            "primary_p_joint_identity": bool(
                np.isfinite(_finite_float(primary.get("p_joint")))
                and np.isclose(
                    _finite_float(primary.get("p_joint")),
                    max(
                        _finite_float(primary.get("p_crp")),
                        _finite_float(primary.get("p_energy")),
                    ),
                    rtol=0.0,
                    atol=1e-15,
                )
            ),
        }
    )
    return scenario, normalized


def _run_scenario_payload(
    payload: tuple[BenchmarkConfig, ScenarioSpec],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return run_scenario(*payload)


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial rate."""

    successes = int(successes)
    total = int(total)
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if total == 0:
        return np.nan, np.nan
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    z_value = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    proportion = successes / total
    denominator = 1.0 + z_value * z_value / total
    center = (
        proportion + z_value * z_value / (2.0 * total)
    ) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value * z_value / (4.0 * total * total)
        )
        / denominator
    )
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    if abs(low) < 1e-15:
        low = 0.0
    if abs(1.0 - high) < 1e-15:
        high = 1.0
    return low, high


def summarize_metrics(
    detector_results: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize unconditional and evaluable-only operating rates."""

    required = {"call", "evaluable", "truth", *group_columns}
    missing = sorted(required.difference(detector_results.columns))
    if missing:
        raise ValueError(f"detector results are missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    grouped = detector_results.groupby(
        list(group_columns),
        dropna=False,
        sort=True,
    )
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        truth_values = group["truth"].astype(str).drop_duplicates().tolist()
        if len(truth_values) != 1:
            raise ValueError("each summary group must contain exactly one truth state")
        truth = truth_values[0]
        total = int(len(group))
        evaluable = group["evaluable"].astype(bool).to_numpy()
        calls = group["call"].astype(bool).to_numpy()
        n_evaluable = int(evaluable.sum())
        n_calls = int(calls.sum())
        n_calls_evaluable = int((calls & evaluable).sum())
        call_low, call_high = wilson_interval(
            n_calls,
            total,
            confidence=confidence,
        )
        eval_low, eval_high = wilson_interval(
            n_evaluable,
            total,
            confidence=confidence,
        )
        conditional_low, conditional_high = wilson_interval(
            n_calls_evaluable,
            n_evaluable,
            confidence=confidence,
        )
        record.update(
            {
                "truth": truth,
                "metric": (
                    "adjusted_target_call_rate_under_null"
                    if truth == NULL_TRUTH
                    else "injected_target_recovery"
                ),
                "n_scenarios": total,
                "n_evaluable": n_evaluable,
                "n_calls": n_calls,
                "n_calls_evaluable": n_calls_evaluable,
                "evaluable_rate": n_evaluable / total,
                "evaluable_ci_low": eval_low,
                "evaluable_ci_high": eval_high,
                "call_rate_unconditional": n_calls / total,
                "call_rate_unconditional_ci_low": call_low,
                "call_rate_unconditional_ci_high": call_high,
                "call_rate_given_evaluable": (
                    n_calls_evaluable / n_evaluable if n_evaluable else np.nan
                ),
                "call_rate_given_evaluable_ci_low": conditional_low,
                "call_rate_given_evaluable_ci_high": conditional_high,
                "specificity": 1.0 - n_calls / total
                if truth == NULL_TRUTH
                else np.nan,
                "specificity_ci_low": 1.0 - call_high
                if truth == NULL_TRUTH
                else np.nan,
                "specificity_ci_high": 1.0 - call_low
                if truth == NULL_TRUTH
                else np.nan,
                "confidence": float(confidence),
                "interval": "Wilson score",
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    random_state: int | np.random.Generator | None = 0,
) -> tuple[float, float, float]:
    """Return a deterministic percentile bootstrap interval for a mean.

    Each input is one independent simulated acquisition family, so resampling
    occurs at the family level and retains dependence among channels within a
    family.
    """

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return np.nan, np.nan, np.nan
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be at least one")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    draws = np.empty(int(n_resamples), dtype=float)
    chunk_size = 500
    for start in range(0, int(n_resamples), chunk_size):
        stop = min(start + chunk_size, int(n_resamples))
        indices = rng.integers(
            0,
            len(array),
            size=(stop - start, len(array)),
        )
        draws[start:stop] = np.mean(array[indices], axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(np.mean(array)), float(low), float(high)


def family_scenario_results(detector_results: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw channel calls to one calibration record per family/detector."""

    required = {
        "scenario_id",
        "detector",
        "channel",
        "is_target",
        "truth",
        "call",
        "evaluable",
        "multiplicity_adjustment",
        "replicate",
        "n_trials_total",
        "artifact_fraction",
        "n_clean_trials_expected",
        "snr_nominal",
    }
    missing = sorted(required.difference(detector_results.columns))
    if missing:
        raise ValueError(f"detector results are missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (identifier, detector), group in detector_results.groupby(
        ["scenario_id", "detector"],
        sort=True,
        dropna=False,
    ):
        target = group[group["is_target"].astype(bool)]
        references = group[~group["is_target"].astype(bool)]
        if len(target) != 1:
            raise ValueError(
                f"scenario {identifier!r}, detector {detector!r} does not have "
                "exactly one target"
            )
        if references.empty or not references["truth"].astype(str).eq(
            "null_reference"
        ).all():
            raise ValueError(
                f"scenario {identifier!r} contains a non-null reference channel"
            )
        target_row = target.iloc[0]
        target_truth = str(target_row["truth"])
        if target_truth not in {NULL_TRUTH, INJECTED_TRUTH}:
            raise ValueError(f"unknown target truth {target_truth!r}")
        family_type = "all_null" if target_truth == NULL_TRUTH else "mixed_signal"
        null_rows = (
            group
            if family_type == "all_null"
            else references
        )
        calls = group["call"].astype(bool).to_numpy()
        null_evaluable = null_rows["evaluable"].astype(bool).to_numpy()
        false_calls = null_rows["call"].astype(bool).to_numpy()
        n_calls = int(calls.sum())
        n_false_calls = int(false_calls.sum())
        n_null_channels = int(len(null_rows))
        n_evaluable_null_channels = int(null_evaluable.sum())
        if n_evaluable_null_channels == 0 and n_false_calls:
            raise ValueError("an unevaluable null hypothesis cannot be called")
        rows.append(
            {
                "scenario_id": str(identifier),
                "detector": str(detector),
                "multiplicity_adjustment": str(
                    target_row["multiplicity_adjustment"]
                ),
                "family_type": family_type,
                "target_truth": target_truth,
                "replicate": int(target_row["replicate"]),
                "n_trials_total": int(target_row["n_trials_total"]),
                "artifact_fraction": float(target_row["artifact_fraction"]),
                "n_clean_trials_expected": int(
                    target_row["n_clean_trials_expected"]
                ),
                "snr_nominal": float(target_row["snr_nominal"]),
                "n_family_channels": int(len(group)),
                "n_null_channels": n_null_channels,
                "n_evaluable_null_channels": n_evaluable_null_channels,
                "n_total_calls": n_calls,
                "n_false_calls": n_false_calls,
                "null_channel_call_fraction": (
                    n_false_calls / n_evaluable_null_channels
                    if n_evaluable_null_channels
                    else np.nan
                ),
                "generated_null_channel_call_fraction": (
                    n_false_calls / n_null_channels
                ),
                "any_null_call": bool(n_false_calls > 0),
                "any_rejection": bool(n_calls > 0),
                "target_call": bool(target_row["call"]),
                "target_evaluable": bool(target_row["evaluable"]),
                "false_discovery_proportion": (
                    n_false_calls / n_calls if n_calls else 0.0
                ),
                "references_verified_null": True,
            }
        )
    return pd.DataFrame(rows)


def _stable_group_seed(base_seed: int, keys: Sequence[Any]) -> int:
    payload = json.dumps(
        [_json_value(value) for value in keys],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    group_code = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return int(
        np.random.SeedSequence([int(base_seed), group_code]).generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


def summarize_family_calibration(
    family_results: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Summarize family-level false calls, FDP, and target power.

    Binary probabilities use Wilson intervals across independent families.
    Means of within-family quantities use a percentile bootstrap that resamples
    complete simulated families, preserving within-family call dependence.
    """

    required = {
        "family_type",
        "target_call",
        "target_evaluable",
        "any_null_call",
        "null_channel_call_fraction",
        "n_false_calls",
        "false_discovery_proportion",
        *group_columns,
    }
    missing = sorted(required.difference(family_results.columns))
    if missing:
        raise ValueError(f"family results are missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for keys, group in family_results.groupby(
        list(group_columns),
        sort=True,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        family_types = group["family_type"].astype(str).drop_duplicates().tolist()
        if len(family_types) != 1:
            raise ValueError("each calibration group must have one family type")
        family_type = family_types[0]
        n_families = int(len(group))
        any_null_count = int(group["any_null_call"].astype(bool).sum())
        any_low, any_high = wilson_interval(
            any_null_count,
            n_families,
            confidence=confidence,
        )
        target_call_count = int(group["target_call"].astype(bool).sum())
        target_evaluable_count = int(
            group["target_evaluable"].astype(bool).sum()
        )
        target_power_low, target_power_high = wilson_interval(
            target_call_count,
            n_families,
            confidence=confidence,
        )
        target_eval_low, target_eval_high = wilson_interval(
            target_evaluable_count,
            n_families,
            confidence=confidence,
        )
        seed = _stable_group_seed(int(random_state), keys)
        null_rate, null_low, null_high = bootstrap_mean_interval(
            group["null_channel_call_fraction"],
            confidence=confidence,
            n_resamples=n_resamples,
            random_state=seed,
        )
        false_count, false_count_low, false_count_high = bootstrap_mean_interval(
            group["n_false_calls"],
            confidence=confidence,
            n_resamples=n_resamples,
            random_state=seed + 1,
        )
        mean_fdp, fdp_low, fdp_high = bootstrap_mean_interval(
            group["false_discovery_proportion"],
            confidence=confidence,
            n_resamples=n_resamples,
            random_state=seed + 2,
        )
        record.update(
            {
                "family_type": family_type,
                "n_families": n_families,
                "n_generated_null_contacts": int(
                    group["n_null_channels"].sum()
                ),
                "n_evaluable_null_hypotheses": int(
                    group["n_evaluable_null_channels"].sum()
                ),
                "n_null_hypotheses": int(
                    group["n_evaluable_null_channels"].sum()
                ),
                "total_false_calls": int(group["n_false_calls"].sum()),
                "total_false_call_rate": null_rate,
                "total_false_call_rate_ci_low": null_low,
                "total_false_call_rate_ci_high": null_high,
                "mean_false_calls_per_family": false_count,
                "mean_false_calls_per_family_ci_low": false_count_low,
                "mean_false_calls_per_family_ci_high": false_count_high,
                "probability_any_null_call": any_null_count / n_families,
                "probability_any_null_call_ci_low": any_low,
                "probability_any_null_call_ci_high": any_high,
                "mean_false_discovery_proportion": mean_fdp,
                "mean_false_discovery_proportion_ci_low": fdp_low,
                "mean_false_discovery_proportion_ci_high": fdp_high,
                "target_power_unconditional": (
                    target_call_count / n_families
                    if family_type == "mixed_signal"
                    else np.nan
                ),
                "target_power_unconditional_ci_low": (
                    target_power_low if family_type == "mixed_signal" else np.nan
                ),
                "target_power_unconditional_ci_high": (
                    target_power_high if family_type == "mixed_signal" else np.nan
                ),
                "target_evaluable_rate": (
                    target_evaluable_count / n_families
                    if family_type == "mixed_signal"
                    else np.nan
                ),
                "target_evaluable_rate_ci_low": (
                    target_eval_low if family_type == "mixed_signal" else np.nan
                ),
                "target_evaluable_rate_ci_high": (
                    target_eval_high if family_type == "mixed_signal" else np.nan
                ),
                "confidence": float(confidence),
                "binary_interval": "Wilson score across families",
                "mean_interval": (
                    f"percentile bootstrap of complete families ({int(n_resamples)} "
                    "resamples)"
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> tuple[str, bool | None]:
    """Return a commit only for a clean tree; source hashes cover dirty runs."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", None
    dirty = bool(status.stdout.strip())
    if dirty:
        return "not_recorded_uncommitted_worktree", True
    return revision.stdout.strip(), False


def provenance_record(root: Path, script_path: Path) -> dict[str, Any]:
    """Return reproducibility provenance without machine- or user-specific paths."""

    dependencies: dict[str, str] = {}
    for module in (np, pd):
        dependencies[module.__name__] = str(module.__version__)
    try:
        import scipy

        dependencies["scipy"] = str(scipy.__version__)
    except ImportError:
        dependencies["scipy"] = "unavailable"
    git_commit, git_worktree_dirty = _git_state(root)
    source_paths = (
        script_path,
        root / "ERPy" / "crp_energy.py",
        root / "ERPy" / "erp_detection.py",
        root / "ERPy" / "inference.py",
    )
    source_sha256 = {
        str(path.relative_to(root)): _sha256(path)
        for path in source_paths
        if path.is_file()
    }
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "erpy_compatibility_id": ERPY_COMPATIBILITY_ID,
        "script": script_path.name,
        "script_sha256": _sha256(script_path),
        "source_sha256": source_sha256,
        "git_commit": git_commit,
        "git_worktree_dirty": git_worktree_dirty,
        "erpy_version": str(ep.__version__),
        "crp_energy_detector_version": str(CRP_ENERGY_DETECTOR_VERSION),
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "dependencies": dependencies,
        "data_source": "fully synthetic; no protected or patient data",
        "stochastic_reproducibility": (
            "NumPy SeedSequence coordinates uniquely identify each factorial-cell "
            "replicate; simulation and detector-root streams are separated. "
            "Each CRP-energy channel stream is the first 63 bits of a SHA-256 "
            "digest of the detector root seed and exact channel label, making "
            "the stream independent of channel order. Exact root and effective "
            "seeds are retained in raw outputs."
        ),
    }


def _metric_record(frame: pd.DataFrame, **filters: Any) -> dict[str, Any]:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row for {filters}, found {len(selected)}")
    return {
        key: _json_value(value)
        for key, value in selected.iloc[0].to_dict().items()
    }


def _benchmark_summary(
    config: BenchmarkConfig,
    scenario_frame: pd.DataFrame,
    detector_frame: pd.DataFrame,
    overall: pd.DataFrame,
    stratified: pd.DataFrame,
    family_results: pd.DataFrame,
    family_overall: pd.DataFrame,
) -> dict[str, Any]:
    primary_null = _metric_record(
        overall,
        detector="crp_energy",
        truth=NULL_TRUTH,
    )
    primary_injected = _metric_record(
        overall,
        detector="crp_energy",
        truth="injected",
    )
    primary_all_null_family = _metric_record(
        family_overall,
        detector="crp_energy",
        family_type="all_null",
    )
    primary_mixed_family = _metric_record(
        family_overall,
        detector="crp_energy",
        family_type="mixed_signal",
    )
    artifact_free_power: dict[str, Any] = {}
    for snr in config.snr_values:
        if float(snr) <= 0:
            continue
        artifact_free_power[str(float(snr))] = _metric_record(
            stratified,
            detector="crp_energy",
            truth="injected",
            snr_nominal=float(snr),
            artifact_fraction=0.0,
        )
    p_identity = scenario_frame.loc[
        scenario_frame["primary_qc_status"].eq("pass"),
        "primary_p_joint_identity",
    ].astype(bool)
    primary_seed_rows = detector_frame[
        detector_frame["detector"].eq("crp_energy")
    ]
    distinct_channel_streams = primary_seed_rows.groupby(
        "scenario_id",
        sort=False,
    )["crp_energy_random_state_effective"].nunique()
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "erpy_compatibility_id": ERPY_COMPATIBILITY_ID,
        "design": {
            "n_scenarios": int(len(scenario_frame)),
            "n_null_scenarios": int(
                scenario_frame["truth"].eq(NULL_TRUTH).sum()
            ),
            "n_injected_scenarios": int(
                scenario_frame["truth"].eq("injected").sum()
            ),
            "n_replicates_per_cell": int(config.n_replicates),
            "trial_counts": list(config.trial_counts),
            "snr_values": list(config.snr_values),
            "artifact_fractions": list(config.artifact_fractions),
            "testing_family_channels": int(config.n_family_channels),
        },
        "primary_overall_null": primary_null,
        "primary_balanced_grid_injected": primary_injected,
        "primary_family_all_null_calibration": primary_all_null_family,
        "primary_family_mixed_calibration": primary_mixed_family,
        "primary_unmasked_recovery_by_snr": artifact_free_power,
        "primary_joint_identity_all_evaluable": bool(p_identity.all()),
        "primary_joint_identity_n_evaluable": int(len(p_identity)),
        "primary_random_stream_provenance_verified_all_channels": bool(
            len(primary_seed_rows)
            == len(scenario_frame) * int(config.n_family_channels)
            and distinct_channel_streams.eq(
                int(config.n_family_channels)
            ).all()
        ),
        "primary_random_stream_derivation": (
            "sha256-v1(root-seed, exact channel label; first 63 bits)"
        ),
        "null_references_verified_all_scenarios": bool(
            family_results["references_verified_null"].astype(bool).all()
        ),
        "multiplicity_adjustment": MULTIPLICITY_ADJUSTMENT,
        "interpretation_note": (
            "The injected aggregate is a balanced-grid conditional-recovery "
            "summary under the declared model. Cell and stratified tables retain "
            "the trial-count, SNR, and masking-burden dependence."
        ),
    }


def _report_markdown(
    config: BenchmarkConfig,
    summary: dict[str, Any],
) -> str:
    null = summary["primary_overall_null"]
    injected = summary["primary_balanced_grid_injected"]
    all_null_family = summary["primary_family_all_null_calibration"]
    mixed_family = summary["primary_family_mixed_calibration"]
    return f"""# ERPy CRP-energy synthetic benchmark

This directory is generated from fully synthetic data. It contains no patient,
clinical, or protected source data.

## Design

The deterministic factorial design contains {summary['design']['n_scenarios']}
independent scenarios: {len(config.trial_counts)} trial counts ×
{len(config.snr_values)} response-to-noise ratios (including the true null) ×
{len(config.artifact_fractions)} random target-trial masking levels ×
{config.n_replicates} replicates. Every scenario contains a fixed
{config.n_family_channels}-channel testing family. A biphasic response is
injected only into the designated target channel; the remaining channels are
null references used by operational detector preprocessing and multiplicity
correction.

Here, a *family* means the four simulated channels processed together for one
stimulation acquisition. It does not mean a participant family or a group of
datasets. The grouping mirrors the recording contacts considered for one
multiple-testing correction in a real analysis. Only detector-eligible
contacts with finite joint p-values enter that adjustment: the adjusted set
contains three contacts when random target-trial masking leaves too few usable
target trials and four contacts otherwise (600 and 1,800 of the 2,400
families, respectively). Because the generator supplies known signal/no-signal
truth, this synthetic benchmark tests detector behavior under declared
noise, trial count, response strength, and masking burden. It
complements, but is not pooled with, the public or governed real-data examples.

The primary rule is ERPy's CRP-energy intersection-union test with
`p_joint = max(p_crp, p_energy)` and Benjamini-Hochberg adjustment across the
detector-eligible contacts with finite joint p-values. Randomly masked target trials
are represented by nonfinite matched baseline and response windows. A
non-evaluable primary result is excluded from adjustment, counted as a negative
in unconditional recovery, and reported separately through evaluability
and conditional recovery.

## Primary topline

| Quantity | Estimate | {int(config.confidence * 100)}% interval | N |
| --- | ---: | ---: | ---: |
| Adjusted target calls in all-null cells | {null['call_rate_unconditional']:.3f} | {null['call_rate_unconditional_ci_low']:.3f}–{null['call_rate_unconditional_ci_high']:.3f} | {null['n_scenarios']} |
| Balanced-grid unconditional injected-target recovery | {injected['call_rate_unconditional']:.3f} | {injected['call_rate_unconditional_ci_low']:.3f}–{injected['call_rate_unconditional_ci_high']:.3f} | {injected['n_scenarios']} |
| Recovery conditional on evaluability | {injected['call_rate_given_evaluable']:.3f} | {injected['call_rate_given_evaluable_ci_low']:.3f}–{injected['call_rate_given_evaluable_ci_high']:.3f} | {injected['n_evaluable']} |
| Balanced-grid injected evaluability | {injected['evaluable_rate']:.3f} | {injected['evaluable_ci_low']:.3f}–{injected['evaluable_ci_high']:.3f} | {injected['n_scenarios']} |
| All-null family: any rejection | {all_null_family['probability_any_null_call']:.3f} | {all_null_family['probability_any_null_call_ci_low']:.3f}–{all_null_family['probability_any_null_call_ci_high']:.3f} | {all_null_family['n_families']} |
| Mixed family: any null-reference call | {mixed_family['probability_any_null_call']:.3f} | {mixed_family['probability_any_null_call_ci_low']:.3f}–{mixed_family['probability_any_null_call_ci_high']:.3f} | {mixed_family['n_families']} |
| Mixed family: mean false-discovery proportion | {mixed_family['mean_false_discovery_proportion']:.3f} | {mixed_family['mean_false_discovery_proportion_ci_low']:.3f}–{mixed_family['mean_false_discovery_proportion_ci_high']:.3f} | {mixed_family['n_families']} |

The balanced-grid recovery rates characterize the declared stationary-noise
model. Use `cell_metrics.csv` or `stratified_metrics.csv` for signal-strength-,
trial-, and masking-specific estimates.

## Files

- `scenario_results.csv.gz`: one target-level record per independent scenario,
  including truth, realized SNR, detector statistics, calls, and evaluability.
- `detector_results.csv.gz`: normalized raw decisions for every detector and
  every channel in each testing family, including the exact CRP-energy root
  and order-invariant per-channel random-stream seeds.
- `family_results.csv.gz`: one family-calibration record per scenario and
  detector, retaining false calls, any-null-call status, target call, and FDP.
- `cell_metrics.csv`: declared factorial-cell rates and Wilson intervals.
- `stratified_metrics.csv`: rates aggregated over trial counts for each SNR and
  random masking level.
- `primary_figure_table.csv`: compact primary-detector values ready for a
  trial/SNR/masking operating-characteristic figure.
- `overall_metrics.csv`: balanced-design aggregate rates.
- `family_cell_metrics.csv`, `family_stratified_metrics.csv`, and
  `family_calibration.csv`: family-level calibration with Wilson intervals for
  binary probabilities and family-cluster bootstrap intervals for mean false
  call rates and FDP.
- `family_figure_table.csv`: compact primary family-calibration values.
- `benchmark_config.json`: complete resolved simulation and detector settings.
- `provenance.json`: compatibility identifier, source-code hashes, versions,
  seed construction, and data origin.
- `benchmark_summary.json`: machine-readable topline results.
- `SHA256SUMS.txt`: integrity hashes for all preceding files.

The companion `composite_null_stress_test/` directory evaluates the two
one-component-null branches with exact tests in 1,000 independently seeded
families per branch.

The compatibility identifier is a version-level software checkpoint shared by
all equivalent ERPy installations. It is not a release-record identifier and
does not label or upload a dataset.

## Comparator scope and limitations

The comparator labels refer to ERPy v1.0.0's operational implementations under
the same synthetic acquisition. They do not constitute exact reproductions of
the cited publications, whose preprocessing, recording characteristics, and
cohorts differ. The benchmark uses one stationary Gaussian AR(1) noise model,
one biphasic morphology, modest positive amplitude variation and latency jitter,
and QC-style trial removal rather than every possible residual artifact. It
does not establish clinical validity, anatomical generalizability, calibrated
effect-size recovery, or equivalence to a prospectively labeled dataset.

CRP-energy, CRP-only, and paired-RMS p-value fields receive Benjamini-Hochberg
adjustment across each simulated channel family. The standalone CRP field uses
a data-selected duration and a t-test over shared-trial projections; its marked
anti-calibration under the all-null simulation makes it exploratory. Paired RMS
uses separately demeaned matched segments in this final release. Kundu-Rolston
and N1-z retain unadjusted operational thresholds. Interpretation of adjusted
calls additionally requires valid input p-values and independent or
positive-regression-dependent null hypotheses; the simulation uses independent
contacts. Family-level and eligible-hypothesis denominators are retained
separately.
"""


def execute_benchmark(
    config: BenchmarkConfig,
    *,
    output_dir: Path,
    jobs: int = 1,
) -> dict[str, Any]:
    """Run the complete design and write auditable result artifacts."""

    config.validate()
    jobs = max(int(jobs), 1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = scenario_specs(config)
    started = time.perf_counter()
    payloads = [(config, spec) for spec in specs]
    if jobs == 1:
        evaluated = [_run_scenario_payload(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            evaluated = list(
                executor.map(_run_scenario_payload, payloads, chunksize=1)
            )
    elapsed_seconds = time.perf_counter() - started

    scenario_rows = [item[0] for item in evaluated]
    detector_rows = [
        record
        for _, records in evaluated
        for record in records
    ]
    scenario_frame = pd.DataFrame(scenario_rows).sort_values(
        ["n_trials_total", "artifact_fraction", "snr_nominal", "replicate"],
        kind="stable",
    )
    detector_frame = pd.DataFrame(detector_rows).sort_values(
        [
            "n_trials_total",
            "artifact_fraction",
            "snr_nominal",
            "replicate",
            "channel",
            "detector",
        ],
        kind="stable",
    )
    family_frame = family_scenario_results(detector_frame).sort_values(
        [
            "n_trials_total",
            "artifact_fraction",
            "snr_nominal",
            "replicate",
            "detector",
        ],
        kind="stable",
    )
    target = detector_frame[detector_frame["is_target"]].copy()
    cell_metrics = summarize_metrics(
        target,
        group_columns=[
            "detector",
            "multiplicity_adjustment",
            "truth",
            "n_trials_total",
            "artifact_fraction",
            "n_clean_trials_expected",
            "snr_nominal",
        ],
        confidence=float(config.confidence),
    )
    stratified_metrics = summarize_metrics(
        target,
        group_columns=[
            "detector",
            "multiplicity_adjustment",
            "truth",
            "artifact_fraction",
            "snr_nominal",
        ],
        confidence=float(config.confidence),
    )
    overall_metrics = summarize_metrics(
        target,
        group_columns=["detector", "multiplicity_adjustment", "truth"],
        confidence=float(config.confidence),
    )
    family_cell_metrics = summarize_family_calibration(
        family_frame,
        group_columns=[
            "detector",
            "multiplicity_adjustment",
            "family_type",
            "target_truth",
            "n_trials_total",
            "artifact_fraction",
            "n_clean_trials_expected",
            "snr_nominal",
        ],
        confidence=float(config.confidence),
        n_resamples=int(config.bootstrap_resamples),
        random_state=int(config.master_seed),
    )
    family_stratified_metrics = summarize_family_calibration(
        family_frame,
        group_columns=[
            "detector",
            "multiplicity_adjustment",
            "family_type",
            "target_truth",
            "artifact_fraction",
            "snr_nominal",
        ],
        confidence=float(config.confidence),
        n_resamples=int(config.bootstrap_resamples),
        random_state=int(config.master_seed),
    )
    family_overall_metrics = summarize_family_calibration(
        family_frame,
        group_columns=[
            "detector",
            "multiplicity_adjustment",
            "family_type",
            "target_truth",
        ],
        confidence=float(config.confidence),
        n_resamples=int(config.bootstrap_resamples),
        random_state=int(config.master_seed),
    )
    primary_figure_table = stratified_metrics[
        stratified_metrics["detector"].eq("crp_energy")
    ][
        [
            "truth",
            "multiplicity_adjustment",
            "artifact_fraction",
            "snr_nominal",
            "metric",
            "n_scenarios",
            "n_evaluable",
            "evaluable_rate",
            "evaluable_ci_low",
            "evaluable_ci_high",
            "call_rate_unconditional",
            "call_rate_unconditional_ci_low",
            "call_rate_unconditional_ci_high",
            "call_rate_given_evaluable",
            "call_rate_given_evaluable_ci_low",
            "call_rate_given_evaluable_ci_high",
            "specificity",
            "specificity_ci_low",
            "specificity_ci_high",
        ]
    ].sort_values(["artifact_fraction", "snr_nominal"], kind="stable")
    family_figure_table = family_stratified_metrics[
        family_stratified_metrics["detector"].eq("crp_energy")
    ].sort_values(
        ["family_type", "artifact_fraction", "snr_nominal"],
        kind="stable",
    )

    paths = {
        "scenario_results": output_dir / "scenario_results.csv.gz",
        "detector_results": output_dir / "detector_results.csv.gz",
        "family_results": output_dir / "family_results.csv.gz",
        "cell_metrics": output_dir / "cell_metrics.csv",
        "stratified_metrics": output_dir / "stratified_metrics.csv",
        "primary_figure_table": output_dir / "primary_figure_table.csv",
        "overall_metrics": output_dir / "overall_metrics.csv",
        "family_cell_metrics": output_dir / "family_cell_metrics.csv",
        "family_stratified_metrics": output_dir
        / "family_stratified_metrics.csv",
        "family_calibration": output_dir / "family_calibration.csv",
        "family_figure_table": output_dir / "family_figure_table.csv",
        "config": output_dir / "benchmark_config.json",
        "provenance": output_dir / "provenance.json",
        "summary": output_dir / "benchmark_summary.json",
        "report": output_dir / "README.md",
    }
    _write_deterministic_gzip_csv(scenario_frame, paths["scenario_results"])
    _write_deterministic_gzip_csv(detector_frame, paths["detector_results"])
    _write_deterministic_gzip_csv(family_frame, paths["family_results"])
    cell_metrics.to_csv(paths["cell_metrics"], index=False, lineterminator="\n")
    stratified_metrics.to_csv(
        paths["stratified_metrics"], index=False, lineterminator="\n"
    )
    primary_figure_table.to_csv(
        paths["primary_figure_table"], index=False, lineterminator="\n"
    )
    overall_metrics.to_csv(
        paths["overall_metrics"], index=False, lineterminator="\n"
    )
    family_cell_metrics.to_csv(
        paths["family_cell_metrics"], index=False, lineterminator="\n"
    )
    family_stratified_metrics.to_csv(
        paths["family_stratified_metrics"], index=False, lineterminator="\n"
    )
    family_overall_metrics.to_csv(
        paths["family_calibration"], index=False, lineterminator="\n"
    )
    family_figure_table.to_csv(
        paths["family_figure_table"], index=False, lineterminator="\n"
    )
    _write_json(paths["config"], config.to_json_record())
    root = Path(__file__).resolve().parents[1]
    _write_json(
        paths["provenance"],
        provenance_record(root, Path(__file__).resolve()),
    )
    summary = _benchmark_summary(
        config,
        scenario_frame,
        detector_frame,
        overall_metrics,
        stratified_metrics,
        family_frame,
        family_overall_metrics,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(
        _report_markdown(config, summary),
        encoding="utf-8",
    )
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in paths.values()
        ),
        encoding="utf-8",
    )
    summary["elapsed_seconds"] = float(elapsed_seconds)
    summary["output_dir"] = str(output_dir)
    summary["checksums"] = str(checksum_path)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release_artifacts/benchmark"),
        help="Output directory (default: release_artifacts/benchmark)",
    )
    parser.add_argument("--master-seed", type=int, default=20_260_830)
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--trial-counts", default="8,12,24,36")
    parser.add_argument("--snr-values", default="0,0.5,1,2,4")
    parser.add_argument("--artifact-fractions", default="0,0.25,0.5")
    parser.add_argument("--family-channels", type=int, default=4)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Independent worker processes (default: 1; process parallelism can "
            "be slower with multithreaded numerical libraries)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small deterministic smoke design instead of the full grid",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    if args.quick:
        return BenchmarkConfig(
            master_seed=int(args.master_seed),
            n_replicates=2,
            trial_counts=(8, 24),
            snr_values=(0.0, 2.0),
            artifact_fractions=(0.0, 0.5),
            n_family_channels=int(args.family_channels),
        )
    return BenchmarkConfig(
        master_seed=int(args.master_seed),
        n_replicates=int(args.replicates),
        trial_counts=parse_number_list(args.trial_counts, cast=int),
        snr_values=parse_number_list(args.snr_values, cast=float),
        artifact_fractions=parse_number_list(
            args.artifact_fractions,
            cast=float,
        ),
        n_family_channels=int(args.family_channels),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    summary = execute_benchmark(
        config,
        output_dir=args.output_dir,
        jobs=int(args.jobs),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
