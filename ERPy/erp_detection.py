from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import signal, stats

from .baseline import baseline_center_array, baseline_time_mask
from .crp import CRPConfig, run_crp_array
from .crp_energy import (
    CRPEnergyConfig,
    _empty_result,
    _parameter_json,
    run_crp_energy_array,
)
from .inference import fdr_bh


ALL_METHODS = [
    "kundu_rolston",
    "keller_zscore",
    "crp_significance",
    "crp_energy",
    "crowther_gamma",
    "peak_amplitude",
    "rms_response",
]

LEGACY_CONSENSUS_METHODS = tuple(
    method for method in ALL_METHODS if method != "crp_energy"
)

PRIMARY_CRITERION = "crp_energy_bh_conjunction"

DETECTOR_REFERENCES = {
    "kundu_rolston": "Kundu et al., Brain Stimulation 2020; doi:10.1016/j.brs.2020.06.002",
    "keller_zscore": "Keller et al., Journal of Neuroscience 2014; doi:10.1523/JNEUROSCI.4289-13.2014",
    "crp_significance": "Miller et al., PLOS Computational Biology 2023; doi:10.1371/journal.pcbi.1011105",
    "crp_energy": "CRP-derived fixed-window whole-trial reproducibility and separately demeaned paired log-RMS energy joined by an intersection-union test; data-selected CRP duration remains descriptive",
    "crowther_gamma": "Crowther et al., Journal of Neuroscience Methods 2019; doi:10.1016/j.jneumeth.2018.09.034",
    "peak_amplitude": "Hays et al., Brain Stimulation 2023; doi:10.1016/j.brs.2023.04.023",
    "rms_response": "Hays et al., Brain Stimulation 2023; doi:10.1016/j.brs.2023.04.023",
}

DETECTOR_DEFINITIONS = {
    "kundu_rolston": "10-Hz high-pass, squared and 10-Hz smoothed envelope; 15-ms persistence above 3x baseline and post median >30 uV",
    "keller_zscore": "maximum absolute mean-waveform z score in 10-50 and 50-250 ms; z>=6 with post-artifact polarity reversal",
    "crp_significance": "exploratory data-selected CRP duration and one-sided t-test over shared-trial projections; waveform quantities are descriptive",
    "crp_energy": "intersection-union conjunction of fixed-window whole-trial sign-flip reproducibility and separately demeaned matched-window excess energy; BH adjustment is applied to max(p_crp, p_energy)",
    "crowther_gamma": "SIGNI 70-170-Hz envelope SNR with log-null randomization and channel-wise Bonferroni correction; requires an explicitly declared passband-preserving input",
    "peak_amplitude": "N1 peak amplitude and baseline-normalized z score in 10-50 ms; z>=6",
    "rms_response": "10-100-ms RMS magnitude; response presence follows the source protocol's 10-50-ms N1 z>=6 rule",
}

DETECTOR_QUANTITY_COLUMNS = {
    "kundu_rolston": (
        "kundu_baseline_median_uv",
        "kundu_envelope_threshold_uv",
        "kundu_post_median_uv",
        "kundu_peak_envelope_ratio",
        "kundu_longest_suprathreshold_ms",
        "kundu_persistence_pass",
        "kundu_magnitude_pass",
        "kundu_mean_absolute_amplitude_uv",
        "kundu_n_valid_trials",
    ),
    "keller_zscore": (
        "keller_a1_peak_uv",
        "keller_a1_signed_peak_uv",
        "keller_a1_latency_ms",
        "keller_a1_z",
        "keller_a2_peak_uv",
        "keller_a2_signed_peak_uv",
        "keller_a2_latency_ms",
        "keller_a2_z",
        "keller_polarity_reversal",
        "keller_saturation_like",
        "keller_max_z",
    ),
    "crp_significance": (
        "crp_t_value",
        "crp_response_duration_ms",
        "crp_alpha_prime_mean_uv",
        "crp_alpha_prime_median_uv",
        "crp_alpha_prime_abs_median_uv",
        "crp_residual_rms_median_uv",
        "crp_snr_median",
        "crp_explained_variance_median",
        "crp_full_window_t_value",
        "crp_full_window_p_value",
        "crp_q_value",
        "crp_fdr_significant",
        "crp_inference_status",
    ),
    "crp_energy": (
        "p_crp",
        "p_energy",
        "p_joint",
        "q_joint",
        "classification",
        "crp_statistic",
        "energy_statistic",
        "rms_response",
        "rms_baseline",
        "rms_ratio_db",
        "canonical_energy",
        "canonical_energy_fraction",
        "response_duration",
        "n_trials_total",
        "n_trials_clean",
        "crp_explained_variance",
        "crp_snr",
        "reproducibility_test_exact",
        "reproducibility_n_randomizations",
        "energy_test_exact",
        "energy_n_permutations",
        "random_state_root",
        "random_state_effective",
        "random_stream_derivation",
        "qc_status",
        "detector_version",
    ),
    "crowther_gamma": (
        "signi_snr",
        "signi_p_value_raw",
        "signi_empirical_p_value",
        "signi_null_log_mean",
        "signi_null_log_std",
        "signi_null_ks_p_value",
        "signi_latency_ms",
        "signi_n_permutations",
        "signi_p_value_bonferroni",
        "high_gamma_z",
        "signi_n_valid_trials",
    ),
    "peak_amplitude": (
        "hays_n1_peak_uv",
        "hays_n1_signed_peak_uv",
        "hays_n1_latency_ms",
        "hays_n1_z",
        "hays_n1_threshold_z",
    ),
    "rms_response": (
        "rms_response_trial_median_uv",
        "rms_baseline_trial_median_uv",
        "rms_delta_uv",
        "rms_ratio",
        "rms_robust_z",
        "rms_wilcoxon_statistic",
        "rms_wilcoxon_p_value",
        "rms_paired_q_value",
        "rms_paired_fdr_significant",
        "rms_waveform_10_100_uv",
        "rms_n_paired_trials",
        "rms_detection_n1_z",
        "rms_detection_n1_peak_uv",
        "rms_detection_n1_latency_ms",
        "rms_significance_basis",
    ),
}


@dataclass
class ERPDetection:
    """Backward-compatible representation of one detector result."""

    channel: str
    method: str
    significant: bool
    score: float
    p_value: float | None = None
    threshold: float | None = None
    notes: str = ""


def _trapezoid(values: np.ndarray, times: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(values, times))


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else np.nan


def _safe_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 1e-12
    return float(finite.std(ddof=1) + 1e-12)


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return np.nan
    median = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - median)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite, ddof=1))
    return scale if np.isfinite(scale) and scale > 1e-12 else np.nan


def _row_rms(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[0], np.nan)
    for index, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            out[index] = np.sqrt(np.mean(finite * finite))
    return out


def _feature_table(
    epochs,
    baseline_window=(-0.5, -0.03),
    response_window=(0.01, 0.3),
    response_boundary_guard_s: float = 0.002,
) -> pd.DataFrame:
    arr, times, channels = epochs.as_array()
    return _feature_table_from_array(
        arr,
        times,
        channels,
        sfreq=float(epochs.sfreq),
        baseline_window=baseline_window,
        response_window=response_window,
        response_boundary_guard_s=response_boundary_guard_s,
    )


def _feature_table_from_array(
    arr: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    *,
    sfreq: float,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
    response_boundary_guard_s: float = 0.002,
    baseline_centered: bool = False,
) -> pd.DataFrame:
    """Compute common voltage-domain quantities once for every detector."""

    baseline_mask = baseline_time_mask(times, baseline_window)
    response_mask = (times >= response_window[0]) & (times <= response_window[1])
    if not response_mask.any():
        raise ValueError("response_window does not overlap epoch times")
    referenced = (
        np.asarray(arr, dtype=float)
        if baseline_centered
        else baseline_center_array(arr, times, baseline_window)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mean = np.nanmean(referenced, axis=0)
    rows = []
    for channel_index, channel in enumerate(channels):
        base_trials = referenced[:, baseline_mask, channel_index].reshape(-1)
        base_std = _safe_std(base_trials)
        base_mean = _safe_mean(base_trials)
        wave = mean[:, channel_index]
        response = wave[response_mask]
        response_times = times[response_mask]
        absolute_response = np.abs(response)
        if np.isfinite(absolute_response).any():
            peak_index = int(np.nanargmax(absolute_response))
            peak_amplitude = float(absolute_response[peak_index])
            peak_latency = float(response_times[peak_index] * 1000.0)
            boundary_distances = np.asarray(
                [
                    abs(float(response_times[peak_index] - response_times[0])),
                    abs(float(response_times[-1] - response_times[peak_index])),
                ]
            )
            boundary_index = int(np.argmin(boundary_distances))
            boundary_distance_s = float(boundary_distances[boundary_index])
            peak_at_boundary = bool(
                boundary_distance_s
                <= max(float(response_boundary_guard_s), 1.0 / float(sfreq))
            )
            peak_boundary_side = (
                "start" if boundary_index == 0 else "end"
            ) if peak_at_boundary else ""
        else:
            peak_amplitude = peak_latency = boundary_distance_s = np.nan
            peak_at_boundary = False
            peak_boundary_side = ""

        finite_response = np.isfinite(response) & np.isfinite(response_times)
        if finite_response.any():
            response_start = float(response[np.flatnonzero(finite_response)[0]])
            response_end = float(response[np.flatnonzero(finite_response)[-1]])
            response_drift = response_end - response_start
        else:
            response_start = response_end = response_drift = np.nan
        response_slope = (
            float(np.polyfit(response_times[finite_response], response[finite_response], deg=1)[0])
            if finite_response.sum() >= 2
            else np.nan
        )
        n1_mask = (times >= 0.01) & (times <= 0.1)
        n1_wave = wave[n1_mask]
        n1_times = times[n1_mask]
        if n1_wave.size and np.isfinite(n1_wave).any():
            n1_index = int(np.nanargmax(np.abs(n1_wave)))
            n1_latency = float(n1_times[n1_index] * 1000.0)
        else:
            n1_latency = np.nan
        rms = float(np.sqrt(_safe_mean(response * response))) if np.isfinite(response).any() else np.nan
        auc = _trapezoid(absolute_response, response_times) * 1000.0 if np.isfinite(absolute_response).any() else np.nan
        derivative = np.gradient(wave, 1.0 / sfreq)
        response_derivative = derivative[response_mask]
        max_descent = (
            float(response_times[int(np.nanargmin(response_derivative))] * 1000.0)
            if np.isfinite(response_derivative).any()
            else np.nan
        )
        baseline_z = float((peak_amplitude - abs(base_mean)) / base_std)
        response_trial_rms = _row_rms(referenced[:, response_mask, channel_index])
        baseline_trial_rms = _row_rms(referenced[:, baseline_mask, channel_index])
        rms_z = float(
            (_safe_mean(response_trial_rms) - _safe_mean(baseline_trial_rms))
            / _safe_std(baseline_trial_rms)
        )
        rows.append(
            {
                "channel": str(channel),
                "peak_amplitude_uv": peak_amplitude,
                "peak_latency_ms": peak_latency,
                "peak_at_response_boundary": peak_at_boundary,
                "peak_boundary_side": peak_boundary_side,
                "peak_boundary_distance_ms": boundary_distance_s * 1000.0,
                "response_start_uv": response_start,
                "response_end_uv": response_end,
                "response_drift_uv": response_drift,
                "response_linear_slope_uv_s": response_slope,
                "n1_latency_ms": n1_latency,
                "rms_uv": rms,
                "auc_uv_ms": auc,
                "max_descent_ms": max_descent,
                "baseline_zscore": baseline_z,
                "rms_zscore": rms_z,
                "baseline_std": base_std,
                "n_trials": int(arr.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _continuous_threshold(
    wave: np.ndarray,
    times: np.ndarray,
    threshold: float,
    min_duration_s: float,
    sfreq: float,
) -> bool:
    del times
    above = np.abs(np.asarray(wave, dtype=float)) >= float(threshold)
    required = max(int(np.ceil(float(min_duration_s) * float(sfreq))), 1)
    return _longest_true_run(above) >= required


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _fill_nonfinite_for_filter(values: np.ndarray) -> np.ndarray:
    """Interpolate nonfinite samples along time without changing validity flags."""

    filled = np.asarray(values, dtype=float).copy()
    sample_index = np.arange(filled.shape[1], dtype=float)
    for trial in range(filled.shape[0]):
        for channel in range(filled.shape[2]):
            trace = filled[trial, :, channel]
            finite = np.isfinite(trace)
            if finite.all():
                continue
            if finite.sum() >= 2:
                trace[~finite] = np.interp(sample_index[~finite], sample_index[finite], trace[finite])
            elif finite.any():
                trace[~finite] = trace[finite][0]
            else:
                trace[:] = 0.0
    return filled


def _aligned_epoch_arrays(epochs, channels: list[str]) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    arr, times, source_channels = epochs.as_array()
    source_lookup = {str(channel): index for index, channel in enumerate(source_channels)}
    aligned = np.full((arr.shape[0], arr.shape[1], len(channels)), np.nan, dtype=float)
    missing = []
    for target_index, channel in enumerate(channels):
        if channel in source_lookup:
            aligned[:, :, target_index] = arr[:, :, source_lookup[channel]]
        else:
            missing.append(channel)
    return aligned, np.asarray(times, dtype=float), float(epochs.sfreq), missing


def _base_method_frame(features: pd.DataFrame, method: str, detector_input: str) -> pd.DataFrame:
    frame = features.copy()
    frame["method"] = str(method)
    frame["method_available"] = True
    frame["availability_reason"] = ""
    frame["significant"] = False
    frame["score"] = np.nan
    frame["p_value"] = np.nan
    frame["threshold"] = np.nan
    frame["notes"] = ""
    frame["detector_input"] = detector_input
    frame["detector_definition"] = DETECTOR_DEFINITIONS[method]
    frame["detector_reference"] = DETECTOR_REFERENCES[method]
    return frame


def _unavailable_signi_frame(
    features: pd.DataFrame,
    *,
    detector_input: str,
    reason: str,
) -> pd.DataFrame:
    """Return a schema-stable unavailable SIGNI result."""

    frame = _base_method_frame(features, "crowther_gamma", detector_input)
    frame["method_available"] = False
    frame["availability_reason"] = str(reason)
    frame["notes"] = str(reason)
    for column in DETECTOR_QUANTITY_COLUMNS["crowther_gamma"]:
        frame[column] = np.nan
    return frame


def _peak_in_window(wave: np.ndarray, times: np.ndarray, window: tuple[float, float], baseline_mean: float, baseline_std: float) -> dict[str, float]:
    mask = (times >= window[0]) & (times <= window[1])
    values = wave[mask]
    window_times = times[mask]
    if not values.size or not np.isfinite(values).any():
        return {"amplitude_uv": np.nan, "signed_amplitude_uv": np.nan, "latency_ms": np.nan, "z": np.nan}
    index = int(np.nanargmax(np.abs(values - baseline_mean)))
    signed = float(values[index] - baseline_mean)
    return {
        "amplitude_uv": abs(signed),
        "signed_amplitude_uv": signed,
        "latency_ms": float(window_times[index] * 1000.0),
        "z": abs(signed) / (float(baseline_std) + 1e-12),
    }


def _keller_quantities(
    referenced: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    *,
    baseline_window: tuple[float, float],
    z_threshold: float = 6.0,
    polarity_reversal_min_z: float = 1.0,
) -> pd.DataFrame:
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    mean_waveforms = np.nanmean(referenced, axis=0)
    rows = []
    for channel_index, channel in enumerate(channels):
        wave = mean_waveforms[:, channel_index]
        baseline = wave[baseline_mask]
        baseline_mean = _safe_mean(baseline)
        baseline_std = _safe_std(baseline)
        a1 = _peak_in_window(wave, times, (0.010, 0.050), baseline_mean, baseline_std)
        a2 = _peak_in_window(wave, times, (0.050, 0.250), baseline_mean, baseline_std)
        overall = _peak_in_window(wave, times, (0.010, 0.250), baseline_mean, baseline_std)
        response_mask = (times >= 0.010) & (times <= 0.250)
        response_z = (wave[response_mask] - baseline_mean) / (baseline_std + 1e-12)
        polarity_reversal = bool(
            np.isfinite(response_z).any()
            and np.nanmin(response_z) <= -abs(float(polarity_reversal_min_z))
            and np.nanmax(response_z) >= abs(float(polarity_reversal_min_z))
        )
        max_z = float(np.nanmax([a1["z"], a2["z"]]))
        rows.append(
            {
                "channel": channel,
                "keller_a1_peak_uv": a1["amplitude_uv"],
                "keller_a1_signed_peak_uv": a1["signed_amplitude_uv"],
                "keller_a1_latency_ms": a1["latency_ms"],
                "keller_a1_z": a1["z"],
                "keller_a2_peak_uv": a2["amplitude_uv"],
                "keller_a2_signed_peak_uv": a2["signed_amplitude_uv"],
                "keller_a2_latency_ms": a2["latency_ms"],
                "keller_a2_z": a2["z"],
                "peak_10_250_uv": overall["amplitude_uv"],
                "peak_10_250_signed_uv": overall["signed_amplitude_uv"],
                "peak_10_250_latency_ms": overall["latency_ms"],
                "peak_10_250_z": overall["z"],
                "keller_polarity_reversal": polarity_reversal,
                "keller_saturation_like": bool(max_z >= z_threshold and not polarity_reversal),
                "keller_max_z": max_z,
            }
        )
    return pd.DataFrame(rows)


def _detect_keller(
    features: pd.DataFrame,
    referenced: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    baseline_window: tuple[float, float],
    params: dict[str, Any],
) -> pd.DataFrame:
    z_threshold = float(params.get("z_threshold", 6.0))
    native = _keller_quantities(
        referenced,
        times,
        channels,
        baseline_window=baseline_window,
        z_threshold=z_threshold,
        polarity_reversal_min_z=float(params.get("polarity_reversal_min_z", 1.0)),
    )
    frame = _base_method_frame(features, "keller_zscore", "primary_epochs").merge(native, on="channel", how="left")
    frame["score"] = frame["keller_max_z"]
    frame["threshold"] = z_threshold
    frame["significant"] = (
        frame["keller_max_z"].ge(z_threshold)
        & frame["keller_polarity_reversal"].fillna(False).astype(bool)
    )
    frame.loc[frame["keller_saturation_like"].fillna(False).astype(bool), "notes"] = "large unidirectional deflection; Keller polarity-reversal screen failed"
    return frame


def _detect_peak(
    features: pd.DataFrame,
    referenced: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    baseline_window: tuple[float, float],
    params: dict[str, Any],
) -> pd.DataFrame:
    z_threshold = float(params.get("z_threshold", 6.0))
    native = _keller_quantities(
        referenced,
        times,
        channels,
        baseline_window=baseline_window,
        z_threshold=z_threshold,
        polarity_reversal_min_z=float(params.get("polarity_reversal_min_z", 1.0)),
    )
    frame = _base_method_frame(features, "peak_amplitude", "primary_epochs").merge(native, on="channel", how="left")
    frame["hays_n1_peak_uv"] = frame["keller_a1_peak_uv"]
    frame["hays_n1_signed_peak_uv"] = frame["keller_a1_signed_peak_uv"]
    frame["hays_n1_latency_ms"] = frame["keller_a1_latency_ms"]
    frame["hays_n1_z"] = frame["keller_a1_z"]
    frame["hays_n1_threshold_z"] = z_threshold
    frame["score"] = frame["hays_n1_z"]
    frame["threshold"] = z_threshold
    frame["significant"] = frame["hays_n1_z"].ge(z_threshold)
    return frame


def _detect_crp(
    features: pd.DataFrame,
    referenced: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
    params: dict[str, Any],
) -> pd.DataFrame:
    alpha = float(params.get("alpha", 0.05))
    configured_response = tuple(params.get("response_window", response_window))
    crp_response = (max(float(configured_response[0]), float(params.get("minimum_response_start_s", 0.015))), float(configured_response[1]))
    config = CRPConfig(
        response_window=crp_response,
        baseline_window=tuple(params.get("baseline_window", baseline_window)),
        min_trials=int(params.get("min_trials", 3)),
        alpha=alpha,
        projection_start_s=float(params.get("projection_start_s", 0.010)),
        projection_step_s=float(params.get("projection_step_s", 0.005)),
        permutation_n=int(params.get("permutation_n", 0)),
        random_state=params.get("random_state", 0),
    )
    rows = []
    for channel_index, channel in enumerate(channels):
        try:
            result = run_crp_array(
                referenced[:, :, channel_index],
                times,
                channel=channel,
                config=config,
            )
            alpha_prime = np.asarray(result.alpha_prime_uv, dtype=float)
            residual_rms = np.asarray(result.residual_rms_uv, dtype=float)
            snr = np.asarray(result.snr, dtype=float)
            explained = np.asarray(result.explained_variance, dtype=float)
            rows.append(
                {
                    "channel": channel,
                    "significant": result.significant,
                    "score": result.score,
                    "p_value": result.p_value,
                    "threshold": result.threshold,
                    "notes": "",
                    "crp_t_value": result.t_value,
                    "crp_response_duration_ms": result.response_duration_s * 1000.0,
                    "crp_alpha_prime_mean_uv": float(np.nanmean(alpha_prime)),
                    "crp_alpha_prime_median_uv": float(np.nanmedian(alpha_prime)),
                    "crp_alpha_prime_abs_median_uv": float(np.nanmedian(np.abs(alpha_prime))),
                    "crp_residual_rms_median_uv": float(np.nanmedian(residual_rms)),
                    "crp_snr_median": float(np.nanmedian(snr)),
                    "crp_explained_variance_median": float(np.nanmedian(explained)),
                    "crp_full_window_t_value": result.full_window_t_value,
                    "crp_full_window_p_value": result.full_window_p_value,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "channel": channel,
                    "significant": False,
                    "score": np.nan,
                    "p_value": np.nan,
                    "threshold": alpha,
                    "notes": str(exc),
                }
            )
    native = pd.DataFrame(rows)
    q_values, rejected = fdr_bh(native["p_value"], alpha=alpha)
    native["crp_q_value"] = q_values
    native["crp_fdr_significant"] = rejected
    native["crp_inference_status"] = (
        "exploratory_uncalibrated_selected_duration_shared_trial_t_test"
    )
    frame = _base_method_frame(features, "crp_significance", "primary_epochs")
    frame = frame.drop(columns=["significant", "score", "p_value", "threshold", "notes"]).merge(native, on="channel", how="left")
    return frame


def _stable_channel_random_state(
    random_state: int | None,
    channel: str,
) -> int | None:
    """Derive an order-invariant per-channel seed from a declared root seed."""

    if random_state is None:
        return None
    payload = (
        "erpy-crp-energy-channel-seed-v1\0"
        f"{int(random_state)}\0{str(channel)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & (
        (1 << 63) - 1
    )


def _seed_decimal(value: int | None) -> str | None:
    """Serialize an optional seed without floating-point coercion risk."""

    return None if value is None else str(int(value))


def _detect_crp_energy(
    features: pd.DataFrame,
    arr: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
    params: dict[str, Any],
    excluded_channels: Iterable[str] = (),
) -> pd.DataFrame:
    """Run both conjunction components on one identical clean-trial set."""

    alpha = float(params.get("alpha", 0.05))
    correction = str(params.get("correction", "fdr_bh")).strip().lower()
    if correction not in {"fdr_bh", "none"}:
        raise ValueError("crp_energy correction must be 'fdr_bh' or 'none'")
    configured_response = tuple(params.get("response_window", response_window))
    configured_baseline = tuple(params.get("baseline_window", baseline_window))
    config = CRPEnergyConfig(
        response_window=(
            float(configured_response[0]),
            float(configured_response[1]),
        ),
        baseline_window=(
            float(configured_baseline[0]),
            float(configured_baseline[1]),
        ),
        alpha=alpha,
        correction=correction,
        min_clean_trials=int(params.get("min_clean_trials", 8)),
        n_permutations=int(params.get("n_permutations", 5_000)),
        max_exact_trials=int(params.get("max_exact_trials", 16)),
        max_exact_reproducibility_trials=int(
            params.get("max_exact_reproducibility_trials", 12)
        ),
        canonical_energy_cv=bool(params.get("canonical_energy_cv", True)),
        random_state=params.get("random_state", 42),
        artifact_interval=tuple(
            params.get("artifact_interval", (0.0, 0.015))
        ),
        projection_start_s=float(params.get("projection_start_s", 0.010)),
        projection_step_s=float(params.get("projection_step_s", 0.005)),
        eps=float(params.get("eps", 1e-12)),
    )
    config.validate()
    include_arrays = bool(params.get("return_arrays", True))
    rows = []
    for channel_index, channel in enumerate(channels):
        effective_random_state = _stable_channel_random_state(
            config.random_state,
            channel,
        )
        channel_config = replace(
            config,
            random_state=effective_random_state,
        )
        try:
            result = run_crp_energy_array(
                arr[:, :, channel_index],
                times,
                channel=channel,
                config=channel_config,
            )
            row = result.to_record(include_arrays=include_arrays)
        except Exception as exc:
            response_start = max(
                float(channel_config.response_window[0]),
                float(channel_config.artifact_interval[1]),
            )
            failure = _empty_result(
                channel=channel,
                n_trials_total=int(arr.shape[0]),
                response_window=(
                    response_start,
                    float(channel_config.response_window[1]),
                ),
                baseline_window=tuple(
                    float(value) for value in channel_config.baseline_window
                ),
                parameters="",
                notes=str(exc),
            )
            # The validated configuration is serializable through the normal
            # result path; assign it after constructing the complete schema.
            failure.parameters = _parameter_json(channel_config)
            row = failure.to_record(include_arrays=include_arrays)
            row["qc_status"] = "detector_error"
        row["random_state_root"] = _seed_decimal(config.random_state)
        row["random_state_effective"] = _seed_decimal(
            effective_random_state
        )
        row["random_stream_derivation"] = (
            "sha256-v1(root-seed,channel-label;63-bit)"
            if config.random_state is not None
            else "nondeterministic"
        )
        rows.append(row)
    native = pd.DataFrame(rows)
    excluded_keys = {
        _channel_identity(channel) for channel in excluded_channels
    }
    excluded = native["channel"].map(_channel_identity).isin(excluded_keys)
    if excluded.any():
        native.loc[excluded, "qc_status"] = "stimulation_contact_excluded"
        native.loc[excluded, "notes"] = native.loc[excluded, "notes"].map(
            lambda value: "; ".join(
                part
                for part in (
                    str(value).strip(),
                    "stimulation contact excluded from the testing family",
                )
                if part
            )
        )
    eligible = native["qc_status"].eq("pass").to_numpy(dtype=bool)
    family_p_values = pd.to_numeric(
        native["p_joint"], errors="coerce"
    ).where(eligible)
    if correction == "fdr_bh":
        q_values, rejected = fdr_bh(family_p_values, alpha=alpha)
    else:
        q_values = family_p_values.to_numpy(dtype=float)
        rejected = np.isfinite(q_values) & (q_values <= alpha)
    native["q_joint"] = q_values
    native["significant"] = np.asarray(rejected, dtype=bool) & eligible
    native["crp_energy_significant"] = native["significant"]
    native["score"] = pd.to_numeric(
        native.get("rms_ratio_db", np.nan), errors="coerce"
    )
    native["p_value"] = pd.to_numeric(native["p_joint"], errors="coerce")
    native["threshold"] = alpha
    native["testing_family_size"] = int(
        np.isfinite(family_p_values.to_numpy(dtype=float)).sum()
    )
    native["multiple_comparison_correction"] = correction
    frame = _base_method_frame(
        features,
        "crp_energy",
        "primary_epochs",
    )
    drop = [
        column
        for column in ("significant", "score", "p_value", "threshold", "notes")
        if column in frame.columns
    ]
    return frame.drop(columns=drop).merge(native, on="channel", how="left")


def _channel_identity(value: Any) -> str:
    text = re.sub(r"(?:[-_ ]?REF)$", "", str(value).strip(), flags=re.I)
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _stimulation_channels_from_epochs(epochs: Any) -> tuple[str, ...]:
    values: list[Any] = []
    stim_ch = getattr(epochs, "stim_ch", None)
    if isinstance(stim_ch, (list, tuple, set, np.ndarray, pd.Index)):
        values.extend(stim_ch)
    elif stim_ch not in (None, ""):
        values.append(stim_ch)
    metadata = getattr(epochs, "metadata", {}) or {}
    metadata_stim = metadata.get("stim_ch") if isinstance(metadata, dict) else None
    if isinstance(metadata_stim, (list, tuple, set, np.ndarray, pd.Index)):
        values.extend(metadata_stim)
    elif metadata_stim not in (None, ""):
        values.append(metadata_stim)
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _kundu_envelope(
    arr: np.ndarray,
    *,
    sfreq: float,
    validity_window_mask: np.ndarray,
    channel_chunk_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sfreq <= 20.0:
        raise ValueError("Kundu detection requires a sampling rate above 20 Hz")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        common_median = np.nanmedian(arr, axis=2, keepdims=True)
    rereferenced = np.asarray(arr, dtype=float) - common_median
    valid_trials = np.all(np.isfinite(rereferenced[:, validity_window_mask, :]), axis=1)
    filled = _fill_nonfinite_for_filter(rereferenced)
    highpass = signal.butter(4, 10.0, btype="highpass", fs=sfreq, output="sos")
    lowpass = signal.butter(2, 10.0, btype="lowpass", fs=sfreq, output="sos")
    envelope = np.full_like(filled, np.nan, dtype=float)
    highpassed = np.full_like(filled, np.nan, dtype=float)
    chunk_size = max(int(channel_chunk_size), 1)
    for start in range(0, filled.shape[2], chunk_size):
        stop = min(start + chunk_size, filled.shape[2])
        filtered = signal.sosfiltfilt(highpass, filled[:, :, start:stop], axis=1)
        smoothed_power = signal.sosfiltfilt(lowpass, filtered * filtered, axis=1)
        highpassed[:, :, start:stop] = filtered
        envelope[:, :, start:stop] = np.sqrt(np.maximum(smoothed_power, 0.0))
    envelope = np.where(valid_trials[:, None, :], envelope, np.nan)
    highpassed = np.where(valid_trials[:, None, :], highpassed, np.nan)
    return envelope, highpassed, valid_trials


def _detect_kundu(
    features: pd.DataFrame,
    arr: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    channels: list[str],
    params: dict[str, Any],
    detector_input: str,
) -> pd.DataFrame:
    baseline_window = tuple(params.get("baseline_window", (-0.100, -0.005)))
    response_window = tuple(params.get("response_window", (0.005, 0.100)))
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    response_mask = (times >= response_window[0]) & (times <= response_window[1])
    validity_mask = baseline_mask | response_mask
    frame = _base_method_frame(features, "kundu_rolston", detector_input)
    if not baseline_mask.any() or not response_mask.any():
        frame["notes"] = "Kundu baseline or response window does not overlap detector input"
        return frame
    try:
        envelope, highpassed, valid_trials = _kundu_envelope(
            arr,
            sfreq=sfreq,
            validity_window_mask=validity_mask,
            channel_chunk_size=int(params.get("channel_chunk_size", 16)),
        )
    except Exception as exc:
        reason = str(exc)
        frame["method_available"] = False
        frame["availability_reason"] = reason
        frame["notes"] = reason
        for column in DETECTOR_QUANTITY_COLUMNS["crowther_gamma"]:
            if column not in frame:
                frame[column] = np.nan
        return frame

    multiplier = float(params.get("baseline_multiplier", 3.0))
    magnitude_threshold = float(params.get("magnitude_threshold_uv", 30.0))
    minimum_duration_s = float(params.get("min_duration_s", 0.015))
    required_samples = max(int(np.ceil(minimum_duration_s * sfreq)), 1)
    rows = []
    for channel_index, channel in enumerate(channels):
        baseline_values = envelope[:, baseline_mask, channel_index]
        response_values = envelope[:, response_mask, channel_index]
        baseline_median = float(np.nanmedian(baseline_values)) if np.isfinite(baseline_values).any() else np.nan
        post_median = float(np.nanmedian(response_values)) if np.isfinite(response_values).any() else np.nan
        trial_median_wave = np.nanmedian(response_values, axis=0)
        envelope_threshold = multiplier * baseline_median
        longest_samples = _longest_true_run(trial_median_wave > envelope_threshold) if np.isfinite(envelope_threshold) else 0
        longest_duration_ms = float(longest_samples / sfreq * 1000.0)
        persistence_pass = bool(longest_samples >= required_samples)
        magnitude_pass = bool(np.isfinite(post_median) and post_median > magnitude_threshold)
        mean_highpassed = np.nanmean(highpassed[:, response_mask, channel_index], axis=0)
        amplitude = float(np.nanmean(np.abs(mean_highpassed))) if np.isfinite(mean_highpassed).any() else np.nan
        ratio = float(np.nanmax(trial_median_wave) / (baseline_median + 1e-12)) if np.isfinite(trial_median_wave).any() and np.isfinite(baseline_median) else np.nan
        rows.append(
            {
                "channel": channel,
                "significant": persistence_pass and magnitude_pass,
                "score": post_median,
                "threshold": magnitude_threshold,
                "kundu_baseline_median_uv": baseline_median,
                "kundu_envelope_threshold_uv": envelope_threshold,
                "kundu_post_median_uv": post_median,
                "kundu_peak_envelope_ratio": ratio,
                "kundu_longest_suprathreshold_ms": longest_duration_ms,
                "kundu_persistence_pass": persistence_pass,
                "kundu_magnitude_pass": magnitude_pass,
                "kundu_mean_absolute_amplitude_uv": amplitude,
                "kundu_n_valid_trials": int(valid_trials[:, channel_index].sum()),
            }
        )
    native = pd.DataFrame(rows)
    frame = frame.drop(columns=["significant", "score", "threshold"]).merge(native, on="channel", how="left")
    return frame


def _detect_rms(
    features: pd.DataFrame,
    referenced: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    baseline_window: tuple[float, float],
    params: dict[str, Any],
) -> pd.DataFrame:
    response_window = tuple(params.get("response_window", (0.010, 0.100)))
    response_mask = (times >= response_window[0]) & (times <= response_window[1])
    baseline_end = float(params.get("baseline_end_s", -0.005))
    baseline_candidates = np.flatnonzero(
        (times >= baseline_window[0]) & (times <= min(baseline_window[1], baseline_end))
    )
    response_indices = np.flatnonzero(response_mask)
    frame = _base_method_frame(features, "rms_response", "primary_epochs")
    if not response_indices.size or not baseline_candidates.size:
        frame["notes"] = "RMS response or baseline window does not overlap epoch times"
        return frame
    n_match = min(len(response_indices), len(baseline_candidates))
    response_indices = response_indices[:n_match]
    baseline_indices = baseline_candidates[-n_match:]
    alpha = float(params.get("alpha", 0.05))
    n1_z_threshold = float(params.get("n1_z_threshold", 6.0))
    keller = _keller_quantities(
        referenced,
        times,
        channels,
        baseline_window=baseline_window,
        z_threshold=n1_z_threshold,
    ).set_index("channel")
    rows = []
    for channel_index, channel in enumerate(channels):
        response_segment = referenced[:, response_indices, channel_index]
        baseline_segment = referenced[:, baseline_indices, channel_index]
        response_residual = response_segment - np.mean(
            response_segment,
            axis=1,
            keepdims=True,
        )
        baseline_residual = baseline_segment - np.mean(
            baseline_segment,
            axis=1,
            keepdims=True,
        )
        response_trial_rms = _row_rms(response_residual)
        baseline_trial_rms = _row_rms(baseline_residual)
        finite = np.isfinite(response_trial_rms) & np.isfinite(baseline_trial_rms)
        response_median = float(np.nanmedian(response_trial_rms[finite])) if finite.any() else np.nan
        baseline_median = float(np.nanmedian(baseline_trial_rms[finite])) if finite.any() else np.nan
        baseline_scale = _robust_scale(baseline_trial_rms[finite])
        robust_z = float((response_median - baseline_median) / baseline_scale) if np.isfinite(baseline_scale) else np.nan
        if finite.sum() >= 3:
            try:
                test = stats.wilcoxon(
                    response_trial_rms[finite],
                    baseline_trial_rms[finite],
                    alternative="greater",
                    zero_method="wilcox",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
        else:
            statistic = p_value = np.nan
        mean_wave = np.nanmean(referenced[finite][:, response_indices, channel_index], axis=0) if finite.any() else np.array([], dtype=float)
        waveform_rms = float(np.sqrt(np.nanmean(mean_wave * mean_wave))) if np.isfinite(mean_wave).any() else np.nan
        rows.append(
            {
                "channel": channel,
                "significant": bool(
                    np.isfinite(keller.loc[channel, "keller_a1_z"])
                    and keller.loc[channel, "keller_a1_z"] >= n1_z_threshold
                ),
                "score": float(keller.loc[channel, "keller_a1_z"]),
                "p_value": np.nan,
                "threshold": n1_z_threshold,
                "rms_response_trial_median_uv": response_median,
                "rms_baseline_trial_median_uv": baseline_median,
                "rms_delta_uv": response_median - baseline_median,
                "rms_ratio": response_median / (baseline_median + 1e-12),
                "rms_robust_z": robust_z,
                "rms_wilcoxon_statistic": statistic,
                "rms_wilcoxon_p_value": p_value,
                "rms_waveform_10_100_uv": waveform_rms,
                "rms_n_paired_trials": int(finite.sum()),
                "rms_detection_n1_z": float(keller.loc[channel, "keller_a1_z"]),
                "rms_detection_n1_peak_uv": float(keller.loc[channel, "keller_a1_peak_uv"]),
                "rms_detection_n1_latency_ms": float(keller.loc[channel, "keller_a1_latency_ms"]),
                "rms_significance_basis": "10-50-ms N1 z>=6; separately demeaned paired 10-100-ms RMS is exploratory",
            }
        )
    native = pd.DataFrame(rows)
    q_values, rejected = fdr_bh(native["rms_wilcoxon_p_value"], alpha=alpha)
    native["rms_paired_q_value"] = q_values
    native["rms_paired_fdr_significant"] = rejected
    frame = frame.drop(columns=["significant", "score", "p_value", "threshold"]).merge(native, on="channel", how="left")
    return frame


def _replace_signi_stimulation_artifact(arr: np.ndarray, times: np.ndarray) -> np.ndarray:
    repaired = np.asarray(arr, dtype=float).copy()
    target = np.flatnonzero((times >= 0.0) & (times <= 0.005))
    before = np.flatnonzero((times >= -0.005) & (times < 0.0))
    after = np.flatnonzero((times >= 0.005) & (times <= 0.010))
    if not target.size or not before.size or not after.size:
        return repaired
    normalized = np.linspace(0.0, 1.0, target.size)
    before_positions = np.linspace(0, before.size - 1, target.size).round().astype(int)
    after_positions = np.linspace(0, after.size - 1, target.size).round().astype(int)
    left = repaired[:, before[before_positions][::-1], :]
    right = repaired[:, after[after_positions][::-1], :]
    taper = normalized[None, :, None]
    repaired[:, target, :] = (1.0 - taper) * left + taper * right
    return repaired


def _signi_snr(values: np.ndarray, response_times: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """SIGNI variance-ratio SNR across trial and time samples.

    Input axes end in ``trial, time, channel``; optional leading axes are
    retained (for example, a permutation axis).
    """

    full_variance = np.nanvar(values, axis=(-3, -2), ddof=1)
    bin_variances = []
    start = float(response_times[0])
    for bin_index in range(6):
        lower = start + bin_index * 0.015
        upper = start + (bin_index + 1) * 0.015
        if bin_index == 5:
            mask = (response_times >= lower) & (response_times <= upper + 1e-12)
        else:
            mask = (response_times >= lower) & (response_times < upper)
        if mask.sum() >= 2:
            bin_variances.append(
                np.nanvar(values[..., mask, :], axis=(-3, -2), ddof=1)
            )
    if not bin_variances:
        return np.full(full_variance.shape, np.nan, dtype=float)
    local_variance = np.nanmean(np.stack(bin_variances, axis=0), axis=0)
    return full_variance / (local_variance + float(eps))


def _signi_quantities(
    arr: np.ndarray,
    times: np.ndarray,
    *,
    sfreq: float,
    n_permutations: int,
    random_state: int | None,
    channel_chunk_size: int,
    permutation_chunk_size: int,
    polarity_labels: np.ndarray | None = None,
) -> pd.DataFrame:
    n_trials, n_times, n_channels = arr.shape
    if sfreq <= 340.0:
        raise ValueError("SIGNI requires a sampling rate above 340 Hz for its 70-170 Hz band")
    response_mask = (times >= 0.010) & (times <= 0.100)
    baseline_mask = (times >= -0.200) & (times < 0.0)
    if response_mask.sum() < 12 or baseline_mask.sum() < 12:
        raise ValueError("SIGNI requires 10-100 ms response and -200-0 ms baseline coverage")

    repaired = _replace_signi_stimulation_artifact(arr, times)
    labels = np.zeros(n_trials, dtype=int) if polarity_labels is None else np.asarray(polarity_labels)
    if len(labels) != n_trials:
        raise ValueError("polarity_labels must contain one label per trial")
    evoked_removed = repaired.copy()
    for label in pd.unique(labels):
        trial_mask = labels == label
        evoked_removed[trial_mask] -= np.nanmean(repaired[trial_mask], axis=0, keepdims=True)

    valid_trials = np.all(np.isfinite(evoked_removed[:, baseline_mask | response_mask, :]), axis=1)
    filled = _fill_nonfinite_for_filter(evoked_removed)
    bandpass = signal.butter(8, [70.0, 170.0], btype="bandpass", fs=sfreq, output="sos")
    envelope = np.full_like(filled, np.nan, dtype=float)
    chunk_size = max(int(channel_chunk_size), 1)
    for start in range(0, n_channels, chunk_size):
        stop = min(start + chunk_size, n_channels)
        filtered = signal.sosfiltfilt(bandpass, filled[:, :, start:stop], axis=1)
        envelope[:, :, start:stop] = np.abs(signal.hilbert(filtered, axis=1))
    envelope = np.where(valid_trials[:, None, :], envelope, np.nan)

    mean_envelope = np.nanmean(envelope, axis=0)
    response_times = times[response_mask]
    observed_snr = _signi_snr(
        envelope[:, response_mask, :],
        response_times,
    )
    response_indices = np.flatnonzero(response_mask)
    rng = np.random.default_rng(random_state)
    n_permutations = max(int(n_permutations), 1)
    shifts = rng.integers(0, n_times, size=(n_permutations, n_trials), endpoint=False)
    reversed_envelope = envelope[:, ::-1, :]
    null_snr = np.full((n_permutations, n_channels), np.nan, dtype=float)
    permutation_chunk_size = max(int(permutation_chunk_size), 1)
    for channel_start in range(0, n_channels, chunk_size):
        channel_stop = min(channel_start + chunk_size, n_channels)
        channel_values = reversed_envelope[:, :, channel_start:channel_stop]
        for permutation_start in range(0, n_permutations, permutation_chunk_size):
            permutation_stop = min(permutation_start + permutation_chunk_size, n_permutations)
            local_shifts = shifts[permutation_start:permutation_stop]
            sampled_trials = []
            for trial in range(n_trials):
                indices = (response_indices[None, :] + local_shifts[:, trial, None]) % n_times
                sampled_trials.append(channel_values[trial, indices, :])
            randomized_trials = np.stack(sampled_trials, axis=1)
            null_snr[permutation_start:permutation_stop, channel_start:channel_stop] = _signi_snr(
                randomized_trials,
                response_times,
            )

    baseline_mean = np.nanmean(mean_envelope[baseline_mask], axis=0)
    baseline_std = np.nanstd(mean_envelope[baseline_mask], axis=0, ddof=1)
    high_gamma_z = (
        np.nanmean(mean_envelope[response_mask], axis=0) - baseline_mean
    ) / (baseline_std + 1e-12)
    latency_threshold = baseline_mean + stats.norm.ppf(0.999) * baseline_std
    latencies = np.full(n_channels, np.nan, dtype=float)
    for channel in range(n_channels):
        crossings = np.flatnonzero(mean_envelope[response_mask, channel] > latency_threshold[channel])
        if crossings.size:
            latencies[channel] = float(response_times[crossings[0]] * 1000.0)

    rows = []
    for channel in range(n_channels):
        finite_null = null_snr[:, channel]
        finite_null = finite_null[np.isfinite(finite_null) & (finite_null > 0)]
        observed = float(observed_snr[channel])
        if finite_null.size >= 20 and np.isfinite(observed) and observed > 0:
            log_null = np.log(finite_null)
            null_mean = float(np.mean(log_null))
            null_std = float(np.std(log_null, ddof=1))
            if null_std > 0:
                raw_p = float(stats.norm.sf(np.log(observed), loc=null_mean, scale=null_std))
                standardized_null = (log_null - null_mean) / null_std
                ks_p = float(
                    stats.kstest(standardized_null, stats.norm.cdf).pvalue
                )
            else:
                raw_p = ks_p = np.nan
            empirical_p = float((1 + np.sum(finite_null >= observed)) / (len(finite_null) + 1))
        else:
            null_mean = null_std = raw_p = ks_p = empirical_p = np.nan
        rows.append(
            {
                "channel_index": channel,
                "signi_snr": observed,
                "signi_p_value_raw": raw_p,
                "signi_empirical_p_value": empirical_p,
                "signi_null_log_mean": null_mean,
                "signi_null_log_std": null_std,
                "signi_null_ks_p_value": ks_p,
                "signi_latency_ms": latencies[channel],
                "signi_n_permutations": int(n_permutations),
                "high_gamma_z": float(high_gamma_z[channel]),
                "signi_n_valid_trials": int(valid_trials[:, channel].sum()),
            }
        )
    return pd.DataFrame(rows)


def _detect_signi(
    features: pd.DataFrame,
    arr: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    channels: list[str],
    params: dict[str, Any],
    detector_input: str,
) -> pd.DataFrame:
    frame = _base_method_frame(features, "crowther_gamma", detector_input)
    alpha = float(params.get("alpha", 0.05))
    try:
        native = _signi_quantities(
            arr,
            times,
            sfreq=sfreq,
            n_permutations=int(params.get("n_permutations", 1000)),
            random_state=params.get("random_state", 0),
            channel_chunk_size=int(params.get("channel_chunk_size", 8)),
            permutation_chunk_size=int(params.get("permutation_chunk_size", 50)),
            polarity_labels=params.get("polarity_labels"),
        )
    except Exception as exc:
        frame["notes"] = str(exc)
        return frame
    native["channel"] = [channels[int(index)] for index in native["channel_index"]]
    native["signi_p_value_bonferroni"] = np.minimum(
        pd.to_numeric(native["signi_p_value_raw"], errors="coerce") * len(channels),
        1.0,
    )
    native["significant"] = native["signi_p_value_bonferroni"].lt(alpha)
    native["score"] = native["signi_snr"]
    native["p_value"] = native["signi_p_value_bonferroni"]
    native["threshold"] = alpha
    native = native.drop(columns=["channel_index"])
    frame = frame.drop(columns=["significant", "score", "p_value", "threshold"]).merge(native, on="channel", how="left")
    return frame


def _high_gamma_z_all(
    arr: np.ndarray,
    times: np.ndarray,
    *,
    sfreq: float,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
    channel_chunk_size: int = 8,
) -> np.ndarray:
    """Legacy descriptive high-gamma envelope z score.

    This remains available as a descriptive quantity. ``crowther_gamma`` now
    uses the full SIGNI procedure rather than thresholding this surrogate.
    """

    n_channels = int(arr.shape[2])
    scores = np.full(n_channels, np.nan, dtype=float)
    if sfreq < 320:
        return scores
    low, high = 70.0, min(150.0, sfreq / 2.0 * 0.9)
    if low >= high or arr.shape[1] < 30:
        return scores
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    response_mask = (times >= response_window[0]) & (times <= response_window[1])
    if not baseline_mask.any() or not response_mask.any():
        return scores
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    chunk_size = max(int(channel_chunk_size), 1)
    filled = _fill_nonfinite_for_filter(np.asarray(arr, dtype=float))
    for start in range(0, n_channels, chunk_size):
        stop = min(start + chunk_size, n_channels)
        filtered = signal.sosfiltfilt(sos, filled[:, :, start:stop], axis=1)
        envelope = np.abs(signal.hilbert(filtered, axis=1))
        baseline = envelope[:, baseline_mask, :].reshape(-1, stop - start)
        response = envelope[:, response_mask, :].reshape(-1, stop - start)
        scores[start:stop] = (
            np.nanmean(response, axis=0) - np.nanmean(baseline, axis=0)
        ) / (np.nanstd(baseline, axis=0, ddof=1) + 1e-12)
    return scores


def _high_gamma_z(
    epochs,
    channel: str,
    baseline_window=(-0.5, -0.03),
    response_window=(0.01, 0.3),
) -> float:
    arr, times, channels = epochs.as_array()
    if channel not in channels:
        return np.nan
    index = channels.index(channel)
    scores = _high_gamma_z_all(
        arr[:, :, index : index + 1],
        times,
        sfreq=float(epochs.sfreq),
        baseline_window=baseline_window,
        response_window=response_window,
    )
    return float(scores[0]) if scores.size else np.nan


def _crp_significance_from_array(
    x: np.ndarray,
    times: np.ndarray,
    *,
    response_window: tuple[float, float],
    baseline_window: tuple[float, float] | None = None,
    min_trials: int = 3,
) -> tuple[float, float]:
    if baseline_window is None:
        prestimulus = np.asarray(times, dtype=float)
        baseline_stop = min(-0.03, float(response_window[0]))
        prestimulus = prestimulus[prestimulus <= baseline_stop]
        if not prestimulus.size:
            raise ValueError("CRP requires a prestimulus baseline")
        baseline_window = (float(prestimulus[0]), float(prestimulus[-1]))
    result = run_crp_array(
        x,
        times,
        config=CRPConfig(
            response_window=(float(response_window[0]), float(response_window[1])),
            baseline_window=baseline_window,
            min_trials=min_trials,
        ),
    )
    return result.score, result.p_value


def detect_erp_all(
    epochs,
    methods: Iterable[str] | None = None,
    min_consensus: int = 2,
    wideband_epochs=None,
    **params: Any,
) -> pd.DataFrame:
    """Run source-informed detector quantities and an explicit primary rule.

    ``consensus`` remains a legacy availability-aware positive-call summary
    over standalone published detectors. SIGNI/high gamma is unavailable unless a
    passband-preserving ``wideband_epochs`` object is supplied explicitly, or
    the caller explicitly declares the primary input wideband with
    ``crowther_gamma={"input_is_wideband": True}``. The recommended
    ``primary_significant`` field is the BH-adjusted CRP-energy
    intersection-union conjunction. The historical CRP-plus-Kundu
    ``shape_magnitude_significant`` field remains available as a legacy
    selection summary. Artifact eligibility is applied by :func:`ERPy.audit_waveforms`
    or the batch QC layer.
    """

    methods = [str(method).lower() for method in (methods or ALL_METHODS)]
    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown ERP detection method(s): {unknown}")
    baseline_window = tuple(params.get("baseline_window", (-0.5, -0.03)))
    response_window = tuple(params.get("response_window", (0.01, 0.3)))
    arr, times, channels = epochs.as_array()
    channels = [str(channel) for channel in channels]
    times = np.asarray(times, dtype=float)
    referenced = baseline_center_array(arr, times, baseline_window)
    features = _feature_table_from_array(
        referenced,
        times,
        channels,
        sfreq=float(epochs.sfreq),
        baseline_window=baseline_window,
        response_window=response_window,
        response_boundary_guard_s=float(params.get("response_boundary_guard_s", 0.002)),
        baseline_centered=True,
    )

    detector_epochs = wideband_epochs if wideband_epochs is not None else epochs
    detector_arr, detector_times, detector_sfreq, missing_wide_channels = _aligned_epoch_arrays(detector_epochs, channels)
    detector_input = "wideband_epochs" if wideband_epochs is not None else "primary_epochs"
    frames = []
    for method in methods:
        method_params = params.get(method, {}) if isinstance(params.get(method, {}), dict) else {}
        shared = {
            key: value
            for key, value in params.items()
            if key not in ALL_METHODS
            and key not in {"baseline_window", "response_window", "response_boundary_guard_s"}
        }
        options = {**shared, **method_params}
        if method == "kundu_rolston":
            frame = _detect_kundu(
                features,
                detector_arr,
                detector_times,
                detector_sfreq,
                channels,
                options,
                detector_input,
            )
        elif method == "keller_zscore":
            frame = _detect_keller(features, referenced, times, channels, baseline_window, options)
        elif method == "crp_significance":
            frame = _detect_crp(features, referenced, times, channels, baseline_window, response_window, options)
        elif method == "crp_energy":
            excluded_channels = (
                _stimulation_channels_from_epochs(epochs)
                if bool(options.get("exclude_stimulation_contacts", True))
                else ()
            )
            frame = _detect_crp_energy(
                features,
                arr,
                times,
                channels,
                baseline_window,
                response_window,
                options,
                excluded_channels=excluded_channels,
            )
        elif method == "crowther_gamma":
            input_declared_wideband = bool(options.get("input_is_wideband", False))
            if wideband_epochs is None and not input_declared_wideband:
                frame = _unavailable_signi_frame(
                    features,
                    detector_input="primary_epochs",
                    reason=(
                        "SIGNI/high gamma is unavailable: pass an explicit "
                        "wideband_epochs input retaining 70-170 Hz, or declare "
                        "a verified wideband primary input with "
                        "crowther_gamma={'input_is_wideband': True}"
                    ),
                )
            else:
                signi_input = (
                    "wideband_epochs"
                    if wideband_epochs is not None
                    else "primary_epochs_declared_wideband"
                )
                frame = _detect_signi(
                    features,
                    detector_arr,
                    detector_times,
                    detector_sfreq,
                    channels,
                    options,
                    signi_input,
                )
        elif method == "peak_amplitude":
            frame = _detect_peak(features, referenced, times, channels, baseline_window, options)
        elif method == "rms_response":
            frame = _detect_rms(features, referenced, times, channels, baseline_window, options)
        else:
            raise AssertionError(method)
        if missing_wide_channels and method in {"kundu_rolston", "crowther_gamma"}:
            missing_mask = frame["channel"].isin(missing_wide_channels)
            frame.loc[missing_mask, "significant"] = False
            frame.loc[missing_mask, "notes"] = "channel absent from wideband detector input"
            frame.loc[missing_mask, "method_available"] = False
            frame.loc[missing_mask, "availability_reason"] = (
                "channel absent from wideband detector input"
            )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        output = pd.concat(frames, ignore_index=True, sort=False)
    output["significant"] = output["significant"].fillna(False).astype(bool)
    output["method_available"] = output["method_available"].fillna(False).astype(bool)
    available_legacy = output[
        output["method"].isin(LEGACY_CONSENSUS_METHODS)
        & output["method_available"]
    ]
    method_counts = (
        available_legacy
        .groupby("channel")["significant"]
        .sum()
        .reindex(channels, fill_value=0)
        .rename("n_methods_significant")
    )
    available_counts = (
        available_legacy.groupby("channel")["method"]
        .nunique()
        .reindex(channels, fill_value=0)
        .rename("n_methods_available")
    )
    output = output.merge(method_counts, on="channel", how="left")
    output = output.merge(available_counts, on="channel", how="left")
    output["consensus_ch"] = output["n_methods_significant"].ge(int(min_consensus))
    output["consensus"] = output["consensus_ch"]

    crp_rows = output[output["method"].eq("crp_significance")].drop_duplicates("channel").set_index("channel")
    kundu_rows = output[output["method"].eq("kundu_rolston")].drop_duplicates("channel").set_index("channel")
    crp_pass = (
        crp_rows.get("crp_fdr_significant", pd.Series(dtype="boolean"))
        .astype("boolean")
        .reindex(channels)
        .fillna(False)
        .astype(bool)
    )
    kundu_pass = (
        kundu_rows.get("significant", pd.Series(dtype="boolean"))
        .astype("boolean")
        .reindex(channels)
        .fillna(False)
        .astype(bool)
    )
    historical_shape_magnitude = crp_pass & kundu_pass
    energy_rows = (
        output[output["method"].eq("crp_energy")]
        .drop_duplicates("channel")
        .set_index("channel")
    )
    primary = (
        energy_rows.get("significant", pd.Series(dtype="boolean"))
        .astype("boolean")
        .reindex(channels)
        .fillna(False)
        .astype(bool)
    )
    reproducibility_pass = (
        energy_rows.get("crp_significant", pd.Series(dtype="boolean"))
        .astype("boolean")
        .reindex(channels)
        .fillna(False)
        .astype(bool)
    )
    energy_pass = (
        energy_rows.get("energy_significant", pd.Series(dtype="boolean"))
        .astype("boolean")
        .reindex(channels)
        .fillna(False)
        .astype(bool)
    )
    joint_p = pd.to_numeric(
        energy_rows.get("p_joint", pd.Series(dtype=float)), errors="coerce"
    ).reindex(channels)
    joint_q = pd.to_numeric(
        energy_rows.get("q_joint", pd.Series(dtype=float)), errors="coerce"
    ).reindex(channels)
    classifications = (
        energy_rows.get("classification", pd.Series(dtype=str))
        .reindex(channels)
        .fillna("not_evaluated")
        .astype(str)
    )
    output["primary_criterion"] = PRIMARY_CRITERION
    output["primary_reproducibility_pass"] = (
        output["channel"].map(reproducibility_pass).fillna(False).astype(bool)
    )
    output["primary_energy_pass"] = (
        output["channel"].map(energy_pass).fillna(False).astype(bool)
    )
    output["primary_joint_p_value"] = output["channel"].map(joint_p)
    output["primary_joint_q_value"] = output["channel"].map(joint_q)
    output["primary_classification"] = output["channel"].map(classifications)
    output["primary_significant"] = (
        output["channel"].map(primary).fillna(False).astype(bool)
    )
    output["primary_shape_pass"] = output["channel"].map(crp_pass).fillna(False).astype(bool)
    output["primary_magnitude_pass"] = output["channel"].map(kundu_pass).fillna(False).astype(bool)
    output["shape_magnitude_significant"] = output["channel"].map(historical_shape_magnitude).fillna(False).astype(bool)
    return output


def detect_erp(
    epochs,
    method: str,
    baseline_window: tuple[float, float] = (-0.5, -0.03),
    response_window: tuple[float, float] = (0.01, 0.3),
    **params: Any,
) -> pd.DataFrame:
    return detect_erp_all(
        epochs,
        methods=[method],
        min_consensus=1,
        baseline_window=baseline_window,
        response_window=response_window,
        **params,
    )


def detect(
    epochs,
    method: str = "crp_energy",
    **params: Any,
) -> pd.DataFrame:
    """Low-syntax detector entry point; defaults to the primary conjunction."""

    return detect_erp(epochs, method=method, **params)


def n1_latency(epochs, channel: str) -> float:
    return float(_feature_table(epochs).set_index("channel").loc[channel, "n1_latency_ms"])


def peak_amplitude(epochs, channel: str) -> float:
    return float(_feature_table(epochs).set_index("channel").loc[channel, "peak_amplitude_uv"])


def rms_response(epochs, channel: str) -> float:
    return float(_feature_table(epochs).set_index("channel").loc[channel, "rms_uv"])


def auc_response(epochs, channel: str) -> float:
    return float(_feature_table(epochs).set_index("channel").loc[channel, "auc_uv_ms"])


def maximal_descent(epochs, channel: str) -> float:
    return float(_feature_table(epochs).set_index("channel").loc[channel, "max_descent_ms"])


def high_gamma_envelope(epochs, channel: str) -> float:
    return _high_gamma_z(epochs, channel)
