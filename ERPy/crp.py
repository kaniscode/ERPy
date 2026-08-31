from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CRPConfig:
    """Configuration for canonical response parametrization.

    The extraction follows Miller et al. (2023): semi-normalized reciprocal
    trial projections determine a data-driven response duration, a balanced
    half of those projections supplies the one-sided extraction test, and a
    linear kernel PCA supplies the canonical response shape.
    """

    response_window: tuple[float, float] = (0.015, 0.3)
    baseline_window: tuple[float, float] = (-0.5, -0.03)
    min_trials: int = 3
    alpha: float = 0.05
    projection_start_s: float = 0.010
    projection_step_s: float = 0.005
    permutation_n: int = 0
    random_state: int | None = 0
    eps: float = 1e-12


@dataclass
class CRPResult:
    """Canonical response parametrization result for one recording channel.

    The object retains the fitted waveform, reciprocal trial projections,
    data-driven response duration, extraction-test quantities, reconstruction
    diagnostics, and optional permutation null. Time windows and durations are
    in seconds; amplitude and residual fields carrying an ``_uv`` suffix are
    in microvolts. ``significant`` and ``p_value`` describe the unadjusted CRP
    extraction test, not the separate CRP-energy conjunction.
    """

    channel: str
    significant: bool
    score: float
    p_value: float
    threshold: float
    canonical_waveform: np.ndarray
    times: np.ndarray
    projections: np.ndarray
    response_window: tuple[float, float]
    baseline_window: tuple[float, float]
    response_duration_s: float = np.nan
    t_value: float = np.nan
    alpha_prime_uv: np.ndarray | None = None
    residual_rms_uv: np.ndarray | None = None
    snr: np.ndarray | None = None
    explained_variance: np.ndarray | None = None
    cross_projections: np.ndarray | None = None
    projection_times: np.ndarray | None = None
    mean_projection_profile: np.ndarray | None = None
    full_window_t_value: float = np.nan
    full_window_p_value: float = np.nan
    canonical_weight_times: np.ndarray | None = None
    canonical_weight_mean: np.ndarray | None = None
    canonical_weight_sem: np.ndarray | None = None
    canonical_weight_z: np.ndarray | None = None
    baseline_weight_mean: float = np.nan
    baseline_weight_std: float = np.nan
    max_canonical_weight_z: float = np.nan
    reconstruction_rms_uv: float = np.nan
    normalized_reconstruction_energy: float = np.nan
    energy_z: float = np.nan
    permutation_p_value: float = np.nan
    null_distribution: np.ndarray | None = None

    @property
    def canonical_response_curve(self) -> np.ndarray:
        """Return the fitted canonical waveform as a compatibility alias."""

        return self.canonical_waveform


class CRPError(ValueError):
    pass


def epochs_to_matrix(epochs, channel: str) -> tuple[np.ndarray, np.ndarray]:
    """Return trial-by-time voltage data and its time axis for one channel."""

    arr, times, channels = epochs.as_array()
    if channel not in channels:
        raise KeyError(channel)
    return np.asarray(arr[:, :, channels.index(channel)], dtype=float), np.asarray(times, dtype=float)


def _sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return std / np.sqrt(counts)


def _zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr, ddof=1)
    if not np.isfinite(std) or std == 0:
        return np.full(arr.shape, np.nan, dtype=float)
    return (arr - mean) / std


def balanced_projection_indices(n_trials: int) -> np.ndarray:
    """Select one projection from every reciprocal trial pair.

    Indices address a row-major flattened off-diagonal projection matrix.
    The construction is the zero-based equivalent of the published MATLAB
    helper and balances which member of each pair is L2-normalized.
    """

    n_trials = int(n_trials)
    if n_trials < 2:
        raise CRPError("CRP requires at least two trials")
    total = n_trials * (n_trials - 1)
    indices = np.arange(0, total, 2, dtype=int)
    if n_trials % 2:
        offsets = np.zeros_like(indices)
        half = (n_trials - 1) // 2
        for trial_number in range(1, n_trials + 1):
            if trial_number % 2 == 0:
                start = (trial_number - 1) * half
                stop = trial_number * half
                offsets[start:stop] = 1
        indices = indices + offsets
    return indices


def semi_normalized_cross_projections(
    voltage: np.ndarray,
    *,
    sfreq: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return off-diagonal semi-normalized CRP projections.

    ``voltage`` is time by trial. The normalizing trial is unit length while
    the projected-into trial retains voltage scale. Division by ``sqrt(fs)``
    converts the discrete projection to the published voltage-sqrt(second)
    scale.
    """

    v = np.asarray(voltage, dtype=float)
    if v.ndim != 2:
        raise CRPError("voltage must have shape (time, trials)")
    norms = np.linalg.norm(v, axis=0)
    normalized = v / np.maximum(norms, float(eps))[None, :]
    projections = normalized.T @ v
    np.fill_diagonal(projections, np.nan)
    values = projections.ravel()
    return values[np.isfinite(values)] / np.sqrt(float(sfreq))


def _one_sample_greater(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or np.nanstd(finite, ddof=1) <= 0:
        return np.nan, np.nan
    try:
        result = stats.ttest_1samp(finite, 0.0, alternative="greater")
    except TypeError:  # scipy < 1.9 compatibility
        result = stats.ttest_1samp(finite, 0.0)
        two_sided = float(result.pvalue)
        p_value = two_sided / 2.0 if float(result.statistic) >= 0 else 1.0 - two_sided / 2.0
        return float(result.statistic), float(p_value)
    return float(result.statistic), float(result.pvalue)


def _projection_sample_counts(n_samples: int, sfreq: float, config: CRPConfig) -> np.ndarray:
    first = max(int(round(float(config.projection_start_s) * sfreq)), 2)
    step = max(int(round(float(config.projection_step_s) * sfreq)), 1)
    if n_samples <= first:
        return np.asarray([n_samples], dtype=int)
    counts = np.arange(first, n_samples + 1, step, dtype=int)
    if counts[-1] != n_samples:
        counts = np.r_[counts, n_samples]
    return counts


def _mean_projection_profile(voltage: np.ndarray, counts: np.ndarray, sfreq: float, eps: float) -> np.ndarray:
    """Compute the CRP duration profile in O(trials x time)."""

    summed_trials = np.sum(voltage, axis=1)
    norm_sq = np.cumsum(voltage * voltage, axis=0)
    off_diagonal_dot = np.cumsum(
        voltage * (summed_trials[:, None] - voltage),
        axis=0,
    )
    n_trials = voltage.shape[1]
    denominator = float(n_trials * (n_trials - 1)) * np.sqrt(float(sfreq))
    profile = np.full(len(counts), np.nan, dtype=float)
    for index, count in enumerate(counts):
        trial_norms = np.sqrt(np.maximum(norm_sq[count - 1], eps))
        profile[index] = float(
            np.sum(off_diagonal_dot[count - 1] / trial_norms) / denominator
        )
    return profile


def _sliding_canonical_weights(
    x: np.ndarray,
    times: np.ndarray,
    canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    window_len = len(canonical)
    if window_len <= 1 or x.shape[1] < window_len:
        return np.array([], dtype=float), np.empty((x.shape[0], 0), dtype=float)
    n_windows = x.shape[1] - window_len + 1
    centers = times[np.arange(n_windows) + window_len // 2]
    weights = np.full((x.shape[0], n_windows), np.nan, dtype=float)
    scale = np.sqrt(float(window_len))
    for start in range(n_windows):
        window = x[:, start : start + window_len]
        finite = np.all(np.isfinite(window), axis=1)
        if finite.any():
            weights[finite, start] = (window[finite] @ canonical) / scale
    return centers, weights


def run_crp_array(
    x: np.ndarray,
    times: np.ndarray,
    *,
    channel: str = "",
    config: CRPConfig | None = None,
) -> CRPResult:
    """Run published CRP extraction on trial-by-time voltage data."""

    config = config or CRPConfig()
    x = np.asarray(x, dtype=float)
    times = np.asarray(times, dtype=float).squeeze()
    if x.ndim != 2 or times.ndim != 1 or x.shape[1] != len(times):
        raise CRPError("x must be trial by time and match the one-dimensional time axis")
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise CRPError("times must be strictly increasing")
    sfreq = float(1.0 / np.median(np.diff(times)))
    baseline_mask = (times >= config.baseline_window[0]) & (times <= config.baseline_window[1])
    response_mask = (times >= config.response_window[0]) & (times <= config.response_window[1])
    if not baseline_mask.any():
        raise CRPError("Baseline window does not overlap epoch times")
    if not response_mask.any():
        raise CRPError("Response window does not overlap epoch times")

    baseline_values = x[:, baseline_mask]
    baseline_counts = np.sum(np.isfinite(baseline_values), axis=1)
    baseline_means = np.divide(
        np.nansum(baseline_values, axis=1),
        baseline_counts,
        out=np.full(x.shape[0], np.nan, dtype=float),
        where=baseline_counts > 0,
    )
    centered = x - baseline_means[:, None]
    finite_trials = np.isfinite(baseline_means) & np.all(np.isfinite(centered[:, response_mask]), axis=1)
    if int(finite_trials.sum()) < int(config.min_trials):
        raise CRPError(
            f"CRP requires at least {config.min_trials} finite baseline and response-window trials"
        )
    centered = centered[finite_trials]
    response_times = times[response_mask]
    voltage = centered[:, response_mask].T

    counts = _projection_sample_counts(len(response_times), sfreq, config)
    profile = _mean_projection_profile(voltage, counts, sfreq, config.eps)
    if not np.isfinite(profile).any():
        raise CRPError("CRP projection profile is degenerate")
    duration_index = int(np.nanargmax(profile))
    response_samples = int(counts[duration_index])
    voltage_response = voltage[:response_samples]

    u, singular_values, _ = np.linalg.svd(voltage_response, full_matrices=False)
    if not singular_values.size or singular_values[0] <= config.eps:
        raise CRPError("CRP canonical response is degenerate")
    canonical = u[:, 0]
    if float(np.dot(canonical, np.mean(voltage_response, axis=1))) < 0:
        canonical = -canonical
    alpha = canonical @ voltage_response
    alpha_prime = alpha / np.sqrt(float(response_samples))
    residual = voltage_response - np.outer(canonical, alpha)
    residual_energy = np.sum(residual * residual, axis=0)
    total_energy = np.sum(voltage_response * voltage_response, axis=0)
    residual_rms = np.sqrt(residual_energy / float(response_samples))
    snr = alpha / np.sqrt(np.maximum(residual_energy, config.eps))
    explained_variance = 1.0 - residual_energy / np.maximum(total_energy, config.eps)

    cross_projections = semi_normalized_cross_projections(
        voltage_response,
        sfreq=sfreq,
        eps=config.eps,
    )
    stat_indices = balanced_projection_indices(voltage_response.shape[1])
    statistical_projections = cross_projections[stat_indices]
    t_value, p_value = _one_sample_greater(statistical_projections)

    full_cross = semi_normalized_cross_projections(voltage, sfreq=sfreq, eps=config.eps)
    full_statistical = full_cross[balanced_projection_indices(voltage.shape[1])]
    full_t, full_p = _one_sample_greater(full_statistical)

    canonical_times = response_times[:response_samples]
    weight_times, weight_trials = _sliding_canonical_weights(centered, times, canonical)
    weight_mean = np.nanmean(weight_trials, axis=0) if weight_trials.size else np.array([], dtype=float)
    weight_sem = _sem(weight_trials, axis=0) if weight_trials.size else np.array([], dtype=float)
    weight_baseline_mask = (weight_times >= config.baseline_window[0]) & (weight_times <= config.baseline_window[1])
    weight_response_mask = (weight_times >= config.response_window[0]) & (weight_times <= config.response_window[1])
    baseline_weights = weight_trials[:, weight_baseline_mask].ravel() if weight_baseline_mask.any() else np.array([], dtype=float)
    baseline_weights = baseline_weights[np.isfinite(baseline_weights)]
    baseline_weight_mean = float(np.mean(baseline_weights)) if baseline_weights.size else np.nan
    baseline_weight_std = float(np.std(baseline_weights, ddof=1)) if baseline_weights.size > 1 else np.nan
    if np.isfinite(baseline_weight_std) and baseline_weight_std > 0:
        weight_z = (weight_mean - baseline_weight_mean) / baseline_weight_std
    else:
        weight_z = np.full(weight_mean.shape, np.nan, dtype=float)
    response_weight_z = weight_z[weight_response_mask] if weight_response_mask.any() else np.array([], dtype=float)
    max_weight_z = float(np.nanmax(np.abs(response_weight_z))) if np.isfinite(response_weight_z).any() else np.nan

    reconstruction = np.outer(canonical, alpha)
    reconstruction_rms = float(np.sqrt(np.nanmean(reconstruction * reconstruction)))
    if baseline_weights.size:
        baseline_reconstruction = np.abs(baseline_weights)
        normalized_reconstruction = float(
            np.nanmedian(np.abs(alpha_prime)) / (np.nanmedian(baseline_reconstruction) + config.eps)
        )
        energy_z = float(
            (np.nanmean(np.abs(alpha_prime)) - np.nanmean(baseline_reconstruction))
            / (np.nanstd(baseline_reconstruction, ddof=1) + config.eps)
        )
    else:
        normalized_reconstruction = np.nan
        energy_z = np.nan

    null_distribution = np.array([], dtype=float)
    permutation_p = np.nan
    if int(config.permutation_n) > 0 and baseline_weights.size:
        rng = np.random.default_rng(config.random_state)
        observed = abs(float(np.nanmean(alpha_prime)))
        null_distribution = np.asarray(
            [
                abs(float(np.mean(rng.choice(baseline_weights, size=len(alpha_prime), replace=True))))
                for _ in range(int(config.permutation_n))
            ],
            dtype=float,
        )
        permutation_p = float(
            (1 + np.sum(null_distribution >= observed)) / (len(null_distribution) + 1)
        )

    significant = bool(np.isfinite(p_value) and p_value < float(config.alpha))
    if np.isfinite(permutation_p):
        significant = significant and permutation_p < float(config.alpha)
    return CRPResult(
        channel=str(channel),
        significant=significant,
        score=float(t_value) if np.isfinite(t_value) else np.nan,
        p_value=float(p_value) if np.isfinite(p_value) else np.nan,
        threshold=float(config.alpha),
        canonical_waveform=canonical,
        times=canonical_times,
        projections=alpha,
        response_window=config.response_window,
        baseline_window=config.baseline_window,
        response_duration_s=float(response_samples / sfreq),
        t_value=float(t_value) if np.isfinite(t_value) else np.nan,
        alpha_prime_uv=alpha_prime,
        residual_rms_uv=residual_rms,
        snr=snr,
        explained_variance=explained_variance,
        cross_projections=statistical_projections,
        projection_times=response_times[counts - 1],
        mean_projection_profile=profile,
        full_window_t_value=float(full_t) if np.isfinite(full_t) else np.nan,
        full_window_p_value=float(full_p) if np.isfinite(full_p) else np.nan,
        canonical_weight_times=weight_times,
        canonical_weight_mean=weight_mean,
        canonical_weight_sem=weight_sem,
        canonical_weight_z=weight_z,
        baseline_weight_mean=baseline_weight_mean,
        baseline_weight_std=baseline_weight_std,
        max_canonical_weight_z=max_weight_z,
        reconstruction_rms_uv=reconstruction_rms,
        normalized_reconstruction_energy=normalized_reconstruction,
        energy_z=energy_z,
        permutation_p_value=permutation_p,
        null_distribution=null_distribution,
    )


def run_crp(epochs, channel: str, config: CRPConfig | None = None) -> CRPResult:
    """Fit canonical response parameterization to one epoch channel."""

    x, times = epochs_to_matrix(epochs, channel)
    return run_crp_array(x, times, channel=channel, config=config)


def run_crp_all(epochs, config: CRPConfig | None = None) -> pd.DataFrame:
    """Fit canonical response parameterization to every epoch channel.

    Per-channel failures are retained as nonsignificant rows with an ``error``
    field so that one unsuitable channel does not discard the full analysis.
    """

    rows: list[dict[str, Any]] = []
    for channel in epochs.channels:
        try:
            result = run_crp(epochs, channel, config=config)
            alpha_prime = np.asarray(result.alpha_prime_uv, dtype=float)
            snr = np.asarray(result.snr, dtype=float)
            explained = np.asarray(result.explained_variance, dtype=float)
            rows.append(
                {
                    "channel": channel,
                    "significant": result.significant,
                    "score": result.score,
                    "mean_projection": float(np.nanmean(result.projections)),
                    "p_value": result.p_value,
                    "threshold": result.threshold,
                    "response_window_start": result.response_window[0],
                    "response_window_stop": result.response_window[1],
                    "baseline_window_start": result.baseline_window[0],
                    "baseline_window_stop": result.baseline_window[1],
                    "response_duration_s": result.response_duration_s,
                    "alpha_prime_mean_uv": float(np.nanmean(alpha_prime)),
                    "alpha_prime_median_uv": float(np.nanmedian(alpha_prime)),
                    "snr_median": float(np.nanmedian(snr)),
                    "explained_variance_median": float(np.nanmedian(explained)),
                    "baseline_weight_mean": result.baseline_weight_mean,
                    "baseline_weight_std": result.baseline_weight_std,
                    "max_canonical_weight_z": result.max_canonical_weight_z,
                    "reconstruction_rms_uv": result.reconstruction_rms_uv,
                    "normalized_reconstruction_energy": result.normalized_reconstruction_energy,
                    "energy_z": result.energy_z,
                    "permutation_p_value": result.permutation_p_value,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "channel": channel,
                    "significant": False,
                    "score": np.nan,
                    "p_value": np.nan,
                    "threshold": (config or CRPConfig()).alpha,
                    "mean_projection": np.nan,
                    "response_duration_s": np.nan,
                    "alpha_prime_mean_uv": np.nan,
                    "alpha_prime_median_uv": np.nan,
                    "snr_median": np.nan,
                    "explained_variance_median": np.nan,
                    "baseline_weight_mean": np.nan,
                    "baseline_weight_std": np.nan,
                    "max_canonical_weight_z": np.nan,
                    "reconstruction_rms_uv": np.nan,
                    "normalized_reconstruction_energy": np.nan,
                    "energy_z": np.nan,
                    "permutation_p_value": np.nan,
                    "notes": str(exc),
                }
            )
    return pd.DataFrame(rows)


def compare_crp_across_stim_sites(
    crp_tables: dict[str, pd.DataFrame] | pd.DataFrame,
    *,
    stim_col: str = "stim_pair",
    channel_col: str = "channel",
    weight_col: str = "mean_projection",
    baseline_col: str = "max_canonical_weight_z",
    reconstruction_col: str = "reconstruction_rms_uv",
    permutation_col: str = "permutation_p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare CRP response expression across stimulation sites.

    Accept either one table containing ``stim_col`` or a mapping from
    stimulation-pair labels to tables returned by ``run_crp_all``. The
    returned long-form table preserves the source columns and adds within-site
    z-scores, baseline-normalized expression, reconstruction ranks, and the
    supplied permutation-test summaries. ``permutation_significant`` applies
    the unadjusted ``alpha`` threshold; this helper does not perform
    multiplicity correction.
    """

    if isinstance(crp_tables, dict):
        pieces = []
        for stim_pair, table in crp_tables.items():
            if table is None or table.empty:
                continue
            piece = table.copy()
            piece[stim_col] = str(stim_pair)
            pieces.append(piece)
        data = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    else:
        data = crp_tables.copy()
    if data.empty:
        return data
    if stim_col not in data.columns:
        raise ValueError(f"CRP comparison requires a '{stim_col}' column or a dict keyed by stimulation site")
    if channel_col not in data.columns:
        raise ValueError(f"CRP comparison requires a '{channel_col}' column")

    out = data.copy()
    for col in [weight_col, baseline_col, reconstruction_col, permutation_col]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["within_stim_z"] = out.groupby(stim_col, group_keys=False)[weight_col].transform(_zscore)
    out["baseline_z"] = out[baseline_col]
    out["reconstruction_rms_within_stim_z"] = out.groupby(stim_col, group_keys=False)[reconstruction_col].transform(_zscore)
    out["permutation_significant"] = out[permutation_col] < alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        out["permutation_log10_p"] = -np.log10(out[permutation_col])
    out.loc[~np.isfinite(out["permutation_log10_p"]), "permutation_log10_p"] = np.nan

    sortable = out[["within_stim_z", "baseline_z", "reconstruction_rms_within_stim_z"]].abs().max(axis=1)
    out["comparison_magnitude"] = sortable
    out["comparison_rank_within_stim"] = out.groupby(stim_col)["comparison_magnitude"].rank(
        ascending=False,
        method="dense",
    )
    ordered_cols = [
        stim_col,
        channel_col,
        weight_col,
        "within_stim_z",
        "baseline_z",
        reconstruction_col,
        "reconstruction_rms_within_stim_z",
        "energy_z",
        permutation_col,
        "permutation_significant",
        "permutation_log10_p",
        "comparison_magnitude",
        "comparison_rank_within_stim",
    ]
    front = [col for col in ordered_cols if col in out.columns]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest].sort_values([stim_col, "comparison_rank_within_stim", channel_col])
