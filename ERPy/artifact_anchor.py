from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


POST_ARTIFACT_ZERO = "post_artifact"


def _as_channel_list(stim_ch: str | Iterable[str] | None) -> list[str]:
    if stim_ch is None:
        return []
    if isinstance(stim_ch, str):
        return [stim_ch]
    return [str(ch) for ch in stim_ch]


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanstd(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def artifact_envelope(epoch_df: pd.DataFrame, stim_ch: str | Iterable[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Return a robust stimulation-artifact envelope for one epoch.

    Stimulation channels are included first when present, but the envelope also
    considers the remaining numeric channels. This catches adjacent-contact or
    amplifier-wide onset spikes without requiring the user to know where the
    artifact is largest.
    """

    numeric = epoch_df.select_dtypes(include=[np.number])
    if numeric.empty:
        return np.array([], dtype=float), []
    stim_cols = [ch for ch in _as_channel_list(stim_ch) if ch in numeric.columns]
    cols = list(dict.fromkeys(stim_cols + list(numeric.columns)))
    values = numeric[cols].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        if values.shape[1] <= 2:
            envelope = np.nanmax(np.abs(values), axis=1)
        else:
            envelope = np.nanpercentile(np.abs(values), 95, axis=1)
    return envelope.astype(float), cols


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    if window <= 1 or values.size <= 2:
        return values
    window = min(window, values.size)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def detect_post_artifact_anchor(
    epoch_df: pd.DataFrame,
    *,
    stim_ch: str | Iterable[str] | None = None,
    baseline_window: tuple[float, float] | None = None,
    search_window: tuple[float, float] = (-0.002, 0.02),
    min_anchor_time_s: float = 0.003,
    settle_z: float = 2.0,
    settle_slope_z: float = 2.0,
    settle_duration_s: float = 0.002,
    smooth_s: float = 0.001,
    onset_artifact_z: float = 6.0,
    consecutive_samples: int | None = None,
) -> dict[str, object]:
    """Detect a post-stimulation-artifact zeroing anchor for one epoch.

    The detector finds the largest artifact-envelope sample near stimulation
    onset, then chooses the first post-peak sample at which the artifact has
    truly *settled* — i.e. the (lightly smoothed) envelope has both returned to
    near baseline (below ``settle_z``) and flattened out (sample-to-sample slope
    below ``settle_slope_z`` in baseline units), and stays settled for
    ``settle_duration_s``. Requiring a sustained return to baseline rather than a
    single threshold crossing places the anchor at the end of the artifact — the
    inflection where the steep stimulation transient gives way to the
    physiological response — instead of part-way down the still-decaying tail.

    If no sustained settled run is found, the anchor falls back to the first
    post-peak sample that meets the amplitude criterion, and finally to the
    quietest sample in the search window; it never lands on the artifact peak.
    """

    times = epoch_df.index.to_numpy(dtype=float)
    if times.size == 0:
        return {
            "anchor_time_s": np.nan,
            "anchor_index": -1,
            "artifact_peak_time_s": np.nan,
            "artifact_peak_score_z": np.nan,
            "onset_score_z": np.nan,
            "onset_is_artifact": False,
            "settled": False,
            "method": "empty_epoch",
            "reference_channels": "",
        }
    envelope, ref_cols = artifact_envelope(epoch_df, stim_ch=stim_ch)
    if envelope.size != times.size or envelope.size == 0:
        zero_idx = int(np.argmin(np.abs(times)))
        return {
            "anchor_time_s": float(times[zero_idx]),
            "anchor_index": zero_idx,
            "artifact_peak_time_s": float(times[zero_idx]),
            "artifact_peak_score_z": np.nan,
            "onset_score_z": np.nan,
            "onset_is_artifact": False,
            "settled": False,
            "method": "no_numeric_reference",
            "reference_channels": "",
        }

    if baseline_window is not None:
        baseline_mask = (times >= float(baseline_window[0])) & (times <= float(baseline_window[1]))
    else:
        baseline_mask = times < 0
    if not baseline_mask.any():
        baseline_mask = times < 0

    sample_dt = float(np.nanmedian(np.diff(times))) if times.size > 1 else 0.0
    if not np.isfinite(sample_dt) or sample_dt <= 0:
        sample_dt = 0.0

    # Lightly smooth the envelope so the settling test reflects the artifact
    # decay rather than per-sample noise in the recording channels.
    smooth_win = max(int(round(float(smooth_s) / sample_dt)), 1) if sample_dt > 0 else 1
    smoothed = _smooth(envelope, smooth_win)

    center, scale = _robust_location_scale(smoothed[baseline_mask] if baseline_mask.any() else smoothed)
    score = (smoothed - center) / scale
    finite_score = np.where(np.isfinite(score), score, -np.inf)

    # Sample-to-sample slope of the smoothed envelope in baseline-robust units.
    # A settled post-artifact region is flat; the decaying tail is not.
    if smoothed.size > 1:
        slope = np.gradient(smoothed) / scale
    else:
        slope = np.zeros_like(smoothed)
    finite_slope = np.where(np.isfinite(slope), np.abs(slope), np.inf)

    search_mask = (times >= float(search_window[0])) & (times <= float(search_window[1]))
    search_indices = np.flatnonzero(search_mask)
    if search_indices.size == 0:
        peak_idx = int(np.argmin(np.abs(times)))
    else:
        peak_idx = int(search_indices[int(np.nanargmax(finite_score[search_indices]))])

    onset_idx = int(np.argmin(np.abs(times)))
    onset_score = float(finite_score[onset_idx]) if np.isfinite(finite_score[onset_idx]) else np.nan
    peak_score = float(finite_score[peak_idx]) if np.isfinite(finite_score[peak_idx]) else np.nan
    anchor_start = max(float(min_anchor_time_s), float(times[peak_idx] + sample_dt))
    candidate_indices = np.flatnonzero((times >= anchor_start) & (times <= float(search_window[1])))

    if consecutive_samples is None:
        n_settle = max(int(round(float(settle_duration_s) / sample_dt)), 1) if sample_dt > 0 else 1
    else:
        n_settle = max(int(consecutive_samples), 1)

    below_amp = finite_score <= float(settle_z)
    settled_full = below_amp & (finite_slope <= float(settle_slope_z))

    anchor_idx: int | None = None
    method = ""
    settled_flag = False
    if candidate_indices.size:
        candidate_settled = settled_full[candidate_indices]
        for pos in range(0, len(candidate_indices) - n_settle + 1):
            if bool(np.all(candidate_settled[pos : pos + n_settle])):
                anchor_idx = int(candidate_indices[pos])
                method = "post_artifact_settled_sample"
                settled_flag = True
                break
        if anchor_idx is None:
            below_candidate = np.flatnonzero(below_amp[candidate_indices])
            if below_candidate.size:
                anchor_idx = int(candidate_indices[int(below_candidate[0])])
                method = "fallback_amplitude_return"
            else:
                anchor_idx = int(candidate_indices[int(np.argmin(finite_score[candidate_indices]))])
                method = "fallback_quietest_sample"
    if anchor_idx is None:
        anchor_idx = int(min(peak_idx + 1, len(times) - 1))
        method = "fallback_peak_neighbor"

    return {
        "anchor_time_s": float(times[anchor_idx]),
        "anchor_index": int(anchor_idx),
        "artifact_peak_time_s": float(times[peak_idx]),
        "artifact_peak_score_z": peak_score,
        "anchor_score_z": float(finite_score[anchor_idx]) if np.isfinite(finite_score[anchor_idx]) else np.nan,
        "anchor_slope_z": float(finite_slope[anchor_idx]) if np.isfinite(finite_slope[anchor_idx]) else np.nan,
        "onset_score_z": onset_score,
        "onset_is_artifact": bool(np.isfinite(onset_score) and onset_score >= float(onset_artifact_z)),
        "settled": bool(settled_flag),
        "method": method,
        "reference_channels": ",".join(ref_cols),
        "search_window_start_s": float(search_window[0]),
        "search_window_stop_s": float(search_window[1]),
        "min_anchor_time_s": float(min_anchor_time_s),
        "settle_z": float(settle_z),
        "settle_slope_z": float(settle_slope_z),
        "settle_duration_s": float(settle_duration_s),
        "n_settle_samples": int(n_settle),
    }


def detect_zero_time_artifact(
    epoch_df: pd.DataFrame,
    *,
    stim_ch: str | Iterable[str] | None = None,
    zero_time: float = 0.0,
    baseline_window: tuple[float, float] | None = None,
    artifact_z: float = 6.0,
) -> dict[str, object]:
    """Report whether a requested zeroing time falls on a stimulation artifact."""

    times = epoch_df.index.to_numpy(dtype=float)
    envelope, ref_cols = artifact_envelope(epoch_df, stim_ch=stim_ch)
    if times.size == 0 or envelope.size != times.size or envelope.size == 0:
        return {
            "zero_time_s": float(zero_time),
            "nearest_sample_s": np.nan,
            "artifact_score_z": np.nan,
            "zero_time_is_artifact": False,
            "reference_channels": "",
        }
    if baseline_window is not None:
        baseline_mask = (times >= float(baseline_window[0])) & (times <= float(baseline_window[1]))
    else:
        baseline_mask = times < 0
    center, scale = _robust_location_scale(envelope[baseline_mask] if baseline_mask.any() else envelope)
    score = (envelope - center) / scale
    zero_idx = int(np.argmin(np.abs(times - float(zero_time))))
    artifact_score = float(score[zero_idx]) if np.isfinite(score[zero_idx]) else np.nan
    return {
        "zero_time_s": float(zero_time),
        "nearest_sample_s": float(times[zero_idx]),
        "artifact_score_z": artifact_score,
        "zero_time_is_artifact": bool(np.isfinite(artifact_score) and artifact_score >= float(artifact_z)),
        "artifact_z_threshold": float(artifact_z),
        "reference_channels": ",".join(ref_cols),
    }
