"""Auditable waveform validation and review figures."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import uuid
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from .baseline import baseline_center_array
from .erp_detection import (
    PRIMARY_CRITERION,
    _feature_table_from_array,
    _standalone_comparator_summary,
)
from .inference import fdr_bh


COMMON_MODE_RECOVERY_DEFAULTS = {
    "common_mode_recovery_enabled": True,
    "common_mode_min_detector_channels": 20,
    "common_mode_min_early_fraction": 0.80,
    "common_mode_early_window_ms": 6.0,
    "common_mode_min_median_start_abs_uv": 25.0,
    "common_mode_min_median_drift_abs_uv": 12.0,
}


def common_mode_recovery_diagnostic(
    frame: pd.DataFrame,
    *,
    response_window: tuple[float, float] = (0.01, 0.35),
    thresholds: Mapping[str, Any] | None = None,
    consensus_column: str | None = None,
) -> dict[str, Any]:
    """Detect an acquisition-wide early step followed by shared recovery.

    A physiological early response may occur on one or several connected
    contacts. Amplifier recovery is different: many anatomically distributed
    detector-positive channels begin with a large offset, peak at nearly the
    same first post-blanking sample, and drift together toward baseline. The
    returned diagnostics make the acquisition-level quarantine auditable.
    """

    options = dict(COMMON_MODE_RECOVERY_DEFAULTS)
    options.update(dict(thresholds or {}))
    result = {
        "common_mode_recovery_artifact": False,
        "common_mode_detector_channels": 0,
        "common_mode_early_channels": 0,
        "common_mode_early_fraction": np.nan,
        "common_mode_median_start_abs_uv": np.nan,
        "common_mode_median_drift_abs_uv": np.nan,
        "common_mode_early_window_start_ms": (
            float(response_window[0]) * 1000.0
        ),
        "common_mode_early_window_stop_ms": (
            float(response_window[0]) * 1000.0
            + float(options["common_mode_early_window_ms"])
        ),
    }
    data = pd.DataFrame(frame).copy()
    if data.empty or "channel" not in data.columns:
        return result

    if consensus_column is None:
        consensus_column = next(
            (
                column
                for column in (
                    "strict_detector_consensus",
                    "consensus",
                )
                if column in data.columns
            ),
            None,
        )
    if consensus_column is None or consensus_column not in data.columns:
        return result

    data["channel"] = data["channel"].astype(str)
    consensus_by_channel = data.groupby("channel", sort=False)[
        consensus_column
    ].apply(lambda values: bool(_bool_series(values).any()))
    channels = data.drop_duplicates("channel", keep="first").copy()
    channels["_detector_consensus"] = channels["channel"].map(
        consensus_by_channel
    ).fillna(False)

    def feature(*names: str) -> pd.Series:
        for name in names:
            if name in channels.columns:
                return pd.to_numeric(channels[name], errors="coerce")
        return pd.Series(np.nan, index=channels.index, dtype=float)

    detector_rows = channels[channels["_detector_consensus"]].copy()
    detector_rows["_latency_ms"] = feature(
        "peak_latency_ms_recomputed",
        "peak_latency_ms",
    ).reindex(detector_rows.index)
    detector_rows["_response_start_uv"] = feature(
        "response_start_uv_recomputed",
        "response_start_uv",
    ).reindex(detector_rows.index)
    detector_rows["_response_drift_uv"] = feature(
        "response_drift_uv_recomputed",
        "response_drift_uv",
    ).reindex(detector_rows.index)

    n_detector = int(len(detector_rows))
    start_ms = float(result["common_mode_early_window_start_ms"])
    stop_ms = float(result["common_mode_early_window_stop_ms"])
    early = detector_rows["_latency_ms"].between(
        start_ms,
        stop_ms,
        inclusive="both",
    )
    n_early = int(early.sum())
    early_fraction = n_early / n_detector if n_detector else np.nan

    def median_absolute(values: pd.Series) -> float:
        finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(
            dtype=float
        )
        return float(np.median(np.abs(finite))) if len(finite) else np.nan

    median_start = median_absolute(detector_rows["_response_start_uv"])
    median_drift = median_absolute(detector_rows["_response_drift_uv"])
    enabled = bool(options["common_mode_recovery_enabled"])
    flagged = bool(
        enabled
        and n_detector
        >= int(options["common_mode_min_detector_channels"])
        and np.isfinite(early_fraction)
        and early_fraction
        >= float(options["common_mode_min_early_fraction"])
        and np.isfinite(median_start)
        and median_start
        >= float(options["common_mode_min_median_start_abs_uv"])
        and np.isfinite(median_drift)
        and median_drift
        >= float(options["common_mode_min_median_drift_abs_uv"])
    )
    result.update(
        {
            "common_mode_recovery_artifact": flagged,
            "common_mode_detector_channels": n_detector,
            "common_mode_early_channels": n_early,
            "common_mode_early_fraction": early_fraction,
            "common_mode_median_start_abs_uv": median_start,
            "common_mode_median_drift_abs_uv": median_drift,
        }
    )
    return result


@dataclass
class WaveformAudit:
    """Waveform metrics, detector concordance, and trial-level QC context."""

    epochs: Any
    summary: pd.DataFrame
    artifact_responses: pd.DataFrame
    response_window: tuple[float, float]
    baseline_window: tuple[float, float]
    _array: np.ndarray = field(repr=False)
    _clean_array: np.ndarray = field(repr=False)
    _times: np.ndarray = field(repr=False)
    _channels: list[str] = field(repr=False)

    def plot(
        self,
        channels: Iterable[str] | None = None,
        *,
        max_channels: int = 12,
        full_scale: bool = False,
        title: str | None = None,
    ):
        """Plot a compact review montage and return the Matplotlib figure."""

        selected = self._selected_channels(channels, max_channels=max_channels)
        n_columns = min(3, max(len(selected), 1))
        n_rows = max(int(math.ceil(len(selected) / n_columns)), 1)
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(4.5 * n_columns, 2.6 * n_rows),
            squeeze=False,
        )
        for axis, channel in zip(axes.flat, selected):
            self._plot_channel(axis, channel, full_scale=full_scale)
        for axis in axes.flat[len(selected) :]:
            axis.set_axis_off()
        if title:
            fig.suptitle(str(title), y=1.01, fontsize=12)
        fig.tight_layout()
        return fig

    def to_pdf(
        self,
        path: str | Path,
        *,
        channels: Iterable[str] | None = None,
        channels_per_page: int = 24,
        full_scale: bool = False,
        include_flagged_full_scale: bool = True,
        max_flagged_full_scale_channels: int | None = 24,
        title: str | None = None,
    ) -> Path:
        """Write an atomic, multipage review atlas for every selected channel.

        The main atlas uses robust limits so physiological morphology remains
        visible in the presence of a large rejected trial. By default, channels
        requiring review are then repeated at full scale so clipping, saturation,
        drift, and sharp transients can be inspected directly.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = self._selected_channels(channels, max_channels=None)
        per_page = max(int(channels_per_page), 1)
        temporary = path.with_name(
            f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{path.suffix}"
        )
        try:
            with PdfPages(temporary) as pdf:
                for page_start in range(0, len(selected), per_page):
                    page_channels = selected[
                        page_start : page_start + per_page
                    ]
                    n_columns = 4
                    n_rows = max(
                        int(math.ceil(len(page_channels) / n_columns)),
                        1,
                    )
                    fig, axes = plt.subplots(
                        n_rows,
                        n_columns,
                        figsize=(16, 2.45 * n_rows),
                        squeeze=False,
                    )
                    for axis, channel in zip(axes.flat, page_channels):
                        self._plot_channel(
                            axis,
                            channel,
                            full_scale=full_scale,
                        )
                    for axis in axes.flat[len(page_channels) :]:
                        axis.set_axis_off()
                    page_number = page_start // per_page + 1
                    page_total = int(math.ceil(len(selected) / per_page))
                    fig.suptitle(
                        (
                            f"{title or 'ERPy waveform review'} | "
                            f"page {page_number}/{page_total}"
                        ),
                        y=0.998,
                        fontsize=12,
                    )
                    fig.tight_layout(rect=(0, 0, 1, 0.98))
                    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
                    plt.close(fig)
                if include_flagged_full_scale and not full_scale:
                    summary = self.summary.set_index("channel")
                    full_scale_reasons = {
                        "recomputed_metric_mismatch",
                        "response_window_boundary_peak",
                        "bad_response_burden",
                        "near_bad_response_burden_cutoff",
                        "hard_artifact_burden",
                        "near_hard_artifact_cutoff",
                        "insufficient_clean_responses",
                        "near_minimum_clean_responses",
                        "artifact_feature_near_threshold",
                        "large_clean_mean_response",
                        "common_mode_recovery_artifact",
                    }
                    flagged = [
                        channel
                        for channel in selected
                        if full_scale_reasons.intersection(
                            str(
                                summary.loc[channel].get(
                                    "review_reasons",
                                    "",
                                )
                            ).split(";")
                        )
                    ]
                    if len(flagged) < min(
                        int(max_flagged_full_scale_channels or 0),
                        len(selected),
                    ):
                        remaining = [
                            channel
                            for channel in selected
                            if channel not in set(flagged)
                            and int(
                                summary.loc[channel].get(
                                    "review_priority",
                                    0,
                                )
                            )
                            >= 2
                        ]
                        flagged.extend(remaining)
                    if max_flagged_full_scale_channels is not None:
                        flagged = flagged[
                            : max(
                                int(max_flagged_full_scale_channels),
                                0,
                            )
                        ]
                    for page_start in range(0, len(flagged), per_page):
                        page_channels = flagged[
                            page_start : page_start + per_page
                        ]
                        n_columns = 4
                        n_rows = max(
                            int(math.ceil(len(page_channels) / n_columns)),
                            1,
                        )
                        fig, axes = plt.subplots(
                            n_rows,
                            n_columns,
                            figsize=(16, 2.45 * n_rows),
                            squeeze=False,
                        )
                        for axis, channel in zip(axes.flat, page_channels):
                            self._plot_channel(
                                axis,
                                channel,
                                full_scale=True,
                            )
                        for axis in axes.flat[len(page_channels) :]:
                            axis.set_axis_off()
                        page_number = page_start // per_page + 1
                        page_total = int(
                            math.ceil(len(flagged) / per_page)
                        )
                        fig.suptitle(
                            (
                                f"{title or 'ERPy waveform review'} | "
                                "flagged channels, full scale | "
                                f"page {page_number}/{page_total}"
                            ),
                            y=0.998,
                            fontsize=12,
                        )
                        fig.tight_layout(rect=(0, 0, 1, 0.98))
                        pdf.savefig(
                            fig,
                            bbox_inches="tight",
                            facecolor="white",
                        )
                        plt.close(fig)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def mean_waveforms(
        self,
        *,
        start_s: float = -0.10,
        stop_s: float = 0.45,
        step_ms: float = 2.0,
    ) -> pd.DataFrame:
        """Return downsampled clean means for cohort-scale waveform review."""

        if float(stop_s) <= float(start_s):
            raise ValueError("stop_s must be greater than start_s")
        if float(step_ms) <= 0:
            raise ValueError("step_ms must be positive")
        selected = np.flatnonzero(
            (self._times >= float(start_s))
            & (self._times <= float(stop_s))
        )
        if not len(selected):
            raise ValueError("Requested review window does not overlap epochs")
        stride = max(
            int(round(float(step_ms) * float(self.epochs.sfreq) / 1000.0)),
            1,
        )
        selected = selected[::stride]
        clean_mean = np.nanmean(self._clean_array, axis=0)
        output = pd.DataFrame(
            clean_mean[selected, :],
            index=pd.Index(
                self._times[selected] * 1000.0,
                name="time_ms",
            ),
            columns=self._channels,
        )
        return output

    def _selected_channels(
        self,
        channels: Iterable[str] | None,
        *,
        max_channels: int | None,
    ) -> list[str]:
        if channels is None:
            ordered = (
                self.summary.sort_values(
                    [
                        "review_priority",
                        "feature_mismatch",
                        "strict_consensus",
                        "peak_amplitude_uv_recomputed",
                    ],
                    ascending=[False, False, False, False],
                    kind="stable",
                )["channel"]
                .astype(str)
                .tolist()
            )
        else:
            requested = [str(channel) for channel in channels]
            available = set(map(str, self.epochs.channels))
            ordered = [channel for channel in requested if channel in available]
        if max_channels is not None:
            ordered = ordered[: max(int(max_channels), 0)]
        return ordered

    def _plot_channel(
        self,
        axis,
        channel: str,
        *,
        full_scale: bool,
    ) -> None:
        times = self._times
        channel_index = self._channels.index(str(channel))
        traces = self._array[:, :, channel_index]
        bad_epochs = _bad_epoch_indices(
            self.artifact_responses,
            str(channel),
        )
        bad_mask = np.zeros(traces.shape[0], dtype=bool)
        bad_mask[
            [index for index in bad_epochs if 0 <= index < len(bad_mask)]
        ] = True
        view = (times >= -0.10) & (times <= 0.45)
        x_values = times[view] * 1000.0
        trace_view = traces[:, view]
        for selected_mask, color, alpha in (
            (~bad_mask, "#8a9099", 0.13),
            (bad_mask, "#b8423f", 0.28),
        ):
            selected_traces = trace_view[selected_mask]
            if not len(selected_traces):
                continue
            x_matrix = np.broadcast_to(
                x_values,
                selected_traces.shape,
            )
            segments = np.stack(
                [x_matrix, selected_traces],
                axis=2,
            )
            axis.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    alpha=alpha,
                    linewidths=0.7,
                    rasterized=True,
                    zorder=1,
                )
            )
        clean = self._clean_array[:, :, channel_index]
        mean = np.nanmean(clean, axis=0)
        sem = _nan_sem(clean)
        row = self.summary.set_index("channel").loc[str(channel)]
        review_reasons = set(
            filter(
                None,
                str(row.get("review_reasons", "")).split(";"),
            )
        )
        analysis_eligible_value = row.get("analysis_eligible", True)
        analysis_eligible = bool(analysis_eligible_value) if pd.notna(
            analysis_eligible_value
        ) else False
        mean_color = "#20262e" if analysis_eligible else "#b8423f"
        interval_color = "#2f6f9f" if analysis_eligible else "#b8423f"
        axis.fill_between(
            times[view] * 1000.0,
            mean[view] - sem[view],
            mean[view] + sem[view],
            color=interval_color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            times[view] * 1000.0,
            mean[view],
            color=mean_color,
            linewidth=1.5,
            zorder=3,
        )
        axis.axvline(0.0, color="#b8423f", linewidth=0.8)
        axis.axvspan(
            self.response_window[0] * 1000.0,
            self.response_window[1] * 1000.0,
            color="#e7eaee",
            alpha=0.35,
            linewidth=0,
            zorder=0,
        )
        peak_time = float(row["peak_latency_ms_recomputed"])
        peak_signed = float(row["peak_signed_uv_recomputed"])
        if np.isfinite(peak_time) and np.isfinite(peak_signed):
            axis.scatter(
                [peak_time],
                [peak_signed],
                s=19,
                color="#d97924",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        if not full_scale:
            finite_clean = clean[:, view][np.isfinite(clean[:, view])]
            finite_mean = mean[view][np.isfinite(mean[view])]
            if len(finite_clean):
                robust = float(np.nanquantile(np.abs(finite_clean), 0.99))
                mean_peak = (
                    float(np.nanmax(np.abs(finite_mean)))
                    if len(finite_mean)
                    else 0.0
                )
                limit = max(robust * 1.35, mean_peak * 1.20, 1.0)
                axis.set_ylim(-limit, limit)
                full_peak = float(np.nanmax(np.abs(traces[:, view])))
                if np.isfinite(full_peak) and full_peak > limit:
                    axis.text(
                        0.99,
                        0.02,
                        f"flagged max {full_peak:.0f} uV",
                        transform=axis.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=6.5,
                        color="#b8423f",
                    )
        else:
            finite_full = trace_view[np.isfinite(trace_view)]
            if len(finite_full):
                lower = float(np.nanmin(finite_full))
                upper = float(np.nanmax(finite_full))
                span = max(upper - lower, 1.0)
                axis.set_ylim(
                    lower - 0.04 * span,
                    upper + 0.04 * span,
                )
        pattern = str(row.get("detector_pattern", "") or "none")
        strict = "strict+" if bool(row.get("strict_consensus", False)) else "strict-"
        primary = (
            "CRP-energy+"
            if bool(row.get("primary_qc_pass", False))
            else "CRP-energy-"
        )
        eligibility = "eligible" if analysis_eligible else "excluded"
        priority = int(row.get("review_priority", 0))
        clean_count = int(row.get("n_clean_response", 0) or 0)
        axis.set_title(
            (
                f"{channel} | {primary} | {strict} | {eligibility} | review {priority} | "
                f"clean n={clean_count}/{len(traces)}\n{pattern}"
            ),
            fontsize=7.4,
        )
        axis.set_xlim(-100.0, 450.0)
        axis.tick_params(labelsize=7, length=2)
        axis.set_xlabel("Time (ms)", fontsize=7)
        axis.set_ylabel("uV", fontsize=7)


def audit_waveforms(
    epochs: Any,
    detections: pd.DataFrame,
    artifact_responses: pd.DataFrame,
    *,
    response_window: tuple[float, float] = (0.01, 0.35),
    baseline_window: tuple[float, float] = (-0.5, -0.03),
    min_consensus: int = 2,
    artifact_thresholds: Mapping[str, float] | None = None,
    max_bad_response_fraction: float = 0.25,
    max_hard_artifact_fraction: float = 0.10,
    min_clean_responses: int = 10,
    fdr_alpha: float = 0.05,
) -> WaveformAudit:
    """Recompute waveform metrics and classify responses needing review.

    Default component decisions are ``primary_reproducibility_pass`` and
    ``primary_energy_pass``. ``primary_detector_pass`` is the joint BH decision
    (``primary_significant`` in the detection table); ``primary_qc_pass`` also
    requires contact-QC eligibility.

    Standalone CRP/Kundu calls use nullable ``comparator_crp_pass`` and
    ``comparator_kundu_pass`` with explicit availability and reason fields.
    ``primary_shape_pass`` and ``primary_magnitude_pass`` are deprecated aliases
    for CRP and Kundu respectively, including their missing states; they are
    not the default reproducibility/energy components. Missing comparator calls
    indicate not run or unavailable, not a negative result. Matching
    ``comparator_*_available`` and ``comparator_*_availability_reason`` columns
    retain the evidence status and distinguish ``not_run``.
    ``comparator_crp_kundu_disagreement`` is reported only when both comparator
    calls have applicable evidence; this diagnostic does not change eligibility.
    """

    detections = pd.DataFrame(detections).copy()
    artifacts = pd.DataFrame(artifact_responses).copy()
    arr, times, channels = epochs.as_array()
    clean = np.asarray(arr, dtype=float).copy()
    for channel_index, channel in enumerate(channels):
        for epoch_index in _bad_epoch_indices(artifacts, str(channel)):
            if 0 <= epoch_index < clean.shape[0]:
                clean[epoch_index, :, channel_index] = np.nan
    referenced = baseline_center_array(
        arr,
        times,
        tuple(baseline_window),
    )
    clean_referenced = baseline_center_array(
        clean,
        times,
        tuple(baseline_window),
    )

    features = _feature_table_from_array(
        clean_referenced,
        times,
        list(map(str, channels)),
        sfreq=float(epochs.sfreq),
        baseline_window=tuple(baseline_window),
        response_window=tuple(response_window),
        response_boundary_guard_s=0.002,
        baseline_centered=True,
    ).rename(
        columns={
            column: f"{column}_recomputed"
            for column in (
                "peak_amplitude_uv",
                "peak_latency_ms",
                "n1_latency_ms",
                "rms_uv",
                "auc_uv_ms",
                "max_descent_ms",
                "baseline_zscore",
                "peak_at_response_boundary",
                "peak_boundary_side",
                "peak_boundary_distance_ms",
                "response_start_uv",
                "response_end_uv",
                "response_drift_uv",
                "response_linear_slope_uv_s",
                "n_trials",
            )
        }
    )
    response_mask = (
        (times >= float(response_window[0]))
        & (times <= float(response_window[1]))
    )
    clean_mean = np.nanmean(clean_referenced, axis=0)
    signed_peaks = []
    for channel_index in range(len(channels)):
        response = clean_mean[response_mask, channel_index]
        if np.isfinite(response).any():
            peak_index = int(np.nanargmax(np.abs(response)))
            signed_peaks.append(float(response[peak_index]))
        else:
            signed_peaks.append(np.nan)
    features["peak_signed_uv_recomputed"] = signed_peaks

    detector_first = _collapse_detection_features(detections)
    summary = features.merge(detector_first, on="channel", how="left")
    method_matrix = _method_significance_matrix(detections, channels)
    crp_q = _crp_adjusted_values(
        detections,
        channels,
        alpha=float(fdr_alpha),
    )
    summary["crp_q_value"] = summary["channel"].map(crp_q["q_value"])
    summary["crp_fdr_significant"] = (
        summary["channel"]
        .map(crp_q["significant"])
        .fillna(False)
        .astype(bool)
    )

    strict_matrix = method_matrix.drop(
        columns=["crp_energy"],
        errors="ignore",
    ).copy()
    if "crp_significance" in strict_matrix.columns:
        strict_matrix["crp_significance"] = (
            strict_matrix.index.to_series()
            .map(crp_q["significant"])
            .fillna(False)
            .astype(bool)
        )
    strict_counts = strict_matrix.sum(axis=1)
    summary["strict_n_methods_significant"] = (
        summary["channel"].map(strict_counts).fillna(0).astype(int)
    )
    summary["strict_detector_consensus"] = (
        summary["strict_n_methods_significant"] >= int(min_consensus)
    )
    kundu_detector_pass = (
        method_matrix.get(
            "kundu_rolston",
            pd.Series(False, index=method_matrix.index),
        )
        .reindex(list(map(str, channels)))
        .fillna(False)
        .astype(bool)
    )
    primary_detector_pass = (
        pd.Series(
            {
                str(channel): bool(crp_q["significant"].get(str(channel), False))
                for channel in channels
            }
        )
        & kundu_detector_pass
    )
    comparators = _standalone_comparator_summary(
        detections, channels, crp_adjustment=crp_q,
    )
    for column in comparators:
        summary[column] = summary["channel"].map(comparators[column])
    # Deprecated compatibility aliases refer to standalone comparators only.
    summary["primary_shape_pass"] = summary["comparator_crp_pass"]
    summary["primary_magnitude_pass"] = summary["comparator_kundu_pass"]
    summary["shape_magnitude_detector_pass"] = (
        summary["channel"]
        .map(primary_detector_pass)
        .fillna(False)
        .astype(bool)
    )
    summary["shape_magnitude_criterion"] = (
        "CRP BH-FDR q<0.05 and published Kundu 15-ms/3x/30-uV rule"
    )
    crp_energy_rows = pd.DataFrame()
    if not detections.empty and {"method", "channel"}.issubset(detections.columns):
        crp_energy_rows = detections[
            detections["method"].astype(str).eq("crp_energy")
        ].drop_duplicates("channel").copy()
    if not crp_energy_rows.empty:
        crp_energy_rows["channel"] = crp_energy_rows["channel"].astype(str)
        crp_energy_rows = crp_energy_rows.set_index("channel")
    primary_detector = _indexed_bool_column(
        crp_energy_rows,
        "significant",
        channels,
    )
    primary_reproducibility = _indexed_bool_column(
        crp_energy_rows,
        "crp_significant",
        channels,
    )
    primary_energy = _indexed_bool_column(
        crp_energy_rows,
        "energy_significant",
        channels,
    )
    summary["primary_detector_pass"] = (
        summary["channel"].map(primary_detector).fillna(False).astype(bool)
    )
    summary["primary_reproducibility_pass"] = (
        summary["channel"]
        .map(primary_reproducibility)
        .fillna(False)
        .astype(bool)
    )
    summary["primary_energy_pass"] = (
        summary["channel"].map(primary_energy).fillna(False).astype(bool)
    )
    for source, target in (
        ("p_joint", "primary_joint_p_value"),
        ("q_joint", "primary_joint_q_value"),
        ("classification", "primary_classification"),
    ):
        values = (
            crp_energy_rows[source]
            if not crp_energy_rows.empty and source in crp_energy_rows.columns
            else pd.Series(dtype=float if source != "classification" else str)
        )
        summary[target] = summary["channel"].map(values)
    summary["primary_criterion"] = PRIMARY_CRITERION
    summary["detector_pattern"] = summary["channel"].map(
        {
            channel: "+".join(
                method
                for method, significant in row.items()
                if bool(significant)
            )
            or "none"
            for channel, row in method_matrix.iterrows()
        }
    )
    summary["strict_detector_pattern"] = summary["channel"].map(
        {
            channel: "+".join(
                method
                for method, significant in row.items()
                if bool(significant)
            )
            or "none"
            for channel, row in strict_matrix.iterrows()
        }
    )
    for method in method_matrix.columns:
        summary[f"detector_{method}"] = (
            summary["channel"]
            .map(method_matrix[method])
            .fillna(False)
            .astype(bool)
        )
        strict_method = strict_matrix.get(
            method,
            pd.Series(False, index=strict_matrix.index, dtype=bool),
        )
        summary[f"strict_detector_{method}"] = (
            summary["channel"]
            .map(strict_method)
            .fillna(False)
            .astype(bool)
        )

    artifact_summary = _artifact_channel_summary(
        artifacts,
        channels=channels,
        n_epochs=int(arr.shape[0]),
    )
    summary = summary.merge(
        artifact_summary,
        on="channel",
        how="left",
    )
    for column in (
        "n_epochs_artifact",
        "n_bad_response",
        "n_hard_artifact_response",
        "n_clean_response",
    ):
        summary[column] = (
            pd.to_numeric(summary[column], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    for column in ("bad_response_fraction", "hard_artifact_fraction"):
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        ).fillna(0.0)

    recomputed_amplitude = pd.to_numeric(
        summary["peak_amplitude_uv_recomputed"], errors="coerce"
    )
    recomputed_latency = pd.to_numeric(
        summary["peak_latency_ms_recomputed"], errors="coerce"
    )
    amplitude_tolerance = np.maximum(0.05, np.abs(recomputed_amplitude) * 5e-5)
    stored_amplitude = pd.to_numeric(
        summary.get(
            "peak_amplitude_uv",
            pd.Series(np.nan, index=summary.index),
        ),
        errors="coerce",
    )
    stored_latency = pd.to_numeric(
        summary.get(
            "peak_latency_ms",
            pd.Series(np.nan, index=summary.index),
        ),
        errors="coerce",
    )
    summary["peak_amplitude_difference_uv"] = (
        stored_amplitude - recomputed_amplitude
    )
    summary["peak_latency_difference_ms"] = (
        stored_latency - recomputed_latency
    )
    # Comparisons with NaN are False, and an infinite amplitude also produces
    # an infinite tolerance. Neither establishes agreement between features.
    features_finite = np.isfinite(
        pd.concat(
            [stored_amplitude, stored_latency, recomputed_amplitude, recomputed_latency],
            axis=1,
        ).to_numpy(dtype=float, na_value=np.nan)
    ).all(axis=1)
    summary["feature_mismatch"] = (
        ~features_finite
        | summary["peak_amplitude_difference_uv"].abs().gt(
            amplitude_tolerance
        )
        | summary["peak_latency_difference_ms"].abs().gt(
            1000.0 / float(epochs.sfreq) + 1e-6
        )
    ).fillna(True)
    artifact_thresholds = dict(artifact_thresholds or {})
    common_mode = common_mode_recovery_diagnostic(
        summary,
        response_window=tuple(response_window),
        thresholds=artifact_thresholds,
        consensus_column="strict_detector_consensus",
    )
    for column, value in common_mode.items():
        summary[column] = value
    common_mode_recovery = _bool_series(
        summary["common_mode_recovery_artifact"]
    )
    boundary_peak = _bool_series(
        summary.get(
            "peak_at_response_boundary_recomputed",
            pd.Series(False, index=summary.index),
        )
    )
    summary["artifact_eligible"] = (
        summary["bad_response_fraction"].lt(
            float(max_bad_response_fraction)
        )
        & summary["hard_artifact_fraction"].lt(
            float(max_hard_artifact_fraction)
        )
        & summary["n_clean_response"].ge(int(min_clean_responses))
        & ~boundary_peak
        & ~common_mode_recovery
    )
    summary["analysis_eligible"] = (
        summary["artifact_eligible"]
        & ~_bool_series(summary["feature_mismatch"])
    )
    summary["strict_consensus"] = (
        _bool_series(summary["strict_detector_consensus"])
        & summary["analysis_eligible"]
    )
    summary["shape_magnitude_qc_pass"] = (
        _bool_series(summary["shape_magnitude_detector_pass"])
        & summary["analysis_eligible"]
    )
    summary["primary_qc_pass"] = (
        _bool_series(summary["primary_detector_pass"])
        & summary["analysis_eligible"]
    )
    summary["crp_energy_qc_pass"] = summary["primary_qc_pass"]

    z_threshold = float(artifact_thresholds.get("zscore_threshold", 6.0))
    late_threshold = float(
        artifact_thresholds.get(
            "late_high_frequency_zscore_threshold",
            6.0,
        )
    )
    trial_borderline = _trial_borderline_channels(
        artifacts,
        channels=channels,
        z_threshold=z_threshold,
        late_threshold=late_threshold,
        thresholds=artifact_thresholds,
    )
    detector_borderline = _detector_borderline_channels(detections)
    review_reasons = []
    priorities = []
    for _, row in summary.iterrows():
        channel = str(row["channel"])
        reasons = []
        priority = 0
        if bool(row["feature_mismatch"]):
            reasons.append("recomputed_metric_mismatch")
            priority = max(priority, 3)
        if bool(row.get("peak_at_response_boundary_recomputed", False)):
            reasons.append("response_window_boundary_peak")
            priority = max(priority, 3)
        if bool(row.get("common_mode_recovery_artifact", False)):
            reasons.append("common_mode_recovery_artifact")
            priority = max(priority, 3)
        if row["bad_response_fraction"] >= float(max_bad_response_fraction):
            reasons.append("bad_response_burden")
            priority = max(priority, 3)
        elif row["bad_response_fraction"] >= max(
            float(max_bad_response_fraction) - 0.05,
            0.0,
        ):
            reasons.append("near_bad_response_burden_cutoff")
            priority = max(priority, 2)
        if row["hard_artifact_fraction"] >= float(
            max_hard_artifact_fraction
        ):
            reasons.append("hard_artifact_burden")
            priority = max(priority, 3)
        elif row["hard_artifact_fraction"] >= max(
            float(max_hard_artifact_fraction) - 0.03,
            0.0,
        ):
            reasons.append("near_hard_artifact_cutoff")
            priority = max(priority, 2)
        if row["n_clean_response"] < int(min_clean_responses):
            reasons.append("insufficient_clean_responses")
            priority = max(priority, 3)
        elif row["n_clean_response"] <= int(min_clean_responses) + 2:
            reasons.append("near_minimum_clean_responses")
            priority = max(priority, 2)
        if channel in trial_borderline:
            reasons.append("artifact_feature_near_threshold")
            priority = max(priority, 2)
        if channel in detector_borderline:
            reasons.append("detector_score_near_threshold")
            priority = max(priority, 2)
        if bool(row.get("strict_detector_consensus", False)) and not bool(
            row.get("analysis_eligible", False)
        ):
            reasons.append("detector_consensus_excluded_by_qc")
            priority = max(priority, 3)
        if bool(row.get("shape_magnitude_detector_pass", False)) and not bool(
            row.get("analysis_eligible", False)
        ):
            reasons.append("shape_magnitude_response_excluded_by_qc")
            priority = max(priority, 3)
        if bool(row.get("primary_detector_pass", False)) and not bool(
            row.get("analysis_eligible", False)
        ):
            reasons.append("crp_energy_response_excluded_by_qc")
            priority = max(priority, 3)
        if bool(row.get("primary_reproducibility_pass", False)) != bool(
            row.get("primary_energy_pass", False)
        ):
            reasons.append("primary_reproducibility_energy_disagreement")
            priority = max(priority, 2)
        if (
            bool(row["comparator_crp_available"])
            and bool(row["comparator_kundu_available"])
            and bool(row["comparator_crp_pass"]) != bool(row["comparator_kundu_pass"])
        ):
            reasons.append("comparator_crp_kundu_disagreement")
            priority = max(priority, 2)
        method_count = int(row.get("n_methods_significant", 0) or 0)
        if 0 < method_count < len(method_matrix.columns):
            reasons.append("detector_disagreement")
            priority = max(priority, 1)
        if method_count in {max(int(min_consensus) - 1, 0), int(min_consensus)}:
            reasons.append("consensus_boundary")
            priority = max(priority, 2)
        peak_amplitude = float(
            row.get("peak_amplitude_uv_recomputed", np.nan)
        )
        if np.isfinite(peak_amplitude) and peak_amplitude >= 1_000.0:
            reasons.append("large_clean_mean_response")
            priority = max(priority, 2)
        review_reasons.append(";".join(dict.fromkeys(reasons)))
        priorities.append(priority)
    summary["review_reasons"] = review_reasons
    summary["review_priority"] = priorities
    summary["borderline"] = summary["review_priority"].ge(2)
    return WaveformAudit(
        epochs=epochs,
        summary=summary,
        artifact_responses=artifacts,
        response_window=tuple(response_window),
        baseline_window=tuple(baseline_window),
        _array=referenced,
        _clean_array=clean_referenced,
        _times=np.asarray(times, dtype=float),
        _channels=list(map(str, channels)),
    )


def _collapse_detection_features(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame({"channel": pd.Series(dtype=str)})
    data = detections.copy()
    data["channel"] = data["channel"].astype(str)
    data = data.sort_values(["channel", "method"], kind="stable")
    first = data.drop_duplicates("channel", keep="first").copy()
    keep = [
        column
        for column in (
            "channel",
            "peak_amplitude_uv",
            "peak_latency_ms",
            "n1_latency_ms",
            "rms_uv",
            "auc_uv_ms",
            "max_descent_ms",
            "baseline_zscore",
            "n_trials",
            "n_methods_significant",
            "consensus",
            "primary_criterion",
            "primary_reproducibility_pass",
            "primary_energy_pass",
            "primary_joint_p_value",
            "primary_joint_q_value",
            "primary_classification",
            "primary_significant",
            "primary_shape_pass",
            "primary_magnitude_pass",
            "shape_magnitude_significant",
            "qc_pass",
            "qc_exclusion_reason",
            "peak_at_response_boundary",
            "peak_boundary_side",
            "peak_boundary_distance_ms",
        )
        if column in first.columns
    ]
    output = first[keep].copy()
    if "n_methods_significant" not in output:
        counts = (
            data.assign(_significant=_bool_series(data["significant"]))
            .groupby("channel")["_significant"]
            .sum()
        )
        output["n_methods_significant"] = output["channel"].map(counts)
    return output


def _method_significance_matrix(
    detections: pd.DataFrame,
    channels: Iterable[str],
) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(index=list(map(str, channels)))
    data = detections[["channel", "method", "significant"]].copy()
    data["channel"] = data["channel"].astype(str)
    data["method"] = data["method"].astype(str)
    data["significant"] = _bool_series(data["significant"])
    matrix = data.pivot_table(
        index="channel",
        columns="method",
        values="significant",
        aggfunc="max",
        fill_value=False,
    ).astype(bool)
    return matrix.reindex(list(map(str, channels)), fill_value=False)


def _crp_adjusted_values(
    detections: pd.DataFrame,
    channels: Iterable[str],
    *,
    alpha: float,
) -> dict[str, dict[str, float | bool]]:
    crp = detections[
        detections.get(
            "method",
            pd.Series("", index=detections.index),
        )
        .astype(str)
        .eq("crp_significance")
    ].copy()
    output = {
        "q_value": {str(channel): np.nan for channel in channels},
        "significant": {str(channel): False for channel in channels},
    }
    if crp.empty:
        return output
    adjusted, rejected = fdr_bh(
        pd.to_numeric(crp["p_value"], errors="coerce"),
        alpha=float(alpha),
    )
    for channel, q_value, significant in zip(
        crp["channel"].astype(str),
        adjusted,
        rejected,
    ):
        output["q_value"][channel] = float(q_value)
        output["significant"][channel] = bool(significant)
    return output


def _artifact_channel_summary(
    artifacts: pd.DataFrame,
    *,
    channels: Iterable[str],
    n_epochs: int | None = None,
) -> pd.DataFrame:
    columns = [
        "channel",
        "n_epochs_artifact",
        "n_bad_response",
        "n_hard_artifact_response",
        "n_clean_response",
        "bad_response_fraction",
        "hard_artifact_fraction",
        "artifact_reasons",
    ]
    if artifacts.empty:
        total = max(int(n_epochs or 0), 0)
        return pd.DataFrame(
            [
                {
                    "channel": str(channel),
                    "n_epochs_artifact": total,
                    "n_bad_response": 0,
                    "n_hard_artifact_response": 0,
                    "n_clean_response": total,
                    "bad_response_fraction": 0.0,
                    "hard_artifact_fraction": 0.0,
                    "artifact_reasons": "",
                }
                for channel in channels
            ],
            columns=columns,
        )
    data = artifacts.copy()
    data["channel"] = data["channel"].astype(str)
    data["_bad"] = _bool_series(data["bad_response"])
    data["_hard"] = _bool_series(
        data.get(
            "hard_artifact_response",
            pd.Series(False, index=data.index),
        )
    )

    def reasons(values: pd.Series) -> str:
        result = set()
        for value in values.fillna("").astype(str):
            result.update(part for part in value.split(";") if part)
        return ";".join(sorted(result))

    summary = (
        data.groupby("channel", dropna=False)
        .agg(
            n_epochs_artifact=("epoch", "nunique"),
            n_bad_response=("_bad", "sum"),
            n_hard_artifact_response=("_hard", "sum"),
            artifact_reasons=("reason", reasons),
        )
        .reset_index()
    )
    summary["n_clean_response"] = (
        summary["n_epochs_artifact"] - summary["n_bad_response"]
    )
    summary["bad_response_fraction"] = (
        summary["n_bad_response"]
        / summary["n_epochs_artifact"].replace(0, np.nan)
    )
    summary["hard_artifact_fraction"] = (
        summary["n_hard_artifact_response"]
        / summary["n_epochs_artifact"].replace(0, np.nan)
    )
    return summary[columns]


def _trial_borderline_channels(
    artifacts: pd.DataFrame,
    *,
    channels: Iterable[str],
    z_threshold: float,
    late_threshold: float,
    thresholds: Mapping[str, float] | None = None,
) -> set[str]:
    if artifacts.empty:
        return set()
    near = pd.Series(False, index=artifacts.index)
    for column, threshold in (
        ("peak_abs_z", z_threshold),
        ("ptp_z", z_threshold),
        ("max_gradient_z", z_threshold),
        ("late_high_frequency_ratio_z", late_threshold),
    ):
        if column not in artifacts:
            continue
        values = pd.to_numeric(artifacts[column], errors="coerce")
        near |= values.between(
            0.80 * float(threshold),
            1.20 * float(threshold),
            inclusive="both",
        )
    thresholds = dict(thresholds or {})
    for column, threshold_name in (
        ("saturation_fraction", "saturation_fraction"),
        ("plateau_fraction", "plateau_fraction"),
        ("plateau_run_s", "plateau_min_run_s"),
        ("rail_fraction", "rail_fraction"),
    ):
        if column not in artifacts or threshold_name not in thresholds:
            continue
        values = pd.to_numeric(artifacts[column], errors="coerce")
        threshold = float(thresholds[threshold_name])
        column_near = values.between(
            0.80 * threshold,
            1.20 * threshold,
            inclusive="both",
        )
        if column == "rail_fraction":
            rail_limits_known = _bool_series(
                artifacts.get(
                    "rail_limits_known",
                    pd.Series(False, index=artifacts.index),
                )
            )
            column_near &= rail_limits_known
        near |= column_near
    return set(
        artifacts.loc[near, "channel"].dropna().astype(str)
    ).intersection(set(map(str, channels)))


def _detector_borderline_channels(detections: pd.DataFrame) -> set[str]:
    if detections.empty:
        return set()
    data = detections.copy()
    score = pd.to_numeric(data.get("score"), errors="coerce")
    threshold = pd.to_numeric(data.get("threshold"), errors="coerce")
    method = data.get(
        "method",
        pd.Series("", index=data.index),
    ).astype(str)
    near = pd.Series(False, index=data.index)
    comparable = method.isin(
        {
            "keller_zscore",
            "crowther_gamma",
            "peak_amplitude",
            "rms_response",
        }
    )
    ratio = score / threshold.replace(0, np.nan)
    near |= comparable & ratio.between(0.80, 1.20, inclusive="both")
    p_value = pd.to_numeric(data.get("p_value"), errors="coerce")
    near |= method.eq("crp_significance") & p_value.between(
        0.025,
        0.10,
        inclusive="both",
    )
    return set(data.loc[near, "channel"].dropna().astype(str))


def _bad_epoch_indices(
    artifacts: pd.DataFrame,
    channel: str,
) -> list[int]:
    if artifacts.empty or "bad_response" not in artifacts:
        return []
    selected = artifacts[
        artifacts["channel"].astype(str).eq(str(channel))
        & _bool_series(artifacts["bad_response"])
    ]
    return (
        pd.to_numeric(selected["epoch"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )


def _indexed_bool_column(
    frame: pd.DataFrame,
    column: str,
    channels: Iterable[str],
) -> pd.Series:
    index = pd.Index(list(map(str, channels)), dtype=str)
    if frame.empty or column not in frame.columns:
        return pd.Series(False, index=index, dtype=bool)
    values = _bool_series(frame[column])
    values.index = frame.index.astype(str)
    return values.reindex(index).fillna(False).astype(bool)


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return (
        values.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _nan_sem(values: np.ndarray) -> np.ndarray:
    finite_count = np.isfinite(values).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(finite_count)
    sem[finite_count < 2] = np.nan
    return sem
