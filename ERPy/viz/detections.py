from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from ..crp import CRPConfig, CRPError, run_crp_array
from ..erp_detection import (
    _aligned_epoch_arrays,
    _kundu_envelope,
)


METHOD_ORDER = [
    "kundu_rolston",
    "keller_zscore",
    "crp_significance",
    "crp_energy",
    "crowther_gamma",
    "peak_amplitude",
    "rms_response",
]


def method_significance_matrix(detections: pd.DataFrame) -> pd.DataFrame:
    """Return channel x method boolean significance matrix."""

    if detections.empty:
        return pd.DataFrame()
    matrix = detections.pivot_table(
        index="channel",
        columns="method",
        values="significant",
        aggfunc="max",
        fill_value=False,
    ).astype(bool)
    ordered_methods = [m for m in METHOD_ORDER if m in matrix.columns] + [m for m in matrix.columns if m not in METHOD_ORDER]
    matrix = matrix[ordered_methods]
    row_order = (
        matrix.sum(axis=1)
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    if "peak_amplitude_uv" in detections.columns:
        peak = detections.drop_duplicates("channel").set_index("channel")["peak_amplitude_uv"]
        row_order = sorted(row_order, key=lambda ch: (matrix.loc[ch].sum(), peak.get(ch, 0.0)), reverse=True)
    return matrix.loc[row_order]


def plot_detection_method_matrix(detections: pd.DataFrame, ax=None, top_n: int = 32):
    """Plot implemented significance methods against channels."""

    ax = ax or plt.subplots(figsize=(7.5, 6.0))[1]
    matrix = method_significance_matrix(detections).head(top_n)
    if matrix.empty:
        ax.text(0.5, 0.5, "No detection rows", ha="center", va="center")
        ax.axis("off")
        return ax
    data = matrix.to_numpy(dtype=float)
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    ax.set_xlabel("Detection method")
    ax.set_ylabel("Channel")
    ax.set_title("Method-level significance calls")
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x, y, "1" if data[y, x] else "", ha="center", va="center", fontsize=6, color="#052e16")
    return ax


def plot_method_significance_counts(detections: pd.DataFrame, ax=None):
    """Bar plot of significant channel counts per method."""

    ax = ax or plt.subplots(figsize=(6.4, 3.2))[1]
    if detections.empty:
        ax.text(0.5, 0.5, "No detection rows", ha="center", va="center")
        ax.axis("off")
        return ax
    counts = (
        detections.assign(significant=lambda d: d["significant"].astype(bool))
        .groupby("method")["significant"]
        .sum()
        .reindex([m for m in METHOD_ORDER if m in detections["method"].unique()])
    )
    ax.bar(counts.index, counts.values, color="#2563eb", alpha=0.86)
    ax.set_ylabel("Significant channels")
    ax.set_xlabel("Detection method")
    ax.set_title("Detection method yield")
    ax.tick_params(axis="x", rotation=35)
    return ax


def plot_detection_summary(detections: pd.DataFrame, top_n: int = 32):
    """Combined method matrix and per-method count figure."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained", gridspec_kw={"width_ratios": [1.3, 0.9]})
    plot_detection_method_matrix(detections, ax=axes[0], top_n=top_n)
    plot_method_significance_counts(detections, ax=axes[1])
    return fig


def choose_significant_and_nonsignificant_channels(detections: pd.DataFrame, channels: list[str]) -> tuple[str, str]:
    """Pick one strong significant channel and one low-scoring non-significant channel."""

    if detections.empty:
        return channels[0], channels[-1]
    channel_summary = (
        detections.groupby("channel")
        .agg(
            n_methods=("significant", lambda s: int(np.asarray(s, dtype=bool).sum())),
            peak=("peak_amplitude_uv", "max") if "peak_amplitude_uv" in detections.columns else ("significant", "size"),
        )
        .reset_index()
    )
    channel_summary = channel_summary[channel_summary["channel"].isin(channels)]
    if channel_summary.empty:
        return channels[0], channels[-1]
    sig = channel_summary.sort_values(["n_methods", "peak"], ascending=False)["channel"].iloc[0]
    nonsig_candidates = channel_summary[channel_summary["n_methods"] == 0]
    if nonsig_candidates.empty:
        nonsig_candidates = channel_summary[channel_summary["channel"] != sig]
    if nonsig_candidates.empty:
        return str(sig), str(sig)
    nonsig = nonsig_candidates.sort_values(["n_methods", "peak"], ascending=True)["channel"].iloc[0]
    return str(sig), str(nonsig)


def plot_published_detector_diagnostic(
    epochs,
    channel: str,
    *,
    detections: pd.DataFrame | None = None,
    wideband_epochs=None,
    baseline_window: tuple[float, float] | None = None,
    response_window: tuple[float, float] = (0.01, 0.30),
    signi_n_permutations: int = 1000,
):
    """Visualize the source-native quantities behind one response decision."""

    arr, times, channels = epochs.as_array()
    channels = [str(value) for value in channels]
    if channel not in channels:
        raise KeyError(channel)
    baseline_window = tuple(
        baseline_window
        or getattr(epochs, "baseline", None)
        or (-0.5, -0.03)
    )
    if detections is None:
        detections = epochs.detect_erp_all(
            wideband_epochs=wideband_epochs,
            baseline_window=baseline_window,
            response_window=response_window,
            crowther_gamma={
                "n_permutations": int(signi_n_permutations),
                "random_state": 0,
            },
        )
    detector_rows = pd.DataFrame(detections).copy()
    detector_rows = detector_rows[
        detector_rows["channel"].astype(str).eq(str(channel))
    ].drop_duplicates("method", keep="first")
    detector_rows = detector_rows.set_index("method", drop=False)

    channel_index = channels.index(channel)
    channel_values = np.asarray(arr[:, :, channel_index], dtype=float)
    response_start = max(0.015, float(response_window[0]))
    response_indices = np.flatnonzero(
        (times >= response_start) & (times <= float(response_window[1]))
    )
    baseline_candidates = np.flatnonzero(
        (times >= baseline_window[0])
        & (times <= min(baseline_window[1], -0.005))
    )
    if len(response_indices) and len(baseline_candidates) >= len(response_indices):
        baseline_indices = baseline_candidates[-len(response_indices) :]
        clean_trials = np.all(
            np.isfinite(
                channel_values[:, np.r_[baseline_indices, response_indices]]
            ),
            axis=1,
        )
    else:
        baseline_indices = np.array([], dtype=int)
        clean_trials = np.zeros(channel_values.shape[0], dtype=bool)
    if len(baseline_indices):
        baseline_means = np.mean(
            channel_values[:, baseline_indices],
            axis=1,
        )
        trial_values = channel_values - baseline_means[:, None]
    else:
        trial_values = np.full(channel_values.shape, np.nan, dtype=float)
    detector_trials = trial_values[clean_trials]
    displayed_trials = detector_trials if len(detector_trials) else trial_values
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mean_wave = np.nanmean(displayed_trials, axis=0)
        finite_counts = np.sum(np.isfinite(displayed_trials), axis=0)
        sem_wave = np.nanstd(displayed_trials, axis=0, ddof=1) / np.sqrt(
            np.maximum(finite_counts, 1)
        )

    crp_result = None
    crp_error = ""
    try:
        crp_result = run_crp_array(
            detector_trials,
            times,
            channel=channel,
            config=CRPConfig(
                response_window=(response_start, float(response_window[1])),
                baseline_window=(
                    float(times[baseline_indices[0]]),
                    float(times[baseline_indices[-1]]),
                ),
            ),
        )
    except (CRPError, ValueError, IndexError) as exc:
        crp_error = str(exc)

    detector_epochs = wideband_epochs if wideband_epochs is not None else epochs
    detector_arr, detector_times, detector_sfreq, _ = _aligned_epoch_arrays(
        detector_epochs,
        channels,
    )
    kundu_baseline = (detector_times >= -0.100) & (detector_times <= -0.005)
    kundu_response = (detector_times >= 0.005) & (detector_times <= 0.100)
    try:
        kundu_envelope, _, _ = _kundu_envelope(
            detector_arr,
            sfreq=detector_sfreq,
            validity_window_mask=kundu_baseline | kundu_response,
        )
        channel_envelope = kundu_envelope[:, :, channel_index]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            kundu_median_wave = np.nanmedian(
                channel_envelope[:, kundu_response],
                axis=0,
            )
            kundu_baseline_median = float(
                np.nanmedian(channel_envelope[:, kundu_baseline])
            )
        if not np.isfinite(kundu_median_wave).any():
            raise ValueError("Kundu response envelope has no finite samples")
        kundu_threshold = 3.0 * kundu_baseline_median
    except Exception:
        channel_envelope = np.empty((0, 0), dtype=float)
        kundu_median_wave = np.array([], dtype=float)
        kundu_baseline_median = kundu_threshold = np.nan

    if len(detector_trials) and len(baseline_indices):
        response_rms = np.sqrt(
            np.mean(detector_trials[:, response_indices] ** 2, axis=1)
        )
        baseline_rms = np.sqrt(
            np.mean(detector_trials[:, baseline_indices] ** 2, axis=1)
        )
    else:
        response_rms = np.array([], dtype=float)
        baseline_rms = np.array([], dtype=float)

    colors = {
        "ink": "#20262e",
        "blue": "#2f6f9f",
        "teal": "#16817a",
        "orange": "#d97924",
        "red": "#b23a48",
        "green": "#2f7d4a",
        "gray": "#9aa3ad",
        "light": "#e8ecef",
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.6, 8.2),
        layout="constrained",
    )

    axis = axes[0, 0]
    time_ms = np.asarray(times, dtype=float) * 1000.0
    axis.fill_between(
        time_ms,
        mean_wave - sem_wave,
        mean_wave + sem_wave,
        color=colors["blue"],
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(time_ms, mean_wave, color=colors["blue"], linewidth=1.8)
    axis.axvspan(10, 50, color=colors["orange"], alpha=0.11, label="N1 10-50 ms")
    axis.axvspan(50, 250, color=colors["teal"], alpha=0.08, label="Late 50-250 ms")
    axis.axvline(0, color=colors["ink"], linestyle="--", linewidth=0.9)
    axis.axhline(0, color=colors["gray"], linewidth=0.8)
    axis.set(xlim=(-100, 300), xlabel="Time after stimulation (ms)", ylabel="Amplitude (uV)", title="A  Mean evoked response")
    axis.legend(frameon=False, fontsize=8, loc="upper right")

    axis = axes[0, 1]
    if kundu_median_wave.size:
        kundu_times_ms = detector_times[kundu_response] * 1000.0
        axis.plot(kundu_times_ms, kundu_median_wave, color=colors["orange"], linewidth=1.9, label="Trial-median envelope")
        axis.axhline(kundu_threshold, color=colors["red"], linestyle="--", linewidth=1.2, label="3x prestimulus median")
        axis.axhline(30.0, color=colors["ink"], linestyle=":", linewidth=1.2, label="30 uV magnitude floor")
        axis.fill_between(kundu_times_ms, kundu_threshold, kundu_median_wave, where=kundu_median_wave > kundu_threshold, color=colors["orange"], alpha=0.18)
        axis.set_xlim(5, 100)
        axis.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.92,
            edgecolor="none",
            fontsize=7.5,
            loc="upper right",
        )
    else:
        axis.text(0.5, 0.5, "Kundu envelope unavailable", ha="center", va="center", transform=axis.transAxes)
    axis.set(xlabel="Time after stimulation (ms)", ylabel="Envelope amplitude (uV)", title="B  Published Kundu envelope")

    axis = axes[0, 2]
    if crp_result is not None:
        reconstruction = crp_result.canonical_waveform * float(
            np.nanmedian(crp_result.projections)
        )
        canonical_times_ms = crp_result.times * 1000.0
        axis.plot(time_ms, mean_wave, color=colors["gray"], linewidth=1.2, alpha=0.8, label="Matched-trial mean")
        axis.plot(canonical_times_ms, reconstruction, color=colors["teal"], linewidth=2.2, label="Canonical reconstruction")
        axis.axvline(0, color=colors["ink"], linestyle="--", linewidth=0.9)
        axis.axhline(0, color=colors["gray"], linewidth=0.8)
        axis.set_xlim(-25, max(150, crp_result.times[-1] * 1000.0 + 15))
        axis.legend(frameon=False, fontsize=8)
    else:
        axis.text(0.5, 0.54, "CRP unavailable", ha="center", va="center", transform=axis.transAxes, weight="bold")
        axis.text(0.5, 0.42, crp_error or "Insufficient matched clean trials", ha="center", va="center", transform=axis.transAxes, fontsize=8, wrap=True)
    axis.set(xlabel="Time after stimulation (ms)", ylabel="Amplitude (uV)", title="C  CRP canonical response")

    axis = axes[1, 0]
    if crp_result is not None:
        duration_index = int(np.nanargmax(crp_result.mean_projection_profile))
        duration_endpoint_ms = float(crp_result.projection_times[duration_index] * 1000.0)
        axis.plot(crp_result.projection_times * 1000.0, crp_result.mean_projection_profile, color=colors["teal"], linewidth=2.0)
        axis.axvline(duration_endpoint_ms, color=colors["red"], linestyle="--", linewidth=1.2)
        axis.scatter([duration_endpoint_ms], [crp_result.mean_projection_profile[duration_index]], color=colors["red"], s=34, zorder=3)
        crp_p_label = "<1e-300" if crp_result.p_value == 0 else f"{crp_result.p_value:.2g}"
        axis.set_title(f"D  CRP duration: {crp_result.response_duration_s * 1000.0:.1f} ms, p={crp_p_label}")
    else:
        axis.text(0.5, 0.5, "No CRP duration estimate", ha="center", va="center", transform=axis.transAxes)
        axis.set_title("D  CRP duration unavailable")
    axis.set(xlabel="Candidate response endpoint (ms)", ylabel="Mean cross-projection")

    axis = axes[1, 1]
    finite = np.isfinite(response_rms) & np.isfinite(baseline_rms)
    for baseline_value, response_value in zip(baseline_rms[finite], response_rms[finite]):
        axis.plot([0, 1], [baseline_value, response_value], color=colors["gray"], alpha=0.28, linewidth=0.8)
    axis.scatter(np.zeros(finite.sum()), baseline_rms[finite], color=colors["gray"], s=18, alpha=0.72)
    axis.scatter(np.ones(finite.sum()), response_rms[finite], color=colors["blue"], s=18, alpha=0.72)
    energy_row = (
        detector_rows.loc["crp_energy"]
        if "crp_energy" in detector_rows.index
        else pd.Series(dtype=object)
    )
    n_clean = pd.to_numeric(
        energy_row.get("n_trials_clean", finite.sum()),
        errors="coerce",
    )
    n_clean = int(n_clean) if np.isfinite(n_clean) else int(finite.sum())
    energy_title = (
        "E  Matched-trial RMS: "
        f"{energy_row.get('rms_ratio_db', np.nan):.1f} dB, "
        f"pE={energy_row.get('p_energy', np.nan):.2g}, "
        f"n={n_clean}"
        if finite.any()
        else f"E  Matched-trial RMS unavailable (n={n_clean})"
    )
    axis.set_xticks([0, 1], ["Matched baseline", "Response"])
    axis.set(ylabel="Trial RMS (uV)", title=energy_title)

    axis = axes[1, 2]
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.6, len(METHOD_ORDER) - 0.4)
    axis.axis("off")
    display_names = {
        "kundu_rolston": "Kundu",
        "keller_zscore": "Keller",
        "crp_significance": "CRP",
        "crp_energy": "CRP-energy conjunction",
        "crowther_gamma": "SIGNI",
        "peak_amplitude": "N1 peak",
        "rms_response": "RMS magnitude",
    }
    for row_index, method in enumerate(METHOD_ORDER):
        y = len(METHOD_ORDER) - 1 - row_index
        available = method in detector_rows.index
        passed = bool(detector_rows.loc[method, "significant"]) if available else False
        circle_color = colors["green"] if passed else colors["light"]
        axis.scatter([0.06], [y], s=145, color=circle_color, edgecolor=colors["ink"], linewidth=0.6)
        axis.text(0.06, y, "yes" if passed else "no", ha="center", va="center", fontsize=6.5, color="white" if passed else colors["ink"])
        details = "not computed"
        if available:
            row = detector_rows.loc[method]
            if method == "kundu_rolston":
                details = f"post {row.get('kundu_post_median_uv', np.nan):.1f} uV; {row.get('kundu_longest_suprathreshold_ms', np.nan):.1f} ms"
            elif method == "keller_zscore":
                details = f"max z {row.get('keller_max_z', np.nan):.1f}; polarity {'yes' if bool(row.get('keller_polarity_reversal', False)) else 'no'}"
            elif method == "crp_significance":
                details = f"t {row.get('crp_t_value', np.nan):.1f}; q {row.get('crp_q_value', np.nan):.2g}"
            elif method == "crp_energy":
                details = (
                    f"pR {row.get('p_crp', np.nan):.2g}; "
                    f"pE {row.get('p_energy', np.nan):.2g}; "
                    f"pJ {row.get('p_joint', np.nan):.2g}; "
                    f"qJ {row.get('q_joint', np.nan):.2g}; "
                    f"{row.get('classification', 'not evaluated')}"
                )
            elif method == "crowther_gamma":
                details = f"SNR {row.get('signi_snr', np.nan):.1f}; pBonf {row.get('signi_p_value_bonferroni', np.nan):.2g}"
            elif method == "peak_amplitude":
                details = f"N1 z {row.get('keller_a1_z', np.nan):.1f}; {row.get('keller_a1_peak_uv', np.nan):.1f} uV"
            elif method == "rms_response":
                details = f"RMS {row.get('rms_waveform_10_100_uv', np.nan):.1f} uV; N1 z {row.get('rms_detection_n1_z', np.nan):.1f}"
        axis.text(0.15, y + 0.10, display_names[method], ha="left", va="center", fontsize=9, weight="bold", color=colors["ink"])
        axis.text(0.15, y - 0.16, details, ha="left", va="center", fontsize=7.5, color=colors["ink"])
    axis.set_title("F  Source-native detector decisions", loc="left")

    primary_column = next(
        (
            column
            for column in (
                "primary_qc_pass",
                "crp_energy_qc_pass",
                "primary_significant",
            )
            if column in detector_rows.columns
        ),
        None,
    )
    primary = bool(detector_rows[primary_column].any()) if primary_column else False
    fig.suptitle(
        f"{channel}: CRP-RMS reproducibility-energy conjunction {'PASS' if primary else 'FAIL'}",
        fontsize=15,
        weight="bold",
        color=colors["ink"],
    )
    return fig
