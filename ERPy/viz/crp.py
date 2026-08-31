from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..baseline import baseline_center_frame
from ..crp import CRPConfig, CRPResult, run_crp


def plot_crp_curve(result: CRPResult, ax=None, color: str = "#0f766e"):
    """Plot the canonical response curve learned from the response window."""

    ax = ax or plt.subplots(figsize=(5.5, 3.0))[1]
    ax.plot(result.times * 1000.0, result.canonical_response_curve, color=color, lw=2.0)
    ax.axhline(0, color="0.82", lw=0.8)
    ax.set_xlabel("Time after stimulation (ms)")
    ax.set_ylabel("Canonical response (unit norm)")
    ax.set_title(f"{result.channel} canonical response curve")
    return ax


def plot_crp_weight_timecourse(result: CRPResult, ax=None, color: str = "#7c3aed", zscore: bool = False):
    """Plot canonical response profile expression over the full epoch."""

    ax = ax or plt.subplots(figsize=(5.8, 3.0))[1]
    times = result.canonical_weight_times
    if times is None or len(times) == 0 or result.canonical_weight_mean is None:
        ax.text(0.5, 0.5, "No CRP weight timecourse", ha="center", va="center")
        ax.axis("off")
        return ax
    y = result.canonical_weight_z if zscore and result.canonical_weight_z is not None else result.canonical_weight_mean
    ylabel = "Canonical weight z" if zscore else "Canonical expression (uV)"
    ax.plot(times * 1000.0, y, color=color, lw=2.0)
    if not zscore and result.canonical_weight_sem is not None and len(result.canonical_weight_sem) == len(y):
        sem = result.canonical_weight_sem
        ax.fill_between(times * 1000.0, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)
    ax.axvline(0, color="0.2", ls="--", lw=0.9)
    ax.axhline(0, color="0.82", lw=0.8)
    ax.axvspan(result.baseline_window[0] * 1000.0, result.baseline_window[1] * 1000.0, color="#64748b", alpha=0.10)
    ax.axvspan(result.response_window[0] * 1000.0, result.response_window[1] * 1000.0, color="#0f766e", alpha=0.10)
    ax.set_xlabel("Time after stimulation (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title("Canonical weight over time")
    return ax


def plot_crp_projections(result: CRPResult, ax=None, color: str = "#2563eb"):
    """Plot trial projections onto the canonical response profile."""

    ax = ax or plt.subplots(figsize=(5.5, 3.0))[1]
    x = np.arange(1, len(result.projections) + 1)
    ax.bar(x, result.projections, color=color, alpha=0.82)
    ax.axhline(0, color="0.25", lw=0.8)
    ax.axhline(np.nanmean(result.projections), color="#dc2626", lw=1.4, ls="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Alpha (uV sqrt(samples))")
    ax.set_title("Trial canonical-response weights")
    return ax


def plot_crp_score_map(crp_table: pd.DataFrame, ax=None, top_n: int = 20):
    """Rank channels by CRP score with significance-aware coloring."""

    ax = ax or plt.subplots(figsize=(6.5, 4.0))[1]
    data = crp_table.copy()
    if data.empty or "score" not in data.columns:
        ax.text(0.5, 0.5, "No CRP scores", ha="center", va="center")
        ax.axis("off")
        return ax
    data = data.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n)
    colors = np.where(data.get("significant", False).astype(bool), "#0f766e", "#94a3b8")
    ax.barh(data["channel"], data["score"], color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_xlabel("CRP score")
    ax.set_ylabel("Channel")
    ax.set_title("CRP score map")
    return ax


def plot_crp_summary(
    epochs,
    channel: str,
    config: CRPConfig | None = None,
    result: CRPResult | None = None,
):
    """Summary CRP panel for one channel."""

    result = result or run_crp(epochs, channel, config=config)
    mean = baseline_center_frame(
        epochs.get_mean_waveform(),
        result.baseline_window,
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), layout="constrained")
    ax = axes[0, 0]
    ax.plot(mean.index * 1000.0, mean[channel], color="#2563eb", lw=1.9)
    ax.axvline(0, color="0.2", ls="--", lw=1.0)
    ax.axhline(0, color="0.82", lw=0.8)
    ax.axvspan(result.response_window[0] * 1000.0, result.response_window[1] * 1000.0, color="#0f766e", alpha=0.10)
    ax.set_xlabel("Time after stimulation (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title(f"{channel} mean response")

    plot_crp_curve(result, ax=axes[0, 1])
    plot_crp_weight_timecourse(result, ax=axes[0, 2])
    plot_crp_projections(result, ax=axes[1, 0])
    plot_crp_weight_timecourse(result, ax=axes[1, 1], zscore=True, color="#dc2626")

    axes[1, 2].axis("off")
    lines = [
        "CRP Summary",
        f"Channel       {result.channel}",
        f"t statistic   {result.t_value:.3f}",
        f"p value       {result.p_value:.3g}",
        f"Duration      {result.response_duration_s * 1000.0:.1f} ms",
        (
            f"Median alpha' {np.nanmedian(result.alpha_prime_uv):.2f} uV"
            if result.alpha_prime_uv is not None
            and np.isfinite(result.alpha_prime_uv).any()
            else "Median alpha' n/a"
        ),
        (
            f"Median SNR    {np.nanmedian(result.snr):.2f}"
            if result.snr is not None and np.isfinite(result.snr).any()
            else "Median SNR    n/a"
        ),
        f"Perm p        {result.permutation_p_value:.3g}" if np.isfinite(result.permutation_p_value) else "Perm p        not run",
        f"Max weight z  {result.max_canonical_weight_z:.2f}" if np.isfinite(result.max_canonical_weight_z) else "Max weight z  n/a",
        f"Recon RMS     {result.reconstruction_rms_uv:.2f} uV" if np.isfinite(result.reconstruction_rms_uv) else "Recon RMS     n/a",
        f"Significant   {'yes' if result.significant else 'no'}",
        f"Window        {result.response_window[0] * 1000:.0f}-{result.response_window[1] * 1000:.0f} ms",
        f"Trials        {len(result.projections)}",
    ]
    axes[1, 2].text(0.04, 0.94, lines[0], va="top", ha="left", fontsize=10, weight="bold")
    axes[1, 2].text(
        0.04,
        0.82,
        "\n".join(lines[1:]),
        va="top",
        ha="left",
        fontsize=9,
        linespacing=1.55,
        family="monospace",
    )
    return fig


def plot_crp_response_contrast(
    epochs,
    significant_channel: str,
    nonsignificant_channel: str,
    config: CRPConfig | None = None,
):
    """Contrast significant and non-significant channels with mean response, CRP curves, and CRP weights."""

    config = config or CRPConfig()
    mean = baseline_center_frame(
        epochs.get_mean_waveform(),
        config.baseline_window,
    )
    results = {
        "Significant / strong": run_crp(epochs, significant_channel, config=config),
        "Non-significant / weak": run_crp(epochs, nonsignificant_channel, config=config),
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6), layout="constrained")
    for row_i, (label, result) in enumerate(results.items()):
        channel = result.channel
        ax = axes[row_i, 0]
        ax.plot(mean.index * 1000.0, mean[channel], color="#2563eb" if row_i == 0 else "#64748b", lw=1.9)
        ax.axvline(0, color="0.2", ls="--", lw=1.0)
        ax.axhline(0, color="0.82", lw=0.8)
        ax.axvspan(result.response_window[0] * 1000.0, result.response_window[1] * 1000.0, color="#0f766e", alpha=0.10)
        ax.set_title(f"{label}: {channel} mean")
        ax.set_xlabel("Time after stimulation (ms)")
        ax.set_ylabel("Amplitude (uV)")

        ax = axes[row_i, 1]
        plot_crp_curve(result, ax=ax, color="#0f766e" if row_i == 0 else "#64748b")
        ax.set_title(f"{channel} CRP: score {result.score:.2f}, p={result.p_value:.3g}")

        ax = axes[row_i, 2]
        plot_crp_weight_timecourse(result, ax=ax, color="#7c3aed" if row_i == 0 else "#64748b")
        if np.isfinite(result.max_canonical_weight_z):
            ax.set_title(f"{channel} weight over time, max z={result.max_canonical_weight_z:.2f}")
    return fig


def plot_crp_site_comparison(
    comparison_table: pd.DataFrame,
    metric: str = "within_stim_z",
    ax=None,
    top_n_channels: int = 24,
    title: str | None = None,
):
    """Plot a stimulation-site by recording-channel CRP comparison heatmap."""

    ax = ax or plt.subplots(figsize=(8.2, 4.6))[1]
    required = {"stim_pair", "channel", metric}
    if comparison_table.empty or not required.issubset(comparison_table.columns):
        ax.text(0.5, 0.5, "No CRP comparison data", ha="center", va="center")
        ax.axis("off")
        return ax
    data = comparison_table.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    def finite_maximum(values: pd.Series) -> float:
        finite = np.abs(values.to_numpy(dtype=float))
        finite = finite[np.isfinite(finite)]
        return float(finite.max()) if finite.size else np.nan

    channel_order = (
        data.groupby("channel")[metric]
        .apply(finite_maximum)
        .sort_values(ascending=False)
        .head(top_n_channels)
        .index.tolist()
    )
    matrix = data[data["channel"].isin(channel_order)].pivot_table(
        index="stim_pair",
        columns="channel",
        values=metric,
        aggfunc="max",
    )
    matrix = matrix.reindex(columns=channel_order)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_xlabel("Recording contact")
    ax.set_ylabel("Stimulation site")
    ax.set_title(title or metric.replace("_", " ").title())
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label(metric.replace("_", " "))
    return ax
