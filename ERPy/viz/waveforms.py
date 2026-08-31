from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..baseline import baseline_center_array


def _referenced_array(epochs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values, times, channels = epochs.as_array()
    return (
        baseline_center_array(
            values,
            times,
            getattr(epochs, "baseline", None),
        ),
        times,
        list(map(str, channels)),
    )


def _referenced_mean_sem(epochs) -> tuple[pd.DataFrame, pd.DataFrame]:
    values, times, channels = _referenced_array(epochs)
    mean = pd.DataFrame(
        np.nanmean(values, axis=0),
        index=times,
        columns=channels,
    )
    counts = np.isfinite(values).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem_values = np.nanstd(values, axis=0, ddof=1) / np.sqrt(counts)
    sem_values[counts < 2] = np.nan
    sem = pd.DataFrame(sem_values, index=times, columns=channels)
    return mean, sem


def _finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _channel_list(epochs, channels=None) -> list[str]:
    if channels is None:
        return epochs.channels
    if isinstance(channels, str):
        return [channels]
    return list(channels)


def _nearest_zero_sample(epochs) -> float | None:
    times = epochs.times
    if len(times) == 0:
        return None
    if hasattr(epochs, "zero_time_report"):
        report = epochs.zero_time_report()
        anchor = _finite_float(report.get("nearest_sample_s"))
        if anchor is not None:
            return float(times[int(np.argmin(np.abs(times - anchor)))])
    zero_time = getattr(epochs, "zero_time", 0.0)
    anchor = _finite_float(zero_time)
    if anchor is None:
        return None
    return float(times[int(np.argmin(np.abs(times - anchor)))])


def _zero_anchor_by_epoch(epochs) -> dict[int, float]:
    # All trials are zero-anchored at the same common time, so mark that time on
    # every trial rather than each trial's individually detected artifact end.
    anchor = _nearest_zero_sample(epochs)
    if anchor is not None:
        return {int(ep): anchor for ep in epochs.epochs_df.index.get_level_values("epoch").unique()}
    if hasattr(epochs, "post_artifact_anchor_report"):
        report = epochs.post_artifact_anchor_report()
        if not report.empty and {"epoch", "anchor_time_s"}.issubset(report.columns):
            out: dict[int, float] = {}
            times = epochs.times
            for _, row in report.iterrows():
                anchor = _finite_float(row.get("anchor_time_s"))
                if anchor is None or len(times) == 0:
                    continue
                nearest = float(times[int(np.argmin(np.abs(times - anchor)))])
                out[int(row["epoch"])] = nearest
            return out
    return {}


def _mark_zero_time(ax, epochs, y: float = 0.0):
    zero_sample = _nearest_zero_sample(epochs)
    if zero_sample is None:
        return
    if abs(zero_sample) > 1e-12:
        ax.axvline(zero_sample, color="#111827", ls=":", lw=1.0)
    ax.scatter([zero_sample], [y], s=22, color="#111827", edgecolors="white", linewidths=0.55, zorder=6)


def plot_mean(epochs, channel: str, ax=None, color="#2563eb", label: str | None = None):
    """Plot a channel's trial mean with a shaded standard-error band."""

    mean, sem = _referenced_mean_sem(epochs)
    ax = ax or plt.subplots(figsize=(6, 3))[1]
    ax.plot(mean.index, mean[channel], color=color, lw=1.8, label=label or channel)
    if channel in sem:
        ax.fill_between(mean.index, mean[channel] - sem[channel], mean[channel] + sem[channel], color=color, alpha=0.18, lw=0)
    ax.axvline(0, color="0.2", ls="--", lw=1)
    ax.axhline(0, color="0.8", lw=0.8)
    zero_sample = _nearest_zero_sample(epochs)
    if zero_sample is not None and channel in mean:
        y0 = float(mean.loc[zero_sample, channel]) if zero_sample in mean.index else 0.0
        _mark_zero_time(ax, epochs, y=y0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title(channel)
    return ax


def plot_overlay(epochs, channel: str, ax=None, alpha: float = 0.22):
    """Overlay individual trials and the mean response for one channel."""

    ax = ax or plt.subplots(figsize=(6, 3))[1]
    values, times, channels = _referenced_array(epochs)
    channel_index = channels.index(str(channel))
    for trial in values[:, :, channel_index]:
        ax.plot(times, trial, color="0.55", alpha=alpha, lw=0.7)
    plot_mean(epochs, channel, ax=ax, color="#dc2626", label="mean")
    return ax


def plot_heatmap(epochs, channel: str, ax=None, cmap="RdBu_r"):
    """Plot trial-by-time amplitudes for one channel as a heat map."""

    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    values, times, channels = _referenced_array(epochs)
    data = values[:, :, channels.index(str(channel))]
    vmax = np.nanpercentile(np.abs(data), 98) or 1.0
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[times.min(), times.max(), 0.5, data.shape[0] + 0.5],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(0, color="k", ls="--", lw=0.8)
    zero_sample = _nearest_zero_sample(epochs)
    if zero_sample is not None and abs(zero_sample) > 1e-12:
        ax.axvline(zero_sample, color="#111827", ls=":", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial")
    if data.shape[0] <= 10:
        ax.set_yticks(np.arange(1, data.shape[0] + 1))
    ax.set_title(f"{channel} trials")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    return ax


def plot_grid(epochs, channels: Iterable[str] | None = None, ncols: int = 4):
    """Plot mean waveforms for several channels in a shared grid."""

    channels = _channel_list(epochs, channels)
    ncols = max(1, ncols)
    nrows = math.ceil(len(channels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False)
    for ax, ch in zip(axes.ravel(), channels):
        plot_mean(epochs, ch, ax=ax)
    for ax in axes.ravel()[len(channels) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_ranked_grid(
    epochs,
    detections: pd.DataFrame,
    metric: str = "peak_amplitude_uv",
    top_n: int = 16,
    ncols: int = 4,
    consensus_only: bool = True,
):
    """Small-multiple mean waveforms ordered by a detection metric."""

    data = detections.copy()
    if consensus_only and "consensus_ch" in data.columns:
        data = data[data["consensus_ch"]]
    if data.empty or metric not in data.columns:
        return plot_grid(epochs, channels=epochs.channels[:top_n], ncols=ncols)
    order = (
        data.drop_duplicates("channel")
        .sort_values(metric, ascending=False)["channel"]
        .tolist()
    )
    channels = [ch for ch in order if ch in epochs.channels][:top_n]
    return plot_grid(epochs, channels=channels, ncols=ncols)


def plot_grouped_grid(
    epochs,
    channel_metadata: pd.DataFrame,
    group_col: str,
    channel_col: str = "channel",
    channels: Iterable[str] | None = None,
    ncols: int = 4,
):
    """Plot mean waveforms grouped by anatomy, lead, method label, or any metadata column."""

    channels = _channel_list(epochs, channels)
    meta = channel_metadata.copy()
    if channel_col not in meta.columns:
        raise ValueError(f"channel_metadata must contain {channel_col!r}")
    if group_col not in meta.columns:
        raise ValueError(f"channel_metadata must contain {group_col!r}")
    meta = meta[meta[channel_col].isin(channels)].drop_duplicates(channel_col)
    groups = [(g, sub[channel_col].tolist()) for g, sub in meta.groupby(group_col, dropna=False)]
    if not groups:
        return plot_grid(epochs, channels=channels, ncols=ncols)

    nrows = sum(math.ceil(len(chs) / ncols) for _, chs in groups)
    fig, axes = plt.subplots(max(nrows, 1), ncols, figsize=(3.2 * ncols, 2.35 * max(nrows, 1)), squeeze=False)
    ax_iter = iter(axes.ravel())
    for group, group_channels in groups:
        for i, ch in enumerate(group_channels):
            ax = next(ax_iter)
            plot_mean(epochs, ch, ax=ax)
            if i == 0:
                ax.set_title(f"{group}: {ch}")
    for ax in ax_iter:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_mean_erp_comparison(
    epoch_map: dict[str, object],
    channel: str,
    ax=None,
    colors: Iterable[str] | None = None,
):
    """Overlay the same channel from multiple Epochs objects."""

    ax = ax or plt.subplots(figsize=(6.5, 3.5))[1]
    colors = list(colors or plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    for i, (label, epochs) in enumerate(epoch_map.items()):
        mean, _ = _referenced_mean_sem(epochs)
        if channel not in mean.columns:
            continue
        ax.plot(mean.index, mean[channel], lw=1.8, label=label, color=colors[i % len(colors)] if colors else None)
    ax.axvline(0, color="0.2", ls="--", lw=1)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{channel} comparison")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_trials_grid(epochs, channel: str, ncols: int = 5):
    """Plot each trial from one channel in a separate small panel."""

    trial_ids = list(epochs.epochs_df.index.get_level_values("epoch").unique())
    values, times, channels = _referenced_array(epochs)
    channel_index = channels.index(str(channel))
    nrows = math.ceil(len(trial_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.8 * nrows), squeeze=False)
    anchor_by_epoch = _zero_anchor_by_epoch(epochs)
    for trial_index, (ax, ep) in enumerate(zip(axes.ravel(), trial_ids)):
        ax.plot(
            times,
            values[trial_index, :, channel_index],
            color="#334155",
            lw=0.8,
        )
        ax.axvline(0, color="0.3", ls="--", lw=0.7)
        if int(ep) in anchor_by_epoch and abs(anchor_by_epoch[int(ep)]) > 1e-12:
            ax.axvline(anchor_by_epoch[int(ep)], color="#111827", ls=":", lw=0.7)
        ax.set_title(f"trial {ep}", fontsize=8)
    for ax in axes.ravel()[len(trial_ids) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_butterfly(epochs, channels: Iterable[str] | None = None, ax=None):
    """Overlay trial-mean waveforms from multiple channels on one axis."""

    channels = _channel_list(epochs, channels)
    mean, _ = _referenced_mean_sem(epochs)
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    for ch in channels:
        ax.plot(mean.index, mean[ch], lw=0.9, alpha=0.75, label=ch)
    ax.axvline(0, color="0.2", ls="--", lw=1)
    ax.axhline(0, color="0.85", lw=0.8)
    _mark_zero_time(ax, epochs, y=0.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("Mean response butterfly")
    return ax


def plot_response_map(detections: pd.DataFrame, metric: str = "peak_amplitude_uv", ax=None, top_n: int = 20):
    """Plot the highest-ranked channel values from a detection metric."""

    ax = ax or plt.subplots(figsize=(7, 4))[1]
    data = detections.copy()
    if "consensus_ch" in data.columns:
        data = data[data["consensus_ch"]]
    data = data.drop_duplicates("channel").nlargest(top_n, metric)
    ax.barh(data["channel"], data[metric], color="#2563eb")
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_ylabel("Channel")
    ax.set_title("Response map")
    return ax


def plot_summary_panel(epochs, channel: str, detections: pd.DataFrame | None = None):
    """Four-panel response summary: mean +/- SEM, trial overlay, trial heatmap, and a text summary."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    plot_mean(epochs, channel, ax=axes[0, 0])
    plot_overlay(epochs, channel, ax=axes[0, 1])
    plot_heatmap(epochs, channel, ax=axes[1, 0])
    axes[1, 1].axis("off")
    lines = [
        channel,
        f"Trials        {epochs.n_trials()}",
        f"Sampling      {epochs.sfreq:.1f} Hz",
    ]
    if hasattr(epochs, "zero_time_report"):
        zero_report = epochs.zero_time_report()
        if zero_report.get("max_abs_uv") is not None and np.isfinite(float(zero_report["max_abs_uv"])):
            median_anchor = _finite_float(zero_report.get("median_anchor_s"))
            if median_anchor is not None:
                lines.append(f"Zero anchor  {median_anchor * 1000:.1f} ms")
            else:
                nearest = _finite_float(zero_report.get("nearest_sample_s"))
                if nearest is not None:
                    lines.append(f"Zero anchor  {nearest * 1000:.1f} ms")
            lines.append(f"Residual     {float(zero_report['max_abs_uv']):.3g} uV max")
            if zero_report.get("n_anchor_reports"):
                lines.append(f"Onset artifact {int(zero_report.get('n_onset_artifact_epochs', 0))}/{int(zero_report['n_anchor_reports'])}")
    if detections is not None and not detections.empty:
        sub = detections[detections["channel"] == channel]
        if not sub.empty:
            n = int(sub["significant"].sum())
            consensus = bool(sub["consensus_ch"].iloc[0]) if "consensus_ch" in sub else False
            peak = float(sub["peak_amplitude_uv"].max()) if "peak_amplitude_uv" in sub else np.nan
            lines.extend(
                [
                    f"Methods       {n}",
                    f"Consensus     {'yes' if consensus else 'no'}",
                    f"Peak          {peak:.2f} uV" if np.isfinite(peak) else "Peak          n/a",
                ]
            )
    axes[1, 1].text(0.04, 0.94, "Response Summary", va="top", ha="left", fontsize=10, weight="bold")
    axes[1, 1].text(
        0.04,
        0.82,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        linespacing=1.55,
        family="monospace",
    )
    return fig


class EpochsPlotter:
    """Waveform plotting accessor available as ``epochs.plot``."""

    def __init__(self, epochs) -> None:
        self.epochs = epochs

    def mean(self, channel: str, **kwargs):
        """Plot a channel's trial mean and standard error."""

        return plot_mean(self.epochs, channel, **kwargs)

    def grid(self, channels=None, **kwargs):
        """Plot trial-mean waveforms for several channels in a grid."""

        return plot_grid(self.epochs, channels=channels, **kwargs)

    def ranked_grid(self, detections: pd.DataFrame, **kwargs):
        """Plot channel waveforms ordered by a detection metric."""

        return plot_ranked_grid(self.epochs, detections=detections, **kwargs)

    def grouped_grid(self, channel_metadata: pd.DataFrame, group_col: str, **kwargs):
        """Plot channel waveforms grouped by a metadata column."""

        return plot_grouped_grid(self.epochs, channel_metadata=channel_metadata, group_col=group_col, **kwargs)

    def overlay(self, channel: str, **kwargs):
        """Overlay individual trials and the mean for one channel."""

        return plot_overlay(self.epochs, channel, **kwargs)

    def trials_grid(self, channel: str, **kwargs):
        """Plot each trial from one channel in its own panel."""

        return plot_trials_grid(self.epochs, channel, **kwargs)

    def butterfly(self, channels=None, **kwargs):
        """Overlay trial-mean waveforms from selected channels."""

        return plot_butterfly(self.epochs, channels=channels, **kwargs)

    def heatmap(self, channel: str, **kwargs):
        """Plot trial-by-time amplitudes for one channel."""

        return plot_heatmap(self.epochs, channel, **kwargs)

    def summary(self, channel: str, detections: pd.DataFrame | None = None, **kwargs):
        """Create a four-panel waveform and detection summary."""

        return plot_summary_panel(self.epochs, channel, detections=detections, **kwargs)

    def detectors(
        self,
        channel: str,
        detections: pd.DataFrame | None = None,
        **kwargs,
    ):
        """Plot source-native detector quantities for one channel."""

        from .detections import plot_published_detector_diagnostic

        return plot_published_detector_diagnostic(
            self.epochs,
            channel,
            detections=detections,
            **kwargs,
        )
