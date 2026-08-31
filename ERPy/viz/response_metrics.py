from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..response_metrics import zscore_metric_within_stim


def _infer_channel_col(data: pd.DataFrame, channel_col: str | None) -> str:
    if channel_col is not None:
        return channel_col
    if "channel" in data.columns:
        return "channel"
    if "record_elec" in data.columns:
        return "record_elec"
    raise ValueError("Provide channel_col or include 'channel'/'record_elec' in the table")


def plot_within_stim_zscore_heatmap(
    data: pd.DataFrame,
    metric: str,
    *,
    stim_col: str = "stim_pair",
    channel_col: str | None = None,
    z_col: str | None = None,
    absolute: bool = False,
    top_n_channels: int = 28,
    ax=None,
    title: str | None = None,
):
    """Heatmap of a metric z-scored across contacts within each stimulation site."""

    ax = ax or plt.subplots(figsize=(8.4, 4.8))[1]
    channel_col = _infer_channel_col(data, channel_col)
    z_col = z_col or f"{metric}_within_stim_z"
    if z_col not in data.columns:
        data = zscore_metric_within_stim(
            data,
            metric,
            stim_col=stim_col,
            channel_col=channel_col,
            absolute=absolute,
            output_col=z_col,
        )
    if data.empty or not {stim_col, channel_col, z_col}.issubset(data.columns):
        ax.text(0.5, 0.5, "No within-stim z-score data", ha="center", va="center")
        ax.axis("off")
        return ax
    work = data.copy()
    work[z_col] = pd.to_numeric(work[z_col], errors="coerce")
    channel_order = (
        work.groupby(channel_col)[z_col]
        .apply(lambda values: np.nanmax(np.abs(values.to_numpy(dtype=float))))
        .sort_values(ascending=False)
        .head(top_n_channels)
        .index.astype(str)
        .tolist()
    )
    matrix = work[work[channel_col].astype(str).isin(channel_order)].pivot_table(
        index=stim_col,
        columns=channel_col,
        values=z_col,
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
    ax.set_title(title or f"{metric} within-stim z-score")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label(z_col)
    return ax


def plot_within_stim_zscore_bars(
    data: pd.DataFrame,
    metric: str,
    *,
    stim_pair: str | None = None,
    stim_col: str = "stim_pair",
    channel_col: str | None = None,
    z_col: str | None = None,
    absolute: bool = False,
    top_n: int = 18,
    ax=None,
    title: str | None = None,
):
    """Rank recording contacts by within-stimulation-site z-scored response."""

    ax = ax or plt.subplots(figsize=(6.8, 4.2))[1]
    channel_col = _infer_channel_col(data, channel_col)
    z_col = z_col or f"{metric}_within_stim_z"
    if z_col not in data.columns:
        data = zscore_metric_within_stim(
            data,
            metric,
            stim_col=stim_col,
            channel_col=channel_col,
            absolute=absolute,
            output_col=z_col,
        )
    work = data.copy()
    if stim_pair is not None:
        work = work[work[stim_col].astype(str) == str(stim_pair)]
    if work.empty or z_col not in work.columns:
        ax.text(0.5, 0.5, "No within-stim z-score data", ha="center", va="center")
        ax.axis("off")
        return ax
    work[z_col] = pd.to_numeric(work[z_col], errors="coerce")
    work = work.dropna(subset=[z_col]).sort_values(z_col, ascending=False).head(top_n)
    colors = np.where(work[z_col] >= 0, "#dc2626", "#2563eb")
    ax.barh(work[channel_col].astype(str), work[z_col], color=colors, alpha=0.86)
    ax.invert_yaxis()
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_xlabel(z_col)
    ax.set_ylabel("Recording contact")
    label = f"{stim_pair} " if stim_pair is not None else ""
    ax.set_title(title or f"{label}{metric} within-stim z-score")
    return ax
