"""Notebook-era ERP grids and comparison plots."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import pandas as pd

from .theme import apply_erpy_theme


def plot_erp_grid(
    epochs_df: pd.DataFrame,
    channels: Optional[Sequence[str]] = None,
    ncols: int = 4,
    mode: str = "mean_sem",
    figsize_per: tuple[float, float] = (3.2, 2.4),
    suptitle: Optional[str] = None,
):
    """Faceted grid of mean ERPs, one subplot per channel."""

    import matplotlib.pyplot as plt

    apply_erpy_theme()
    if channels is None:
        channels = [c for c in epochs_df.columns if pd.api.types.is_numeric_dtype(epochs_df[c])]
    channels = list(channels)
    if not channels:
        raise ValueError("No channels to plot.")
    ncols = max(int(ncols), 1)
    nrows = int(math.ceil(len(channels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        layout="constrained",
        squeeze=False,
    )
    for ax, channel in zip(axes.ravel(), channels):
        mean = epochs_df[channel].groupby(level="time").mean()
        err = epochs_df[channel].groupby(level="time").sem() if mode == "mean_sem" else epochs_df[channel].groupby(level="time").std()
        ax.fill_between(mean.index, mean - err, mean + err, color="C0", alpha=0.25)
        ax.plot(mean.index, mean.values, color="C0", lw=1.2)
        ax.axvline(0.0, color="0.4", ls="--", lw=0.6)
        ax.set_title(str(channel), fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("uV", fontsize=8)
    for ax in axes.ravel()[len(channels) :]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle)
    return fig, axes


def plot_mean_erp_comparison(
    series: Iterable[tuple[str, pd.DataFrame]],
    channel: str,
    ax=None,
    colors: Optional[Sequence[str]] = None,
):
    """Overlay mean +/- SEM for several epoched DataFrames."""

    import matplotlib.pyplot as plt

    apply_erpy_theme()
    ax = ax or plt.subplots(figsize=(7, 4), layout="constrained")[1]
    for i, (label, df) in enumerate(series):
        if channel not in df.columns:
            continue
        color = colors[i] if colors else f"C{i}"
        mean = df[channel].groupby(level="time").mean()
        sem = df[channel].groupby(level="time").sem()
        ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
        ax.plot(mean.index, mean.values, color=color, lw=1.8, label=label)
    ax.axvline(0.0, color="0.4", ls="--", lw=0.8)
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return ax

