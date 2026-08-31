"""Heatmap visualizations for ERP time series."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .theme import apply_erpy_theme


def heatmap_epochs(
    epochs_df: pd.DataFrame,
    channel: str,
    ax=None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = "Amplitude",
):
    """Plot a time-by-epoch heatmap for one channel."""

    import matplotlib.pyplot as plt

    apply_erpy_theme()
    wide = epochs_df[channel].unstack(level="epoch")
    times = wide.index.to_numpy()
    data = wide.values.T
    ax = ax or plt.subplots(figsize=(8, 4), layout="constrained")[1]
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(times[0], times[-1], data.shape[0] - 0.5, -0.5),
        interpolation="nearest",
    )
    ax.axvline(0.0, color="0.2", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Epoch index")
    ax.set_title(f"{channel}: time x epoch")
    cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label(cbar_label)
    return ax, im


def heatmap_channels_time(
    epochs_df: pd.DataFrame,
    channels: Optional[list[str]] = None,
    ax=None,
    cmap: str = "viridis",
):
    """Plot a time-by-channel heatmap after averaging over epochs."""

    import matplotlib.pyplot as plt

    apply_erpy_theme()
    if channels is None:
        channels = [c for c in epochs_df.columns if pd.api.types.is_numeric_dtype(epochs_df[c])]
    mean = epochs_df.groupby(level="time")[channels].mean()
    times = mean.index.to_numpy()
    data = mean.values.T
    ax = ax or plt.subplots(figsize=(8, max(3, 0.25 * len(channels))), layout="constrained")[1]
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        extent=(times[0], times[-1], len(channels) - 0.5, -0.5),
        interpolation="nearest",
    )
    ax.axvline(0.0, color="0.2", ls="--", lw=0.8)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title("Mean amplitude: time x channel")
    plt.colorbar(im, ax=ax, shrink=0.6)
    return ax, im

