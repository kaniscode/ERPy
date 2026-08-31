"""Quick ERP visualizations for notebook workflows."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .theme import apply_erpy_theme


def quick_erp(
    epochs_df: pd.DataFrame,
    channel: str,
    ax=None,
    mode: str = "mean_sem",
    color: Optional[str] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot one channel from an ``(epoch, time)`` indexed epochs DataFrame."""

    import matplotlib.pyplot as plt

    apply_erpy_theme()
    if "epoch" not in epochs_df.index.names or "time" not in epochs_df.index.names:
        raise ValueError("epochs_df must have a MultiIndex with 'epoch' and 'time' levels.")
    if channel not in epochs_df.columns:
        raise KeyError(channel)

    ax = ax or plt.subplots(figsize=(7, 4), layout="constrained")[1]
    grouped_epoch = epochs_df[channel].groupby(level="epoch")

    if mode == "trials":
        for _, series in grouped_epoch:
            ax.plot(series.index.get_level_values("time"), series.values, color=color or "0.6", alpha=0.35, lw=0.8)
        mean = epochs_df[channel].groupby(level="time").mean()
        ax.plot(mean.index, mean.values, color=color or "C0", lw=2.0, label=label or "mean")
    else:
        mean = epochs_df[channel].groupby(level="time").mean()
        spread = epochs_df[channel].groupby(level="time").sem() if mode == "mean_sem" else epochs_df[channel].groupby(level="time").std()
        c = color or "C0"
        ax.fill_between(mean.index, mean - spread, mean + spread, color=c, alpha=0.25)
        ax.plot(mean.index, mean.values, color=c, lw=2.0, label=label or "mean")

    ax.axvline(0.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if title:
        ax.set_title(title)
    if label or mode == "trials":
        ax.legend(loc="best")
    return ax

