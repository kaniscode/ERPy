"""Visualize metrics against stimulation order."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .theme import apply_erpy_theme


def plot_metric_vs_epoch(
    epochs_df: pd.DataFrame,
    channel: str,
    metric_fn: Optional[Callable[[np.ndarray], float]] = None,
    ax=None,
    rolling: int = 0,
    title: Optional[str] = None,
):
    """Plot a per-epoch scalar metric in stimulation order."""

    import matplotlib.pyplot as plt

    apply_erpy_theme()
    if metric_fn is None:

        def metric_fn(waveform: np.ndarray) -> float:
            return float(np.sqrt(np.nanmean(np.asarray(waveform, dtype=float) ** 2)))

    epochs = np.sort(epochs_df.index.get_level_values("epoch").unique())
    values = []
    for epoch in epochs:
        series = epochs_df.xs(epoch, level="epoch")[channel]
        times = series.index.to_numpy() if hasattr(series.index, "to_numpy") else np.asarray(series.index)
        waveform = series.to_numpy(dtype=float)
        mask = times > 0
        values.append(metric_fn(waveform[mask]) if np.any(mask) else metric_fn(waveform))

    ax = ax or plt.subplots(figsize=(7, 3.5), layout="constrained")[1]
    x = np.arange(len(epochs))
    ax.plot(x, values, "o-", ms=3, lw=1.0, color="C0")
    if rolling and rolling > 1 and len(values) >= rolling:
        smoothed = pd.Series(values).rolling(rolling, min_periods=1).mean()
        ax.plot(x, smoothed.values, color="C3", lw=2.0, label=f"rolling {rolling}")
        ax.legend()
    ax.set_xlabel("Epoch order")
    ax.set_ylabel("Metric")
    ax.set_title(title or f"{channel}: per-epoch metric")
    return ax

