"""Shared baseline-referencing utilities for epoched signals."""

from __future__ import annotations

from typing import Iterable
import warnings

import numpy as np
import pandas as pd


def baseline_time_mask(
    times: Iterable[float],
    baseline_window: tuple[float, float] | None,
) -> np.ndarray:
    """Return the requested baseline mask, falling back to prestimulus data."""

    values = np.asarray(times, dtype=float)
    if baseline_window is None:
        mask = values < 0.0
    else:
        start, stop = map(float, baseline_window)
        if stop < start:
            start, stop = stop, start
        mask = (values >= start) & (values <= stop)
        if not mask.any():
            mask = values < 0.0
    if not mask.any():
        raise ValueError("No baseline samples overlap the epoch times")
    return mask


def baseline_center_array(
    values: np.ndarray,
    times: Iterable[float],
    baseline_window: tuple[float, float] | None,
) -> np.ndarray:
    """Subtract each trial-channel baseline mean from a 3D epoch array."""

    array = np.asarray(values, dtype=float)
    time_values = np.asarray(times, dtype=float)
    if array.ndim != 3:
        raise ValueError("values must have shape (trials, times, channels)")
    if array.shape[1] != len(time_values):
        raise ValueError("The array time axis does not match times")
    mask = baseline_time_mask(time_values, baseline_window)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        offsets = np.nanmean(array[:, mask, :], axis=1, keepdims=True)
    return array - offsets


def baseline_center_frame(
    frame: pd.DataFrame,
    baseline_window: tuple[float, float] | None,
) -> pd.DataFrame:
    """Return a time-indexed waveform frame referenced to its baseline."""

    output = pd.DataFrame(frame).copy()
    times = pd.to_numeric(
        pd.Series(output.index, index=output.index),
        errors="coerce",
    ).to_numpy(dtype=float)
    mask = baseline_time_mask(times, baseline_window)
    offsets = output.iloc[np.flatnonzero(mask)].mean(axis=0)
    return output.subtract(offsets, axis="columns")
