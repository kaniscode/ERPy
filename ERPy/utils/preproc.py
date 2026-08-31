from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .utils import infer_sfreq


def _numeric_values(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric = df.select_dtypes(include=[np.number])
    return numeric.to_numpy(dtype=float), list(numeric.columns)


def common_reference(
    df: pd.DataFrame,
    *,
    reference_channels: list[str] | None = None,
    method: str = "mean",
    zscore_threshold: float = 8.0,
    min_reference_channels: int = 4,
) -> pd.DataFrame:
    """Subtract a common reference without changing the channel set.

    ``robust_mean`` computes a sample-wise mean after excluding contacts that
    are extreme relative to the cross-contact median and MAD. This prevents a
    transiently saturated contact from being copied into every channel by the
    reference while retaining common-average behavior for the remaining
    contacts.
    """

    numeric = df.select_dtypes(include=[np.number])
    refs = list(reference_channels) if reference_channels is not None else list(numeric.columns)
    refs = [channel for channel in refs if channel in numeric.columns]
    if not refs:
        out = df.copy()
        out.attrs.update(df.attrs)
        return out

    values = numeric[refs].to_numpy(dtype=float)
    method_key = str(method).strip().lower().replace("-", "_")
    if method_key in {"mean", "average", "common_average"}:
        reference = np.nanmean(values, axis=1)
    elif method_key in {"median", "common_median"}:
        reference = np.nanmedian(values, axis=1)
    elif method_key in {"robust_mean", "robust_average"}:
        center = np.nanmedian(values, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(values - center), axis=1, keepdims=True)
        scale = 1.4826 * mad
        finite = np.isfinite(values)
        nonzero_scale = np.isfinite(scale) & (scale > np.finfo(float).eps)
        within = np.abs(values - center) <= float(zscore_threshold) * scale
        eligible = finite & np.where(nonzero_scale, within, True)
        enough = eligible.sum(axis=1) >= min(max(int(min_reference_channels), 1), len(refs))
        robust_reference = np.nanmean(np.where(eligible, values, np.nan), axis=1)
        fallback = np.nanmedian(values, axis=1)
        reference = np.where(enough & np.isfinite(robust_reference), robust_reference, fallback)
    else:
        raise ValueError("method must be 'mean', 'median', or 'robust_mean'")

    out = df.copy()
    out.loc[:, numeric.columns] = numeric.sub(pd.Series(reference, index=df.index), axis=0)
    out.attrs.update(df.attrs)
    out.attrs["reference_method"] = method_key
    out.attrs["reference_channels"] = refs
    if method_key == "robust_mean":
        out.attrs["reference_zscore_threshold"] = float(zscore_threshold)
        out.attrs["reference_min_channels"] = int(min_reference_channels)
    return out


def remove_drift(df: pd.DataFrame, fs: float | None = None, cutoff: float = 0.01) -> pd.DataFrame:
    fs = fs or infer_sfreq(df)
    if cutoff <= 0:
        return df.copy()
    sos = signal.butter(2, cutoff, btype="highpass", fs=fs, output="sos")
    out = df.copy()
    values, cols = _numeric_values(df)
    if len(df) > 12:
        out.loc[:, cols] = signal.sosfiltfilt(sos, values, axis=0)
    return out


def bandpass(df: pd.DataFrame, lowcut: float, highcut: float, fs: float | None = None, order: int = 4) -> pd.DataFrame:
    fs = fs or infer_sfreq(df)
    nyq = fs / 2.0
    low = max(float(lowcut), 1e-6)
    high = min(float(highcut), nyq * 0.99)
    if low >= high:
        raise ValueError(f"Invalid bandpass range {lowcut}-{highcut} Hz for fs={fs}")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    out = df.copy()
    values, cols = _numeric_values(df)
    if len(df) > max(24, order * 6):
        out.loc[:, cols] = signal.sosfiltfilt(sos, values, axis=0)
    return out


def notch_line(
    df: pd.DataFrame,
    notch_freq: float = 60.0,
    fs: float | None = None,
    bw: float = 2.0,
    harm: bool = True,
) -> pd.DataFrame:
    fs = fs or infer_sfreq(df)
    out = df.copy()
    values, cols = _numeric_values(out)
    if len(df) < 24:
        return out
    fundamental = float(notch_freq)
    upper = float(fs) / 2.0 * 0.98
    freqs = []
    frequency = fundamental
    while frequency < upper:
        freqs.append(frequency)
        if not harm:
            break
        frequency += fundamental
    if not freqs:
        return out
    filt = values
    for freq in freqs:
        q = max(freq / max(float(bw), 1e-6), 1.0)
        b, a = signal.iirnotch(freq, q, fs=fs)
        filt = signal.filtfilt(b, a, filt, axis=0)
    out.loc[:, cols] = filt
    return out


def decimate(df: pd.DataFrame, fs_new: float, fs: float | None = None) -> pd.DataFrame:
    fs = fs or infer_sfreq(df)
    if fs_new >= fs:
        out = df.copy()
        out.attrs["sfreq"] = fs
        return out
    q = max(int(round(fs / fs_new)), 1)
    values, cols = _numeric_values(df)
    dec = signal.decimate(values, q, axis=0, zero_phase=True)
    index = df.index[::q][: len(dec)]
    out = pd.DataFrame(dec, index=index, columns=cols)
    for col in df.columns:
        if col not in cols:
            out[col] = df[col].iloc[::q].to_numpy()[: len(out)]
    out = out[df.columns]
    out.attrs["sfreq"] = fs / q
    return out


def artifact_blank(
    df: pd.DataFrame,
    event_times,
    fs: float | None = None,
    width_s: float = 0.003,
    pre_s: float | None = None,
    post_s: float | None = None,
) -> pd.DataFrame:
    """Linearly interpolate across stimulation artifacts."""

    if event_times is None or len(event_times) == 0:
        return df.copy()
    fs = fs or infer_sfreq(df)
    pre_s = width_s if pre_s is None else pre_s
    post_s = width_s if post_s is None else post_s
    pre = max(int(round(pre_s * fs)), 1)
    post = max(int(round(post_s * fs)), 1)
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns

    if isinstance(out.index, pd.DatetimeIndex):
        events = pd.to_datetime(event_times)
        positions = out.index.searchsorted(events)
    else:
        idx = pd.to_numeric(pd.Series(out.index), errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(pd.Series(event_times), errors="coerce").to_numpy(dtype=float)
        positions = np.searchsorted(idx, events)

    for pos in positions:
        start = max(int(pos) - pre, 0)
        stop = min(int(pos) + post + 1, len(out))
        left = max(start - 1, 0)
        right = min(stop, len(out) - 1)
        if right <= left:
            continue
        x = np.array([left, right], dtype=float)
        xi = np.arange(start, stop, dtype=float)
        for col in numeric_cols:
            y = out[col].iloc[[left, right]].to_numpy(dtype=float)
            out.iloc[start:stop, out.columns.get_loc(col)] = np.interp(xi, x, y)
    out.attrs.update(df.attrs)
    return out
