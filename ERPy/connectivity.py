"""Functional connectivity metrics for evoked and post-stimulation windows."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import signal, stats


def canonicalize_connectivity_edges(
    edges: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    directed_col: str = "directed",
    method_col: str = "method",
    undirected_methods: Iterable[str] = (
        "plv",
        "coherence",
        "mutual_information",
    ),
) -> pd.DataFrame:
    """Store undirected edge endpoints in one stable orientation.

    Directed rows, including Granger-causality edges, are left unchanged.
    Source/target anatomy and coordinate columns are swapped with their
    corresponding endpoint labels.
    """

    data = pd.DataFrame(edges).copy()
    if data.empty:
        return data
    missing = [
        col for col in (source_col, target_col) if col not in data.columns
    ]
    if missing:
        raise ValueError(f"edges is missing endpoint columns: {missing}")

    if directed_col in data.columns:
        directed = _coerce_bool_series(data[directed_col])
        undirected = ~directed
    elif method_col in data.columns:
        methods = {str(value).strip().lower() for value in undirected_methods}
        undirected = data[method_col].fillna("").astype(str).str.lower().isin(
            methods
        )
    else:
        undirected = pd.Series(True, index=data.index)

    source_key = data[source_col].fillna("").astype(str).str.casefold()
    target_key = data[target_col].fillna("").astype(str).str.casefold()
    swap = undirected & (source_key > target_key)
    endpoint_pairs = [(source_col, target_col)]
    for left, right in (
        ("source_region", "target_region"),
        ("source_mni_x", "target_mni_x"),
        ("source_mni_y", "target_mni_y"),
        ("source_mni_z", "target_mni_z"),
        ("stim_region", "record_region"),
        ("stim_elec", "record_elec"),
    ):
        if (
            left in data.columns
            and right in data.columns
            and (left, right) not in endpoint_pairs
        ):
            endpoint_pairs.append((left, right))
    for left, right in endpoint_pairs:
        left_values = data.loc[swap, left].copy()
        data.loc[swap, left] = data.loc[swap, right].to_numpy()
        data.loc[swap, right] = left_values.to_numpy()
    return data


def _coerce_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def _as_channel_list(columns: Iterable[str], channels: Iterable[str] | None = None) -> list[str]:
    available = [str(c) for c in columns]
    if channels is None:
        return available
    wanted = [str(c) for c in channels]
    return [c for c in wanted if c in available]


def _time_window_mask(index: pd.Index, window: tuple[object, object] | None) -> np.ndarray:
    if window is None:
        return np.ones(len(index), dtype=bool)
    start, stop = window
    if isinstance(index, pd.DatetimeIndex):
        return np.asarray((index >= pd.Timestamp(start)) & (index <= pd.Timestamp(stop)), dtype=bool)
    values = pd.to_numeric(pd.Series(index), errors="coerce").to_numpy(dtype=float)
    return (values >= float(start)) & (values <= float(stop))


def dataframe_time_window(df: pd.DataFrame, start: object, stop: object) -> pd.DataFrame:
    """Return rows between two numeric or datetime boundaries."""

    return df.loc[_time_window_mask(df.index, (start, stop))].copy()


def post_stim_rest_window(
    df: pd.DataFrame,
    stim_stop: object,
    *,
    start_offset_s: float = 0.0,
    duration_s: float = 60.0,
) -> pd.DataFrame:
    """Extract a continuous rest window after a stimulation train stops."""

    if isinstance(df.index, pd.DatetimeIndex):
        start = pd.Timestamp(stim_stop) + pd.to_timedelta(start_offset_s, unit="s")
        stop = start + pd.to_timedelta(duration_s, unit="s")
    else:
        start = float(stim_stop) + float(start_offset_s)
        stop = start + float(duration_s)
    return dataframe_time_window(df, start, stop)


def mutual_information_matrix_from_array(
    data: np.ndarray,
    channels: list[str],
    *,
    n_bins: int = 16,
    normalized: bool = True,
) -> pd.DataFrame:
    """Histogram mutual information matrix from ``[n_samples, n_channels]`` data."""

    data = np.asarray(data, dtype=float)
    n_ch = data.shape[1]
    mat = np.eye(n_ch) if normalized else np.zeros((n_ch, n_ch), dtype=float)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            x = data[:, i]
            y = data[:, j]
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < max(8, n_bins):
                mat[i, j] = mat[j, i] = np.nan
                continue
            mi, hx, hy = _histogram_mutual_information(x[finite], y[finite], n_bins=n_bins)
            if normalized:
                denom = np.sqrt(hx * hy)
                mi = mi / denom if denom > 0 else np.nan
            mat[i, j] = mat[j, i] = mi
    return pd.DataFrame(mat, index=channels, columns=channels)


def _histogram_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[float, float, float]:
    hist, _, _ = np.histogram2d(x, y, bins=int(n_bins))
    pxy = hist / np.sum(hist)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return mi, hx, hy


def granger_causality_matrix_from_array(
    data: np.ndarray,
    channels: list[str],
    *,
    maxlag: int = 5,
    min_samples: int | None = None,
    return_pvalues: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise linear Granger causality matrix.

    Matrix rows are source channels and columns are target channels. Values are
    ``log(RSS_restricted / RSS_full)``; larger positive values indicate that the
    source history improves prediction of the target beyond the target history.
    """

    data = np.asarray(data, dtype=float)
    n_ch = data.shape[1]
    lag = int(maxlag)
    min_samples = int(min_samples or max(30, 6 * lag))
    if (
        data.ndim == 2
        and data.shape[0] > max(min_samples, 2 * lag + 2)
        and np.isfinite(data).all()
    ):
        return _granger_complete_matrix(
            data,
            channels,
            lag=lag,
            return_pvalues=return_pvalues,
        )
    score = np.zeros((n_ch, n_ch), dtype=float)
    pvals = np.ones((n_ch, n_ch), dtype=float)
    for src in range(n_ch):
        for dst in range(n_ch):
            if src == dst:
                score[src, dst] = 0.0
                pvals[src, dst] = 1.0
                continue
            result = _pairwise_granger_score(data[:, src], data[:, dst], lag=lag, min_samples=min_samples)
            score[src, dst] = result[0]
            pvals[src, dst] = result[1]
    score_df = pd.DataFrame(score, index=channels, columns=channels)
    pval_df = pd.DataFrame(pvals, index=channels, columns=channels)
    score_df.attrs["maxlag"] = lag
    score_df.attrs["metric"] = "granger_log_rss_ratio"
    if return_pvalues:
        pval_df.attrs["maxlag"] = lag
        pval_df.attrs["metric"] = "granger_f_test_pvalue"
        return score_df, pval_df
    return score_df


def _granger_complete_matrix(
    data: np.ndarray,
    channels: list[str],
    *,
    lag: int,
    return_pvalues: bool,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Fast pairwise Granger matrix for complete continuous observations."""

    n_samples, n_ch = data.shape
    target_length = n_samples - lag
    lagged = np.empty((target_length, n_ch, lag), dtype=float)
    for offset in range(lag):
        lagged[:, :, offset] = data[lag - offset - 1 : n_samples - offset - 1, :]

    score = np.zeros((n_ch, n_ch), dtype=float)
    pvals = np.ones((n_ch, n_ch), dtype=float)
    df_num = lag
    df_den = target_length - (1 + 2 * lag)
    for dst in range(n_ch):
        target = data[lag:, dst]
        restricted = np.column_stack([np.ones(target_length), lagged[:, dst, :]])
        beta_r, *_ = np.linalg.lstsq(restricted, target, rcond=None)
        residual_y = target - restricted @ beta_r
        rss_r = float(residual_y @ residual_y)
        rank = np.linalg.matrix_rank(restricted)
        q = np.linalg.qr(restricted, mode="reduced")[0] if rank == restricted.shape[1] else None

        for src in range(n_ch):
            if src == dst:
                continue
            source_lags = lagged[:, src, :]
            if q is not None:
                source_residual = source_lags - q @ (q.T @ source_lags)
                beta_x, *_ = np.linalg.lstsq(source_residual, residual_y, rcond=None)
                residual_full = residual_y - source_residual @ beta_x
                rss_f = float(residual_full @ residual_full)
            else:
                full = np.column_stack([restricted, source_lags])
                rss_f = _rss(target, full)
            if not np.isfinite(rss_r) or not np.isfinite(rss_f) or rss_f <= 0:
                score[src, dst] = np.nan
                pvals[src, dst] = np.nan
                continue
            score[src, dst] = float(np.log(max(rss_r, 1e-300) / max(rss_f, 1e-300)))
            if df_den > 0:
                f_stat = max(((rss_r - rss_f) / df_num) / (rss_f / df_den), 0.0)
                pvals[src, dst] = float(stats.f.sf(f_stat, df_num, df_den))
            else:
                pvals[src, dst] = np.nan

    score_df = pd.DataFrame(score, index=channels, columns=channels)
    pval_df = pd.DataFrame(pvals, index=channels, columns=channels)
    score_df.attrs["maxlag"] = lag
    score_df.attrs["metric"] = "granger_log_rss_ratio"
    if return_pvalues:
        pval_df.attrs["maxlag"] = lag
        pval_df.attrs["metric"] = "granger_f_test_pvalue"
        return score_df, pval_df
    return score_df


def mutual_information_matrix_from_trials(
    data: np.ndarray,
    channels: list[str],
    *,
    n_bins: int = 16,
    normalized: bool = True,
) -> pd.DataFrame:
    """Mutual information after flattening finite samples across trials."""

    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("trial data must have shape [n_trials, n_times, n_channels]")
    return mutual_information_matrix_from_array(
        data.reshape((-1, data.shape[-1])),
        channels,
        n_bins=n_bins,
        normalized=normalized,
    )


def granger_causality_matrix_from_trials(
    data: np.ndarray,
    channels: list[str],
    *,
    maxlag: int = 5,
    min_samples: int | None = None,
    return_pvalues: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Granger matrix from trial-wise data without crossing trials."""

    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("trial data must have shape [n_trials, n_times, n_channels]")
    n_ch = data.shape[-1]
    lag = int(maxlag)
    min_samples = int(min_samples or max(30, 6 * lag))
    score = np.zeros((n_ch, n_ch), dtype=float)
    pvals = np.ones((n_ch, n_ch), dtype=float)
    for src in range(n_ch):
        for dst in range(n_ch):
            if src == dst:
                continue
            result = _pairwise_granger_trials(data[:, :, src], data[:, :, dst], lag=lag, min_samples=min_samples)
            score[src, dst] = result[0]
            pvals[src, dst] = result[1]
    score_df = pd.DataFrame(score, index=channels, columns=channels)
    pval_df = pd.DataFrame(pvals, index=channels, columns=channels)
    score_df.attrs["maxlag"] = lag
    score_df.attrs["metric"] = "granger_log_rss_ratio"
    if return_pvalues:
        pval_df.attrs["maxlag"] = lag
        pval_df.attrs["metric"] = "granger_f_test_pvalue"
        return score_df, pval_df
    return score_df


def _pairwise_granger_score(x: np.ndarray, y: np.ndarray, *, lag: int, min_samples: int) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) <= max(min_samples, 2 * lag + 2):
        return np.nan, np.nan
    target = y[lag:]
    y_lags = np.column_stack([y[lag - k - 1 : -k - 1] for k in range(lag)])
    x_lags = np.column_stack([x[lag - k - 1 : -k - 1] for k in range(lag)])
    restricted = np.column_stack([np.ones(len(target)), y_lags])
    full = np.column_stack([np.ones(len(target)), y_lags, x_lags])
    rss_r = _rss(target, restricted)
    rss_f = _rss(target, full)
    if not np.isfinite(rss_r) or not np.isfinite(rss_f) or rss_f <= 0:
        return np.nan, np.nan
    score = float(np.log(max(rss_r, 1e-300) / max(rss_f, 1e-300)))
    df_num = lag
    df_den = len(target) - full.shape[1]
    if df_den <= 0:
        return score, np.nan
    f_stat = max(((rss_r - rss_f) / df_num) / (rss_f / df_den), 0.0)
    pval = float(stats.f.sf(f_stat, df_num, df_den))
    return score, pval


def _pairwise_granger_trials(x_trials: np.ndarray, y_trials: np.ndarray, *, lag: int, min_samples: int) -> tuple[float, float]:
    targets, y_lags_all, x_lags_all = [], [], []
    for x, y in zip(np.asarray(x_trials, dtype=float), np.asarray(y_trials, dtype=float)):
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) <= 2 * lag + 2:
            continue
        targets.append(y[lag:])
        y_lags_all.append(np.column_stack([y[lag - k - 1 : -k - 1] for k in range(lag)]))
        x_lags_all.append(np.column_stack([x[lag - k - 1 : -k - 1] for k in range(lag)]))
    if not targets:
        return np.nan, np.nan
    target = np.concatenate(targets)
    y_lags = np.vstack(y_lags_all)
    x_lags = np.vstack(x_lags_all)
    if len(target) <= min_samples:
        return np.nan, np.nan
    restricted = np.column_stack([np.ones(len(target)), y_lags])
    full = np.column_stack([np.ones(len(target)), y_lags, x_lags])
    rss_r = _rss(target, restricted)
    rss_f = _rss(target, full)
    if not np.isfinite(rss_r) or not np.isfinite(rss_f) or rss_f <= 0:
        return np.nan, np.nan
    score = float(np.log(max(rss_r, 1e-300) / max(rss_f, 1e-300)))
    df_num = lag
    df_den = len(target) - full.shape[1]
    if df_den <= 0:
        return score, np.nan
    f_stat = max(((rss_r - rss_f) / df_num) / (rss_f / df_den), 0.0)
    return score, float(stats.f.sf(f_stat, df_num, df_den))


def _rss(y: np.ndarray, design: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(np.sum(resid**2))


def plv_matrix_from_dataframe(
    df: pd.DataFrame,
    sfreq: float,
    *,
    band: tuple[float, float] = (8.0, 13.0),
    channels: Iterable[str] | None = None,
    time_window: tuple[object, object] | None = None,
    order: int = 4,
) -> pd.DataFrame:
    """Estimate band-limited phase-locking value in a continuous window.

    ERPy band-pass filters each selected numeric channel, obtains its analytic
    phase with a Hilbert transform, and computes the pairwise phase-locking
    value (PLV)

    ``PLV_ij = abs(mean(exp(1j * (phase_i - phase_j))))``

    across samples in the requested interval. Unlike ``plv_matrix``, this
    function treats continuous time samples, rather than trials, as the
    averaging observations.

    Parameters
    ----------
    df:
        Samples-by-channels table. The index may contain numeric time values or
        datetimes; nonnumeric columns are ignored.
    sfreq:
        Sampling frequency in hertz.
    band:
        Inclusive lower and upper passband frequencies in hertz.
    channels:
        Channel columns to include, in output order. By default, all numeric
        columns are used.
    time_window:
        Inclusive start and stop boundaries in the same coordinate system as
        ``df.index``. The complete table is used when omitted.
    order:
        Order of the Butterworth band-pass filter applied before phase
        extraction.

    Returns
    -------
    pd.DataFrame
        Symmetric channel-by-channel PLV matrix with values from zero to one
        and a unit diagonal. A pair is ``NaN`` when no finite phase samples are
        available for that comparison.
    """

    data = df.select_dtypes(include=[np.number])
    channels = _as_channel_list(data.columns, channels)
    data = data.loc[_time_window_mask(data.index, time_window), channels].to_numpy(dtype=float)
    analytic = _band_analytic_2d(data, sfreq, band, order=order)
    n_ch = len(channels)
    magnitude = np.abs(analytic)
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_phase = analytic / magnitude
    if np.isfinite(unit_phase).all() and len(unit_phase):
        mat = np.abs(unit_phase.conj().T @ unit_phase) / float(len(unit_phase))
        np.fill_diagonal(mat, 1.0)
        return pd.DataFrame(mat.real, index=channels, columns=channels)

    phase = np.angle(analytic)
    mat = np.eye(n_ch)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            finite = np.isfinite(phase[:, i]) & np.isfinite(phase[:, j])
            if not finite.any():
                mat[i, j] = mat[j, i] = np.nan
                continue
            mat[i, j] = mat[j, i] = float(np.abs(np.mean(np.exp(1j * (phase[finite, i] - phase[finite, j])))))
    return pd.DataFrame(mat, index=channels, columns=channels)


def coherence_matrix_from_dataframe(
    df: pd.DataFrame,
    sfreq: float,
    *,
    band: tuple[float, float] = (8.0, 13.0),
    channels: Iterable[str] | None = None,
    time_window: tuple[object, object] | None = None,
    mode: str = "coherence",
    nperseg: int | None = None,
) -> pd.DataFrame:
    """Estimate band-averaged coherence in a continuous multichannel window.

    For ``mode="coherence"``, ERPy estimates magnitude-squared coherence,

    ``C_ij(f) = abs(S_ij(f))**2 / (S_ii(f) * S_jj(f))``,

    from Welch cross-spectral densities and averages it across frequencies in
    ``band``. For ``mode="imaginary"``, the returned quantity is the band mean
    of the absolute imaginary part of coherency, which reduces contributions
    with zero phase lag. Both modes summarize continuous samples; use
    ``coherence_matrix`` when trials provide the averaging ensemble.

    Parameters
    ----------
    df:
        Samples-by-channels table. The index may contain numeric time values or
        datetimes; nonnumeric columns are ignored.
    sfreq:
        Sampling frequency in hertz.
    band:
        Inclusive lower and upper frequency bounds in hertz.
    channels:
        Channel columns to include, in output order. By default, all numeric
        columns are used.
    time_window:
        Inclusive start and stop boundaries in the same coordinate system as
        ``df.index``. The complete table is used when omitted.
    mode:
        ``"coherence"`` for magnitude-squared coherence or ``"imaginary"``
        for absolute imaginary coherency.
    nperseg:
        Welch segment length in samples. ERPy chooses a length compatible with
        the available window when omitted.

    Returns
    -------
    pd.DataFrame
        Symmetric channel-by-channel matrix. Magnitude-squared coherence has a
        unit diagonal; imaginary coherency has a zero diagonal. Pairwise values
        are ``NaN`` when fewer than eight joint finite samples are available.
    """

    data = df.select_dtypes(include=[np.number])
    channels = _as_channel_list(data.columns, channels)
    data = data.loc[_time_window_mask(data.index, time_window), channels]
    values = data.to_numpy(dtype=float)
    if mode != "imaginary" and np.isfinite(values).all():
        mat = _welch_coherence_matrix(values, sfreq, band=band, nperseg=nperseg)
        return pd.DataFrame(mat, index=channels, columns=channels)

    n_ch = len(channels)
    mat = np.eye(n_ch) if mode != "imaginary" else np.zeros((n_ch, n_ch), dtype=float)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            x = data.iloc[:, i].to_numpy(dtype=float)
            y = data.iloc[:, j].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 8:
                mat[i, j] = mat[j, i] = np.nan
                continue
            if mode == "imaginary":
                value = _imaginary_coherence_continuous(x[finite], y[finite], sfreq, band=band, nperseg=nperseg)
            else:
                freqs, coh = signal.coherence(x[finite], y[finite], fs=sfreq, nperseg=nperseg)
                fmask = (freqs >= band[0]) & (freqs <= band[1])
                value = float(np.nanmean(coh[fmask])) if fmask.any() else np.nan
            mat[i, j] = mat[j, i] = value
    return pd.DataFrame(mat, index=channels, columns=channels)


def multitaper_coherence_matrices_from_dataframe(
    df: pd.DataFrame,
    sfreq: float,
    *,
    band: tuple[float, float] = (8.0, 13.0),
    channels: Iterable[str] | None = None,
    time_window: tuple[object, object] | None = None,
    time_bandwidth: float = 3.0,
    n_tapers: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multitaper coherence and absolute imaginary-coherency matrices.

    Both matrices use one shared DPSS decomposition. This is especially useful
    for short matched windows where Welch segmentation would leave few stable
    alpha-band estimates. Imaginary coherency is returned as a prespecified
    sensitivity measure for zero-lag coupling.
    """

    data = df.select_dtypes(include=[np.number])
    selected_channels = _as_channel_list(data.columns, channels)
    data = data.loc[
        _time_window_mask(data.index, time_window), selected_channels
    ]
    values = data.to_numpy(dtype=float)
    coherence, imaginary = _multitaper_coherence_matrices(
        values,
        sfreq,
        band=band,
        time_bandwidth=time_bandwidth,
        n_tapers=n_tapers,
    )
    return (
        pd.DataFrame(
            coherence, index=selected_channels, columns=selected_channels
        ),
        pd.DataFrame(
            imaginary, index=selected_channels, columns=selected_channels
        ),
    )


def _multitaper_coherence_matrices(
    data: np.ndarray,
    sfreq: float,
    *,
    band: tuple[float, float],
    time_bandwidth: float,
    n_tapers: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 16 or values.shape[1] < 2:
        raise ValueError("multitaper coherence requires samples x >=2 channels")
    if not np.isfinite(values).all():
        raise ValueError("multitaper coherence requires finite values")
    values = signal.detrend(values, axis=0, type="linear")
    tapers, concentrations = signal.windows.dpss(
        values.shape[0],
        float(time_bandwidth),
        Kmax=int(n_tapers),
        sym=False,
        norm=2,
        return_ratios=True,
    )
    spectra = np.fft.rfft(tapers[:, :, None] * values[None, :, :], axis=1)
    frequencies = np.fft.rfftfreq(values.shape[0], d=1.0 / float(sfreq))
    selected = (frequencies >= float(band[0])) & (
        frequencies <= float(band[1])
    )
    if not selected.any():
        raise ValueError("no Fourier bins fall inside the requested band")
    spectra = spectra[:, selected, :]
    weights = np.asarray(concentrations, dtype=float)
    weights = weights / weights.sum()
    coherence_by_frequency = []
    imaginary_by_frequency = []
    for frequency_index in range(spectra.shape[1]):
        frequency_spectra = spectra[:, frequency_index, :]
        cross_spectrum = np.einsum(
            "k,ki,kj->ij",
            weights,
            frequency_spectra.conj(),
            frequency_spectra,
            optimize=True,
        )
        power = np.maximum(np.real(np.diag(cross_spectrum)), 0.0)
        denominator = np.sqrt(power[:, None] * power[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            coherency = np.divide(
                cross_spectrum,
                denominator,
                out=np.full_like(cross_spectrum, np.nan, dtype=complex),
                where=denominator > 1e-15,
            )
        coherence_by_frequency.append(np.abs(coherency) ** 2)
        imaginary_by_frequency.append(np.abs(np.imag(coherency)))
    coherence = np.nanmean(np.stack(coherence_by_frequency), axis=0).real
    imaginary = np.nanmean(np.stack(imaginary_by_frequency), axis=0).real
    coherence = np.clip(coherence, 0.0, 1.0)
    imaginary = np.clip(imaginary, 0.0, 1.0)
    np.fill_diagonal(coherence, 1.0)
    np.fill_diagonal(imaginary, 0.0)
    return coherence, imaginary


def _welch_coherence_matrix(
    data: np.ndarray,
    sfreq: float,
    *,
    band: tuple[float, float],
    nperseg: int | None,
) -> np.ndarray:
    """Compute every channel pair from one shared Welch decomposition."""

    data = np.asarray(data, dtype=float)
    n_samples, n_ch = data.shape
    segment_length = min(int(nperseg or 256), n_samples)
    if segment_length < 8:
        return np.full((n_ch, n_ch), np.nan, dtype=float)
    step = max(segment_length // 2, 1)
    segments = np.lib.stride_tricks.sliding_window_view(
        data,
        window_shape=segment_length,
        axis=0,
    )[::step]
    segments = np.transpose(segments, (0, 2, 1))
    segments = segments - segments.mean(axis=1, keepdims=True)
    window = signal.windows.hann(segment_length, sym=False)
    spectra = np.fft.rfft(segments * window[None, :, None], axis=1)
    spectral_matrix = np.einsum(
        "sfi,sfj->fij",
        spectra.conj(),
        spectra,
        optimize=True,
    ) / max(len(spectra), 1)
    power = np.real(np.diagonal(spectral_matrix, axis1=1, axis2=2))
    denominator = power[:, :, None] * power[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.abs(spectral_matrix) ** 2 / denominator
    frequencies = np.fft.rfftfreq(segment_length, d=1.0 / float(sfreq))
    selected = (frequencies >= float(band[0])) & (frequencies <= float(band[1]))
    if not selected.any():
        return np.full((n_ch, n_ch), np.nan, dtype=float)
    matrix = np.nanmean(coherence[selected], axis=0).real
    matrix = np.clip(matrix, 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _imaginary_coherence_continuous(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    *,
    band: tuple[float, float],
    nperseg: int | None,
) -> float:
    nperseg = nperseg or min(1024, len(x))
    freqs, pxy = signal.csd(x, y, fs=sfreq, nperseg=nperseg)
    _, pxx = signal.welch(x, fs=sfreq, nperseg=nperseg)
    _, pyy = signal.welch(y, fs=sfreq, nperseg=nperseg)
    denom = np.sqrt(pxx * pyy)
    with np.errstate(divide="ignore", invalid="ignore"):
        coherency = pxy / denom
    fmask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.nanmean(np.abs(np.imag(coherency[fmask])))) if fmask.any() else np.nan


def mutual_information_matrix_from_dataframe(
    df: pd.DataFrame,
    *,
    channels: Iterable[str] | None = None,
    time_window: tuple[object, object] | None = None,
    n_bins: int = 16,
    normalized: bool = True,
) -> pd.DataFrame:
    """Estimate pairwise mutual information in a continuous data window.

    Numeric columns are discretized into ``n_bins`` before estimating the
    symmetric channel matrix. Normalized values range from zero to one.
    """

    data = df.select_dtypes(include=[np.number])
    channels = _as_channel_list(data.columns, channels)
    arr = data.loc[_time_window_mask(data.index, time_window), channels].to_numpy(dtype=float)
    return mutual_information_matrix_from_array(arr, channels, n_bins=n_bins, normalized=normalized)


def granger_causality_matrix_from_dataframe(
    df: pd.DataFrame,
    *,
    channels: Iterable[str] | None = None,
    time_window: tuple[object, object] | None = None,
    maxlag: int = 5,
    return_pvalues: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate directed pairwise Granger causality in a data window.

    Matrix rows are putative source channels and columns are targets. When
    ``return_pvalues`` is true, the F-statistic and p-value matrices are
    returned as a pair.
    """

    data = df.select_dtypes(include=[np.number])
    channels = _as_channel_list(data.columns, channels)
    arr = data.loc[_time_window_mask(data.index, time_window), channels].to_numpy(dtype=float)
    return granger_causality_matrix_from_array(arr, channels, maxlag=maxlag, return_pvalues=return_pvalues)


def _band_analytic_2d(data: np.ndarray, sfreq: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    low = max(float(band[0]), 1e-6)
    high = min(float(band[1]), 0.49 * float(sfreq))
    if low >= high:
        raise ValueError(f"Invalid spectral band {band} for sfreq={sfreq}")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    filt = signal.sosfiltfilt(sos, np.nan_to_num(data), axis=0)
    return signal.hilbert(filt, axis=0)


def connectivity_matrix_from_dataframe(
    df: pd.DataFrame,
    sfreq: float,
    *,
    method: str,
    channels: Iterable[str] | None = None,
    band: tuple[float, float] = (8.0, 13.0),
    time_window: tuple[object, object] | None = None,
    **kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch common continuous-window connectivity metrics."""

    method = method.lower().replace("-", "_")
    if method == "plv":
        return plv_matrix_from_dataframe(df, sfreq, band=band, channels=channels, time_window=time_window, **kwargs)
    if method in {"coherence", "imaginary_coherence", "imaginary"}:
        mode = "imaginary" if method in {"imaginary_coherence", "imaginary"} else kwargs.pop("mode", "coherence")
        return coherence_matrix_from_dataframe(df, sfreq, band=band, channels=channels, time_window=time_window, mode=mode, **kwargs)
    if method in {"mutual_information", "mi"}:
        return mutual_information_matrix_from_dataframe(df, channels=channels, time_window=time_window, **kwargs)
    if method in {"granger", "granger_causality"}:
        return granger_causality_matrix_from_dataframe(df, channels=channels, time_window=time_window, **kwargs)
    raise ValueError("method must be one of plv, coherence, imaginary_coherence, mutual_information, or granger")
