"""Spectral analysis for stimulation-evoked intracranial responses.

This module provides standard time-frequency, spectral-power, and
spectral-connectivity tools used in EEG/iEEG pipelines, computed either per
trial or on the trial-mean response waveform. Everything is built on
``numpy``/``scipy`` only (no MNE dependency):

* power spectral density (Welch), per trial or on the mean waveform;
* complex Morlet time-frequency representations (TFR), power and phase;
* event-related spectral perturbation (ERSP, baseline-normalized power);
* inter-trial phase coherence / clustering (ITPC, "phase locking to stimulus");
* band power summaries;
* phase-locking value (PLV) between contacts, time-frequency resolved and as a
  band/time-window connectivity matrix;
* magnitude-squared and imaginary coherence connectivity matrices;
* phase-amplitude coupling (PAC) via the Tort modulation index or the Canolty
  mean-vector-length, and PAC comodulograms;
* n:m phase-phase cross-frequency coupling.

The :class:`Epochs` object exposes these through an ``epochs.spectral`` accessor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import signal

from .connectivity import granger_causality_matrix_from_trials, mutual_information_matrix_from_trials


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
    "high_gamma": (80.0, 150.0),
}


@dataclass
class SpectralConfig:
    """Settings for Morlet time-frequency analysis."""

    fmin: float = 4.0
    fmax: float = 150.0
    n_freqs: int = 40
    spacing: str = "log"  # "log" or "linear"
    n_cycles: float | str = 7.0  # scalar, per-frequency array, or "adaptive" (freqs/2)
    baseline: tuple[float, float] | None = (-0.5, -0.05)
    baseline_mode: str = "logratio"  # logratio, db, ratio, percent, zscore, mean
    decim: int = 1

    def frequencies(self, sfreq: float | None = None) -> np.ndarray:
        """Return configured frequencies, capped below the Nyquist limit."""

        fmax = float(self.fmax)
        if sfreq is not None:
            fmax = min(fmax, 0.45 * float(sfreq))
        return make_frequencies(self.fmin, fmax, self.n_freqs, spacing=self.spacing)


@dataclass
class TFRResult:
    """Time-frequency result for one channel.

    ``power`` and ``itc`` are frequency-by-time arrays. ``power`` is the
    trial-averaged quantity after the recorded baseline transform, while
    ``power_per_trial`` optionally retains untransformed trial-level power as
    a trial-by-frequency-by-time array.
    """

    channel: str
    freqs: np.ndarray
    times: np.ndarray
    power: np.ndarray  # [n_freqs, n_times] trial-averaged, baseline-normalized when baseline set
    itc: np.ndarray  # [n_freqs, n_times] inter-trial phase coherence
    n_trials: int
    baseline: tuple[float, float] | None = None
    baseline_mode: str | None = None
    power_per_trial: np.ndarray | None = None  # [n_trials, n_freqs, n_times] raw power

    @property
    def ersp(self) -> np.ndarray:
        """Event-related spectral perturbation (baseline-normalized power)."""

        return self.power


@dataclass
class PLVResult:
    """Time-frequency phase-locking value between two channels.

    ``plv`` is a frequency-by-time array bounded from zero to one and computed
    across the retained trials. ``freqs`` are in hertz and ``times`` are epoch-
    relative seconds.
    """

    ch_x: str
    ch_y: str
    freqs: np.ndarray
    times: np.ndarray
    plv: np.ndarray  # [n_freqs, n_times]
    n_trials: int


@dataclass
class PACResult:
    """Phase-amplitude coupling summary for one channel pair and band pair.

    The result retains the coupling method and scalar modulation index together
    with the phase-bin centers, mean amplitude profile, preferred phase, and
    number of finite samples used in the estimate.
    """

    phase_channel: str
    amp_channel: str
    phase_band: tuple[float, float]
    amp_band: tuple[float, float]
    method: str
    mi: float
    phase_bin_centers: np.ndarray
    amplitude_by_phase: np.ndarray
    preferred_phase: float = np.nan
    n_samples: int = 0


@dataclass
class ComodulogramResult:
    """Phase-amplitude coupling values across frequency-band centers.

    ``mi`` is arranged as amplitude-frequency by phase-frequency. Frequency
    centers and bandwidths are expressed in hertz, and ``method`` records the
    modulation-index estimator.
    """

    phase_channel: str
    amp_channel: str
    phase_freqs: np.ndarray  # band centers
    amp_freqs: np.ndarray  # band centers
    mi: np.ndarray  # [n_amp, n_phase]
    method: str
    bandwidth_phase: float = field(default=2.0)
    bandwidth_amp: float = field(default=20.0)


# ---------------------------------------------------------------------------
# Low-level signal helpers
# ---------------------------------------------------------------------------

def make_frequencies(fmin: float, fmax: float, n_freqs: int, spacing: str = "log") -> np.ndarray:
    fmin = max(float(fmin), 1e-6)
    fmax = max(float(fmax), fmin * 1.001)
    n_freqs = max(int(n_freqs), 1)
    if spacing == "linear":
        return np.linspace(fmin, fmax, n_freqs)
    return np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)


def _resolve_n_cycles(n_cycles: float | str | Iterable[float], freqs: np.ndarray) -> np.ndarray:
    if isinstance(n_cycles, str):
        if n_cycles in {"adaptive", "auto"}:
            return np.maximum(freqs / 2.0, 1.0)
        raise ValueError(f"Unknown n_cycles spec {n_cycles!r}")
    arr = np.asarray(n_cycles, dtype=float)
    if arr.ndim == 0:
        return np.full(len(freqs), float(arr))
    if arr.shape[0] != len(freqs):
        raise ValueError("n_cycles array must match the number of frequencies")
    return arr


def morlet_wavelet(freq: float, sfreq: float, n_cycles: float, n_times: int | None = None, sigma_factor: float = 5.0) -> np.ndarray:
    """Return a zero-mean, unit-energy complex Morlet wavelet."""

    freq = float(freq)
    sigma_t = n_cycles / (2.0 * np.pi * freq)
    half = int(np.ceil(sigma_factor * sigma_t * sfreq))
    if n_times is not None:
        half = min(half, max((int(n_times) - 1) // 2, 1))
    half = max(half, 1)
    t = np.arange(-half, half + 1) / sfreq
    gauss = np.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
    wavelet = np.exp(2j * np.pi * freq * t) * gauss
    wavelet = wavelet - wavelet.mean()  # remove DC for admissibility
    norm = np.sqrt(np.sum(np.abs(wavelet) ** 2))
    if norm > 0:
        wavelet = wavelet / norm
    return wavelet


def cwt_morlet(x: np.ndarray, sfreq: float, freqs: np.ndarray, n_cycles: float | str = 7.0) -> np.ndarray:
    """Complex Morlet continuous wavelet transform along the last axis.

    ``x`` may be ``[n_times]`` or ``[n_trials, n_times]``; the return has an
    inserted frequency axis: ``[..., n_freqs, n_times]`` (complex).
    """

    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[np.newaxis, :]
    n_rows, n_times = x.shape
    freqs = np.asarray(freqs, dtype=float)
    n_cycles_arr = _resolve_n_cycles(n_cycles, freqs)
    out = np.empty((n_rows, len(freqs), n_times), dtype=complex)
    for fi, (f, nc) in enumerate(zip(freqs, n_cycles_arr)):
        wavelet = morlet_wavelet(f, sfreq, nc, n_times=n_times)
        conv = signal.fftconvolve(x, wavelet[np.newaxis, :], mode="same", axes=1)
        out[:, fi, :] = conv
    if squeeze:
        return out[0]
    return out


def _band_analytic(x: np.ndarray, sfreq: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    """Bandpass-filter then Hilbert-transform along the last axis -> analytic signal."""

    low = max(float(band[0]), 1e-6)
    high = min(float(band[1]), 0.49 * float(sfreq))
    if low >= high:
        raise ValueError(f"Invalid spectral band {band} for sfreq={sfreq}")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    filt = signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)
    return signal.hilbert(filt, axis=-1)


def _time_mask(times: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    if window is None:
        return np.ones(times.shape, dtype=bool)
    return (times >= float(window[0])) & (times <= float(window[1]))


def _channel_matrix(epochs, channel: str) -> tuple[np.ndarray, np.ndarray]:
    arr, times, channels = epochs.as_array()
    if channel not in channels:
        raise KeyError(channel)
    return arr[:, :, channels.index(channel)], times


def _finite_trials(mat: np.ndarray) -> np.ndarray:
    """Keep only trials (axis 0) that are fully finite."""

    finite = np.all(np.isfinite(mat), axis=tuple(range(1, mat.ndim)))
    return mat[finite]


def _apply_baseline(power: np.ndarray, times: np.ndarray, baseline: tuple[float, float] | None, mode: str) -> np.ndarray:
    if baseline is None:
        return power
    bmask = _time_mask(times, baseline)
    if not bmask.any():
        return power
    base = np.nanmean(power[..., bmask], axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "db":
            out = 10.0 * np.log10(power / base)
        elif mode == "logratio":
            out = np.log10(power / base)
        elif mode == "ratio":
            out = power / base
        elif mode == "percent":
            out = 100.0 * (power - base) / base
        elif mode in {"zscore", "zlogratio"}:
            bstd = np.nanstd(power[..., bmask], axis=-1, keepdims=True)
            out = (power - base) / bstd
        elif mode in {"mean", "subtract"}:
            out = power - base
        else:
            raise ValueError(f"Unknown baseline_mode {mode!r}")
    return out


# ---------------------------------------------------------------------------
# Power spectral density
# ---------------------------------------------------------------------------

def compute_psd(
    epochs,
    channels: Iterable[str] | str | None = None,
    *,
    fmin: float = 1.0,
    fmax: float | None = None,
    on: str = "trials",
    nperseg: int | None = None,
    detrend: str = "constant",
    relative: bool = False,
) -> pd.DataFrame:
    """Welch power spectral density per channel.

    ``on="trials"`` computes the PSD per trial and averages across trials (the
    standard estimate for evoked data); ``on="mean"`` computes the PSD of the
    trial-mean response waveform. Returns a DataFrame indexed by frequency with
    one column per channel.
    """

    arr, _, all_ch = epochs.as_array()
    sfreq = float(epochs.sfreq)
    if channels is None:
        channels = all_ch
    elif isinstance(channels, str):
        channels = [channels]
    channels = [c for c in channels if c in all_ch]
    idx = [all_ch.index(c) for c in channels]
    n_times = arr.shape[1]
    if nperseg is None:
        nperseg = min(n_times, 256) if n_times >= 32 else n_times
    nperseg = max(int(nperseg), 8)

    psd_cols: dict[str, np.ndarray] = {}
    freqs_ref: np.ndarray | None = None
    for col, ci in zip(channels, idx):
        if on == "mean":
            sig = np.nanmean(arr[:, :, ci], axis=0)
            freqs, pxx = signal.welch(sig, fs=sfreq, nperseg=min(nperseg, len(sig)), detrend=detrend)
        else:
            trials = _finite_trials(arr[:, :, ci])
            if trials.shape[0] == 0:
                trials = np.nan_to_num(arr[:, :, ci])
            spectra = []
            for trial in trials:
                freqs, pxx = signal.welch(trial, fs=sfreq, nperseg=min(nperseg, len(trial)), detrend=detrend)
                spectra.append(pxx)
            pxx = np.nanmean(np.vstack(spectra), axis=0)
        freqs_ref = freqs if freqs_ref is None else freqs_ref
        psd_cols[col] = pxx

    if freqs_ref is None:
        return pd.DataFrame()
    fmax = fmax if fmax is not None else sfreq / 2.0
    fmask = (freqs_ref >= float(fmin)) & (freqs_ref <= float(fmax))
    out = pd.DataFrame({c: v[fmask] for c, v in psd_cols.items()}, index=np.round(freqs_ref[fmask], 6))
    out.index.name = "frequency_hz"
    if relative:
        totals = out.sum(axis=0).replace(0.0, np.nan)
        out = out.divide(totals, axis=1)
    out.attrs["sfreq"] = sfreq
    out.attrs["on"] = on
    return out


def band_power(
    epochs,
    bands: Mapping[str, tuple[float, float]] | None = None,
    channels: Iterable[str] | None = None,
    *,
    on: str = "trials",
    relative: bool = False,
) -> pd.DataFrame:
    """Average PSD power within named frequency bands -> channels x bands DataFrame."""

    bands = dict(bands or DEFAULT_BANDS)
    psd = compute_psd(epochs, channels=channels, fmin=min(b[0] for b in bands.values()),
                      fmax=max(b[1] for b in bands.values()), on=on, relative=relative)
    if psd.empty:
        return pd.DataFrame()
    freqs = psd.index.to_numpy(dtype=float)
    rows = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        rows[name] = psd.loc[mask].mean(axis=0) if mask.any() else pd.Series(np.nan, index=psd.columns)
    out = pd.DataFrame(rows)
    out.index.name = "channel"
    return out


# ---------------------------------------------------------------------------
# Time-frequency: power, ERSP, ITPC
# ---------------------------------------------------------------------------

def compute_tfr(
    epochs,
    channel: str,
    config: SpectralConfig | None = None,
    *,
    return_per_trial: bool = False,
) -> TFRResult:
    """Morlet time-frequency power and inter-trial phase coherence for one channel.

    When ``config.baseline`` is set, ``power`` is the baseline-normalized
    event-related spectral perturbation (ERSP). ``itc`` is the inter-trial phase
    coherence (a.k.a. inter-trial phase clustering / phase locking to stimulus).
    """

    config = config or SpectralConfig()
    x, times = _channel_matrix(epochs, channel)
    sfreq = float(epochs.sfreq)
    trials = _finite_trials(x)
    if trials.shape[0] == 0:
        trials = np.nan_to_num(x)
    freqs = config.frequencies(sfreq)
    coefs = cwt_morlet(trials, sfreq, freqs, n_cycles=config.n_cycles)  # [n_trials, n_freqs, n_times]
    power_per_trial = np.abs(coefs) ** 2
    power = power_per_trial.mean(axis=0)  # [n_freqs, n_times]

    with np.errstate(invalid="ignore", divide="ignore"):
        unit_phase = coefs / np.abs(coefs)
    unit_phase[~np.isfinite(unit_phase)] = 0.0
    itc = np.abs(unit_phase.mean(axis=0))  # [n_freqs, n_times]

    power = _apply_baseline(power, times, config.baseline, config.baseline_mode)

    decim = max(int(config.decim), 1)
    if decim > 1:
        times = times[::decim]
        power = power[:, ::decim]
        itc = itc[:, ::decim]
        if return_per_trial:
            power_per_trial = power_per_trial[:, :, ::decim]

    return TFRResult(
        channel=channel,
        freqs=freqs,
        times=times,
        power=power,
        itc=itc,
        n_trials=int(trials.shape[0]),
        baseline=config.baseline,
        baseline_mode=config.baseline_mode,
        power_per_trial=power_per_trial if return_per_trial else None,
    )


def compute_ersp(epochs, channel: str, config: SpectralConfig | None = None) -> TFRResult:
    """Event-related spectral perturbation (baseline-normalized TFR power)."""

    config = config or SpectralConfig()
    if config.baseline is None:
        config = SpectralConfig(**{**config.__dict__, "baseline": (epochs.tmin, min(-0.02, epochs.tmin + 0.05))})
    return compute_tfr(epochs, channel, config=config)


def inter_trial_phase_coherence(epochs, channel: str, config: SpectralConfig | None = None) -> TFRResult:
    """Inter-trial phase coherence / clustering (ITPC) time-frequency map."""

    return compute_tfr(epochs, channel, config=config)


# ---------------------------------------------------------------------------
# Phase-locking value (PLV)
# ---------------------------------------------------------------------------

def phase_locking_value(epochs, ch_x: str, ch_y: str, config: SpectralConfig | None = None) -> PLVResult:
    """Time-frequency phase-locking value between two contacts across trials.

    PLV(f, t) = |mean over trials exp(i (phi_x - phi_y))|, a measure of how
    consistently the two contacts hold a fixed phase relationship at each
    time-frequency point.
    """

    config = config or SpectralConfig()
    sfreq = float(epochs.sfreq)
    x, times = _channel_matrix(epochs, ch_x)
    y, _ = _channel_matrix(epochs, ch_y)
    finite = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    x, y = x[finite], y[finite]
    if x.shape[0] == 0:
        raise ValueError("PLV requires at least one fully finite trial in both channels")
    freqs = config.frequencies(sfreq)
    cx = cwt_morlet(x, sfreq, freqs, n_cycles=config.n_cycles)
    cy = cwt_morlet(y, sfreq, freqs, n_cycles=config.n_cycles)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (cx / np.abs(cx)) * np.conj(cy / np.abs(cy))
    rel[~np.isfinite(rel)] = 0.0
    plv = np.abs(rel.mean(axis=0))
    decim = max(int(config.decim), 1)
    if decim > 1:
        times = times[::decim]
        plv = plv[:, ::decim]
    return PLVResult(ch_x=ch_x, ch_y=ch_y, freqs=freqs, times=times, plv=plv, n_trials=int(x.shape[0]))


def plv_matrix(
    epochs,
    band: tuple[float, float] = (8.0, 13.0),
    channels: Iterable[str] | None = None,
    *,
    time_window: tuple[float, float] | None = None,
    order: int = 4,
) -> pd.DataFrame:
    """Band-limited inter-trial PLV connectivity matrix.

    For each channel the band-limited analytic phase is computed per trial; the
    PLV between every channel pair is taken across trials at each time point and
    averaged over ``time_window`` (defaults to the full post-stimulus epoch).
    Returns a symmetric channels x channels DataFrame.
    """

    arr, times, all_ch = epochs.as_array()
    sfreq = float(epochs.sfreq)
    channels = list(channels) if channels is not None else list(all_ch)
    channels = [c for c in channels if c in all_ch]
    idx = [all_ch.index(c) for c in channels]
    if time_window is None:
        time_window = (max(0.0, float(times.min())), float(times.max()))
    tmask = _time_mask(times, time_window)

    # Per-channel band-limited analytic phase over the window, tracking which
    # trials are finite for each channel so a NaN-masked (artifact-rejected)
    # channel only removes trials from the pairs that involve it.
    phase_per_ch = []
    finite_per_ch = []
    for ci in idx:
        ch_finite = np.all(np.isfinite(arr[:, :, ci]), axis=1)
        analytic = _band_analytic(np.nan_to_num(arr[:, :, ci]), sfreq, band, order=order)
        phase_per_ch.append(np.angle(analytic[:, tmask]))
        finite_per_ch.append(ch_finite)
    n_ch = len(channels)
    mat = np.eye(n_ch)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            both = finite_per_ch[i] & finite_per_ch[j]
            if not both.any():
                mat[i, j] = mat[j, i] = np.nan
                continue
            dphi = phase_per_ch[i][both] - phase_per_ch[j][both]
            plv_t = np.abs(np.mean(np.exp(1j * dphi), axis=0))
            mat[i, j] = mat[j, i] = float(np.nanmean(plv_t)) if plv_t.size else np.nan
    out = pd.DataFrame(mat, index=channels, columns=channels)
    out.attrs["band"] = band
    out.attrs["time_window"] = time_window
    return out


# ---------------------------------------------------------------------------
# Coherence connectivity (magnitude-squared and imaginary)
# ---------------------------------------------------------------------------

def coherence_matrix(
    epochs,
    band: tuple[float, float] = (8.0, 13.0),
    channels: Iterable[str] | None = None,
    *,
    mode: str = "coherence",
    time_window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Band-averaged spectral coherence connectivity matrix across trials.

    ``mode="coherence"`` returns magnitude-squared coherence; ``mode="imaginary"``
    returns the imaginary part of coherency (insensitive to zero-lag/volume-
    conduction effects). Cross-spectra are estimated over trials, which act as
    the averaging ensemble for evoked data.
    """

    arr, times, all_ch = epochs.as_array()
    sfreq = float(epochs.sfreq)
    channels = list(channels) if channels is not None else list(all_ch)
    channels = [c for c in channels if c in all_ch]
    idx = [all_ch.index(c) for c in channels]
    tmask = _time_mask(times, time_window)
    data = arr[:, tmask, :][:, :, idx]  # [n_trials, n_t, n_ch]
    valid = np.all(np.isfinite(data), axis=(1, 2))
    data = data[valid]
    if data.shape[0] == 0:
        raise ValueError("Coherence requires at least one fully finite trial")
    n_t = data.shape[1]
    window = signal.windows.hann(n_t)
    data = (data - data.mean(axis=1, keepdims=True)) * window[np.newaxis, :, np.newaxis]
    fft = np.fft.rfft(data, axis=1)  # [n_trials, n_freq, n_ch]
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
    fmask = (freqs >= float(band[0])) & (freqs <= float(band[1]))
    if not fmask.any():
        fmask = np.argmin(np.abs(freqs - np.mean(band)))[np.newaxis]
    fft = fft[:, fmask, :]  # [n_trials, n_band, n_ch]
    # Cross-spectral matrix averaged over trials and band: [n_ch, n_ch]
    csd = np.einsum("tfi,tfj->ij", fft, np.conj(fft)) / (fft.shape[0] * fft.shape[1])
    psd = np.real(np.diag(csd))
    denom = np.sqrt(np.outer(psd, psd))
    with np.errstate(invalid="ignore", divide="ignore"):
        coherency = csd / denom
    if mode == "imaginary":
        mat = np.abs(np.imag(coherency))
    elif mode in {"coherence", "msc"}:
        mat = np.abs(coherency) ** 2
    elif mode == "coherency":
        mat = np.abs(coherency)
    else:
        raise ValueError(f"Unknown coherence mode {mode!r}")
    np.fill_diagonal(mat, 1.0 if mode != "imaginary" else 0.0)
    out = pd.DataFrame(np.real(mat), index=channels, columns=channels)
    out.attrs["band"] = band
    out.attrs["mode"] = mode
    return out


# ---------------------------------------------------------------------------
# Information-theoretic and directed connectivity
# ---------------------------------------------------------------------------

def mutual_information_matrix(
    epochs,
    channels: Iterable[str] | None = None,
    *,
    time_window: tuple[float, float] | None = None,
    n_bins: int = 16,
    normalized: bool = True,
) -> pd.DataFrame:
    """Histogram mutual-information matrix across trial/time samples."""

    arr, times, all_ch = epochs.as_array()
    channels = list(channels) if channels is not None else list(all_ch)
    channels = [c for c in channels if c in all_ch]
    idx = [all_ch.index(c) for c in channels]
    tmask = _time_mask(times, time_window)
    data = arr[:, tmask, :][:, :, idx]
    out = mutual_information_matrix_from_trials(data, channels, n_bins=n_bins, normalized=normalized)
    out.attrs["time_window"] = time_window
    out.attrs["n_bins"] = int(n_bins)
    out.attrs["normalized"] = bool(normalized)
    return out


def granger_causality_matrix(
    epochs,
    channels: Iterable[str] | None = None,
    *,
    time_window: tuple[float, float] | None = None,
    maxlag: int = 5,
    return_pvalues: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise linear Granger causality matrix across trials.

    Rows are source channels and columns are target channels. Trial boundaries
    are preserved when constructing lagged regressions.
    """

    arr, times, all_ch = epochs.as_array()
    channels = list(channels) if channels is not None else list(all_ch)
    channels = [c for c in channels if c in all_ch]
    idx = [all_ch.index(c) for c in channels]
    tmask = _time_mask(times, time_window)
    data = arr[:, tmask, :][:, :, idx]
    result = granger_causality_matrix_from_trials(data, channels, maxlag=maxlag, return_pvalues=return_pvalues)
    frames = result if isinstance(result, tuple) else (result,)
    for frame in frames:
        frame.attrs["time_window"] = time_window
        frame.attrs["maxlag"] = int(maxlag)
    return result


# ---------------------------------------------------------------------------
# Phase-amplitude coupling (PAC)
# ---------------------------------------------------------------------------

def _modulation_index(phase: np.ndarray, amplitude: np.ndarray, method: str, n_bins: int) -> tuple[float, np.ndarray, np.ndarray, float]:
    finite = np.isfinite(phase) & np.isfinite(amplitude)
    phase = phase[finite]
    amplitude = amplitude[finite]
    if phase.size == 0:
        bins = np.linspace(-np.pi, np.pi, n_bins, endpoint=False) + np.pi / n_bins
        return np.nan, bins, np.full(n_bins, np.nan), np.nan

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = edges[:-1] + np.diff(edges) / 2.0
    which = np.digitize(phase, edges) - 1
    which = np.clip(which, 0, n_bins - 1)
    mean_amp = np.array([amplitude[which == b].mean() if np.any(which == b) else 0.0 for b in range(n_bins)])
    preferred_phase = float(centers[int(np.nanargmax(mean_amp))]) if np.isfinite(mean_amp).any() else np.nan

    if method == "mvl":
        # Canolty mean vector length, normalized by mean amplitude.
        z = amplitude * np.exp(1j * phase)
        mi = float(np.abs(np.mean(z)) / (np.mean(amplitude) + 1e-12))
    elif method == "tort":
        total = mean_amp.sum()
        if total <= 0:
            mi = np.nan
        else:
            p = mean_amp / total
            p = np.clip(p, 1e-12, None)
            entropy = -np.sum(p * np.log(p))
            mi = float((np.log(n_bins) - entropy) / np.log(n_bins))
    else:
        raise ValueError(f"Unknown PAC method {method!r}")
    return mi, centers, mean_amp, preferred_phase


def phase_amplitude_coupling(
    epochs,
    phase_channel: str,
    amp_channel: str | None = None,
    *,
    phase_band: tuple[float, float] = (4.0, 8.0),
    amp_band: tuple[float, float] = (80.0, 150.0),
    method: str = "tort",
    n_bins: int = 18,
    time_window: tuple[float, float] | None = None,
    order: int = 4,
) -> PACResult:
    """Phase-amplitude coupling within or between contacts.

    The low-frequency phase of ``phase_channel`` modulates the high-frequency
    amplitude of ``amp_channel`` (defaults to the same contact). ``method="tort"``
    returns the normalized Kullback-Leibler modulation index; ``method="mvl"``
    returns the Canolty mean-vector-length. Trials are concatenated over the
    selected ``time_window`` to estimate the coupling.
    """

    amp_channel = amp_channel or phase_channel
    sfreq = float(epochs.sfreq)
    xp, times = _channel_matrix(epochs, phase_channel)
    xa, _ = _channel_matrix(epochs, amp_channel)
    tmask = _time_mask(times, time_window)
    finite = np.all(np.isfinite(xp), axis=1) & np.all(np.isfinite(xa), axis=1)
    xp, xa = xp[finite], xa[finite]
    if xp.shape[0] == 0:
        raise ValueError("PAC requires at least one fully finite trial")
    phase = np.angle(_band_analytic(xp, sfreq, phase_band, order=order))[:, tmask].ravel()
    amplitude = np.abs(_band_analytic(xa, sfreq, amp_band, order=order))[:, tmask].ravel()
    mi, centers, mean_amp, preferred = _modulation_index(phase, amplitude, method=method, n_bins=n_bins)
    return PACResult(
        phase_channel=phase_channel,
        amp_channel=amp_channel,
        phase_band=phase_band,
        amp_band=amp_band,
        method=method,
        mi=mi,
        phase_bin_centers=centers,
        amplitude_by_phase=mean_amp,
        preferred_phase=preferred,
        n_samples=int(phase.size),
    )


def phase_amplitude_comodulogram(
    epochs,
    phase_channel: str,
    amp_channel: str | None = None,
    *,
    phase_freqs: Iterable[float] | None = None,
    amp_freqs: Iterable[float] | None = None,
    bandwidth_phase: float = 2.0,
    bandwidth_amp: float = 20.0,
    method: str = "tort",
    n_bins: int = 18,
    time_window: tuple[float, float] | None = None,
) -> ComodulogramResult:
    """PAC comodulogram: modulation index over a grid of phase and amplitude bands."""

    amp_channel = amp_channel or phase_channel
    nyq = 0.49 * float(epochs.sfreq)
    if phase_freqs is None:
        phase_freqs = np.arange(2.0, 14.0, 2.0)
    if amp_freqs is None:
        amp_freqs = np.arange(30.0, min(150.0, nyq - bandwidth_amp), 15.0)
    phase_freqs = np.asarray(list(phase_freqs), dtype=float)
    amp_freqs = np.asarray(list(amp_freqs), dtype=float)
    mi = np.full((len(amp_freqs), len(phase_freqs)), np.nan)
    for pi, pf in enumerate(phase_freqs):
        p_band = (max(pf - bandwidth_phase / 2.0, 0.5), pf + bandwidth_phase / 2.0)
        for ai, af in enumerate(amp_freqs):
            a_band = (max(af - bandwidth_amp / 2.0, p_band[1] + 1.0), min(af + bandwidth_amp / 2.0, nyq))
            if a_band[0] >= a_band[1]:
                continue
            try:
                res = phase_amplitude_coupling(
                    epochs, phase_channel, amp_channel,
                    phase_band=p_band, amp_band=a_band,
                    method=method, n_bins=n_bins, time_window=time_window,
                )
                mi[ai, pi] = res.mi
            except Exception:
                continue
    return ComodulogramResult(
        phase_channel=phase_channel,
        amp_channel=amp_channel,
        phase_freqs=phase_freqs,
        amp_freqs=amp_freqs,
        mi=mi,
        method=method,
        bandwidth_phase=bandwidth_phase,
        bandwidth_amp=bandwidth_amp,
    )


# ---------------------------------------------------------------------------
# Phase-phase (n:m) cross-frequency coupling
# ---------------------------------------------------------------------------

def phase_phase_coupling(
    epochs,
    ch_x: str,
    ch_y: str,
    *,
    freq_x: float,
    freq_y: float,
    n: int = 1,
    m: int = 1,
    n_cycles: float = 7.0,
    time_window: tuple[float, float] | None = None,
) -> float:
    """n:m phase-phase coupling between two contacts (or one contact, two bands).

    Returns the n:m phase-locking value
    ``|mean exp(i (n*phi_x - m*phi_y))|`` over trials and the selected time
    window, where ``phi_x`` is the phase at ``freq_x`` and ``phi_y`` at
    ``freq_y``. ``ch_x == ch_y`` gives within-contact harmonic coupling.
    """

    sfreq = float(epochs.sfreq)
    x, times = _channel_matrix(epochs, ch_x)
    y, _ = _channel_matrix(epochs, ch_y)
    finite = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    x, y = x[finite], y[finite]
    if x.shape[0] == 0:
        raise ValueError("Phase-phase coupling requires at least one fully finite trial")
    tmask = _time_mask(times, time_window)
    cx = cwt_morlet(x, sfreq, np.array([freq_x]), n_cycles=n_cycles)[:, 0, :]
    cy = cwt_morlet(y, sfreq, np.array([freq_y]), n_cycles=n_cycles)[:, 0, :]
    phi_x = np.angle(cx[:, tmask])
    phi_y = np.angle(cy[:, tmask])
    rel = np.exp(1j * (n * phi_x - m * phi_y))
    return float(np.abs(np.mean(rel)))


def phase_phase_matrix(
    epochs,
    ch_x: str,
    ch_y: str | None = None,
    *,
    freqs_x: Iterable[float] | None = None,
    freqs_y: Iterable[float] | None = None,
    n: int = 1,
    m: int = 1,
    n_cycles: float = 7.0,
    time_window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """n:m phase-locking over a grid of frequencies -> DataFrame [freq_y x freq_x]."""

    ch_y = ch_y or ch_x
    nyq = 0.49 * float(epochs.sfreq)
    if freqs_x is None:
        freqs_x = np.arange(4.0, 20.0, 2.0)
    if freqs_y is None:
        freqs_y = np.arange(4.0, min(40.0, nyq), 4.0)
    freqs_x = np.asarray(list(freqs_x), dtype=float)
    freqs_y = np.asarray(list(freqs_y), dtype=float)
    mat = np.full((len(freqs_y), len(freqs_x)), np.nan)
    for yi, fy in enumerate(freqs_y):
        for xi, fx in enumerate(freqs_x):
            try:
                mat[yi, xi] = phase_phase_coupling(
                    epochs, ch_x, ch_y, freq_x=fx, freq_y=fy, n=n, m=m,
                    n_cycles=n_cycles, time_window=time_window,
                )
            except Exception:
                continue
    out = pd.DataFrame(mat, index=np.round(freqs_y, 3), columns=np.round(freqs_x, 3))
    out.index.name = f"{ch_y}_freq_hz"
    out.columns.name = f"{ch_x}_freq_hz"
    return out


# ---------------------------------------------------------------------------
# Epochs accessor
# ---------------------------------------------------------------------------

class EpochsSpectral:
    """Spectral-analysis accessor available as ``epochs.spectral``."""

    def __init__(self, epochs) -> None:
        self.epochs = epochs

    # power / time-frequency
    def psd(self, channels=None, **kwargs) -> pd.DataFrame:
        """Estimate power spectral density for selected epoch channels."""

        return compute_psd(self.epochs, channels=channels, **kwargs)

    def band_power(self, bands=None, channels=None, **kwargs) -> pd.DataFrame:
        """Summarize epoch power within named frequency bands."""

        return band_power(self.epochs, bands=bands, channels=channels, **kwargs)

    def tfr(self, channel: str, config: SpectralConfig | None = None, **kwargs) -> TFRResult:
        """Compute a Morlet time-frequency representation for one channel."""

        return compute_tfr(self.epochs, channel, config=config, **kwargs)

    def ersp(self, channel: str, config: SpectralConfig | None = None) -> TFRResult:
        """Compute baseline-normalized event-related spectral perturbation."""

        return compute_ersp(self.epochs, channel, config=config)

    def itpc(self, channel: str, config: SpectralConfig | None = None) -> TFRResult:
        """Compute inter-trial phase coherence for one channel."""

        return inter_trial_phase_coherence(self.epochs, channel, config=config)

    # connectivity
    def plv(self, ch_x: str, ch_y: str, config: SpectralConfig | None = None) -> PLVResult:
        """Compute time-frequency phase locking between two channels."""

        return phase_locking_value(self.epochs, ch_x, ch_y, config=config)

    def plv_matrix(self, band=(8.0, 13.0), channels=None, **kwargs) -> pd.DataFrame:
        """Compute a band-limited pairwise phase-locking matrix."""

        return plv_matrix(self.epochs, band=band, channels=channels, **kwargs)

    def coherence_matrix(self, band=(8.0, 13.0), channels=None, **kwargs) -> pd.DataFrame:
        """Compute a band-limited pairwise coherence matrix."""

        return coherence_matrix(self.epochs, band=band, channels=channels, **kwargs)

    def mutual_information_matrix(self, channels=None, **kwargs) -> pd.DataFrame:
        """Compute a pairwise mutual-information matrix across epoch samples."""

        return mutual_information_matrix(self.epochs, channels=channels, **kwargs)

    def mi_matrix(self, channels=None, **kwargs) -> pd.DataFrame:
        """Compute mutual information; alias of ``mutual_information_matrix``."""

        return mutual_information_matrix(self.epochs, channels=channels, **kwargs)

    def granger_matrix(self, channels=None, **kwargs) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Compute a directed pairwise Granger-causality matrix."""

        return granger_causality_matrix(self.epochs, channels=channels, **kwargs)

    def granger_causality_matrix(self, channels=None, **kwargs) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Compute Granger causality; explicit alias of ``granger_matrix``."""

        return granger_causality_matrix(self.epochs, channels=channels, **kwargs)

    def pac(self, phase_channel: str, amp_channel: str | None = None, **kwargs) -> PACResult:
        """Estimate phase-amplitude coupling for a channel pair."""

        return phase_amplitude_coupling(self.epochs, phase_channel, amp_channel, **kwargs)

    def comodulogram(self, phase_channel: str, amp_channel: str | None = None, **kwargs) -> ComodulogramResult:
        """Compute phase-amplitude coupling across frequency pairs."""

        return phase_amplitude_comodulogram(self.epochs, phase_channel, amp_channel, **kwargs)

    def phase_phase(self, ch_x: str, ch_y: str | None = None, **kwargs) -> float | pd.DataFrame:
        """Estimate n:m phase coupling or a cross-frequency coupling matrix."""

        if {"freq_x", "freq_y"}.issubset(kwargs):
            return phase_phase_coupling(self.epochs, ch_x, ch_y or ch_x, **kwargs)
        return phase_phase_matrix(self.epochs, ch_x, ch_y, **kwargs)

    @property
    def plot(self):
        """Return a spectral plotting accessor bound to these epochs."""

        from .viz.spectral import SpectralPlotter

        return SpectralPlotter(self.epochs)
