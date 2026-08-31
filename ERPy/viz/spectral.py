"""Visualizations for ERPy spectral analyses."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..spectral import (
    ComodulogramResult,
    PACResult,
    PLVResult,
    SpectralConfig,
    TFRResult,
    compute_psd,
    compute_tfr,
    phase_amplitude_coupling,
    phase_locking_value,
)


def _freq_yticks(ax, freqs: np.ndarray, n: int = 6) -> None:
    idx = np.linspace(0, len(freqs) - 1, min(n, len(freqs))).astype(int)
    ax.set_yticks(idx)
    ax.set_yticklabels([f"{freqs[i]:.0f}" for i in idx])


def plot_psd(
    psd: pd.DataFrame,
    channels: Iterable[str] | None = None,
    ax=None,
    *,
    logx: bool = True,
    logy: bool = True,
    top_n: int | None = 8,
):
    """Plot Welch power spectra (one line per channel)."""

    ax = ax or plt.subplots(figsize=(6.2, 3.6))[1]
    if psd.empty:
        ax.text(0.5, 0.5, "No PSD", ha="center", va="center")
        ax.axis("off")
        return ax
    cols = list(channels) if channels is not None else list(psd.columns)
    cols = [c for c in cols if c in psd.columns]
    if top_n is not None and channels is None and len(cols) > top_n:
        cols = psd[cols].mean(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    freqs = psd.index.to_numpy(dtype=float)
    for col in cols:
        ax.plot(freqs, psd[col].to_numpy(dtype=float), lw=1.4, label=col)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_title("Power spectrum")
    if len(cols) <= 12:
        ax.legend(loc="best", fontsize=7, ncol=2)
    return ax


def plot_tfr(result: TFRResult, ax=None, *, kind: str = "power", cmap: str | None = None, vmax: float | None = None):
    """Plot a time-frequency map (``kind`` = ``power``/``ersp`` or ``itc``)."""

    ax = ax or plt.subplots(figsize=(6.4, 3.8))[1]
    if kind in {"itc", "itpc"}:
        data = result.itc
        cmap = cmap or "viridis"
        vmin, vmax = 0.0, (vmax or float(np.nanmax(data)) or 1.0)
        label = "Inter-trial phase coherence"
    else:
        data = result.power
        cmap = cmap or "RdBu_r"
        if vmax is None:
            vmax = float(np.nanpercentile(np.abs(data), 98)) or 1.0
        vmin = -vmax
        label = "ERSP" if result.baseline is not None else "Power"
    times = result.times * 1000.0
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[times.min(), times.max(), 0, len(result.freqs) - 1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    _freq_yticks(ax, result.freqs)
    ax.axvline(0, color="k", ls="--", lw=0.9)
    ax.set_xlabel("Time after stimulation (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{result.channel} {label.lower()}")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(label)
    return ax


def plot_ersp(result: TFRResult, ax=None, **kwargs):
    """Plot baseline-normalized event-related spectral perturbation."""

    return plot_tfr(result, ax=ax, kind="power", **kwargs)


def plot_itpc(result: TFRResult, ax=None, **kwargs):
    """Plot inter-trial phase coherence as a time-frequency map."""

    return plot_tfr(result, ax=ax, kind="itc", **kwargs)


def plot_plv_timefreq(result: PLVResult, ax=None, cmap: str = "magma"):
    """Time-frequency phase-locking value heatmap between two contacts."""

    ax = ax or plt.subplots(figsize=(6.4, 3.8))[1]
    times = result.times * 1000.0
    im = ax.imshow(
        result.plv,
        aspect="auto",
        origin="lower",
        extent=[times.min(), times.max(), 0, len(result.freqs) - 1],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    _freq_yticks(ax, result.freqs)
    ax.axvline(0, color="w", ls="--", lw=0.9)
    ax.set_xlabel("Time after stimulation (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"PLV {result.ch_x} ↔ {result.ch_y}")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Phase-locking value")
    return ax


def plot_connectivity_matrix(
    matrix: pd.DataFrame,
    ax=None,
    *,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    label: str | None = None,
):
    """Plot a channel x channel connectivity matrix (PLV, coherence, etc.)."""

    ax = ax or plt.subplots(figsize=(5.6, 5.0))[1]
    if matrix.empty:
        ax.text(0.5, 0.5, "No connectivity", ha="center", va="center")
        ax.axis("off")
        return ax
    values = matrix.to_numpy(dtype=float)
    mode = str(matrix.attrs.get("mode", ""))
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        finite = values[np.isfinite(values)]
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
        vmax = vmax or 1.0
    cmap = cmap or "viridis"
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=6)
    band = matrix.attrs.get("band")
    default_title = "Connectivity"
    if band is not None:
        default_title = f"{mode or 'connectivity'} {band[0]:.0f}-{band[1]:.0f} Hz"
    ax.set_title(title or default_title)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(label or (mode or "connectivity"))
    return ax


def plot_comodulogram(result: ComodulogramResult, ax=None, cmap: str = "inferno"):
    """Plot a PAC comodulogram (amplitude frequency x phase frequency)."""

    ax = ax or plt.subplots(figsize=(5.8, 4.4))[1]
    data = result.mi
    finite = data[np.isfinite(data)]
    vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[result.phase_freqs.min(), result.phase_freqs.max(), result.amp_freqs.min(), result.amp_freqs.max()],
        vmin=0.0,
        vmax=vmax or 1.0,
    )
    ax.set_xlabel("Phase frequency (Hz)")
    ax.set_ylabel("Amplitude frequency (Hz)")
    title = f"{result.phase_channel} → {result.amp_channel}" if result.phase_channel != result.amp_channel else result.phase_channel
    ax.set_title(f"PAC comodulogram ({result.method}): {title}")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Modulation index")
    return ax


def plot_pac(result: PACResult, ax=None, color: str = "#7c3aed"):
    """Plot the amplitude-by-phase distribution for a single PAC estimate."""

    ax = ax or plt.subplots(figsize=(5.4, 3.2))[1]
    centers = np.degrees(result.phase_bin_centers)
    width = (centers[1] - centers[0]) * 0.9 if len(centers) > 1 else 18.0
    ax.bar(centers, result.amplitude_by_phase, width=width, color=color, alpha=0.85)
    ax.set_xlabel("Phase (degrees)")
    ax.set_ylabel("Mean amplitude")
    coupling = "self" if result.phase_channel == result.amp_channel else f"→ {result.amp_channel}"
    ax.set_title(
        f"PAC {result.phase_channel} {coupling}\n"
        f"{result.phase_band[0]:.0f}-{result.phase_band[1]:.0f} Hz phase, "
        f"{result.amp_band[0]:.0f}-{result.amp_band[1]:.0f} Hz amp, MI={result.mi:.3g}"
    )
    return ax


def plot_phase_phase_matrix(matrix: pd.DataFrame, ax=None, cmap: str = "magma"):
    """Plot an n:m phase-phase coupling matrix."""

    ax = ax or plt.subplots(figsize=(5.6, 4.4))[1]
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[float(matrix.columns.min()), float(matrix.columns.max()), float(matrix.index.min()), float(matrix.index.max())],
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel(matrix.columns.name or "Phase frequency x (Hz)")
    ax.set_ylabel(matrix.index.name or "Phase frequency y (Hz)")
    ax.set_title("n:m phase-phase coupling")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("n:m phase-locking value")
    return ax


def plot_spectral_summary(epochs, channel: str, config: SpectralConfig | None = None):
    """Multi-panel spectral summary for one channel: mean waveform, PSD, ERSP, ITPC."""

    config = config or SpectralConfig()
    tfr = compute_tfr(epochs, channel, config=config)
    psd = compute_psd(epochs, channels=[channel])
    mean = epochs.get_mean_waveform()

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.0), layout="constrained")
    ax = axes[0, 0]
    ax.plot(mean.index * 1000.0, mean[channel], color="#2563eb", lw=1.7)
    ax.axvline(0, color="0.2", ls="--", lw=1.0)
    ax.axhline(0, color="0.82", lw=0.8)
    ax.set_xlabel("Time after stimulation (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title(f"{channel} mean response")

    plot_psd(psd, ax=axes[0, 1])
    plot_tfr(tfr, ax=axes[1, 0], kind="power")
    plot_tfr(tfr, ax=axes[1, 1], kind="itc")
    fig.suptitle(f"Spectral summary: {channel} ({tfr.n_trials} trials)", fontsize=12)
    return fig


class SpectralPlotter:
    """Plotting accessor available as ``epochs.spectral.plot``."""

    def __init__(self, epochs) -> None:
        self.epochs = epochs

    def psd(self, channels=None, **kwargs):
        """Compute and plot power spectral density for selected channels."""

        return plot_psd(compute_psd(self.epochs, channels=channels), **kwargs)

    def tfr(self, channel: str, config: SpectralConfig | None = None, **kwargs):
        """Compute and plot a time-frequency representation for one channel."""

        return plot_tfr(compute_tfr(self.epochs, channel, config=config), **kwargs)

    def ersp(self, channel: str, config: SpectralConfig | None = None, **kwargs):
        """Compute and plot event-related spectral perturbation."""

        from ..spectral import compute_ersp

        return plot_ersp(compute_ersp(self.epochs, channel, config=config), **kwargs)

    def itpc(self, channel: str, config: SpectralConfig | None = None, **kwargs):
        """Compute and plot inter-trial phase coherence."""

        return plot_itpc(compute_tfr(self.epochs, channel, config=config), **kwargs)

    def plv(self, ch_x: str, ch_y: str, config: SpectralConfig | None = None, **kwargs):
        """Compute and plot time-frequency phase locking for two channels."""

        return plot_plv_timefreq(phase_locking_value(self.epochs, ch_x, ch_y, config=config), **kwargs)

    def connectivity(self, matrix: pd.DataFrame, **kwargs):
        """Plot a square channel-connectivity matrix."""

        return plot_connectivity_matrix(matrix, **kwargs)

    def comodulogram(self, phase_channel: str, amp_channel: str | None = None, **kwargs):
        """Compute and plot phase-amplitude coupling across frequency pairs."""

        from ..spectral import phase_amplitude_comodulogram

        plot_kwargs = {k: kwargs.pop(k) for k in ("ax", "cmap") if k in kwargs}
        return plot_comodulogram(
            phase_amplitude_comodulogram(self.epochs, phase_channel, amp_channel, **kwargs),
            **plot_kwargs,
        )

    def pac(self, phase_channel: str, amp_channel: str | None = None, **kwargs):
        """Compute and plot amplitude as a function of oscillatory phase."""

        plot_kwargs = {k: kwargs.pop(k) for k in ("ax", "color") if k in kwargs}
        return plot_pac(
            phase_amplitude_coupling(self.epochs, phase_channel, amp_channel, **kwargs),
            **plot_kwargs,
        )

    def summary(self, channel: str, config: SpectralConfig | None = None):
        """Create a multi-panel spectral summary for one channel."""

        return plot_spectral_summary(self.epochs, channel, config=config)
