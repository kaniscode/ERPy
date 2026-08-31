from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import ERPy as ep
from ERPy.epochs import Epochs


def _synthetic_epochs(n_trials: int = 14, sfreq: float = 600.0) -> Epochs:
    """Build epochs with known spectral structure.

    * ``A`` -- 10 Hz burst phase-locked to stimulation onset.
    * ``B`` -- 10 Hz burst at a fixed phase offset from ``A`` (strong PLV with A).
    * ``NOISE`` -- broadband noise (weak PLV with A).
    * ``PAC`` -- explicit 6 Hz theta plus a 90 Hz gamma whose amplitude is
      modulated by theta phase (strong phase-amplitude coupling).
    """

    rng = np.random.default_rng(7)
    times = np.round(np.arange(-0.5, 1.0 + 1.0 / sfreq, 1.0 / sfreq), 6)
    n_t = len(times)
    burst = (times > 0) & (times < 0.5)
    rows, index = [], []
    for tr in range(n_trials):
        a = 0.3 * rng.standard_normal(n_t)
        b = 0.3 * rng.standard_normal(n_t)
        a[burst] += 5 * np.sin(2 * np.pi * 10 * times[burst])
        b[burst] += 5 * np.sin(2 * np.pi * 10 * times[burst] + 0.7)
        noise = 1.0 * rng.standard_normal(n_t)
        theta_phase = 2 * np.pi * 6 * times + rng.uniform(0, 2 * np.pi)
        amp = (1 + np.cos(theta_phase)) / 2.0
        pac = 3 * np.sin(theta_phase) + 2 * amp * np.sin(2 * np.pi * 90 * times) + 0.3 * rng.standard_normal(n_t)
        rows.append(np.vstack([a, b, noise, pac]).T)
        index.extend((tr, float(t)) for t in times)
    df = pd.DataFrame(
        np.vstack(rows),
        index=pd.MultiIndex.from_tuples(index, names=["epoch", "time"]),
        columns=["A", "B", "NOISE", "PAC"],
    )
    return Epochs(epochs_df=df, sfreq=sfreq, tmin=-0.5, tmax=1.0, baseline=(-0.5, -0.05))


def test_psd_recovers_oscillation_peak():
    epochs = _synthetic_epochs()
    psd = epochs.spectral.psd(fmin=2, fmax=40)
    assert list(psd.columns) == ["A", "B", "NOISE", "PAC"]
    peak = float(psd["A"].idxmax())
    assert 8.0 <= peak <= 12.0
    # PSD of the mean waveform also runs and returns the same frequency grid.
    psd_mean = epochs.spectral.psd(channels=["A"], on="mean", fmin=2, fmax=40)
    assert not psd_mean.empty


def test_tfr_itc_high_during_phase_locked_burst():
    epochs = _synthetic_epochs()
    cfg = ep.SpectralConfig(fmin=4, fmax=120, n_freqs=30, baseline=(-0.5, -0.05))
    tfr = epochs.spectral.tfr("A", config=cfg)
    assert tfr.power.shape == (30, tfr.times.size)
    assert tfr.itc.shape == (30, tfr.times.size)
    assert tfr.n_trials == epochs.n_trials()
    fi = int(np.argmin(np.abs(tfr.freqs - 10.0)))
    burst = (tfr.times > 0.05) & (tfr.times < 0.45)
    pre = tfr.times < -0.1
    assert np.nanmean(tfr.itc[fi, burst]) > 0.6
    assert np.nanmean(tfr.itc[fi, burst]) > np.nanmean(tfr.itc[fi, pre])
    # ERSP convenience applies a baseline by default.
    ersp = epochs.spectral.ersp("A")
    assert ersp.baseline is not None


def test_plv_distinguishes_coupled_from_noise():
    epochs = _synthetic_epochs()
    cfg = ep.SpectralConfig(fmin=4, fmax=40, n_freqs=20)
    plv = epochs.spectral.plv("A", "B", config=cfg)
    fi = int(np.argmin(np.abs(plv.freqs - 10.0)))
    burst = (plv.times > 0.05) & (plv.times < 0.45)
    assert np.nanmean(plv.plv[fi, burst]) > 0.8

    mat = epochs.spectral.plv_matrix(band=(8, 12), time_window=(0.0, 0.5))
    assert mat.shape == (4, 4)
    assert mat.loc["A", "B"] > mat.loc["A", "NOISE"]
    assert np.allclose(np.diag(mat.to_numpy()), 1.0)


def test_coherence_matrix_modes():
    epochs = _synthetic_epochs()
    coh = epochs.spectral.coherence_matrix(band=(8, 12), time_window=(0.0, 0.5))
    assert coh.loc["A", "B"] > coh.loc["A", "NOISE"]
    icoh = epochs.spectral.coherence_matrix(band=(8, 12), mode="imaginary", time_window=(0.0, 0.5))
    assert icoh.loc["A", "B"] >= 0.0
    assert (icoh.to_numpy() <= 1.0 + 1e-9).all()


def test_pac_detects_coupling_and_comodulogram_peak():
    epochs = _synthetic_epochs()
    coupled = epochs.spectral.pac("PAC", phase_band=(4, 8), amp_band=(70, 110), method="tort")
    control = epochs.spectral.pac("NOISE", phase_band=(4, 8), amp_band=(70, 110), method="tort")
    assert coupled.mi > control.mi
    assert coupled.mi > 0.01
    mvl = epochs.spectral.pac("PAC", phase_band=(4, 8), amp_band=(70, 110), method="mvl")
    assert mvl.mi > 0.05

    como = epochs.spectral.comodulogram(
        "PAC", phase_freqs=[4, 6, 8, 10], amp_freqs=[70, 90, 110], bandwidth_phase=2, bandwidth_amp=20
    )
    assert como.mi.shape == (3, 4)
    amp_i, phase_i = np.unravel_index(np.nanargmax(como.mi), como.mi.shape)
    assert como.phase_freqs[phase_i] == pytest.approx(6.0)
    assert como.amp_freqs[amp_i] == pytest.approx(90.0)


def test_phase_phase_and_band_power():
    epochs = _synthetic_epochs()
    nm = epochs.spectral.phase_phase("A", freq_x=10.0, freq_y=10.0, n=1, m=1, time_window=(0.0, 0.5))
    assert 0.0 <= float(nm) <= 1.0
    matrix = epochs.spectral.phase_phase("A", freqs_x=[8, 10, 12], freqs_y=[8, 10, 12], time_window=(0.0, 0.5))
    assert matrix.shape == (3, 3)

    bp = epochs.spectral.band_power()
    assert list(bp.index) == ["A", "B", "NOISE", "PAC"]
    assert set(ep.DEFAULT_BANDS).issubset(bp.columns)


def test_spectral_visualizations_build_figures():
    epochs = _synthetic_epochs()
    sp = epochs.spectral
    fig = sp.plot.summary("A")
    assert fig is not None
    plt.close(fig)
    for ax in (
        sp.plot.tfr("A"),
        sp.plot.itpc("A"),
        sp.plot.psd(),
        sp.plot.plv("A", "B"),
        sp.plot.connectivity(sp.plv_matrix(band=(8, 12))),
        sp.plot.pac("PAC", phase_band=(4, 8), amp_band=(70, 110)),
        sp.plot.comodulogram("PAC", phase_freqs=[4, 6, 8], amp_freqs=[70, 90]),
    ):
        assert ax.figure is not None
        plt.close(ax.figure)


def test_nan_response_trials_are_ignored():
    epochs = _synthetic_epochs()
    # Mask one trial's channel A response (mimics reject_artifacts nan_response mode).
    epochs.epochs_df.loc[(0, slice(None)), "A"] = np.nan
    tfr = epochs.spectral.tfr("A")
    assert tfr.n_trials == epochs.n_trials() - 1
    psd = epochs.spectral.psd(channels=["A"])
    assert np.isfinite(psd["A"].to_numpy()).all()
