# Spectral analysis

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

Spectral measures are available from the `epochs.spectral` accessor and as
top-level functions. All of them use only NumPy and SciPy.

Spectral analysis runs per trial or on the mean response waveform via the `epochs.spectral` accessor (and the equivalent `ep.compute_*` functions):

```python
# Power spectral density (Welch), per trial averaged or on the mean waveform
psd = epochs.spectral.psd(channels=["CONTACT_B1"])              # frequency x channel
psd_mean = epochs.spectral.psd(channels=["CONTACT_B1"], on="mean")
bands = epochs.spectral.band_power()                       # channel x delta/theta/alpha/beta/gamma/high_gamma

# Morlet time-frequency: power, ERSP (baseline-normalized), and ITPC
from ERPy import SpectralConfig
cfg = SpectralConfig(fmin=4, fmax=150, n_freqs=40, baseline=(-0.5, -0.05))
tfr = epochs.spectral.tfr("CONTACT_B1", config=cfg)             # tfr.power, tfr.itc
ersp = epochs.spectral.ersp("CONTACT_B1")                       # event-related spectral perturbation
itpc = epochs.spectral.itpc("CONTACT_B1")                       # inter-trial phase coherence

epochs.spectral.plot.summary("CONTACT_B1")                      # mean waveform + PSD + ERSP + ITPC panel
epochs.spectral.plot.tfr("CONTACT_B1")
epochs.spectral.plot.itpc("CONTACT_B1")
```

Spectral connectivity covers the standard EEG/iEEG measures — phase-locking value, coherence, and cross-frequency coupling:

```python
# Phase-locking value (PLV) between contacts: time-frequency resolved and as a matrix
plv = epochs.spectral.plv("STIM_A1", "CONTACT_B1", config=cfg)    # plv.plv is freq x time
plv_mat = epochs.spectral.plv_matrix(band=(8, 13), time_window=(0.0, 0.5))

# Coherence connectivity (magnitude-squared or imaginary coherency)
coh = epochs.spectral.coherence_matrix(band=(8, 13))
icoh = epochs.spectral.coherence_matrix(band=(8, 13), mode="imaginary")

# Phase-amplitude coupling (Tort modulation index or Canolty mean-vector-length) and comodulograms
pac = epochs.spectral.pac("CONTACT_B1", phase_band=(4, 8), amp_band=(80, 150), method="tort")
como = epochs.spectral.comodulogram("CONTACT_B1")               # MI over phase x amplitude band grid

# n:m phase-phase cross-frequency coupling (single estimate or a frequency grid)
nm = epochs.spectral.phase_phase("STIM_A1", "CONTACT_B1", freq_x=6.0, freq_y=12.0, n=2, m=1)
nm_matrix = epochs.spectral.phase_phase("STIM_A1", "CONTACT_B1")

epochs.spectral.plot.plv("STIM_A1", "CONTACT_B1")
epochs.spectral.plot.connectivity(plv_mat)
epochs.spectral.plot.comodulogram("CONTACT_B1")
epochs.spectral.plot.pac("CONTACT_B1", phase_band=(4, 8), amp_band=(80, 150))
```

PLV is computed across trials at each time-frequency point (phase consistency of stimulus-locked activity); coherence cross-spectra are estimated over trials; PAC and n:m coupling concatenate trials over the selected response window. NaN-masked (artifact-rejected) trials are dropped automatically. All spectral computations use only `numpy`/`scipy`, so they require no extra dependencies.

## See also

- [Methods: spectral and connectivity quantities](METHODS.md#spectral-and-connectivity-quantities)
- [Visualization](VISUALIZATION.md)
- [Notebook 03: spectral visualizations](../notebooks/examples/03_spectral_visualizations.ipynb)
