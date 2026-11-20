# Analysis workflow

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

This walkthrough follows one stimulation acquisition from a configured project
to saved, auditable results, first with the one-call analysis and then with the
individual steps it is built from. Identifiers and electrode labels are
fictional placeholders.

## The pieces

| Layer | Use |
| --- | --- |
| `Patient` | Quick wrapper for routine work: detect events, epoch data, warm caches, and run analyses. |
| `DataLoader` | Universal connector for stimulation iEEG projects: ERPy CSV metadata, EDF/BDF, BrainVision, NWB, BIDS/OpenNeuro folders, and DANDI-style NWB files. |
| `Events` | Creates canonical `events/<stim_pair>_events.csv` files from timestamps, binary vectors, artifacts, EDF trigger channels, or EDF annotations. |
| `Pipeline` | Registry-based preprocessing: artifact blanking, notch, bandpass, decimation, re-referencing (common-average reference by default, plus bipolar and Laplacian montages), and bad-channel rejection. |
| `Epochs` | Trial x time x channel data with HDF5 caching, artifact-response QC, plotting, CRP, spectral analysis, and ERP detection methods. |
| `erp_detection` | Source-informed quantities for Kundu/Rolston, Keller z-score, Miller CRP, Crowther SIGNI, peak amplitude, and matched-window RMS, plus a fixed-window reproducibility-and-energy primary result. |
| `spectral` | Per-trial or mean-waveform spectral analysis: Welch PSD, Morlet time-frequency power/ERSP, inter-trial phase coherence (ITPC), band power, and spectral connectivity (phase-locking value, magnitude-squared and imaginary coherence, phase-amplitude and n:m phase-phase cross-frequency coupling). |
| `viz` | Mean/SEM waveforms, overlays, trial heatmaps, response maps, ranked grids, metadata-organized grids, CRP canonical response curves, CRP canonical-weight timecourses, within-stim metric z-score maps, response matrices, dynamic graph metrics, NetworkX graphs, interactive brain connectomes, single-site and aggregate evoked-response brain graph animations. |
| `warmup` | One call to precompute event, processed, and epoch caches before cohort-scale analysis. |

## One call

`Patient.analyze` epochs the acquisition, scores every trial-by-channel
response for artifacts, masks contaminated responses, runs the detectors on the
cleaned epochs, and annotates each detection with artifact-QC eligibility.

```python
import ERPy as ep

patient = ep.Patient("EXAMPLE_PATIENT", config_path="config.yaml")
patient                                   # <Patient EXAMPLE_PATIENT | sessions: EXAMPLE_SESSION>
patient.stim_pairs("EXAMPLE_SESSION")     # ['STIM_A_1_2', ...]

result = patient.analyze("EXAMPLE_SESSION", "STIM_A_1_2", pipeline="blank_filt")
result                                    # <AnalysisResult ... | 20 trials, 64 channels | 7 primary responses after QC>

result.primary_significant.head()
result.save("analysis_tables")
fig = result.clean_epochs.plot.summary("CONTACT_B1", detections=result.qc_detections)
```

The returned `AnalysisResult` keeps every intermediate product:

| Attribute | Contents |
| --- | --- |
| `epochs` | Epochs as extracted, before artifact rejection. |
| `clean_epochs` | Epochs after trial-by-channel artifact rejection; the detector input. |
| `artifact_report`, `artifact_summary` | Per-response artifact scores and their per-channel summary. |
| `detections` | Long-form detector table, one row per channel and method. |
| `qc_detections` | `detections` with artifact-QC eligibility columns. |
| `primary_significant` | One CRP-energy row per response that passes the joint test and artifact QC. The recommended final result. |
| `qc_pass_detections`, `significant` | All rows passing artifact QC; rows flagged by the legacy consensus. |
| `metadata` | Audit record of every effective setting. |

`result.save(folder)` writes the tables as CSV plus one JSON audit record to
that local folder and returns the written paths. Nothing is uploaded; see
[data privacy](DATA_PRIVACY.md).

A session can also be addressed directly. `patient["EXAMPLE_SESSION"]` returns a
`Recording` with the same `detect_events`, `epoch`, and `analyze` methods, without
repeating the session identifier.

## Step by step

The lower-level pieces remain available when you want manual control:

```python
epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2", pipeline="blank_filt")
detections = epochs.detect_all(min_consensus=2)   # every detector
primary = epochs.detect()                         # primary CRP-energy detector only
artifact_report = epochs.flag_artifacts()
clean_epochs = epochs.reject_artifacts(report=artifact_report, mode="drop_epoch")
```

The standard `blank_filt` stream is 0.1–50 Hz, so its SIGNI/high-gamma row is
reported as unavailable rather than negative. To evaluate 70–170-Hz gamma,
pass a separately artifact-masked, passband-preserving object through
`wideband_epochs=`. The legacy `consensus` field counts only
methods whose required inputs are available.

`epochs.detect_all(...)` was called `detect_erp_all` before version 1.1; the
older name still works. Each step has its own guide:
[events](EVENTS.md), [preprocessing and quality control](PREPROCESSING_AND_QC.md),
[response detection](DETECTION.md).

## Working with epochs

`Epochs` holds trial x time x channel data together with its sampling and
alignment contract.

```python
epochs                      # <Epochs | 20 trials x 751 samples x 64 channels | -0.5 to 1 s at 500 Hz | stim A1, A2>
epochs.shape                # (n_trials, n_times, n_channels)
epochs.channels, epochs.times
epochs.mean()               # trial-averaged waveform, time x channel (also .std(), .sem())
samples, times, channels = epochs.as_array()

epochs.plot.summary("CONTACT_B1")     # plotting accessor
epochs.spectral.psd()                 # spectral accessor
epochs.to_hdf("epochs.h5")            # and ep.Epochs.from_hdf("epochs.h5")
```

Epoch defaults are -0.5 to 1.0 s with a (-0.5, -0.03) s baseline. Pass `tmin`,
`tmax`, `baseline`, or `zero_time` to `patient.epoch(...)` or
`patient.analyze(...)` to change them. Extracted epochs are cached on disk and
reused only when the events, source recording, settings, and installed code
are unchanged.

## Zero-time anchoring

Epochs are baseline corrected and, by default, recentred on a short window
after an anchor shared by all trials. `epochs.zero_time_report()` and
`epochs.post_artifact_anchor_report()` show the anchor and residual; see
[preprocessing and quality control](PREPROCESSING_AND_QC.md#zero-time-anchoring).

## Starting from BIDS or NWB

For OpenNeuro/BIDS-style iEEG projects:

```python
patient = ep.Patient.from_bids(
    "./open_data/ds004080",
    subject="ccepAgeUMCU01",
    session="1",
    task="SPESclin",
)

epochs = patient.epoch("1", "PT_1_2")
```

For NWB/DANDI-style iEEG projects:

```python
patient = ep.Patient.from_nwb(
    "./recordings/example_recording.nwb",
    output_root="./erpy_nwb_project",
    patient_id="EXAMPLE_PATIENT",
    session_id="EXAMPLE_SESSION",
    stim_pair="STIM_A_1_2",    # optional if NWB trials include a stimulation-site column
    series_name="ElectricalSeries",
)

epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2")
```

See [data formats](DATA_FORMATS.md) for what the importers read and write.

## Preparing a whole patient

```python
status = patient.warmup()          # events, processed signals and epochs for every acquisition
status[~status["ok"]]              # acquisitions that failed, with the failing stage and error
```

## See also

- [Data formats and the project configuration](DATA_FORMATS.md)
- [Methods and mathematical definitions](METHODS.md)
- [API reference](API_REFERENCE.md)
- [Troubleshooting](TROUBLESHOOTING.md)
