# ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis

**Auditable stimulation response pipeline**

A Python workflow for event detection, preprocessing, quality control, response
detection, waveform measurements and reproducible result exports.

[Getting started](docs/GETTING_STARTED.md) | [Methods and equations](docs/METHODS.md) | [API reference](docs/API_REFERENCE.md) | [Visualization notebooks](docs/VISUALIZATION_GALLERY.md) | [Validation](docs/VALIDATION.md) | [External expert-label evaluation](docs/EXTERNAL_VALIDATION.md) | [Detector–annotation agreement](docs/ANNOTATION_AGREEMENT.md) | [Label-free N1](docs/N1_LABEL_FREE.md) | [Secondary N1 models](docs/N1_DEVELOPMENT.md) | [Data privacy](docs/DATA_PRIVACY.md)

ERPy is research software. It does not make clinical decisions and is not a
medical device. Inspect waveforms, acquisition settings, and quality-control
reports before interpreting any automated response label.

Unless a named public dataset is cited, identifiers and electrode labels in
the examples below are fictional placeholders.

ERPy turns raw EDF/BDF, BrainVision, NWB, or per-stimulation CSV recordings into event files, cleaned continuous data, epoched HDF5 caches, stimulation-evoked response detection tables, spectral analyses, and summary figures. This package implements a simple clinical research oriented workflow:

```python
import ERPy as ep

patient = ep.Patient("EXAMPLE_PATIENT", config_path="config.yaml")
result = patient.analyze("EXAMPLE_SESSION", "STIM_A_1_2", pipeline="blank_filt")

result.primary_significant.head()
result.save("analysis_tables")
fig = result.clean_epochs.plot.summary("CONTACT_B1", detections=result.qc_detections)
```

The lower-level pieces remain available when you want manual control:

```python
epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2", pipeline="blank_filt")
detections = epochs.detect_erp_all(min_consensus=2)
artifact_report = epochs.flag_artifacts()
clean_epochs = epochs.reject_artifacts(report=artifact_report, mode="drop_epoch")
```

The standard `blank_filt` stream is 0.1–50 Hz, so its SIGNI/high-gamma row is
reported as unavailable rather than negative. To evaluate 70–170-Hz gamma,
pass a separately artifact-masked, passband-preserving object through
`wideband_epochs=`. The legacy `consensus` field counts only
methods whose required inputs are available.

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

Epochs are baseline-window corrected and, by default, recentered using a short window after a **common** anchor shared across trials. ERPy searches the trial-averaged peri-onset artifact envelope for a sustained amplitude-and-slope return toward baseline. If no settled interval is found, it uses a reported amplitude-return, quietest-sample, or peak-neighbor fallback. The anchor report records the selected method and residual values for inspection. Subtracting a post-anchor window sets an amplitude reference; it does not establish a flat waveform, artifact removal, or physiological baseline recovery. Time 0 remains the stimulation-onset reference. Audit the anchor and residual directly:

```python
epochs.zero_time_report()
epochs.post_artifact_anchor_report()
epochs.detect_zero_time_artifacts(zero_time=0.0)
```

## What ERPy Provides

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

## Install

This source checkout is `1.1.0rc1`. It includes the current pipeline, public-data
examples and optional negative-N1 classifier. Install the complete source folder:

```bash
python -m pip install ".[edf,nwb,viz]"
git rev-parse HEAD
```

Record the printed Git commit with the results. The source installation reports
`1.1.0rc1`; the tagged installation below retrieves historical `1.0.0`. The
commands using `main` retrieve the latest public branch. Confirm its version
before use, and use an exact recorded commit for reproducing published results.

ERPy's import name is `ERPy`, while its installable distribution is
`erpy-neuro`. The unrelated name `erpy` on the Python Package Index is an
Erlang communication library; do not use `pip install erpy` for this project.
The package declares Python 3.9 or newer. This release candidate's checks were
run on Python 3.12.5; an expanded Python-version matrix has not been verified
for these changes. Use the recorded environment for numerical reproduction.

For the tagged GitHub release, create an isolated environment and install the
features needed for EDF/BIDS, NWB, and the full visualization gallery. A public
GitHub release does not require a GitHub account or password.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "erpy-neuro[edf,nwb,viz] @ git+https://github.com/kaniscode/ERPy.git@v1.0.0"
python -c "import ERPy; print(ERPy.__version__)"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "erpy-neuro[edf,nwb,viz] @ git+https://github.com/kaniscode/ERPy.git@v1.0.0"
python -c "import ERPy; print(ERPy.__version__)"
```

The final command should print `1.0.0`. The command above installs directly
from the immutable `v1.0.0` tag and therefore requires Git. It installs the
library but does not create a local copy of the notebooks.

For the current worked notebooks, beginning with the actual deidentified cohort
recording, clone `main`. Those examples postdate the historical `v1.0.0` tag:

```bash
git clone --branch main --depth 1 https://github.com/kaniscode/ERPy.git ERPy-source
python -m pip install "./ERPy-source[edf,nwb,viz]"
python -m pip install jupyterlab
python -m jupyter lab ERPy-source/notebooks/examples/00_quickstart.ipynb
```

The lead notebook's stored figures can be viewed directly. Reexecution requires
the authorized trial export and `ERPY_COHORT_EXAMPLE_DIR` configured before
starting Jupyter. Without Git, download the
[current main source ZIP](https://github.com/kaniscode/ERPy/archive/refs/heads/main.zip),
extract the complete folder, rename it `ERPy-source`, and use the same install
command shown above. A step-by-step guide for readers who are new to Python,
with separate macOS/Linux and Windows commands, is in
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

For a local source checkout used for development:

```bash
python -m pip install -e ".[dev,edf,nwb,viz]"
```

Minimal CSV workflows do not require MNE. EDF/BIDS workflows should install the `edf` extra:

```bash
python -m pip install -e ".[edf]"
```

NWB workflows should install the `nwb` extra:

```bash
python -m pip install -e ".[nwb]"
```

Coordinate-based brain graphs use the `viz` extra. High-fidelity FreeSurfer/MNE/PyVista surface rendering is optional:

```bash
python -m pip install -e ".[viz]"
python -m pip install -e ".[viz,surface]"
```

Bootstrap a portable config and metadata stubs:

```bash
erpy-init --config ./config.yaml --data-root ./data
erpy-init show-config --config ./config.yaml
```

These commands create a project-local configuration file and blank metadata
tables; they never upload recording data.

### Local outputs and optional release-record labels

ERPy analysis stays on the computer or storage system where you run it.
`result.save("analysis_tables")` creates CSV tables and one JSON audit record
inside that local folder. It does not upload a recording, register a result,
or send a release-record identifier to GitHub or any ERPy server. ERPy has no
such server.

An identifier written as `ERPY-RR-<UUID>` is an optional, author-assigned audit
label for a governed local derivative. It is not an upload address. ERPy 1.0.0
does not create or attach that label automatically. If your study uses the
convention, generate a random UUIDv4 locally and keep the mapping from that
opaque label to the permitted local files inside the governed research
environment:

```python
from uuid import uuid4

release_record_id = f"ERPY-RR-{str(uuid4()).upper()}"
print(release_record_id)
```

The random label should never be calculated from participant identifiers,
filenames, dates, contact labels, or waveform values. The version-level
`ERPy.CHECKPOINT_COMPATIBILITY_ID` and the short preprocessing hashes used in
cache filenames are separate software/local-file labels; neither identifies a
participant or triggers network transfer. The optional public-data validator
downloads public OpenNeuro ranges, and optional brain plotting can download
public template assets, but neither uploads local recordings.

The default `result.save(...)` filenames and metadata can contain the analysis
patient, session, and stimulation-pair values. Adding an opaque label or merely
renaming the folder is not deidentification; inspect and approve a separate
sharing derivative under the applicable governance process.

## Data Contract

ERPy uses a small YAML config:

```yaml
rawdata_path: ./data/raw
procdata_path: ./data/processed
rawdata_meta_path: ./data/metadata/raw_metadata.csv
stim_meta_path: ./data/metadata/stim_metadata.csv
elec_meta_path: ./data/metadata/electrode_metadata.csv
session_ids:
  EXAMPLE_PATIENT: [EXAMPLE_SESSION]
```

Required metadata tables:

| File | Key columns |
| --- | --- |
| `raw_metadata.csv` | `patient_id`, `session_id`, `raw_file`, optional `start_time`, `stop_time`, `sampling_freq`, `nwb_series` |
| `stim_metadata.csv` | `patient_id`, `session_id`, `stim_pair`, `stim_start`, `stim_stop`, `stim_freq` |
| `electrode_metadata.csv` | `patient_id`, `session_id`, `elec_label`, optional `anat_label`, `mni_x`, `mni_y`, `mni_z`, `bad_channel` |

Event files always live at:

```text
<procdata>/<patient>/<session>/events/<stim_pair>_events.csv
```

with a `times` column containing datetimes or sample-aligned numeric seconds.

## Native BIDS / OpenNeuro

ERPy can import BIDS iEEG projects without hand-writing metadata:

```python
import ERPy as ep

patient = ep.Patient.from_bids(
    bids_root="./open_data/ds003708",
    subject="01",
    session="ieeg01",
    task="ccep",
    output_root="./erpy_ds003708",
)
```

The importer:

- discovers `*_ieeg.vhdr`, `*_ieeg.edf`, `*_events.tsv`, `*_channels.tsv`, and `*_electrodes.tsv`;
- converts BIDS `electrical_stimulation_site` values such as `LTG1-LTG2` into ERPy names such as `LTG_1_2`;
- writes ERPy metadata and canonical event CSVs;
- keeps raw files in place by default, so large OpenNeuro datasets do not need to be duplicated;
- reads BrainVision windows natively around stimulation blocks.

## Native NWB / DANDI

ERPy can read NWB `ElectricalSeries` objects directly while keeping the same event, epoching, QC, detection, and visualization workflow:

```python
import ERPy as ep

patient = ep.Patient.from_nwb(
    nwb_path="./recordings/example_recording.nwb",
    output_root="./erpy_nwb_project",
    patient_id="EXAMPLE_PATIENT",
    session_id="EXAMPLE_SESSION",
    stim_pair="STIM_A_1_2",
    stim_times=[12.50, 17.50, 22.50],   # seconds from NWB session start
    series_name="ElectricalSeries",
)

epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2")
```

When `stim_times` is omitted, ERPy inspects the NWB trials table for electrical-stimulation-site columns such as `electrical_stimulation_site`, `stim_pair`, `stimulation_site`, or `stimulus_pair`. Numeric event times are treated as seconds relative to NWB session start. Datetime event times are preserved and aligned to the NWB `session_start_time`. Generic cognitive task stimulus columns are not treated as electrical stimulation events unless explicit `stim_times` are provided.

Configured projects can also point `raw_metadata.csv` directly to `.nwb` files. Add `nwb_series` when a file contains more than one `ElectricalSeries`.

## Events

Convert provided timestamps:

```python
events = ep.create_events_from_timestamps(["2026-01-01T00:00:02", "2026-01-01T00:00:04"])
events.to_csv("EXAMPLE_PATIENT/EXAMPLE_SESSION/events/STIM_A_1_2_events.csv", index=False)
```

Convert a binary marker vector:

```python
indexed, events = ep.create_events_from_binary_series(
    binary_df,
    start_datetime="2026-01-01T00:00:00",
    end_datetime="2026-01-01T00:01:00",
    event_column=0,
)
```

Detect from artifacts or EDF metadata:

```python
dl = ep.DataLoader("EXAMPLE_PATIENT")
events = ep.detect_events_from_artifacts(
    dl, "EXAMPLE_SESSION", "STIM_A_1_2",
    method="adjacent",          # auto, adjacent, consensus, matched_filter, trigger, annotations, binary
    stim_freq=0.2,
    min_peak_height=1000,
    save_events=True,
)
```

## Bad Channels And Artifact Responses

Continuous bad-channel rejection catches persistently bad contacts:

```python
report = ep.detect_bad_channels(raw_df)
clean_df = ep.reject_bad_channels(raw_df, bad_channels="auto", method="drop")
```

Epoch-level artifact QC catches trial-specific failures such as amplifier
saturation, plateau or known-rail clipping, late-response roughness-ratio
outliers, sharp transients, NaN-heavy windows, and movement/noise bursts. The
roughness ratio compares mean absolute sample gradients in late and early
poststimulus intervals; its serialized `late_high_frequency_ratio` field name
is historical and does not denote spectral or high-gamma energy:

```python
report = epochs.flag_artifacts(zscore_threshold=6)
epochs_clean = epochs.reject_artifacts(report=report, mode="drop_epoch")
```

Use `mode="nan_response"` when you want to keep trial counts but mask contaminated trial-channel responses, or `mode="drop_channel"` when a channel is dominated by artifact responses.
Channel-level QC is based on the fraction of contaminated responses and the
number of clean responses remaining; one isolated clipped trial does not
invalidate an otherwise usable channel.

The one-call analysis always runs this trial-by-channel response QC. Persistent
continuous-channel screening is optional: the default `blank_filt` pipeline
does not run it automatically. Select `blank_filt_reject` or add an explicit
`reject_bad_channels` step when that screening is appropriate, and retain its
report with the analysis provenance.

`audit_waveforms()` also quarantines an acquisition when detector-positive
channels show a spatially widespread, near-synchronous early voltage step and
shared recovery. The exported common-mode diagnostics report the channel
count, early-latency concentration, onset offset, and recovery drift so the
decision can be inspected rather than inferred from a generic artifact flag.

## Referencing

ERPy re-references through pipeline steps. A **robust common-average reference
(CAR)** is the default and is applied automatically by the standard
`blank_filt`, `blank_dec_filt`, `blank_dec_wide`, and `blank_filt_reject`
pipelines. The reference excludes stimulation contacts, known bad channels,
and sample-wise cross-contact amplitude outliers before averaging, preventing
a saturated contact from being copied into every channel. Bipolar and
Laplacian montages are available as alternatives:

```python
# Default CAR (built into "blank_filt")
epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2", pipeline="blank_filt")

# Named variants
epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2", pipeline="blank_filt_noref")       # no re-referencing
epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2", pipeline="blank_filt_bipolar")     # adjacent-contact bipolar
epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2", pipeline="blank_filt_laplacian")   # weighted Laplacian

# Custom step composition
epochs = patient.epoch(
    "EXAMPLE_SESSION", "STIM_A_1_2",
    pipeline=[
        ("artifact_blank", {"width_s": 0.003}),
        ("bandpass_filter", {"lowcut": 0.5, "highcut": 80.0, "order": 4}),
        ("commonavg_ref", {}),                 # alias: "car_ref"; skips stim contacts by default
        # ("bipolar_ref", {}),                 # X1, X2 -> "X1-X2" = X1 - X2 per lead
        # ("laplacian_ref", {"decay": 1.0}),   # distance-weighted neighbor reference
    ],
)
```

Bipolar referencing turns adjacent contacts `X1`/`X2` into a single channel `X1-X2`; Laplacian referencing subtracts a distance-weighted average of each contact's neighbors on the same lead. Common-average and Laplacian montages preserve contact names, while bipolar montages produce paired channel labels.

## Spectral Analysis

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

## Visualization

ERPy’s plotting API is available directly from an `Epochs` object:

```python
detections = epochs.detect_erp_all(min_consensus=2)
fig = epochs.plot.summary("CONTACT_B1", detections=detections)
fig.savefig("erpy_summary_example.png", dpi=300, bbox_inches="tight")
```

When the epochs are not explicitly wideband, `crowther_gamma` remains in the
long-form table with `method_available=False`, an availability reason, and no
scientific call. Pass `wideband_epochs=clean_wideband_epochs` only when that
object truly retains 70–170 Hz.

The recommended detector is `crp_energy`. It tests two necessary components on
the same artifact-cleaned trials. Amplitude-weighted waveform reproducibility
uses whole-trial sign flips of an ordered, semi-normalized cross-projection
statistic over the full response window declared before inference; the
CRP-selected duration remains descriptive. Excess energy
uses a one-sided paired sign-flip test of the mean log RMS ratio after the
matched response and baseline segments are each demeaned separately. Their
intersection-union value is \(p_{\mathrm{joint}}=\max(p_R,p_E)\); `q_joint`
applies Benjamini-Hochberg adjustment over the eligible recording-channel
family. The historical result fields `p_crp` and `p_energy` store \(p_R\) and
\(p_E\), respectively. Both contacts defining the bipolar
stimulation pair and channels with insufficient matched clean trials are
outside that family. Their unadjusted detector quantities remain auditable,
but stimulation contacts receive `qc_status="stimulation_contact_excluded"`,
no `q_joint`, and cannot enter network or travelling-wave outputs. In the
long-form detection table, the `primary_significant` column records the
BH-adjusted statistical decision before the separate response-artifact gate,
whereas `primary_qc_pass` records the combined decision. By contrast,
`AnalysisResult.primary_significant` is a convenience property that already
returns only primary rows passing artifact QC and is the recommended final
result for a one-call analysis. CRP-only, energy-only, the former
`shape_magnitude_significant` CRP-Kundu rule, and count-based `consensus_ch`
remain available as explicitly labeled descriptive comparisons.

Detection and waveform-audit tables distinguish the default components from
standalone comparator calls:

| Field | Meaning |
| --- | --- |
| `primary_reproducibility_pass`, `primary_energy_pass` | Unadjusted component decisions within the default `crp_energy` method. |
| `primary_significant` (detection table), `primary_detector_pass` (audit) | Joint BH decision before the separate contact-QC gate. |
| `primary_qc_pass` (audit) | Final intersection of the joint decision and contact-QC eligibility. |
| `comparator_crp_pass` | Standalone `crp_significance` comparator, using its own adjusted result. |
| `comparator_kundu_pass` | Standalone `kundu_rolston` comparator's published-rule call. |

Each comparator has a matching `comparator_*_available` boolean and
`comparator_*_availability_reason`. Calls use nullable booleans: missing
(`pd.NA`, empty in CSV) means not run or unavailable, not a negative result.
Reasons distinguish `not_run`, missing availability metadata, unavailable
method input, and missing required comparator evidence. Availability requires
explicit method metadata and finite applicable quantities; a boolean call alone
is insufficient. Only two available comparator calls can produce the audit
reason `comparator_crp_kundu_disagreement`; this review diagnostic does not
change detector decisions or contact eligibility.

`primary_shape_pass` and `primary_magnitude_pass` are deprecated compatibility
aliases for `comparator_crp_pass` and `comparator_kundu_pass`, respectively,
including their missing states. They do not represent the default detector's
components. Code using these aliases as ordinary booleans should migrate to the
comparator names and inspect availability first. The historical
`shape_magnitude_significant`/`shape_magnitude_detector_pass` conjunction and
consensus fields retain their existing selection semantics.

The primary result retains the historical field names `p_crp` and
`crp_statistic`; in version 1.0.0 they contain the fixed-window
reproducibility value \(p_R\) and statistic \(T_R\). The standalone
`crp_significance` comparator keeps its own selected-duration extraction-test
quantities.

```python
result = ep.detect(
    clean_epochs,
    method="crp_energy",
    response_window=(0.015, 1.0),
    baseline_window=(-1.0, -0.015),
    alpha=0.05,
    correction="fdr_bh",
    min_clean_trials=8,
    n_permutations=5000,
    canonical_energy_cv=True,
    random_state=42,
)
```

The result retains four unadjusted component-test states:
`reproducible_energetic`, `reproducible_low_energy`,
`energetic_inconsistent`, and `no_response`. They are determined by
`p_crp <= 0.05` and `p_energy <= 0.05`; the final call still uses `q_joint`
and artifact eligibility. Canonical projection energy and its fraction are
computed leave-one-trial-out at the duration selected from all clean trials
and are reported as effect sizes. Too few clean trials yield
`insufficient_data` instead of an exception.

`validation/validate_crp_energy_real_baseline.py` injects the four diagnostic
response patterns into matched prestimulus noise from a saved epoch file. It
writes only derived detector quantities and can be run with `--strict` to make
class recovery and the joint-p identity executable release checks.

To adjudicate one channel with the exact source-native quantities, use the
six-panel detector diagnostic. It shows the mean response and Keller/Hays
windows, Kundu envelope and thresholds, the descriptive CRP canonical
reconstruction and data-driven duration, the fixed-window reproducibility
test, the paired CRP-energy RMS comparison, and all detector decisions. Pass
the same artifact-masked standard and wideband
epochs used for detection when reproducing a saved result.

`ERPy.DETECTOR_QUANTITY_COLUMNS` lists the source-native output columns for
each detector. Hays N1 quantities use explicit `hays_n1_*` aliases even though
their 10-50-ms measurement window is shared with the Keller implementation.

```python
fig = clean_epochs.plot.detectors(
    "CONTACT_B1",
    detections=detections,
    wideband_epochs=clean_wideband_epochs,
)
```

Common waveform views:

```python
epochs.plot.mean("CONTACT_B1")                         # mean +/- SEM
epochs.plot.overlay("CONTACT_B1")                      # all trials with mean
epochs.plot.heatmap("CONTACT_B1")                      # trial x time
epochs.plot.butterfly(channels=["CONTACT_B1", "CONTACT_B2"])
epochs.plot.grid(channels=["CONTACT_B1", "CONTACT_B2", "CONTACT_C1"])
epochs.plot.ranked_grid(detections, metric="peak_amplitude_uv")
epochs.plot.grouped_grid(channel_metadata, group_col="anat_label")
```

Response maps, response matrices, and graph-ready edge tables:

```python
from ERPy.response_metrics import zscore_metric_within_stim
from ERPy.viz.waveforms import plot_response_map
from ERPy.viz.networks import edges_from_detections, graph_from_edges, response_matrix_from_edges
from ERPy.viz.response_metrics import plot_within_stim_zscore_bars

plot_response_map(detections, metric="peak_amplitude_uv")

metric_zscores = zscore_metric_within_stim(
    detections.assign(stim_pair="STIM_A_1_2"),
    metric="peak_amplitude_uv",
    absolute=True,
)
plot_within_stim_zscore_bars(metric_zscores, metric="peak_amplitude_uv", stim_pair="STIM_A_1_2")

edges = edges_from_detections(
    detections.assign(stim_pair="STIM_A_1_2"),
    elec_meta=patient.elec_meta,
    metric="peak_amplitude_uv",
    significance_column="primary_qc_pass",
)
matrix = response_matrix_from_edges(edges, elec_meta=patient.elec_meta)
graph = graph_from_edges(edges)
```

CRP analysis exposes both the learned canonical response curve and the time-resolved canonical weight across the epoch, plus comparison metrics for stimulation-site contrasts:

```python
from ERPy.crp import CRPConfig, compare_crp_across_stim_sites, run_crp_all
from ERPy.viz.crp import plot_crp_site_comparison, plot_crp_summary

crp_config = CRPConfig(
    response_window=(0.015, 0.30),
    baseline_window=(-0.30, -0.05),
    permutation_n=200,
)
crp_table = run_crp_all(epochs, config=crp_config)
plot_crp_summary(epochs, "CONTACT_B1", config=crp_config)

comparison = compare_crp_across_stim_sites({"STIM_A_1_2": crp_table})
plot_crp_site_comparison(comparison, metric="within_stim_z")
```

`compare_crp_across_stim_sites` reports within-stimulation-site z-scored canonical weights, within-channel baseline-normalized CRP expression, reconstructed response RMS in microvolts, and optional permutation/null p-values. The more general `zscore_metric_within_stim` helper works with any response metric column, including peak amplitude, RMS, canonical weight, reconstructed CRP RMS, or user-defined metrics.

Network analysis, brain visualization, and evoked-response graph animations:

```python
from ERPy.graph_metrics import compute_dynamic_graph_metrics, evoked_response_edges_over_time
from ERPy.viz.graph_metrics import plot_graph_metric_heatmap, plot_graph_metric_timecourse
from ERPy.viz.networks import (
    graph_from_edges,
    plot_aggregate_evoked_response_graph,
    plot_evoked_response_graph,
    plot_glass_brain_network,
    plot_interactive_connectome,
    plot_network,
    plot_node_metric_glass_brain,
    plot_node_metric_template_brain,
    plot_response_matrix_heatmap,
    plot_template_brain_network,
    validate_mni_coordinates,
)

graph = graph_from_edges(edges)       # NetworkX DiGraph
plot_network(graph)                   # Kamada-Kawai topology view
plot_response_matrix_heatmap(edges, elec_meta=patient.elec_meta)

validate_mni_coordinates(patient.elec_meta)
plot_glass_brain_network(edges, patient.elec_meta, edge_linewidth=1.45, edge_alpha=0.82)
plot_template_brain_network(edges, patient.elec_meta)
plot_interactive_connectome(edges, patient.elec_meta)
plot_evoked_response_graph(epochs, patient.elec_meta, stim_pair="STIM_A_1_2")
plot_aggregate_evoked_response_graph(
    [
        {"epochs": epochs_a, "stim_pair": "STIM_A_1_2", "detections": detections_a},
        {"epochs": epochs_b, "stim_pair": "STIM_A_3_4", "detections": detections_b},
    ],
    patient.elec_meta,
)

edge_time = evoked_response_edges_over_time(
    [{"epochs": epochs, "stim_pair": "STIM_A_1_2", "detections": detections}],
    elec_meta=patient.elec_meta,
)
node_metrics = compute_dynamic_graph_metrics(edge_time, metrics=("hub", "authority", "in_strength", "out_strength"))
plot_graph_metric_timecourse(node_metrics, node="STIM_A1", metric="hub")       # stimulation-source hubness
plot_graph_metric_timecourse(node_metrics, node="CONTACT_B1", metric="authority") # response-contact authority
plot_graph_metric_heatmap(node_metrics, metric="hub")
plot_node_metric_template_brain(node_metrics, patient.elec_meta, metric="hub", summary="max")
plot_node_metric_glass_brain(node_metrics, patient.elec_meta, metric="hub", summary="max")
```

In the aggregate evoked-response brain graph, edge color, width, and opacity encode the selected response-magnitude metric at the current time point. The default is absolute mean evoked voltage in microvolts; custom time-indexed metric frames can animate CRP canonical weight, z-scored CRP expression, or other user-defined response metrics. Response-node size is cumulative incoming response magnitude from all rendered stimulation sites at that time point, and response-node color is the active count of ERPy significance methods designating that response significant. Stimulation sources are rendered separately as black/gold diamond markers across the network, glass-brain, cortical-surface, and animated Plotly views so source contacts remain visually distinct from response contacts.

`evoked_response_edges_over_time` converts the same animation inputs into a time-indexed edge table. `compute_dynamic_graph_metrics` then computes node metrics such as hub centrality, authority, PageRank, betweenness, in-strength, out-strength, and total strength for each sampled time point. In a directed stimulation-response graph, hubness is concentrated on stimulation/source contacts and authority is concentrated on recording-response contacts; ERPy chooses graph-metric showcase nodes accordingly and falls back to normalized weighted out-strength/in-strength when HITS is numerically degenerate. These metrics can be shown as timecourses, node x time heatmaps, max-over-epoch fsaverage cortical maps, or Nilearn glass-brain metric maps.

For opportunity-aware source-node cartography and target-distribution dynamics:

```python
from ERPy import (
    dynamic_target_distribution_metrics,
    weighted_directed_node_metrics,
)

node_roles = weighted_directed_node_metrics(
    edge_table,
    group_cols=["patient_id"],
    source_col="source",
    target_col="target",
    weight_col="response_magnitude_uv",
)

timecourse, site_summary = dynamic_target_distribution_metrics(
    target_distribution,
    group_cols=["patient_id", "stim_pair"],
    time_col="time_ms",
    target_col="anatomical_family",
    weight_col="magnitude_per_eligible_contact",
    scaffold_nodes=["ACC", "Thalamus", "Striatum/Pallidum", "PAG/PVG"],
)
```

`weighted_directed_node_metrics` expects nonnegative edge strengths. It uses
inverse strength for shortest-path quantities and the original strength for local
reaching centrality. `dynamic_target_distribution_metrics` reports normalized
participation, entropy, effective target count, scaffold weight fraction, and
Jensen-Shannon reconfiguration for one stimulated source through time. Bins with
zero total response weight are undefined rather than maximally integrated. These
source-conditioned summaries are not a substitute for whole-connectome community
analysis.

For clinically sampled networks, use `ERPy.opportunity_conditioned_reciprocity`
on the directed opportunity table. It excludes anatomical pairs for which the
reverse stimulation direction was never tested, preventing missing coverage
from being counted as evidence for a unidirectional connection.

`validate_mni_coordinates` audits electrode coordinates against the MNI152 brain mask. MNI-template brain plots keep exact electrode coordinates and, by default, drop/report contacts outside the template brain rather than projecting them onto the cortex. `plot_glass_brain_network` uses Nilearn's MNI brain display with fine symmetric edges by default for dense stimulation maps; pass `directed=True` when arrow geometry is preferred. `plot_template_brain_network`, `plot_interactive_connectome`, `plot_evoked_response_graph`, and `plot_aggregate_evoked_response_graph` render electrodes and response edges on fsaverage cortical surfaces when MNI coordinates are available, with a lightweight shell fallback if Nilearn surfaces are unavailable. Static template-brain networks can also use explicit node-size and node-color value maps for max-over-epoch summaries. Matrix-based Nilearn connectome helpers are also available:

```bash
python -m pip install -e ".[viz]"
```

```python
from ERPy.viz.networks import ordered_adjacency_from_edges, plot_electrode_connectome, plot_region_connectome

adjacency = ordered_adjacency_from_edges(edges, elec_meta=patient.elec_meta)
display = plot_electrode_connectome(adjacency, patient.elec_meta)
```

For subject-specific cortical surface figures, provide coordinates in the same frame as the FreeSurfer reconstruction and use the optional PyVista/MNE surface renderer:

```python
from ERPy.viz.networks import plot_surface_connectome

plot_surface_connectome(
    edges,
    patient.elec_meta,
    subjects_dir="./freesurfer_subjects",
    subject="sub-EXAMPLE_PATIENT",
    output_png="surface_connectome.png",
)
```

The lead [quick-start notebook](notebooks/examples/00_quickstart.ipynb) uses an
actual deidentified recording from the CNS/ACC–PAG cohort, with source trial QC,
waveforms, and single-contact reproducibility/energy results. Its stored figures
are viewable directly on GitHub; rerunning requires the authorized local trial
export configured through `ERPY_COHORT_EXAMPLE_DIR`. Restricted signals and
private source locators are not distributed. Compact notebooks 01 through 06
under [`notebooks/examples`](notebooks/examples) cover every public visualization
family using deterministic synthetic data. The gallery index maps every
plotting function to a notebook and identifies optional Nilearn, Plotly, MNE,
or PyVista requirements. Notebook 07 runs a compact public Miller/Hermes
OpenNeuro `ds003708` example and includes its input provenance, detector
tables, and figures. Raw downloads remain outside version control.

[Notebook 08](notebooks/examples/08_n1_development.ipynb) applies the secondary
negative-N1 development classifier to actual public `ds004774` derived features
and matching stored inference values. It verifies source/model hashes, uses a
model that excluded the example participant, and demonstrates portable model
save/load. The example needs no raw download or scikit-learn. Notebook outputs
retain their actual execution versions; earlier examples are not relabeled as
having been executed under the current release candidate.

## Label-free negative-N1 detection

`ERPy.n1_detection.detect_n1_family` is the proposed N1 method. It requires no
training labels or fitted model: an interior negative peak in the fixed early
window must exceed 3.4 times the mean-baseline sample SD, with a 50 µV SD floor.
The original early projection–energy joint p becomes 1 when this gate fails,
then BH is recomputed over the complete original finite contact family.
Unannotated contacts remain in that family. The general polarity-invariant
detector keeps its existing defaults.

On the previously examined paired public cohort, the fixed rule had 54.80%
sensitivity, 98.50% specificity and 84.60% PPV. Sensitivity was lower than
archived ER-detect; paired intervals did not establish higher specificity or
PPV. Its p/q values concern the original response conjunction, not N1 truth.
See the [label-free guide](docs/N1_LABEL_FREE.md) for the checked array API,
complete-family public example, uncertainty, synthetic checks and limitations.

## Secondary supervised N1 development models

`ERPy.n1` retains separate supervised negative-N1 ECoG annotation scores and decisions;
the default polarity-invariant detector and its p/q values remain unchanged.
The morphology arm includes trial-consistency descriptors; the hybrid adds
reproducibility, energy and RMS inference features. Training uses fixed C=1,
soft expert-vote targets with total weight one per record, train-only scaling,
and nested leave-one-participant-out threshold selection. Its score is not a
p value or q value, and the fitted population does not validate sEEG use.
Hybrid prediction requires checked early-window evidence. Use
`ERPy.n1.prepare_n1_hybrid_inputs(trials_uv, times_seconds)` to construct both
paths from matching arrays and retain their separate finite-trial masks; bare
inference dictionaries are rejected. The guide also provides a checksum-verified
saved-reference adapter with explicit historical trial-provenance limits.

The comparison uses archived negative-N1 outputs from ER-detect. ER-detect also
supports other polarities and detection methods, including CRP similarity across
trials; the ERPy hybrid is a separate classifier and does not use those calls as
predictors. See the [method comparison](docs/N1_DEVELOPMENT.md#how-this-relates-to-er-detect)
and [ER-detect paper](https://doi.org/10.1016/j.jneumeth.2025.110389).

This is corrected post hoc development after the original external evaluation,
with all candidates and the identifier-join correction disclosed. Saved reference
inputs and models are checksum-pinned. The [N1 development guide](docs/N1_DEVELOPMENT.md)
explains the model, limitations, API and complete reproduction commands.
Runtime prediction uses the existing NumPy/SciPy dependencies; install
`python -m pip install -e ".[n1-training]"` for optional scikit-learn training.
The N1 extraction, training and secondary-summary scripts require Python
3.11+ and were validated on 3.12.5. Portable model prediction retains the
package's broader declared Python range and does not require scikit-learn.

The [optional multiple-window experiment](docs/MULTISCALE_DEVELOPMENT.md) tests
several response intervals with a correction for searching across them. Neither
equal weights nor one declared set of unequal weights established a general
improvement in a separate confirmation simulation. The single-window default
remains. Plans, source versions, result tables and two small public baseline
samples are retained so the experiment can be reproduced.

## How the synthetic benchmark relates to the real-data examples

The benchmark in
[`validation/benchmark_crp_energy.py`](validation/benchmark_crp_energy.py)
characterizes the joint projection–energy decision under one declared synthetic
model. It contains generated values exclusively and creates 2,400 independent
simulated stimulation acquisitions by
varying trial count, response-to-noise ratio, and the fraction of target trials
randomly masked after upstream QC. Each acquisition contains stationary colored noise in
four channels. A known, RMS-normalized biphasic response with small amplitude
and latency variation is added only to one designated target channel; the
other three channels remain known nulls.

In the benchmark, a **testing family** means the four simulated channels
processed together in one acquisition. It is the small synthetic counterpart
of the recording contacts considered for one multiple-testing correction in a
real stimulation acquisition; only detector-eligible contacts with finite
joint p-values enter the Benjamini-Hochberg adjustment. The adjusted set has
three contacts when random masking leaves the target with too few usable trials
and four contacts otherwise (600 and 1,800 of the 2,400 families,
respectively). Here, “family” has no kinship or cross-dataset meaning. At zero response-to-noise ratio, all four
simulated channels are null. At nonzero ratios, the target contains the
injected response and the three references remain null.

Known synthetic truth makes false calls, evaluability, and response detection
measurable under these specific conditions. The model uses stationary,
independent Gaussian AR(1) noise, one biphasic morphology, four-channel
families, and signal-independent random target-trial removal. The removal step
tests detector behavior after missingness; it does not evaluate the accuracy
of artifact identification. The public `ds003708` example shows that ERPy can
read and analyze an independently hosted real recording, while a governed
deidentified example shows the same end-to-end workflow on local data. The
three sources serve distinct roles and are reported separately.

The four-channel family is smaller than the 30- and 143-contact real examples.
For `n <= 12`, the exact reproducibility component has minimum attainable
`p_R = 2**(-(n - 1))`; with eight clean trials an isolated BH discovery at
alpha 0.05 is arithmetically possible only through six contacts. With 5,000
Monte Carlo draws, the minimum estimated p-value is `1 / 5001`, making an
isolated discovery arithmetically possible through 250 contacts. The exact
energy component has minimum `p_E = 2**(-n)`, so the exact conjunction floor
is governed by `p_R`. These are resolution bounds rather than power
guarantees. Plan trial counts, family size, and randomization budget together.
The exact implementation omits the observed all-positive assignment from the
matrix calculation, counts it explicitly through a plus-one numerator, and
uses a scale-aware floating-point tolerance for numerically tied statistics.

The saved target-level component comparison uses the primary row's own component values.
Among 360 evaluable noise-only targets, `p_R <= 0.05`, `p_E <= 0.05`, and
`max(p_R, p_E) <= 0.05` occurred 19, 21, and 1 times, respectively; no joint
call survived family adjustment. Among 1,440 evaluable injected targets, the
corresponding recovery counts were 1,423, 1,302, and 1,298 before 1,205 joint
calls survived adjustment. The separately named `crp_only_fdr` and
`paired_rms_fdr` benchmark rows are source-informed comparator
implementations, rather than these primary component tests.

A focused companion benchmark in
[`validation/benchmark_crp_energy_composite_nulls.py`](validation/benchmark_crp_energy_composite_nulls.py)
tests both one-component-null branches with 1,000 independently seeded
four-contact families per branch and 12 trials per contact. Both tests use
exact enumeration. In the projection-alternative/energy-null branch, the raw
joint and adjusted target call rates were 5.6% and 1.9%; in the
energy-alternative/projection-null branch, they were 5.1% and 1.0%. Complete
results, Wilson intervals, provenance, and checksums are retained in
`release_artifacts/benchmark/composite_null_stress_test/` when the command is
run.

The standalone selected-duration CRP p-value is explicitly exploratory: its
duration optimization and shared-trial projection t-test were anti-calibrated
under the all-null simulation. The paired-Wilcoxon RMS field now separately
demeans its matched segments, correcting baseline-centering leakage observed
during release review. The reader-facing RMS binary field continues to share
the peak-z rule with the peak-amplitude comparator and supplies no additional
independent vote.

## Public-data portability targets

The validation script in [`validation/validate_public_spes.py`](validation/validate_public_spes.py) is designed for small, reproducible OpenNeuro slices:

| Dataset | Why it matters |
| --- | --- |
| `ds003708` | Mayo one-patient iEEG SPES dataset used around basis profile curve work; open BrainVision signal derivative. |
| `ds004080` | 74-subject CCEP ECoG SPES dataset across age 4-51; open raw BrainVision signals. |

From a source checkout, install the validation dependencies once and then run:

```bash
python -m pip install -e ".[validation]"
python validation/validate_public_spes.py --dataset ds003708 --min-events 8 --max-events 8
python validation/validate_public_spes.py --dataset ds004080 --min-events 8 --max-events 8
```

Eight artifact-clean trials are required by the primary CRP-energy detector.
Smaller slices can still test download, event, epoch, and QC interoperability,
but their primary detector result is correctly reported as insufficient data.

Generate high quality visualizations from a larger real-data sample spanning
multiple stimulation sites:

```bash
python validation/validate_public_spes.py \
  --dataset all \
  --sites-per-dataset 3 \
  --max-events 10 \
  --max-channels 48 \
  --figures
```

These high quality visualizations include waveform panels with
post-artifact zero-anchor verification, response-ranked grids,
anatomy-organized grids, response maps, exploratory within-stimulation metric
z-score plots, CRP canonical response views, response matrices, network
summaries, and time-resolved brain views.

Large datasets are intentionally sampled by stimulation event window; ERPy should not require downloading a full multi-GB BIDS run to prove that open raw signal loading, event creation, epoching, QC, and detection work.


## Detector–annotation agreement figures

The [agreement guide](docs/ANNOTATION_AGREEMENT.md) explains the current broad-response Figure 6 and focused N1 Figure 7. The broad detector is compared with released negative-N1 ratings using equal weight per contact/pair record and participant-level uncertainty. Its four recorded examples retain their original q values and source hashes. N1-negative ratings are not a complete reference for every evoked response.

Reproduce the aggregate tables with `python validation/analyze_annotation_agreement.py --verify`. The [figure directory](validation/manuscript_figures/README.md) provides current artwork and plotting commands. Existing model fits, calls, thresholds, notebooks and historical result files remain unchanged.

## Citation

```bibtex
@software{kanungo_erpy_2026,
  author = {Kanungo, Ishan},
  title = {ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis},
  year = {2026},
  version = {1.1.0rc1},
  url = {https://github.com/kaniscode/ERPy},
  license = {MIT}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
