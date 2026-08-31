# BIDS, OpenNeuro, And NWB In ERPy

ERPy is intended to feel natural for stereo-EEG and ECoG stimulation studies. If your project is already BIDS-like or NWB-based, you should not need to build ERPy metadata by hand.

## One-Minute Path

```python
import ERPy as ep

patient = ep.Patient.from_bids(
    bids_root="./open_data/ds004080",
    subject="ccepAgeUMCU01",
    session="1",
    task="SPESclin",
    output_root="./erpy_ds004080",
)

epochs = patient.epoch("1", "PT_1_2")
detections = epochs.detect_erp_all()
fig = epochs.plot.summary("PT03", detections=detections)
```

For NWB/DANDI-style files:

```python
import ERPy as ep

patient = ep.Patient.from_nwb(
    "./recordings/example_recording.nwb",
    output_root="./erpy_nwb",
    patient_id="EXAMPLE_PATIENT",
    session_id="EXAMPLE_SESSION",
    stim_pair="STIM_A_1_2",
    series_name="ElectricalSeries",
)

epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2")
```

## What The Importer Does

`Patient.from_bids(...)` calls `import_bids_project(...)`, which:

- discovers BIDS iEEG files under `sub-*/ses-*/ieeg/`;
- recognizes BrainVision (`.vhdr/.eeg/.vmrk`), EDF/BDF, CSV, and TSV;
- reads BIDS `*_events.tsv` tables;
- converts `electrical_stimulation_site` labels like `PT01-PT02` into ERPy `stim_pair` names like `PT_1_2`;
- writes canonical ERPy event CSVs under `<procdata>/<patient>/<session>/events/`;
- writes ERPy metadata stubs under `<output_root>/metadata/`;
- keeps large raw data in place by default.

The resulting project can be used exactly like a hand-authored ERPy project:

```python
patient = ep.Patient("ccepAgeUMCU01", config_path="./erpy_ds004080/config.yaml")
epochs = patient.epoch("1", "PT_1_2")
```

## NWB Import

`Patient.from_nwb(...)` and `DataLoader.from_nwb(...)` create the same ERPy metadata/config structure from one NWB file. ERPy reads `ElectricalSeries` data directly, writes electrode metadata from the NWB electrode table, and creates canonical event CSVs from either provided `stim_times` or electrical-stimulation columns in the NWB trials table.

Use `series_name` when the file contains more than one `ElectricalSeries`. Numeric event times are interpreted as seconds relative to NWB session start. Datetime event times are aligned to `session_start_time`.

## Why This Matters

Open stimulation datasets are not uniform. Some use datetimes, some use seconds from recording start, some use BrainVision, some use EDF, and some use NWB. ERPy’s job is to normalize those differences into one clinical-research workflow:

```text
open data folder -> Patient -> events -> epochs -> QC -> detections -> figures / brain graphs
```

## Current Limits

- BrainVision support currently assumes multiplexed `IEEE_FLOAT_32`, which covers the public SPES datasets used in validation.
- NWB stimulation events are inferred from electrical-stimulation-like trials-table columns; for unusual schemas or cognitive task files, pass `stim_pair` and `stim_times` explicitly.
- ERPy does not infer anatomy beyond what `*_electrodes.tsv` or user metadata provide.
- Surface brain renderings require electrode coordinates in the same coordinate frame as the chosen FreeSurfer/MNE surface.
