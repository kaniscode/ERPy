# Data formats and project configuration

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

ERPy reads per-stimulation CSV exports, EDF/BDF, BrainVision, and NWB
recordings. A project is described by one YAML configuration and three small
metadata tables; BIDS and NWB projects can generate these automatically.

## The project contract

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

Common alternative column names (for example `subject`, `session`, `file`,
`onset`, `name`, `region`) are recognized and mapped to the canonical names
above. `erpy-init` creates a configuration and blank tables to fill in; see
[installation](INSTALLATION.md#create-a-project).

The loaded tables are available as `patient.raw_meta`, `patient.stim_meta`
and `patient.elec_meta`, and the resolved locations as attributes of
`patient.dataloader` (`rawdata_path`, `procdata_path`, `patient_path`,
`config_path`).

## BIDS and OpenNeuro

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

The [BIDS, OpenNeuro, and NWB guide](BIDS_OPENNEURO.md) describes the importer in more detail and lists its current limits.

## NWB and DANDI

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

## See also

- [BIDS, OpenNeuro, and NWB in depth](BIDS_OPENNEURO.md)
- [Creating stimulation events](EVENTS.md)
- [Data privacy and safe sharing](DATA_PRIVACY.md)
