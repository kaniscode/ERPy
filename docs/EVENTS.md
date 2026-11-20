# Stimulation events

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

Every analysis is locked to stimulation onset times. ERPy stores them in one
canonical CSV per stimulation pair,
`<procdata>/<patient>/<session>/events/<stim_pair>_events.csv`, with a `times`
column of datetimes or sample-aligned numeric seconds. `patient.epoch(...)`
and `patient.analyze(...)` create this file automatically when it is missing;
the functions below create or inspect it explicitly.

## From a patient

```python
events = patient.detect_events("EXAMPLE_SESSION", "STIM_A_1_2", method="auto", stim_freq=0.2)
```

`method` is one of `auto`, `adjacent`, `consensus`, `matched_filter`,
`trigger`, `annotations`, or `binary`. The result is saved to the canonical
location unless `save_events=False`.

## From known timestamps

Convert provided timestamps:

```python
events = ep.create_events_from_timestamps(["2026-01-01T00:00:02", "2026-01-01T00:00:04"])
events.to_csv("EXAMPLE_PATIENT/EXAMPLE_SESSION/events/STIM_A_1_2_events.csv", index=False)
```

## From a binary marker vector

Convert a binary marker vector:

```python
indexed, events = ep.create_events_from_binary_series(
    binary_df,
    start_datetime="2026-01-01T00:00:00",
    end_datetime="2026-01-01T00:01:00",
    event_column=0,
)
```

## From stimulation artifacts or EDF metadata

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

Always confirm that detected event times align with the stimulation artifacts
before interpreting results; `epochs.plot.overlay(channel)` on a contact near
the stimulation site makes misaligned events obvious. If no events are found,
see [troubleshooting](TROUBLESHOOTING.md#no-events-are-detected).

## See also

- [Analysis workflow](WORKFLOW.md)
- [Preprocessing and quality control](PREPROCESSING_AND_QC.md)
