# Preprocessing and quality control

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

Preprocessing is a named or custom sequence of steps applied to the continuous
recording before epoching. Quality control then acts at two levels: contacts
that are bad throughout a recording, and individual trial-by-channel responses
contaminated by artifact.

## Pipelines

A pipeline is an ordered list of `(step, parameters)` tuples. The standard
pipelines are available by name (`ep.STANDARD_PIPELINES` lists them):

| Name | Steps |
| --- | --- |
| `blank_filt` (default) | 3 ms artifact blanking, 60 Hz notch with harmonics, 0.1-50 Hz band-pass, robust common-average reference |
| `blank_filt_noref` | As `blank_filt`, without re-referencing |
| `blank_filt_bipolar` | As `blank_filt`, with an adjacent-contact bipolar montage |
| `blank_filt_laplacian` | As `blank_filt`, with a distance-weighted Laplacian montage |
| `blank_dec_filt` | As `blank_filt`, decimated to 500 Hz after blanking |
| `blank_dec_wide` | As `blank_dec_filt`, with a 0.1-180 Hz band-pass |
| `blank_filt_reject` | Automatic persistent bad-channel rejection, then `blank_filt` |
| `raw`, `none` | No preprocessing |

The available steps are `artifact_blank`, `notch_filter`, `bandpass_filter`,
`decimate`, `commonavg_ref` (alias `car_ref`), `bipolar_ref`, `laplacian_ref`,
and `reject_bad_channels`.

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

## Bad channels and artifact responses

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

## Zero-time anchoring

Epochs are baseline-window corrected and, by default, recentered using a short window after a **common** anchor shared across trials. ERPy searches the trial-averaged peri-onset artifact envelope for a sustained amplitude-and-slope return toward baseline. If no settled interval is found, it uses a reported amplitude-return, quietest-sample, or peak-neighbor fallback. The anchor report records the selected method and residual values for inspection. Subtracting a post-anchor window sets an amplitude reference; it does not establish a flat waveform, artifact removal, or physiological baseline recovery. Time 0 remains the stimulation-onset reference. Audit the anchor and residual directly:

```python
epochs.zero_time_report()
epochs.post_artifact_anchor_report()
epochs.detect_zero_time_artifacts(zero_time=0.0)
```

## See also

- [Methods: preprocessing, anchor, and QC definitions](METHODS.md)
- [Response detection](DETECTION.md)
- [Troubleshooting: too few clean trials](TROUBLESHOOTING.md#too-few-clean-trials-remain)
