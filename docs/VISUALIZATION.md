# Visualization

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

Plots for a set of epochs are reached through `epochs.plot`; table- and
network-level plots are functions in `ERPy.viz`. The
[visualization gallery](VISUALIZATION_GALLERY.md) maps every plotting function
to an executed example notebook.

ERPy’s plotting API is available directly from an `Epochs` object:

```python
detections = epochs.detect_all(min_consensus=2)
fig = epochs.plot.summary("CONTACT_B1", detections=detections)
fig.savefig("erpy_summary_example.png", dpi=300, bbox_inches="tight")
```

## Waveform views

```python
epochs.plot.mean("CONTACT_B1")                         # mean +/- SEM
epochs.plot.overlay("CONTACT_B1")                      # all trials with mean
epochs.plot.heatmap("CONTACT_B1")                      # trial x time
epochs.plot.butterfly(channels=["CONTACT_B1", "CONTACT_B2"])
epochs.plot.grid(channels=["CONTACT_B1", "CONTACT_B2", "CONTACT_C1"])
epochs.plot.ranked_grid(detections, metric="peak_amplitude_uv")
epochs.plot.grouped_grid(channel_metadata, group_col="anat_label")
```

## Response maps, matrices, and edge tables

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

## Detector and CRP figures

The six-panel detector diagnostic and the CRP summary plots are described with
their detectors in the [response detection guide](DETECTION.md#adjudicating-one-channel).
Spectral plots are listed in the [spectral guide](SPECTRAL.md), and brain and
network figures in the [network guide](NETWORKS.md).

## Saving figures

Every plotting function returns a Matplotlib figure or axes (or a Plotly figure
for interactive views), so figures are saved with the library's own methods,
for example `fig.savefig("figure.png", dpi=300, bbox_inches="tight")`.
[Notebook 06](../notebooks/examples/06_exporting_figures.ipynb) covers layout
checks and reproducible PNG/PDF export.

## See also

- [Visualization gallery (every plotting function)](VISUALIZATION_GALLERY.md)
- [Network analysis and brain visualization](NETWORKS.md)
- [Example notebooks](../notebooks/examples/README.md)
