# ERPy 1.0.0 API reference

This reference is generated from the released public signatures and docstrings.
It documents the stable `ERPy` import surface and every exported visualization.
Scientific definitions and defaults are explained in [METHODS.md](METHODS.md).

## Core API (`ERPy`)

### Classes

### `Analysis`

```python
Analysis(pipeline) -> 'None'
```

High-level analysis and visualization helpers bound to a pipeline.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `pipeline` | not specified | required |

**Returns:** None.

#### Public members

##### `Analysis.adjacency_from_edges`

```python
adjacency_from_edges(self, edges: 'pd.DataFrame', **kwargs) -> 'pd.DataFrame'
```

Build an electrode-level adjacency matrix from an edge table.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.band_power`

```python
band_power(self, epochs, bands=None, channels=None, **kwargs) -> 'pd.DataFrame'
```

Summarize spectral power within named frequency bands.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `bands` | not specified | `None` |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.coherence_matrix`

```python
coherence_matrix(self, epochs, band=(8.0, 13.0), channels=None, **kwargs) -> 'pd.DataFrame'
```

Compute a band-limited channel coherence matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `band` | not specified | `(8.0, 13.0)` |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.compare_crp_across_stim_sites`

```python
compare_crp_across_stim_sites(self, crp_tables, **kwargs) -> 'pd.DataFrame'
```

Compare CRP metrics across stimulation sites.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `crp_tables` | not specified | required |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.compare_groups`

```python
compare_groups(self, *args: 'Any', **kwargs: 'Any') -> 'pd.DataFrame'
```

Run ERPy's tabular group-comparison helper.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `args` | Any | required |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Analysis.compute_dynamic_graph_metrics`

```python
compute_dynamic_graph_metrics(self, edge_time_table: 'pd.DataFrame', **kwargs) -> 'pd.DataFrame'
```

Compute graph metrics for successive bins of an edge-time table.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edge_time_table` | pd.DataFrame | required |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.compute_psd`

```python
compute_psd(self, epochs, channels=None, **kwargs) -> 'pd.DataFrame'
```

Estimate power spectral density for selected channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.compute_tfr`

```python
compute_tfr(self, epochs, channel: 'str', config: 'SpectralConfig | None' = None, **kwargs)
```

Compute a time-frequency representation for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `kwargs` | not specified | required |

##### `Analysis.edges_from_detections`

```python
edges_from_detections(self, detections: 'pd.DataFrame', **kwargs) -> 'pd.DataFrame'
```

Convert channel detections into anatomically annotated directed edges.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.evoked_graph_metric_timecourse`

```python
evoked_graph_metric_timecourse(self, runs: 'list[dict]', **kwargs) -> 'pd.DataFrame'
```

Derive a graph-metric time course directly from epoch runs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `runs` | list[dict] | required |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.evoked_response_edges_over_time`

```python
evoked_response_edges_over_time(self, runs: 'list[dict]', **kwargs) -> 'pd.DataFrame'
```

Build time-resolved directed response edges from epoch runs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `runs` | list[dict] | required |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.graph_from_edges`

```python
graph_from_edges(self, edges: 'pd.DataFrame', **kwargs)
```

Convert a directed edge table to a NetworkX graph.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.graph_from_matrix`

```python
graph_from_matrix(self, matrix: 'pd.DataFrame', threshold: 'float' = 0.0)
```

Convert an adjacency matrix to a thresholded NetworkX graph.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `threshold` | float | `0.0` |

##### `Analysis.heatmap_epochs`

```python
heatmap_epochs(self, epochs, channel: 'str', **kwargs)
```

Plot single-trial activity for one channel as a heat map.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `kwargs` | not specified | required |

##### `Analysis.install_standard_scripts`

```python
install_standard_scripts(cls) -> 'None'
```

Reset external analysis-script hooks to ERPy's built-in execution.

**Returns:** None.

##### `Analysis.phase_amplitude_comodulogram`

```python
phase_amplitude_comodulogram(self, epochs, phase_channel: 'str', amp_channel: 'str | None' = None, **kwargs)
```

Compute phase-amplitude coupling across frequency pairs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `kwargs` | not specified | required |

##### `Analysis.phase_amplitude_coupling`

```python
phase_amplitude_coupling(self, epochs, phase_channel: 'str', amp_channel: 'str | None' = None, **kwargs)
```

Estimate phase-amplitude coupling for a channel pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `kwargs` | not specified | required |

##### `Analysis.phase_locking_value`

```python
phase_locking_value(self, epochs, ch_x: 'str', ch_y: 'str', config: 'SpectralConfig | None' = None)
```

Estimate time-frequency phase locking between two channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `ch_x` | str | required |
| `ch_y` | str | required |
| `config` | SpectralConfig \| None | `None` |

##### `Analysis.plot_adjacency`

```python
plot_adjacency(self, matrix: 'pd.DataFrame', **kwargs)
```

Plot an electrode-level adjacency matrix as a heat map.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_aggregate_evoked_response_graph`

```python
plot_aggregate_evoked_response_graph(self, runs: 'list[dict]', **kwargs)
```

Plot an aggregate directed response graph across runs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `runs` | list[dict] | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_comodulogram`

```python
plot_comodulogram(self, epochs, phase_channel: 'str', amp_channel: 'str | None' = None, plot_kwargs: 'dict | None' = None, **kwargs)
```

Compute and plot a phase-amplitude comodulogram.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `plot_kwargs` | dict \| None | `None` |
| `kwargs` | not specified | required |

##### `Analysis.plot_connectivity_matrix`

```python
plot_connectivity_matrix(self, matrix: 'pd.DataFrame', **kwargs)
```

Plot a square channel-connectivity matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_coordinate_network`

```python
plot_coordinate_network(self, edges: 'pd.DataFrame', **kwargs)
```

Plot response edges in electrode-coordinate space.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_crp_response_contrast`

```python
plot_crp_response_contrast(self, epochs, significant_channel: 'str', nonsignificant_channel: 'str', **kwargs)
```

Contrast representative significant and nonsignificant CRP responses.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `significant_channel` | str | required |
| `nonsignificant_channel` | str | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_crp_score_map`

```python
plot_crp_score_map(self, crp_table: 'pd.DataFrame', **kwargs)
```

Plot channel-level CRP scores and return the figure.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `crp_table` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_crp_site_comparison`

```python
plot_crp_site_comparison(self, comparison_table: 'pd.DataFrame', **kwargs)
```

Plot a CRP comparison across stimulation sites.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `comparison_table` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_crp_summary`

```python
plot_crp_summary(self, epochs, channel: 'str', config: 'CRPConfig | None' = None, **kwargs)
```

Plot the canonical response fit and diagnostics for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | CRPConfig \| None | `None` |
| `kwargs` | not specified | required |

##### `Analysis.plot_detection_summary`

```python
plot_detection_summary(self, detections: 'pd.DataFrame', **kwargs)
```

Summarize detector calls and scores in one figure.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_erp_grid`

```python
plot_erp_grid(self, epochs, channels=None, **kwargs)
```

Plot mean ERPs for several channels in a grid.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

##### `Analysis.plot_evoked_response_graph`

```python
plot_evoked_response_graph(self, epochs, **kwargs)
```

Plot the directed response graph for one epoch collection.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_glass_brain_network`

```python
plot_glass_brain_network(self, edges: 'pd.DataFrame', **kwargs)
```

Plot response edges in a glass-brain projection.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_graph_metric_heatmap`

```python
plot_graph_metric_heatmap(self, metric_table: 'pd.DataFrame', **kwargs)
```

Plot dynamic graph metrics as a node-by-time heat map.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `metric_table` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_graph_metric_timecourse`

```python
plot_graph_metric_timecourse(self, metric_table: 'pd.DataFrame', **kwargs)
```

Plot a dynamic graph metric against post-stimulus time.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `metric_table` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_interactive_connectome`

```python
plot_interactive_connectome(self, edges: 'pd.DataFrame', **kwargs)
```

Create an interactive 3-D connectome from response edges.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_network`

```python
plot_network(self, matrix: 'pd.DataFrame', threshold: 'float' = 0.0, **kwargs)
```

Threshold an adjacency matrix and plot the resulting network.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `threshold` | float | `0.0` |
| `kwargs` | not specified | required |

##### `Analysis.plot_node_metric_glass_brain`

```python
plot_node_metric_glass_brain(self, metric_table, **kwargs)
```

Plot a node metric in a glass-brain projection.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `metric_table` | not specified | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_node_metric_template_brain`

```python
plot_node_metric_template_brain(self, metric_table, **kwargs)
```

Plot a node metric on a cortical surface template.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `metric_table` | not specified | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_ordered_adjacency`

```python
plot_ordered_adjacency(self, edges: 'pd.DataFrame', **kwargs)
```

Plot adjacency with electrodes ordered by anatomical metadata.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_pac`

```python
plot_pac(self, epochs, phase_channel: 'str', amp_channel: 'str | None' = None, plot_kwargs: 'dict | None' = None, **kwargs)
```

Compute and plot a phase-amplitude coupling estimate.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `plot_kwargs` | dict \| None | `None` |
| `kwargs` | not specified | required |

##### `Analysis.plot_plv`

```python
plot_plv(self, epochs, ch_x: 'str', ch_y: 'str', config: 'SpectralConfig | None' = None, **kwargs)
```

Plot time-frequency phase locking for a channel pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `ch_x` | str | required |
| `ch_y` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `kwargs` | not specified | required |

##### `Analysis.plot_psd`

```python
plot_psd(self, epochs, channels=None, **kwargs)
```

Plot power spectral density for selected channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

##### `Analysis.plot_ranked_erp_grid`

```python
plot_ranked_erp_grid(self, epochs, detections: 'pd.DataFrame', **kwargs)
```

Plot channel ERPs ordered by a detection result table.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `detections` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_response_matrix`

```python
plot_response_matrix(self, edges: 'pd.DataFrame', **kwargs)
```

Plot an anatomically aggregated response matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_spectral_summary`

```python
plot_spectral_summary(self, epochs, channel: 'str', config: 'SpectralConfig | None' = None)
```

Create a multi-panel spectral summary for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |

##### `Analysis.plot_template_brain_network`

```python
plot_template_brain_network(self, edges: 'pd.DataFrame', **kwargs)
```

Plot response edges on a cortical surface template.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_tfr`

```python
plot_tfr(self, epochs, channel: 'str', config: 'SpectralConfig | None' = None, **kwargs)
```

Plot the time-frequency representation of one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `kwargs` | not specified | required |

##### `Analysis.plot_within_stim_zscore_bars`

```python
plot_within_stim_zscore_bars(self, data: 'pd.DataFrame', metric: 'str', **kwargs)
```

Plot within-stimulation metric z-scores as bars.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `data` | pd.DataFrame | required |
| `metric` | str | required |
| `kwargs` | not specified | required |

##### `Analysis.plot_within_stim_zscore_heatmap`

```python
plot_within_stim_zscore_heatmap(self, data: 'pd.DataFrame', metric: 'str', **kwargs)
```

Plot within-stimulation metric z-scores as a heat map.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `data` | pd.DataFrame | required |
| `metric` | str | required |
| `kwargs` | not specified | required |

##### `Analysis.plv_matrix`

```python
plv_matrix(self, epochs, band=(8.0, 13.0), channels=None, **kwargs) -> 'pd.DataFrame'
```

Compute a band-limited phase-locking matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `band` | not specified | `(8.0, 13.0)` |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.process`

```python
process(self, analysis_type: 'str', epochs=None, **kwargs: 'Any')
```

Dispatch a supported high-level analysis by name.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `analysis_type` | str | required |
| `epochs` | not specified | `None` |
| `kwargs` | Any | required |

##### `Analysis.queue_job`

```python
queue_job(self, session_ids: 'Iterable[str] | None' = None, backend: 'str | None' = None, **kwargs: 'Any')
```

Create an analysis job runner and return its dispatch summary.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_ids` | Iterable[str] \| None | `None` |
| `backend` | str \| None | `None` |
| `kwargs` | Any | required |

##### `Analysis.quick_erp`

```python
quick_erp(self, epochs, channel: 'str', **kwargs)
```

Plot the mean ERP for one channel and return the figure.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `kwargs` | not specified | required |

##### `Analysis.response_matrix_from_edges`

```python
response_matrix_from_edges(self, edges: 'pd.DataFrame', **kwargs) -> 'pd.DataFrame'
```

Aggregate an electrode edge table into a region response matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `Analysis.run_crp`

```python
run_crp(self, epochs, channel: 'str', config: 'CRPConfig | None' = None)
```

Fit canonical response parameterization for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | CRPConfig \| None | `None` |

##### `Analysis.run_crp_from_epochs`

```python
run_crp_from_epochs(self, epochs, config: 'CRPConfig | None' = None) -> 'pd.DataFrame'
```

Fit canonical response parameterization for every epoch channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `config` | CRPConfig \| None | `None` |

**Returns:** pd.DataFrame.

##### `Analysis.run_crp_from_file`

```python
run_crp_from_file(self, path: 'str | Path', config: 'CRPConfig | None' = None) -> 'pd.DataFrame'
```

Load an epoch file and run canonical response analysis on all channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `config` | CRPConfig \| None | `None` |

**Returns:** pd.DataFrame.

##### `Analysis.save_figure`

```python
save_figure(self, fig, filename: 'str', session_id: 'str | None' = None, dpi: 'int' = 300) -> 'Path'
```

Save a figure in the configured analysis directory and return its path.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `fig` | not specified | required |
| `filename` | str | required |
| `session_id` | str \| None | `None` |
| `dpi` | int | `300` |

**Returns:** Path.

##### `Analysis.use_standard_scripts`

```python
use_standard_scripts(cls, bash_type: 'str' = 'simple') -> 'None'
```

Select ERPy's built-in local analysis execution hooks.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bash_type` | str | `'simple'` |

**Returns:** None.

##### `Analysis.zscore_metric_within_stim`

```python
zscore_metric_within_stim(self, data: 'pd.DataFrame', metric: 'str', **kwargs) -> 'pd.DataFrame'
```

Standardize a response metric within each stimulation condition.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `data` | pd.DataFrame | required |
| `metric` | str | required |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

### `AnalysisResult`

```python
AnalysisResult(epochs: 'Epochs', detections: 'pd.DataFrame', artifact_report: 'ArtifactResponseReport', artifact_summary: 'pd.DataFrame', qc_detections: 'pd.DataFrame', clean_epochs: 'Epochs | None' = None, metadata: 'dict[str, Any]' = <factory>) -> None
```

One-call result for the common ERPy stimulation-response workflow.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | Epochs | required |
| `detections` | pd.DataFrame | required |
| `artifact_report` | ArtifactResponseReport | required |
| `artifact_summary` | pd.DataFrame | required |
| `qc_detections` | pd.DataFrame | required |
| `clean_epochs` | Epochs \| None | `None` |
| `metadata` | dict[str, Any] | `<factory>` |

**Returns:** None.

#### Public members

##### `AnalysisResult.primary_significant`

```python
primary_significant  # property
```

Return one CRP-energy row per primary response after artifact QC.

**Returns:** pd.DataFrame.

##### `AnalysisResult.qc_pass_detections`

```python
qc_pass_detections  # property
```

Return detection rows that pass response-artifact QC.

**Returns:** pd.DataFrame.

##### `AnalysisResult.save`

```python
save(self, output_dir: 'str | Path', prefix: 'str | None' = None) -> 'dict[str, Path]'
```

Save the standard tables and deterministic audit metadata.

The metadata record includes resolved epoch and pipeline options,
whether persistent-channel screening was configured, effective
response-artifact settings, detector windows and overrides, and one
method-native parameter record for every executed detector.

This method writes only to ``output_dir``. It does not upload data,
contact a registry, or create a release-record identifier.
Projects that use an opaque release-record label should keep its local
mapping under their own governance controls.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `output_dir` | str \| Path | required |
| `prefix` | str \| None | `None` |

**Returns:** dict[str, Path].

##### `AnalysisResult.significant`

```python
significant  # property
```

Return significant response rows from the full detection table.

**Returns:** pd.DataFrame.

### `ArtifactResponseReport`

```python
ArtifactResponseReport(table: 'pd.DataFrame') -> None
```

Trial-by-channel response-artifact quality-control report.

``table`` contains one row per evaluated trial and channel, including the
measured artifact features, the ``bad_response`` decision, and its reason.
Use ``by_channel`` to summarize burden without discarding the
trial-level audit trail.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `table` | pd.DataFrame | required |

**Returns:** None.

#### Public members

##### `ArtifactResponseReport.bad_channels`

```python
bad_channels  # property
```

Return channels whose bad-response fraction exceeds 25 percent.

**Returns:** list[str].

##### `ArtifactResponseReport.bad_epochs`

```python
bad_epochs  # property
```

Return trial identifiers containing at least one bad response.

**Returns:** list[int].

##### `ArtifactResponseReport.by_channel`

```python
by_channel(self, group_cols: 'Iterable[str] | None' = None, hard_artifact_reasons: 'Iterable[str] | None' = None) -> 'pd.DataFrame'
```

Summarize response-artifact burden by channel.

This is the table used by cohort notebooks before building ranked
response edges. It keeps saturated/rail-clipped responses auditable
without requiring downstream code to reimplement the trial-level QC
rules.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `group_cols` | Iterable[str] \| None | `None` |
| `hard_artifact_reasons` | Iterable[str] \| None | `None` |

**Returns:** pd.DataFrame.

##### `ArtifactResponseReport.to_csv`

```python
to_csv(self, path) -> 'None'
```

Write the trial-by-channel artifact report to a CSV file.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | not specified | required |

**Returns:** None.

### `BIDSImportResult`

```python
BIDSImportResult(config_path: 'Path', raw_metadata_path: 'Path', stim_metadata_path: 'Path', electrode_metadata_path: 'Path', n_raw_files: 'int', n_stim_rows: 'int', n_electrodes: 'int') -> None
```

Files and row counts produced by ``import_bids_project``.

The path fields identify the generated ERPy configuration and metadata
tables. The count fields report how many raw recordings, stimulation rows,
and electrode rows were imported for a quick completeness check.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `config_path` | Path | required |
| `raw_metadata_path` | Path | required |
| `stim_metadata_path` | Path | required |
| `electrode_metadata_path` | Path | required |
| `n_raw_files` | int | required |
| `n_stim_rows` | int | required |
| `n_electrodes` | int | required |

**Returns:** None.
### `BadChannelReport`

```python
BadChannelReport(table: 'pd.DataFrame') -> None
```

Auditable summary of continuous-recording channel quality control.

``table`` is indexed by channel and retains the measured features, Boolean
decision, and reason string used by ``detect_bad_channels``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `table` | pd.DataFrame | required |

**Returns:** None.

#### Public members

##### `BadChannelReport.bad_channels`

```python
bad_channels  # property
```

Return channel labels marked bad in the report table.

**Returns:** list[str].

##### `BadChannelReport.to_csv`

```python
to_csv(self, path) -> 'None'
```

Write the full channel-QC table to a CSV file.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | not specified | required |

**Returns:** None.

### `BootstrapInterval`

```python
BootstrapInterval(estimate: 'float', ci_low: 'float', ci_high: 'float', confidence: 'float', n_resamples: 'int', n_observations: 'int', n_top_level_clusters: 'int') -> None
```

Patient-balanced hierarchical bootstrap estimate and interval.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `estimate` | float | required |
| `ci_low` | float | required |
| `ci_high` | float | required |
| `confidence` | float | required |
| `n_resamples` | int | required |
| `n_observations` | int | required |
| `n_top_level_clusters` | int | required |

**Returns:** None.
### `ComodulogramResult`

```python
ComodulogramResult(phase_channel: 'str', amp_channel: 'str', phase_freqs: 'np.ndarray', amp_freqs: 'np.ndarray', mi: 'np.ndarray', method: 'str', bandwidth_phase: 'float' = 2.0, bandwidth_amp: 'float' = 20.0) -> None
```

Phase-amplitude coupling values across frequency-band centers.

``mi`` is arranged as amplitude-frequency by phase-frequency. Frequency
centers and bandwidths are expressed in hertz, and ``method`` records the
modulation-index estimator.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `phase_channel` | str | required |
| `amp_channel` | str | required |
| `phase_freqs` | np.ndarray | required |
| `amp_freqs` | np.ndarray | required |
| `mi` | np.ndarray | required |
| `method` | str | required |
| `bandwidth_phase` | float | `2.0` |
| `bandwidth_amp` | float | `20.0` |

**Returns:** None.
### `CRPConfig`

```python
CRPConfig(response_window: 'tuple[float, float]' = (0.015, 0.3), baseline_window: 'tuple[float, float]' = (-0.5, -0.03), min_trials: 'int' = 3, alpha: 'float' = 0.05, projection_start_s: 'float' = 0.01, projection_step_s: 'float' = 0.005, permutation_n: 'int' = 0, random_state: 'int | None' = 0, eps: 'float' = 1e-12) -> None
```

Configuration for canonical response parametrization.

The extraction follows Miller et al. (2023): semi-normalized reciprocal
trial projections determine a data-driven response duration, a balanced
half of those projections supplies the one-sided extraction test, and a
linear kernel PCA supplies the canonical response shape.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `response_window` | tuple[float, float] | `(0.015, 0.3)` |
| `baseline_window` | tuple[float, float] | `(-0.5, -0.03)` |
| `min_trials` | int | `3` |
| `alpha` | float | `0.05` |
| `projection_start_s` | float | `0.01` |
| `projection_step_s` | float | `0.005` |
| `permutation_n` | int | `0` |
| `random_state` | int \| None | `0` |
| `eps` | float | `1e-12` |

**Returns:** None.
### `CRPEnergyConfig`

```python
CRPEnergyConfig(response_window: 'tuple[float, float]' = (0.015, 1.0), baseline_window: 'tuple[float, float]' = (-1.0, -0.015), alpha: 'float' = 0.05, correction: 'str' = 'fdr_bh', min_clean_trials: 'int' = 8, n_permutations: 'int' = 5000, max_exact_trials: 'int' = 16, max_exact_reproducibility_trials: 'int' = 12, canonical_energy_cv: 'bool' = True, random_state: 'int | None' = 42, artifact_interval: 'tuple[float, float]' = (0.0, 0.015), projection_start_s: 'float' = 0.01, projection_step_s: 'float' = 0.005, eps: 'float' = 1e-12) -> None
```

Configuration for the CRP-energy conjunction.

The response and baseline windows are matched sample-for-sample. Samples
in ``artifact_interval`` are excluded before either component is fitted.
The inferential reproducibility window is the complete effective response
window; CRP duration selection is retained only for descriptive effects.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `response_window` | tuple[float, float] | `(0.015, 1.0)` |
| `baseline_window` | tuple[float, float] | `(-1.0, -0.015)` |
| `alpha` | float | `0.05` |
| `correction` | str | `'fdr_bh'` |
| `min_clean_trials` | int | `8` |
| `n_permutations` | int | `5000` |
| `max_exact_trials` | int | `16` |
| `max_exact_reproducibility_trials` | int | `12` |
| `canonical_energy_cv` | bool | `True` |
| `random_state` | int \| None | `42` |
| `artifact_interval` | tuple[float, float] | `(0.0, 0.015)` |
| `projection_start_s` | float | `0.01` |
| `projection_step_s` | float | `0.005` |
| `eps` | float | `1e-12` |

**Returns:** None.

#### Public members

##### `CRPEnergyConfig.validate`

```python
validate(self) -> 'None'
```

Validate scientific windows and finite inference settings.

Validation is explicit because silently coercing an invalid window,
seed, or randomization count into an ``insufficient_data`` result would
make configuration errors indistinguishable from unsuitable data.

**Returns:** None.

### `CRPEnergyResult`

```python
CRPEnergyResult(channel: 'str', significant: 'bool', classification: 'str', p_crp: 'float', p_energy: 'float', p_joint: 'float', q_joint: 'float', crp_significant: 'bool', energy_significant: 'bool', crp_statistic: 'float', energy_statistic: 'float', rms_response: 'float', rms_baseline: 'float', rms_ratio_db: 'float', canonical_energy: 'float', canonical_energy_fraction: 'float', response_duration: 'float', n_trials_total: 'int', n_trials_clean: 'int', canonical_waveform: 'np.ndarray', canonical_waveform_times: 'np.ndarray', trial_coefficients: 'np.ndarray', clean_trial_indices: 'np.ndarray', crp_explained_variance: 'float', crp_snr: 'float', reproducibility_test_exact: 'bool', reproducibility_n_randomizations: 'int', energy_test_exact: 'bool', energy_n_permutations: 'int', response_window: 'tuple[float, float]', baseline_window: 'tuple[float, float]', qc_status: 'str', parameters: 'str', detector_version: 'str' = '1.0.0', notes: 'str' = '') -> None
```

One channel's unadjusted reproducibility-energy conjunction result.

For backward compatibility, ``p_crp`` and ``crp_statistic`` store the
fixed-window reproducibility value `p_R` and statistic `T_R`.
They are distinct from the standalone, selected-duration CRP comparator.
The canonical waveform, response duration, coefficients, SNR, and
explained fraction remain descriptive CRP quantities.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `significant` | bool | required |
| `classification` | str | required |
| `p_crp` | float | required |
| `p_energy` | float | required |
| `p_joint` | float | required |
| `q_joint` | float | required |
| `crp_significant` | bool | required |
| `energy_significant` | bool | required |
| `crp_statistic` | float | required |
| `energy_statistic` | float | required |
| `rms_response` | float | required |
| `rms_baseline` | float | required |
| `rms_ratio_db` | float | required |
| `canonical_energy` | float | required |
| `canonical_energy_fraction` | float | required |
| `response_duration` | float | required |
| `n_trials_total` | int | required |
| `n_trials_clean` | int | required |
| `canonical_waveform` | np.ndarray | required |
| `canonical_waveform_times` | np.ndarray | required |
| `trial_coefficients` | np.ndarray | required |
| `clean_trial_indices` | np.ndarray | required |
| `crp_explained_variance` | float | required |
| `crp_snr` | float | required |
| `reproducibility_test_exact` | bool | required |
| `reproducibility_n_randomizations` | int | required |
| `energy_test_exact` | bool | required |
| `energy_n_permutations` | int | required |
| `response_window` | tuple[float, float] | required |
| `baseline_window` | tuple[float, float] | required |
| `qc_status` | str | required |
| `parameters` | str | required |
| `detector_version` | str | `'1.0.0'` |
| `notes` | str | `''` |

**Returns:** None.

#### Public members

##### `CRPEnergyResult.to_record`

```python
to_record(self, *, include_arrays: 'bool' = True) -> 'dict[str, Any]'
```

Return a detector-table record with optionally compact array fields.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `include_arrays` | bool | `True` |

**Returns:** dict[str, Any].

### `CRPResult`

```python
CRPResult(channel: 'str', significant: 'bool', score: 'float', p_value: 'float', threshold: 'float', canonical_waveform: 'np.ndarray', times: 'np.ndarray', projections: 'np.ndarray', response_window: 'tuple[float, float]', baseline_window: 'tuple[float, float]', response_duration_s: 'float' = nan, t_value: 'float' = nan, alpha_prime_uv: 'np.ndarray | None' = None, residual_rms_uv: 'np.ndarray | None' = None, snr: 'np.ndarray | None' = None, explained_variance: 'np.ndarray | None' = None, cross_projections: 'np.ndarray | None' = None, projection_times: 'np.ndarray | None' = None, mean_projection_profile: 'np.ndarray | None' = None, full_window_t_value: 'float' = nan, full_window_p_value: 'float' = nan, canonical_weight_times: 'np.ndarray | None' = None, canonical_weight_mean: 'np.ndarray | None' = None, canonical_weight_sem: 'np.ndarray | None' = None, canonical_weight_z: 'np.ndarray | None' = None, baseline_weight_mean: 'float' = nan, baseline_weight_std: 'float' = nan, max_canonical_weight_z: 'float' = nan, reconstruction_rms_uv: 'float' = nan, normalized_reconstruction_energy: 'float' = nan, energy_z: 'float' = nan, permutation_p_value: 'float' = nan, null_distribution: 'np.ndarray | None' = None) -> None
```

Canonical response parametrization result for one recording channel.

The object retains the fitted waveform, reciprocal trial projections,
data-driven response duration, extraction-test quantities, reconstruction
diagnostics, and optional permutation null. Time windows and durations are
in seconds; amplitude and residual fields carrying an ``_uv`` suffix are
in microvolts. ``significant`` and ``p_value`` describe the unadjusted CRP
extraction test, not the separate CRP-energy conjunction.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `significant` | bool | required |
| `score` | float | required |
| `p_value` | float | required |
| `threshold` | float | required |
| `canonical_waveform` | np.ndarray | required |
| `times` | np.ndarray | required |
| `projections` | np.ndarray | required |
| `response_window` | tuple[float, float] | required |
| `baseline_window` | tuple[float, float] | required |
| `response_duration_s` | float | `nan` |
| `t_value` | float | `nan` |
| `alpha_prime_uv` | np.ndarray \| None | `None` |
| `residual_rms_uv` | np.ndarray \| None | `None` |
| `snr` | np.ndarray \| None | `None` |
| `explained_variance` | np.ndarray \| None | `None` |
| `cross_projections` | np.ndarray \| None | `None` |
| `projection_times` | np.ndarray \| None | `None` |
| `mean_projection_profile` | np.ndarray \| None | `None` |
| `full_window_t_value` | float | `nan` |
| `full_window_p_value` | float | `nan` |
| `canonical_weight_times` | np.ndarray \| None | `None` |
| `canonical_weight_mean` | np.ndarray \| None | `None` |
| `canonical_weight_sem` | np.ndarray \| None | `None` |
| `canonical_weight_z` | np.ndarray \| None | `None` |
| `baseline_weight_mean` | float | `nan` |
| `baseline_weight_std` | float | `nan` |
| `max_canonical_weight_z` | float | `nan` |
| `reconstruction_rms_uv` | float | `nan` |
| `normalized_reconstruction_energy` | float | `nan` |
| `energy_z` | float | `nan` |
| `permutation_p_value` | float | `nan` |
| `null_distribution` | np.ndarray \| None | `None` |

**Returns:** None.

#### Public members

##### `CRPResult.canonical_response_curve`

```python
canonical_response_curve  # property
```

Return the fitted canonical waveform as a compatibility alias.

**Returns:** np.ndarray.

### `FixedWindowReproducibilityResult`

```python
FixedWindowReproducibilityResult(statistic: 'float', p_value: 'float', exact: 'bool', n_randomizations: 'int') -> None
```

Result of the trial-level fixed-window reproducibility randomization.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `statistic` | float | required |
| `p_value` | float | required |
| `exact` | bool | required |
| `n_randomizations` | int | required |

**Returns:** None.
### `DataLoader`

```python
DataLoader(patient_id: 'str', config_path: 'str | Path | None' = None, load_data: 'bool' = False, create_tree: 'bool' = True, verbose: 'bool' = True) -> 'None'
```

Configuration and metadata access for one ERPy patient.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `patient_id` | str | required |
| `config_path` | str \| Path \| None | `None` |
| `load_data` | bool | `False` |
| `create_tree` | bool | `True` |
| `verbose` | bool | `True` |

**Returns:** None.

#### Public members

##### `DataLoader.bad_channels`

```python
bad_channels(self, session_id: 'str | None' = None) -> 'list[str]'
```

Return electrode labels explicitly marked bad in project metadata.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str \| None | `None` |

**Returns:** list[str].

##### `DataLoader.clear_memory_cache`

```python
clear_memory_cache(self) -> 'None'
```

Discard the currently retained stimulation window from memory.

**Returns:** None.

##### `DataLoader.create_session_tree`

```python
create_session_tree(self) -> 'None'
```

Create the standard raw, event, epoch, analysis, and log folders.

**Returns:** None.

##### `DataLoader.from_bids`

```python
from_bids(cls, bids_root: 'str | Path', subject: 'str', output_root: 'str | Path | None' = None, session: 'str | None' = None, task: 'str | None' = None, patient_id: 'str | None' = None, copy_raw: 'bool' = False, derivatives: 'bool' = True, **kwargs: 'Any') -> "'DataLoader'"
```

Import one BIDS iEEG subject and return its configured data loader.

ERPy metadata tables and configuration are created under
``output_root``; raw recordings remain in place unless ``copy_raw`` is
enabled.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bids_root` | str \| Path | required |
| `subject` | str | required |
| `output_root` | str \| Path \| None | `None` |
| `session` | str \| None | `None` |
| `task` | str \| None | `None` |
| `patient_id` | str \| None | `None` |
| `copy_raw` | bool | `False` |
| `derivatives` | bool | `True` |
| `kwargs` | Any | required |

**Returns:** 'DataLoader'.

##### `DataLoader.from_nwb`

```python
from_nwb(cls, nwb_path: 'str | Path', output_root: 'str | Path | None' = None, patient_id: 'str | None' = None, session_id: 'str | None' = None, stim_pair: 'str | None' = None, stim_times: 'Any | None' = None, stim_duration_s: 'float' = 0.001, series_name: 'str | None' = None, copy_raw: 'bool' = False, **kwargs: 'Any') -> "'DataLoader'"
```

Create a minimal ERPy project around one NWB file.

Parameters
----------
nwb_path
    Local NWB file.
output_root
    Folder where ERPy metadata, config, and processed outputs are written.
patient_id, session_id
    Optional ERPy identifiers. When omitted, ERPy uses the NWB subject
    and session identifiers when available.
stim_pair, stim_times
    Optional stimulation-site name and event times. Numeric times are
    interpreted as seconds relative to NWB session start; datetimes are
    preserved as absolute event times. If omitted, ERPy tries to infer a
    stimulation table from NWB trials.
series_name
    Name of the NWB ElectricalSeries to load. If omitted, the first
    ElectricalSeries is used.
copy_raw
    Copy the NWB into the ERPy project. Defaults to keeping large NWB
    files in place.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `nwb_path` | str \| Path | required |
| `output_root` | str \| Path \| None | `None` |
| `patient_id` | str \| None | `None` |
| `session_id` | str \| None | `None` |
| `stim_pair` | str \| None | `None` |
| `stim_times` | Any \| None | `None` |
| `stim_duration_s` | float | `0.001` |
| `series_name` | str \| None | `None` |
| `copy_raw` | bool | `False` |
| `kwargs` | Any | required |

**Returns:** 'DataLoader'.

##### `DataLoader.get_analysis_path`

```python
get_analysis_path(self, session_id: 'str | None' = None) -> 'Path'
```

Return and create the patient- or session-level analysis directory.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str \| None | `None` |

**Returns:** Path.

##### `DataLoader.get_event_path`

```python
get_event_path(self, session_id: 'str', stim_pair: 'str') -> 'Path'
```

Return the conventional CSV path for a detected event table.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |

**Returns:** Path.

##### `DataLoader.get_raw_cache_path`

```python
get_raw_cache_path(self, session_id: 'str', stim_pair: 'str', *, raw_file: 'str | Path | None' = None) -> 'Path'
```

Return the configured binary or text cache path for an acquisition.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `raw_file` | str \| Path \| None | `None` |

**Returns:** Path.

##### `DataLoader.get_raw_csv_path`

```python
get_raw_csv_path(self, session_id: 'str', stim_pair: 'str', *, raw_file: 'str | Path | None' = None) -> 'Path'
```

Return the acquisition-specific path for a text raw-data cache.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `raw_file` | str \| Path \| None | `None` |

**Returns:** Path.

##### `DataLoader.get_raw_files`

```python
get_raw_files(self, session_id: 'str | None' = None, stim_pair: 'str | None' = None, *, raw_file: 'str | Path | None' = None, stim_start: 'Any | None' = None) -> 'list[Path]'
```

Resolve raw recording paths for a session or stimulation acquisition.

``raw_file`` and ``stim_start`` disambiguate repeated acquisitions when
the metadata table contains more than one candidate.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str \| None | `None` |
| `stim_pair` | str \| None | `None` |
| `raw_file` | str \| Path \| None | `None` |
| `stim_start` | Any \| None | `None` |

**Returns:** list[Path].

##### `DataLoader.get_sampling_freq`

```python
get_sampling_freq(self, session_id: 'str | None' = None, stim_pair: 'str | None' = None) -> 'float | None'
```

Resolve the registered sampling frequency for a run or session.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str \| None | `None` |
| `stim_pair` | str \| None | `None` |

**Returns:** float \| None.

##### `DataLoader.get_session_path`

```python
get_session_path(self, session_id: 'str', kind: 'str | None' = None) -> 'Path'
```

Return a session directory or one named subdirectory within it.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `kind` | str \| None | `None` |

**Returns:** Path.

##### `DataLoader.get_stim_row`

```python
get_stim_row(self, session_id: 'str', stim_pair: 'str', *, stim_start: 'Any | None' = None) -> 'pd.Series'
```

Return the metadata row matching a session and stimulation pair.

When repeated acquisitions exist, ``stim_start`` selects the closest
registered start time.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `stim_start` | Any \| None | `None` |

**Returns:** pd.Series.

##### `DataLoader.has_stim_cache`

```python
has_stim_cache(self, session_id: 'str', stim_pair: 'str', *, raw_file: 'str | Path | None' = None, allow_legacy_cache: 'bool' = False) -> 'bool'
```

Return whether a matching on-disk stimulation cache is available.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `raw_file` | str \| Path \| None | `None` |
| `allow_legacy_cache` | bool | `False` |

**Returns:** bool.

##### `DataLoader.load_stim_data`

```python
load_stim_data(self, session_id: 'str', stim_pair: 'str', source: 'str' = 'auto', save_csv: 'bool' = True, nrows: 'int | None' = None, raw_file: 'str | Path | None' = None, stim_start: 'Any | None' = None, allow_legacy_cache: 'bool' = False) -> 'pd.DataFrame'
```

Load a stimulation window as a time-indexed DataFrame.

``source='auto'`` prefers the local per-acquisition cache and otherwise
reads a registered raw file. ``raw_file`` disambiguates repeated
stimulation acquisitions that share a session and electrode pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `source` | str | `'auto'` |
| `save_csv` | bool | `True` |
| `nrows` | int \| None | `None` |
| `raw_file` | str \| Path \| None | `None` |
| `stim_start` | Any \| None | `None` |
| `allow_legacy_cache` | bool | `False` |

**Returns:** pd.DataFrame.

##### `DataLoader.prefetch_stim_data`

```python
prefetch_stim_data(self, session_id: 'str', stim_pair: 'str', *, raw_file: 'str | Path', stim_start: 'Any | None' = None, stop_event: 'Any | None' = None) -> 'list[Path]'
```

Stage registered text exports locally using large sequential reads.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `raw_file` | str \| Path | required |
| `stim_start` | Any \| None | `None` |
| `stop_event` | Any \| None | `None` |

**Returns:** list[Path].

##### `DataLoader.prepare_raw_windows`

```python
prepare_raw_windows(self, overwrite: 'bool' = False) -> 'list[Path]'
```

Precompute registered stimulation-window caches for all sessions.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `overwrite` | bool | `False` |

**Returns:** list[Path].

##### `DataLoader.remove_stim_cache`

```python
remove_stim_cache(self, session_id: 'str', stim_pair: 'str', *, raw_file: 'str | Path | None' = None, include_legacy: 'bool' = False) -> 'list[Path]'
```

Remove matching cache files and return the paths that were deleted.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `raw_file` | str \| Path \| None | `None` |
| `include_legacy` | bool | `False` |

**Returns:** list[Path].

##### `DataLoader.stim_pairs`

```python
stim_pairs(self, session_id: 'str | None' = None) -> 'list[str]'
```

Return sorted stimulation-pair labels, optionally for one session.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str \| None | `None` |

**Returns:** list[str].

### `Epochs`

```python
Epochs(epochs_df: 'pd.DataFrame', sfreq: 'float', tmin: 'float', tmax: 'float', baseline: 'tuple[float, float] | None' = None, stim_ch: 'str | list[str] | None' = None, metadata: 'dict[str, Any]' = <factory>, zero_time: 'float | str | None' = 'post_artifact') -> None
```

Epoched stimulation data backed by a MultiIndex DataFrame.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs_df` | pd.DataFrame | required |
| `sfreq` | float | required |
| `tmin` | float | required |
| `tmax` | float | required |
| `baseline` | tuple[float, float] \| None | `None` |
| `stim_ch` | str \| list[str] \| None | `None` |
| `metadata` | dict[str, Any] | `<factory>` |
| `zero_time` | float \| str \| None | `'post_artifact'` |

**Returns:** None.

#### Public members

##### `Epochs.as_array`

```python
as_array(self) -> 'tuple[np.ndarray, np.ndarray, list[str]]'
```

Return samples, time coordinates, and channel labels.

The sample array is ordered as ``(trials, times, channels)``.

**Returns:** tuple[np.ndarray, np.ndarray, list[str]].

##### `Epochs.channels`

```python
channels  # property
```

Return channel labels in their stored column order.

**Returns:** list[str].

##### `Epochs.detect_erp`

```python
detect_erp(self, method: 'str', *args: 'Any', **kwargs: 'Any') -> 'pd.DataFrame'
```

Run one named ERP detector and return its channel-level results.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `method` | str | required |
| `args` | Any | required |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Epochs.detect_erp_all`

```python
detect_erp_all(self, *args: 'Any', **kwargs: 'Any') -> 'pd.DataFrame'
```

Run every registered ERP detector and combine the channel results.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `args` | Any | required |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Epochs.detect_zero_time_artifacts`

```python
detect_zero_time_artifacts(self, zero_time: 'float' = 0.0, artifact_z: 'float' = 6.0) -> 'pd.DataFrame'
```

Detect whether a requested zeroing sample overlaps the onset artifact.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `zero_time` | float | `0.0` |
| `artifact_z` | float | `6.0` |

**Returns:** pd.DataFrame.

##### `Epochs.flag_artifacts`

```python
flag_artifacts(self, *args: 'Any', **kwargs: 'Any')
```

Score trial-channel responses against the artifact criteria.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `args` | Any | required |
| `kwargs` | Any | required |

##### `Epochs.from_hdf`

```python
from_hdf(cls, path: 'str | Path', key: 'str' = 'epochs', *, allow_unsafe_legacy_pickle: 'bool' = False) -> "'Epochs'"
```

Load epochs and restore their audit metadata.

Files written by ERPy 1.0 and later use deterministic JSON metadata.
Older ERPy files stored metadata with Python pickle, which can execute
arbitrary code when opened. Such files are rejected unless
``allow_unsafe_legacy_pickle=True`` is supplied for a file the caller
created and trusts. Re-saving a trusted legacy file migrates it to the
safe format.

Parameters
----------
path : str or pathlib.Path
    HDF5 epoch file to read.
key : str, default ``"epochs"``
    HDF5 group containing the epoch table.
allow_unsafe_legacy_pickle : bool, default ``False``
    Explicitly permit executable pickle metadata from a trusted legacy
    file. Never enable this for an untrusted or downloaded file.

Returns
-------
Epochs
    Restored epoch data and metadata.

Raises
------
UnsafeLegacyEpochMetadataError
    If pickle metadata are present and unsafe loading was not enabled.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `key` | str | `'epochs'` |
| `allow_unsafe_legacy_pickle` | bool | `False` |

**Returns:** 'Epochs'.

##### `Epochs.get_mean_waveform`

```python
get_mean_waveform(self) -> 'pd.DataFrame'
```

Return the trial-averaged waveform for every channel.

**Returns:** pd.DataFrame.

##### `Epochs.get_sem_waveform`

```python
get_sem_waveform(self) -> 'pd.DataFrame'
```

Return the standard error across trials at each time.

**Returns:** pd.DataFrame.

##### `Epochs.get_std_waveform`

```python
get_std_waveform(self) -> 'pd.DataFrame'
```

Return the sample standard deviation across trials at each time.

**Returns:** pd.DataFrame.

##### `Epochs.n_trials`

```python
n_trials(self) -> 'int'
```

Return the number of distinct trials in the epoch table.

**Returns:** int.

##### `Epochs.plot`

```python
plot  # property
```

Return a plotting accessor bound to these epochs.

##### `Epochs.post_artifact_anchor_report`

```python
post_artifact_anchor_report(self) -> 'pd.DataFrame'
```

Return per-epoch post-artifact zeroing anchors when available.

**Returns:** pd.DataFrame.

##### `Epochs.reject_artifacts`

```python
reject_artifacts(self, *args: 'Any', **kwargs: 'Any') -> "'Epochs'"
```

Return a copy with artifact-contaminated responses rejected.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `args` | Any | required |
| `kwargs` | Any | required |

**Returns:** 'Epochs'.

##### `Epochs.rereference`

```python
rereference(self, *, method: 'str' = 'robust_mean', skipped_chs: 'Iterable[str] | None' = None, zscore_threshold: 'float' = 8.0, min_reference_channels: 'int' = 4) -> "'Epochs'"
```

Return a re-referenced copy while preserving every channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `method` | str | `'robust_mean'` |
| `skipped_chs` | Iterable[str] \| None | `None` |
| `zscore_threshold` | float | `8.0` |
| `min_reference_channels` | int | `4` |

**Returns:** 'Epochs'.

##### `Epochs.rescale`

```python
rescale(self, factor: 'float', *, signal_units: 'str' = 'uV', reason: 'str | None' = None) -> "'Epochs'"
```

Return a copy with channel values multiplied by ``factor``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `factor` | float | required |
| `signal_units` | str | `'uV'` |
| `reason` | str \| None | `None` |

**Returns:** 'Epochs'.

##### `Epochs.spectral`

```python
spectral  # property
```

Return a spectral-analysis accessor bound to these epochs.

##### `Epochs.times`

```python
times  # property
```

Return unique epoch-relative sample times in seconds.

**Returns:** np.ndarray.

##### `Epochs.to_hdf`

```python
to_hdf(self, path: 'str | Path', key: 'str' = 'epochs', *, storage_dtype: 'str | np.dtype | None' = 'float32', complevel: 'int' = 5, complib: 'str' = 'blosc:zstd', **metadata: 'Any') -> 'Path'
```

Atomically persist all trials in a compact, loss-bounded HDF5 file.

Float32 storage preserves sub-microvolt precision across the practical
iEEG range while substantially reducing read time and disk use. Data are
promoted by NumPy operations as needed after loading.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `key` | str | `'epochs'` |
| `storage_dtype` | str \| np.dtype \| None | `'float32'` |
| `complevel` | int | `5` |
| `complib` | str | `'blosc:zstd'` |
| `metadata` | Any | required |

**Returns:** Path.

##### `Epochs.zero_time_report`

```python
zero_time_report(self, zero_time: 'float | str | None' = None) -> 'dict[str, float | int | bool | str | None]'
```

Return residual amplitude at the zeroing anchor used for each epoch.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `zero_time` | float \| str \| None | `None` |

**Returns:** dict[str, float \| int \| bool \| str \| None].

### `EpochsCore`

```python
EpochsCore(epochs_df: 'pd.DataFrame', sfreq: 'float', tmin: 'float', tmax: 'float', baseline: 'tuple[float, float] | None' = None, stim_ch: 'str | list[str] | None' = None, metadata: 'dict[str, Any]' = <factory>, zero_time: 'float | str | None' = 'post_artifact') -> None
```

Trial-by-time-by-channel epoch container with audit metadata.

``epochs_df`` uses a two-level ``(epoch, time)`` row index and channel
columns. ``sfreq``, ``tmin``, ``tmax``, ``baseline``, and ``zero_time``
retain the sampling and alignment contract used by detection, quality
control, spectral analysis, and visualization methods.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs_df` | pd.DataFrame | required |
| `sfreq` | float | required |
| `tmin` | float | required |
| `tmax` | float | required |
| `baseline` | tuple[float, float] \| None | `None` |
| `stim_ch` | str \| list[str] \| None | `None` |
| `metadata` | dict[str, Any] | `<factory>` |
| `zero_time` | float \| str \| None | `'post_artifact'` |

**Returns:** None.

#### Public members

##### `EpochsCore.as_array`

```python
as_array(self) -> 'tuple[np.ndarray, np.ndarray, list[str]]'
```

Return samples, time coordinates, and channel labels.

The sample array is ordered as ``(trials, times, channels)``.

**Returns:** tuple[np.ndarray, np.ndarray, list[str]].

##### `EpochsCore.channels`

```python
channels  # property
```

Return channel labels in their stored column order.

**Returns:** list[str].

##### `EpochsCore.detect_zero_time_artifacts`

```python
detect_zero_time_artifacts(self, zero_time: 'float' = 0.0, artifact_z: 'float' = 6.0) -> 'pd.DataFrame'
```

Detect whether a requested zeroing sample overlaps the onset artifact.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `zero_time` | float | `0.0` |
| `artifact_z` | float | `6.0` |

**Returns:** pd.DataFrame.

##### `EpochsCore.get_mean_waveform`

```python
get_mean_waveform(self) -> 'pd.DataFrame'
```

Return the trial-averaged waveform for every channel.

**Returns:** pd.DataFrame.

##### `EpochsCore.get_sem_waveform`

```python
get_sem_waveform(self) -> 'pd.DataFrame'
```

Return the standard error across trials at each time.

**Returns:** pd.DataFrame.

##### `EpochsCore.get_std_waveform`

```python
get_std_waveform(self) -> 'pd.DataFrame'
```

Return the sample standard deviation across trials at each time.

**Returns:** pd.DataFrame.

##### `EpochsCore.n_trials`

```python
n_trials(self) -> 'int'
```

Return the number of distinct trials in the epoch table.

**Returns:** int.

##### `EpochsCore.post_artifact_anchor_report`

```python
post_artifact_anchor_report(self) -> 'pd.DataFrame'
```

Return per-epoch post-artifact zeroing anchors when available.

**Returns:** pd.DataFrame.

##### `EpochsCore.times`

```python
times  # property
```

Return unique epoch-relative sample times in seconds.

**Returns:** np.ndarray.

##### `EpochsCore.zero_time_report`

```python
zero_time_report(self, zero_time: 'float | str | None' = None) -> 'dict[str, float | int | bool | str | None]'
```

Return residual amplitude at the zeroing anchor used for each epoch.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `zero_time` | float \| str \| None | `None` |

**Returns:** dict[str, float \| int \| bool \| str \| None].

### `EpochsResult`

```python
EpochsResult(epochs_df: 'pd.DataFrame', sfreq: 'float', tmin: 'float', tmax: 'float', baseline: 'tuple[float, float] | None' = None, stim_ch: 'str | list[str] | None' = None, metadata: 'dict[str, Any]' = <factory>, zero_time: 'float | str | None' = 'post_artifact') -> None
```

Epoched stimulation data backed by a MultiIndex DataFrame.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs_df` | pd.DataFrame | required |
| `sfreq` | float | required |
| `tmin` | float | required |
| `tmax` | float | required |
| `baseline` | tuple[float, float] \| None | `None` |
| `stim_ch` | str \| list[str] \| None | `None` |
| `metadata` | dict[str, Any] | `<factory>` |
| `zero_time` | float \| str \| None | `'post_artifact'` |

**Returns:** None.

#### Public members

##### `EpochsResult.as_array`

```python
as_array(self) -> 'tuple[np.ndarray, np.ndarray, list[str]]'
```

Return samples, time coordinates, and channel labels.

The sample array is ordered as ``(trials, times, channels)``.

**Returns:** tuple[np.ndarray, np.ndarray, list[str]].

##### `EpochsResult.channels`

```python
channels  # property
```

Return channel labels in their stored column order.

**Returns:** list[str].

##### `EpochsResult.detect_erp`

```python
detect_erp(self, method: 'str', *args: 'Any', **kwargs: 'Any') -> 'pd.DataFrame'
```

Run one named ERP detector and return its channel-level results.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `method` | str | required |
| `args` | Any | required |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `EpochsResult.detect_erp_all`

```python
detect_erp_all(self, *args: 'Any', **kwargs: 'Any') -> 'pd.DataFrame'
```

Run every registered ERP detector and combine the channel results.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `args` | Any | required |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `EpochsResult.detect_zero_time_artifacts`

```python
detect_zero_time_artifacts(self, zero_time: 'float' = 0.0, artifact_z: 'float' = 6.0) -> 'pd.DataFrame'
```

Detect whether a requested zeroing sample overlaps the onset artifact.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `zero_time` | float | `0.0` |
| `artifact_z` | float | `6.0` |

**Returns:** pd.DataFrame.

##### `EpochsResult.flag_artifacts`

```python
flag_artifacts(self, *args: 'Any', **kwargs: 'Any')
```

Score trial-channel responses against the artifact criteria.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `args` | Any | required |
| `kwargs` | Any | required |

##### `EpochsResult.from_hdf`

```python
from_hdf(cls, path: 'str | Path', key: 'str' = 'epochs', *, allow_unsafe_legacy_pickle: 'bool' = False) -> "'Epochs'"
```

Load epochs and restore their audit metadata.

Files written by ERPy 1.0 and later use deterministic JSON metadata.
Older ERPy files stored metadata with Python pickle, which can execute
arbitrary code when opened. Such files are rejected unless
``allow_unsafe_legacy_pickle=True`` is supplied for a file the caller
created and trusts. Re-saving a trusted legacy file migrates it to the
safe format.

Parameters
----------
path : str or pathlib.Path
    HDF5 epoch file to read.
key : str, default ``"epochs"``
    HDF5 group containing the epoch table.
allow_unsafe_legacy_pickle : bool, default ``False``
    Explicitly permit executable pickle metadata from a trusted legacy
    file. Never enable this for an untrusted or downloaded file.

Returns
-------
Epochs
    Restored epoch data and metadata.

Raises
------
UnsafeLegacyEpochMetadataError
    If pickle metadata are present and unsafe loading was not enabled.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `key` | str | `'epochs'` |
| `allow_unsafe_legacy_pickle` | bool | `False` |

**Returns:** 'Epochs'.

##### `EpochsResult.get_mean_waveform`

```python
get_mean_waveform(self) -> 'pd.DataFrame'
```

Return the trial-averaged waveform for every channel.

**Returns:** pd.DataFrame.

##### `EpochsResult.get_sem_waveform`

```python
get_sem_waveform(self) -> 'pd.DataFrame'
```

Return the standard error across trials at each time.

**Returns:** pd.DataFrame.

##### `EpochsResult.get_std_waveform`

```python
get_std_waveform(self) -> 'pd.DataFrame'
```

Return the sample standard deviation across trials at each time.

**Returns:** pd.DataFrame.

##### `EpochsResult.n_trials`

```python
n_trials(self) -> 'int'
```

Return the number of distinct trials in the epoch table.

**Returns:** int.

##### `EpochsResult.plot`

```python
plot  # property
```

Return a plotting accessor bound to these epochs.

##### `EpochsResult.post_artifact_anchor_report`

```python
post_artifact_anchor_report(self) -> 'pd.DataFrame'
```

Return per-epoch post-artifact zeroing anchors when available.

**Returns:** pd.DataFrame.

##### `EpochsResult.reject_artifacts`

```python
reject_artifacts(self, *args: 'Any', **kwargs: 'Any') -> "'Epochs'"
```

Return a copy with artifact-contaminated responses rejected.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `args` | Any | required |
| `kwargs` | Any | required |

**Returns:** 'Epochs'.

##### `EpochsResult.rereference`

```python
rereference(self, *, method: 'str' = 'robust_mean', skipped_chs: 'Iterable[str] | None' = None, zscore_threshold: 'float' = 8.0, min_reference_channels: 'int' = 4) -> "'Epochs'"
```

Return a re-referenced copy while preserving every channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `method` | str | `'robust_mean'` |
| `skipped_chs` | Iterable[str] \| None | `None` |
| `zscore_threshold` | float | `8.0` |
| `min_reference_channels` | int | `4` |

**Returns:** 'Epochs'.

##### `EpochsResult.rescale`

```python
rescale(self, factor: 'float', *, signal_units: 'str' = 'uV', reason: 'str | None' = None) -> "'Epochs'"
```

Return a copy with channel values multiplied by ``factor``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `factor` | float | required |
| `signal_units` | str | `'uV'` |
| `reason` | str \| None | `None` |

**Returns:** 'Epochs'.

##### `EpochsResult.spectral`

```python
spectral  # property
```

Return a spectral-analysis accessor bound to these epochs.

##### `EpochsResult.times`

```python
times  # property
```

Return unique epoch-relative sample times in seconds.

**Returns:** np.ndarray.

##### `EpochsResult.to_hdf`

```python
to_hdf(self, path: 'str | Path', key: 'str' = 'epochs', *, storage_dtype: 'str | np.dtype | None' = 'float32', complevel: 'int' = 5, complib: 'str' = 'blosc:zstd', **metadata: 'Any') -> 'Path'
```

Atomically persist all trials in a compact, loss-bounded HDF5 file.

Float32 storage preserves sub-microvolt precision across the practical
iEEG range while substantially reducing read time and disk use. Data are
promoted by NumPy operations as needed after loading.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `key` | str | `'epochs'` |
| `storage_dtype` | str \| np.dtype \| None | `'float32'` |
| `complevel` | int | `5` |
| `complib` | str | `'blosc:zstd'` |
| `metadata` | Any | required |

**Returns:** Path.

##### `EpochsResult.zero_time_report`

```python
zero_time_report(self, zero_time: 'float | str | None' = None) -> 'dict[str, float | int | bool | str | None]'
```

Return residual amplitude at the zeroing anchor used for each epoch.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `zero_time` | float \| str \| None | `None` |

**Returns:** dict[str, float \| int \| bool \| str \| None].

### `EpochsSpectral`

```python
EpochsSpectral(epochs) -> 'None'
```

Spectral-analysis accessor available as ``epochs.spectral``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |

**Returns:** None.

#### Public members

##### `EpochsSpectral.band_power`

```python
band_power(self, bands=None, channels=None, **kwargs) -> 'pd.DataFrame'
```

Summarize epoch power within named frequency bands.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bands` | not specified | `None` |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `EpochsSpectral.coherence_matrix`

```python
coherence_matrix(self, band=(8.0, 13.0), channels=None, **kwargs) -> 'pd.DataFrame'
```

Compute a band-limited pairwise coherence matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `band` | not specified | `(8.0, 13.0)` |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `EpochsSpectral.comodulogram`

```python
comodulogram(self, phase_channel: 'str', amp_channel: 'str | None' = None, **kwargs) -> 'ComodulogramResult'
```

Compute phase-amplitude coupling across frequency pairs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `kwargs` | not specified | required |

**Returns:** ComodulogramResult.

##### `EpochsSpectral.ersp`

```python
ersp(self, channel: 'str', config: 'SpectralConfig | None' = None) -> 'TFRResult'
```

Compute baseline-normalized event-related spectral perturbation.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |

**Returns:** TFRResult.

##### `EpochsSpectral.granger_causality_matrix`

```python
granger_causality_matrix(self, channels=None, **kwargs) -> 'pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]'
```

Compute Granger causality; explicit alias of ``granger_matrix``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame \| tuple[pd.DataFrame, pd.DataFrame].

##### `EpochsSpectral.granger_matrix`

```python
granger_matrix(self, channels=None, **kwargs) -> 'pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]'
```

Compute a directed pairwise Granger-causality matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame \| tuple[pd.DataFrame, pd.DataFrame].

##### `EpochsSpectral.itpc`

```python
itpc(self, channel: 'str', config: 'SpectralConfig | None' = None) -> 'TFRResult'
```

Compute inter-trial phase coherence for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |

**Returns:** TFRResult.

##### `EpochsSpectral.mi_matrix`

```python
mi_matrix(self, channels=None, **kwargs) -> 'pd.DataFrame'
```

Compute mutual information; alias of ``mutual_information_matrix``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `EpochsSpectral.mutual_information_matrix`

```python
mutual_information_matrix(self, channels=None, **kwargs) -> 'pd.DataFrame'
```

Compute a pairwise mutual-information matrix across epoch samples.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `EpochsSpectral.pac`

```python
pac(self, phase_channel: 'str', amp_channel: 'str | None' = None, **kwargs) -> 'PACResult'
```

Estimate phase-amplitude coupling for a channel pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `kwargs` | not specified | required |

**Returns:** PACResult.

##### `EpochsSpectral.phase_phase`

```python
phase_phase(self, ch_x: 'str', ch_y: 'str | None' = None, **kwargs) -> 'float | pd.DataFrame'
```

Estimate n:m phase coupling or a cross-frequency coupling matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `ch_x` | str | required |
| `ch_y` | str \| None | `None` |
| `kwargs` | not specified | required |

**Returns:** float \| pd.DataFrame.

##### `EpochsSpectral.plot`

```python
plot  # property
```

Return a spectral plotting accessor bound to these epochs.

##### `EpochsSpectral.plv`

```python
plv(self, ch_x: 'str', ch_y: 'str', config: 'SpectralConfig | None' = None) -> 'PLVResult'
```

Compute time-frequency phase locking between two channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `ch_x` | str | required |
| `ch_y` | str | required |
| `config` | SpectralConfig \| None | `None` |

**Returns:** PLVResult.

##### `EpochsSpectral.plv_matrix`

```python
plv_matrix(self, band=(8.0, 13.0), channels=None, **kwargs) -> 'pd.DataFrame'
```

Compute a band-limited pairwise phase-locking matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `band` | not specified | `(8.0, 13.0)` |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `EpochsSpectral.psd`

```python
psd(self, channels=None, **kwargs) -> 'pd.DataFrame'
```

Estimate power spectral density for selected epoch channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

##### `EpochsSpectral.tfr`

```python
tfr(self, channel: 'str', config: 'SpectralConfig | None' = None, **kwargs) -> 'TFRResult'
```

Compute a Morlet time-frequency representation for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `kwargs` | not specified | required |

**Returns:** TFRResult.

### `EventDetectionError`

```python
EventDetectionError
```

Raised when no defensible stimulation-event sequence can be detected.

#### Public members

##### `EventDetectionError.add_note`

```python
add_note
```

Exception.add_note(note) --
add a note to the exception

##### `EventDetectionError.with_traceback`

```python
with_traceback
```

Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.

### `Events`

```python
Events(dataloader: 'Any', PYTHON_SCRIPT: 'str | None' = None, BASH_SCRIPT: 'str | None' = None) -> None
```

Session-event workflow bound to an ERPy ``DataLoader``.

The facade creates canonical event tables for selected stimulation pairs
and exposes the same processing path for local execution or a configured
job runner. Event-source parameters remain explicit in each method call.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `dataloader` | Any | required |
| `PYTHON_SCRIPT` | str \| None | `None` |
| `BASH_SCRIPT` | str \| None | `None` |

**Returns:** None.

#### Public members

##### `Events.install_standard_scripts`

```python
install_standard_scripts(cls) -> 'None'
```

Reset event detection to ERPy's built-in execution path.

**Returns:** None.

##### `Events.process`

```python
process(self, session_id: 'str', method: 'str' = 'consensus', stim_pairs: 'Iterable[str] | None' = None, **kwargs: 'Any')
```

Detect events for the requested session and stimulation pairs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `method` | str | `'consensus'` |
| `stim_pairs` | Iterable[str] \| None | `None` |
| `kwargs` | Any | required |

##### `Events.queue_job`

```python
queue_job(self, session_ids: 'Iterable[str] | None' = None, backend: 'str | None' = None, **kwargs: 'Any')
```

Process event jobs and return the runner plus execution summaries.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_ids` | Iterable[str] \| None | `None` |
| `backend` | str \| None | `None` |
| `kwargs` | Any | required |

##### `Events.use_standard_scripts`

```python
use_standard_scripts(cls, bash_type: 'str' = 'simple') -> 'None'
```

Select ERPy's built-in local event-processing hooks.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bash_type` | str | `'simple'` |

**Returns:** None.

### `PACResult`

```python
PACResult(phase_channel: 'str', amp_channel: 'str', phase_band: 'tuple[float, float]', amp_band: 'tuple[float, float]', method: 'str', mi: 'float', phase_bin_centers: 'np.ndarray', amplitude_by_phase: 'np.ndarray', preferred_phase: 'float' = nan, n_samples: 'int' = 0) -> None
```

Phase-amplitude coupling summary for one channel pair and band pair.

The result retains the coupling method and scalar modulation index together
with the phase-bin centers, mean amplitude profile, preferred phase, and
number of finite samples used in the estimate.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `phase_channel` | str | required |
| `amp_channel` | str | required |
| `phase_band` | tuple[float, float] | required |
| `amp_band` | tuple[float, float] | required |
| `method` | str | required |
| `mi` | float | required |
| `phase_bin_centers` | np.ndarray | required |
| `amplitude_by_phase` | np.ndarray | required |
| `preferred_phase` | float | `nan` |
| `n_samples` | int | `0` |

**Returns:** None.
### `PLVResult`

```python
PLVResult(ch_x: 'str', ch_y: 'str', freqs: 'np.ndarray', times: 'np.ndarray', plv: 'np.ndarray', n_trials: 'int') -> None
```

Time-frequency phase-locking value between two channels.

``plv`` is a frequency-by-time array bounded from zero to one and computed
across the retained trials. ``freqs`` are in hertz and ``times`` are epoch-
relative seconds.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `ch_x` | str | required |
| `ch_y` | str | required |
| `freqs` | np.ndarray | required |
| `times` | np.ndarray | required |
| `plv` | np.ndarray | required |
| `n_trials` | int | required |

**Returns:** None.
### `Pipeline`

```python
Pipeline(dataloader) -> 'None'
```

Registry-based preprocessing and epoch extraction.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `dataloader` | not specified | required |

**Returns:** None.

#### Public members

##### `Pipeline.build_custom_pipeline`

```python
build_custom_pipeline(self, steps: 'list[tuple[str, dict[str, Any]]]') -> 'list[tuple[str, dict[str, Any]]]'
```

Validate and normalize a user-supplied preprocessing sequence.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `steps` | list[tuple[str, dict[str, Any]]] | required |

**Returns:** list[tuple[str, dict[str, Any]]].

##### `Pipeline.epoch_data`

```python
epoch_data(self, df: 'pd.DataFrame', event_times: 'Iterable[Any]', tmin: 'float' = -0.5, tmax: 'float' = 1.0, baseline: 'tuple[float, float] | None' = None, zero_time: 'float | str | None' = 'post_artifact', stim_ch: 'str | list[str] | None' = None, artifact_anchor_params: 'dict[str, Any] | None' = None) -> 'Epochs'
```

Extract event-locked trials from a continuous processed recording.

Epochs may use a fixed zero-time sample or ERPy's post-artifact anchor;
the latter stores per-trial anchor diagnostics in audit metadata.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `event_times` | Iterable[Any] | required |
| `tmin` | float | `-0.5` |
| `tmax` | float | `1.0` |
| `baseline` | tuple[float, float] \| None | `None` |
| `zero_time` | float \| str \| None | `'post_artifact'` |
| `stim_ch` | str \| list[str] \| None | `None` |
| `artifact_anchor_params` | dict[str, Any] \| None | `None` |

**Returns:** Epochs.

##### `Pipeline.epochs`

```python
epochs  # property
```

Return the epoch-job accessor bound to this pipeline.

**Returns:** EpochsRunner.

##### `Pipeline.find_epochs_files`

```python
find_epochs_files(self, session_id: 'str', stim_pair: 'str', pipeline_steps=None) -> 'list[Path]'
```

Return matching epoch caches from newest to oldest.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `pipeline_steps` | not specified | `None` |

**Returns:** list[Path].

##### `Pipeline.get_standard_pipeline`

```python
get_standard_pipeline(self, name: 'str' = 'blank_filt') -> 'list[tuple[str, dict[str, Any]]]'
```

Return a copy-ready named preprocessing step sequence.

Available presets combine artifact blanking, line-noise filtering,
band-pass filtering, optional decimation, and a selected reference.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `name` | str | `'blank_filt'` |

**Returns:** list[tuple[str, dict[str, Any]]].

##### `Pipeline.install_standard_scripts`

```python
install_standard_scripts(cls) -> 'None'
```

Reset preprocessing to ERPy's built-in execution path.

**Returns:** None.

##### `Pipeline.load_epochs`

```python
load_epochs(self, session_id: 'str', stim_pair: 'str', pipeline_steps=None, *, allow_unsafe_legacy_pickle: 'bool' = False) -> 'Epochs'
```

Load the newest matching epoch cache for an acquisition.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `pipeline_steps` | not specified | `None` |
| `allow_unsafe_legacy_pickle` | bool | `False` |

**Returns:** Epochs.

##### `Pipeline.load_epochs_file`

```python
load_epochs_file(self, path: 'str | Path', *, allow_unsafe_legacy_pickle: 'bool' = False) -> 'Epochs'
```

Load a cached epoch file.

Set ``allow_unsafe_legacy_pickle`` only for a pre-1.0 file that you
created and trust; legacy pickle metadata can execute arbitrary code.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `allow_unsafe_legacy_pickle` | bool | `False` |

**Returns:** Epochs.

##### `Pipeline.load_processed_data`

```python
load_processed_data(self, path: 'str | Path', *, allow_unsafe_legacy_pickle: 'bool' = False) -> 'pd.DataFrame'
```

Load processed data and safely restore its provenance metadata.

Parameters
----------
path : str or pathlib.Path
    HDF5 processed-data file to read.
allow_unsafe_legacy_pickle : bool, default ``False``
    Permit metadata from a trusted pre-1.0 file. Legacy PyTables
    attributes use pickle and may execute arbitrary code; never enable
    this option for an untrusted or downloaded file.

Returns
-------
pandas.DataFrame
    Processed samples with restored metadata in ``DataFrame.attrs``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `allow_unsafe_legacy_pickle` | bool | `False` |

**Returns:** pd.DataFrame.

##### `Pipeline.pipeline_id`

```python
pipeline_id(self, pipeline_steps: 'list[tuple[str, dict[str, Any]]] | None') -> 'str'
```

Return a stable identifier for a preprocessing step sequence.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `pipeline_steps` | list[tuple[str, dict[str, Any]]] \| None | required |

**Returns:** str.

##### `Pipeline.process`

```python
process(self, pipeline_steps: 'list[tuple[str, dict[str, Any]]] | None' = None, session_id: 'str | None' = None, stim_pair: 'str | None' = None, **kwargs: 'Any')
```

Run preprocessing for one acquisition or all matching acquisitions.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `pipeline_steps` | list[tuple[str, dict[str, Any]]] \| None | `None` |
| `session_id` | str \| None | `None` |
| `stim_pair` | str \| None | `None` |
| `kwargs` | Any | required |

##### `Pipeline.process_all_sessions`

```python
process_all_sessions(self, pipeline_steps=None, session_ids: 'Iterable[str] | None' = None, **kwargs: 'Any')
```

Preprocess selected or registered sessions and stimulation pairs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `pipeline_steps` | not specified | `None` |
| `session_ids` | Iterable[str] \| None | `None` |
| `kwargs` | Any | required |

##### `Pipeline.process_and_epoch`

```python
process_and_epoch(self, session_id: 'str', stim_pair: 'str', pipeline_steps: 'list[tuple[str, dict[str, Any]]] | None' = None, event_times: 'Iterable[Any] | None' = None, tmin: 'float' = -0.5, tmax: 'float' = 1.0, baseline: 'tuple[float, float] | None' = (-0.5, -0.03), zero_time: 'float | str | None' = 'post_artifact', artifact_anchor_params: 'dict[str, Any] | None' = None, processing_margin_s: 'float | None' = None, save_processed: 'bool' = False, save_epochs: 'bool' = False, **kwargs: 'Any') -> 'tuple[Epochs, pd.DataFrame]'
```

Preprocess one acquisition and extract its event-locked epochs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `pipeline_steps` | list[tuple[str, dict[str, Any]]] \| None | `None` |
| `event_times` | Iterable[Any] \| None | `None` |
| `tmin` | float | `-0.5` |
| `tmax` | float | `1.0` |
| `baseline` | tuple[float, float] \| None | `(-0.5, -0.03)` |
| `zero_time` | float \| str \| None | `'post_artifact'` |
| `artifact_anchor_params` | dict[str, Any] \| None | `None` |
| `processing_margin_s` | float \| None | `None` |
| `save_processed` | bool | `False` |
| `save_epochs` | bool | `False` |
| `kwargs` | Any | required |

**Returns:** tuple[Epochs, pd.DataFrame].

##### `Pipeline.process_session`

```python
process_session(self, session_id: 'str', pipeline_steps=None, stim_pairs: 'Iterable[str] | None' = None, **kwargs: 'Any')
```

Preprocess selected stimulation pairs within one session.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `pipeline_steps` | not specified | `None` |
| `stim_pairs` | Iterable[str] \| None | `None` |
| `kwargs` | Any | required |

##### `Pipeline.process_stim_session`

```python
process_stim_session(self, session_id: 'str', stim_pair: 'str', pipeline_steps: 'list[tuple[str, dict[str, Any]]] | None' = None, input_source: 'str' = 'auto', load_from_raw: 'bool' = True, save_processed: 'bool' = False, overwrite: 'bool' = False, raw_file: 'str | Path | None' = None, stim_start: 'Any | None' = None, time_window: 'tuple[Any, Any] | None' = None, **context: 'Any') -> 'pd.DataFrame'
```

Load and preprocess one session/stimulation-pair acquisition.

The returned continuous table records sampling frequency and pipeline
identity in ``DataFrame.attrs`` and can optionally be cached on disk.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `pipeline_steps` | list[tuple[str, dict[str, Any]]] \| None | `None` |
| `input_source` | str | `'auto'` |
| `load_from_raw` | bool | `True` |
| `save_processed` | bool | `False` |
| `overwrite` | bool | `False` |
| `raw_file` | str \| Path \| None | `None` |
| `stim_start` | Any \| None | `None` |
| `time_window` | tuple[Any, Any] \| None | `None` |
| `context` | Any | required |

**Returns:** pd.DataFrame.

##### `Pipeline.queue_job`

```python
queue_job(self, pipeline_steps=None, session_ids: 'Iterable[str] | None' = None, backend: 'str | None' = None, **kwargs: 'Any')
```

Process selected acquisitions and return the job-runner summary.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `pipeline_steps` | not specified | `None` |
| `session_ids` | Iterable[str] \| None | `None` |
| `backend` | str \| None | `None` |
| `kwargs` | Any | required |

##### `Pipeline.save_processed_data`

```python
save_processed_data(self, df: 'pd.DataFrame', path: 'str | Path', **metadata: 'Any') -> 'Path'
```

Atomically save processed data with deterministic JSON provenance.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `path` | str \| Path | required |
| `metadata` | Any | required |

**Returns:** Path.

##### `Pipeline.use_standard_scripts`

```python
use_standard_scripts(cls, bash_type: 'str' = 'simple') -> 'None'
```

Select ERPy's built-in local preprocessing hooks.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bash_type` | str | `'simple'` |

**Returns:** None.

### `Patient`

```python
Patient(patient_id: 'str', config_path: 'str | None' = None, verbose: 'bool' = True) -> 'None'
```

Quick-use public wrapper for one patient.

Examples
--------
>>> patient = ep.Patient("EXAMPLE_PATIENT")
>>> epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2")
>>> detections = epochs.detect_erp_all()

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `patient_id` | str | required |
| `config_path` | str \| None | `None` |
| `verbose` | bool | `True` |

**Returns:** None.

#### Public members

##### `Patient.analyze`

```python
analyze(self, session_id: 'str', stim_pair: 'str', **kwargs: 'Any') -> 'AnalysisResult'
```

Run epoching, response-artifact QC, and ERP detection in one call.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `kwargs` | Any | required |

**Returns:** AnalysisResult.

##### `Patient.detect_events`

```python
detect_events(self, session_id: 'str', stim_pair: 'str', method: 'str' = 'auto', **kwargs: 'Any') -> 'pd.DataFrame'
```

Detect stimulation events for one session and stimulation pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `method` | str | `'auto'` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Patient.epoch`

```python
epoch(self, session_id: 'str', stim_pair: 'str', pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', **kwargs: 'Any') -> 'EpochsResult'
```

Extract or load event-locked epochs for one acquisition.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `kwargs` | Any | required |

**Returns:** EpochsResult.

##### `Patient.from_bids`

```python
from_bids(cls, bids_root: 'str', subject: 'str', output_root: 'str | None' = None, session: 'str | None' = None, task: 'str | None' = None, patient_id: 'str | None' = None, copy_raw: 'bool' = False, derivatives: 'bool' = True, verbose: 'bool' = True) -> "'Patient'"
```

Import a BIDS iEEG subject and return a ready-to-use patient facade.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bids_root` | str | required |
| `subject` | str | required |
| `output_root` | str \| None | `None` |
| `session` | str \| None | `None` |
| `task` | str \| None | `None` |
| `patient_id` | str \| None | `None` |
| `copy_raw` | bool | `False` |
| `derivatives` | bool | `True` |
| `verbose` | bool | `True` |

**Returns:** 'Patient'.

##### `Patient.from_nwb`

```python
from_nwb(cls, nwb_path: 'str', output_root: 'str | None' = None, patient_id: 'str | None' = None, session_id: 'str | None' = None, stim_pair: 'str | None' = None, stim_times: 'Any | None' = None, stim_duration_s: 'float' = 0.001, series_name: 'str | None' = None, copy_raw: 'bool' = False, verbose: 'bool' = True) -> "'Patient'"
```

Import one NWB recording and return a ready-to-use patient facade.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `nwb_path` | str | required |
| `output_root` | str \| None | `None` |
| `patient_id` | str \| None | `None` |
| `session_id` | str \| None | `None` |
| `stim_pair` | str \| None | `None` |
| `stim_times` | Any \| None | `None` |
| `stim_duration_s` | float | `0.001` |
| `series_name` | str \| None | `None` |
| `copy_raw` | bool | `False` |
| `verbose` | bool | `True` |

**Returns:** 'Patient'.

##### `Patient.recording`

```python
recording(self, session_id: 'str') -> "'Recording'"
```

Return the recording facade for one session.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |

**Returns:** 'Recording'.

##### `Patient.run`

```python
run(self, session_id: 'str', stim_pair: 'str', pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', tmin: 'float' = -0.5, tmax: 'float' = 1.0, baseline: 'tuple[float, float] | None' = (-0.5, -0.03), zero_time: 'float | str | None' = 'post_artifact', artifact_anchor_params: 'dict[str, Any] | None' = None, event_method: 'str' = 'auto', **kwargs: 'Any') -> 'EpochsResult'
```

Run preprocessing and epoch extraction without reusing a cache.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `tmin` | float | `-0.5` |
| `tmax` | float | `1.0` |
| `baseline` | tuple[float, float] \| None | `(-0.5, -0.03)` |
| `zero_time` | float \| str \| None | `'post_artifact'` |
| `artifact_anchor_params` | dict[str, Any] \| None | `None` |
| `event_method` | str | `'auto'` |
| `kwargs` | Any | required |

**Returns:** EpochsResult.

##### `Patient.session`

```python
session(self, session_id: 'str') -> "'Recording'"
```

Return a recording facade; alias of ``recording``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |

**Returns:** 'Recording'.

##### `Patient.sessions`

```python
sessions(self) -> 'list[str]'
```

Return registered session identifiers for this patient.

**Returns:** list[str].

##### `Patient.stim_pairs`

```python
stim_pairs(self, session_id: 'str') -> 'list[str]'
```

Return registered stimulation pairs for one session.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |

**Returns:** list[str].

##### `Patient.warmup`

```python
warmup(self, session_ids: 'Iterable[str] | None' = None, **kwargs: 'Any') -> 'pd.DataFrame'
```

Precompute local caches for selected patient sessions.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_ids` | Iterable[str] \| None | `None` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Patient.warmup_jobs`

```python
warmup_jobs(self, session_ids: 'Iterable[str] | None' = None, **kwargs: 'Any') -> 'pd.DataFrame'
```

Submit cache-warmup jobs for selected patient sessions.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_ids` | Iterable[str] \| None | `None` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

### `Recording`

```python
Recording(patient: 'Patient', session_id: 'str') -> None
```

Session-scoped analysis facade owned by a ``Patient``.

A recording fixes ``session_id`` while exposing event creation, epoching,
cache loading, and the one-call analysis contract for any registered
stimulation pair. ``Session`` is a compatibility alias for this class.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `patient` | Patient | required |
| `session_id` | str | required |

**Returns:** None.

#### Public members

##### `Recording.analyze`

```python
analyze(self, stim_pair: 'str', *, pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', methods: 'list[str] | None' = None, min_consensus: 'int' = 2, response_window: 'tuple[float, float] | None' = None, artifact_kwargs: 'dict[str, Any] | None' = None, artifact_reject: 'str | None' = 'nan_response', max_bad_channel_fraction: 'float' = 0.25, qc_max_bad_response_fraction: 'float' = 0.25, qc_max_hard_artifact_fraction: 'float' = 0.1, qc_min_clean_responses: 'int' = 8, qc_exclude_boundary_peaks: 'bool' = True, hard_artifact_reasons: 'Iterable[str]' = ('absolute_peak', 'absolute_ptp', 'extreme_amplitude_outlier', 'plateau_clipping', 'rail_clipping', 'saturation_or_clipping'), detect_kwargs: 'dict[str, Any] | None' = None, **epoch_kwargs: 'Any') -> 'AnalysisResult'
```

Run the default ERPy analysis contract for one stimulation pair.

This keeps the common notebook path compact while returning the same
auditable artifacts available from the lower-level API. Trial-contact
artifact QC is always evaluated. Persistent-channel detection is
performed only when the preprocessing pipeline contains an explicit
``reject_bad_channels`` step, such as ``blank_filt_reject``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `methods` | list[str] \| None | `None` |
| `min_consensus` | int | `2` |
| `response_window` | tuple[float, float] \| None | `None` |
| `artifact_kwargs` | dict[str, Any] \| None | `None` |
| `artifact_reject` | str \| None | `'nan_response'` |
| `max_bad_channel_fraction` | float | `0.25` |
| `qc_max_bad_response_fraction` | float | `0.25` |
| `qc_max_hard_artifact_fraction` | float | `0.1` |
| `qc_min_clean_responses` | int | `8` |
| `qc_exclude_boundary_peaks` | bool | `True` |
| `hard_artifact_reasons` | Iterable[str] | `('absolute_peak', 'absolute_ptp', 'extreme_amplitude_outlier', 'plateau_clipping', 'rail_clipping', 'saturation_or_clipping')` |
| `detect_kwargs` | dict[str, Any] \| None | `None` |
| `epoch_kwargs` | Any | required |

**Returns:** AnalysisResult.

##### `Recording.dataloader`

```python
dataloader  # property
```

Return the patient data loader used by this recording.

**Returns:** DataLoader.

##### `Recording.epoch`

```python
epoch(self, stim_pair: 'str', pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', tmin: 'float' = -0.5, tmax: 'float' = 1.0, baseline: 'tuple[float, float] | None' = (-0.5, -0.03), zero_time: 'float | str | None' = 'post_artifact', artifact_anchor_params: 'dict[str, Any] | None' = None, cache: 'bool' = True, event_method: 'str' = 'auto', save_epochs: 'bool' = True, **kwargs: 'Any') -> 'EpochsResult'
```

Preprocess and epoch one stimulation pair, optionally using a cache.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `tmin` | float | `-0.5` |
| `tmax` | float | `1.0` |
| `baseline` | tuple[float, float] \| None | `(-0.5, -0.03)` |
| `zero_time` | float \| str \| None | `'post_artifact'` |
| `artifact_anchor_params` | dict[str, Any] \| None | `None` |
| `cache` | bool | `True` |
| `event_method` | str | `'auto'` |
| `save_epochs` | bool | `True` |
| `kwargs` | Any | required |

**Returns:** EpochsResult.

##### `Recording.events_for`

```python
events_for(self, stim_pair: 'str', method: 'str' = 'auto', save_events: 'bool' = True, **kwargs: 'Any') -> 'pd.DataFrame'
```

Detect and optionally save events for one stimulation pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |
| `method` | str | `'auto'` |
| `save_events` | bool | `True` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Recording.load_epochs`

```python
load_epochs(self, stim_pair: 'str', pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', **kwargs: 'Any') -> 'EpochsResult'
```

Load the newest matching epoch cache for a stimulation pair.

``allow_unsafe_legacy_pickle=True`` may be passed only for a trusted
pre-1.0 cache whose pickle metadata the caller accepts executing.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `kwargs` | Any | required |

**Returns:** EpochsResult.

##### `Recording.pipeline_obj`

```python
pipeline_obj  # property
```

Return the patient preprocessing pipeline used by this recording.

**Returns:** Pipeline.

##### `Recording.stim_pairs`

```python
stim_pairs(self) -> 'list[str]'
```

Return stimulation pairs registered for this recording session.

**Returns:** list[str].

### `Session`

```python
Session(patient: 'Patient', session_id: 'str') -> None
```

Session-scoped analysis facade owned by a ``Patient``.

A recording fixes ``session_id`` while exposing event creation, epoching,
cache loading, and the one-call analysis contract for any registered
stimulation pair. ``Session`` is a compatibility alias for this class.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `patient` | Patient | required |
| `session_id` | str | required |

**Returns:** None.

#### Public members

##### `Session.analyze`

```python
analyze(self, stim_pair: 'str', *, pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', methods: 'list[str] | None' = None, min_consensus: 'int' = 2, response_window: 'tuple[float, float] | None' = None, artifact_kwargs: 'dict[str, Any] | None' = None, artifact_reject: 'str | None' = 'nan_response', max_bad_channel_fraction: 'float' = 0.25, qc_max_bad_response_fraction: 'float' = 0.25, qc_max_hard_artifact_fraction: 'float' = 0.1, qc_min_clean_responses: 'int' = 8, qc_exclude_boundary_peaks: 'bool' = True, hard_artifact_reasons: 'Iterable[str]' = ('absolute_peak', 'absolute_ptp', 'extreme_amplitude_outlier', 'plateau_clipping', 'rail_clipping', 'saturation_or_clipping'), detect_kwargs: 'dict[str, Any] | None' = None, **epoch_kwargs: 'Any') -> 'AnalysisResult'
```

Run the default ERPy analysis contract for one stimulation pair.

This keeps the common notebook path compact while returning the same
auditable artifacts available from the lower-level API. Trial-contact
artifact QC is always evaluated. Persistent-channel detection is
performed only when the preprocessing pipeline contains an explicit
``reject_bad_channels`` step, such as ``blank_filt_reject``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `methods` | list[str] \| None | `None` |
| `min_consensus` | int | `2` |
| `response_window` | tuple[float, float] \| None | `None` |
| `artifact_kwargs` | dict[str, Any] \| None | `None` |
| `artifact_reject` | str \| None | `'nan_response'` |
| `max_bad_channel_fraction` | float | `0.25` |
| `qc_max_bad_response_fraction` | float | `0.25` |
| `qc_max_hard_artifact_fraction` | float | `0.1` |
| `qc_min_clean_responses` | int | `8` |
| `qc_exclude_boundary_peaks` | bool | `True` |
| `hard_artifact_reasons` | Iterable[str] | `('absolute_peak', 'absolute_ptp', 'extreme_amplitude_outlier', 'plateau_clipping', 'rail_clipping', 'saturation_or_clipping')` |
| `detect_kwargs` | dict[str, Any] \| None | `None` |
| `epoch_kwargs` | Any | required |

**Returns:** AnalysisResult.

##### `Session.dataloader`

```python
dataloader  # property
```

Return the patient data loader used by this recording.

**Returns:** DataLoader.

##### `Session.epoch`

```python
epoch(self, stim_pair: 'str', pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', tmin: 'float' = -0.5, tmax: 'float' = 1.0, baseline: 'tuple[float, float] | None' = (-0.5, -0.03), zero_time: 'float | str | None' = 'post_artifact', artifact_anchor_params: 'dict[str, Any] | None' = None, cache: 'bool' = True, event_method: 'str' = 'auto', save_epochs: 'bool' = True, **kwargs: 'Any') -> 'EpochsResult'
```

Preprocess and epoch one stimulation pair, optionally using a cache.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `tmin` | float | `-0.5` |
| `tmax` | float | `1.0` |
| `baseline` | tuple[float, float] \| None | `(-0.5, -0.03)` |
| `zero_time` | float \| str \| None | `'post_artifact'` |
| `artifact_anchor_params` | dict[str, Any] \| None | `None` |
| `cache` | bool | `True` |
| `event_method` | str | `'auto'` |
| `save_epochs` | bool | `True` |
| `kwargs` | Any | required |

**Returns:** EpochsResult.

##### `Session.events_for`

```python
events_for(self, stim_pair: 'str', method: 'str' = 'auto', save_events: 'bool' = True, **kwargs: 'Any') -> 'pd.DataFrame'
```

Detect and optionally save events for one stimulation pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |
| `method` | str | `'auto'` |
| `save_events` | bool | `True` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Session.load_epochs`

```python
load_epochs(self, stim_pair: 'str', pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', **kwargs: 'Any') -> 'EpochsResult'
```

Load the newest matching epoch cache for a stimulation pair.

``allow_unsafe_legacy_pickle=True`` may be passed only for a trusted
pre-1.0 cache whose pickle metadata the caller accepts executing.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `kwargs` | Any | required |

**Returns:** EpochsResult.

##### `Session.pipeline_obj`

```python
pipeline_obj  # property
```

Return the patient preprocessing pipeline used by this recording.

**Returns:** Pipeline.

##### `Session.stim_pairs`

```python
stim_pairs(self) -> 'list[str]'
```

Return stimulation pairs registered for this recording session.

**Returns:** list[str].

### `SignFlipResult`

```python
SignFlipResult(statistic: 'float', p_value: 'float', alternative: 'str', n_observations: 'int', n_permutations: 'int', exact: 'bool') -> None
```

Result of an exact or Monte Carlo one-sample sign-flip test.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `statistic` | float | required |
| `p_value` | float | required |
| `alternative` | str | required |
| `n_observations` | int | required |
| `n_permutations` | int | required |
| `exact` | bool | required |

**Returns:** None.
### `SpectralConfig`

```python
SpectralConfig(fmin: 'float' = 4.0, fmax: 'float' = 150.0, n_freqs: 'int' = 40, spacing: 'str' = 'log', n_cycles: 'float | str' = 7.0, baseline: 'tuple[float, float] | None' = (-0.5, -0.05), baseline_mode: 'str' = 'logratio', decim: 'int' = 1) -> None
```

Settings for Morlet time-frequency analysis.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `fmin` | float | `4.0` |
| `fmax` | float | `150.0` |
| `n_freqs` | int | `40` |
| `spacing` | str | `'log'` |
| `n_cycles` | float \| str | `7.0` |
| `baseline` | tuple[float, float] \| None | `(-0.5, -0.05)` |
| `baseline_mode` | str | `'logratio'` |
| `decim` | int | `1` |

**Returns:** None.

#### Public members

##### `SpectralConfig.frequencies`

```python
frequencies(self, sfreq: 'float | None' = None) -> 'np.ndarray'
```

Return configured frequencies, capped below the Nyquist limit.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `sfreq` | float \| None | `None` |

**Returns:** np.ndarray.

### `Study`

```python
Study(patient_id: 'str', config_path: 'str | None' = None, verbose: 'bool' = True) -> 'None'
```

Quick-use public wrapper for one patient.

Examples
--------
>>> patient = ep.Patient("EXAMPLE_PATIENT")
>>> epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2")
>>> detections = epochs.detect_erp_all()

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `patient_id` | str | required |
| `config_path` | str \| None | `None` |
| `verbose` | bool | `True` |

**Returns:** None.

#### Public members

##### `Study.analyze`

```python
analyze(self, session_id: 'str', stim_pair: 'str', **kwargs: 'Any') -> 'AnalysisResult'
```

Run epoching, response-artifact QC, and ERP detection in one call.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `kwargs` | Any | required |

**Returns:** AnalysisResult.

##### `Study.detect_events`

```python
detect_events(self, session_id: 'str', stim_pair: 'str', method: 'str' = 'auto', **kwargs: 'Any') -> 'pd.DataFrame'
```

Detect stimulation events for one session and stimulation pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `method` | str | `'auto'` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Study.epoch`

```python
epoch(self, session_id: 'str', stim_pair: 'str', pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', **kwargs: 'Any') -> 'EpochsResult'
```

Extract or load event-locked epochs for one acquisition.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `kwargs` | Any | required |

**Returns:** EpochsResult.

##### `Study.from_bids`

```python
from_bids(cls, bids_root: 'str', subject: 'str', output_root: 'str | None' = None, session: 'str | None' = None, task: 'str | None' = None, patient_id: 'str | None' = None, copy_raw: 'bool' = False, derivatives: 'bool' = True, verbose: 'bool' = True) -> "'Patient'"
```

Import a BIDS iEEG subject and return a ready-to-use patient facade.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bids_root` | str | required |
| `subject` | str | required |
| `output_root` | str \| None | `None` |
| `session` | str \| None | `None` |
| `task` | str \| None | `None` |
| `patient_id` | str \| None | `None` |
| `copy_raw` | bool | `False` |
| `derivatives` | bool | `True` |
| `verbose` | bool | `True` |

**Returns:** 'Patient'.

##### `Study.from_nwb`

```python
from_nwb(cls, nwb_path: 'str', output_root: 'str | None' = None, patient_id: 'str | None' = None, session_id: 'str | None' = None, stim_pair: 'str | None' = None, stim_times: 'Any | None' = None, stim_duration_s: 'float' = 0.001, series_name: 'str | None' = None, copy_raw: 'bool' = False, verbose: 'bool' = True) -> "'Patient'"
```

Import one NWB recording and return a ready-to-use patient facade.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `nwb_path` | str | required |
| `output_root` | str \| None | `None` |
| `patient_id` | str \| None | `None` |
| `session_id` | str \| None | `None` |
| `stim_pair` | str \| None | `None` |
| `stim_times` | Any \| None | `None` |
| `stim_duration_s` | float | `0.001` |
| `series_name` | str \| None | `None` |
| `copy_raw` | bool | `False` |
| `verbose` | bool | `True` |

**Returns:** 'Patient'.

##### `Study.recording`

```python
recording(self, session_id: 'str') -> "'Recording'"
```

Return the recording facade for one session.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |

**Returns:** 'Recording'.

##### `Study.run`

```python
run(self, session_id: 'str', stim_pair: 'str', pipeline: 'str | list[tuple[str, dict[str, Any]]]' = 'blank_filt', tmin: 'float' = -0.5, tmax: 'float' = 1.0, baseline: 'tuple[float, float] | None' = (-0.5, -0.03), zero_time: 'float | str | None' = 'post_artifact', artifact_anchor_params: 'dict[str, Any] | None' = None, event_method: 'str' = 'auto', **kwargs: 'Any') -> 'EpochsResult'
```

Run preprocessing and epoch extraction without reusing a cache.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `pipeline` | str \| list[tuple[str, dict[str, Any]]] | `'blank_filt'` |
| `tmin` | float | `-0.5` |
| `tmax` | float | `1.0` |
| `baseline` | tuple[float, float] \| None | `(-0.5, -0.03)` |
| `zero_time` | float \| str \| None | `'post_artifact'` |
| `artifact_anchor_params` | dict[str, Any] \| None | `None` |
| `event_method` | str | `'auto'` |
| `kwargs` | Any | required |

**Returns:** EpochsResult.

##### `Study.session`

```python
session(self, session_id: 'str') -> "'Recording'"
```

Return a recording facade; alias of ``recording``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |

**Returns:** 'Recording'.

##### `Study.sessions`

```python
sessions(self) -> 'list[str]'
```

Return registered session identifiers for this patient.

**Returns:** list[str].

##### `Study.stim_pairs`

```python
stim_pairs(self, session_id: 'str') -> 'list[str]'
```

Return registered stimulation pairs for one session.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_id` | str | required |

**Returns:** list[str].

##### `Study.warmup`

```python
warmup(self, session_ids: 'Iterable[str] | None' = None, **kwargs: 'Any') -> 'pd.DataFrame'
```

Precompute local caches for selected patient sessions.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_ids` | Iterable[str] \| None | `None` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

##### `Study.warmup_jobs`

```python
warmup_jobs(self, session_ids: 'Iterable[str] | None' = None, **kwargs: 'Any') -> 'pd.DataFrame'
```

Submit cache-warmup jobs for selected patient sessions.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `session_ids` | Iterable[str] \| None | `None` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

### `TFRResult`

```python
TFRResult(channel: 'str', freqs: 'np.ndarray', times: 'np.ndarray', power: 'np.ndarray', itc: 'np.ndarray', n_trials: 'int', baseline: 'tuple[float, float] | None' = None, baseline_mode: 'str | None' = None, power_per_trial: 'np.ndarray | None' = None) -> None
```

Time-frequency result for one channel.

``power`` and ``itc`` are frequency-by-time arrays. ``power`` is the
trial-averaged quantity after the recorded baseline transform, while
``power_per_trial`` optionally retains untransformed trial-level power as
a trial-by-frequency-by-time array.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `freqs` | np.ndarray | required |
| `times` | np.ndarray | required |
| `power` | np.ndarray | required |
| `itc` | np.ndarray | required |
| `n_trials` | int | required |
| `baseline` | tuple[float, float] \| None | `None` |
| `baseline_mode` | str \| None | `None` |
| `power_per_trial` | np.ndarray \| None | `None` |

**Returns:** None.

#### Public members

##### `TFRResult.ersp`

```python
ersp  # property
```

Event-related spectral perturbation (baseline-normalized power).

**Returns:** np.ndarray.

### `UnsafeLegacyEpochMetadataError`

```python
UnsafeLegacyEpochMetadataError
```

Raised when a legacy pickle-backed epoch cache is not explicitly trusted.

#### Public members

##### `UnsafeLegacyEpochMetadataError.add_note`

```python
add_note
```

Exception.add_note(note) --
add a note to the exception

##### `UnsafeLegacyEpochMetadataError.with_traceback`

```python
with_traceback
```

Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.

### `UnsafeLegacyProcessedMetadataError`

```python
UnsafeLegacyProcessedMetadataError
```

Raised when pickle-backed processed-data metadata are not trusted.

#### Public members

##### `UnsafeLegacyProcessedMetadataError.add_note`

```python
add_note
```

Exception.add_note(note) --
add a note to the exception

##### `UnsafeLegacyProcessedMetadataError.with_traceback`

```python
with_traceback
```

Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.

### `WaveformAudit`

```python
WaveformAudit(epochs: 'Any', summary: 'pd.DataFrame', artifact_responses: 'pd.DataFrame', response_window: 'tuple[float, float]', baseline_window: 'tuple[float, float]', _array: 'np.ndarray', _clean_array: 'np.ndarray', _times: 'np.ndarray', _channels: 'list[str]') -> None
```

Waveform metrics, detector concordance, and trial-level QC context.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | Any | required |
| `summary` | pd.DataFrame | required |
| `artifact_responses` | pd.DataFrame | required |
| `response_window` | tuple[float, float] | required |
| `baseline_window` | tuple[float, float] | required |
| `_array` | np.ndarray | required |
| `_clean_array` | np.ndarray | required |
| `_times` | np.ndarray | required |
| `_channels` | list[str] | required |

**Returns:** None.

#### Public members

##### `WaveformAudit.mean_waveforms`

```python
mean_waveforms(self, *, start_s: 'float' = -0.1, stop_s: 'float' = 0.45, step_ms: 'float' = 2.0) -> 'pd.DataFrame'
```

Return downsampled clean means for cohort-scale waveform review.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `start_s` | float | `-0.1` |
| `stop_s` | float | `0.45` |
| `step_ms` | float | `2.0` |

**Returns:** pd.DataFrame.

##### `WaveformAudit.plot`

```python
plot(self, channels: 'Iterable[str] | None' = None, *, max_channels: 'int' = 12, full_scale: 'bool' = False, title: 'str | None' = None)
```

Plot a compact review montage and return the Matplotlib figure.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | Iterable[str] \| None | `None` |
| `max_channels` | int | `12` |
| `full_scale` | bool | `False` |
| `title` | str \| None | `None` |

##### `WaveformAudit.to_pdf`

```python
to_pdf(self, path: 'str | Path', *, channels: 'Iterable[str] | None' = None, channels_per_page: 'int' = 24, full_scale: 'bool' = False, include_flagged_full_scale: 'bool' = True, max_flagged_full_scale_channels: 'int | None' = 24, title: 'str | None' = None) -> 'Path'
```

Write an atomic, multipage review atlas for every selected channel.

The main atlas uses robust limits so physiological morphology remains
visible in the presence of a large rejected trial. By default, channels
requiring review are then repeated at full scale so clipping, saturation,
drift, and sharp transients can be inspected directly.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `channels` | Iterable[str] \| None | `None` |
| `channels_per_page` | int | `24` |
| `full_scale` | bool | `False` |
| `include_flagged_full_scale` | bool | `True` |
| `max_flagged_full_scale_channels` | int \| None | `24` |
| `title` | str \| None | `None` |

**Returns:** Path.

### Functions

### `band_power`

```python
band_power(epochs, bands: 'Mapping[str, tuple[float, float]] | None' = None, channels: 'Iterable[str] | None' = None, *, on: 'str' = 'trials', relative: 'bool' = False) -> 'pd.DataFrame'
```

Average PSD power within named frequency bands -> channels x bands DataFrame.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `bands` | Mapping[str, tuple[float, float]] \| None | `None` |
| `channels` | Iterable[str] \| None | `None` |
| `on` | str | `'trials'` |
| `relative` | bool | `False` |

**Returns:** pd.DataFrame.

### `bidirectional_edge_summary`

```python
bidirectional_edge_summary(edges: 'pd.DataFrame', *, source_col: 'str' = 'source', target_col: 'str' = 'target', weight_col: 'str' = 'weight', group_cols: 'Iterable[str] | None' = None, min_weight: 'float' = 0.0, agg: 'str' = 'mean', eps: 'float' = 1e-12) -> 'pd.DataFrame'
```

Summarize reciprocal directed edges and directional asymmetry.

Rows are unordered node pairs within optional grouping columns. ``a_to_b``
and ``b_to_a`` are aggregated directed weights; ``asymmetry_log2`` is
positive when the first listed direction is stronger and negative when the
reverse direction is stronger.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `source_col` | str | `'source'` |
| `target_col` | str | `'target'` |
| `weight_col` | str | `'weight'` |
| `group_cols` | Iterable[str] \| None | `None` |
| `min_weight` | float | `0.0` |
| `agg` | str | `'mean'` |
| `eps` | float | `1e-12` |

**Returns:** pd.DataFrame.

### `canonicalize_connectivity_edges`

```python
canonicalize_connectivity_edges(edges: 'pd.DataFrame', *, source_col: 'str' = 'source', target_col: 'str' = 'target', directed_col: 'str' = 'directed', method_col: 'str' = 'method', undirected_methods: 'Iterable[str]' = ('plv', 'coherence', 'mutual_information')) -> 'pd.DataFrame'
```

Store undirected edge endpoints in one stable orientation.

Directed rows, including Granger-causality edges, are left unchanged.
Source/target anatomy and coordinate columns are swapped with their
corresponding endpoint labels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `source_col` | str | `'source'` |
| `target_col` | str | `'target'` |
| `directed_col` | str | `'directed'` |
| `method_col` | str | `'method'` |
| `undirected_methods` | Iterable[str] | `('plv', 'coherence', 'mutual_information')` |

**Returns:** pd.DataFrame.

### `coherence_matrix`

```python
coherence_matrix(epochs, band: 'tuple[float, float]' = (8.0, 13.0), channels: 'Iterable[str] | None' = None, *, mode: 'str' = 'coherence', time_window: 'tuple[float, float] | None' = None) -> 'pd.DataFrame'
```

Band-averaged spectral coherence connectivity matrix across trials.

``mode="coherence"`` returns magnitude-squared coherence; ``mode="imaginary"``
returns the imaginary part of coherency (insensitive to zero-lag/volume-
conduction effects). Cross-spectra are estimated over trials, which act as
the averaging ensemble for evoked data.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `band` | tuple[float, float] | `(8.0, 13.0)` |
| `channels` | Iterable[str] \| None | `None` |
| `mode` | str | `'coherence'` |
| `time_window` | tuple[float, float] \| None | `None` |

**Returns:** pd.DataFrame.

### `coherence_matrix_from_dataframe`

```python
coherence_matrix_from_dataframe(df: 'pd.DataFrame', sfreq: 'float', *, band: 'tuple[float, float]' = (8.0, 13.0), channels: 'Iterable[str] | None' = None, time_window: 'tuple[object, object] | None' = None, mode: 'str' = 'coherence', nperseg: 'int | None' = None) -> 'pd.DataFrame'
```

Estimate band-averaged coherence in a continuous multichannel window.

For ``mode="coherence"``, ERPy estimates magnitude-squared coherence,

``C_ij(f) = abs(S_ij(f))**2 / (S_ii(f) * S_jj(f))``,

from Welch cross-spectral densities and averages it across frequencies in
``band``. For ``mode="imaginary"``, the returned quantity is the band mean
of the absolute imaginary part of coherency, which reduces contributions
with zero phase lag. Both modes summarize continuous samples; use
``coherence_matrix`` when trials provide the averaging ensemble.

Parameters
----------
df:
    Samples-by-channels table. The index may contain numeric time values or
    datetimes; nonnumeric columns are ignored.
sfreq:
    Sampling frequency in hertz.
band:
    Inclusive lower and upper frequency bounds in hertz.
channels:
    Channel columns to include, in output order. By default, all numeric
    columns are used.
time_window:
    Inclusive start and stop boundaries in the same coordinate system as
    ``df.index``. The complete table is used when omitted.
mode:
    ``"coherence"`` for magnitude-squared coherence or ``"imaginary"``
    for absolute imaginary coherency.
nperseg:
    Welch segment length in samples. ERPy chooses a length compatible with
    the available window when omitted.

Returns
-------
pd.DataFrame
    Symmetric channel-by-channel matrix. Magnitude-squared coherence has a
    unit diagonal; imaginary coherency has a zero diagonal. Pairwise values
    are ``NaN`` when fewer than eight joint finite samples are available.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `sfreq` | float | required |
| `band` | tuple[float, float] | `(8.0, 13.0)` |
| `channels` | Iterable[str] \| None | `None` |
| `time_window` | tuple[object, object] \| None | `None` |
| `mode` | str | `'coherence'` |
| `nperseg` | int \| None | `None` |

**Returns:** pd.DataFrame.

### `connectivity_matrix_from_dataframe`

```python
connectivity_matrix_from_dataframe(df: 'pd.DataFrame', sfreq: 'float', *, method: 'str', channels: 'Iterable[str] | None' = None, band: 'tuple[float, float]' = (8.0, 13.0), time_window: 'tuple[object, object] | None' = None, **kwargs) -> 'pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]'
```

Dispatch common continuous-window connectivity metrics.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `sfreq` | float | required |
| `method` | str | required |
| `channels` | Iterable[str] \| None | `None` |
| `band` | tuple[float, float] | `(8.0, 13.0)` |
| `time_window` | tuple[object, object] \| None | `None` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame \| tuple[pd.DataFrame, pd.DataFrame].

### `complete_contact_coordinates`

```python
complete_contact_coordinates(contacts: 'pd.DataFrame', *, maximum_fit_rmse_mm: 'float' = 2.5, maximum_contact_step_mm: 'float' = 15.0) -> 'pd.DataFrame'
```

Fill an isolated missing contact from a well-fit linear lead trajectory.

Contacts are grouped by patient, session, and the nonnumeric prefix of
``elec_label``. A three-dimensional linear fit is accepted only when at
least four localized contacts are available, the fit error is within
``maximum_fit_rmse_mm``, and the estimated contact spacing is plausible.
Interpolation or one-contact extrapolation is allowed; an entirely
unlocalized lead is left unchanged.

The returned table records ``mni_coordinate_source``, fit RMSE, and the
number of localized contacts used. These provenance columns distinguish
inferred positions from measured or imported coordinates.

Parameters
----------
contacts:
    Electrode metadata containing ``elec_label`` and, when available,
    ``mni_x``, ``mni_y``, and ``mni_z``. ``patient_id`` and ``session_id``
    are used to prevent fits from crossing acquisitions.
maximum_fit_rmse_mm:
    Largest accepted three-dimensional root-mean-square fit residual.
maximum_contact_step_mm:
    Largest accepted distance between adjacent fitted contacts. The lower
    bound is fixed at 0.5 mm.

Returns
-------
pandas.DataFrame
    A copy of the input table with conservatively completed coordinates
    and explicit coordinate-provenance columns.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `contacts` | pd.DataFrame | required |
| `maximum_fit_rmse_mm` | float | `2.5` |
| `maximum_contact_step_mm` | float | `15.0` |

**Returns:** pd.DataFrame.

### `compute_ersp`

```python
compute_ersp(epochs, channel: 'str', config: 'SpectralConfig | None' = None) -> 'TFRResult'
```

Event-related spectral perturbation (baseline-normalized TFR power).

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |

**Returns:** TFRResult.

### `compute_psd`

```python
compute_psd(epochs, channels: 'Iterable[str] | str | None' = None, *, fmin: 'float' = 1.0, fmax: 'float | None' = None, on: 'str' = 'trials', nperseg: 'int | None' = None, detrend: 'str' = 'constant', relative: 'bool' = False) -> 'pd.DataFrame'
```

Welch power spectral density per channel.

``on="trials"`` computes the PSD per trial and averages across trials (the
standard estimate for evoked data); ``on="mean"`` computes the PSD of the
trial-mean response waveform. Returns a DataFrame indexed by frequency with
one column per channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channels` | Iterable[str] \| str \| None | `None` |
| `fmin` | float | `1.0` |
| `fmax` | float \| None | `None` |
| `on` | str | `'trials'` |
| `nperseg` | int \| None | `None` |
| `detrend` | str | `'constant'` |
| `relative` | bool | `False` |

**Returns:** pd.DataFrame.

### `compute_tfr`

```python
compute_tfr(epochs, channel: 'str', config: 'SpectralConfig | None' = None, *, return_per_trial: 'bool' = False) -> 'TFRResult'
```

Morlet time-frequency power and inter-trial phase coherence for one channel.

When ``config.baseline`` is set, ``power`` is the baseline-normalized
event-related spectral perturbation (ERSP). ``itc`` is the inter-trial phase
coherence (a.k.a. inter-trial phase clustering / phase locking to stimulus).

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `return_per_trial` | bool | `False` |

**Returns:** TFRResult.

### `common_mode_recovery_diagnostic`

```python
common_mode_recovery_diagnostic(frame: 'pd.DataFrame', *, response_window: 'tuple[float, float]' = (0.01, 0.35), thresholds: 'Mapping[str, Any] | None' = None, consensus_column: 'str | None' = None) -> 'dict[str, Any]'
```

Detect an acquisition-wide early step followed by shared recovery.

A physiological early response may occur on one or several connected
contacts. Amplifier recovery is different: many anatomically distributed
detector-positive channels begin with a large offset, peak at nearly the
same first post-blanking sample, and drift together toward baseline. The
returned diagnostics make the acquisition-level quarantine auditable.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `frame` | pd.DataFrame | required |
| `response_window` | tuple[float, float] | `(0.01, 0.35)` |
| `thresholds` | Mapping[str, Any] \| None | `None` |
| `consensus_column` | str \| None | `None` |

**Returns:** dict[str, Any].

### `compare_crp_across_stim_sites`

```python
compare_crp_across_stim_sites(crp_tables: 'dict[str, pd.DataFrame] | pd.DataFrame', *, stim_col: 'str' = 'stim_pair', channel_col: 'str' = 'channel', weight_col: 'str' = 'mean_projection', baseline_col: 'str' = 'max_canonical_weight_z', reconstruction_col: 'str' = 'reconstruction_rms_uv', permutation_col: 'str' = 'permutation_p_value', alpha: 'float' = 0.05) -> 'pd.DataFrame'
```

Compare CRP response expression across stimulation sites.

Accept either one table containing ``stim_col`` or a mapping from
stimulation-pair labels to tables returned by ``run_crp_all``. The
returned long-form table preserves the source columns and adds within-site
z-scores, baseline-normalized expression, reconstruction ranks, and the
supplied permutation-test summaries. ``permutation_significant`` applies
the unadjusted ``alpha`` threshold; this helper does not perform
multiplicity correction.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `crp_tables` | dict[str, pd.DataFrame] \| pd.DataFrame | required |
| `stim_col` | str | `'stim_pair'` |
| `channel_col` | str | `'channel'` |
| `weight_col` | str | `'mean_projection'` |
| `baseline_col` | str | `'max_canonical_weight_z'` |
| `reconstruction_col` | str | `'reconstruction_rms_uv'` |
| `permutation_col` | str | `'permutation_p_value'` |
| `alpha` | float | `0.05` |

**Returns:** pd.DataFrame.

### `create_events_from_binary_series`

```python
create_events_from_binary_series(series: 'Iterable[Any] | pd.Series | pd.DataFrame', start_datetime: 'Any | None' = None, end_datetime: 'Any | None' = None, sfreq: 'float | None' = None, event_value: 'int | float | str' = 1, event_column: 'int | str' = 0, edge: 'str' = 'rising') -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Convert a binary marker vector into a datetime-indexed table and events.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `series` | Iterable[Any] \| pd.Series \| pd.DataFrame | required |
| `start_datetime` | Any \| None | `None` |
| `end_datetime` | Any \| None | `None` |
| `sfreq` | float \| None | `None` |
| `event_value` | int \| float \| str | `1` |
| `event_column` | int \| str | `0` |
| `edge` | str | `'rising'` |

**Returns:** tuple[pd.DataFrame, pd.DataFrame].

### `create_events_from_timestamps`

```python
create_events_from_timestamps(timestamps: 'Iterable[Any] | pd.Series | pd.DataFrame', label: 'str | None' = None, time_column: 'str' = 'times') -> 'pd.DataFrame'
```

Create canonical ERPy event annotations from timestamp rows.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `timestamps` | Iterable[Any] \| pd.Series \| pd.DataFrame | required |
| `label` | str \| None | `None` |
| `time_column` | str | `'times'` |

**Returns:** pd.DataFrame.

### `compute_dynamic_graph_metrics`

```python
compute_dynamic_graph_metrics(edge_time_table: 'pd.DataFrame', *, metrics: 'Iterable[str]' = ('in_strength', 'out_strength', 'total_strength', 'hub', 'authority'), time_col: 'str' = 'time_s', source_col: 'str' = 'source', target_col: 'str' = 'target', weight_col: 'str' = 'weight', min_weight: 'float' = 0.0) -> 'pd.DataFrame'
```

Compute node-level graph metrics for every sampled response time.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edge_time_table` | pd.DataFrame | required |
| `metrics` | Iterable[str] | `('in_strength', 'out_strength', 'total_strength', 'hub', 'authority')` |
| `time_col` | str | `'time_s'` |
| `source_col` | str | `'source'` |
| `target_col` | str | `'target'` |
| `weight_col` | str | `'weight'` |
| `min_weight` | float | `0.0` |

**Returns:** pd.DataFrame.

### `dynamic_target_distribution_metrics`

```python
dynamic_target_distribution_metrics(distributions: 'pd.DataFrame', *, group_cols: 'Iterable[str]', time_col: 'str' = 'time_s', target_col: 'str' = 'target', weight_col: 'str' = 'weight', scaffold_nodes: 'Iterable[str] | None' = None, agg: 'str' = 'sum') -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Quantify a perturbed node's target distribution through time.

This is a source-node broadcast summary, not a whole-network community
analysis. Participation and entropy describe how evenly nonnegative evoked
weight is distributed across the target nodes sampled for that source.
Consecutive Jensen-Shannon distance measures target-profile reconfiguration.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `distributions` | pd.DataFrame | required |
| `group_cols` | Iterable[str] | required |
| `time_col` | str | `'time_s'` |
| `target_col` | str | `'target'` |
| `weight_col` | str | `'weight'` |
| `scaffold_nodes` | Iterable[str] \| None | `None` |
| `agg` | str | `'sum'` |

**Returns:** tuple[pd.DataFrame, pd.DataFrame].

### `dataframe_time_window`

```python
dataframe_time_window(df: 'pd.DataFrame', start: 'object', stop: 'object') -> 'pd.DataFrame'
```

Return rows between two numeric or datetime boundaries.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `start` | object | required |
| `stop` | object | required |

**Returns:** pd.DataFrame.

### `estimate_latency_gradient`

```python
estimate_latency_gradient(wavefront: 'pd.DataFrame', *, group_cols: 'Iterable[str] | None' = None, latency_col: 'str' = 'latency_ms', coord_cols: 'tuple[str, str, str]' = ('mni_x', 'mni_y', 'mni_z'), weight_col: 'str | None' = 'metric_abs', min_points: 'int' = 4) -> 'pd.DataFrame'
```

Estimate latency gradients and apparent propagation speeds.

Latency is modeled as ``latency_ms ~ x + y + z`` within each group. The
gradient norm has units of milliseconds per millimeter; its reciprocal is
reported as ``apparent_speed_m_per_s`` because 1 mm/ms equals 1 m/s.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `wavefront` | pd.DataFrame | required |
| `group_cols` | Iterable[str] \| None | `None` |
| `latency_col` | str | `'latency_ms'` |
| `coord_cols` | tuple[str, str, str] | `('mni_x', 'mni_y', 'mni_z')` |
| `weight_col` | str \| None | `'metric_abs'` |
| `min_points` | int | `4` |

**Returns:** pd.DataFrame.

### `latency_distance_permutation_test`

```python
latency_distance_permutation_test(wavefront: 'pd.DataFrame', *, group_cols: 'Iterable[str] | None' = None, latency_col: 'str' = 'latency_ms', distance_col: 'str' = 'distance_from_stim_mm', weight_col: 'str | None' = 'metric_abs', min_points: 'int' = 4, n_permutations: 'int' = 2000, fdr_alpha: 'float' = 0.05, random_state: 'int | np.random.Generator | None' = 0) -> 'pd.DataFrame'
```

Test whether response latency is ordered by distance from stimulation.

Latency is modeled as ``latency_ms ~ distance_from_stim_mm``. A positive
slope indicates later responses at more distant contacts. The permutation
null shuffles latency, and optional response weights with it, over the fixed
contact distances. Euclidean distance is a spatial proxy rather than tract
length, so the returned speed is explicitly an apparent radial speed.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `wavefront` | pd.DataFrame | required |
| `group_cols` | Iterable[str] \| None | `None` |
| `latency_col` | str | `'latency_ms'` |
| `distance_col` | str | `'distance_from_stim_mm'` |
| `weight_col` | str \| None | `'metric_abs'` |
| `min_points` | int | `4` |
| `n_permutations` | int | `2000` |
| `fdr_alpha` | float | `0.05` |
| `random_state` | int \| np.random.Generator \| None | `0` |

**Returns:** pd.DataFrame.

### `latency_gradient_permutation_test`

```python
latency_gradient_permutation_test(wavefront: 'pd.DataFrame', *, group_cols: 'Iterable[str] | None' = None, latency_col: 'str' = 'latency_ms', coord_cols: 'tuple[str, str, str]' = ('mni_x', 'mni_y', 'mni_z'), weight_col: 'str | None' = 'metric_abs', min_points: 'int' = 4, n_permutations: 'int' = 2000, fdr_alpha: 'float' = 0.05, random_state: 'int | np.random.Generator | None' = 0) -> 'pd.DataFrame'
```

Test spatial latency gradients against a contact-permutation null.

Latency and optional response weight are permuted together across fixed MNI
coordinates. This preserves the latency-amplitude relationship while
breaking spatial organization. The returned ``gradient_p_value`` tests
whether the observed weighted planar-model R-squared exceeds the null, and
``gradient_q_value`` applies Benjamini-Hochberg correction across groups.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `wavefront` | pd.DataFrame | required |
| `group_cols` | Iterable[str] \| None | `None` |
| `latency_col` | str | `'latency_ms'` |
| `coord_cols` | tuple[str, str, str] | `('mni_x', 'mni_y', 'mni_z')` |
| `weight_col` | str \| None | `'metric_abs'` |
| `min_points` | int | `4` |
| `n_permutations` | int | `2000` |
| `fdr_alpha` | float | `0.05` |
| `random_state` | int \| np.random.Generator \| None | `0` |

**Returns:** pd.DataFrame.

### `exact_sign_flip_test`

```python
exact_sign_flip_test(values: 'Iterable[float]', *, null_value: 'float' = 0.0, alternative: 'str' = 'two-sided', statistic: 'str | Callable[[np.ndarray], float]' = 'mean', max_exact_observations: 'int' = 20, n_resamples: 'int' = 100000, random_state: 'int | np.random.Generator | None' = 0) -> 'SignFlipResult'
```

Test a paired or one-sample contrast by flipping observation signs.

The test is exact when the number of finite observations does not exceed
``max_exact_observations``. For larger samples, deterministic Monte Carlo
resampling is used by default.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `values` | Iterable[float] | required |
| `null_value` | float | `0.0` |
| `alternative` | str | `'two-sided'` |
| `statistic` | str \| Callable[[np.ndarray], float] | `'mean'` |
| `max_exact_observations` | int | `20` |
| `n_resamples` | int | `100000` |
| `random_state` | int \| np.random.Generator \| None | `0` |

**Returns:** SignFlipResult.

### `fdr_bh`

```python
fdr_bh(p_values: 'Iterable[float]', *, alpha: 'float' = 0.05) -> 'tuple[np.ndarray, np.ndarray]'
```

Adjust p-values with the Benjamini-Hochberg FDR procedure.

Nonfinite inputs remain nonfinite in the adjusted array and are never
rejected. The returned arrays preserve the input order.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `p_values` | Iterable[float] | required |
| `alpha` | float | `0.05` |

**Returns:** tuple[np.ndarray, np.ndarray].

### `fixed_window_reproducibility_sign_flip_test`

```python
fixed_window_reproducibility_sign_flip_test(response: 'np.ndarray', *, sfreq: 'float', n_permutations: 'int' = 5000, max_exact_trials: 'int' = 12, random_state: 'int | np.random.Generator | None' = 42, eps: 'float' = 1e-12) -> 'FixedWindowReproducibilityResult'
```

Test fixed-window trial reproducibility by whole-trial sign flips.

The statistic is the mean of all ordered, semi-normalized cross-trial
projections. Whole-trial Rademacher signs preserve every trial norm and
account for the dependence among projections that share trials. The
response window must be fixed before this function is called; a
data-selected CRP duration must not define this inferential array.

For exact enumeration, the observed all-positive assignment is omitted
from the matrix calculation and counted explicitly. This guarantees the
exact randomization floor even when floating-point summation order makes
the separately calculated observed statistic differ by roundoff.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `response` | np.ndarray | required |
| `sfreq` | float | required |
| `n_permutations` | int | `5000` |
| `max_exact_trials` | int | `12` |
| `random_state` | int \| np.random.Generator \| None | `42` |
| `eps` | float | `1e-12` |

**Returns:** FixedWindowReproducibilityResult.

### `granger_causality_matrix`

```python
granger_causality_matrix(epochs, channels: 'Iterable[str] | None' = None, *, time_window: 'tuple[float, float] | None' = None, maxlag: 'int' = 5, return_pvalues: 'bool' = False) -> 'pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]'
```

Pairwise linear Granger causality matrix across trials.

Rows are source channels and columns are target channels. Trial boundaries
are preserved when constructing lagged regressions.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channels` | Iterable[str] \| None | `None` |
| `time_window` | tuple[float, float] \| None | `None` |
| `maxlag` | int | `5` |
| `return_pvalues` | bool | `False` |

**Returns:** pd.DataFrame \| tuple[pd.DataFrame, pd.DataFrame].

### `granger_causality_matrix_from_dataframe`

```python
granger_causality_matrix_from_dataframe(df: 'pd.DataFrame', *, channels: 'Iterable[str] | None' = None, time_window: 'tuple[object, object] | None' = None, maxlag: 'int' = 5, return_pvalues: 'bool' = False) -> 'pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]'
```

Estimate directed pairwise Granger causality in a data window.

Matrix rows are putative source channels and columns are targets. When
``return_pvalues`` is true, the F-statistic and p-value matrices are
returned as a pair.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `channels` | Iterable[str] \| None | `None` |
| `time_window` | tuple[object, object] \| None | `None` |
| `maxlag` | int | `5` |
| `return_pvalues` | bool | `False` |

**Returns:** pd.DataFrame \| tuple[pd.DataFrame, pd.DataFrame].

### `inter_trial_phase_coherence`

```python
inter_trial_phase_coherence(epochs, channel: 'str', config: 'SpectralConfig | None' = None) -> 'TFRResult'
```

Inter-trial phase coherence / clustering (ITPC) time-frequency map.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |

**Returns:** TFRResult.

### `hierarchical_bootstrap_interval`

```python
hierarchical_bootstrap_interval(data: 'pd.DataFrame', *, value_col: 'str', levels: 'Sequence[str]', statistic: 'str | Callable[[np.ndarray], float]' = 'median', confidence: 'float' = 0.95, n_resamples: 'int' = 5000, random_state: 'int | np.random.Generator | None' = 0) -> 'BootstrapInterval'
```

Estimate an equal-top-cluster interval for nested observations.

The first level is treated as the biological replicate. Each bootstrap
draw resamples those clusters, then recursively resamples lower-level
clusters and observations. A statistic is computed within every selected
top-level cluster and then across those cluster statistics, preventing
patients with more acquisitions or contacts from dominating the estimate.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `data` | pd.DataFrame | required |
| `value_col` | str | required |
| `levels` | Sequence[str] | required |
| `statistic` | str \| Callable[[np.ndarray], float] | `'median'` |
| `confidence` | float | `0.95` |
| `n_resamples` | int | `5000` |
| `random_state` | int \| np.random.Generator \| None | `0` |

**Returns:** BootstrapInterval.

### `load_continuous_npz`

```python
load_continuous_npz(path: 'str | Path', *, nrows: 'int | None' = None) -> 'pd.DataFrame'
```

Load an ERPy continuous NPZ file without enabling pickle deserialization.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `path` | str \| Path | required |
| `nrows` | int \| None | `None` |

**Returns:** pd.DataFrame.

### `mutual_information_matrix`

```python
mutual_information_matrix(epochs, channels: 'Iterable[str] | None' = None, *, time_window: 'tuple[float, float] | None' = None, n_bins: 'int' = 16, normalized: 'bool' = True) -> 'pd.DataFrame'
```

Histogram mutual-information matrix across trial/time samples.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channels` | Iterable[str] \| None | `None` |
| `time_window` | tuple[float, float] \| None | `None` |
| `n_bins` | int | `16` |
| `normalized` | bool | `True` |

**Returns:** pd.DataFrame.

### `mutual_information_matrix_from_dataframe`

```python
mutual_information_matrix_from_dataframe(df: 'pd.DataFrame', *, channels: 'Iterable[str] | None' = None, time_window: 'tuple[object, object] | None' = None, n_bins: 'int' = 16, normalized: 'bool' = True) -> 'pd.DataFrame'
```

Estimate pairwise mutual information in a continuous data window.

Numeric columns are discretized into ``n_bins`` before estimating the
symmetric channel matrix. Normalized values range from zero to one.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `channels` | Iterable[str] \| None | `None` |
| `time_window` | tuple[object, object] \| None | `None` |
| `n_bins` | int | `16` |
| `normalized` | bool | `True` |

**Returns:** pd.DataFrame.

### `multitaper_coherence_matrices_from_dataframe`

```python
multitaper_coherence_matrices_from_dataframe(df: 'pd.DataFrame', sfreq: 'float', *, band: 'tuple[float, float]' = (8.0, 13.0), channels: 'Iterable[str] | None' = None, time_window: 'tuple[object, object] | None' = None, time_bandwidth: 'float' = 3.0, n_tapers: 'int' = 5) -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Multitaper coherence and absolute imaginary-coherency matrices.

Both matrices use one shared DPSS decomposition. This is especially useful
for short matched windows where Welch segmentation would leave few stable
alpha-band estimates. Imaginary coherency is returned as a prespecified
sensitivity measure for zero-lag coupling.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `sfreq` | float | required |
| `band` | tuple[float, float] | `(8.0, 13.0)` |
| `channels` | Iterable[str] \| None | `None` |
| `time_window` | tuple[object, object] \| None | `None` |
| `time_bandwidth` | float | `3.0` |
| `n_tapers` | int | `5` |

**Returns:** tuple[pd.DataFrame, pd.DataFrame].

### `opportunity_conditioned_reciprocity`

```python
opportunity_conditioned_reciprocity(edges: 'pd.DataFrame', *, source_col: 'str' = 'source', target_col: 'str' = 'target', response_col: 'str' = 'responding', group_cols: 'Iterable[str] | None' = None) -> 'pd.DataFrame'
```

Estimate reciprocity only where both directions were sampled.

The denominator is the number of bidirectionally tested unordered node
pairs with at least one detected response. This prevents an unavailable
reverse stimulation direction from being interpreted as a negative edge.
Duplicate directed opportunities are collapsed with an ``any`` response
rule before pair-level counting.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `source_col` | str | `'source'` |
| `target_col` | str | `'target'` |
| `response_col` | str | `'responding'` |
| `group_cols` | Iterable[str] \| None | `None` |

**Returns:** pd.DataFrame.

### `phase_amplitude_comodulogram`

```python
phase_amplitude_comodulogram(epochs, phase_channel: 'str', amp_channel: 'str | None' = None, *, phase_freqs: 'Iterable[float] | None' = None, amp_freqs: 'Iterable[float] | None' = None, bandwidth_phase: 'float' = 2.0, bandwidth_amp: 'float' = 20.0, method: 'str' = 'tort', n_bins: 'int' = 18, time_window: 'tuple[float, float] | None' = None) -> 'ComodulogramResult'
```

PAC comodulogram: modulation index over a grid of phase and amplitude bands.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `phase_freqs` | Iterable[float] \| None | `None` |
| `amp_freqs` | Iterable[float] \| None | `None` |
| `bandwidth_phase` | float | `2.0` |
| `bandwidth_amp` | float | `20.0` |
| `method` | str | `'tort'` |
| `n_bins` | int | `18` |
| `time_window` | tuple[float, float] \| None | `None` |

**Returns:** ComodulogramResult.

### `phase_amplitude_coupling`

```python
phase_amplitude_coupling(epochs, phase_channel: 'str', amp_channel: 'str | None' = None, *, phase_band: 'tuple[float, float]' = (4.0, 8.0), amp_band: 'tuple[float, float]' = (80.0, 150.0), method: 'str' = 'tort', n_bins: 'int' = 18, time_window: 'tuple[float, float] | None' = None, order: 'int' = 4) -> 'PACResult'
```

Phase-amplitude coupling within or between contacts.

The low-frequency phase of ``phase_channel`` modulates the high-frequency
amplitude of ``amp_channel`` (defaults to the same contact). ``method="tort"``
returns the normalized Kullback-Leibler modulation index; ``method="mvl"``
returns the Canolty mean-vector-length. Trials are concatenated over the
selected ``time_window`` to estimate the coupling.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `phase_band` | tuple[float, float] | `(4.0, 8.0)` |
| `amp_band` | tuple[float, float] | `(80.0, 150.0)` |
| `method` | str | `'tort'` |
| `n_bins` | int | `18` |
| `time_window` | tuple[float, float] \| None | `None` |
| `order` | int | `4` |

**Returns:** PACResult.

### `phase_locking_value`

```python
phase_locking_value(epochs, ch_x: 'str', ch_y: 'str', config: 'SpectralConfig | None' = None) -> 'PLVResult'
```

Time-frequency phase-locking value between two contacts across trials.

PLV(f, t) = |mean over trials exp(i (phi_x - phi_y))|, a measure of how
consistently the two contacts hold a fixed phase relationship at each
time-frequency point.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `ch_x` | str | required |
| `ch_y` | str | required |
| `config` | SpectralConfig \| None | `None` |

**Returns:** PLVResult.

### `paired_log_rms_sign_flip_test`

```python
paired_log_rms_sign_flip_test(log_rms_differences: 'np.ndarray', *, n_permutations: 'int' = 5000, max_exact_trials: 'int' = 16, random_state: 'int | np.random.Generator | None' = 42) -> 'tuple[float, float, bool, int]'
```

One-sided paired sign-flip test of mean log RMS ratio against zero.

Exact enumeration omits the observed all-positive assignment and adds it
back through the standard plus-one formula. This is algebraically the
exact randomization p-value while sharing the Monte Carlo implementation's
finite-sample form.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `log_rms_differences` | np.ndarray | required |
| `n_permutations` | int | `5000` |
| `max_exact_trials` | int | `16` |
| `random_state` | int \| np.random.Generator \| None | `42` |

**Returns:** tuple[float, float, bool, int].

### `phase_phase_coupling`

```python
phase_phase_coupling(epochs, ch_x: 'str', ch_y: 'str', *, freq_x: 'float', freq_y: 'float', n: 'int' = 1, m: 'int' = 1, n_cycles: 'float' = 7.0, time_window: 'tuple[float, float] | None' = None) -> 'float'
```

n:m phase-phase coupling between two contacts (or one contact, two bands).

Returns the n:m phase-locking value
``|mean exp(i (n*phi_x - m*phi_y))|`` over trials and the selected time
window, where ``phi_x`` is the phase at ``freq_x`` and ``phi_y`` at
``freq_y``. ``ch_x == ch_y`` gives within-contact harmonic coupling.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `ch_x` | str | required |
| `ch_y` | str | required |
| `freq_x` | float | required |
| `freq_y` | float | required |
| `n` | int | `1` |
| `m` | int | `1` |
| `n_cycles` | float | `7.0` |
| `time_window` | tuple[float, float] \| None | `None` |

**Returns:** float.

### `phase_phase_matrix`

```python
phase_phase_matrix(epochs, ch_x: 'str', ch_y: 'str | None' = None, *, freqs_x: 'Iterable[float] | None' = None, freqs_y: 'Iterable[float] | None' = None, n: 'int' = 1, m: 'int' = 1, n_cycles: 'float' = 7.0, time_window: 'tuple[float, float] | None' = None) -> 'pd.DataFrame'
```

n:m phase-locking over a grid of frequencies -> DataFrame [freq_y x freq_x].

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `ch_x` | str | required |
| `ch_y` | str \| None | `None` |
| `freqs_x` | Iterable[float] \| None | `None` |
| `freqs_y` | Iterable[float] \| None | `None` |
| `n` | int | `1` |
| `m` | int | `1` |
| `n_cycles` | float | `7.0` |
| `time_window` | tuple[float, float] \| None | `None` |

**Returns:** pd.DataFrame.

### `plv_matrix_from_dataframe`

```python
plv_matrix_from_dataframe(df: 'pd.DataFrame', sfreq: 'float', *, band: 'tuple[float, float]' = (8.0, 13.0), channels: 'Iterable[str] | None' = None, time_window: 'tuple[object, object] | None' = None, order: 'int' = 4) -> 'pd.DataFrame'
```

Estimate band-limited phase-locking value in a continuous window.

ERPy band-pass filters each selected numeric channel, obtains its analytic
phase with a Hilbert transform, and computes the pairwise phase-locking
value (PLV)

``PLV_ij = abs(mean(exp(1j * (phase_i - phase_j))))``

across samples in the requested interval. Unlike ``plv_matrix``, this
function treats continuous time samples, rather than trials, as the
averaging observations.

Parameters
----------
df:
    Samples-by-channels table. The index may contain numeric time values or
    datetimes; nonnumeric columns are ignored.
sfreq:
    Sampling frequency in hertz.
band:
    Inclusive lower and upper passband frequencies in hertz.
channels:
    Channel columns to include, in output order. By default, all numeric
    columns are used.
time_window:
    Inclusive start and stop boundaries in the same coordinate system as
    ``df.index``. The complete table is used when omitted.
order:
    Order of the Butterworth band-pass filter applied before phase
    extraction.

Returns
-------
pd.DataFrame
    Symmetric channel-by-channel PLV matrix with values from zero to one
    and a unit diagonal. A pair is ``NaN`` when no finite phase samples are
    available for that comparison.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `sfreq` | float | required |
| `band` | tuple[float, float] | `(8.0, 13.0)` |
| `channels` | Iterable[str] \| None | `None` |
| `time_window` | tuple[object, object] \| None | `None` |
| `order` | int | `4` |

**Returns:** pd.DataFrame.

### `plv_matrix`

```python
plv_matrix(epochs, band: 'tuple[float, float]' = (8.0, 13.0), channels: 'Iterable[str] | None' = None, *, time_window: 'tuple[float, float] | None' = None, order: 'int' = 4) -> 'pd.DataFrame'
```

Band-limited inter-trial PLV connectivity matrix.

For each channel the band-limited analytic phase is computed per trial; the
PLV between every channel pair is taken across trials at each time point and
averaged over ``time_window`` (defaults to the full post-stimulus epoch).
Returns a symmetric channels x channels DataFrame.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `band` | tuple[float, float] | `(8.0, 13.0)` |
| `channels` | Iterable[str] \| None | `None` |
| `time_window` | tuple[float, float] \| None | `None` |
| `order` | int | `4` |

**Returns:** pd.DataFrame.

### `post_stim_rest_window`

```python
post_stim_rest_window(df: 'pd.DataFrame', stim_stop: 'object', *, start_offset_s: 'float' = 0.0, duration_s: 'float' = 60.0) -> 'pd.DataFrame'
```

Extract a continuous rest window after a stimulation train stops.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `stim_stop` | object | required |
| `start_offset_s` | float | `0.0` |
| `duration_s` | float | `60.0` |

**Returns:** pd.DataFrame.

### `run_crp`

```python
run_crp(epochs, channel: 'str', config: 'CRPConfig | None' = None) -> 'CRPResult'
```

Fit canonical response parameterization to one epoch channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | CRPConfig \| None | `None` |

**Returns:** CRPResult.

### `run_crp_all`

```python
run_crp_all(epochs, config: 'CRPConfig | None' = None) -> 'pd.DataFrame'
```

Fit canonical response parameterization to every epoch channel.

Per-channel failures are retained as nonsignificant rows with an ``error``
field so that one unsuitable channel does not discard the full analysis.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `config` | CRPConfig \| None | `None` |

**Returns:** pd.DataFrame.

### `run_crp_energy_array`

```python
run_crp_energy_array(x: 'np.ndarray', times: 'np.ndarray', *, channel: 'str' = '', config: 'CRPEnergyConfig | None' = None) -> 'CRPEnergyResult'
```

Evaluate fixed-window reproducibility and paired energy for one channel.

Whole-trial sign flips supply the reproducibility component over the full
declared response window. Matched response and baseline segments are each
demeaned separately before paired log-RMS sign flips. A descriptive CRP
model is fitted independently of the inferential window selection.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `x` | np.ndarray | required |
| `times` | np.ndarray | required |
| `channel` | str | `''` |
| `config` | CRPEnergyConfig \| None | `None` |

**Returns:** CRPEnergyResult.

### `artifact_envelope`

```python
artifact_envelope(epoch_df: 'pd.DataFrame', stim_ch: 'str | Iterable[str] | None' = None) -> 'tuple[np.ndarray, list[str]]'
```

Return a robust stimulation-artifact envelope for one epoch.

Stimulation channels are included first when present, but the envelope also
considers the remaining numeric channels. This catches adjacent-contact or
amplifier-wide onset spikes without requiring the user to know where the
artifact is largest.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epoch_df` | pd.DataFrame | required |
| `stim_ch` | str \| Iterable[str] \| None | `None` |

**Returns:** tuple[np.ndarray, list[str]].

### `audit_waveforms`

```python
audit_waveforms(epochs: 'Any', detections: 'pd.DataFrame', artifact_responses: 'pd.DataFrame', *, response_window: 'tuple[float, float]' = (0.01, 0.35), baseline_window: 'tuple[float, float]' = (-0.5, -0.03), min_consensus: 'int' = 2, artifact_thresholds: 'Mapping[str, float] | None' = None, max_bad_response_fraction: 'float' = 0.25, max_hard_artifact_fraction: 'float' = 0.1, min_clean_responses: 'int' = 10, fdr_alpha: 'float' = 0.05) -> 'WaveformAudit'
```

Recompute waveform metrics and classify responses needing review.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | Any | required |
| `detections` | pd.DataFrame | required |
| `artifact_responses` | pd.DataFrame | required |
| `response_window` | tuple[float, float] | `(0.01, 0.35)` |
| `baseline_window` | tuple[float, float] | `(-0.5, -0.03)` |
| `min_consensus` | int | `2` |
| `artifact_thresholds` | Mapping[str, float] \| None | `None` |
| `max_bad_response_fraction` | float | `0.25` |
| `max_hard_artifact_fraction` | float | `0.1` |
| `min_clean_responses` | int | `10` |
| `fdr_alpha` | float | `0.05` |

**Returns:** WaveformAudit.

### `discover_bids_ieeg`

```python
discover_bids_ieeg(bids_root: 'str | Path', subject: 'str | None' = None, session: 'str | None' = None, task: 'str | None' = None, derivatives: 'bool' = True) -> 'pd.DataFrame'
```

Discover BIDS iEEG recordings and their available sidecar files.

The returned table contains one row per supported ``*_ieeg`` recording,
including parsed subject, session, task, and run entities plus paths to
matching events, channels, electrodes, and JSON sidecars when present.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bids_root` | str \| Path | required |
| `subject` | str \| None | `None` |
| `session` | str \| None | `None` |
| `task` | str \| None | `None` |
| `derivatives` | bool | `True` |

**Returns:** pd.DataFrame.

### `detect`

```python
detect(epochs, method: 'str' = 'crp_energy', **params: 'Any') -> 'pd.DataFrame'
```

Low-syntax detector entry point; defaults to the primary conjunction.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `method` | str | `'crp_energy'` |
| `params` | Any | required |

**Returns:** pd.DataFrame.

### `detect_artifactual_responses`

```python
detect_artifactual_responses(epochs, response_window: 'tuple[float, float] | None' = None, baseline_window: 'tuple[float, float] | None' = None, zscore_threshold: 'float' = 6.0, max_nan_fraction: 'float' = 0.02, saturation_fraction: 'float' = 0.02, plateau_fraction: 'float' = 0.015, plateau_min_run_s: 'float' = 0.003, plateau_epsilon_uv: 'float | None' = None, rail_fraction: 'float' = 0.02, rail_epsilon_fraction: 'float' = 0.002, rail_epsilon_uv: 'float | None' = None, rail_limits_uv: 'tuple[float, float] | Mapping[str, tuple[float, float]] | None' = None, absolute_peak_uv: 'float | None' = None, absolute_ptp_uv: 'float | None' = None, extreme_zscore_threshold: 'float | None' = 12.0, ringing_zscore_threshold: 'float' = 6.0, late_high_frequency_zscore_threshold: 'float | None' = None) -> 'ArtifactResponseReport'
```

Flag aberrant trial-by-channel responses after epoching.

This catches problems that are not persistent bad channels: amplifier
saturations on one trial, plateau or known-rail clipping, extreme
trial-relative amplitude outliers, late-response roughness-ratio outliers,
isolated movement/noise bursts, and NaN-heavy extracted windows. The
roughness ratio compares mean absolute sample gradients from 50–500 ms and
0–50 ms (bounded by the available epoch); the serialized
``late_high_frequency_ratio`` name is retained for compatibility and does
not denote spectral energy. Local waveform
minima and maxima are retained as an ``extreme_dwell_fraction`` diagnostic,
but are not treated as amplifier rails unless explicit ``rail_limits_uv``
are supplied.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `response_window` | tuple[float, float] \| None | `None` |
| `baseline_window` | tuple[float, float] \| None | `None` |
| `zscore_threshold` | float | `6.0` |
| `max_nan_fraction` | float | `0.02` |
| `saturation_fraction` | float | `0.02` |
| `plateau_fraction` | float | `0.015` |
| `plateau_min_run_s` | float | `0.003` |
| `plateau_epsilon_uv` | float \| None | `None` |
| `rail_fraction` | float | `0.02` |
| `rail_epsilon_fraction` | float | `0.002` |
| `rail_epsilon_uv` | float \| None | `None` |
| `rail_limits_uv` | tuple[float, float] \| Mapping[str, tuple[float, float]] \| None | `None` |
| `absolute_peak_uv` | float \| None | `None` |
| `absolute_ptp_uv` | float \| None | `None` |
| `extreme_zscore_threshold` | float \| None | `12.0` |
| `ringing_zscore_threshold` | float | `6.0` |
| `late_high_frequency_zscore_threshold` | float \| None | `None` |

**Returns:** ArtifactResponseReport.

### `detect_bad_channels`

```python
detect_bad_channels(df: 'pd.DataFrame', zscore_threshold: 'float' = 5.0, flat_std: 'float' = 1e-12, max_nan_fraction: 'float' = 0.05, min_correlation: 'float | None' = -0.2) -> 'BadChannelReport'
```

Detect flat, noisy, NaN-heavy, and anticorrelated channels.

The rule is deliberately transparent: each feature receives a robust z-score
and a channel is bad when any hard QC flag fires. The report table records
the reason string so clinical users can audit the decision.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `zscore_threshold` | float | `5.0` |
| `flat_std` | float | `1e-12` |
| `max_nan_fraction` | float | `0.05` |
| `min_correlation` | float \| None | `-0.2` |

**Returns:** BadChannelReport.

### `detect_events_from_artifacts`

```python
detect_events_from_artifacts(dataloader, session_id: 'str', stim_pair: 'str', method: 'str' = 'consensus', stim_freq: 'float | None' = None, min_peak_height: 'float' = 1000.0, raw_source: 'str' = 'auto', min_window_coverage: 'float' = 0.95, save_events: 'bool' = False, raw_file: 'str | Path | None' = None, stim_start: 'Any | None' = None, signal_data: 'pd.DataFrame | None' = None, **kwargs: 'Any') -> 'pd.DataFrame'
```

Unified event detector for artifact, trigger, annotation, and binary sources.

``method="auto"`` searches only the annotated stimulation interval. It first
tries explicit annotations/triggers and the configured amplitude threshold,
then uses a robust noise-qualified threshold down to
``auto_min_peak_height`` (100 microvolts by default). Every automatic result
must match the expected pulse count and cadence.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `dataloader` | not specified | required |
| `session_id` | str | required |
| `stim_pair` | str | required |
| `method` | str | `'consensus'` |
| `stim_freq` | float \| None | `None` |
| `min_peak_height` | float | `1000.0` |
| `raw_source` | str | `'auto'` |
| `min_window_coverage` | float | `0.95` |
| `save_events` | bool | `False` |
| `raw_file` | str \| Path \| None | `None` |
| `stim_start` | Any \| None | `None` |
| `signal_data` | pd.DataFrame \| None | `None` |
| `kwargs` | Any | required |

**Returns:** pd.DataFrame.

### `detect_post_artifact_anchor`

```python
detect_post_artifact_anchor(epoch_df: 'pd.DataFrame', *, stim_ch: 'str | Iterable[str] | None' = None, baseline_window: 'tuple[float, float] | None' = None, search_window: 'tuple[float, float]' = (-0.002, 0.02), min_anchor_time_s: 'float' = 0.003, settle_z: 'float' = 2.0, settle_slope_z: 'float' = 2.0, settle_duration_s: 'float' = 0.002, smooth_s: 'float' = 0.001, onset_artifact_z: 'float' = 6.0, consecutive_samples: 'int | None' = None) -> 'dict[str, object]'
```

Detect a post-stimulation-artifact zeroing anchor for one epoch.

The detector finds the largest artifact-envelope sample near stimulation
onset, then chooses the first post-peak sample at which the artifact has
truly *settled* — i.e. the (lightly smoothed) envelope has both returned to
near baseline (below ``settle_z``) and flattened out (sample-to-sample slope
below ``settle_slope_z`` in baseline units), and stays settled for
``settle_duration_s``. Requiring a sustained return to baseline rather than a
single threshold crossing places the anchor at the end of the artifact — the
inflection where the steep stimulation transient gives way to the
physiological response — instead of part-way down the still-decaying tail.

If no sustained settled run is found, the anchor falls back to the first
post-peak sample that meets the amplitude criterion, and finally to the
quietest sample in the search window; it never lands on the artifact peak.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epoch_df` | pd.DataFrame | required |
| `stim_ch` | str \| Iterable[str] \| None | `None` |
| `baseline_window` | tuple[float, float] \| None | `None` |
| `search_window` | tuple[float, float] | `(-0.002, 0.02)` |
| `min_anchor_time_s` | float | `0.003` |
| `settle_z` | float | `2.0` |
| `settle_slope_z` | float | `2.0` |
| `settle_duration_s` | float | `0.002` |
| `smooth_s` | float | `0.001` |
| `onset_artifact_z` | float | `6.0` |
| `consecutive_samples` | int \| None | `None` |

**Returns:** dict[str, object].

### `detect_zero_time_artifact`

```python
detect_zero_time_artifact(epoch_df: 'pd.DataFrame', *, stim_ch: 'str | Iterable[str] | None' = None, zero_time: 'float' = 0.0, baseline_window: 'tuple[float, float] | None' = None, artifact_z: 'float' = 6.0) -> 'dict[str, object]'
```

Report whether a requested zeroing time falls on a stimulation artifact.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epoch_df` | pd.DataFrame | required |
| `stim_ch` | str \| Iterable[str] \| None | `None` |
| `zero_time` | float | `0.0` |
| `baseline_window` | tuple[float, float] \| None | `None` |
| `artifact_z` | float | `6.0` |

**Returns:** dict[str, object].

### `evoked_graph_metric_timecourse`

```python
evoked_graph_metric_timecourse(runs: 'list[dict]', elec_meta: 'pd.DataFrame | None' = None, *, metrics: 'Iterable[str]' = ('in_strength', 'out_strength', 'total_strength', 'hub', 'authority'), **kwargs) -> 'pd.DataFrame'
```

Build dynamic evoked-response edges and return node graph metrics.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `runs` | list[dict] | required |
| `elec_meta` | pd.DataFrame \| None | `None` |
| `metrics` | Iterable[str] | `('in_strength', 'out_strength', 'total_strength', 'hub', 'authority')` |
| `kwargs` | not specified | required |

**Returns:** pd.DataFrame.

### `evoked_response_edges_over_time`

```python
evoked_response_edges_over_time(runs: 'list[dict]', elec_meta: 'pd.DataFrame | None' = None, *, response_window: 'tuple[float, float]' = (0.0, 0.35), frame_step_ms: 'float' = 12.0, top_n_per_stim: 'int' = 18, top_n_total: 'int | None' = 90, detection_metric: 'str' = 'peak_amplitude_uv', consensus_only: 'bool' = True, significance_column: 'str | None' = None, response_metric_frame_key: 'str' = 'response_metric_frame', metric_label: 'str' = 'Mean evoked voltage (uV)', elec_col: 'str' = 'elec_label', baseline_window: 'tuple[float, float] | None' = None, exclude_stimulation_contacts: 'bool' = True) -> 'pd.DataFrame'
```

Create a dynamic edge table from evoked-response epochs.

Rows are directed stimulation-source -> recording-contact edges at sampled
post-stimulus times. ``weight`` is the nonnegative response magnitude used
for graph metrics; ``signed_value`` preserves the original metric polarity.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `runs` | list[dict] | required |
| `elec_meta` | pd.DataFrame \| None | `None` |
| `response_window` | tuple[float, float] | `(0.0, 0.35)` |
| `frame_step_ms` | float | `12.0` |
| `top_n_per_stim` | int | `18` |
| `top_n_total` | int \| None | `90` |
| `detection_metric` | str | `'peak_amplitude_uv'` |
| `consensus_only` | bool | `True` |
| `significance_column` | str \| None | `None` |
| `response_metric_frame_key` | str | `'response_metric_frame'` |
| `metric_label` | str | `'Mean evoked voltage (uV)'` |
| `elec_col` | str | `'elec_label'` |
| `baseline_window` | tuple[float, float] \| None | `None` |
| `exclude_stimulation_contacts` | bool | `True` |

**Returns:** pd.DataFrame.

### `evoked_to_datetime`

```python
evoked_to_datetime(df: 'pd.DataFrame', start_datetime: 'Any', end_datetime: 'Any', event_column: 'int | str' = 0, event_value: 'int | float | str' = 1) -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Assign a datetime axis and extract rising binary-event onsets.

This compatibility helper returns both the timestamp-indexed input frame
and a one-column ``times`` table containing detected onset timestamps.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `start_datetime` | Any | required |
| `end_datetime` | Any | required |
| `event_column` | int \| str | `0` |
| `event_value` | int \| float \| str | `1` |

**Returns:** tuple[pd.DataFrame, pd.DataFrame].

### `import_bids_project`

```python
import_bids_project(bids_root: 'str | Path', output_root: 'str | Path', subject: 'str', session: 'str | None' = None, task: 'str | None' = None, patient_id: 'str | None' = None, copy_raw: 'bool' = False, derivatives: 'bool' = True) -> 'BIDSImportResult'
```

Build an ERPy config/metadata project from BIDS/OpenNeuro iEEG files.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `bids_root` | str \| Path | required |
| `output_root` | str \| Path | required |
| `subject` | str | required |
| `session` | str \| None | `None` |
| `task` | str \| None | `None` |
| `patient_id` | str \| None | `None` |
| `copy_raw` | bool | `False` |
| `derivatives` | bool | `True` |

**Returns:** BIDSImportResult.

### `process_events_for_session`

```python
process_events_for_session(dataloader, session_id: 'str', method: 'str' = 'consensus', stim_pairs: 'Iterable[str] | None' = None, save_events: 'bool' = True, **kwargs: 'Any') -> 'dict[str, pd.DataFrame]'
```

Detect and optionally save events for each stimulation pair in a session.

Returns a mapping from stimulation-pair label to its normalized event table.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `dataloader` | not specified | required |
| `session_id` | str | required |
| `method` | str | `'consensus'` |
| `stim_pairs` | Iterable[str] \| None | `None` |
| `save_events` | bool | `True` |
| `kwargs` | Any | required |

**Returns:** dict[str, pd.DataFrame].

### `reject_artifactual_responses`

```python
reject_artifactual_responses(epochs, report: 'ArtifactResponseReport | None' = None, mode: 'str' = 'drop_epoch', max_bad_channel_fraction: 'float' = 0.25, max_bad_response_fraction: 'float' = 0.25, **detect_kwargs)
```

Remove or mask aberrant trial-channel responses from an Epochs object.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `report` | ArtifactResponseReport \| None | `None` |
| `mode` | str | `'drop_epoch'` |
| `max_bad_channel_fraction` | float | `0.25` |
| `max_bad_response_fraction` | float | `0.25` |
| `detect_kwargs` | not specified | required |

### `reject_bad_channels`

```python
reject_bad_channels(df: 'pd.DataFrame', bad_channels: 'Iterable[str] | str | None' = None, method: 'str' = 'drop', detection_params: 'dict | None' = None, exclude: 'Iterable[str] | None' = None, return_report: 'bool' = False)
```

Reject or repair bad channels.

Parameters
----------
bad_channels:
    Explicit channel list, ``"auto"`` for data-driven detection, or ``None``.
method:
    ``"drop"`` removes bad columns, ``"nan"`` keeps them as NaN, and
    ``"interpolate"`` replaces them with the median good-channel signal.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `bad_channels` | Iterable[str] \| str \| None | `None` |
| `method` | str | `'drop'` |
| `detection_params` | dict \| None | `None` |
| `exclude` | Iterable[str] \| None | `None` |
| `return_report` | bool | `False` |

### `response_artifact_summary`

```python
response_artifact_summary(report: 'ArtifactResponseReport | pd.DataFrame', group_cols: 'Iterable[str] | None' = None, hard_artifact_reasons: 'Iterable[str] | None' = None) -> 'pd.DataFrame'
```

Return an auditable channel-level response-artifact summary.

Parameters
----------
report:
    Artifact response report or its table.
group_cols:
    Optional columns such as ``patient_id``, ``session_id``, or
    ``stim_pair`` to preserve in the grouping.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `report` | ArtifactResponseReport \| pd.DataFrame | required |
| `group_cols` | Iterable[str] \| None | `None` |
| `hard_artifact_reasons` | Iterable[str] \| None | `None` |

**Returns:** pd.DataFrame.

### `resample_continuous`

```python
resample_continuous(df: 'pd.DataFrame', target_sfreq: 'float', *, source_sfreq: 'float | None' = None) -> 'pd.DataFrame'
```

Resample every numeric channel while preserving a time-like index.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `target_sfreq` | float | required |
| `source_sfreq` | float \| None | `None` |

**Returns:** pd.DataFrame.

### `save_continuous_npz`

```python
save_continuous_npz(df: 'pd.DataFrame', path: 'str | Path', *, dtype: 'str | np.dtype' = 'float32', compressed: 'bool' = True, metadata: 'dict[str, Any] | None' = None) -> 'Path'
```

Atomically save continuous numeric channels in ERPy's portable NPZ format.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `df` | pd.DataFrame | required |
| `path` | str \| Path | required |
| `dtype` | str \| np.dtype | `'float32'` |
| `compressed` | bool | `True` |
| `metadata` | dict[str, Any] \| None | `None` |

**Returns:** Path.

### `latency_wavefront_table`

```python
latency_wavefront_table(detections: 'pd.DataFrame', elec_meta: 'pd.DataFrame', *, latency_col: 'str' = 'n1_latency_ms', metric_col: 'str' = 'peak_amplitude_uv', channel_col: 'str' = 'channel', consensus_col: 'str' = 'consensus_ch', consensus_only: 'bool' = True, patient_col: 'str' = 'patient_id', session_col: 'str' = 'session_id', elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', coord_cols: 'tuple[str, str, str]' = ('mni_x', 'mni_y', 'mni_z'), exclude_stimulation_contacts: 'bool' = True) -> 'pd.DataFrame'
```

Merge response latencies with recording-contact anatomy and MNI coordinates.

The returned table is suitable for wavefront plots and
:func:`estimate_latency_gradient`.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame | required |
| `latency_col` | str | `'n1_latency_ms'` |
| `metric_col` | str | `'peak_amplitude_uv'` |
| `channel_col` | str | `'channel'` |
| `consensus_col` | str | `'consensus_ch'` |
| `consensus_only` | bool | `True` |
| `patient_col` | str | `'patient_id'` |
| `session_col` | str | `'session_id'` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `coord_cols` | tuple[str, str, str] | `('mni_x', 'mni_y', 'mni_z')` |
| `exclude_stimulation_contacts` | bool | `True` |

**Returns:** pd.DataFrame.

### `scaffold_region_mask`

```python
scaffold_region_mask(values: 'pd.Series', pattern: 'str' = '\\bACC\\b|anterior[\\s_-]*cingulat|cingul(?:ate)?(?:[\\s_-]*mid)?[\\s_-]*ant\\b|\\bACgG\\b|brodmann[\\s_-]*area[\\s_-]*32|\\bSGC\\b|subgenual|thalam|caudate|putamen|striat|\\bSFG\\b|frontal|pallid|capsule') -> 'pd.Series'
```

Return rows whose anatomical labels belong to the ACC-thalamic-striatal scaffold.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `values` | pd.Series | required |
| `pattern` | str | `'\\bACC\\b\|anterior[\\s_-]*cingulat\|cingul(?:ate)?(?:[\\s_-]*mid)?[\\s_-]*ant\\b\|\\bACgG\\b\|brodmann[\\s_-]*area[\\s_-]*32\|\\bSGC\\b\|subgenual\|thalam\|caudate\|putamen\|striat\|\\bSFG\\b\|frontal\|pallid\|capsule'` |

**Returns:** pd.Series.

### `zscore_metric_within_stim`

```python
zscore_metric_within_stim(data: 'pd.DataFrame', metric: 'str', *, stim_col: 'str' = 'stim_pair', channel_col: 'str | None' = None, absolute: 'bool' = False, output_col: 'str | None' = None) -> 'pd.DataFrame'
```

Z-score a response metric across recording contacts within each stim site.

This is an exploratory ranking transform: for each stimulation site, ERPy
normalizes the selected response metric across recording electrodes so that
unusually high contacts for that stimulation site are easy to inspect.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `data` | pd.DataFrame | required |
| `metric` | str | required |
| `stim_col` | str | `'stim_pair'` |
| `channel_col` | str \| None | `None` |
| `absolute` | bool | `False` |
| `output_col` | str \| None | `None` |

**Returns:** pd.DataFrame.

### `weighted_directed_node_metrics`

```python
weighted_directed_node_metrics(edges: 'pd.DataFrame', *, source_col: 'str' = 'source', target_col: 'str' = 'target', weight_col: 'str' = 'weight', group_cols: 'Iterable[str] | None' = None, min_weight: 'float' = 0.0, agg: 'str' = 'median') -> 'pd.DataFrame'
```

Compute interpretable weighted metrics for directed network nodes.

Edge weights must be nonnegative connection strengths. Shortest-path
measures use their reciprocal as distance, while local reaching centrality
receives the original strength because its definition already treats larger
weights as shorter, stronger connections.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `source_col` | str | `'source'` |
| `target_col` | str | `'target'` |
| `weight_col` | str | `'weight'` |
| `group_cols` | Iterable[str] \| None | `None` |
| `min_weight` | float | `0.0` |
| `agg` | str | `'median'` |

**Returns:** pd.DataFrame.

### Constants

- `CRP_ENERGY_DETECTOR_VERSION` — Public `str` package constant.
- `CHECKPOINT_COMPATIBILITY_ID` — Public `str` package constant.
- `DEFAULT_BANDS` — Public `dict` package constant.
- `DEFAULT_HARD_ARTIFACT_REASONS` — Public `frozenset` package constant.
- `DEFAULT_SCAFFOLD_PATTERN` — Public `str` package constant.
- `DETECTOR_DEFINITIONS` — Public `dict` package constant.
- `DETECTOR_QUANTITY_COLUMNS` — Public `dict` package constant.
- `DETECTOR_REFERENCES` — Public `dict` package constant.
- `PRIMARY_CRITERION` — Public `str` package constant.
- `POST_ARTIFACT_ZERO` — Public `str` package constant.
- `ALL_METHODS` — Public `list` package constant.
- `__version__` — Public `str` package constant.

## Visualization API (`ERPy.viz`)

### Classes

### `EpochsPlotter`

```python
EpochsPlotter(epochs) -> 'None'
```

Waveform plotting accessor available as ``epochs.plot``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |

**Returns:** None.

#### Public members

##### `EpochsPlotter.butterfly`

```python
butterfly(self, channels=None, **kwargs)
```

Overlay trial-mean waveforms from selected channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

##### `EpochsPlotter.detectors`

```python
detectors(self, channel: 'str', detections: 'pd.DataFrame | None' = None, **kwargs)
```

Plot source-native detector quantities for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `detections` | pd.DataFrame \| None | `None` |
| `kwargs` | not specified | required |

##### `EpochsPlotter.grid`

```python
grid(self, channels=None, **kwargs)
```

Plot trial-mean waveforms for several channels in a grid.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

##### `EpochsPlotter.grouped_grid`

```python
grouped_grid(self, channel_metadata: 'pd.DataFrame', group_col: 'str', **kwargs)
```

Plot channel waveforms grouped by a metadata column.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel_metadata` | pd.DataFrame | required |
| `group_col` | str | required |
| `kwargs` | not specified | required |

##### `EpochsPlotter.heatmap`

```python
heatmap(self, channel: 'str', **kwargs)
```

Plot trial-by-time amplitudes for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `kwargs` | not specified | required |

##### `EpochsPlotter.mean`

```python
mean(self, channel: 'str', **kwargs)
```

Plot a channel's trial mean and standard error.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `kwargs` | not specified | required |

##### `EpochsPlotter.overlay`

```python
overlay(self, channel: 'str', **kwargs)
```

Overlay individual trials and the mean for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `kwargs` | not specified | required |

##### `EpochsPlotter.ranked_grid`

```python
ranked_grid(self, detections: 'pd.DataFrame', **kwargs)
```

Plot channel waveforms ordered by a detection metric.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `EpochsPlotter.summary`

```python
summary(self, channel: 'str', detections: 'pd.DataFrame | None' = None, **kwargs)
```

Create a four-panel waveform and detection summary.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `detections` | pd.DataFrame \| None | `None` |
| `kwargs` | not specified | required |

##### `EpochsPlotter.trials_grid`

```python
trials_grid(self, channel: 'str', **kwargs)
```

Plot each trial from one channel in its own panel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `kwargs` | not specified | required |

### `SpectralPlotter`

```python
SpectralPlotter(epochs) -> 'None'
```

Plotting accessor available as ``epochs.spectral.plot``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |

**Returns:** None.

#### Public members

##### `SpectralPlotter.comodulogram`

```python
comodulogram(self, phase_channel: 'str', amp_channel: 'str | None' = None, **kwargs)
```

Compute and plot phase-amplitude coupling across frequency pairs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `kwargs` | not specified | required |

##### `SpectralPlotter.connectivity`

```python
connectivity(self, matrix: 'pd.DataFrame', **kwargs)
```

Plot a square channel-connectivity matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `kwargs` | not specified | required |

##### `SpectralPlotter.ersp`

```python
ersp(self, channel: 'str', config: 'SpectralConfig | None' = None, **kwargs)
```

Compute and plot event-related spectral perturbation.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `kwargs` | not specified | required |

##### `SpectralPlotter.itpc`

```python
itpc(self, channel: 'str', config: 'SpectralConfig | None' = None, **kwargs)
```

Compute and plot inter-trial phase coherence.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `kwargs` | not specified | required |

##### `SpectralPlotter.pac`

```python
pac(self, phase_channel: 'str', amp_channel: 'str | None' = None, **kwargs)
```

Compute and plot amplitude as a function of oscillatory phase.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `phase_channel` | str | required |
| `amp_channel` | str \| None | `None` |
| `kwargs` | not specified | required |

##### `SpectralPlotter.plv`

```python
plv(self, ch_x: 'str', ch_y: 'str', config: 'SpectralConfig | None' = None, **kwargs)
```

Compute and plot time-frequency phase locking for two channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `ch_x` | str | required |
| `ch_y` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `kwargs` | not specified | required |

##### `SpectralPlotter.psd`

```python
psd(self, channels=None, **kwargs)
```

Compute and plot power spectral density for selected channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channels` | not specified | `None` |
| `kwargs` | not specified | required |

##### `SpectralPlotter.summary`

```python
summary(self, channel: 'str', config: 'SpectralConfig | None' = None)
```

Create a multi-panel spectral summary for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |

##### `SpectralPlotter.tfr`

```python
tfr(self, channel: 'str', config: 'SpectralConfig | None' = None, **kwargs)
```

Compute and plot a time-frequency representation for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |
| `kwargs` | not specified | required |

### Functions

### `choose_significant_and_nonsignificant_channels`

```python
choose_significant_and_nonsignificant_channels(detections: 'pd.DataFrame', channels: 'list[str]') -> 'tuple[str, str]'
```

Pick one strong significant channel and one low-scoring non-significant channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `channels` | list[str] | required |

**Returns:** tuple[str, str].

### `method_significance_matrix`

```python
method_significance_matrix(detections: 'pd.DataFrame') -> 'pd.DataFrame'
```

Return channel x method boolean significance matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |

**Returns:** pd.DataFrame.

### `plot_crp_curve`

```python
plot_crp_curve(result: 'CRPResult', ax=None, color: 'str' = '#0f766e')
```

Plot the canonical response curve learned from the response window.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | CRPResult | required |
| `ax` | not specified | `None` |
| `color` | str | `'#0f766e'` |

### `plot_crp_projections`

```python
plot_crp_projections(result: 'CRPResult', ax=None, color: 'str' = '#2563eb')
```

Plot trial projections onto the canonical response profile.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | CRPResult | required |
| `ax` | not specified | `None` |
| `color` | str | `'#2563eb'` |

### `plot_crp_response_contrast`

```python
plot_crp_response_contrast(epochs, significant_channel: 'str', nonsignificant_channel: 'str', config: 'CRPConfig | None' = None)
```

Contrast significant and non-significant channels with mean response, CRP curves, and CRP weights.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `significant_channel` | str | required |
| `nonsignificant_channel` | str | required |
| `config` | CRPConfig \| None | `None` |

### `plot_crp_score_map`

```python
plot_crp_score_map(crp_table: 'pd.DataFrame', ax=None, top_n: 'int' = 20)
```

Rank channels by CRP score with significance-aware coloring.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `crp_table` | pd.DataFrame | required |
| `ax` | not specified | `None` |
| `top_n` | int | `20` |

### `plot_crp_site_comparison`

```python
plot_crp_site_comparison(comparison_table: 'pd.DataFrame', metric: 'str' = 'within_stim_z', ax=None, top_n_channels: 'int' = 24, title: 'str | None' = None)
```

Plot a stimulation-site by recording-channel CRP comparison heatmap.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `comparison_table` | pd.DataFrame | required |
| `metric` | str | `'within_stim_z'` |
| `ax` | not specified | `None` |
| `top_n_channels` | int | `24` |
| `title` | str \| None | `None` |

### `plot_crp_summary`

```python
plot_crp_summary(epochs, channel: 'str', config: 'CRPConfig | None' = None, result: 'CRPResult | None' = None)
```

Summary CRP panel for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | CRPConfig \| None | `None` |
| `result` | CRPResult \| None | `None` |

### `plot_crp_weight_timecourse`

```python
plot_crp_weight_timecourse(result: 'CRPResult', ax=None, color: 'str' = '#7c3aed', zscore: 'bool' = False)
```

Plot canonical response profile expression over the full epoch.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | CRPResult | required |
| `ax` | not specified | `None` |
| `color` | str | `'#7c3aed'` |
| `zscore` | bool | `False` |

### `plot_detection_method_matrix`

```python
plot_detection_method_matrix(detections: 'pd.DataFrame', ax=None, top_n: 'int' = 32)
```

Plot implemented significance methods against channels.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `ax` | not specified | `None` |
| `top_n` | int | `32` |

### `plot_detection_summary`

```python
plot_detection_summary(detections: 'pd.DataFrame', top_n: 'int' = 32)
```

Combined method matrix and per-method count figure.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `top_n` | int | `32` |

### `plot_method_significance_counts`

```python
plot_method_significance_counts(detections: 'pd.DataFrame', ax=None)
```

Bar plot of significant channel counts per method.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `ax` | not specified | `None` |

### `plot_published_detector_diagnostic`

```python
plot_published_detector_diagnostic(epochs, channel: 'str', *, detections: 'pd.DataFrame | None' = None, wideband_epochs=None, baseline_window: 'tuple[float, float] | None' = None, response_window: 'tuple[float, float]' = (0.01, 0.3), signi_n_permutations: 'int' = 1000)
```

Visualize the source-native quantities behind one response decision.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `detections` | pd.DataFrame \| None | `None` |
| `wideband_epochs` | not specified | `None` |
| `baseline_window` | tuple[float, float] \| None | `None` |
| `response_window` | tuple[float, float] | `(0.01, 0.3)` |
| `signi_n_permutations` | int | `1000` |

### `plot_butterfly`

```python
plot_butterfly(epochs, channels: 'Iterable[str] | None' = None, ax=None)
```

Overlay trial-mean waveforms from multiple channels on one axis.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channels` | Iterable[str] \| None | `None` |
| `ax` | not specified | `None` |

### `plot_grid`

```python
plot_grid(epochs, channels: 'Iterable[str] | None' = None, ncols: 'int' = 4)
```

Plot mean waveforms for several channels in a shared grid.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channels` | Iterable[str] \| None | `None` |
| `ncols` | int | `4` |

### `plot_heatmap`

```python
plot_heatmap(epochs, channel: 'str', ax=None, cmap='RdBu_r')
```

Plot trial-by-time amplitudes for one channel as a heat map.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `ax` | not specified | `None` |
| `cmap` | not specified | `'RdBu_r'` |

### `plot_grouped_grid`

```python
plot_grouped_grid(epochs, channel_metadata: 'pd.DataFrame', group_col: 'str', channel_col: 'str' = 'channel', channels: 'Iterable[str] | None' = None, ncols: 'int' = 4)
```

Plot mean waveforms grouped by anatomy, lead, method label, or any metadata column.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel_metadata` | pd.DataFrame | required |
| `group_col` | str | required |
| `channel_col` | str | `'channel'` |
| `channels` | Iterable[str] \| None | `None` |
| `ncols` | int | `4` |

### `plot_mean`

```python
plot_mean(epochs, channel: 'str', ax=None, color='#2563eb', label: 'str | None' = None)
```

Plot a channel's trial mean with a shaded standard-error band.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `ax` | not specified | `None` |
| `color` | not specified | `'#2563eb'` |
| `label` | str \| None | `None` |

### `plot_mean_erp_comparison`

```python
plot_mean_erp_comparison(epoch_map: 'dict[str, object]', channel: 'str', ax=None, colors: 'Iterable[str] | None' = None)
```

Overlay the same channel from multiple Epochs objects.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epoch_map` | dict[str, object] | required |
| `channel` | str | required |
| `ax` | not specified | `None` |
| `colors` | Iterable[str] \| None | `None` |

### `plot_overlay`

```python
plot_overlay(epochs, channel: 'str', ax=None, alpha: 'float' = 0.22)
```

Overlay individual trials and the mean response for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `ax` | not specified | `None` |
| `alpha` | float | `0.22` |

### `plot_ranked_grid`

```python
plot_ranked_grid(epochs, detections: 'pd.DataFrame', metric: 'str' = 'peak_amplitude_uv', top_n: 'int' = 16, ncols: 'int' = 4, consensus_only: 'bool' = True)
```

Small-multiple mean waveforms ordered by a detection metric.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `detections` | pd.DataFrame | required |
| `metric` | str | `'peak_amplitude_uv'` |
| `top_n` | int | `16` |
| `ncols` | int | `4` |
| `consensus_only` | bool | `True` |

### `plot_response_map`

```python
plot_response_map(detections: 'pd.DataFrame', metric: 'str' = 'peak_amplitude_uv', ax=None, top_n: 'int' = 20)
```

Plot the highest-ranked channel values from a detection metric.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `metric` | str | `'peak_amplitude_uv'` |
| `ax` | not specified | `None` |
| `top_n` | int | `20` |

### `plot_summary_panel`

```python
plot_summary_panel(epochs, channel: 'str', detections: 'pd.DataFrame | None' = None)
```

Four-panel response summary: mean +/- SEM, trial overlay, trial heatmap, and a text summary.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `detections` | pd.DataFrame \| None | `None` |

### `plot_trials_grid`

```python
plot_trials_grid(epochs, channel: 'str', ncols: 'int' = 5)
```

Plot each trial from one channel in a separate small panel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `ncols` | int | `5` |

### `plot_within_stim_zscore_bars`

```python
plot_within_stim_zscore_bars(data: 'pd.DataFrame', metric: 'str', *, stim_pair: 'str | None' = None, stim_col: 'str' = 'stim_pair', channel_col: 'str | None' = None, z_col: 'str | None' = None, absolute: 'bool' = False, top_n: 'int' = 18, ax=None, title: 'str | None' = None)
```

Rank recording contacts by within-stimulation-site z-scored response.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `data` | pd.DataFrame | required |
| `metric` | str | required |
| `stim_pair` | str \| None | `None` |
| `stim_col` | str | `'stim_pair'` |
| `channel_col` | str \| None | `None` |
| `z_col` | str \| None | `None` |
| `absolute` | bool | `False` |
| `top_n` | int | `18` |
| `ax` | not specified | `None` |
| `title` | str \| None | `None` |

### `plot_within_stim_zscore_heatmap`

```python
plot_within_stim_zscore_heatmap(data: 'pd.DataFrame', metric: 'str', *, stim_col: 'str' = 'stim_pair', channel_col: 'str | None' = None, z_col: 'str | None' = None, absolute: 'bool' = False, top_n_channels: 'int' = 28, ax=None, title: 'str | None' = None)
```

Heatmap of a metric z-scored across contacts within each stimulation site.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `data` | pd.DataFrame | required |
| `metric` | str | required |
| `stim_col` | str | `'stim_pair'` |
| `channel_col` | str \| None | `None` |
| `z_col` | str \| None | `None` |
| `absolute` | bool | `False` |
| `top_n_channels` | int | `28` |
| `ax` | not specified | `None` |
| `title` | str \| None | `None` |

### `plot_comodulogram`

```python
plot_comodulogram(result: 'ComodulogramResult', ax=None, cmap: 'str' = 'inferno')
```

Plot a PAC comodulogram (amplitude frequency x phase frequency).

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | ComodulogramResult | required |
| `ax` | not specified | `None` |
| `cmap` | str | `'inferno'` |

### `plot_connectivity_matrix`

```python
plot_connectivity_matrix(matrix: 'pd.DataFrame', ax=None, *, cmap: 'str | None' = None, vmin: 'float | None' = None, vmax: 'float | None' = None, title: 'str | None' = None, label: 'str | None' = None)
```

Plot a channel x channel connectivity matrix (PLV, coherence, etc.).

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `ax` | not specified | `None` |
| `cmap` | str \| None | `None` |
| `vmin` | float \| None | `None` |
| `vmax` | float \| None | `None` |
| `title` | str \| None | `None` |
| `label` | str \| None | `None` |

### `plot_ersp`

```python
plot_ersp(result: 'TFRResult', ax=None, **kwargs)
```

Plot baseline-normalized event-related spectral perturbation.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | TFRResult | required |
| `ax` | not specified | `None` |
| `kwargs` | not specified | required |

### `plot_itpc`

```python
plot_itpc(result: 'TFRResult', ax=None, **kwargs)
```

Plot inter-trial phase coherence as a time-frequency map.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | TFRResult | required |
| `ax` | not specified | `None` |
| `kwargs` | not specified | required |

### `plot_pac`

```python
plot_pac(result: 'PACResult', ax=None, color: 'str' = '#7c3aed')
```

Plot the amplitude-by-phase distribution for a single PAC estimate.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | PACResult | required |
| `ax` | not specified | `None` |
| `color` | str | `'#7c3aed'` |

### `plot_phase_phase_matrix`

```python
plot_phase_phase_matrix(matrix: 'pd.DataFrame', ax=None, cmap: 'str' = 'magma')
```

Plot an n:m phase-phase coupling matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `ax` | not specified | `None` |
| `cmap` | str | `'magma'` |

### `plot_plv_timefreq`

```python
plot_plv_timefreq(result: 'PLVResult', ax=None, cmap: 'str' = 'magma')
```

Time-frequency phase-locking value heatmap between two contacts.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | PLVResult | required |
| `ax` | not specified | `None` |
| `cmap` | str | `'magma'` |

### `plot_psd`

```python
plot_psd(psd: 'pd.DataFrame', channels: 'Iterable[str] | None' = None, ax=None, *, logx: 'bool' = True, logy: 'bool' = True, top_n: 'int | None' = 8)
```

Plot Welch power spectra (one line per channel).

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `psd` | pd.DataFrame | required |
| `channels` | Iterable[str] \| None | `None` |
| `ax` | not specified | `None` |
| `logx` | bool | `True` |
| `logy` | bool | `True` |
| `top_n` | int \| None | `8` |

### `plot_spectral_summary`

```python
plot_spectral_summary(epochs, channel: 'str', config: 'SpectralConfig | None' = None)
```

Multi-panel spectral summary for one channel: mean waveform, PSD, ERSP, ITPC.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `channel` | str | required |
| `config` | SpectralConfig \| None | `None` |

### `plot_tfr`

```python
plot_tfr(result: 'TFRResult', ax=None, *, kind: 'str' = 'power', cmap: 'str | None' = None, vmax: 'float | None' = None)
```

Plot a time-frequency map (``kind`` = ``power``/``ersp`` or ``itc``).

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `result` | TFRResult | required |
| `ax` | not specified | `None` |
| `kind` | str | `'power'` |
| `cmap` | str \| None | `None` |
| `vmax` | float \| None | `None` |

### `plot_graph_metric_heatmap`

```python
plot_graph_metric_heatmap(metric_table: 'pd.DataFrame', *, metric: 'str' = 'hub', top_n_nodes: 'int' = 24, ax=None, title: 'str | None' = None)
```

Plot node graph metrics as a node x time heatmap.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `metric_table` | pd.DataFrame | required |
| `metric` | str | `'hub'` |
| `top_n_nodes` | int | `24` |
| `ax` | not specified | `None` |
| `title` | str \| None | `None` |

### `plot_graph_metric_timecourse`

```python
plot_graph_metric_timecourse(metric_table: 'pd.DataFrame', *, node: 'str', metric: 'str' = 'hub', ax=None, color: 'str' = '#2563eb', title: 'str | None' = None)
```

Plot how one node-level graph metric changes through the stim epoch.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `metric_table` | pd.DataFrame | required |
| `node` | str | required |
| `metric` | str | `'hub'` |
| `ax` | not specified | `None` |
| `color` | str | `'#2563eb'` |
| `title` | str \| None | `None` |

### `quick_erp`

```python
quick_erp(epochs_df: 'pd.DataFrame', channel: 'str', ax=None, mode: 'str' = 'mean_sem', color: 'Optional[str]' = None, label: 'Optional[str]' = None, title: 'Optional[str]' = None)
```

Plot one channel from an ``(epoch, time)`` indexed epochs DataFrame.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs_df` | pd.DataFrame | required |
| `channel` | str | required |
| `ax` | not specified | `None` |
| `mode` | str | `'mean_sem'` |
| `color` | Optional[str] | `None` |
| `label` | Optional[str] | `None` |
| `title` | Optional[str] | `None` |

### `plot_erp_grid`

```python
plot_erp_grid(epochs_df: 'pd.DataFrame', channels: 'Optional[Sequence[str]]' = None, ncols: 'int' = 4, mode: 'str' = 'mean_sem', figsize_per: 'tuple[float, float]' = (3.2, 2.4), suptitle: 'Optional[str]' = None)
```

Faceted grid of mean ERPs, one subplot per channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs_df` | pd.DataFrame | required |
| `channels` | Optional[Sequence[str]] | `None` |
| `ncols` | int | `4` |
| `mode` | str | `'mean_sem'` |
| `figsize_per` | tuple[float, float] | `(3.2, 2.4)` |
| `suptitle` | Optional[str] | `None` |

### `plot_mean_erp_dataframe_comparison`

```python
plot_mean_erp_dataframe_comparison(series: 'Iterable[tuple[str, pd.DataFrame]]', channel: 'str', ax=None, colors: 'Optional[Sequence[str]]' = None)
```

Overlay mean +/- SEM for several epoched DataFrames.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `series` | Iterable[tuple[str, pd.DataFrame]] | required |
| `channel` | str | required |
| `ax` | not specified | `None` |
| `colors` | Optional[Sequence[str]] | `None` |

### `heatmap_channels_time`

```python
heatmap_channels_time(epochs_df: 'pd.DataFrame', channels: 'Optional[list[str]]' = None, ax=None, cmap: 'str' = 'viridis')
```

Plot a time-by-channel heatmap after averaging over epochs.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs_df` | pd.DataFrame | required |
| `channels` | Optional[list[str]] | `None` |
| `ax` | not specified | `None` |
| `cmap` | str | `'viridis'` |

### `heatmap_epochs`

```python
heatmap_epochs(epochs_df: 'pd.DataFrame', channel: 'str', ax=None, cmap: 'str' = 'RdBu_r', vmin: 'Optional[float]' = None, vmax: 'Optional[float]' = None, cbar_label: 'str' = 'Amplitude')
```

Plot a time-by-epoch heatmap for one channel.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs_df` | pd.DataFrame | required |
| `channel` | str | required |
| `ax` | not specified | `None` |
| `cmap` | str | `'RdBu_r'` |
| `vmin` | Optional[float] | `None` |
| `vmax` | Optional[float] | `None` |
| `cbar_label` | str | `'Amplitude'` |

### `plot_metric_vs_epoch`

```python
plot_metric_vs_epoch(epochs_df: 'pd.DataFrame', channel: 'str', metric_fn: 'Optional[Callable[[np.ndarray], float]]' = None, ax=None, rolling: 'int' = 0, title: 'Optional[str]' = None)
```

Plot a per-epoch scalar metric in stimulation order.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs_df` | pd.DataFrame | required |
| `channel` | str | required |
| `metric_fn` | Optional[Callable[[np.ndarray], float]] | `None` |
| `ax` | not specified | `None` |
| `rolling` | int | `0` |
| `title` | Optional[str] | `None` |

### `plot_electrode_mni`

```python
plot_electrode_mni(coords: 'np.ndarray', values: 'Optional[np.ndarray]' = None, node_size: 'Union[float, np.ndarray]' = 50.0, title: 'Optional[str]' = None, display_mode: 'str' = 'ortho', **kwargs)
```

Plot electrode locations in MNI space using ``nilearn.plot_connectome``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `coords` | np.ndarray | required |
| `values` | Optional[np.ndarray] | `None` |
| `node_size` | Union[float, np.ndarray] | `50.0` |
| `title` | Optional[str] | `None` |
| `display_mode` | str | `'ortho'` |
| `kwargs` | not specified | required |

### `plot_electrodes_from_metadata`

```python
plot_electrodes_from_metadata(elec_df: 'pd.DataFrame', value_col: 'Optional[str]' = None, mni_cols: 'tuple[str, str, str]' = ('mni_x', 'mni_y', 'mni_z'), **kwargs)
```

Plot electrodes using coordinate columns from electrode metadata.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `elec_df` | pd.DataFrame | required |
| `value_col` | Optional[str] | `None` |
| `mni_cols` | tuple[str, str, str] | `('mni_x', 'mni_y', 'mni_z')` |
| `kwargs` | not specified | required |

### `plot_erps_on_brain`

```python
plot_erps_on_brain(elec_df: 'pd.DataFrame', metric: 'pd.Series', elec_label_col: 'str' = 'elec_label', mni_cols: 'tuple[str, str, str]' = ('mni_x', 'mni_y', 'mni_z'), **kwargs)
```

Color electrodes by a per-contact scalar metric.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `elec_df` | pd.DataFrame | required |
| `metric` | pd.Series | required |
| `elec_label_col` | str | `'elec_label'` |
| `mni_cols` | tuple[str, str, str] | `('mni_x', 'mni_y', 'mni_z')` |
| `kwargs` | not specified | required |

### `save_analysis_figure`

```python
save_analysis_figure(fig: 'Union[matplotlib.figure.Figure, matplotlib.axes.Axes]', name: 'str', patient_path: 'Optional[str]' = None, subdir: 'str' = '', formats: 'Iterable[str]' = ('png', 'pdf')) -> 'list[str]'
```

Write a figure to ``{patient_path}/analysis/{subdir}/{name}.{fmt}``.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `fig` | Union[matplotlib.figure.Figure, matplotlib.axes.Axes] | required |
| `name` | str | required |
| `patient_path` | Optional[str] | `None` |
| `subdir` | str | `''` |
| `formats` | Iterable[str] | `('png', 'pdf')` |

**Returns:** list[str].

### `adjacency_from_edges`

```python
adjacency_from_edges(edges: 'pd.DataFrame', source_col: 'str' = 'stim_elec', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', agg: 'str' = 'mean') -> 'pd.DataFrame'
```

Pivot a directed edge table into a square electrode adjacency matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `source_col` | str | `'stim_elec'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `agg` | str | `'mean'` |

**Returns:** pd.DataFrame.

### `electrode_label_identity`

```python
electrode_label_identity(label: 'str') -> 'str'
```

Return a reference- and zero-padding-invariant contact identity.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `label` | str | required |

**Returns:** str.

### `edges_from_detections`

```python
edges_from_detections(detections: 'pd.DataFrame', stim_pair: 'str | None' = None, elec_meta: 'pd.DataFrame | None' = None, metric: 'str' = 'peak_amplitude_uv', channel_col: 'str' = 'channel', consensus_only: 'bool' = True, significance_column: 'str | None' = None, qc_only: 'bool' = True, exclude_stimulation_contacts: 'bool' = True, identity_cols: 'Iterable[str] | None' = None) -> 'pd.DataFrame'
```

Convert detection rows into a tidy directed edge table.

The result is intentionally plain pandas so users can pass it to NetworkX,
Nilearn, Plotly, BCT-style graph packages, or their own statistics code.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `detections` | pd.DataFrame | required |
| `stim_pair` | str \| None | `None` |
| `elec_meta` | pd.DataFrame \| None | `None` |
| `metric` | str | `'peak_amplitude_uv'` |
| `channel_col` | str | `'channel'` |
| `consensus_only` | bool | `True` |
| `significance_column` | str \| None | `None` |
| `qc_only` | bool | `True` |
| `exclude_stimulation_contacts` | bool | `True` |
| `identity_cols` | Iterable[str] \| None | `None` |

**Returns:** pd.DataFrame.

### `graph_from_edges`

```python
graph_from_edges(edges: 'pd.DataFrame', source_col: 'str' = 'stim_elec', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', directed: 'bool' = True, threshold: 'float' = 0.0) -> 'nx.Graph'
```

Convert a response-edge table to a weighted directed or undirected graph.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `source_col` | str | `'stim_elec'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `directed` | bool | `True` |
| `threshold` | float | `0.0` |

**Returns:** nx.Graph.

### `graph_from_matrix`

```python
graph_from_matrix(matrix: 'pd.DataFrame', threshold: 'float' = 0.0, directed: 'bool' = True) -> 'nx.Graph'
```

Convert finite supra-threshold matrix entries to weighted graph edges.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `threshold` | float | `0.0` |
| `directed` | bool | `True` |

**Returns:** nx.Graph.

### `node_coordinates_from_metadata`

```python
node_coordinates_from_metadata(elec_meta: 'pd.DataFrame', nodes: 'list[str] | None' = None, elec_col: 'str' = 'elec_label', coord_cols=('mni_x', 'mni_y', 'mni_z'), region_col: 'str' = 'anat_label') -> 'pd.DataFrame'
```

Return electrode coordinates in a predictable plotting schema.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `elec_meta` | pd.DataFrame | required |
| `nodes` | list[str] \| None | `None` |
| `elec_col` | str | `'elec_label'` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `region_col` | str | `'anat_label'` |

**Returns:** pd.DataFrame.

### `ordered_adjacency_from_edges`

```python
ordered_adjacency_from_edges(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame | None' = None, source_col: 'str' = 'stim_elec', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', agg: 'str' = 'mean') -> 'pd.DataFrame'
```

Build an adjacency matrix ordered by electrode metadata and label.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame \| None | `None` |
| `source_col` | str | `'stim_elec'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `agg` | str | `'mean'` |

**Returns:** pd.DataFrame.

### `order_nodes_by_metadata`

```python
order_nodes_by_metadata(nodes: 'list[str]', elec_meta: 'pd.DataFrame | None' = None, elec_col: 'str' = 'elec_label', group_cols=('hemisphere', 'group'), coord_cols=('mni_x', 'mni_y', 'mni_z')) -> 'list[str]'
```

Order electrodes by shaft metadata, natural contact label, then coordinates.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `nodes` | list[str] | required |
| `elec_meta` | pd.DataFrame \| None | `None` |
| `elec_col` | str | `'elec_label'` |
| `group_cols` | not specified | `('hemisphere', 'group')` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |

**Returns:** list[str].

### `order_sources_by_metadata`

```python
order_sources_by_metadata(sources: 'list[str]', elec_meta: 'pd.DataFrame | None' = None) -> 'list[str]'
```

Order stimulation sources by their first contact, then by pair label.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `sources` | list[str] | required |
| `elec_meta` | pd.DataFrame \| None | `None` |

**Returns:** list[str].

### `plot_adjacency_heatmap`

```python
plot_adjacency_heatmap(matrix: 'pd.DataFrame', ax=None, cmap: 'str' = 'magma', title: 'str' = 'Connectivity matrix')
```

Plot a labeled stimulation-by-recording adjacency heat map.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `matrix` | pd.DataFrame | required |
| `ax` | not specified | `None` |
| `cmap` | str | `'magma'` |
| `title` | str | `'Connectivity matrix'` |

### `plot_coordinate_network`

```python
plot_coordinate_network(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame', source_col: 'str' = 'stim_elec', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', coord_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', projection: 'tuple[str, str]' = ('mni_x', 'mni_y'), top_n: 'int | None' = 60, ax=None, title: 'str' = 'Coordinate response network', cmap: 'str' = 'plasma')
```

Plot a data-aware 2D network using electrode coordinates instead of a spring layout.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame | required |
| `source_col` | str | `'stim_elec'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `projection` | tuple[str, str] | `('mni_x', 'mni_y')` |
| `top_n` | int \| None | `60` |
| `ax` | not specified | `None` |
| `title` | str | `'Coordinate response network'` |
| `cmap` | str | `'plasma'` |

### `plot_aggregate_evoked_response_graph`

```python
plot_aggregate_evoked_response_graph(runs: 'list[dict]', elec_meta: 'pd.DataFrame', metric: 'str' = 'peak_amplitude_uv', response_metric: 'str' = 'voltage', response_metric_frame_key: 'str' = 'response_metric_frame', metric_label: 'str | None' = None, consensus_only: 'bool' = True, response_window: 'tuple[float, float]' = (0.0, 0.35), frame_step_ms: 'float' = 12.0, top_n_per_stim: 'int' = 18, top_n_total: 'int | None' = 90, coord_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', title: 'str' = 'Aggregate evoked response graph over time', output_html: 'str | Path | None' = None, cmap: 'str' = 'plasma', show_brain_shell: 'bool' = True, mesh: 'str' = 'fsaverage5', surface: 'str' = 'pial', fallback_shell: 'bool' = True, validate_mni: 'bool' = True, mni_mask_margin_mm: 'float' = 4.0, outside_mni: 'str' = 'drop', significance_time_gate: 'float' = 0.2)
```

Animate evoked responses from multiple stimulation sites on one brain.

Each run dictionary should contain ``epochs`` and ``stim_pair``. Optional
keys include ``detections``, ``channels``, and ``response_metric_frame``.
By default, edges use absolute mean evoked voltage in uV. To animate another
response metric such as CRP canonical weight or z-scored CRP expression,
pass a time-indexed DataFrame in ``response_metric_frame`` with recording
contacts as columns and set ``metric_label`` accordingly.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `runs` | list[dict] | required |
| `elec_meta` | pd.DataFrame | required |
| `metric` | str | `'peak_amplitude_uv'` |
| `response_metric` | str | `'voltage'` |
| `response_metric_frame_key` | str | `'response_metric_frame'` |
| `metric_label` | str \| None | `None` |
| `consensus_only` | bool | `True` |
| `response_window` | tuple[float, float] | `(0.0, 0.35)` |
| `frame_step_ms` | float | `12.0` |
| `top_n_per_stim` | int | `18` |
| `top_n_total` | int \| None | `90` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `title` | str | `'Aggregate evoked response graph over time'` |
| `output_html` | str \| Path \| None | `None` |
| `cmap` | str | `'plasma'` |
| `show_brain_shell` | bool | `True` |
| `mesh` | str | `'fsaverage5'` |
| `surface` | str | `'pial'` |
| `fallback_shell` | bool | `True` |
| `validate_mni` | bool | `True` |
| `mni_mask_margin_mm` | float | `4.0` |
| `outside_mni` | str | `'drop'` |
| `significance_time_gate` | float | `0.2` |

### `plot_electrode_connectome`

```python
plot_electrode_connectome(adjacency: 'pd.DataFrame', elec_meta: 'pd.DataFrame', mni_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', metric_label: 'str | None' = None, title: 'str' = 'Electrode connectome', edge_cmap: 'str' = 'plasma', display_mode: 'str' = 'lyrz')
```

Plot an electrode adjacency matrix at registered MNI coordinates.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `adjacency` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame | required |
| `mni_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `metric_label` | str \| None | `None` |
| `title` | str | `'Electrode connectome'` |
| `edge_cmap` | str | `'plasma'` |
| `display_mode` | str | `'lyrz'` |

### `plot_evoked_response_graph`

```python
plot_evoked_response_graph(epochs, elec_meta: 'pd.DataFrame', stim_pair: 'str | None' = None, stim_elec: 'str | None' = None, detections: 'pd.DataFrame | None' = None, channels: 'list[str] | None' = None, metric: 'str' = 'peak_amplitude_uv', consensus_only: 'bool' = True, response_window: 'tuple[float, float]' = (0.0, 0.35), frame_step_ms: 'float' = 8.0, top_n: 'int' = 24, coord_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', title: 'str' = 'Evoked response graph over time', output_html: 'str | Path | None' = None, cmap: 'str' = 'RdBu_r', show_brain_shell: 'bool' = True, mesh: 'str' = 'fsaverage5', surface: 'str' = 'pial', fallback_shell: 'bool' = True, validate_mni: 'bool' = True, mni_mask_margin_mm: 'float' = 4.0, outside_mni: 'str' = 'drop')
```

Animate mean evoked response amplitudes on a template cortical surface.

Node color and size follow the mean response waveform at each post-stimulus
time point. Edges run from the stimulation contact to recording contacts and
brighten as response magnitude increases. Static fsaverage cortical meshes
are kept outside the animation frames so the HTML remains tractable.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `epochs` | not specified | required |
| `elec_meta` | pd.DataFrame | required |
| `stim_pair` | str \| None | `None` |
| `stim_elec` | str \| None | `None` |
| `detections` | pd.DataFrame \| None | `None` |
| `channels` | list[str] \| None | `None` |
| `metric` | str | `'peak_amplitude_uv'` |
| `consensus_only` | bool | `True` |
| `response_window` | tuple[float, float] | `(0.0, 0.35)` |
| `frame_step_ms` | float | `8.0` |
| `top_n` | int | `24` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `title` | str | `'Evoked response graph over time'` |
| `output_html` | str \| Path \| None | `None` |
| `cmap` | str | `'RdBu_r'` |
| `show_brain_shell` | bool | `True` |
| `mesh` | str | `'fsaverage5'` |
| `surface` | str | `'pial'` |
| `fallback_shell` | bool | `True` |
| `validate_mni` | bool | `True` |
| `mni_mask_margin_mm` | float | `4.0` |
| `outside_mni` | str | `'drop'` |

### `plot_glass_brain_network`

```python
plot_glass_brain_network(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame', source_col: 'str' = 'stim_elec', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', mni_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', title: 'str | None' = 'Glass brain response network', edge_cmap: 'str' = 'plasma', display_mode: 'str' = 'lyrz', edge_linewidth: 'float' = 1.45, edge_alpha: 'float' = 0.82, edge_threshold: 'str | float | None' = None, node_size: 'float' = 42.0, directed: 'bool' = False, brain_alpha: 'float' = 0.72, validate_mni: 'bool' = True, mni_mask_margin_mm: 'float' = 4.0, outside_mni: 'str' = 'drop')
```

Plot an edge table on Nilearn's MNI glass-brain display.

The default is a symmetric edge display because Nilearn's directed arrows
become visually heavy in dense intracranial stimulation networks. Source
electrodes remain highlighted, and ``directed=True`` is available when
arrow geometry is explicitly desired.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame | required |
| `source_col` | str | `'stim_elec'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `mni_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `title` | str \| None | `'Glass brain response network'` |
| `edge_cmap` | str | `'plasma'` |
| `display_mode` | str | `'lyrz'` |
| `edge_linewidth` | float | `1.45` |
| `edge_alpha` | float | `0.82` |
| `edge_threshold` | str \| float \| None | `None` |
| `node_size` | float | `42.0` |
| `directed` | bool | `False` |
| `brain_alpha` | float | `0.72` |
| `validate_mni` | bool | `True` |
| `mni_mask_margin_mm` | float | `4.0` |
| `outside_mni` | str | `'drop'` |

### `plot_interactive_connectome`

```python
plot_interactive_connectome(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame', source_col: 'str' = 'stim_elec', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', coord_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', top_n: 'int | None' = 80, title: 'str' = 'ERPy stimulation-response connectome', output_html: 'str | Path | None' = None, cmap: 'str' = 'plasma', show_brain_shell: 'bool' = True, mesh: 'str' = 'fsaverage5', surface: 'str' = 'pial', fallback_shell: 'bool' = True, validate_mni: 'bool' = True, mni_mask_margin_mm: 'float' = 4.0, outside_mni: 'str' = 'drop')
```

Interactive 3D connectome on a template cortical surface.

This is the most portable brain-network view: it needs electrode coordinates
but not a subject MRI, FreeSurfer reconstruction, or graphical backend.
When Nilearn is installed, the brain is rendered with fsaverage cortical
meshes; otherwise ERPy falls back to a lightweight translucent shell.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame | required |
| `source_col` | str | `'stim_elec'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `top_n` | int \| None | `80` |
| `title` | str | `'ERPy stimulation-response connectome'` |
| `output_html` | str \| Path \| None | `None` |
| `cmap` | str | `'plasma'` |
| `show_brain_shell` | bool | `True` |
| `mesh` | str | `'fsaverage5'` |
| `surface` | str | `'pial'` |
| `fallback_shell` | bool | `True` |
| `validate_mni` | bool | `True` |
| `mni_mask_margin_mm` | float | `4.0` |
| `outside_mni` | str | `'drop'` |

### `plot_network`

```python
plot_network(graph: 'nx.Graph', ax=None, weight_attr: 'str' = 'weight', layout: 'str' = 'kamada_kawai', seed: 'int' = 8, title: 'str | None' = None)
```

Plot a weighted response graph with strength-scaled nodes and edges.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `graph` | nx.Graph | required |
| `ax` | not specified | `None` |
| `weight_attr` | str | `'weight'` |
| `layout` | str | `'kamada_kawai'` |
| `seed` | int | `8` |
| `title` | str \| None | `None` |

### `plot_node_metric_glass_brain`

```python
plot_node_metric_glass_brain(metric_table: 'pd.DataFrame | pd.Series | dict[str, float]', elec_meta: 'pd.DataFrame', metric: 'str' = 'hub', summary: 'str' = 'max', time_s: 'float | None' = None, stimulation_nodes: 'list[str] | set[str] | tuple[str, ...] | None' = None, mni_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', title: 'str | None' = None, cmap: 'str' = 'magma', display_mode: 'str' = 'lyrz', node_size: 'float' = 42.0, brain_alpha: 'float' = 0.72, validate_mni: 'bool' = True, mni_mask_margin_mm: 'float' = 4.0, outside_mni: 'str' = 'drop')
```

Visualize a node graph metric summary on Nilearn's MNI glass brain.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `metric_table` | pd.DataFrame \| pd.Series \| dict[str, float] | required |
| `elec_meta` | pd.DataFrame | required |
| `metric` | str | `'hub'` |
| `summary` | str | `'max'` |
| `time_s` | float \| None | `None` |
| `stimulation_nodes` | list[str] \| set[str] \| tuple[str, ...] \| None | `None` |
| `mni_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `title` | str \| None | `None` |
| `cmap` | str | `'magma'` |
| `display_mode` | str | `'lyrz'` |
| `node_size` | float | `42.0` |
| `brain_alpha` | float | `0.72` |
| `validate_mni` | bool | `True` |
| `mni_mask_margin_mm` | float | `4.0` |
| `outside_mni` | str | `'drop'` |

### `plot_node_metric_template_brain`

```python
plot_node_metric_template_brain(metric_table: 'pd.DataFrame | pd.Series | dict[str, float]', elec_meta: 'pd.DataFrame', metric: 'str' = 'hub', summary: 'str' = 'max', time_s: 'float | None' = None, stimulation_nodes: 'list[str] | set[str] | tuple[str, ...] | None' = None, coord_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', views: 'tuple[str, ...]' = ('left_lateral', 'dorsal'), title: 'str | None' = None, cmap: 'str' = 'magma', mesh: 'str' = 'fsaverage5', surface: 'str' = 'pial', fallback_shell: 'bool' = True, validate_mni: 'bool' = True, mni_mask_margin_mm: 'float' = 4.0, outside_mni: 'str' = 'drop', top_n_labels: 'int' = 8)
```

Render a dynamic graph metric summary on an fsaverage cortical surface.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `metric_table` | pd.DataFrame \| pd.Series \| dict[str, float] | required |
| `elec_meta` | pd.DataFrame | required |
| `metric` | str | `'hub'` |
| `summary` | str | `'max'` |
| `time_s` | float \| None | `None` |
| `stimulation_nodes` | list[str] \| set[str] \| tuple[str, ...] \| None | `None` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `views` | tuple[str, ...] | `('left_lateral', 'dorsal')` |
| `title` | str \| None | `None` |
| `cmap` | str | `'magma'` |
| `mesh` | str | `'fsaverage5'` |
| `surface` | str | `'pial'` |
| `fallback_shell` | bool | `True` |
| `validate_mni` | bool | `True` |
| `mni_mask_margin_mm` | float | `4.0` |
| `outside_mni` | str | `'drop'` |
| `top_n_labels` | int | `8` |

### `plot_ordered_adjacency_heatmap`

```python
plot_ordered_adjacency_heatmap(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame | None' = None, ax=None, cmap: 'str' = 'magma', title: 'str' = 'Ordered response adjacency', **kwargs)
```

Build and plot an anatomically ordered electrode adjacency matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame \| None | `None` |
| `ax` | not specified | `None` |
| `cmap` | str | `'magma'` |
| `title` | str | `'Ordered response adjacency'` |
| `kwargs` | not specified | required |

### `plot_region_connectome`

```python
plot_region_connectome(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame', stim_col: 'str' = 'stim_region', record_col: 'str' = 'record_region', weight_col: 'str' = 'weight', mni_cols=('mni_x', 'mni_y', 'mni_z'), region_col: 'str' = 'anat_label', metric_label: 'str | None' = None, title: 'str' = 'Region connectome', edge_cmap: 'str' = 'plasma', display_mode: 'str' = 'lyrz')
```

Plot region-level directed response edges on an MNI connectome view.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame | required |
| `stim_col` | str | `'stim_region'` |
| `record_col` | str | `'record_region'` |
| `weight_col` | str | `'weight'` |
| `mni_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `region_col` | str | `'anat_label'` |
| `metric_label` | str \| None | `None` |
| `title` | str | `'Region connectome'` |
| `edge_cmap` | str | `'plasma'` |
| `display_mode` | str | `'lyrz'` |

### `plot_response_matrix_heatmap`

```python
plot_response_matrix_heatmap(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame | None' = None, ax=None, cmap: 'str' = 'magma', title: 'str' = 'Stimulation response matrix', **kwargs)
```

Build and plot a stimulation-source by recording-channel response matrix.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame \| None | `None` |
| `ax` | not specified | `None` |
| `cmap` | str | `'magma'` |
| `title` | str | `'Stimulation response matrix'` |
| `kwargs` | not specified | required |

### `plot_surface_connectome`

```python
plot_surface_connectome(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame', subjects_dir: 'str | Path', subject: 'str' = 'fsaverage', surface: 'str' = 'pial', source_col: 'str' = 'stim_elec', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', coord_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', top_n: 'int | None' = 80, output_png: 'str | Path | None' = None, off_screen: 'bool' = True, title: 'str' = 'ERPy surface connectome', cmap: 'str' = 'plasma', edge_radius: 'float' = 0.18, edge_radius_scale: 'float' = 1.2, edge_opacity: 'float' = 0.82)
```

Render a high-fidelity surface connectome with PyVista/MNE.

The electrode coordinates must be in the same coordinate frame as the
FreeSurfer surface. For subject-specific figures, pass that subject's
`subjects_dir` and coordinates from the same reconstruction.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame | required |
| `subjects_dir` | str \| Path | required |
| `subject` | str | `'fsaverage'` |
| `surface` | str | `'pial'` |
| `source_col` | str | `'stim_elec'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `top_n` | int \| None | `80` |
| `output_png` | str \| Path \| None | `None` |
| `off_screen` | bool | `True` |
| `title` | str | `'ERPy surface connectome'` |
| `cmap` | str | `'plasma'` |
| `edge_radius` | float | `0.18` |
| `edge_radius_scale` | float | `1.2` |
| `edge_opacity` | float | `0.82` |

### `plot_template_brain_network`

```python
plot_template_brain_network(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame', source_col: 'str' = 'stim_elec', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', coord_cols=('mni_x', 'mni_y', 'mni_z'), elec_col: 'str' = 'elec_label', region_col: 'str' = 'anat_label', top_n: 'int | None' = 80, views: 'tuple[str, ...]' = ('left_lateral', 'dorsal'), title: 'str' = 'Template brain response network', cmap: 'str' = 'plasma', node_size_values: 'dict[str, float] | pd.Series | None' = None, node_color_values: 'dict[str, float] | pd.Series | None' = None, node_color_label: 'str' = 'Active significance metrics', node_cmap: 'str' = 'YlOrRd', mesh: 'str' = 'fsaverage5', surface: 'str' = 'pial', fallback_shell: 'bool' = True, validate_mni: 'bool' = True, mni_mask_margin_mm: 'float' = 4.0, outside_mni: 'str' = 'drop')
```

Render response edges and electrodes on a template cortical surface.

ERPy first tries Nilearn's packaged fsaverage cortical meshes, which are
derived from FreeSurfer surfaces and work well for MNI-coordinate public
datasets. If Nilearn is unavailable, the function can fall back to the
lightweight translucent shell. For patient-specific surgical figures, use
`plot_surface_connectome` with that subject's FreeSurfer reconstruction.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame | required |
| `source_col` | str | `'stim_elec'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `coord_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `elec_col` | str | `'elec_label'` |
| `region_col` | str | `'anat_label'` |
| `top_n` | int \| None | `80` |
| `views` | tuple[str, ...] | `('left_lateral', 'dorsal')` |
| `title` | str | `'Template brain response network'` |
| `cmap` | str | `'plasma'` |
| `node_size_values` | dict[str, float] \| pd.Series \| None | `None` |
| `node_color_values` | dict[str, float] \| pd.Series \| None | `None` |
| `node_color_label` | str | `'Active significance metrics'` |
| `node_cmap` | str | `'YlOrRd'` |
| `mesh` | str | `'fsaverage5'` |
| `surface` | str | `'pial'` |
| `fallback_shell` | bool | `True` |
| `validate_mni` | bool | `True` |
| `mni_mask_margin_mm` | float | `4.0` |
| `outside_mni` | str | `'drop'` |

### `response_matrix_from_edges`

```python
response_matrix_from_edges(edges: 'pd.DataFrame', elec_meta: 'pd.DataFrame | None' = None, source_col: 'str' = 'stim_pair', target_col: 'str' = 'record_elec', weight_col: 'str' = 'weight', agg: 'str' = 'mean') -> 'pd.DataFrame'
```

Return stimulation-source by recording-channel response weights.

Unlike a graph adjacency matrix, this table keeps stimulation sources and
recording contacts on separate axes. That is the view most users want for
CCEP/SPES response fields, especially when comparing multiple stimulation
sites in the same subject or public dataset.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `edges` | pd.DataFrame | required |
| `elec_meta` | pd.DataFrame \| None | `None` |
| `source_col` | str | `'stim_pair'` |
| `target_col` | str | `'record_elec'` |
| `weight_col` | str | `'weight'` |
| `agg` | str | `'mean'` |

**Returns:** pd.DataFrame.

### `resolve_electrode_label`

```python
resolve_electrode_label(label: 'str', available_labels: 'list[str] | None' = None) -> 'str'
```

Match compact ERPy labels such as `PT1` to coordinate labels such as `PT01`.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `label` | str | required |
| `available_labels` | list[str] \| None | `None` |

**Returns:** str.

### `stim_pair_to_contact_electrodes`

```python
stim_pair_to_contact_electrodes(stim_pair: 'str') -> 'tuple[str, ...]'
```

Return both contact labels represented by a stimulation-pair label.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `stim_pair` | str | required |

**Returns:** tuple[str, ...].

### `validate_mni_coordinates`

```python
validate_mni_coordinates(elec_meta: 'pd.DataFrame', elec_col: 'str' = 'elec_label', mni_cols=('mni_x', 'mni_y', 'mni_z'), margin_mm: 'float' = 4.0) -> 'pd.DataFrame'
```

Flag whether electrode coordinates fall inside the MNI152 brain mask.

**Parameters**

| Name | Type | Default |
| --- | --- | --- |
| `elec_meta` | pd.DataFrame | required |
| `elec_col` | str | `'elec_label'` |
| `mni_cols` | not specified | `('mni_x', 'mni_y', 'mni_z')` |
| `margin_mm` | float | `4.0` |

**Returns:** pd.DataFrame.
