# Visualization gallery

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

The example notebooks include executed tables and inline PNG figures that can
be viewed directly on GitHub. The lead notebook 00 uses an actual deidentified
recording from the CNS/ACC–PAG cohort, with anatomical display labels.
Notebooks 01–06 use deterministic synthetic signals and fictitious coordinates.
Notebook 07 runs a small, explicitly
identified public OpenNeuro ds003708 recording subset. Notebook 08 applies a
participant-held-out negative-N1 development model to actual public ds004774
derived features, with input/model hash checks and no raw download. Its score
is separate from general response p/q values. Earlier stored outputs retain
their historical execution versions. Keep the complete
`notebooks/examples` folder because the
notebooks import its `_cohort.py` and `_synthetic.py` helpers. To rerun the lead
recording, configure `ERPY_COHORT_EXAMPLE_DIR` to its authorized local source
directory before starting Jupyter. From the repository root, open it with:

```text
python -m pip install jupyterlab
python -m jupyter lab notebooks/examples/00_quickstart.ipynb
```

Expensive surface renderers are opt-in because they require local template
assets.

The tables below enumerate every name exported by `ERPy.viz`. Plotting
callables are shown in a runnable cell unless an external template or
FreeSurfer directory is required; those callables have complete opt-in recipes
in notebook 05. Data-transformation helpers exported beside the plots are
demonstrated in notebook 04. The generated [API reference](API_REFERENCE.md)
provides the complete signatures and parameter descriptions.

| Notebook | What it teaches |
| --- | --- |
| [`00_quickstart.ipynb`](../notebooks/examples/00_quickstart.ipynb) | Actual cohort trials, source QC, waveforms, and freshly computed single-contact reproducibility/energy evidence. |
| [`01_waveform_visualizations.ipynb`](../notebooks/examples/01_waveform_visualizations.ipynb) | Means/SEM, trials, heatmaps, comparisons, ranked and grouped grids. |
| [`02_detection_qc_and_crp.ipynb`](../notebooks/examples/02_detection_qc_and_crp.ipynb) | Artifact reasons, detector matrices, response metrics, and CRP panels. |
| [`03_spectral_visualizations.ipynb`](../notebooks/examples/03_spectral_visualizations.ipynb) | PSD, ERSP/TFR, ITPC, PLV, PAC, and phase coupling. |
| [`04_network_visualizations.ipynb`](../notebooks/examples/04_network_visualizations.ipynb) | Edge tables, adjacency/response matrices, topology, coordinates, and graph metrics. |
| [`05_brain_and_interactive_visualizations.ipynb`](../notebooks/examples/05_brain_and_interactive_visualizations.ipynb) | MNI glass-brain, Plotly, animation, and opt-in surface recipes. |
| [`06_exporting_figures.ipynb`](../notebooks/examples/06_exporting_figures.ipynb) | High quality layout checks and reproducible PNG/PDF export. |
| [`07_public_ds003708_recipe.ipynb`](../notebooks/examples/07_public_ds003708_recipe.ipynb) | Run a public recording subset and view input provenance, detector results, and waveform figures. |
| [`08_n1_development.ipynb`](../notebooks/examples/08_n1_development.ipynb) | Verify an actual public record and held-out N1 model, reproduce its score, round-trip model save/load, and inspect signed logit contributions. |

## Waveform and response views

| Public callable | Notebook | Output |
| --- | --- | --- |
| `EpochsPlotter`, `plot_summary_panel` | 00, 01, 06 | Bound plotting accessor and audit panel. |
| `plot_mean`, `plot_overlay`, `plot_butterfly` | 01 | Mean/SEM, individual trials, and multichannel overlay. |
| `plot_heatmap`, `plot_trials_grid` | 01 | Trial-by-time heatmap and one panel per trial. |
| `plot_grid`, `plot_grouped_grid`, `plot_ranked_grid` | 01 | Channel small multiples by order, metadata, or response metric. |
| `plot_mean_erp_comparison` | 01 | Same channel across multiple `Epochs` objects. |
| `plot_response_map` | 01 | Ranked channel response quantity. |
| `quick_erp`, `plot_erp_grid` | 01 | Compact views from a MultiIndex DataFrame. |
| `plot_mean_erp_dataframe_comparison` | 01 | DataFrame-based condition overlay. |
| `heatmap_channels_time`, `heatmap_epochs` | 01 | Mean channel-by-time and trial-by-time matrices. |
| `plot_metric_vs_epoch` | 01 | Trial-order metric with optional rolling summary. |

## Detection, QC, and canonical response views

| Public callable | Notebook | Output |
| --- | --- | --- |
| `method_significance_matrix` | 02 | Channel-by-method Boolean table used by plots. |
| `plot_detection_method_matrix` | 02 | Visual detector agreement by channel. |
| `plot_method_significance_counts` | 02 | Per-method positive-channel counts. |
| `plot_detection_summary` | 02 | Combined agreement matrix and counts. |
| `choose_significant_and_nonsignificant_channels` | 02 | Deterministic illustrative-channel helper. |
| `plot_published_detector_diagnostic` | 02 | Executed source-native detector diagnostic with 1,000 SIGNI permutations. |
| `plot_within_stim_zscore_bars`, `plot_within_stim_zscore_heatmap` | 02 | Within-stimulation response metric normalization. |
| `plot_crp_curve`, `plot_crp_projections` | 02 | Canonical waveform and trial coefficients. |
| `plot_crp_weight_timecourse` | 02 | Sliding canonical expression, raw or baseline z-scored. |
| `plot_crp_score_map`, `plot_crp_site_comparison` | 02 | CRP quantities within and across stimulation sites. |
| `plot_crp_response_contrast`, `plot_crp_summary` | 02 | Positive/negative contrast and full CRP audit panel. |

## Spectral views

| Public callable | Notebook | Output |
| --- | --- | --- |
| `SpectralPlotter`, `plot_spectral_summary` | 03 | Bound accessor and combined channel summary. |
| `plot_psd` | 03 | Welch spectrum by channel. |
| `plot_tfr`, `plot_ersp` | 03 | Morlet power or baseline-transformed ERSP. |
| `plot_itpc` | 03 | Inter-trial phase coherence. |
| `plot_plv_timefreq` | 03 | Time-frequency PLV between two contacts. |
| `plot_connectivity_matrix` | 03 | PLV, coherence, mutual-information, or related matrix. |
| `plot_pac`, `plot_comodulogram` | 03 | Phase-amplitude distribution and frequency grid. |
| `plot_phase_phase_matrix` | 03 | n:m phase coupling across frequency pairs. |

`plot_spectral_summary` is also available from
`epochs.spectral.plot.summary(...)`; notebook 03 calls the complete summary and
constructs its constituent panels separately so every parameter is visible.

## Network and anatomy views

| Public callable | Notebook | Output |
| --- | --- | --- |
| `plot_adjacency_heatmap`, `plot_ordered_adjacency_heatmap` | 04 | Raw and metadata-ordered adjacency. |
| `plot_response_matrix_heatmap` | 04 | Stimulation-pair by recording-contact response matrix. |
| `plot_network`, `plot_coordinate_network` | 04 | Topological and coordinate-projected graphs. |
| `plot_graph_metric_timecourse`, `plot_graph_metric_heatmap` | 04 | Dynamic node metrics. |
| `plot_electrode_mni`, `plot_electrodes_from_metadata`, `plot_erps_on_brain` | 05 | Contacts and scalar response values in MNI views. |
| `plot_glass_brain_network` | 05 | Response edges on a Nilearn glass brain. |
| `plot_node_metric_glass_brain` | 05 | Node metric on a glass brain. |
| `plot_interactive_connectome` | 05 | Rotatable Plotly connectome. |
| `plot_evoked_response_graph` | 05 | Time-resolved single-stimulation Plotly graph. |
| `plot_aggregate_evoked_response_graph` | 05 | Executed aggregate animation with a static preview. |
| `plot_template_brain_network`, `plot_node_metric_template_brain` | 05 | Template cortical views; opt-in template assets. |
| `plot_region_connectome`, `plot_electrode_connectome` | 05 | Nilearn region/electrode connectomes; opt-in recipe. |
| `plot_surface_connectome` | 05 | FreeSurfer/MNE/PyVista surface; requires `subjects_dir`. |

The transformation helpers used before network plotting are demonstrated in
notebook 04: `edges_from_detections`, `adjacency_from_edges`,
`ordered_adjacency_from_edges`, `response_matrix_from_edges`,
`graph_from_edges`, `graph_from_matrix`, `node_coordinates_from_metadata`,
`order_nodes_by_metadata`, `order_sources_by_metadata`,
`resolve_electrode_label`, `electrode_label_identity`,
`stim_pair_to_contact_electrodes`, and `validate_mni_coordinates`.

## Saving figures

`save_analysis_figure` is demonstrated in notebook 06. Prefer PDF or SVG for
editable line art and a 300-dpi PNG for review systems. Figure exports should
name the metric, window, and analysis scope in a nearby caption or provenance
table. Inspect the final image at journal/poster size, not only on a large
monitor. The export is local; it does not transmit the figure or source data.

## Optional dependency boundaries

- Base Matplotlib views require only the core package.
- Plotly/Nilearn views require the `viz` extra.
- FreeSurfer surface rendering requires the `surface` extra plus local surface
  assets and a known coordinate frame.
- Public validation data are downloaded when notebook 07 is executed;
  workspaces remain outside version control.

From a source checkout with the environment active, install coordinate-based
visualizations with `python -m pip install ".[viz]"`. For the optional
FreeSurfer/MNE/PyVista renderer, use
`python -m pip install ".[viz,surface]"` instead.

Notebook 07 also needs the public-validation dependencies. Install them from
the same source checkout before executing its cells:

```text
python -m pip install ".[validation]"
```

## Running the worked notebooks

Notebook 00 requires authorized access to the documented cohort trial export.
Set `ERPY_COHORT_EXAMPLE_DIR` to its local method-example directory before
starting Jupyter. The loader checks the input checksum,
reads only the numeric trial export, and uses generic anatomical display labels.
Restricted recordings and private locator metadata are not distributed in this
repository. The stored lead figures can be viewed without those inputs. The
lead contact was selected previously for method illustration; its unadjusted
results are not an acquisition-wide response call or an accuracy benchmark.
It does not substitute a synthetic or OpenNeuro signal when input is missing.

The recorded outputs are real cell executions. Execution counts, source hashes,
Python version, and key package versions are retained in each notebook's
`erpy_execution` metadata. Machine paths and execution timing are removed.
Warnings that affect interpretation remain visible. The public example also
displays its source URLs and the SHA-256 hash of its downloaded window CSV.

With the authorized cohort input configured, open a worked notebook in Jupyter
and run its cells from top to bottom with a fresh kernel. Install its optional
dependencies as described above. Save a separate copy when changing the inputs
or analysis settings so the stored example remains available for comparison.

Plotly PNG export uses Kaleido and requires Chrome or Chromium. In Jupyter,
call `interactive.show()` or `animated.show()` to rotate/play those figures;
GitHub displays the stored PNG snapshots. Template/FreeSurfer surface recipes
remain explicitly opt-in because they need separately installed assets.
