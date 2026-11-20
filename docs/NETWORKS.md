# Network analysis and brain visualization

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

Detections become directed stimulation-to-response edges, which can be
summarized as matrices and graphs, tracked through time, and drawn on template
or subject-specific brains. Brain figures need MNI coordinates in the electrode
metadata and the `viz` extra; surface rendering needs the `surface` extra.

## From detections to a graph

```python
from ERPy.viz.networks import edges_from_detections, graph_from_edges, response_matrix_from_edges

edges = edges_from_detections(
    detections.assign(stim_pair="STIM_A_1_2"),
    elec_meta=patient.elec_meta,
    metric="peak_amplitude_uv",
    significance_column="primary_qc_pass",
)
matrix = response_matrix_from_edges(edges, elec_meta=patient.elec_meta)
graph = graph_from_edges(edges)       # NetworkX DiGraph
```

## Graphs, brain views, and animations

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

## Source-node roles and target distributions

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

## Coordinates and brain templates

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

## See also

- [Methods: graph construction](METHODS.md#graph-construction)
- [Notebook 04: network visualizations](../notebooks/examples/04_network_visualizations.ipynb)
- [Notebook 05: brain and interactive visualizations](../notebooks/examples/05_brain_and_interactive_visualizations.ipynb)
- [Visualization gallery](VISUALIZATION_GALLERY.md)
