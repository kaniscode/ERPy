"""Build the small, deterministic notebooks shipped with ERPy."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "examples"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


SETUP = code(
    """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow this notebook to run from JupyterLab or from the repository root.
HERE = Path.cwd()
EXAMPLES = HERE if (HERE / "_synthetic.py").exists() else HERE / "notebooks" / "examples"
sys.path.insert(0, str(EXAMPLES.resolve()))
REPOSITORY = EXAMPLES.parents[1]
if (REPOSITORY / "ERPy").exists():
    sys.path.insert(0, str(REPOSITORY.resolve()))

import ERPy as ep
import ERPy.viz as viz
from _synthetic import (
    make_detection_table,
    make_edges,
    make_electrode_metadata,
    make_epochs,
    make_metric_table,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})
"""
)


def notebook(
    title: str,
    purpose: str,
    cells: list,
    *,
    data_note: str | None = None,
) -> nbf.NotebookNode:
    if data_note is None:
        data_note = (
            "All signals, labels, and coordinates in this notebook are deterministic and\n"
            "synthetic. They do not represent a participant. Run cells from top to bottom."
        )
    intro = md(
        f"""
# {title}

{purpose}

{data_note}
"""
    )
    nb = nbf.v4.new_notebook(cells=[intro, copy.deepcopy(SETUP), *cells])
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    return nb


NOTEBOOKS = {
    "00_quickstart.ipynb": notebook(
        "ERPy quick start",
        "Create an `Epochs` object, inspect it, run the primary detector, and make one audit-ready summary.",
        [
            code(
                """
epochs = make_epochs()
print(f"Python: {sys.executable}")
print(f"ERPy {ep.__version__}: {epochs.n_trials()} trials, {len(epochs.channels)} channels")
epochs.get_mean_waveform().head(3)
"""
            ),
            code(
                """
detections = epochs.detect_erp_all(
    methods=["crp_energy", "peak_amplitude", "rms_response"],
    min_consensus=1,
)
columns = ["channel", "method", "significant", "primary_significant", "primary_classification"]
detections[columns].drop_duplicates().head(12)
"""
            ),
            code(
                """
# The `p_crp` field name is retained for compatibility. On the primary
# `crp_energy` row it stores the fixed-window reproducibility value p_R.
primary_columns = [
    "channel", "p_crp", "p_energy", "p_joint", "q_joint",
    "primary_classification", "primary_significant",
]
detections.loc[detections["method"].eq("crp_energy"), primary_columns].head()
"""
            ),
            md(
                """
The primary result requires both components: fixed-window whole-trial
amplitude-weighted waveform reproducibility $p_R$ and separately demeaned
matched excess energy $p_E$. The displayed four-class label uses the
unadjusted component thresholds; the final response call uses the
family-adjusted joint value and artifact eligibility.
The standalone `crp_significance` method shown later is a different,
data-selected-duration extraction test retained for descriptive comparison.
"""
            ),
            code(
                """
fig = epochs.plot.summary("CONTACT_B1", detections=detections)
fig.suptitle("Synthetic response: waveform, trials, and detector context", y=1.02)
plt.show()
"""
            ),
            md(
                """
The result is an audit aid, not a clinical interpretation. In real work, next
review event alignment, the artifact report, the post-artifact anchor, and the
clean-trial count before using a response label.
"""
            ),
        ],
    ),
    "01_waveform_visualizations.ipynb": notebook(
        "Waveform visualizations",
        "Use trial overlays, means/SEM, heatmaps, small multiples, comparisons, and the compact DataFrame plotting helpers.",
        [
            code("epochs = make_epochs()\ndetections = make_detection_table(epochs)\nmetadata = make_electrode_metadata()"),
            code(
                """
fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
viz.plot_mean(epochs, "CONTACT_B1", ax=axes[0, 0])
viz.plot_overlay(epochs, "CONTACT_B1", ax=axes[0, 1])
viz.plot_heatmap(epochs, "CONTACT_B1", ax=axes[1, 0])
viz.plot_butterfly(
    epochs,
    channels=["STIM_A1", "STIM_A2", "CONTACT_B1", "CONTACT_B2"],
    ax=axes[1, 1],
)
plt.show()
"""
            ),
            code(
                """
fig, _ = viz.plot_grid(epochs, channels=epochs.channels, ncols=3)
fig.suptitle("Channel grid", y=1.01)
plt.show()

fig, _ = viz.plot_grouped_grid(
    epochs, metadata, group_col="anat_label", channel_col="channel", ncols=2
)
fig.suptitle("Anatomy-organized grid", y=1.01)
plt.show()
"""
            ),
            code(
                """
weaker = make_epochs(seed=24, response_scale=0.55)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
viz.plot_mean_erp_comparison(
    {"full response": epochs, "weaker response": weaker},
    "CONTACT_B1",
    ax=axes[0],
)
viz.plot_response_map(detections, metric="peak_amplitude_uv", ax=axes[1])
plt.show()
"""
            ),
            code(
                """
fig, _ = viz.plot_ranked_grid(
    epochs, detections, metric="peak_amplitude_uv", top_n=6, ncols=3
)
fig.suptitle("Response-ranked grid", y=1.01)
plt.show()

fig = viz.plot_summary_panel(epochs, "CONTACT_B1", detections=detections)
plt.show()

fig, _ = viz.plot_trials_grid(epochs, "CONTACT_B1", ncols=6)
fig.suptitle("Individual trials", y=1.01)
plt.show()
"""
            ),
            code(
                """
# The compact helpers accept the underlying MultiIndex DataFrame directly.
fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
viz.quick_erp(epochs.epochs_df, "CONTACT_B1", ax=axes[0, 0])
viz.heatmap_epochs(epochs.epochs_df, "CONTACT_B1", ax=axes[0, 1])
viz.heatmap_channels_time(epochs.epochs_df, channels=epochs.channels[:4], ax=axes[1, 0])
viz.plot_metric_vs_epoch(epochs.epochs_df, "CONTACT_B1", rolling=3, ax=axes[1, 1])
plt.show()

fig, _ = viz.plot_erp_grid(epochs.epochs_df, channels=epochs.channels[:4], ncols=2)
plt.show()

ax = viz.plot_mean_erp_dataframe_comparison(
    [("full", epochs.epochs_df), ("weaker", weaker.epochs_df)], "CONTACT_B1"
)
plt.show()
"""
            ),
        ],
    ),
    "02_detection_qc_and_crp.ipynb": notebook(
        "Detection, artifact QC, and CRP",
        "Inspect detector agreement, artifact reasons, response metrics, and canonical response parametrization without hiding method-specific quantities.",
        [
            code("epochs = make_epochs()\ndetections = make_detection_table(epochs)"),
            md(
                """
This compact teaching fixture is separate from ERPy's declared synthetic
benchmark. In that benchmark, one *testing family* is a four-channel group
processed together for one simulated stimulation acquisition: one designated
target and three known-null references. Only detector-eligible contacts with
finite joint p-values enter the multiple-testing adjustment, so the adjusted
set has three contacts when target-trial QC leaves too few usable target trials
and four contacts otherwise. The group mirrors the recording contacts
considered for one adjustment in real data; the word *family* refers only to
that multiple-testing set. The benchmark varies trial count, response-to-noise
ratio, and random target-trial removal so detector behavior can be measured
against known synthetic truth under stationary independent Gaussian AR(1)
noise. It characterizes the four-contact model rather than the multiplicity
burden of larger real acquisitions or the accuracy of artifact identification.
The public-data recipe in notebook 07 instead checks end-to-end portability on
an independently hosted real recording; results from the two sources are
reported separately.
"""
            ),
            code(
                """
artifact_report = epochs.flag_artifacts()
artifact_report.table.loc[artifact_report.table["bad_response"],
                          ["epoch", "channel", "reason", "peak_abs", "ptp"]].head(10)
"""
            ),
            code(
                """
fig = viz.plot_detection_summary(detections, top_n=12)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
viz.plot_detection_method_matrix(detections, ax=axes[0])
viz.plot_method_significance_counts(detections, ax=axes[1])
plt.show()

matrix = viz.method_significance_matrix(detections)
strong, weak = viz.choose_significant_and_nonsignificant_channels(detections, epochs.channels)
print("Illustrative channels:", strong, weak)
matrix
"""
            ),
            code(
                """
metric_table = detections.drop_duplicates("channel")[["channel", "peak_amplitude_uv"]].copy()
metric_table["stim_pair"] = "STIM_A_1_2"
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
viz.plot_within_stim_zscore_bars(metric_table, "peak_amplitude_uv", ax=axes[0])
viz.plot_within_stim_zscore_heatmap(metric_table, "peak_amplitude_uv", ax=axes[1])
plt.show()
"""
            ),
            md(
                """
The next cells visualize the standalone canonical response parametrization
and its selected-duration extraction quantities. These CRP plots are
descriptive/comparator views; they do not define the primary fixed-window
whole-trial reproducibility value $p_R$.
"""
            ),
            code(
                """
crp = ep.run_crp(epochs, "CONTACT_B1")
fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), layout="constrained")
viz.plot_crp_curve(crp, ax=axes[0])
viz.plot_crp_projections(crp, ax=axes[1])
viz.plot_crp_weight_timecourse(crp, ax=axes[2], zscore=True)
plt.show()

fig = viz.plot_crp_summary(epochs, "CONTACT_B1", result=crp)
plt.show()
"""
            ),
            code(
                """
crp_table = ep.run_crp_all(epochs)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
viz.plot_crp_score_map(crp_table, ax=axes[0], top_n=6)
comparison = crp_table[["channel", "score"]].rename(columns={"score": "within_stim_z"})
comparison["stim_pair"] = "STIM_A_1_2"
viz.plot_crp_site_comparison(comparison, ax=axes[1], top_n_channels=6)
plt.show()

fig = viz.plot_crp_response_contrast(epochs, strong, weak)
plt.show()
"""
            ),
            code(
                """
# This audit panel recomputes all source-native detector quantities and is
# slower. Enable it when reviewing a real candidate response.
RUN_FULL_DIAGNOSTIC = False
if RUN_FULL_DIAGNOSTIC:
    fig = viz.plot_published_detector_diagnostic(
        epochs, "CONTACT_B1", signi_n_permutations=1_000
    )
    plt.show()
"""
            ),
        ],
    ),
    "03_spectral_visualizations.ipynb": notebook(
        "Spectral visualizations",
        "Compute compact Welch, Morlet, phase, connectivity, and cross-frequency examples with explicit bands and windows.",
        [
            code(
                """
epochs = make_epochs(n_trials=12, sfreq=500.0)
config = ep.SpectralConfig(
    fmin=4.0, fmax=80.0, n_freqs=18, n_cycles=5.0,
    baseline=(-0.4, -0.05), baseline_mode="logratio", decim=4,
)
"""
            ),
            code(
                """
psd = ep.compute_psd(epochs, channels=epochs.channels[:4], fmin=2, fmax=100, nperseg=256)
ax = viz.plot_psd(psd, logx=False)
plt.show()

tfr = ep.compute_tfr(epochs, "STIM_A1", config=config)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
viz.plot_tfr(tfr, ax=axes[0])
viz.plot_ersp(tfr, ax=axes[1])
itpc = ep.inter_trial_phase_coherence(epochs, "STIM_A1", config=config)
viz.plot_itpc(itpc, ax=axes[2])
plt.show()
"""
            ),
            code(
                """
plv_tf = ep.phase_locking_value(epochs, "STIM_A1", "STIM_A2", config=config)
plv = ep.plv_matrix(epochs, band=(8, 13), channels=epochs.channels[:4], time_window=(0.01, 0.25))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
viz.plot_plv_timefreq(plv_tf, ax=axes[0])
viz.plot_connectivity_matrix(plv, ax=axes[1], title="Alpha-band PLV")
plt.show()
"""
            ),
            code(
                """
pac = ep.phase_amplitude_coupling(
    epochs, "STIM_A1", "STIM_A1", phase_band=(4, 8), amp_band=(40, 70),
    time_window=(0.02, 0.30)
)
comod = ep.phase_amplitude_comodulogram(
    epochs, "STIM_A1", phase_freqs=[4, 6, 8], amp_freqs=[35, 50, 65],
    bandwidth_phase=2, bandwidth_amp=14, time_window=(0.02, 0.30)
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
viz.plot_pac(pac, ax=axes[0])
viz.plot_comodulogram(comod, ax=axes[1])
plt.show()
"""
            ),
            code(
                """
phase_matrix = ep.phase_phase_matrix(
    epochs, "STIM_A1", "STIM_A2", freqs_x=[4, 6, 8, 10],
    freqs_y=[4, 6, 8, 10], time_window=(0.02, 0.30)
)
ax = viz.plot_phase_phase_matrix(phase_matrix)
plt.show()

# The accessor exposes the same calculations and plotting family.
accessor = epochs.spectral
print(type(accessor).__name__, type(accessor.plot).__name__)

fig = viz.plot_spectral_summary(epochs, "STIM_A1", config=config)
plt.show()
"""
            ),
        ],
    ),
    "04_network_visualizations.ipynb": notebook(
        "Network and graph visualizations",
        "Turn an explicit edge table into adjacency matrices, directed graphs, coordinate views, response matrices, and dynamic metric panels.",
        [
            code(
                """
edges = make_edges()
metadata = make_electrode_metadata()
metrics = make_metric_table()
epochs = make_epochs()
detections = make_detection_table(epochs)
derived_edges = viz.edges_from_detections(
    detections, stim_pair="STIM_A_1_2", elec_meta=metadata,
    significance_column="primary_significant", qc_only=False,
)
print("Canonical identity:", viz.electrode_label_identity("STIM_A_1"))
derived_edges.head()
"""
            ),
            code(
                """
adjacency = viz.adjacency_from_edges(edges)
ordered = viz.ordered_adjacency_from_edges(edges, metadata)
response_matrix = viz.response_matrix_from_edges(edges, metadata)
graph = viz.graph_from_edges(edges)
graph_from_matrix = viz.graph_from_matrix(adjacency)
print("Resolved label:", viz.resolve_electrode_label("STIM_A_1", metadata.elec_label.tolist()))
print("Stim contacts:", viz.stim_pair_to_contact_electrodes("STIM_A_1_2"))
print("Node order:", viz.order_nodes_by_metadata(list(graph.nodes), metadata))
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
viz.plot_adjacency_heatmap(adjacency, ax=axes[0])
viz.plot_ordered_adjacency_heatmap(edges, metadata, ax=axes[1])
viz.plot_response_matrix_heatmap(edges, metadata, ax=axes[2])
plt.show()
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
viz.plot_network(graph, ax=axes[0], title="Topology view")
viz.plot_coordinate_network(edges, metadata, ax=axes[1], top_n=None)
plt.show()
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
viz.plot_graph_metric_timecourse(metrics, node="STIM_A1", metric="hub", ax=axes[0])
viz.plot_graph_metric_heatmap(metrics, metric="hub", top_n_nodes=6, ax=axes[1])
plt.show()
"""
            ),
            code(
                """
coords = viz.node_coordinates_from_metadata(metadata, list(graph.nodes))
sources = viz.order_sources_by_metadata(edges.stim_elec.unique().tolist(), metadata)
qc = viz.validate_mni_coordinates(metadata)
print("Sources:", sources)
coords
"""
            ),
        ],
    ),
    "05_brain_and_interactive_visualizations.ipynb": notebook(
        "Brain and interactive visualizations",
        "Use fictitious coordinates to demonstrate glass-brain and Plotly views. Surface recipes are kept opt-in because they require large template assets.",
        [
            code("epochs = make_epochs(n_trials=10)\nedges = make_edges()\nmetadata = make_electrode_metadata()\nmetrics = make_metric_table()"),
            code(
                """
display = viz.plot_electrode_mni(
    metadata[["mni_x", "mni_y", "mni_z"]].to_numpy(),
    values=np.linspace(0.2, 1.0, len(metadata)),
    title="Fictitious teaching coordinates",
)
plt.show()

display = viz.plot_electrodes_from_metadata(metadata, value_col="mni_z")
plt.show()

metric = pd.Series(np.linspace(0.1, 1.0, len(metadata)), index=metadata.elec_label)
display = viz.plot_erps_on_brain(metadata, metric, title="Synthetic response metric")
plt.show()
"""
            ),
            code(
                """
display = viz.plot_glass_brain_network(
    edges, metadata, validate_mni=False, title="Synthetic response network"
)
plt.show()

display = viz.plot_node_metric_glass_brain(
    metrics, metadata, metric="hub", validate_mni=False,
    stimulation_nodes=["STIM_A1"], title="Synthetic hub magnitude"
)
plt.show()
"""
            ),
            code(
                """
# Plotly figures are assigned rather than rendered into the notebook so the
# file stays small. In JupyterLab, put the variable name on the last line to
# display and rotate the view.
interactive = viz.plot_interactive_connectome(
    edges, metadata, show_brain_shell=False, validate_mni=False
)
animated = viz.plot_evoked_response_graph(
    epochs, metadata, stim_pair="STIM_A_1_2", channels=epochs.channels,
    show_brain_shell=False, validate_mni=False, top_n=6, frame_step_ms=40
)
print(type(interactive).__name__, len(animated.frames), "animation frames")
"""
            ),
            code(
                """
# These renderers need template surfaces and, for `plot_surface_connectome`,
# a local FreeSurfer subjects directory. They are opt-in by design.
RUN_SURFACE_RECIPES = False
if RUN_SURFACE_RECIPES:
    template = viz.plot_template_brain_network(edges, metadata)
    node_template = viz.plot_node_metric_template_brain(metrics, metadata, metric="hub")
    region = viz.plot_region_connectome(edges, metadata)
    electrode = viz.plot_electrode_connectome(
        viz.adjacency_from_edges(edges), metadata
    )
    surface = viz.plot_surface_connectome(
        edges, metadata, subjects_dir=Path("freesurfer_subjects")
    )

# For pooled runs, provide dictionaries containing epochs, detections, and a
# stimulation label, then build one multi-source animation:
RUN_AGGREGATE_RECIPE = False
if RUN_AGGREGATE_RECIPE:
    runs = [{
        "epochs": epochs,
        "stim_pair": "STIM_A_1_2",
        "detections": make_detection_table(epochs),
    }]
    aggregate = viz.plot_aggregate_evoked_response_graph(
        runs, metadata, show_brain_shell=False, validate_mni=False
    )
"""
            ),
        ],
    ),
    "06_exporting_figures.ipynb": notebook(
        "Exporting high quality visualizations",
        "Apply final layout checks and save Matplotlib figures in editable PDF plus high-resolution PNG without embedding project paths.",
        [
            code("epochs = make_epochs()\ndetections = make_detection_table(epochs)"),
            code(
                """
fig = viz.plot_summary_panel(epochs, "CONTACT_B1", detections=detections)
fig.set_size_inches(7.2, 6.0)
fig.suptitle("Synthetic ERPy audit panel", y=1.01)
fig.canvas.draw()
plt.show()
"""
            ),
            code(
                """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as temporary_folder:
    paths = viz.save_analysis_figure(
        fig, "synthetic_audit_panel", patient_path=temporary_folder,
        formats=("png", "pdf")
    )
    print([Path(path).name for path in paths])
"""
            ),
            md(
                """
Before submission, inspect every file at the intended column width. Confirm
that labels, legends, panel letters, confidence bands, and color scales are
readable; keep at least 3 mm of quiet space around text; use vector PDF/SVG for
line art; and verify that PNGs are at least 300 dpi. Remove private identifiers
from visible content and file metadata, but retain a governed provenance record.

These export functions write to the path you provide and do not upload the
figure or its source data. An optional `ERPY-RR-<UUID>` release-record label is
an author-assigned local governance label, not a server address, and ERPy does
not create or register one automatically.
"""
            ),
        ],
    ),
    "07_public_ds003708_recipe.ipynb": notebook(
        "Public OpenNeuro ds003708 recipe",
        "Reproduce the public-data portability workflow without embedding a multi-gigabyte dataset or downloaded outputs in the repository.",
        [
            md(
                """
The Miller–Müller–Hermes basis-profile dataset is available as OpenNeuro
`ds003708`, snapshot v1.0.4. ERPy's validation command reads only event-centered
byte ranges by default. Run the following in a terminal from the repository:

```text
python validation/validate_public_spes.py --dataset ds003708 \\
  --min-events 8 --sites-per-dataset 1 --max-events 17 \\
  --max-channels 32 --figures
```

The generated workspace is intentionally ignored by Git. Review its manifest,
event table, QC table, detector parameters, and source URL before interpreting
the figures. This is an external portability and numerical-reproduction
exercise rather than a performance benchmark.
"""
            ),
            code(
                """
from pathlib import Path

workspace = Path("validation/public_spes_work/ds003708")
if workspace.exists():
    files = sorted(path.relative_to(workspace) for path in workspace.rglob("*") if path.is_file())
    print(f"Found {len(files)} local validation files")
    files[:20]
else:
    print("No local workspace found. Run the terminal command above when ready.")
"""
            ),
        ],
        data_note=(
            "This notebook embeds no recording data or downloaded outputs. Its terminal "
            "command retrieves event-centered windows from a public participant dataset; "
            "review that dataset's terms and provenance before running it. Run the Python "
            "cells from top to bottom."
        ),
    ),
}


def assign_deterministic_cell_ids(filename: str, nb: nbf.NotebookNode) -> None:
    """Assign stable nbformat cell IDs from file name, position, type, and source."""

    for index, cell in enumerate(nb.cells):
        material = "\0".join(
            [filename, str(index), str(cell.cell_type), str(cell.source)]
        ).encode("utf-8")
        cell["id"] = "cell-" + hashlib.sha256(material).hexdigest()[:16]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, nb in NOTEBOOKS.items():
        assign_deterministic_cell_ids(filename, nb)
        nbf.validate(nb)
        nbf.write(nb, OUT / filename)


if __name__ == "__main__":
    main()
