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
import platform
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
from IPython.display import Image, display as show_output
from _synthetic import (
    make_detection_table,
    make_edges,
    make_electrode_metadata,
    make_epochs,
    make_metric_table,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# The publication workflow stores PNGs so the figures render directly on GitHub.
get_ipython().run_line_magic("matplotlib", "inline")

def show_plotly_snapshot(figure):
    # Use figure.show() to interact locally; these PNGs also render on GitHub.
    import plotly.graph_objects as go
    preview = go.Figure(figure)
    preview.layout.sliders = ()
    preview.layout.updatemenus = ()
    preview.update_layout(
        legend={"orientation": "h", "y": -0.08, "x": 0.05},
        margin={"l": 30, "r": 100, "t": 65, "b": 70},
    )
    show_output(Image(preview.to_image(format="png", width=1000, height=650, scale=1)))
"""
)


def notebook(
    title: str,
    purpose: str,
    cells: list,
    *,
    data_note: str | None = None,
    setup=None,
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
    nb = nbf.v4.new_notebook(cells=[intro, copy.deepcopy(SETUP if setup is None else setup), *cells])
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
        "ERPy quick start: a deidentified cohort recording",
        "Inspect an actual recording from the CNS/ACC–PAG analysis cohort, preserve its source trial QC, and recompute fixed-window reproducibility and matched excess energy.",
        [
            code(
                """
import json
from _cohort import load_cohort_example, DISPLAY_CHANNEL

# Set ERPY_COHORT_EXAMPLE_DIR in the kernel environment to the authorized
# method-example directory. The loader verifies the exact input checksum.
epochs, trial_qc, provenance = load_cohort_example()
print(f"Python {platform.python_version()}")
print(f"ERPy {ep.__version__}: {epochs.n_trials()} trials, {len(epochs.channels)} channels")
print(json.dumps(provenance, indent=2))
trial_qc.groupby("source_status").size().rename("Trials").to_frame()
"""
            ),
            code(
                """
values = epochs.epochs_df[DISPLAY_CHANNEL].unstack("time").to_numpy()
times = epochs.times
source_retained = trial_qc.source_status.eq("retained").to_numpy()
retained_values = values[source_retained]
config = ep.CRPEnergyConfig(
    response_window=(0.015, 0.300), baseline_window=(-0.400, -0.030),
    artifact_interval=(0.0, 0.015), min_clean_trials=8,
    alpha=0.05, correction="none", n_permutations=5_000, random_state=42,
)
result = ep.run_crp_energy_array(
    retained_values, times, channel=DISPLAY_CHANNEL, config=config,
)
pd.Series({
    "Source-retained trials": int(source_retained.sum()),
    "Finite trials used by both components": result.n_trials_clean,
    "Fixed-window p_R": result.p_crp,
    "Matched-energy p_E": result.p_energy,
    "Unadjusted p_joint = max(p_R, p_E)": result.p_joint,
    "Unadjusted component class": result.classification,
    "Matched response/baseline RMS ratio (dB)": result.rms_ratio_db,
    "Response window (s)": result.response_window,
    "Matched baseline window (s)": result.baseline_window,
}, name="Selected recording results").to_frame()
"""
            ),
            code(
                """
# The source export retains the original processed amplitudes. Center each
# trial over the declared prestimulus interval for the waveform display only.
baseline = (times >= config.baseline_window[0]) & (times <= config.baseline_window[1])
display_values = retained_values - retained_values[:, baseline].mean(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].plot(times * 1000, display_values.T, color="0.6", alpha=0.24, linewidth=0.6)
mean = display_values.mean(axis=0)
sem = display_values.std(axis=0, ddof=1) / np.sqrt(len(display_values))
axes[0].plot(times * 1000, mean, color="#2455A4", linewidth=2, label="Mean")
axes[0].fill_between(times * 1000, mean-sem, mean+sem, color="#2455A4", alpha=0.2, label="SEM")
axes[0].legend()
axes[0].set(xlabel="Time from stimulation (ms)", ylabel="Amplitude (µV)", title="Recorded trials and mean ± SEM")
limit = np.quantile(np.abs(display_values), 0.99)
heat = axes[1].imshow(display_values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                      extent=[times[0]*1000, times[-1]*1000, 0.5, len(display_values)+0.5])
axes[1].set(xlabel="Time from stimulation (ms)", ylabel="Example trial", title="Source-retained trial waveforms")
fig.colorbar(heat, ax=axes[1], label="Amplitude (µV)")
for ax in axes:
    ax.axvspan(0, 15, color="0.5", alpha=0.16)
    ax.axvline(0, color="0.3", linewidth=0.8, linestyle="--")
fig.suptitle("Actual cohort recording: ACC stimulation → subgenual cingulate")
plt.show()
"""
            ),
            md(
                """
Both components use the same finite source-retained trials. The `p_crp` field
in this result stores the fixed-window whole-trial reproducibility value
$p_R$; matched excess energy is assessed after separately demeaning response
and baseline. Their maximum is the unadjusted conjunction value.

This export contains one previously selected recording contact. Its original
selection favored clear method examples, so it cannot estimate detection
accuracy or response prevalence. We do not reconstruct the original
acquisition-wide testing family, report a family-adjusted response call, or
claim a new cohort analysis here. The 15–300 ms computation above is a fresh
ERPy analysis of the exported trials; it is distinct from the generating
analysis's selected-duration CRP comparator and 15–350 ms energy example.
"""
            ),
            code(
                """
# Inspect matched, separately demeaned trial energies used by the component.
response = (times >= result.response_window[0]) & (times <= result.response_window[1])
matched_baseline = (times >= result.baseline_window[0]) & (times <= result.baseline_window[1])
used = retained_values[result.clean_trial_indices]
def demeaned_rms(segment):
    return np.sqrt(np.mean((segment - segment.mean(axis=1, keepdims=True)) ** 2, axis=1))
baseline_rms = demeaned_rms(used[:, matched_baseline])
response_rms = demeaned_rms(used[:, response])
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
axes[0].scatter(baseline_rms, response_rms, color="#2455A4", alpha=0.8)
maximum = max(baseline_rms.max(), response_rms.max()) * 1.05
axes[0].plot([0, maximum], [0, maximum], "--", color="0.5", label="Equal RMS")
axes[0].set(xlabel="Matched baseline RMS (µV)", ylabel="Response RMS (µV)", title="Trial-level paired energy")
axes[0].legend()
p_values = [result.p_crp, result.p_energy, result.p_joint]
axes[1].bar(["p_R", "p_E", "p_joint"], -np.log10(p_values), color=["#2455A4", "#D28B26", "#30856B"])
axes[1].axhline(-np.log10(config.alpha), linestyle="--", color="0.35", label="Unadjusted α = 0.05")
axes[1].set(ylabel="−log10(p)", title="Single-contact unadjusted evidence")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
plt.show()

fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
ax.plot(result.canonical_waveform_times*1000, result.canonical_waveform, color="#30856B")
ax.set(xlabel="Time from stimulation (ms)", ylabel="Normalized canonical amplitude",
       title=f"Descriptive CRP waveform: duration {result.response_duration*1000:.1f} ms")
plt.show()
"""
            ),
            md(
                """
The gray interval marks the declared 0–15 ms exclusion. The source trial QC
and processed signal are reused from the generating cohort analysis; this
notebook does not repeat continuous-data filtering, event extraction, or
artifact detection. The canonical waveform is a descriptive view and does
not select the primary fixed inference window. Synthetic visualization
fixtures follow in notebooks 01–06, and the independent public OpenNeuro
portability example remains in notebook 07.
"""
            ),
        ],
        data_note=(
            "The stored figures are derived from an actual deidentified CNS/ACC–PAG "
            "cohort recording. Restricted trial data are not distributed. Reexecution "
            "requires the authorized local source export via ERPY_COHORT_EXAMPLE_DIR; "
            "there is no synthetic or public-dataset substitution. Display labels are anatomical."
        ),
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
# Recompute source-native quantities on this synthetic example. The explicit
# permutation count keeps the teaching run manageable.
RUN_FULL_DIAGNOSTIC = True
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
# GitHub cannot run Plotly JavaScript. Save faithful PNG snapshots of the
# generated figures; call interactive.show() or animated.show() in Jupyter
# to rotate the connectome or play the complete animation.
interactive = viz.plot_interactive_connectome(
    edges, metadata, show_brain_shell=False, validate_mni=False
)
animated = viz.plot_evoked_response_graph(
    epochs, metadata, stim_pair="STIM_A_1_2", channels=epochs.channels,
    show_brain_shell=False, validate_mni=False, top_n=6, frame_step_ms=40
)
print(type(interactive).__name__, len(animated.frames), "animation frames")
show_plotly_snapshot(interactive)

# Show a response-period frame, with the original animation data and layout.
import plotly.graph_objects as go
frame_index = min(3, len(animated.frames) - 1)
frame = animated.frames[frame_index]
snapshot = go.Figure(animated)
for index, trace in zip(frame.traces or range(len(frame.data)), frame.data):
    snapshot.data[index].update(trace)
snapshot.update_layout(sliders=[], updatemenus=[], title=f"Synthetic response animation — frame {frame.name}")
show_plotly_snapshot(snapshot)
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
RUN_AGGREGATE_RECIPE = True
if RUN_AGGREGATE_RECIPE:
    runs = [{
        "epochs": epochs,
        "stim_pair": "STIM_A_1_2",
        "detections": make_detection_table(epochs),
    }]
    aggregate = viz.plot_aggregate_evoked_response_graph(
        runs, metadata, show_brain_shell=False, validate_mni=False
    )
    print(len(aggregate.frames), "aggregate animation frames")
    show_plotly_snapshot(aggregate)
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
        "Run the public-data portability workflow and display a compact detector audit and signal figure directly on GitHub.",
        [
            md(
                """
The Miller–Müller–Hermes basis-profile dataset is available as OpenNeuro
[`ds003708`](https://openneuro.org/datasets/ds003708/versions/1.0.4), snapshot
v1.0.4. The cells below download a bounded interval spanning 17 stimulation
events, keep at most 32 channels, and run ERPy's public validation pipeline.
The recipe pins explicit S3 object versions and verifies expected metadata
checksums and the previously recorded signal-window checksum before analysis.
The source contract is in `validation/public_ds003708_source_manifest.json`.
It records the snapshot commit and the scope of verification: the full 2.9 GB
recording is not downloaded or rehashed by this bounded recipe.

This is a real public recording. Notebooks 01–06 use synthetic fixtures;
notebook 00 uses an access-controlled cohort illustration. This public recipe
is a portability demonstration, with no
adjudicated response ground truth. Its calls are not sensitivity/specificity
estimates or clinical interpretations. The full local workspace is ignored by
Git; only the compact outputs below are committed. Install the `validation`
extra before running this notebook. An initial run requires network access.
"""
            ),
            code(
                """
import hashlib
import json
from validation.validate_public_spes import DATASETS, run_dataset

workspace = REPOSITORY / "validation" / "public_spes_work_notebook"
source_manifest = json.loads(
    (REPOSITORY / "validation" / "public_ds003708_source_manifest.json").read_text()
)
summaries = run_dataset(
    DATASETS["ds003708"], out_root=workspace,
    min_events=8, max_events=17, max_sites=1, max_channels=32,
    make_figures=False, prefer_cache=False, source_manifest=source_manifest,
)
summary = summaries[0]
site_root = Path(summary["workspace"])
raw_window = next((site_root / "raw").glob("*.csv"))
provenance = {
    "dataset": summary["dataset_id"],
    "source": summary["source_note"],
    "events_url": summary["source_events_url"],
    "signal_url": summary["source_eeg_url"],
    "downloaded_window_sha256": hashlib.sha256(raw_window.read_bytes()).hexdigest(),
    "source_sample_range": [summary["source_start_sample"], summary["source_stop_sample"]],
    "range_bytes_downloaded": summary["range_bytes_downloaded"],
    "source_window_cache_used": summary.get("source_window_cache_used", False),
    "source_identity_verified": summary["source_identity_verified"],
    "snapshot_git_commit": summary["source_snapshot_commit"],
    "downloaded_range_sha256": summary["source_range_sha256"],
    "ERPy_version": ep.__version__,
}
print(json.dumps(provenance, indent=2))
"""
            ),
            code(
                """
summary_keys = [
    "stim_pair", "n_source_events_used", "sfreq", "n_epochs",
    "n_epochs_after_artifact_qc", "n_flagged_artifact_responses",
    "post_artifact_anchor_median_ms", "primary_epoch_passband_hz",
    "crowther_gamma_available",
]
show_output(pd.Series({key: summary[key] for key in summary_keys}, name="Run summary").to_frame())
public_detections = pd.read_csv(site_root / "detections.csv")
primary = public_detections.loc[public_detections.method.eq("crp_energy")].copy()
columns = [
    "channel", "p_crp", "p_energy", "p_joint", "q_joint",
    "primary_classification", "primary_significant",
]
print(int(primary.primary_significant.sum()), "primary response calls across", len(primary), "retained channels")
primary[columns].head(12)
"""
            ),
            md(
                """
The retained `p_crp` field on a `crp_energy` row is the fixed-window
reproducibility value $p_R$. The final primary call requires both components,
family adjustment, and artifact eligibility. SIGNI/high gamma is explicitly
unavailable because this public pipeline uses 0.5–80 Hz epochs without a
separate 70–170 Hz input. The agreement figure below includes available
methods only; the unavailable SIGNI row is excluded.
"""
            ),
            code(
                """
available_detections = public_detections.loc[public_detections.method.ne("crowther_gamma")]
fig = viz.plot_detection_summary(available_detections, top_n=16)
fig.suptitle("Public ds003708: available detector audit (one stimulation site)", y=1.02)
plt.show()

# Rebuild the same epoch stream for a waveform view from the downloaded data.
patient = ep.Patient("DS003708", config_path=str(site_root / "config.yaml"))
steps = [
    ("reject_bad_channels", {"bad_channels": "auto", "method": "drop", "detection_params": {"zscore_threshold": 8.0}}),
    ("artifact_blank", {"width_s": 0.004}),
    ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
    ("bandpass_filter", {"lowcut": 0.5, "highcut": 80.0, "order": 3}),
]
public_epochs = patient.epoch(
    "A", summary["stim_pair"], pipeline=steps,
    tmin=-0.5, tmax=0.6, baseline=(-0.5, -0.03), cache=False,
)
report = public_epochs.flag_artifacts(zscore_threshold=3.0)
public_clean = public_epochs.reject_artifacts(report=report, mode="nan_response")
eligible = primary.loc[primary.channel.isin(public_clean.channels)].sort_values("peak_amplitude_uv", ascending=False)
channel = eligible.channel.iloc[0]
fig = viz.plot_summary_panel(public_clean, channel, detections=public_detections)
fig.suptitle(f"Public ds003708: {summary['stim_pair']} → {channel}", y=1.02)
plt.show()
"""
            ),
        ],
        data_note=(
            "The stored tables and figures are derived exclusively from the public "
            "OpenNeuro ds003708 dataset. Raw downloads stay outside version control. "
            "Run the Python cells from top to bottom to reproduce the analysis."
        ),
    ),
}


NOTEBOOKS["08_n1_development.ipynb"] = notebook(
    "Public negative-N1 development model: a held-out participant",
    "Apply a portable N1 annotation classifier to one actual ds004774 record, verify its saved held-out prediction, and inspect its feature contributions.",
    [
        md(
            """
This optional classifier targets early **negative N1 annotations in cortical-surface
ECoG**. Default ERPy remains polarity-invariant; its p/q values and calls are
unchanged. A logistic N1 score is **not a p value, q value or established
out-of-population probability**, and this model is not validated for sEEG.

The source is CC0 [OpenNeuro ds004774 v1.0.0](https://doi.org/10.18112/openneuro.ds004774.v1.0.0).
We use frozen features and matching inference from recorded signals, with no
synthetic substitution or invented waveform. No raw download or scikit-learn
is required. Install ERPy plus Jupyter to run these cells.

The extension was developed after the original evaluation was examined.
Participant-held-out predictions are development cross-validation, not a new
untouched external validation. A deterministic identifier-join correction
required a full rerun after preliminary outcomes had been seen; the correction
receipt and both authentic protocol revisions accompany the final evidence.
See `docs/N1_DEVELOPMENT.md` for training and reproduction.
"""
        ),
        code(
            """
from ERPy.n1 import N1MorphologyModel, n1_feature_vector
from validation.verify_n1_artifacts import verify
from validation.verify_n1_input_contract import verify as verify_input_contract
from validation.n1_frozen_inputs import FrozenN1Inputs
from validation.score_erdetect_validation import canonicalize, require_unique

N1_ROOT = REPOSITORY / "validation" / "frozen_results" / "n1"
print(verify(N1_ROOT))  # Verifies every packaged artifact and the shared inference table.
manifest = json.loads((N1_ROOT / "artifact_manifest.json").read_text())
entries = {item["path"]: item for group in ("files", "external_inputs") for item in manifest[group]}

# Require the current checked-interface replay, bound to these exact source files.
contract_output = REPOSITORY / "validation/n1_input_contract"
print(verify_input_contract(contract_output))
contract_receipt = json.loads((contract_output / "verification.json").read_text())
assert contract_receipt["changed_calls"] == 0
assert contract_receipt["individual_prediction_checks"] == 128192
checked_inputs = FrozenN1Inputs(N1_ROOT)

features = canonicalize(pd.read_csv(N1_ROOT / "features/n1_features.csv.gz"))
KEY = ["subject", "stimpair", "channel"]
require_unique(features, KEY, "public N1 features")
record_columns = KEY + [
    "configuration", "scoring_eligible", "evaluable", "archived_record_present",
    "archived_pair_comparable", "reference_positive", "development_overlap",
    "epoch_sha256", "p_reproducibility", "p_energy", "p_joint", "q_joint",
    "rms_ratio_db", "detected", "n_trials_total", "n_trials_clean", "random_seed",
]
records = pd.read_csv(
    N1_ROOT / "../erdetect/record_predictions_and_labels.csv.gz",
    usecols=record_columns, low_memory=False, dtype={"random_seed": "string"},
)
print(f"ERPy {ep.__version__}; Python {platform.python_version()}")
print(f"{len(features):,} recorded contact/pair feature rows; all source identities verified")
"""
        ),
        md(
            """
Selection rule: use the **first lexicographic feature-available MAYO02 record**
in the frozen early-window paired evaluation frame. Eligibility includes a
reference being present; its value, model score and detector call do not select
the record. The join uses the scorer's exact undirected pair convention and
requires matching epoch hashes. Identifier columns locate the record and are
never model features.
"""
        ),
        code(
            """
eligible = (
    records.configuration.eq("early_10_90ms") & records.scoring_eligible
    & records.evaluable & records.archived_record_present
    & records.archived_pair_comparable & records.reference_positive.notna()
    & ~records.development_overlap
)
frame = canonicalize(records.loc[eligible]).merge(
    features, on=KEY, how="left", validate="one_to_one", suffixes=("", "_feature"),
)
assert len(frame) == 32048 and frame.subject.nunique() == 13
assert frame.feature_available.all()
assert frame.epoch_sha256.eq(frame.epoch_sha256_feature).all()
row = frame.loc[frame.subject.eq("MAYO02") & frame.feature_available].sort_values(KEY).iloc[0]
feature_record, inference = checked_inputs.prepare(
    checked_inputs.features.loc[tuple(row[key] for key in KEY)], row.to_dict(),
)
print("Trial provenance:", feature_record["_n1_input_provenance"]["trial_provenance_scope"])
print("Selected public record:", " / ".join(str(row[key]) for key in KEY))
print("Feature contract:", row.feature_version, row.voltage_unit, row.feature_config_sha256)
show_output(row[["n_feature_trials", "negative_peak_uv", "peak_latency_ms", "peak_width_ms",
                 "trial_negative_fraction", "trial_cosine"]].rename("Recorded feature").to_frame())
"""
        ),
        code(
            """
model_path = N1_ROOT / f"nested/N1_logistic_hybrid_heldout_{row.subject}.json"
model = N1MorphologyModel.load(model_path)
assert row.subject in model.provenance["excluded_subjects"]
assert row.subject not in model.provenance["training_subjects"]
assert len(model.provenance["training_subjects"]) == 12
assert model.provenance["feature_manifest_sha256"] == entries["features/feature_manifest.json"]["sha256"]
assert model.provenance["run_protocol_sha256"] == entries["nested/run_protocol.json"]["sha256"]
result = model.predict(feature_record, inference)

# Compare with the existing out-of-fold score; no fitting or threshold selection.
saved = pd.read_csv(N1_ROOT / "nested/out_of_fold_predictions.csv.gz")
match = saved.loc[(saved[KEY] == row[KEY]).all(axis=1)]
assert len(match) == 1
expected = match.iloc[0]
assert np.isclose(result["score"], expected.N1_logistic_hybrid_score, rtol=0, atol=1e-12)
assert result["detected"] == bool(expected.N1_logistic_hybrid_detected)
assert np.isclose(model.threshold, expected.N1_logistic_hybrid_threshold, rtol=0, atol=1e-12)

# Save/load retains the schema, coefficient values, threshold and training provenance.
with TemporaryDirectory(prefix="erpy-n1-example-") as directory:
    copy_path = Path(directory) / "heldout_model.json"
    model.save(copy_path)
    restored = N1MorphologyModel.load(copy_path)
    assert asdict(restored) == asdict(model)
    assert restored.predict(feature_record, inference) == result

# A unit mismatch fails instead of silently rescaling a fitted model's inputs.
try:
    model.predict({**feature_record, "voltage_unit": "V"}, inference)
except ValueError:
    print("Unit-contract mismatch correctly rejected; save/load prediction matched")
else:
    raise AssertionError("Model accepted a mismatched voltage unit")

show_output(pd.DataFrame({"Quantity": ["N1 development score", "Training-selected N1 threshold",
    "N1 development call", "Frozen general-detector p_R", "Frozen general-detector p_E",
    "Frozen general-detector q_joint", "Frozen general-detector call"],
    "Value": [result["score"], model.threshold, result["detected"], row.p_reproducibility,
              row.p_energy, row.q_joint, row.detected]}))
print("Prediction matched the saved participant-held-out result; no model was fitted here")
print("Measured N1 peak latency (ms):", result["measured_peak_latency_ms"])
print("Model latency was imputed:", result["peak_latency_imputed"])
"""
        ),
        md(
            """
The hybrid adds reproducibility/energy/RMS inference features to morphology and
trial-consistency features. The display below decomposes this single model's
logit after **training-only** standardization. These signed terms sum with its
intercept to the logit; they are not causal effects or independent feature
importance estimates. The correlated descriptors can share information.
"""
        ),
        code(
            """
vector = n1_feature_vector(feature_record, inference.validated_values(feature_record))
contributions = ((vector - np.asarray(model.mean)) / np.asarray(model.scale)) * model.coefficients
logit = model.intercept + contributions.sum()
assert np.isclose(1 / (1 + np.exp(-logit)), result["score"], rtol=0, atol=1e-12)
labels = ["Negative peak / baseline SD", "Negative prominence / baseline SD", "Peak latency",
          "Half-prominence width", "Negative area fraction", "Negative-trial fraction",
          "Trial cosine consistency", "Preceding positive / baseline SD", "Response RMS / baseline SD",
          "Mean-waveform baseline SD", "Absolute negative peak", "Finite-trial count",
          "Reproducibility evidence", "Energy evidence", "RMS ratio (dB)"]
order = np.argsort(np.abs(contributions))[::-1]
fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
ax.barh(np.arange(len(order)), contributions[order],
        color=np.where(contributions[order] >= 0, "#176B87", "#A34A28"))
ax.set_yticks(np.arange(len(order)), np.asarray(labels)[order], fontsize=10)
ax.invert_yaxis()
ax.axvline(0, color="0.25", linewidth=1)
ax.set_xlabel("Signed contribution to the N1 model logit", fontsize=11)
ax.set_title(f"Held-out public record: {row.subject}, {row.stimpair}, {row.channel}\\n"
             f"Score {result['score']:.3f}; threshold {model.threshold:.3f}; intercept {model.intercept:.3f}",
             fontsize=12, pad=14)
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)
plt.show()
"""
        ),
        md(
            """
For new trial data, call `prepare_n1_hybrid_inputs(trials_uv, times_seconds)`
and pass the returned pair to `model.predict(*inputs)`. It constructs morphology
and trained 10–90 ms inference from the same arrays, retains their distinct
finite-trial masks, and rejects unavailable inference. Hybrid `predict` rejects
bare dictionaries because they do not establish window or trial identity.
The checked frozen-record adapter above preserves the explicit limitation that
historical clean-trial indices were not archived. Check
`available` and keep a processing no-call separate from physiological absence.
Use `measured_peak_latency_ms` for reported timing; it is unavailable when no
interior negative peak exists. `peak_latency_imputed` flags the midpoint
feature used by the classifier in that case; it is not a measured latency.
Inspect training provenance before interpreting a model on a participant.
The full-cohort `development_model` files are for further development/testing;
they must not be used to claim held-out accuracy on their training participants.

The frozen dataset license, exact checksums, all candidate results, model files,
source snapshots and reproduction commands are linked in
`validation/frozen_results/n1/README.md` and `docs/N1_DEVELOPMENT.md`.
"""
        ),
    ],
    data_note=("All features in this notebook derive from actual public OpenNeuro ds004774 recordings. "
               "The example model excluded its participant from fitting and threshold selection. "
               "Run cells from top to bottom; input and model checksums are verified before use."),
    setup=code(
        """
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display as show_output

HERE = Path.cwd()
REPOSITORY = HERE if (HERE / "ERPy").is_dir() else HERE.parents[1]
assert (REPOSITORY / "validation" / "frozen_results" / "n1").is_dir()
sys.path.insert(0, str(REPOSITORY.resolve()))
import ERPy as ep

get_ipython().run_line_magic("matplotlib", "inline")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "savefig.bbox": "tight"})
"""
    ),
)


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
