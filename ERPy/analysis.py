from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .crp import CRPConfig, compare_crp_across_stim_sites, run_crp, run_crp_all
from .graph_metrics import compute_dynamic_graph_metrics, evoked_graph_metric_timecourse, evoked_response_edges_over_time
from .response_metrics import zscore_metric_within_stim
from .spectral import (
    SpectralConfig,
    band_power,
    coherence_matrix,
    compute_psd,
    compute_tfr,
    phase_amplitude_comodulogram,
    phase_amplitude_coupling,
    phase_locking_value,
    plv_matrix,
)
from .stats_utils import compare_groups
from .utils.jobrunner import JobRunner
from .viz.networks import (
    adjacency_from_edges,
    edges_from_detections,
    graph_from_edges,
    graph_from_matrix,
    plot_adjacency_heatmap,
    plot_aggregate_evoked_response_graph,
    plot_coordinate_network,
    plot_evoked_response_graph,
    plot_glass_brain_network,
    plot_node_metric_glass_brain,
    plot_node_metric_template_brain,
    plot_interactive_connectome,
    plot_network,
    plot_ordered_adjacency_heatmap,
    plot_response_matrix_heatmap,
    plot_template_brain_network,
    response_matrix_from_edges,
)
from .viz.crp import plot_crp_response_contrast, plot_crp_score_map, plot_crp_site_comparison, plot_crp_summary
from .viz.detections import plot_detection_summary
from .viz.spectral import (
    plot_comodulogram,
    plot_connectivity_matrix,
    plot_pac,
    plot_plv_timefreq,
    plot_psd,
    plot_spectral_summary,
    plot_tfr,
)
from .viz.response_metrics import plot_within_stim_zscore_bars, plot_within_stim_zscore_heatmap
from .viz.graph_metrics import plot_graph_metric_heatmap, plot_graph_metric_timecourse
from .viz.waveforms import plot_grid, plot_heatmap, plot_mean, plot_ranked_grid


class Analysis:
    """High-level analysis and visualization helpers bound to a pipeline."""

    PYTHON_SCRIPT: str | None = None
    BASH_SCRIPT: str | None = None

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def run_crp(self, epochs, channel: str, config: CRPConfig | None = None):
        """Fit canonical response parameterization for one channel."""

        return run_crp(epochs, channel, config=config)

    def run_crp_from_epochs(self, epochs, config: CRPConfig | None = None) -> pd.DataFrame:
        """Fit canonical response parameterization for every epoch channel."""

        return run_crp_all(epochs, config=config)

    def run_crp_from_file(self, path: str | Path, config: CRPConfig | None = None) -> pd.DataFrame:
        """Load an epoch file and run canonical response analysis on all channels."""

        epochs = self.pipeline.load_epochs_file(path)
        return self.run_crp_from_epochs(epochs, config=config)

    def compare_crp_across_stim_sites(self, crp_tables, **kwargs) -> pd.DataFrame:
        """Compare CRP metrics across stimulation sites."""

        return compare_crp_across_stim_sites(crp_tables, **kwargs)

    def zscore_metric_within_stim(self, data: pd.DataFrame, metric: str, **kwargs) -> pd.DataFrame:
        """Standardize a response metric within each stimulation condition."""

        return zscore_metric_within_stim(data, metric, **kwargs)

    def evoked_response_edges_over_time(self, runs: list[dict], **kwargs) -> pd.DataFrame:
        """Build time-resolved directed response edges from epoch runs."""

        return evoked_response_edges_over_time(runs, elec_meta=self.pipeline.dataloader.elec_meta, **kwargs)

    def compute_dynamic_graph_metrics(self, edge_time_table: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute graph metrics for successive bins of an edge-time table."""

        return compute_dynamic_graph_metrics(edge_time_table, **kwargs)

    def evoked_graph_metric_timecourse(self, runs: list[dict], **kwargs) -> pd.DataFrame:
        """Derive a graph-metric time course directly from epoch runs."""

        return evoked_graph_metric_timecourse(runs, elec_meta=self.pipeline.dataloader.elec_meta, **kwargs)

    def plot_crp_summary(self, epochs, channel: str, config: CRPConfig | None = None, **kwargs):
        """Plot the canonical response fit and diagnostics for one channel."""

        return plot_crp_summary(epochs, channel=channel, config=config, **kwargs)

    def plot_crp_score_map(self, crp_table: pd.DataFrame, **kwargs):
        """Plot channel-level CRP scores and return the figure."""

        ax = plot_crp_score_map(crp_table, **kwargs)
        return ax.figure

    def plot_crp_site_comparison(self, comparison_table: pd.DataFrame, **kwargs):
        """Plot a CRP comparison across stimulation sites."""

        ax = plot_crp_site_comparison(comparison_table, **kwargs)
        return ax.figure

    def plot_within_stim_zscore_heatmap(self, data: pd.DataFrame, metric: str, **kwargs):
        """Plot within-stimulation metric z-scores as a heat map."""

        ax = plot_within_stim_zscore_heatmap(data, metric, **kwargs)
        return ax.figure

    def plot_within_stim_zscore_bars(self, data: pd.DataFrame, metric: str, **kwargs):
        """Plot within-stimulation metric z-scores as bars."""

        ax = plot_within_stim_zscore_bars(data, metric, **kwargs)
        return ax.figure

    def plot_graph_metric_timecourse(self, metric_table: pd.DataFrame, **kwargs):
        """Plot a dynamic graph metric against post-stimulus time."""

        ax = plot_graph_metric_timecourse(metric_table, **kwargs)
        return ax.figure

    def plot_graph_metric_heatmap(self, metric_table: pd.DataFrame, **kwargs):
        """Plot dynamic graph metrics as a node-by-time heat map."""

        ax = plot_graph_metric_heatmap(metric_table, **kwargs)
        return ax.figure

    def plot_crp_response_contrast(self, epochs, significant_channel: str, nonsignificant_channel: str, **kwargs):
        """Contrast representative significant and nonsignificant CRP responses."""

        return plot_crp_response_contrast(epochs, significant_channel, nonsignificant_channel, **kwargs)

    # --- Spectral analysis ---
    def compute_psd(self, epochs, channels=None, **kwargs) -> pd.DataFrame:
        """Estimate power spectral density for selected channels."""

        return compute_psd(epochs, channels=channels, **kwargs)

    def band_power(self, epochs, bands=None, channels=None, **kwargs) -> pd.DataFrame:
        """Summarize spectral power within named frequency bands."""

        return band_power(epochs, bands=bands, channels=channels, **kwargs)

    def compute_tfr(self, epochs, channel: str, config: SpectralConfig | None = None, **kwargs):
        """Compute a time-frequency representation for one channel."""

        return compute_tfr(epochs, channel, config=config, **kwargs)

    def phase_locking_value(self, epochs, ch_x: str, ch_y: str, config: SpectralConfig | None = None):
        """Estimate time-frequency phase locking between two channels."""

        return phase_locking_value(epochs, ch_x, ch_y, config=config)

    def plv_matrix(self, epochs, band=(8.0, 13.0), channels=None, **kwargs) -> pd.DataFrame:
        """Compute a band-limited phase-locking matrix."""

        return plv_matrix(epochs, band=band, channels=channels, **kwargs)

    def coherence_matrix(self, epochs, band=(8.0, 13.0), channels=None, **kwargs) -> pd.DataFrame:
        """Compute a band-limited channel coherence matrix."""

        return coherence_matrix(epochs, band=band, channels=channels, **kwargs)

    def phase_amplitude_coupling(self, epochs, phase_channel: str, amp_channel: str | None = None, **kwargs):
        """Estimate phase-amplitude coupling for a channel pair."""

        return phase_amplitude_coupling(epochs, phase_channel, amp_channel, **kwargs)

    def phase_amplitude_comodulogram(self, epochs, phase_channel: str, amp_channel: str | None = None, **kwargs):
        """Compute phase-amplitude coupling across frequency pairs."""

        return phase_amplitude_comodulogram(epochs, phase_channel, amp_channel, **kwargs)

    def plot_psd(self, epochs, channels=None, **kwargs):
        """Plot power spectral density for selected channels."""

        ax = plot_psd(compute_psd(epochs, channels=channels), **kwargs)
        return ax.figure

    def plot_tfr(self, epochs, channel: str, config: SpectralConfig | None = None, **kwargs):
        """Plot the time-frequency representation of one channel."""

        ax = plot_tfr(compute_tfr(epochs, channel, config=config), **kwargs)
        return ax.figure

    def plot_plv(self, epochs, ch_x: str, ch_y: str, config: SpectralConfig | None = None, **kwargs):
        """Plot time-frequency phase locking for a channel pair."""

        ax = plot_plv_timefreq(phase_locking_value(epochs, ch_x, ch_y, config=config), **kwargs)
        return ax.figure

    def plot_connectivity_matrix(self, matrix: pd.DataFrame, **kwargs):
        """Plot a square channel-connectivity matrix."""

        ax = plot_connectivity_matrix(matrix, **kwargs)
        return ax.figure

    def plot_comodulogram(self, epochs, phase_channel: str, amp_channel: str | None = None, plot_kwargs: dict | None = None, **kwargs):
        """Compute and plot a phase-amplitude comodulogram."""

        ax = plot_comodulogram(phase_amplitude_comodulogram(epochs, phase_channel, amp_channel, **kwargs), **(plot_kwargs or {}))
        return ax.figure

    def plot_pac(self, epochs, phase_channel: str, amp_channel: str | None = None, plot_kwargs: dict | None = None, **kwargs):
        """Compute and plot a phase-amplitude coupling estimate."""

        ax = plot_pac(phase_amplitude_coupling(epochs, phase_channel, amp_channel, **kwargs), **(plot_kwargs or {}))
        return ax.figure

    def plot_spectral_summary(self, epochs, channel: str, config: SpectralConfig | None = None):
        """Create a multi-panel spectral summary for one channel."""

        return plot_spectral_summary(epochs, channel, config=config)

    def process(self, analysis_type: str, epochs=None, **kwargs: Any):
        """Dispatch a supported high-level analysis by name."""

        if analysis_type == "crp":
            return self.run_crp(epochs, kwargs["channel"], config=kwargs.get("config"))
        if analysis_type == "crp_epochs":
            return self.run_crp_from_epochs(epochs, config=kwargs.get("config"))
        if analysis_type == "noop":
            return {"ok": True}
        raise ValueError(f"Unknown analysis_type {analysis_type!r}")

    def quick_erp(self, epochs, channel: str, **kwargs):
        """Plot the mean ERP for one channel and return the figure."""

        ax = plot_mean(epochs, channel, **kwargs)
        return ax.figure

    def plot_erp_grid(self, epochs, channels=None, **kwargs):
        """Plot mean ERPs for several channels in a grid."""

        return plot_grid(epochs, channels=channels, **kwargs)[0]

    def plot_ranked_erp_grid(self, epochs, detections: pd.DataFrame, **kwargs):
        """Plot channel ERPs ordered by a detection result table."""

        return plot_ranked_grid(epochs, detections=detections, **kwargs)[0]

    def heatmap_epochs(self, epochs, channel: str, **kwargs):
        """Plot single-trial activity for one channel as a heat map."""

        ax = plot_heatmap(epochs, channel, **kwargs)
        return ax.figure

    def plot_network(self, matrix: pd.DataFrame, threshold: float = 0.0, **kwargs):
        """Threshold an adjacency matrix and plot the resulting network."""

        graph = graph_from_matrix(matrix, threshold=threshold)
        ax = plot_network(graph, **kwargs)
        return ax.figure

    def graph_from_matrix(self, matrix: pd.DataFrame, threshold: float = 0.0):
        """Convert an adjacency matrix to a thresholded NetworkX graph."""

        return graph_from_matrix(matrix, threshold=threshold)

    def edges_from_detections(self, detections: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Convert channel detections into anatomically annotated directed edges."""

        return edges_from_detections(detections, elec_meta=self.pipeline.dataloader.elec_meta, **kwargs)

    def adjacency_from_edges(self, edges: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Build an electrode-level adjacency matrix from an edge table."""

        return adjacency_from_edges(edges, **kwargs)

    def response_matrix_from_edges(self, edges: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Aggregate an electrode edge table into a region response matrix."""

        return response_matrix_from_edges(edges, elec_meta=self.pipeline.dataloader.elec_meta, **kwargs)

    def graph_from_edges(self, edges: pd.DataFrame, **kwargs):
        """Convert a directed edge table to a NetworkX graph."""

        return graph_from_edges(edges, **kwargs)

    def plot_adjacency(self, matrix: pd.DataFrame, **kwargs):
        """Plot an electrode-level adjacency matrix as a heat map."""

        ax = plot_adjacency_heatmap(matrix, **kwargs)
        return ax.figure

    def plot_ordered_adjacency(self, edges: pd.DataFrame, **kwargs):
        """Plot adjacency with electrodes ordered by anatomical metadata."""

        ax = plot_ordered_adjacency_heatmap(edges, elec_meta=self.pipeline.dataloader.elec_meta, **kwargs)
        return ax.figure

    def plot_response_matrix(self, edges: pd.DataFrame, **kwargs):
        """Plot an anatomically aggregated response matrix."""

        ax = plot_response_matrix_heatmap(edges, elec_meta=self.pipeline.dataloader.elec_meta, **kwargs)
        return ax.figure

    def plot_coordinate_network(self, edges: pd.DataFrame, **kwargs):
        """Plot response edges in electrode-coordinate space."""

        ax = plot_coordinate_network(edges, self.pipeline.dataloader.elec_meta, **kwargs)
        return ax.figure

    def plot_template_brain_network(self, edges: pd.DataFrame, **kwargs):
        """Plot response edges on a cortical surface template."""

        return plot_template_brain_network(edges, self.pipeline.dataloader.elec_meta, **kwargs)

    def plot_glass_brain_network(self, edges: pd.DataFrame, **kwargs):
        """Plot response edges in a glass-brain projection."""

        return plot_glass_brain_network(edges, self.pipeline.dataloader.elec_meta, **kwargs)

    def plot_node_metric_glass_brain(self, metric_table, **kwargs):
        """Plot a node metric in a glass-brain projection."""

        return plot_node_metric_glass_brain(metric_table, self.pipeline.dataloader.elec_meta, **kwargs)

    def plot_node_metric_template_brain(self, metric_table, **kwargs):
        """Plot a node metric on a cortical surface template."""

        return plot_node_metric_template_brain(metric_table, self.pipeline.dataloader.elec_meta, **kwargs)

    def plot_detection_summary(self, detections: pd.DataFrame, **kwargs):
        """Summarize detector calls and scores in one figure."""

        return plot_detection_summary(detections, **kwargs)

    def plot_interactive_connectome(self, edges: pd.DataFrame, **kwargs):
        """Create an interactive 3-D connectome from response edges."""

        return plot_interactive_connectome(edges, self.pipeline.dataloader.elec_meta, **kwargs)

    def plot_evoked_response_graph(self, epochs, **kwargs):
        """Plot the directed response graph for one epoch collection."""

        return plot_evoked_response_graph(epochs, self.pipeline.dataloader.elec_meta, **kwargs)

    def plot_aggregate_evoked_response_graph(self, runs: list[dict], **kwargs):
        """Plot an aggregate directed response graph across runs."""

        return plot_aggregate_evoked_response_graph(runs, self.pipeline.dataloader.elec_meta, **kwargs)

    def compare_groups(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Run ERPy's tabular group-comparison helper."""

        return compare_groups(*args, **kwargs)

    def save_figure(self, fig, filename: str, session_id: str | None = None, dpi: int = 300) -> Path:
        """Save a figure in the configured analysis directory and return its path."""

        root = self.pipeline.dataloader.get_analysis_path(session_id)
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    @classmethod
    def install_standard_scripts(cls) -> None:
        """Reset external analysis-script hooks to ERPy's built-in execution."""

        return None

    @classmethod
    def use_standard_scripts(cls, bash_type: str = "simple") -> None:
        """Select ERPy's built-in local analysis execution hooks."""

        cls.PYTHON_SCRIPT = None
        cls.BASH_SCRIPT = None

    def queue_job(self, session_ids: Iterable[str] | None = None, backend: str | None = None, **kwargs: Any):
        """Create an analysis job runner and return its dispatch summary."""

        runner = JobRunner(backend=backend, job_name=kwargs.get("job_name", "erpy_analysis"))
        return runner, "Analysis.queue_job configured; use local notebooks or warmup for full analysis dispatch.", ""


def dispatch_analysis_batch(*args: Any, **kwargs: Any):
    return Analysis(*args, **kwargs)
