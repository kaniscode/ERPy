"""ERPy: stimulation-evoked intracranial response detection and analysis.

The usual entry point is :class:`Patient`::

    import ERPy as ep

    patient = ep.Patient("EXAMPLE_PATIENT", config_path="config.yaml")
    result = patient.analyze("EXAMPLE_SESSION", "STIM_A_1_2")
    result.primary_significant

The names exported here follow the workflow:

* **Projects and data** - :class:`Patient`, :class:`Recording`,
  :class:`DataLoader`, BIDS/NWB import, continuous-signal storage.
* **Events and preprocessing** - :class:`Events`, :class:`Pipeline`,
  bad-channel screening, artifact anchors.
* **Epochs and quality control** - :class:`Epochs`, trial-by-channel artifact
  QC, :func:`audit_waveforms`.
* **Response detection** - :func:`detect`, CRP and CRP-energy, statistical
  inference helpers.
* **Spectral, connectivity, and network analysis** - time-frequency and
  coupling measures, directed graph metrics, travelling waves.

Plotting lives in :mod:`ERPy.viz` and is reachable from ``epochs.plot``.
"""

from ._version import CHECKPOINT_COMPATIBILITY_ID, __version__
from .analysis import Analysis
from .artifact_anchor import (
    POST_ARTIFACT_ZERO,
    artifact_envelope,
    detect_post_artifact_anchor,
    detect_zero_time_artifact,
)
from .bad_channels import (
    DEFAULT_HARD_ARTIFACT_REASONS,
    ArtifactResponseReport,
    BadChannelReport,
    detect_artifactual_responses,
    detect_bad_channels,
    reject_artifactual_responses,
    reject_bad_channels,
    response_artifact_summary,
)
from .bids import BIDSImportResult, discover_bids_ieeg, import_bids_project
from .connectivity import (
    canonicalize_connectivity_edges,
    coherence_matrix_from_dataframe,
    connectivity_matrix_from_dataframe,
    dataframe_time_window,
    granger_causality_matrix_from_dataframe,
    multitaper_coherence_matrices_from_dataframe,
    mutual_information_matrix_from_dataframe,
    plv_matrix_from_dataframe,
    post_stim_rest_window,
)
from .coordinates import complete_contact_coordinates
from .crp import CRPConfig, CRPResult, compare_crp_across_stim_sites, run_crp, run_crp_all
from .crp_energy import (
    CRP_ENERGY_DETECTOR_VERSION,
    CRPEnergyConfig,
    CRPEnergyResult,
    FixedWindowReproducibilityResult,
    fixed_window_reproducibility_sign_flip_test,
    paired_log_rms_sign_flip_test,
    run_crp_energy_array,
)
from .dataloader import DataLoader
from .epochs import Epochs, EpochsCore, UnsafeLegacyEpochMetadataError
from .erp_detection import (
    ALL_METHODS,
    DETECTOR_DEFINITIONS,
    DETECTOR_QUANTITY_COLUMNS,
    DETECTOR_REFERENCES,
    PRIMARY_CRITERION,
    detect,
)
from .events import (
    EventDetectionError,
    Events,
    create_events_from_binary_series,
    create_events_from_timestamps,
    detect_events_from_artifacts,
    evoked_to_datetime,
    process_events_for_session,
)
from .facade import AnalysisResult, EpochsResult, Patient, Recording, Session, Study
from .graph_metrics import (
    bidirectional_edge_summary,
    compute_dynamic_graph_metrics,
    dynamic_target_distribution_metrics,
    evoked_graph_metric_timecourse,
    evoked_response_edges_over_time,
    opportunity_conditioned_reciprocity,
    weighted_directed_node_metrics,
)
from .inference import (
    BootstrapInterval,
    SignFlipResult,
    exact_sign_flip_test,
    fdr_bh,
    hierarchical_bootstrap_interval,
)
from .pipeline import STANDARD_PIPELINES, Pipeline, UnsafeLegacyProcessedMetadataError
from .response_metrics import zscore_metric_within_stim
from .spectral import (
    DEFAULT_BANDS,
    ComodulogramResult,
    EpochsSpectral,
    PACResult,
    PLVResult,
    SpectralConfig,
    TFRResult,
    band_power,
    coherence_matrix,
    compute_ersp,
    compute_psd,
    compute_tfr,
    granger_causality_matrix,
    inter_trial_phase_coherence,
    mutual_information_matrix,
    phase_amplitude_comodulogram,
    phase_amplitude_coupling,
    phase_locking_value,
    phase_phase_coupling,
    phase_phase_matrix,
    plv_matrix,
)
from .storage import load_continuous_npz, resample_continuous, save_continuous_npz
from .traveling_waves import (
    DEFAULT_SCAFFOLD_PATTERN,
    estimate_latency_gradient,
    latency_distance_permutation_test,
    latency_gradient_permutation_test,
    latency_wavefront_table,
    scaffold_region_mask,
)
from .waveform_qc import (
    WaveformAudit,
    audit_waveforms,
    common_mode_recovery_diagnostic,
)

__all__ = [
    "__version__",
    "CHECKPOINT_COMPATIBILITY_ID",
    # Projects and data
    "Patient",
    "Recording",
    "Session",
    "Study",
    "DataLoader",
    "Analysis",
    "AnalysisResult",
    "BIDSImportResult",
    "discover_bids_ieeg",
    "import_bids_project",
    "complete_contact_coordinates",
    "load_continuous_npz",
    "resample_continuous",
    "save_continuous_npz",
    # Events and preprocessing
    "Events",
    "EventDetectionError",
    "create_events_from_binary_series",
    "create_events_from_timestamps",
    "detect_events_from_artifacts",
    "evoked_to_datetime",
    "process_events_for_session",
    "Pipeline",
    "STANDARD_PIPELINES",
    "UnsafeLegacyProcessedMetadataError",
    "BadChannelReport",
    "detect_bad_channels",
    "reject_bad_channels",
    "POST_ARTIFACT_ZERO",
    "artifact_envelope",
    "detect_post_artifact_anchor",
    "detect_zero_time_artifact",
    # Epochs and quality control
    "Epochs",
    "EpochsCore",
    "EpochsResult",
    "UnsafeLegacyEpochMetadataError",
    "ArtifactResponseReport",
    "DEFAULT_HARD_ARTIFACT_REASONS",
    "detect_artifactual_responses",
    "reject_artifactual_responses",
    "response_artifact_summary",
    "WaveformAudit",
    "audit_waveforms",
    "common_mode_recovery_diagnostic",
    # Response detection
    "detect",
    "ALL_METHODS",
    "PRIMARY_CRITERION",
    "DETECTOR_DEFINITIONS",
    "DETECTOR_QUANTITY_COLUMNS",
    "DETECTOR_REFERENCES",
    "CRPConfig",
    "CRPResult",
    "run_crp",
    "run_crp_all",
    "compare_crp_across_stim_sites",
    "CRPEnergyConfig",
    "CRPEnergyResult",
    "CRP_ENERGY_DETECTOR_VERSION",
    "FixedWindowReproducibilityResult",
    "run_crp_energy_array",
    "fixed_window_reproducibility_sign_flip_test",
    "paired_log_rms_sign_flip_test",
    "BootstrapInterval",
    "SignFlipResult",
    "exact_sign_flip_test",
    "fdr_bh",
    "hierarchical_bootstrap_interval",
    "zscore_metric_within_stim",
    # Spectral analysis
    "EpochsSpectral",
    "SpectralConfig",
    "DEFAULT_BANDS",
    "TFRResult",
    "PLVResult",
    "PACResult",
    "ComodulogramResult",
    "compute_psd",
    "band_power",
    "compute_tfr",
    "compute_ersp",
    "inter_trial_phase_coherence",
    "phase_locking_value",
    "plv_matrix",
    "coherence_matrix",
    "granger_causality_matrix",
    "mutual_information_matrix",
    "phase_amplitude_coupling",
    "phase_amplitude_comodulogram",
    "phase_phase_coupling",
    "phase_phase_matrix",
    # Connectivity from continuous recordings
    "canonicalize_connectivity_edges",
    "connectivity_matrix_from_dataframe",
    "coherence_matrix_from_dataframe",
    "multitaper_coherence_matrices_from_dataframe",
    "plv_matrix_from_dataframe",
    "granger_causality_matrix_from_dataframe",
    "mutual_information_matrix_from_dataframe",
    "dataframe_time_window",
    "post_stim_rest_window",
    # Network analysis and travelling waves
    "bidirectional_edge_summary",
    "compute_dynamic_graph_metrics",
    "dynamic_target_distribution_metrics",
    "evoked_graph_metric_timecourse",
    "evoked_response_edges_over_time",
    "opportunity_conditioned_reciprocity",
    "weighted_directed_node_metrics",
    "DEFAULT_SCAFFOLD_PATTERN",
    "estimate_latency_gradient",
    "latency_distance_permutation_test",
    "latency_gradient_permutation_test",
    "latency_wavefront_table",
    "scaffold_region_mask",
]
