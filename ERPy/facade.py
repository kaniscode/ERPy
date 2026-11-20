"""High-level entry points: :class:`Patient` and its per-session :class:`Recording`."""

from __future__ import annotations

import inspect
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._epoch_cache import epoch_cache_identity
from ._version import CHECKPOINT_COMPATIBILITY_ID, __version__
from .analysis import Analysis
from .artifact_anchor import POST_ARTIFACT_ZERO
from .bad_channels import DEFAULT_HARD_ARTIFACT_REASONS, detect_artifactual_responses
from .dataloader import DataLoader
from .defaults import DEFAULT_BASELINE, DEFAULT_PIPELINE, DEFAULT_RESPONSE_WINDOW, DEFAULT_TMAX, DEFAULT_TMIN
from .epochs import Epochs
from .erp_detection import ALL_METHODS
from .events import Events, detect_events_from_artifacts
from .pipeline import Pipeline, PipelineSpec, PipelineSteps

# Re-exported so existing imports from ``ERPy.facade`` keep working.
from .results import (  # noqa: F401
    AnalysisResult,
    _annotate_detection_artifact_qc,
    _metadata_value,
    _result_prefix,
    _significant_rows,
)
from .utils.utils import parse_event_times

EpochsResult = Epochs

#: Epoch options with a named parameter in :meth:`Recording.epoch`.
_EPOCH_OPTION_DEFAULTS: dict[str, Any] = {
    "tmin": DEFAULT_TMIN,
    "tmax": DEFAULT_TMAX,
    "baseline": DEFAULT_BASELINE,
    "zero_time": POST_ARTIFACT_ZERO,
    "artifact_anchor_params": None,
    "cache": True,
    "event_method": "auto",
    "save_epochs": True,
}

#: Extra keyword arguments of :meth:`Recording.epoch` that select or bound the
#: recording to process, and therefore take part in the cache identity.
_PROCESSING_OPTION_NAMES = (
    "input_source",
    "raw_file",
    "stim_start",
    "processing_margin_s",
    "load_from_raw",
    "overwrite",
)

#: Detection options recorded in the audit metadata's ``effective_core``.
_CORE_DETECTION_OPTION_NAMES = frozenset(
    {
        "methods",
        "min_consensus",
        "baseline_window",
        "response_window",
        "response_boundary_guard_s",
        "wideband_epochs",
    }
)

#: Detector-table columns summarized once per method in the audit metadata.
_DETECTOR_PARAMETER_COLUMNS = (
    "detector_version",
    "detector_input",
    "detector_definition",
    "detector_reference",
    "threshold",
    "response_window",
    "baseline_window",
    "parameters",
    "testing_family_size",
    "multiple_comparison_correction",
    "reproducibility_test_exact",
    "reproducibility_n_randomizations",
    "energy_n_permutations",
    "random_state_root",
    "random_state_effective",
    "random_stream_derivation",
    "signi_n_permutations",
    "hays_n1_threshold_z",
    "kundu_envelope_threshold_uv",
)


class Patient:
    """Quick-use public wrapper for one patient.

    A patient bundles the configured :class:`DataLoader`, the preprocessing
    :class:`Pipeline`, event detection, and the analysis helpers. Index it by
    session (``patient["A"]``) to obtain a :class:`Recording`.

    Parameters
    ----------
    patient_id
        Identifier used in the metadata tables.
    config_path
        ERPy YAML configuration. When omitted, ``$ERPY_CONFIG``, the working
        directory and ``~/.erpy`` are searched.
    verbose
        Forwarded to the :class:`DataLoader`.

    Examples
    --------
    >>> patient = ep.Patient("EXAMPLE_PATIENT", config_path="config.yaml")
    >>> result = patient.analyze("EXAMPLE_SESSION", "STIM_A_1_2")
    >>> result.primary_significant.head()

    >>> epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2")
    >>> detections = epochs.detect_all()
    """

    def __init__(self, patient_id: str, config_path: str | None = None, verbose: bool = True) -> None:
        self.patient_id = patient_id
        self.dataloader = DataLoader(patient_id, config_path=config_path, load_data=False, verbose=verbose)
        self.pipeline = Pipeline(self.dataloader)
        self.events = Events(self.dataloader)
        self.analysis = Analysis(self.pipeline)
        self.elec_meta = self.dataloader.elec_meta
        self.stim_meta = self.dataloader.stim_meta
        self.raw_meta = self.dataloader.raw_meta

    def __repr__(self) -> str:
        sessions = ", ".join(map(str, self.session_ids)) or "none"
        return f"<Patient {self.patient_id} | sessions: {sessions}>"

    # ------------------------------------------------------------------
    # Construction from standard formats
    # ------------------------------------------------------------------
    @classmethod
    def from_bids(
        cls,
        bids_root: str,
        subject: str,
        output_root: str | None = None,
        session: str | None = None,
        task: str | None = None,
        patient_id: str | None = None,
        copy_raw: bool = False,
        derivatives: bool = True,
        verbose: bool = True,
    ) -> Patient:
        """Import a BIDS iEEG subject and return a ready-to-use patient facade."""

        from .bids import import_bids_project, normalize_subject

        patient_id = patient_id or normalize_subject(subject)
        result = import_bids_project(
            bids_root=bids_root,
            output_root=output_root or f"{bids_root}/erpy_project",
            subject=subject,
            session=session,
            task=task,
            patient_id=patient_id,
            copy_raw=copy_raw,
            derivatives=derivatives,
        )
        return cls(patient_id, config_path=str(result.config_path), verbose=verbose)

    @classmethod
    def from_nwb(
        cls,
        nwb_path: str,
        output_root: str | None = None,
        patient_id: str | None = None,
        session_id: str | None = None,
        stim_pair: str | None = None,
        stim_times: Any | None = None,
        stim_duration_s: float = 0.001,
        series_name: str | None = None,
        copy_raw: bool = False,
        verbose: bool = True,
    ) -> Patient:
        """Import one NWB recording and return a ready-to-use patient facade."""

        dataloader = DataLoader.from_nwb(
            nwb_path=nwb_path,
            output_root=output_root,
            patient_id=patient_id,
            session_id=session_id,
            stim_pair=stim_pair,
            stim_times=stim_times,
            stim_duration_s=stim_duration_s,
            series_name=series_name,
            copy_raw=copy_raw,
            verbose=verbose,
        )
        return cls(dataloader.patient_id, config_path=str(dataloader.config_path), verbose=verbose)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def __getitem__(self, session_id: str) -> Recording:
        return self.recording(session_id)

    def recording(self, session_id: str) -> Recording:
        """Return the recording facade for one session."""

        return Recording(self, session_id)

    def session(self, session_id: str) -> Recording:
        """Return a recording facade; alias of ``recording``."""

        return self.recording(session_id)

    @property
    def session_ids(self) -> list[str]:
        """Registered session identifiers for this patient."""

        return self.dataloader.session_ids

    def sessions(self) -> list[str]:
        """Return registered session identifiers; same as :attr:`session_ids`."""

        return self.session_ids

    def stim_pairs(self, session_id: str | None = None) -> list[str]:
        """Return registered stimulation pairs, optionally for one session."""

        return self.dataloader.stim_pairs(session_id)

    # ------------------------------------------------------------------
    # Workflow shortcuts (each delegates to the session's Recording)
    # ------------------------------------------------------------------
    def detect_events(self, session_id: str, stim_pair: str, method: str = "auto", **kwargs: Any) -> pd.DataFrame:
        """Detect stimulation events for one session and stimulation pair."""

        return self.recording(session_id).detect_events(stim_pair, method=method, **kwargs)

    def epoch(
        self,
        session_id: str,
        stim_pair: str,
        pipeline: PipelineSpec = DEFAULT_PIPELINE,
        **kwargs: Any,
    ) -> Epochs:
        """Extract or load event-locked epochs for one acquisition.

        See :meth:`Recording.epoch` for the epoch and caching options.
        """

        return self.recording(session_id).epoch(stim_pair, pipeline=pipeline, **kwargs)

    def run(
        self,
        session_id: str,
        stim_pair: str,
        pipeline: PipelineSpec = DEFAULT_PIPELINE,
        tmin: float = DEFAULT_TMIN,
        tmax: float = DEFAULT_TMAX,
        baseline: tuple[float, float] | None = DEFAULT_BASELINE,
        zero_time: float | str | None = POST_ARTIFACT_ZERO,
        artifact_anchor_params: dict[str, Any] | None = None,
        event_method: str = "auto",
        **kwargs: Any,
    ) -> Epochs:
        """Run preprocessing and epoch extraction without reusing a cache."""

        return self.recording(session_id).epoch(
            stim_pair,
            pipeline=pipeline,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            zero_time=zero_time,
            artifact_anchor_params=artifact_anchor_params,
            cache=False,
            event_method=event_method,
            save_epochs=True,
            **kwargs,
        )

    def analyze(
        self,
        session_id: str,
        stim_pair: str,
        **kwargs: Any,
    ) -> AnalysisResult:
        """Run epoching, response-artifact QC, and ERP detection in one call.

        See :meth:`Recording.analyze` for the available options.
        """

        return self.recording(session_id).analyze(stim_pair, **kwargs)

    def warmup(self, session_ids: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        """Precompute local caches for selected patient sessions."""

        from .warmup import warm_patient

        return warm_patient(self.patient_id, config_path=self.dataloader.config_path, session_ids=session_ids, **kwargs)

    def warmup_jobs(self, session_ids: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        """Submit cache-warmup jobs for selected patient sessions."""

        from .warmup import submit_warmup_jobs

        return submit_warmup_jobs(
            self.patient_id, config_path=self.dataloader.config_path, session_ids=session_ids, **kwargs
        )


@dataclass
class Recording:
    """Session-scoped analysis facade owned by a ``Patient``.

    A recording fixes ``session_id`` while exposing event creation, epoching,
    cache loading, and the one-call analysis contract for any registered
    stimulation pair. ``Session`` is a compatibility alias for this class.

    Examples
    --------
    >>> recording = patient["EXAMPLE_SESSION"]
    >>> recording.stim_pairs()
    ['STIM_A_1_2', 'STIM_A_3_4']
    >>> result = recording.analyze("STIM_A_1_2")
    """

    patient: Patient
    session_id: str

    @property
    def dataloader(self) -> DataLoader:
        """Return the patient data loader used by this recording."""

        return self.patient.dataloader

    @property
    def pipeline_obj(self) -> Pipeline:
        """Return the patient preprocessing pipeline used by this recording."""

        return self.patient.pipeline

    def stim_pairs(self) -> list[str]:
        """Return stimulation pairs registered for this recording session."""

        return self.dataloader.stim_pairs(self.session_id)

    def detect_events(
        self,
        stim_pair: str,
        method: str = "auto",
        save_events: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Detect and optionally save events for one stimulation pair.

        ``kwargs`` are forwarded to :func:`ERPy.detect_events_from_artifacts`.
        """

        return detect_events_from_artifacts(
            self.dataloader,
            self.session_id,
            stim_pair,
            method=method,
            save_events=save_events,
            **kwargs,
        )

    def events_for(self, stim_pair: str, method: str = "auto", save_events: bool = True, **kwargs: Any) -> pd.DataFrame:
        """Detect events for one stimulation pair; alias of :meth:`detect_events`."""

        return self.detect_events(stim_pair, method=method, save_events=save_events, **kwargs)

    def load_epochs(self, stim_pair: str, pipeline: PipelineSpec = DEFAULT_PIPELINE, **kwargs: Any) -> Epochs:
        """Load the newest matching epoch cache for a stimulation pair.

        ``allow_unsafe_legacy_pickle=True`` may be passed only for a trusted
        pre-1.0 cache whose pickle metadata the caller accepts executing.
        """

        return self.pipeline_obj.load_epochs(
            self.session_id,
            stim_pair,
            pipeline_steps=self._steps(pipeline),
            allow_unsafe_legacy_pickle=bool(kwargs.pop("allow_unsafe_legacy_pickle", False)),
        )

    def epoch(
        self,
        stim_pair: str,
        pipeline: PipelineSpec = DEFAULT_PIPELINE,
        tmin: float = DEFAULT_TMIN,
        tmax: float = DEFAULT_TMAX,
        baseline: tuple[float, float] | None = DEFAULT_BASELINE,
        zero_time: float | str | None = POST_ARTIFACT_ZERO,
        artifact_anchor_params: dict[str, Any] | None = None,
        cache: bool = True,
        event_method: str = "auto",
        save_epochs: bool = True,
        **kwargs: Any,
    ) -> Epochs:
        """Preprocess and epoch one stimulation pair, optionally using a cache.

        Parameters
        ----------
        stim_pair
            Registered stimulation pair, for example ``"STIM_A_1_2"``.
        pipeline
            Name of a standard pipeline or a list of ``(step, parameters)``.
        tmin, tmax
            Epoch limits in seconds relative to stimulation onset.
        baseline
            Window whose mean is subtracted from each trial, or ``None``.
        zero_time
            Post-baseline recentring: ERPy's post-artifact anchor (default), a
            fixed time in seconds, or ``None`` to skip.
        artifact_anchor_params
            Overrides for the post-artifact anchor search.
        cache
            Reuse a verified epoch cache when one matches.
        event_method
            Detector used when no event file exists yet.
        save_epochs
            Write the extracted epochs to the cache.
        **kwargs
            Recording selection (``input_source``, ``raw_file``,
            ``stim_start``, ``processing_margin_s``, ``overwrite``) and, when
            events must be detected first, event-detection options.

        Notes
        -----
        Automatic reuse verifies current event and source contents plus effective
        processing settings. Legacy caches without this identity are recomputed.
        ``auto`` reads registered raw inputs on a miss; a window export is used
        when the original inputs are unavailable. Full source hashing costs I/O.
        NWB and custom processing hooks currently disable automatic reuse.
        Configuration/metadata edits on disk require constructing a new Patient;
        this method uses the DataLoader's current in-memory settings.
        """

        steps = self._steps(pipeline)
        event_path = self.dataloader.get_event_path(self.session_id, stim_pair)
        if not event_path.exists():
            self.detect_events(stim_pair, method=event_method, save_events=True, **kwargs)
        event_bytes = event_path.read_bytes()
        event_info = pd.read_csv(io.BytesIO(event_bytes))
        if "times" not in event_info:
            raise ValueError(f"Event file must contain a 'times' column: {event_path}")
        event_times = parse_event_times(event_info["times"])

        processing_options = {name: kwargs[name] for name in _PROCESSING_OPTION_NAMES if name in kwargs}
        options = {
            "tmin": float(tmin),
            "tmax": float(tmax),
            "baseline": baseline,
            "zero_time": zero_time,
            "artifact_anchor_params": artifact_anchor_params or {},
            "event_method": event_method,
            **processing_options,
        }
        identity, source = epoch_cache_identity(
            self.pipeline_obj, self.session_id, stim_pair, steps, event_bytes, options
        )
        path = self.pipeline_obj._epochs_path(
            self.session_id,
            stim_pair,
            steps,
            tmin,
            tmax,
            baseline,
            zero_time,
            artifact_anchor_params=artifact_anchor_params,
        )
        if cache and identity is not None and not processing_options.get("overwrite", False):
            try:
                cached = self.pipeline_obj.load_epochs_file(path)
                if cached.metadata.get("epoch_cache_identity") == identity:
                    return cached
            except (FileNotFoundError, ValueError):
                pass

        # Lower caches have no binding to current events/source. A facade miss
        # must not silently reconstruct an epoch from stale in-memory signals.
        self.dataloader.clear_memory_cache()
        processing_options["input_source"] = source
        epochs, _ = self.pipeline_obj.process_and_epoch(
            self.session_id,
            stim_pair,
            pipeline_steps=steps,
            event_times=event_times,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            zero_time=zero_time,
            artifact_anchor_params=artifact_anchor_params,
            save_processed=kwargs.get("save_processed", False),
            save_epochs=False,
            **processing_options,
        )
        current_event_bytes = event_path.read_bytes()
        current_identity, _ = epoch_cache_identity(
            self.pipeline_obj, self.session_id, stim_pair, steps, current_event_bytes, options
        )
        if current_identity != identity or event_path.read_bytes() != event_bytes:
            raise RuntimeError("Epoch inputs changed during processing; retry after writes finish")
        epochs.metadata["epoch_cache_identity"] = identity
        epochs.metadata["epoch_source_kind"] = source
        if save_epochs:
            epochs.to_hdf(path)
        return epochs

    def analyze(
        self,
        stim_pair: str,
        *,
        pipeline: PipelineSpec = DEFAULT_PIPELINE,
        methods: list[str] | None = None,
        min_consensus: int = 2,
        response_window: tuple[float, float] | None = None,
        artifact_kwargs: dict[str, Any] | None = None,
        artifact_reject: str | None = "nan_response",
        max_bad_channel_fraction: float = 0.25,
        qc_max_bad_response_fraction: float = 0.25,
        qc_max_hard_artifact_fraction: float = 0.10,
        qc_min_clean_responses: int = 8,
        qc_exclude_boundary_peaks: bool = True,
        hard_artifact_reasons: Iterable[str] = tuple(sorted(DEFAULT_HARD_ARTIFACT_REASONS)),
        detect_kwargs: dict[str, Any] | None = None,
        **epoch_kwargs: Any,
    ) -> AnalysisResult:
        """Run the default ERPy analysis contract for one stimulation pair.

        The call epochs the acquisition, scores every trial-by-channel response
        for artifacts, rejects contaminated responses, runs the detectors on
        the cleaned epochs, and annotates each detection with artifact-QC
        eligibility. It returns the same auditable artifacts available from
        the lower-level API.

        Parameters
        ----------
        stim_pair
            Registered stimulation pair.
        pipeline
            Standard pipeline name or custom step list.
        methods
            Detectors to run; defaults to :data:`ERPy.ALL_METHODS`.
        min_consensus
            Number of standalone detectors required by the legacy consensus.
        response_window
            Response window applied to both artifact QC and detection.
        artifact_kwargs
            Options for :meth:`Epochs.flag_artifacts`.
        artifact_reject
            Rejection mode (``"nan_response"``, ``"drop_epoch"``,
            ``"drop_channel"``) or ``None`` to detect on the unmasked epochs.
        max_bad_channel_fraction
            Rejection threshold forwarded to :meth:`Epochs.reject_artifacts`.
        qc_max_bad_response_fraction, qc_max_hard_artifact_fraction, \
qc_min_clean_responses, qc_exclude_boundary_peaks
            Channel eligibility criteria behind ``artifact_qc_pass``.
        hard_artifact_reasons
            Artifact reasons counted as hard failures.
        detect_kwargs
            Additional options for :meth:`Epochs.detect_all`.
        **epoch_kwargs
            Options for :meth:`epoch` (``tmin``, ``tmax``, ``baseline``, ...).

        Notes
        -----
        Trial-contact artifact QC is always evaluated. Persistent-channel
        detection is performed only when the preprocessing pipeline contains
        an explicit ``reject_bad_channels`` step, such as ``blank_filt_reject``.
        """

        resolved_pipeline_steps = self._steps(pipeline)
        epochs = self.epoch(stim_pair, pipeline=pipeline, **epoch_kwargs)

        hard_reasons = tuple(str(reason) for reason in hard_artifact_reasons)
        artifact_options = dict(artifact_kwargs or {})
        if response_window is not None:
            artifact_options["response_window"] = response_window
        artifact_report = epochs.flag_artifacts(**artifact_options)
        artifact_summary = artifact_report.by_channel(hard_artifact_reasons=hard_reasons)

        clean_epochs = epochs
        if artifact_reject:
            clean_epochs = epochs.reject_artifacts(
                report=artifact_report,
                mode=artifact_reject,
                max_bad_channel_fraction=max_bad_channel_fraction,
            )

        detection_options = dict(detect_kwargs or {})
        if methods is not None:
            detection_options["methods"] = methods
        if response_window is not None:
            detection_options["response_window"] = response_window
        detection_options.setdefault("min_consensus", min_consensus)
        detections = clean_epochs.detect_all(**detection_options)
        qc_detections = _annotate_detection_artifact_qc(
            detections,
            artifact_summary,
            max_bad_response_fraction=qc_max_bad_response_fraction,
            max_hard_artifact_fraction=qc_max_hard_artifact_fraction,
            min_clean_responses=qc_min_clean_responses,
            exclude_boundary_peaks=qc_exclude_boundary_peaks,
        )

        metadata = {
            "audit_schema_version": 1,
            "erpy_version": str(__version__),
            "erpy_compatibility_id": str(CHECKPOINT_COMPATIBILITY_ID),
            "patient_id": self.patient.patient_id,
            "session_id": self.session_id,
            "stim_pair": stim_pair,
            "pipeline": pipeline,
            "methods": methods,
            "min_consensus": min_consensus,
            "artifact_reject": artifact_reject,
            "qc_max_bad_response_fraction": qc_max_bad_response_fraction,
            "qc_max_hard_artifact_fraction": qc_max_hard_artifact_fraction,
            "qc_min_clean_responses": int(qc_min_clean_responses),
            "qc_exclude_boundary_peaks": bool(qc_exclude_boundary_peaks),
            "epoch": _epoch_audit_record(pipeline, resolved_pipeline_steps, epoch_kwargs),
            "epoch_metadata": epochs.metadata,
            "persistent_channel_qc": _persistent_channel_qc_record(resolved_pipeline_steps),
            "artifact_qc": {
                "requested_options": artifact_kwargs or {},
                "effective_options": _effective_artifact_options(artifact_options, epochs),
                "reject_mode": artifact_reject,
                "max_bad_channel_fraction": float(max_bad_channel_fraction),
                "hard_artifact_reasons": sorted(hard_reasons),
                "eligibility": {
                    "max_bad_response_fraction": float(qc_max_bad_response_fraction),
                    "max_hard_artifact_fraction": float(qc_max_hard_artifact_fraction),
                    "min_clean_responses": int(qc_min_clean_responses),
                    "exclude_boundary_peaks": bool(qc_exclude_boundary_peaks),
                },
            },
            "detection": _detection_audit_record(detect_kwargs, detection_options, detections),
        }
        return AnalysisResult(
            epochs=epochs,
            clean_epochs=clean_epochs,
            detections=detections,
            artifact_report=artifact_report,
            artifact_summary=artifact_summary,
            qc_detections=qc_detections,
            metadata=metadata,
        )

    def _steps(self, pipeline: PipelineSpec) -> PipelineSteps:
        return self.pipeline_obj.resolve_pipeline(pipeline)


# Backward-compatible aliases for notebooks written against the recovered docs.
Study = Patient
Session = Recording


# ----------------------------------------------------------------------
# Audit metadata
# ----------------------------------------------------------------------
def _epoch_audit_record(
    pipeline: PipelineSpec, resolved_steps: PipelineSteps, epoch_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Describe the epoch settings in effect, including untouched defaults."""

    record: dict[str, Any] = {"pipeline": pipeline, "resolved_pipeline_steps": resolved_steps}
    for name, default in _EPOCH_OPTION_DEFAULTS.items():
        value = epoch_kwargs.get(name, default)
        record[name] = bool(value) if name in {"cache", "save_epochs"} else value
    record["additional_options"] = {
        key: value for key, value in epoch_kwargs.items() if key not in _EPOCH_OPTION_DEFAULTS
    }
    return record


def _persistent_channel_qc_record(resolved_steps: PipelineSteps) -> dict[str, Any]:
    """Record whether the pipeline screened persistently bad channels."""

    steps = [
        {"step": str(step), "parameters": params}
        for step, params in resolved_steps
        if str(step) == "reject_bad_channels"
    ]
    return {
        "configured": bool(steps),
        "steps": steps,
        "note": (
            "Persistent-channel rejection was configured in the preprocessing pipeline."
            if steps
            else "Persistent-channel detection was not configured; only predeclared exclusions and reference screening may apply."
        ),
    }


def _effective_artifact_options(artifact_options: dict[str, Any], epochs: Epochs) -> dict[str, Any]:
    """Resolve the artifact-QC settings actually applied to ``epochs``."""

    effective = _effective_defaults(detect_artifactual_responses, artifact_options, skip={"epochs"})
    if effective.get("baseline_window") is None:
        effective["baseline_window"] = epochs.baseline
    if effective.get("response_window") is None:
        effective["response_window"] = (max(0.005, float(epochs.tmin)), min(float(epochs.tmax), 0.5))
    return effective


def _detection_audit_record(
    detect_kwargs: dict[str, Any] | None,
    detection_options: dict[str, Any],
    detections: pd.DataFrame,
) -> dict[str, Any]:
    """Describe requested and effective detector settings."""

    selected_methods = [str(method).lower() for method in detection_options.get("methods", ALL_METHODS)]
    return {
        "requested_options": detect_kwargs or {},
        "effective_core": {
            "methods": selected_methods,
            "min_consensus": int(detection_options.get("min_consensus", 2)),
            "baseline_window": detection_options.get("baseline_window", DEFAULT_BASELINE),
            "response_window": detection_options.get("response_window", DEFAULT_RESPONSE_WINDOW),
            "response_boundary_guard_s": detection_options.get("response_boundary_guard_s", 0.002),
            "detector_input": (
                "wideband_epochs" if detection_options.get("wideband_epochs") is not None else "primary_epochs"
            ),
        },
        "shared_overrides": {
            key: value
            for key, value in detection_options.items()
            if key not in _CORE_DETECTION_OPTION_NAMES and key not in ALL_METHODS
        },
        "method_overrides": {
            method: detection_options[method] for method in selected_methods if method in detection_options
        },
        "method_parameter_records": _detector_parameter_records(detections),
    }


def _effective_defaults(function, supplied: dict[str, Any], *, skip: set[str]) -> dict[str, Any]:
    """Return declared defaults updated with explicitly supplied parameters."""

    effective: dict[str, Any] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in skip or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is not inspect.Parameter.empty:
            effective[name] = parameter.default
    effective.update(supplied)
    return effective


def _first_record_value(frame: pd.DataFrame, column: str) -> Any | None:
    """Return the first non-missing value of ``column``, if any."""

    if column not in frame.columns:
        return None
    for value in frame[column]:
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return value
    return None


def _detector_parameter_records(detections: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize method-native parameter fields stored with detector rows."""

    if detections.empty or "method" not in detections.columns:
        return []
    records: list[dict[str, Any]] = []
    for method, frame in detections.groupby("method", sort=True, dropna=False):
        record: dict[str, Any] = {"method": str(method)}
        for column in _DETECTOR_PARAMETER_COLUMNS:
            value = _first_record_value(frame, column)
            if value is not None:
                record[column] = value
        records.append(record)
    return records


def _parse_event_times(values):
    """Retained name of :func:`ERPy.utils.parse_event_times`."""

    return parse_event_times(values)
