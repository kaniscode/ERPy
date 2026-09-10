from __future__ import annotations

import inspect
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .analysis import Analysis
from .artifact_anchor import POST_ARTIFACT_ZERO
from .bad_channels import ArtifactResponseReport, DEFAULT_HARD_ARTIFACT_REASONS
from .dataloader import DataLoader
from .epochs import Epochs
from .events import Events, detect_events_from_artifacts
from ._epoch_cache import epoch_cache_identity
from .pipeline import Pipeline


EpochsResult = Epochs


@dataclass
class AnalysisResult:
    """One-call result for the common ERPy stimulation-response workflow."""

    epochs: Epochs
    detections: pd.DataFrame
    artifact_report: ArtifactResponseReport
    artifact_summary: pd.DataFrame
    qc_detections: pd.DataFrame
    clean_epochs: Epochs | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def significant(self) -> pd.DataFrame:
        """Return significant response rows from the full detection table."""

        return _significant_rows(self.detections)

    @property
    def primary_significant(self) -> pd.DataFrame:
        """Return one CRP-energy row per primary response after artifact QC."""

        data = self.qc_detections.copy()
        if "primary_qc_pass" not in data.columns:
            return data.iloc[0:0].copy()
        data = data[data["primary_qc_pass"].fillna(False)].copy()
        if "method" in data.columns:
            primary_rows = data[data["method"].astype(str).eq("crp_energy")]
            if not primary_rows.empty:
                return primary_rows.reset_index(drop=True)
        identity = [
            column
            for column in (
                "patient_id",
                "session_id",
                "stim_pair",
                "acquisition_id",
                "raw_file",
                "channel",
            )
            if column in data.columns
        ]
        return (
            data.drop_duplicates(identity).reset_index(drop=True)
            if identity
            else data.reset_index(drop=True)
        )

    @property
    def qc_pass_detections(self) -> pd.DataFrame:
        """Return detection rows that pass response-artifact QC."""

        if self.qc_detections.empty or "artifact_qc_pass" not in self.qc_detections.columns:
            return self.qc_detections.copy()
        return self.qc_detections[self.qc_detections["artifact_qc_pass"].fillna(False)].copy()

    def save(self, output_dir: str | Path, prefix: str | None = None) -> dict[str, Path]:
        """Save the standard tables and deterministic audit metadata.

        The metadata record includes resolved epoch and pipeline options,
        whether persistent-channel screening was configured, effective
        response-artifact settings, detector windows and overrides, and one
        method-native parameter record for every executed detector.

        This method writes only to ``output_dir``. It does not upload data,
        contact a registry, or create a release-record identifier.
        Projects that use an opaque release-record label should keep its local
        mapping under their own governance controls.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix or _result_prefix(self.metadata)
        paths = {
            "detections": output_dir / f"{prefix}_detections.csv",
            "detections_qc": output_dir / f"{prefix}_detections_qc.csv",
            "detections_qc_pass": output_dir / f"{prefix}_detections_qc_pass.csv",
            "detections_primary": output_dir / f"{prefix}_detections_primary.csv",
            "artifact_response_table": output_dir / f"{prefix}_artifact_response_table.csv",
            "artifact_channel_summary": output_dir / f"{prefix}_artifact_channel_summary.csv",
            "zero_anchor_report": output_dir / f"{prefix}_zero_anchor_report.csv",
            "metadata": output_dir / f"{prefix}_metadata.json",
        }
        self.detections.to_csv(paths["detections"], index=False)
        self.qc_detections.to_csv(paths["detections_qc"], index=False)
        self.qc_pass_detections.to_csv(paths["detections_qc_pass"], index=False)
        self.primary_significant.to_csv(paths["detections_primary"], index=False)
        self.artifact_report.table.to_csv(paths["artifact_response_table"], index=False)
        self.artifact_summary.to_csv(paths["artifact_channel_summary"], index=False)
        self.epochs.post_artifact_anchor_report().to_csv(paths["zero_anchor_report"], index=False)
        paths["metadata"].write_text(
            json.dumps(
                _metadata_value(self.metadata),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return paths


class Patient:
    """Quick-use public wrapper for one patient.

    Examples
    --------
    >>> patient = ep.Patient("EXAMPLE_PATIENT")
    >>> epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2")
    >>> detections = epochs.detect_erp_all()
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
    ) -> "Patient":
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
    ) -> "Patient":
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
        return cls(dataloader.patient_id, config_path=str(dataloader.CONFIG_PATH), verbose=verbose)

    def __getitem__(self, session_id: str) -> "Recording":
        return self.recording(session_id)

    def recording(self, session_id: str) -> "Recording":
        """Return the recording facade for one session."""

        return Recording(self, session_id)

    def session(self, session_id: str) -> "Recording":
        """Return a recording facade; alias of ``recording``."""

        return self.recording(session_id)

    def sessions(self) -> list[str]:
        """Return registered session identifiers for this patient."""

        return self.dataloader.session_ids

    def stim_pairs(self, session_id: str) -> list[str]:
        """Return registered stimulation pairs for one session."""

        return self.dataloader.stim_pairs(session_id)

    def detect_events(self, session_id: str, stim_pair: str, method: str = "auto", **kwargs: Any) -> pd.DataFrame:
        """Detect stimulation events for one session and stimulation pair."""

        return self.recording(session_id).events_for(stim_pair, method=method, **kwargs)

    def epoch(
        self,
        session_id: str,
        stim_pair: str,
        pipeline: str | list[tuple[str, dict[str, Any]]] = "blank_filt",
        **kwargs: Any,
    ) -> EpochsResult:
        """Extract or load event-locked epochs for one acquisition."""

        return self.recording(session_id).epoch(stim_pair, pipeline=pipeline, **kwargs)

    def run(
        self,
        session_id: str,
        stim_pair: str,
        pipeline: str | list[tuple[str, dict[str, Any]]] = "blank_filt",
        tmin: float = -0.5,
        tmax: float = 1.0,
        baseline: tuple[float, float] | None = (-0.5, -0.03),
        zero_time: float | str | None = POST_ARTIFACT_ZERO,
        artifact_anchor_params: dict[str, Any] | None = None,
        event_method: str = "auto",
        **kwargs: Any,
    ) -> EpochsResult:
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
        """Run epoching, response-artifact QC, and ERP detection in one call."""

        return self.recording(session_id).analyze(stim_pair, **kwargs)

    def warmup(self, session_ids: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        """Precompute local caches for selected patient sessions."""

        from .warmup import warm_patient

        return warm_patient(self.patient_id, config_path=self.dataloader.CONFIG_PATH, session_ids=session_ids, **kwargs)

    def warmup_jobs(self, session_ids: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        """Submit cache-warmup jobs for selected patient sessions."""

        from .warmup import submit_warmup_jobs

        return submit_warmup_jobs(self.patient_id, config_path=self.dataloader.CONFIG_PATH, session_ids=session_ids, **kwargs)


@dataclass
class Recording:
    """Session-scoped analysis facade owned by a ``Patient``.

    A recording fixes ``session_id`` while exposing event creation, epoching,
    cache loading, and the one-call analysis contract for any registered
    stimulation pair. ``Session`` is a compatibility alias for this class.
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

    def events_for(
        self,
        stim_pair: str,
        method: str = "auto",
        save_events: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Detect and optionally save events for one stimulation pair."""

        return detect_events_from_artifacts(
            self.dataloader,
            self.session_id,
            stim_pair,
            method=method,
            save_events=save_events,
            **kwargs,
        )

    def load_epochs(self, stim_pair: str, pipeline: str | list[tuple[str, dict[str, Any]]] = "blank_filt", **kwargs: Any) -> EpochsResult:
        """Load the newest matching epoch cache for a stimulation pair.

        ``allow_unsafe_legacy_pickle=True`` may be passed only for a trusted
        pre-1.0 cache whose pickle metadata the caller accepts executing.
        """

        steps = self._steps(pipeline)
        return self.pipeline_obj.load_epochs(
            self.session_id,
            stim_pair,
            pipeline_steps=steps,
            allow_unsafe_legacy_pickle=bool(
                kwargs.pop("allow_unsafe_legacy_pickle", False)
            ),
        )

    def epoch(
        self,
        stim_pair: str,
        pipeline: str | list[tuple[str, dict[str, Any]]] = "blank_filt",
        tmin: float = -0.5,
        tmax: float = 1.0,
        baseline: tuple[float, float] | None = (-0.5, -0.03),
        zero_time: float | str | None = POST_ARTIFACT_ZERO,
        artifact_anchor_params: dict[str, Any] | None = None,
        cache: bool = True,
        event_method: str = "auto",
        save_epochs: bool = True,
        **kwargs: Any,
    ) -> EpochsResult:
        """Preprocess and epoch one stimulation pair, optionally using a cache.

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
            self.events_for(stim_pair, method=event_method, save_events=True, **kwargs)
        event_bytes = event_path.read_bytes()
        event_info = pd.read_csv(io.BytesIO(event_bytes))
        if "times" not in event_info:
            raise ValueError(f"Event file must contain a 'times' column: {event_path}")
        event_times = _parse_event_times(event_info["times"])
        processing_options = {key: kwargs[key] for key in (
            "input_source", "raw_file", "stim_start", "processing_margin_s",
            "load_from_raw", "overwrite",
        ) if key in kwargs}
        options = {
            "tmin": float(tmin), "tmax": float(tmax), "baseline": baseline,
            "zero_time": zero_time, "artifact_anchor_params": artifact_anchor_params or {},
            "event_method": event_method, **processing_options,
        }
        identity, source = epoch_cache_identity(
            self.pipeline_obj, self.session_id, stim_pair, steps, event_bytes, options,
        )
        path = self.pipeline_obj._epochs_path(
            self.session_id, stim_pair, steps, tmin, tmax, baseline, zero_time,
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
        current, _ = epoch_cache_identity(
            self.pipeline_obj, self.session_id, stim_pair, steps,
            event_path.read_bytes(), options,
        )
        if current != identity or event_path.read_bytes() != event_bytes:
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
        pipeline: str | list[tuple[str, dict[str, Any]]] = "blank_filt",
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

        This keeps the common notebook path compact while returning the same
        auditable artifacts available from the lower-level API. Trial-contact
        artifact QC is always evaluated. Persistent-channel detection is
        performed only when the preprocessing pipeline contains an explicit
        ``reject_bad_channels`` step, such as ``blank_filt_reject``.
        """

        resolved_pipeline_steps = self._steps(pipeline)
        epochs = self.epoch(stim_pair, pipeline=pipeline, **epoch_kwargs)
        hard_reason_values = tuple(str(reason) for reason in hard_artifact_reasons)
        artifact_options = dict(artifact_kwargs or {})
        if response_window is not None:
            artifact_options["response_window"] = response_window
        artifact_report = epochs.flag_artifacts(**artifact_options)
        artifact_summary = artifact_report.by_channel(
            hard_artifact_reasons=hard_reason_values
        )

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
        detections = clean_epochs.detect_erp_all(**detection_options)
        qc_detections = _annotate_detection_artifact_qc(
            detections,
            artifact_summary,
            max_bad_response_fraction=qc_max_bad_response_fraction,
            max_hard_artifact_fraction=qc_max_hard_artifact_fraction,
            min_clean_responses=qc_min_clean_responses,
            exclude_boundary_peaks=qc_exclude_boundary_peaks,
        )
        from . import CHECKPOINT_COMPATIBILITY_ID
        from . import __version__ as erpy_version
        from .bad_channels import detect_artifactual_responses
        from .erp_detection import ALL_METHODS

        effective_artifact_options = _effective_defaults(
            detect_artifactual_responses,
            artifact_options,
            skip={"epochs"},
        )
        if effective_artifact_options.get("baseline_window") is None:
            effective_artifact_options["baseline_window"] = epochs.baseline
        if effective_artifact_options.get("response_window") is None:
            effective_artifact_options["response_window"] = (
                max(0.005, float(epochs.tmin)),
                min(float(epochs.tmax), 0.5),
            )

        selected_methods = [
            str(method).lower()
            for method in detection_options.get("methods", ALL_METHODS)
        ]
        effective_detection_core = {
            "methods": selected_methods,
            "min_consensus": int(detection_options.get("min_consensus", 2)),
            "baseline_window": detection_options.get(
                "baseline_window", (-0.5, -0.03)
            ),
            "response_window": detection_options.get(
                "response_window", (0.01, 0.3)
            ),
            "response_boundary_guard_s": detection_options.get(
                "response_boundary_guard_s", 0.002
            ),
            "detector_input": (
                "wideband_epochs"
                if detection_options.get("wideband_epochs") is not None
                else "primary_epochs"
            ),
        }
        method_overrides = {
            method: detection_options[method]
            for method in selected_methods
            if method in detection_options
        }
        shared_detection_overrides = {
            key: value
            for key, value in detection_options.items()
            if key
            not in {
                "methods",
                "min_consensus",
                "baseline_window",
                "response_window",
                "response_boundary_guard_s",
                "wideband_epochs",
                *ALL_METHODS,
            }
        }
        epoch_options = {
            "pipeline": pipeline,
            "resolved_pipeline_steps": resolved_pipeline_steps,
            "tmin": epoch_kwargs.get("tmin", -0.5),
            "tmax": epoch_kwargs.get("tmax", 1.0),
            "baseline": epoch_kwargs.get("baseline", (-0.5, -0.03)),
            "zero_time": epoch_kwargs.get("zero_time", POST_ARTIFACT_ZERO),
            "artifact_anchor_params": epoch_kwargs.get(
                "artifact_anchor_params", None
            ),
            "cache": bool(epoch_kwargs.get("cache", True)),
            "event_method": epoch_kwargs.get("event_method", "auto"),
            "save_epochs": bool(epoch_kwargs.get("save_epochs", True)),
            "additional_options": {
                key: value
                for key, value in epoch_kwargs.items()
                if key
                not in {
                    "tmin",
                    "tmax",
                    "baseline",
                    "zero_time",
                    "artifact_anchor_params",
                    "cache",
                    "event_method",
                    "save_epochs",
                }
            },
        }
        persistent_steps = [
            {"step": str(step), "parameters": params}
            for step, params in resolved_pipeline_steps
            if str(step) == "reject_bad_channels"
        ]

        metadata = {
            "audit_schema_version": 1,
            "erpy_version": str(erpy_version),
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
            "epoch": epoch_options,
            "epoch_metadata": epochs.metadata,
            "persistent_channel_qc": {
                "configured": bool(persistent_steps),
                "steps": persistent_steps,
                "note": (
                    "Persistent-channel rejection was configured in the preprocessing pipeline."
                    if persistent_steps
                    else "Persistent-channel detection was not configured; only predeclared exclusions and reference screening may apply."
                ),
            },
            "artifact_qc": {
                "requested_options": artifact_kwargs or {},
                "effective_options": effective_artifact_options,
                "reject_mode": artifact_reject,
                "max_bad_channel_fraction": float(max_bad_channel_fraction),
                "hard_artifact_reasons": sorted(hard_reason_values),
                "eligibility": {
                    "max_bad_response_fraction": float(
                        qc_max_bad_response_fraction
                    ),
                    "max_hard_artifact_fraction": float(
                        qc_max_hard_artifact_fraction
                    ),
                    "min_clean_responses": int(qc_min_clean_responses),
                    "exclude_boundary_peaks": bool(qc_exclude_boundary_peaks),
                },
            },
            "detection": {
                "requested_options": detect_kwargs or {},
                "effective_core": effective_detection_core,
                "shared_overrides": shared_detection_overrides,
                "method_overrides": method_overrides,
                "method_parameter_records": _detector_parameter_records(
                    detections
                ),
            },
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

    def _steps(self, pipeline: str | list[tuple[str, dict[str, Any]]]):
        if isinstance(pipeline, str):
            return self.pipeline_obj.get_standard_pipeline(pipeline)
        return self.pipeline_obj.build_custom_pipeline(pipeline)


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
    columns = (
        "method",
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
    records: list[dict[str, Any]] = []
    for method, frame in detections.groupby("method", sort=True, dropna=False):
        record: dict[str, Any] = {"method": str(method)}
        for column in columns:
            if column == "method":
                continue
            value = _first_record_value(frame, column)
            if value is not None:
                record[column] = value
        records.append(record)
    return records


def _metadata_value(value: Any) -> Any:
    """Convert analysis metadata to deterministic, JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _metadata_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_metadata_value(item) for item in sorted(value, key=str)]
    if isinstance(value, Epochs):
        return {
            "type": "Epochs",
            "n_trials": int(value.n_trials()),
            "n_channels": int(len(value.channels)),
            "sfreq": float(value.sfreq),
            "tmin": float(value.tmin),
            "tmax": float(value.tmax),
        }
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return f"{module}.{name}".lstrip(".")
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _metadata_value(item())
        except (TypeError, ValueError):
            pass
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


# Backward-compatible aliases for notebooks written against the recovered docs.
Study = Patient
Session = Recording


def _parse_event_times(values):
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy()
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed if parsed.notna().mean() > 0.8 else values


def _annotate_detection_artifact_qc(
    detections: pd.DataFrame,
    artifact_summary: pd.DataFrame,
    *,
    max_bad_response_fraction: float,
    max_hard_artifact_fraction: float,
    min_clean_responses: int,
    exclude_boundary_peaks: bool = True,
) -> pd.DataFrame:
    out = detections.copy()
    if out.empty:
        out["artifact_qc_pass"] = True
        out["primary_qc_pass"] = False
        out["crp_energy_qc_pass"] = False
        out["shape_magnitude_qc_pass"] = False
        return out
    if "channel" not in out.columns:
        out["artifact_qc_pass"] = True
        out["primary_qc_pass"] = False
        out["crp_energy_qc_pass"] = False
        out["shape_magnitude_qc_pass"] = False
        return out

    if artifact_summary.empty or "channel" not in artifact_summary.columns:
        out["bad_response_fraction"] = 0.0
        out["hard_artifact_fraction"] = 0.0
        out["n_clean_response"] = pd.to_numeric(
            out.get("n_trials", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0)
        out["artifact_reasons"] = ""
    else:
        keep_cols = [
            col
            for col in (
                "channel",
                "n_epochs",
                "n_bad_response",
                "n_clean_response",
                "bad_response_fraction",
                "n_hard_artifact_response",
                "hard_artifact_fraction",
                "artifact_reasons",
                "max_peak_abs",
                "max_ptp",
                "max_saturation_fraction",
                "max_plateau_fraction",
                "max_plateau_run_s",
                "max_rail_fraction",
            )
            if col in artifact_summary.columns
        ]
        out = out.merge(
            artifact_summary[keep_cols],
            on="channel",
            how="left",
            suffixes=("", "_artifact"),
        )
    bad_fraction = out["bad_response_fraction"] if "bad_response_fraction" in out else pd.Series(0.0, index=out.index)
    hard_fraction = out["hard_artifact_fraction"] if "hard_artifact_fraction" in out else pd.Series(0.0, index=out.index)
    if "n_clean_response" in out:
        clean_count = out["n_clean_response"]
    elif {"n_epochs", "n_bad_response"}.issubset(out.columns):
        clean_count = pd.to_numeric(out["n_epochs"], errors="coerce") - pd.to_numeric(
            out["n_bad_response"], errors="coerce"
        )
    else:
        clean_count = pd.Series(0, index=out.index)
    out["bad_response_fraction"] = pd.to_numeric(bad_fraction, errors="coerce").fillna(0.0)
    out["hard_artifact_fraction"] = pd.to_numeric(hard_fraction, errors="coerce").fillna(0.0)
    out["n_clean_response"] = pd.to_numeric(clean_count, errors="coerce").fillna(0).astype(int)
    out["artifact_reasons"] = out.get("artifact_reasons", "").fillna("").astype(str)
    out["artifact_hard_fail"] = out["hard_artifact_fraction"] >= float(max_hard_artifact_fraction)
    out["artifact_insufficient_clean_responses"] = out["n_clean_response"] < int(min_clean_responses)
    if "peak_at_response_boundary" in out.columns:
        boundary_peak = out["peak_at_response_boundary"]
        if not pd.api.types.is_bool_dtype(boundary_peak):
            boundary_peak = (
                boundary_peak.fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )
        else:
            boundary_peak = boundary_peak.fillna(False)
    else:
        boundary_peak = pd.Series(False, index=out.index)
    out["artifact_boundary_peak_fail"] = (
        boundary_peak & bool(exclude_boundary_peaks)
    )
    out["artifact_qc_pass"] = (
        (out["bad_response_fraction"] < float(max_bad_response_fraction))
        & ~out["artifact_hard_fail"]
        & ~out["artifact_insufficient_clean_responses"]
        & ~out["artifact_boundary_peak_fail"]
    )
    if "primary_significant" in out.columns:
        primary = out["primary_significant"]
        if not pd.api.types.is_bool_dtype(primary):
            primary = (
                primary.fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )
        out["primary_qc_pass"] = (
            primary.fillna(False) & out["artifact_qc_pass"]
        )
    else:
        out["primary_qc_pass"] = False
    out["crp_energy_qc_pass"] = out["primary_qc_pass"]
    if "shape_magnitude_significant" in out.columns:
        primary = out["shape_magnitude_significant"]
        if not pd.api.types.is_bool_dtype(primary):
            primary = (
                primary.fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )
        out["shape_magnitude_qc_pass"] = (
            primary.fillna(False) & out["artifact_qc_pass"]
        )
    else:
        out["shape_magnitude_qc_pass"] = False
    return out


def _significant_rows(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return detections.copy()
    if "consensus_ch" in detections.columns:
        return detections[detections["consensus_ch"].fillna(False)].copy()
    if "significant" in detections.columns:
        return detections[detections["significant"].fillna(False)].copy()
    return detections.copy()


def _result_prefix(metadata: dict[str, Any]) -> str:
    parts = [
        str(metadata.get("patient_id", "patient")),
        str(metadata.get("session_id", "session")),
        str(metadata.get("stim_pair", "stim")),
    ]
    return "_".join(part.replace("/", "-").replace(" ", "-") for part in parts if part)
