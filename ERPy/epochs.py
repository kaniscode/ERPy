"""Event-locked epochs: the trial-by-time-by-channel container used throughout ERPy."""

from __future__ import annotations

import dataclasses
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._legacy import LegacyScriptHooks

# The metadata helpers are re-exported for code that imports them from here.
from ._metadata import (  # noqa: F401
    UnsafeLegacyEpochMetadataError,
    UnsafeLegacyMetadataError,
    _decode_metadata_payload,
    _encode_metadata_payload,
    _metadata_from_json_value,
    _metadata_to_json_value,
    _read_metadata_node,
    _write_metadata_node,
)
from .artifact_anchor import POST_ARTIFACT_ZERO, detect_zero_time_artifact
from .defaults import DEFAULT_BASELINE, DEFAULT_PIPELINE, DEFAULT_RESPONSE_WINDOW, DEFAULT_TMAX, DEFAULT_TMIN
from .utils.jobrunner import JobRunner
from .utils.utils import parse_event_times

#: Largest residual, in microvolts, still reported as "zeroed" at the anchor.
_ZEROED_TOLERANCE_UV = 1e-9


def _count_true(values: Any) -> int:
    """Count truthy entries of an optional report column."""

    if values is None:
        return 0
    return int(pd.Series(values).fillna(False).astype(bool).sum())


def _max_abs_finite(values: np.ndarray) -> tuple[int, float]:
    """Return the number of finite values and their largest magnitude."""

    finite = values[np.isfinite(values)]
    return int(finite.size), (float(np.nanmax(np.abs(finite))) if finite.size else np.nan)


@dataclass(repr=False)
class EpochsCore:
    """Trial-by-time-by-channel epoch container with audit metadata.

    ``epochs_df`` uses a two-level ``(epoch, time)`` row index and channel
    columns. ``sfreq``, ``tmin``, ``tmax``, ``baseline``, and ``zero_time``
    retain the sampling and alignment contract used by detection, quality
    control, spectral analysis, and visualization methods.
    """

    epochs_df: pd.DataFrame
    sfreq: float
    tmin: float
    tmax: float
    baseline: tuple[float, float] | None = None
    stim_ch: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    zero_time: float | str | None = POST_ARTIFACT_ZERO

    def __repr__(self) -> str:
        n_trials, n_times, n_channels = self.shape
        stim = self.stim_ch if isinstance(self.stim_ch, str) else ", ".join(self.stim_ch or [])
        parts = [
            f"{n_trials} trials x {n_times} samples x {n_channels} channels",
            f"{self.tmin:g} to {self.tmax:g} s at {self.sfreq:g} Hz",
        ]
        if stim:
            parts.append(f"stim {stim}")
        return f"<{type(self).__name__} | {' | '.join(parts)}>"

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    def n_trials(self) -> int:
        """Return the number of distinct trials in the epoch table."""

        return int(self._epoch_ids.nunique())

    @property
    def channels(self) -> list[str]:
        """Return channel labels in their stored column order."""

        return list(self.epochs_df.columns)

    @property
    def times(self) -> np.ndarray:
        """Return unique epoch-relative sample times in seconds."""

        return self.epochs_df.index.get_level_values("time").unique().to_numpy(dtype=float)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(n_trials, n_times, n_channels)``, matching :meth:`as_array`."""

        return self.n_trials(), int(self.times.size), len(self.channels)

    @property
    def _epoch_ids(self) -> pd.Index:
        return self.epochs_df.index.get_level_values("epoch")

    def _trials(self):
        """Yield ``(epoch_id, time-indexed trial frame)`` in stored order."""

        for epoch_id in self._epoch_ids.unique():
            yield epoch_id, self.epochs_df.xs(epoch_id, level="epoch")

    # ------------------------------------------------------------------
    # Trial summaries
    # ------------------------------------------------------------------
    def mean(self) -> pd.DataFrame:
        """Return the trial-averaged waveform for every channel (time x channel)."""

        return self.epochs_df.groupby(level="time").mean()

    def std(self) -> pd.DataFrame:
        """Return the sample standard deviation across trials at each time."""

        return self.epochs_df.groupby(level="time").std()

    def sem(self) -> pd.DataFrame:
        """Return the standard error across trials at each time."""

        return self.epochs_df.groupby(level="time").sem()

    def get_mean_waveform(self) -> pd.DataFrame:
        """Return the trial-averaged waveform; alias of :meth:`mean`."""

        return self.mean()

    def get_std_waveform(self) -> pd.DataFrame:
        """Return the across-trial standard deviation; alias of :meth:`std`."""

        return self.std()

    def get_sem_waveform(self) -> pd.DataFrame:
        """Return the across-trial standard error; alias of :meth:`sem`."""

        return self.sem()

    def as_array(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return samples, time coordinates, and channel labels.

        The sample array is ordered as ``(trials, times, channels)``.
        """

        times = self.times
        trial_ids = sorted(self._epoch_ids.unique())
        samples = np.stack(
            [self.epochs_df.xs(trial_id, level="epoch").reindex(times).to_numpy(dtype=float) for trial_id in trial_ids]
        )
        return samples, times, self.channels

    # ------------------------------------------------------------------
    # Zero-anchor audit
    # ------------------------------------------------------------------
    def post_artifact_anchor_report(self) -> pd.DataFrame:
        """Return per-epoch post-artifact zeroing anchors when available."""

        reports = self.metadata.get("zero_anchor_reports") or []
        if isinstance(reports, pd.DataFrame):
            return reports.copy()
        if not reports:
            return pd.DataFrame()
        return pd.DataFrame(list(reports))

    def detect_zero_time_artifacts(
        self,
        zero_time: float = 0.0,
        artifact_z: float = 6.0,
    ) -> pd.DataFrame:
        """Detect whether a requested zeroing sample overlaps the onset artifact."""

        anchor_reports = self.post_artifact_anchor_report()
        at_onset = abs(float(zero_time)) <= 1e-12
        if at_onset and not anchor_reports.empty and "onset_score_z" in anchor_reports.columns:
            onset_is_artifact = anchor_reports.get("onset_is_artifact", pd.Series(False, index=anchor_reports.index))
            return pd.DataFrame(
                {
                    "epoch": anchor_reports.get("epoch", pd.Series(dtype=int)).astype(int),
                    "zero_time_s": 0.0,
                    "nearest_sample_s": 0.0,
                    "artifact_score_z": pd.to_numeric(anchor_reports["onset_score_z"], errors="coerce"),
                    "zero_time_is_artifact": onset_is_artifact.fillna(False).astype(bool),
                    "artifact_z_threshold": float(artifact_z),
                    "source": "pre_zero_anchor_report",
                }
            )

        reports = []
        for epoch_id, trial in self._trials():
            report = detect_zero_time_artifact(
                trial,
                stim_ch=self.stim_ch,
                zero_time=zero_time,
                baseline_window=self.baseline,
                artifact_z=artifact_z,
            )
            report["epoch"] = int(epoch_id)
            reports.append(report)
        return pd.DataFrame(reports)

    def zero_time_report(self, zero_time: float | str | None = None) -> dict[str, float | int | bool | str | None]:
        """Return residual amplitude at the zeroing anchor used for each epoch."""

        if zero_time is None:
            zero_time = self.zero_time
        if zero_time is None:
            return {"zero_time": None, "n_values": 0, "max_abs_uv": None, "is_zeroed": False}
        if self.times.size == 0:
            return {"zero_time": str(zero_time), "n_values": 0, "max_abs_uv": None, "is_zeroed": False}

        anchor_reports = self.post_artifact_anchor_report()
        if zero_time == POST_ARTIFACT_ZERO and not anchor_reports.empty:
            return self._post_artifact_zero_report(anchor_reports)
        return self._fixed_zero_report(zero_time)

    def _post_artifact_zero_report(self, anchor_reports: pd.DataFrame) -> dict[str, Any]:
        """Report the residual over the common post-artifact baseline window.

        The zero anchor is applied at a single common time using a short
        baseline window, so the residual is each trial's mean over that window
        (approximately zero by construction) rather than the value at one sample.
        """

        times = self.times
        window_s = float(self.metadata.get("post_baseline_window_s", 0.0) or 0.0)
        anchors = pd.to_numeric(anchor_reports.get("anchor_time_s"), errors="coerce").dropna()
        common = self.metadata.get("common_anchor_time_s")
        if common is None or not np.isfinite(float(common)):
            common = float(anchors.median()) if not anchors.empty else 0.0
        common = float(common)
        common_index = int(np.argmin(np.abs(times - common)))
        n_window = int(self.metadata.get("post_baseline_samples") or max(round(window_s * float(self.sfreq)), 1))
        window_times = times[common_index : common_index + max(n_window, 1)]

        residuals: list[float] = []
        for _, trial in self._trials():
            window_mean = trial.reindex(window_times).mean(axis=0)
            residuals.extend(np.abs(window_mean.to_numpy(dtype=float)).ravel().tolist())
        n_values, max_abs = _max_abs_finite(np.asarray(residuals, dtype=float))
        return {
            "zero_time": POST_ARTIFACT_ZERO,
            "zero_time_strategy": POST_ARTIFACT_ZERO,
            "common_anchor_s": common,
            "median_anchor_s": float(anchors.median()) if not anchors.empty else None,
            "min_anchor_s": float(anchors.min()) if not anchors.empty else None,
            "max_anchor_s": float(anchors.max()) if not anchors.empty else None,
            "nearest_sample_s": float(times[common_index]),
            "post_baseline_window_s": window_s,
            "n_values": n_values,
            "max_abs_uv": max_abs,
            "is_zeroed": bool(np.isfinite(max_abs) and max_abs <= _ZEROED_TOLERANCE_UV),
            "n_onset_artifact_epochs": _count_true(anchor_reports.get("onset_is_artifact")),
            "n_settled_epochs": _count_true(anchor_reports.get("settled")),
            "n_anchor_reports": int(len(anchor_reports)),
        }

    def _fixed_zero_report(self, zero_time: float | str) -> dict[str, Any]:
        """Report the residual at the sample nearest a fixed zero time."""

        try:
            zero_float = float(zero_time)
        except (TypeError, ValueError):
            return {
                "zero_time": str(zero_time),
                "zero_time_strategy": str(zero_time),
                "n_values": 0,
                "max_abs_uv": None,
                "is_zeroed": False,
            }
        times = self.times
        nearest = float(times[int(np.argmin(np.abs(times - zero_float)))])
        values = self.epochs_df.xs(nearest, level="time").to_numpy(dtype=float).ravel()
        n_values, max_abs = _max_abs_finite(values)
        return {
            "zero_time": zero_float,
            "zero_time_strategy": "fixed",
            "nearest_sample_s": nearest,
            "n_values": n_values,
            "max_abs_uv": max_abs,
            "is_zeroed": bool(np.isfinite(max_abs) and max_abs <= _ZEROED_TOLERANCE_UV),
        }


class Epochs(EpochsCore):
    """Epoched stimulation data backed by a MultiIndex DataFrame.

    Examples
    --------
    >>> epochs = patient.epoch("EXAMPLE_SESSION", "STIM_A_1_2")
    >>> epochs
    <Epochs | 20 trials x 751 samples x 64 channels | -0.5 to 1 s at 500 Hz | stim A1, A2>
    >>> clean = epochs.reject_artifacts(mode="nan_response")
    >>> detections = clean.detect()            # primary CRP-energy detector
    >>> fig = clean.plot.summary("CONTACT_B1", detections=detections)
    """

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def plot(self):
        """Return a plotting accessor bound to these epochs."""

        from .viz.waveforms import EpochsPlotter

        return EpochsPlotter(self)

    @property
    def spectral(self):
        """Return a spectral-analysis accessor bound to these epochs."""

        from .spectral import EpochsSpectral

        return EpochsSpectral(self)

    # ------------------------------------------------------------------
    # Response detection
    # ------------------------------------------------------------------
    def detect(self, method: str = "crp_energy", **params: Any) -> pd.DataFrame:
        """Run one detector; defaults to the primary CRP-energy conjunction.

        Equivalent to :func:`ERPy.detect`. ``params`` are forwarded to the
        detector, for example ``response_window``, ``baseline_window``,
        ``alpha`` or ``n_permutations``.
        """

        from .erp_detection import detect

        return detect(self, method=method, **params)

    def detect_all(
        self,
        methods: Iterable[str] | None = None,
        min_consensus: int = 2,
        wideband_epochs: Epochs | None = None,
        **params: Any,
    ) -> pd.DataFrame:
        """Run every requested detector and combine the channel results.

        ``methods`` defaults to :data:`ERPy.ALL_METHODS`. Pass
        ``wideband_epochs`` only when that object truly retains 70-170 Hz; the
        SIGNI/high-gamma detector is otherwise reported as unavailable. See
        :func:`ERPy.erp_detection.detect_erp_all` for the returned columns.
        """

        from .erp_detection import detect_erp_all

        return detect_erp_all(
            self,
            methods=methods,
            min_consensus=min_consensus,
            wideband_epochs=wideband_epochs,
            **params,
        )

    def detect_erp(
        self,
        method: str,
        baseline_window: tuple[float, float] = DEFAULT_BASELINE,
        response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
        **params: Any,
    ) -> pd.DataFrame:
        """Run one named ERP detector and return its channel-level results."""

        from .erp_detection import detect_erp

        return detect_erp(
            self,
            method,
            baseline_window=baseline_window,
            response_window=response_window,
            **params,
        )

    def detect_erp_all(
        self,
        methods: Iterable[str] | None = None,
        min_consensus: int = 2,
        wideband_epochs: Epochs | None = None,
        **params: Any,
    ) -> pd.DataFrame:
        """Run every registered ERP detector; alias of :meth:`detect_all`."""

        return self.detect_all(
            methods=methods,
            min_consensus=min_consensus,
            wideband_epochs=wideband_epochs,
            **params,
        )

    # ------------------------------------------------------------------
    # Trial-by-channel artifact QC
    # ------------------------------------------------------------------
    def flag_artifacts(self, *args: Any, **kwargs: Any):
        """Score trial-channel responses against the artifact criteria.

        Arguments are forwarded to :func:`ERPy.detect_artifactual_responses`.
        """

        from .bad_channels import detect_artifactual_responses

        return detect_artifactual_responses(self, *args, **kwargs)

    def reject_artifacts(self, *args: Any, **kwargs: Any) -> Epochs:
        """Return a copy with artifact-contaminated responses rejected.

        Arguments are forwarded to :func:`ERPy.reject_artifactual_responses`;
        ``mode`` is ``"drop_epoch"``, ``"nan_response"`` or ``"drop_channel"``.
        """

        from .bad_channels import reject_artifactual_responses

        return reject_artifactual_responses(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------
    def rescale(
        self,
        factor: float,
        *,
        signal_units: str = "uV",
        reason: str | None = None,
    ) -> Epochs:
        """Return a copy with channel values multiplied by ``factor``."""

        scaled = self.epochs_df.copy()
        numeric = scaled.select_dtypes(include=[np.number]).columns
        scaled.loc[:, numeric] = scaled.loc[:, numeric] * float(factor)
        prior_factor = float(self.metadata.get("value_scale_to_uv_applied", 1.0) or 1.0)
        metadata = {
            **self.metadata,
            "signal_units": str(signal_units),
            "value_scale_to_uv_applied": prior_factor * float(factor),
        }
        if reason:
            metadata["value_scale_reason"] = str(reason)
        return dataclasses.replace(self, epochs_df=scaled, metadata=metadata)

    def rereference(
        self,
        *,
        method: str = "robust_mean",
        skipped_chs: Iterable[str] | None = None,
        zscore_threshold: float = 8.0,
        min_reference_channels: int = 4,
    ) -> Epochs:
        """Return a re-referenced copy while preserving every channel."""

        from .utils.preproc import common_reference

        skipped = set(skipped_chs or [])
        if self.stim_ch:
            skipped.update([self.stim_ch] if isinstance(self.stim_ch, str) else self.stim_ch)
        reference_channels = [channel for channel in self.channels if channel not in skipped]
        frame = common_reference(
            self.epochs_df,
            reference_channels=reference_channels,
            method=method,
            zscore_threshold=zscore_threshold,
            min_reference_channels=min_reference_channels,
        )
        metadata = {
            **self.metadata,
            "reference_method": frame.attrs.get("reference_method", method),
            "reference_channels": reference_channels,
            "reference_zscore_threshold": frame.attrs.get("reference_zscore_threshold"),
            "reference_applied_post_epoch": True,
        }
        return dataclasses.replace(self, epochs_df=frame, metadata=metadata)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_hdf(
        self,
        path: str | Path,
        key: str = "epochs",
        *,
        storage_dtype: str | np.dtype | None = "float32",
        complevel: int = 5,
        complib: str = "blosc:zstd",
        **metadata: Any,
    ) -> Path:
        """Atomically persist all trials in a compact, loss-bounded HDF5 file.

        Float32 storage preserves sub-microvolt precision across the practical
        iEEG range while substantially reducing read time and disk use. Data are
        promoted by NumPy operations as needed after loading.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix or '.h5'}")
        stored = self.epochs_df.copy()
        numeric = stored.select_dtypes(include=[np.number]).columns
        if storage_dtype is not None and len(numeric):
            target_dtype = np.dtype(storage_dtype)
            stored = stored.astype({column: target_dtype for column in numeric})
        stored_metadata = {
            "sfreq": self.sfreq,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "baseline": self.baseline,
            "stim_ch": self.stim_ch,
            "zero_time": self.zero_time,
            **self.metadata,
            **metadata,
            "storage_dtype": str(np.dtype(storage_dtype)) if storage_dtype is not None else "preserve",
            "storage_format": "hdf5-fixed",
            "storage_complib": str(complib),
            "storage_complevel": int(complevel),
        }
        try:
            with pd.HDFStore(
                temporary,
                mode="w",
                complevel=max(int(complevel), 0),
                complib=str(complib),
            ) as store:
                store.put(key, stored, format="fixed")
            _write_metadata_node(temporary, key, stored_metadata)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def from_hdf(
        cls,
        path: str | Path,
        key: str = "epochs",
        *,
        allow_unsafe_legacy_pickle: bool = False,
    ) -> Epochs:
        """Load epochs and restore their audit metadata.

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
        """

        path = Path(path)
        try:
            metadata = _read_metadata_node(
                path,
                key,
                allow_unsafe_legacy_pickle=allow_unsafe_legacy_pickle,
            )
            with pd.HDFStore(path, mode="r") as store:
                frame = store[key]
                if metadata is None and allow_unsafe_legacy_pickle:
                    metadata = getattr(store.get_storer(key).attrs, "metadata", {}) or {}
        except Exception as exc:
            if isinstance(exc, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Could not load ERPy epochs file {path}") from exc
        if not metadata:
            raise ValueError(f"ERPy epochs file is incomplete or missing metadata: {path}")
        sample_times = frame.index.get_level_values("time")
        return cls(
            epochs_df=frame,
            sfreq=float(metadata.get("sfreq", np.nan)),
            tmin=float(metadata.get("tmin", sample_times.min())),
            tmax=float(metadata.get("tmax", sample_times.max())),
            baseline=tuple(metadata["baseline"]) if metadata.get("baseline") is not None else None,
            stim_ch=metadata.get("stim_ch"),
            metadata=metadata,
            zero_time=metadata.get("zero_time", metadata.get("zero_time_strategy", 0.0)),
        )


class EpochsJob(LegacyScriptHooks):
    """Base class retaining the pre-1.0 epoch job-script hooks."""


class EpochsRunner(EpochsJob):
    """Batch epoching accessor returned by :attr:`ERPy.Pipeline.epochs`."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def queue_job(self, session_ids: Iterable[str] | None = None, backend: str | None = None, **kwargs: Any):
        """Epoch every selected acquisition and return the job-runner summary."""

        runner = JobRunner(backend=backend, job_name=kwargs.get("job_name", "erpy_epochs"))
        dataloader = self.pipeline.dataloader
        steps = kwargs.get("pipeline_steps") or self.pipeline.get_standard_pipeline(
            kwargs.get("pipeline_name", DEFAULT_PIPELINE)
        )
        lines = []
        for session_id in list(session_ids or dataloader.session_ids):
            for stim_pair in kwargs.get("stim_pairs") or dataloader.stim_pairs(session_id):
                event_info = self.pipeline._load_event_info(session_id, stim_pair)
                self.pipeline.process_and_epoch(
                    session_id=session_id,
                    stim_pair=stim_pair,
                    event_times=parse_event_times(event_info["times"]),
                    pipeline_steps=steps,
                    tmin=float(kwargs.get("tmin", DEFAULT_TMIN)),
                    tmax=float(kwargs.get("tmax", DEFAULT_TMAX)),
                    baseline=kwargs.get("baseline", None),
                    zero_time=kwargs.get("zero_time", POST_ARTIFACT_ZERO),
                    save_epochs=True,
                )
                lines.append(f"{session_id}/{stim_pair}: epoched")
        return runner, "\n".join(lines), ""
