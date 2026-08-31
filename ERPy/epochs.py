from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Iterable
import uuid
import warnings
import zlib

import numpy as np
import pandas as pd

from .artifact_anchor import POST_ARTIFACT_ZERO, detect_zero_time_artifact
from .utils.jobrunner import JobRunner


_METADATA_NODE_NAME = "erpy_metadata_v2"
_METADATA_STORAGE_FORMAT = "zlib+json-v2"
_LEGACY_METADATA_NODE_NAME = "erpy_metadata_v1"
_MAX_COMPRESSED_METADATA_BYTES = 16 * 1024 * 1024
_MAX_DECOMPRESSED_METADATA_BYTES = 64 * 1024 * 1024
_JSON_TYPE_KEY = "__erpy_json_type__"


class UnsafeLegacyMetadataError(ValueError):
    """Base error for blocked pickle-backed ERPy metadata."""


class UnsafeLegacyEpochMetadataError(UnsafeLegacyMetadataError):
    """Raised when a legacy pickle-backed epoch cache is not explicitly trusted."""


def _metadata_to_json_value(value: Any) -> Any:
    """Convert supported metadata values to a deterministic tagged JSON tree."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return {_JSON_TYPE_KEY: "float", "value": "nan"}
        if np.isposinf(value):
            return {_JSON_TYPE_KEY: "float", "value": "inf"}
        if np.isneginf(value):
            return {_JSON_TYPE_KEY: "float", "value": "-inf"}
        return value
    if isinstance(value, np.generic):
        return _metadata_to_json_value(value.item())
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("ERPy epoch metadata dictionaries must use string keys")
        return {
            _JSON_TYPE_KEY: "dict",
            "items": [
                [key, _metadata_to_json_value(value[key])]
                for key in sorted(value)
            ],
        }
    if isinstance(value, list):
        return {
            _JSON_TYPE_KEY: "list",
            "items": [_metadata_to_json_value(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            _JSON_TYPE_KEY: "tuple",
            "items": [_metadata_to_json_value(item) for item in value],
        }
    if isinstance(value, Path):
        return {_JSON_TYPE_KEY: "path", "value": str(value)}
    if isinstance(value, pd.Timestamp):
        return {_JSON_TYPE_KEY: "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {_JSON_TYPE_KEY: "timedelta", "value": value.isoformat()}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported in ERPy epoch metadata")
        return {
            _JSON_TYPE_KEY: "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "items": _metadata_to_json_value(value.tolist()),
        }
    raise TypeError(
        "Unsupported ERPy epoch metadata value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _metadata_from_json_value(value: Any) -> Any:
    """Restore a tagged JSON metadata tree without importing or executing code."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, dict):
        raise ValueError("Invalid ERPy JSON metadata value")
    value_type = value.get(_JSON_TYPE_KEY)
    if value_type == "float":
        names = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}
        try:
            return float(names[value["value"]])
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid non-finite float in ERPy metadata") from exc
    if value_type in {"list", "tuple"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Invalid {value_type} in ERPy metadata")
        restored = [_metadata_from_json_value(item) for item in items]
        return restored if value_type == "list" else tuple(restored)
    if value_type == "dict":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Invalid dictionary in ERPy metadata")
        restored: dict[str, Any] = {}
        for pair in items:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or pair[0] in restored
            ):
                raise ValueError("Invalid dictionary entry in ERPy metadata")
            restored[pair[0]] = _metadata_from_json_value(pair[1])
        return restored
    if value_type == "path":
        return Path(str(value.get("value", "")))
    if value_type == "timestamp":
        return pd.Timestamp(value.get("value"))
    if value_type == "timedelta":
        return pd.Timedelta(value.get("value"))
    if value_type == "ndarray":
        dtype = np.dtype(value.get("dtype"))
        if dtype.hasobject:
            raise ValueError("Object arrays are not allowed in ERPy JSON metadata")
        shape = value.get("shape")
        if (
            not isinstance(shape, list)
            or not all(isinstance(size, int) and size >= 0 for size in shape)
        ):
            raise ValueError("Invalid array shape in ERPy metadata")
        items = _metadata_from_json_value(value.get("items"))
        array = np.asarray(items, dtype=dtype)
        expected_size = math.prod(shape) if shape else 1
        if array.size != expected_size:
            raise ValueError("Array shape does not match its ERPy metadata payload")
        return array.reshape(tuple(shape))
    raise ValueError(f"Unsupported ERPy JSON metadata type {value_type!r}")


def _encode_metadata_payload(metadata: dict[str, Any]) -> bytes:
    envelope = {
        "format": _METADATA_STORAGE_FORMAT,
        "metadata": _metadata_to_json_value(metadata),
    }
    raw = json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(raw) > _MAX_DECOMPRESSED_METADATA_BYTES:
        raise ValueError("ERPy epoch metadata exceed the 64 MiB safety limit")
    payload = zlib.compress(raw, level=9)
    if len(payload) > _MAX_COMPRESSED_METADATA_BYTES:
        raise ValueError("Compressed ERPy epoch metadata exceed the 16 MiB safety limit")
    return payload


def _decompress_metadata_payload(payload: bytes) -> bytes:
    if len(payload) > _MAX_COMPRESSED_METADATA_BYTES:
        raise ValueError("Compressed ERPy epoch metadata exceed the safety limit")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(
        payload,
        _MAX_DECOMPRESSED_METADATA_BYTES + 1,
    )
    if (
        len(raw) > _MAX_DECOMPRESSED_METADATA_BYTES
        or decompressor.unconsumed_tail
    ):
        raise ValueError("Decompressed ERPy epoch metadata exceed the safety limit")
    raw += decompressor.flush()
    if (
        len(raw) > _MAX_DECOMPRESSED_METADATA_BYTES
        or not decompressor.eof
        or decompressor.unused_data
    ):
        raise ValueError("Invalid or oversized compressed ERPy epoch metadata")
    return raw


def _decode_metadata_payload(payload: bytes) -> dict[str, Any]:
    try:
        envelope = json.loads(_decompress_metadata_payload(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, zlib.error) as exc:
        raise ValueError("Corrupt ERPy JSON metadata payload") from exc
    if not isinstance(envelope, dict) or envelope.get("format") != _METADATA_STORAGE_FORMAT:
        raise ValueError("Unsupported ERPy JSON metadata envelope")
    metadata = _metadata_from_json_value(envelope.get("metadata"))
    if not isinstance(metadata, dict):
        raise ValueError("ERPy metadata payload is not a dictionary")
    return metadata


def _write_metadata_node(path: Path, key: str, metadata: dict[str, Any]) -> None:
    import tables

    payload = _encode_metadata_payload(metadata)
    group_path = f"/{str(key).strip('/')}"
    with tables.open_file(path, mode="a") as handle:
        group = handle.get_node(group_path)
        for node_name in (_METADATA_NODE_NAME, _LEGACY_METADATA_NODE_NAME):
            if hasattr(group, node_name):
                handle.remove_node(group, node_name)
        if "metadata" in group._v_attrs._f_list():
            del group._v_attrs.metadata
        handle.create_array(
            group,
            _METADATA_NODE_NAME,
            obj=np.frombuffer(payload, dtype=np.uint8),
        )
        group._v_attrs.erpy_metadata_storage = _METADATA_STORAGE_FORMAT


def _read_metadata_node(
    path: Path,
    key: str,
    *,
    allow_unsafe_legacy_pickle: bool,
    legacy_error_type: type[UnsafeLegacyMetadataError] = UnsafeLegacyEpochMetadataError,
    metadata_description: str = "ERPy",
    unsafe_opt_in_example: str = "Epochs.from_hdf(..., allow_unsafe_legacy_pickle=True)",
) -> dict[str, Any] | None:
    import tables

    group_path = f"/{str(key).strip('/')}"
    with tables.open_file(path, mode="r") as handle:
        try:
            group = handle.get_node(group_path)
        except tables.NoSuchNodeError:
            return None
        has_v2_node = hasattr(group, _METADATA_NODE_NAME)
        has_legacy_node = hasattr(group, _LEGACY_METADATA_NODE_NAME)
        has_legacy_attribute = "metadata" in group._v_attrs._f_list()
        has_legacy_metadata = has_legacy_node or has_legacy_attribute
        if has_legacy_metadata and not allow_unsafe_legacy_pickle:
            raise legacy_error_type(
                f"Legacy {metadata_description} metadata use Python pickle and are blocked by "
                "default because loading a crafted pickle can execute arbitrary code. "
                "Only for a file you created and trust, call "
                f"{unsafe_opt_in_example}, then save it "
                "again to migrate it to the safe JSON format."
            )
        if has_v2_node:
            node = handle.get_node(group, _METADATA_NODE_NAME)
            if int(np.prod(node.shape, dtype=np.int64)) > _MAX_COMPRESSED_METADATA_BYTES:
                raise ValueError(f"ERPy metadata payload is too large: {path}")
            payload = np.asarray(node.read(), dtype=np.uint8).tobytes()
            return _decode_metadata_payload(payload)
        if not has_legacy_metadata:
            return None
        warnings.warn(
            f"Unsafe legacy {metadata_description} pickle metadata are being loaded "
            "from a trusted file; "
            "save the loaded data again to migrate it to JSON metadata.",
            RuntimeWarning,
            stacklevel=3,
        )
        if not has_legacy_node:
            return None
        node = handle.get_node(group, _LEGACY_METADATA_NODE_NAME)
        if int(np.prod(node.shape, dtype=np.int64)) > _MAX_COMPRESSED_METADATA_BYTES:
            raise ValueError(f"Legacy ERPy metadata payload is too large: {path}")
        payload = np.asarray(node.read(), dtype=np.uint8).tobytes()
    try:
        metadata = pickle.loads(_decompress_metadata_payload(payload))
    except Exception as exc:
        raise ValueError(f"Corrupt legacy ERPy metadata payload in {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Legacy ERPy metadata payload in {path} is not a dictionary")
    return metadata


@dataclass
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

    def n_trials(self) -> int:
        """Return the number of distinct trials in the epoch table."""

        return int(self.epochs_df.index.get_level_values("epoch").nunique())

    @property
    def channels(self) -> list[str]:
        """Return channel labels in their stored column order."""

        return list(self.epochs_df.columns)

    @property
    def times(self) -> np.ndarray:
        """Return unique epoch-relative sample times in seconds."""

        return self.epochs_df.index.get_level_values("time").unique().to_numpy(dtype=float)

    def get_mean_waveform(self) -> pd.DataFrame:
        """Return the trial-averaged waveform for every channel."""

        return self.epochs_df.groupby(level="time").mean()

    def get_std_waveform(self) -> pd.DataFrame:
        """Return the sample standard deviation across trials at each time."""

        return self.epochs_df.groupby(level="time").std()

    def get_sem_waveform(self) -> pd.DataFrame:
        """Return the standard error across trials at each time."""

        return self.epochs_df.groupby(level="time").sem()

    def as_array(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return samples, time coordinates, and channel labels.

        The sample array is ordered as ``(trials, times, channels)``.
        """

        times = self.epochs_df.index.get_level_values("time").unique().to_numpy(dtype=float)
        epochs = sorted(self.epochs_df.index.get_level_values("epoch").unique())
        arr = np.stack([self.epochs_df.xs(ep, level="epoch").reindex(times).to_numpy(dtype=float) for ep in epochs])
        return arr, times, list(self.epochs_df.columns)

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
        if abs(float(zero_time)) <= 1e-12 and not anchor_reports.empty and "onset_score_z" in anchor_reports.columns:
            return pd.DataFrame(
                {
                    "epoch": anchor_reports.get("epoch", pd.Series(dtype=int)).astype(int),
                    "zero_time_s": 0.0,
                    "nearest_sample_s": 0.0,
                    "artifact_score_z": pd.to_numeric(anchor_reports["onset_score_z"], errors="coerce"),
                    "zero_time_is_artifact": anchor_reports.get("onset_is_artifact", pd.Series(False, index=anchor_reports.index)).fillna(False).astype(bool),
                    "artifact_z_threshold": float(artifact_z),
                    "source": "pre_zero_anchor_report",
                }
            )

        reports = []
        for epoch_id in self.epochs_df.index.get_level_values("epoch").unique():
            trial = self.epochs_df.xs(epoch_id, level="epoch")
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
        times = self.times
        if times.size == 0:
            return {"zero_time": str(zero_time), "n_values": 0, "max_abs_uv": None, "is_zeroed": False}

        anchor_reports = self.post_artifact_anchor_report()
        if zero_time == POST_ARTIFACT_ZERO and not anchor_reports.empty:
            # The zero anchor is applied at a single common time using a short
            # baseline window; the residual is the per-trial mean over that window
            # (which is ~0 by construction), not the value at one sample.
            common = self.metadata.get("common_anchor_time_s")
            win = float(self.metadata.get("post_baseline_window_s", 0.0) or 0.0)
            anchors = pd.to_numeric(anchor_reports.get("anchor_time_s"), errors="coerce").dropna()
            if common is None or not np.isfinite(float(common)):
                common = float(anchors.median()) if not anchors.empty else 0.0
            common = float(common)
            common_idx = int(np.argmin(np.abs(times - common)))
            nearest = float(times[common_idx])
            n_win = int(self.metadata.get("post_baseline_samples") or max(round(win * float(self.sfreq)), 1))
            window_times = times[common_idx : common_idx + max(n_win, 1)]
            residuals = []
            for epoch_id in self.epochs_df.index.get_level_values("epoch").unique():
                trial = self.epochs_df.xs(epoch_id, level="epoch")
                window_mean = trial.reindex(window_times).mean(axis=0)
                residuals.extend(np.abs(window_mean.to_numpy(dtype=float)).ravel().tolist())
            finite = np.asarray(residuals, dtype=float)
            finite = finite[np.isfinite(finite)]
            max_abs = float(np.nanmax(finite)) if finite.size else np.nan
            onset_flags = anchor_reports.get("onset_is_artifact")
            settled = anchor_reports.get("settled")
            return {
                "zero_time": POST_ARTIFACT_ZERO,
                "zero_time_strategy": POST_ARTIFACT_ZERO,
                "common_anchor_s": common,
                "median_anchor_s": float(anchors.median()) if not anchors.empty else None,
                "min_anchor_s": float(anchors.min()) if not anchors.empty else None,
                "max_anchor_s": float(anchors.max()) if not anchors.empty else None,
                "nearest_sample_s": nearest,
                "post_baseline_window_s": win,
                "n_values": int(finite.size),
                "max_abs_uv": max_abs,
                "is_zeroed": bool(np.isfinite(max_abs) and max_abs <= 1e-9),
                "n_onset_artifact_epochs": int(pd.Series(onset_flags).fillna(False).astype(bool).sum()) if onset_flags is not None else 0,
                "n_settled_epochs": int(pd.Series(settled).fillna(False).astype(bool).sum()) if settled is not None else 0,
                "n_anchor_reports": int(len(anchor_reports)),
            }

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
        nearest = float(times[int(np.argmin(np.abs(times - zero_float)))])
        zero_frame = self.epochs_df.xs(nearest, level="time")
        values = zero_frame.to_numpy(dtype=float).ravel()
        finite = values[np.isfinite(values)]
        max_abs = float(np.nanmax(np.abs(finite))) if finite.size else np.nan
        return {
            "zero_time": zero_float,
            "zero_time_strategy": "fixed",
            "nearest_sample_s": nearest,
            "n_values": int(finite.size),
            "max_abs_uv": max_abs,
            "is_zeroed": bool(np.isfinite(max_abs) and max_abs <= 1e-9),
        }


class Epochs(EpochsCore):
    """Epoched stimulation data backed by a MultiIndex DataFrame."""

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

    def detect_erp_all(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Run every registered ERP detector and combine the channel results."""

        from .erp_detection import detect_erp_all

        return detect_erp_all(self, *args, **kwargs)

    def detect_erp(self, method: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Run one named ERP detector and return its channel-level results."""

        from .erp_detection import detect_erp

        return detect_erp(self, method=method, *args, **kwargs)

    def flag_artifacts(self, *args: Any, **kwargs: Any):
        """Score trial-channel responses against the artifact criteria."""

        from .bad_channels import detect_artifactual_responses

        return detect_artifactual_responses(self, *args, **kwargs)

    def reject_artifacts(self, *args: Any, **kwargs: Any) -> "Epochs":
        """Return a copy with artifact-contaminated responses rejected."""

        from .bad_channels import reject_artifactual_responses

        return reject_artifactual_responses(self, *args, **kwargs)

    def rescale(
        self,
        factor: float,
        *,
        signal_units: str = "uV",
        reason: str | None = None,
    ) -> "Epochs":
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
        return Epochs(
            epochs_df=scaled,
            sfreq=self.sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            stim_ch=self.stim_ch,
            metadata=metadata,
            zero_time=self.zero_time,
        )

    def rereference(
        self,
        *,
        method: str = "robust_mean",
        skipped_chs: Iterable[str] | None = None,
        zscore_threshold: float = 8.0,
        min_reference_channels: int = 4,
    ) -> "Epochs":
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
        return Epochs(
            epochs_df=frame,
            sfreq=self.sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            stim_ch=self.stim_ch,
            metadata=metadata,
            zero_time=self.zero_time,
        )

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
            stored = stored.astype(
                {column: target_dtype for column in numeric}
            )
        storage_dtype_name = str(np.dtype(storage_dtype)) if storage_dtype is not None else "preserve"
        md = {
            "sfreq": self.sfreq,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "baseline": self.baseline,
            "stim_ch": self.stim_ch,
            "zero_time": self.zero_time,
            **self.metadata,
            **metadata,
            "storage_dtype": storage_dtype_name,
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
            _write_metadata_node(temporary, key, md)
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
    ) -> "Epochs":
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
            md = _read_metadata_node(
                path,
                key,
                allow_unsafe_legacy_pickle=allow_unsafe_legacy_pickle,
            )
            with pd.HDFStore(path, mode="r") as store:
                df = store[key]
                if md is None and allow_unsafe_legacy_pickle:
                    md = getattr(store.get_storer(key).attrs, "metadata", {}) or {}
        except Exception as exc:
            if isinstance(exc, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Could not load ERPy epochs file {path}") from exc
        if not md:
            raise ValueError(f"ERPy epochs file is incomplete or missing metadata: {path}")
        return cls(
            epochs_df=df,
            sfreq=float(md.get("sfreq", np.nan)),
            tmin=float(md.get("tmin", df.index.get_level_values("time").min())),
            tmax=float(md.get("tmax", df.index.get_level_values("time").max())),
            baseline=tuple(md["baseline"]) if md.get("baseline") is not None else None,
            stim_ch=md.get("stim_ch"),
            metadata=md,
            zero_time=md.get("zero_time", md.get("zero_time_strategy", 0.0)),
        )


class EpochsJob:
    PYTHON_SCRIPT: str | None = None
    BASH_SCRIPT: str | None = None

    @classmethod
    def install_standard_scripts(cls) -> None:
        return None

    @classmethod
    def use_standard_scripts(cls, bash_type: str = "simple") -> None:
        cls.PYTHON_SCRIPT = None
        cls.BASH_SCRIPT = None


class EpochsRunner(EpochsJob):
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def queue_job(self, session_ids: Iterable[str] | None = None, backend: str | None = None, **kwargs: Any):
        runner = JobRunner(backend=backend, job_name=kwargs.get("job_name", "erpy_epochs"))
        sessions = list(session_ids or self.pipeline.dataloader.session_ids)
        lines = []
        for session_id in sessions:
            pairs = kwargs.get("stim_pairs") or self.pipeline.dataloader.stim_pairs(session_id)
            for stim_pair in pairs:
                event_info = self.pipeline._load_event_info(session_id, stim_pair)
                if pd.api.types.is_numeric_dtype(event_info["times"]):
                    events = event_info["times"].to_numpy()
                else:
                    parsed = pd.to_datetime(event_info["times"], errors="coerce")
                    events = parsed if parsed.notna().mean() > 0.8 else event_info["times"]
                self.pipeline.process_and_epoch(
                    session_id=session_id,
                    stim_pair=stim_pair,
                    event_times=events,
                    pipeline_steps=kwargs.get("pipeline_steps")
                    or self.pipeline.get_standard_pipeline(kwargs.get("pipeline_name", "blank_filt")),
                    tmin=float(kwargs.get("tmin", -0.5)),
                    tmax=float(kwargs.get("tmax", 1.0)),
                    baseline=kwargs.get("baseline", None),
                    zero_time=kwargs.get("zero_time", POST_ARTIFACT_ZERO),
                    save_epochs=True,
                )
                lines.append(f"{session_id}/{stim_pair}: epoched")
        return runner, "\n".join(lines), ""
