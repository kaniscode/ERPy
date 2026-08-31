from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd
from scipy import signal


CONTINUOUS_NPZ_FORMAT = "erpy-continuous-npz-v1"


def resample_continuous(
    df: pd.DataFrame,
    target_sfreq: float,
    *,
    source_sfreq: float | None = None,
) -> pd.DataFrame:
    """Resample every numeric channel while preserving a time-like index."""

    source_sfreq = float(source_sfreq or df.attrs.get("sfreq") or _infer_sfreq(df.index))
    target_sfreq = float(target_sfreq)
    if source_sfreq <= 0 or target_sfreq <= 0:
        raise ValueError("Sampling frequencies must be positive")

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("Continuous data contain no numeric channels")
    if target_sfreq >= source_sfreq or np.isclose(target_sfreq, source_sfreq):
        out = numeric.copy()
        out.attrs.update(df.attrs)
        out.attrs["sfreq"] = source_sfreq
        return out

    ratio = Fraction(target_sfreq / source_sfreq).limit_denominator(10_000)
    up, down = int(ratio.numerator), int(ratio.denominator)
    actual_sfreq = source_sfreq * up / down
    values = numeric.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        values = (
            pd.DataFrame(values)
            .interpolate(axis=0, limit_direction="both")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    resampled = signal.resample_poly(values, up, down, axis=0)
    index = _resampled_index(df.index, len(resampled), actual_sfreq)
    out = pd.DataFrame(resampled, index=index, columns=numeric.columns)
    out.attrs.update(df.attrs)
    out.attrs.update(
        {
            "sfreq": float(actual_sfreq),
            "source_sfreq": float(source_sfreq),
            "resample_up": up,
            "resample_down": down,
            "dropped_non_numeric_columns": [
                str(column) for column in df.columns if column not in numeric.columns
            ],
        }
    )
    return out


def save_continuous_npz(
    df: pd.DataFrame,
    path: str | Path,
    *,
    dtype: str | np.dtype = "float32",
    compressed: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save continuous numeric channels in ERPy's portable NPZ format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("Continuous data contain no numeric channels")

    index_kind, index_values, index_metadata = _encode_index(numeric.index)
    attrs = {**df.attrs, **(metadata or {}), **index_metadata}
    metadata_bytes = json.dumps(
        attrs,
        sort_keys=True,
        default=_json_default,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = {
        "format": np.asarray(CONTINUOUS_NPZ_FORMAT),
        "data": numeric.to_numpy(dtype=np.dtype(dtype), copy=True),
        "columns": np.asarray(numeric.columns.astype(str), dtype=str),
        "index_kind": np.asarray(index_kind),
        "index_values": index_values,
        "metadata_json": np.frombuffer(metadata_bytes, dtype=np.uint8),
    }

    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix or '.npz'}")
    try:
        with temporary.open("wb") as handle:
            if compressed:
                np.savez_compressed(handle, **payload)
            else:
                np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_continuous_npz(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Load an ERPy continuous NPZ file without enabling pickle deserialization."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        file_format = str(np.asarray(payload["format"]).item())
        if file_format != CONTINUOUS_NPZ_FORMAT:
            raise ValueError(f"Unsupported ERPy continuous format {file_format!r}: {path}")
        data = np.asarray(payload["data"])
        columns = np.asarray(payload["columns"]).astype(str).tolist()
        index_kind = str(np.asarray(payload["index_kind"]).item())
        index_values = np.asarray(payload["index_values"])
        metadata_raw = np.asarray(payload["metadata_json"], dtype=np.uint8).tobytes()

    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"Corrupt ERPy continuous array dimensions: {path}")
    if len(index_values) != len(data):
        raise ValueError(f"Corrupt ERPy continuous index length: {path}")
    if nrows is not None:
        stop = max(int(nrows), 0)
        data = data[:stop]
        index_values = index_values[:stop]

    metadata = json.loads(metadata_raw.decode("utf-8")) if metadata_raw else {}
    index = _decode_index(index_kind, index_values, metadata)
    out = pd.DataFrame(data, index=index, columns=columns)
    out.attrs.update(metadata)
    return out


def _infer_sfreq(index: pd.Index) -> float:
    if len(index) < 2:
        raise ValueError("Cannot infer sampling frequency from fewer than two rows")
    if isinstance(index, pd.DatetimeIndex):
        delta_s = np.diff(index.asi8.astype(np.float64)) / 1e9
    else:
        values = pd.to_numeric(pd.Series(index), errors="coerce").to_numpy(dtype=float)
        delta_s = np.diff(values)
    delta_s = delta_s[np.isfinite(delta_s) & (delta_s > 0)]
    if not len(delta_s):
        raise ValueError("Cannot infer sampling frequency from the index")
    return float(1.0 / np.median(delta_s))


def _resampled_index(index: pd.Index, length: int, sfreq: float) -> pd.Index:
    offsets = pd.to_timedelta(np.arange(length, dtype=float) / float(sfreq), unit="s")
    if isinstance(index, pd.DatetimeIndex):
        return pd.DatetimeIndex(index[0] + offsets, name=index.name)
    values = pd.to_numeric(pd.Series(index), errors="coerce").to_numpy(dtype=float)
    if len(values) and np.isfinite(values[0]):
        return pd.Index(values[0] + np.arange(length, dtype=float) / float(sfreq), name=index.name)
    return pd.Index(np.arange(length, dtype=float) / float(sfreq), name=index.name or "time")


def _encode_index(index: pd.Index) -> tuple[str, np.ndarray, dict[str, Any]]:
    if isinstance(index, pd.DatetimeIndex):
        timezone = str(index.tz) if index.tz is not None else None
        return "datetime64ns", index.asi8.astype(np.int64), {"index_timezone": timezone}
    values = pd.to_numeric(pd.Series(index), errors="coerce")
    if values.notna().all():
        return "numeric", values.to_numpy(dtype=np.float64), {}
    return "string", np.asarray(index.astype(str), dtype=str), {}


def _decode_index(kind: str, values: np.ndarray, metadata: dict[str, Any]) -> pd.Index:
    if kind == "datetime64ns":
        index = pd.to_datetime(values.astype(np.int64), unit="ns", utc=True)
        timezone = metadata.get("index_timezone")
        if timezone:
            return pd.DatetimeIndex(index.tz_convert(timezone))
        return pd.DatetimeIndex(index.tz_localize(None))
    if kind == "numeric":
        return pd.Index(values.astype(float), name="time")
    if kind == "string":
        return pd.Index(values.astype(str))
    raise ValueError(f"Unsupported ERPy continuous index kind {kind!r}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)
