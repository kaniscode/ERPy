"""Safe, deterministic storage of ERPy metadata inside HDF5 files.

Epoch caches and processed recordings keep their audit metadata as a
zlib-compressed, tagged JSON document stored next to the signal table. Unlike
the pickle attributes written by pre-1.0 releases, this format can be read
without executing code. Legacy pickle metadata are refused unless the caller
explicitly opts in for a file they created and trust.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
            "items": [[key, _metadata_to_json_value(value[key])] for key in sorted(value)],
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
    raise TypeError(f"Unsupported ERPy epoch metadata value {type(value).__module__}.{type(value).__qualname__}")


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
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str) or pair[0] in restored:
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
        if not isinstance(shape, list) or not all(isinstance(size, int) and size >= 0 for size in shape):
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
    if len(raw) > _MAX_DECOMPRESSED_METADATA_BYTES or decompressor.unconsumed_tail:
        raise ValueError("Decompressed ERPy epoch metadata exceed the safety limit")
    raw += decompressor.flush()
    if len(raw) > _MAX_DECOMPRESSED_METADATA_BYTES or not decompressor.eof or decompressor.unused_data:
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
