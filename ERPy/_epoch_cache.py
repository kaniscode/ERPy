"""Conservative content identity for automatic facade epoch reuse.

This is a local cache check, not a signed provenance record or a transaction
against concurrent writers. All source bytes are read on each check. In-memory
configuration/metadata are authoritative until a new DataLoader is constructed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from .epochs import _metadata_to_json_value
from .io import parse_brainvision_header


def file_digest(path: Path) -> str:
    """Hash full contents and reject an ordinary concurrent file replacement."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError("An epoch input changed during content verification; retry after writes finish")
    return digest.hexdigest()


def _digest(value) -> str:
    payload = json.dumps(_metadata_to_json_value(value), sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _files(paths) -> list:
    return [(str(path.resolve()), file_digest(path)) for path in paths]


def _raw_dependencies(paths):
    dependencies = []
    verifiable = True
    for path in paths:
        dependencies.append(path)
        if path.suffix.lower() == ".vhdr":
            header = parse_brainvision_header(path.read_text(encoding="utf-8", errors="ignore"))
            # Resolve the binary exactly as DataLoader._load_brainvision_window.
            dependencies.append(path.with_name(str(header.get("data_file") or path.with_suffix(".eeg").name)))
            marker = header.get("marker_file")
            if marker:
                marker_path = path.parent / str(marker)
                # Markers are not read by the signal loader; bind their presence
                # and contents when supplied, without requiring an unused file.
                if marker_path.exists():
                    dependencies.append(marker_path)
        elif path.suffix.lower() == ".nwb":
            # NWB/HDF5 can refer to external datasets. Until all such links are
            # enumerated, load normally but never automatically reuse epochs.
            verifiable = False
        elif not path.name.lower().endswith((".csv", ".tsv", ".csv.gz", ".tsv.gz", ".edf", ".bdf")):
            verifiable = False
    return dependencies, verifiable


def epoch_cache_identity(pipeline, session_id, stim_pair, steps, event_bytes, options):
    """Return (identity or None, resolved input source).

    Default ``auto`` uses registered raw files when present, bypassing unbound
    memory/window caches. If no registered raw file is available, a local window
    export can be the explicit effective source. Its identity does not certify
    agreement with an unavailable original recording. Custom processing hooks
    and NWB external dependencies disable automatic reuse.
    """
    loader = pipeline.dataloader
    requested = options.get("input_source", "auto")
    raw_file = options.get("raw_file")
    raw_paths = loader.get_raw_files(session_id, stim_pair, raw_file=raw_file,
                                     stim_start=options.get("stim_start"))
    caches = loader._raw_cache_candidates(session_id, stim_pair, raw_file=raw_file,
                                           allow_legacy_cache=False)
    cache_path = next((path for path in caches if path.is_file()), None)
    processed = pipeline._processed_path(session_id, stim_pair, steps)
    verifiable = True
    if requested == "processed" and processed.is_file() and not options.get("overwrite", False):
        source, paths = "processed", [processed]
    elif requested in {"csv", "cache"} or (requested == "auto" and not any(p.exists() for p in raw_paths) and cache_path):
        if cache_path is None:
            raise FileNotFoundError("No current raw-window cache is available")
        source, paths = "cache", [cache_path]
        sidecar = loader._raw_cache_metadata_path(cache_path)
        if sidecar.exists():
            paths.append(sidecar)
    else:
        source = "raw"
        if not raw_paths:
            raise FileNotFoundError("No current raw source is registered")
        paths, verifiable = _raw_dependencies(raw_paths)

    # Unknown registered callables may depend on closures or external state.
    from .pipeline import Pipeline
    for name, _ in steps:
        hook = pipeline.registry[name]
        function = getattr(hook, "__func__", None)
        if (function is None or getattr(hook, "__self__", None) is not pipeline
                or function is not getattr(Pipeline, function.__name__, None)
                or function.__module__ != Pipeline.__module__):
            verifiable = False

    source_files = _files(paths)  # Missing/unreadable required inputs fail closed.
    if not verifiable:
        return None, source
    try:
        package_root = Path(__file__).parent
        implementation = _files(sorted(package_root.rglob("*.py")))
        versions = {name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scipy", "h5py", "tables")}
        if any(path.suffix.lower() in {".edf", ".bdf"} for path in paths):
            versions["mne"] = importlib.metadata.version("mne")
        # Bind all effective metadata, including bad-channel flags, scale,
        # sampling frequency, acquisition selection and window configuration.
        metadata = {name: getattr(loader, name).to_dict(orient="split")
                    for name in ("raw_meta", "stim_meta", "elec_meta")}
        identity = {
            "schema": "erpy-facade-epoch-content-v1",
            "source_kind": source,
            "events_sha256": hashlib.sha256(event_bytes).hexdigest(),
            "source_sha256": _digest(source_files),
            "settings_sha256": _digest({"session_id": session_id, "stim_pair": stim_pair,
                                          "patient_id": loader.patient_id, "steps": steps,
                                          "options": options, "config": loader.config,
                                          "metadata": metadata}),
            "implementation_sha256": _digest({"files": implementation, "versions": versions,
                                                "python": sys.version}),
        }
    except (TypeError, ValueError, importlib.metadata.PackageNotFoundError):
        # Unsupported settings must never be represented by a lossy repr().
        return None, source
    return identity, source
