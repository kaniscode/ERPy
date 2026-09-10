#!/usr/bin/env python3
"""Run fixed ERPy settings on recovered public epochs without reading labels."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ERPy.crp_energy import CRPEnergyConfig, run_crp_energy_array
from ERPy.inference import fdr_bh

WINDOWS = {"broad_15_300ms": ((.015, .3), (.0, .015)),
           "early_10_90ms": ((.010, .09), (.0, .009))}
SEED_NAMESPACE = "ERPy-ds004774-v1|20260907"
NORMALIZATION_BASELINE = (-.5, -.02)
FAMILY_ALPHA = .05
CACHE_SCHEMA_VERSION = 2
PREDICTION_COLUMNS = [
    "subject", "site", "stimpair", "channel", "configuration",
    "n_trials_total", "n_trials_clean", "p_reproducibility", "p_energy", "p_joint",
    "rms_ratio_db", "reproducibility_statistic", "energy_statistic",
    "component_classification", "qc_status", "notes", "projection_exact", "energy_exact",
    "random_seed", "epoch_sha256", "elapsed_seconds", "q_joint", "detected", "evaluable",
    "family_size",
    "response_start_s", "response_stop_s", "baseline_start_s", "baseline_stop_s",
    "n_response_samples", "n_baseline_samples", "clean_trial_indices",
]


def configurations() -> dict[str, CRPEnergyConfig]:
    """Resolve every detector default before both fingerprinting and inference."""
    return {
        name: CRPEnergyConfig(response_window=window, baseline_window=(-1., -.1),
                              artifact_interval=artifact, min_clean_trials=8,
                              n_permutations=5000, canonical_energy_cv=False)
        for name, (window, artifact) in WINDOWS.items()
    }


def _sha256(path: Path) -> str:
    # Stream files on all supported Python versions (file_digest needs 3.11).
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def code_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    # Hash the whole ERPy package so a new internal import cannot evade receipt
    # invalidation. Relative paths let an unchanged checkout move directories.
    paths = [Path(__file__).resolve(), *sorted((root / "ERPy").rglob("*.py"))]
    return {str(path.relative_to(root)): _sha256(path) for path in paths}


def dependency_fingerprint() -> dict:
    builds = {}
    for name, module in (("numpy", np), ("scipy", scipy)):
        with redirect_stdout(io.StringIO()) as stream:
            module.show_config()
        builds[name] = stream.getvalue()
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "machine": platform.machine(), "byteorder": sys.byteorder,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "numerical_builds": builds,
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")},
    }


def prediction_fingerprint(source_sha: str, subject: str, pair: str, names: list[str],
                           configs: dict[str, CRPEnergyConfig]) -> dict:
    return {
        "epoch_sha256": source_sha, "subject": subject, "stimpair": pair,
        "retained_channels": names,
        "configurations": {name: asdict(config) for name, config in configs.items()},
        "configuration_order": list(configs),
        "normalization": {"baseline_s": NORMALIZATION_BASELINE,
                          "operation": "per-trial nanmedian; half-open rounded sample slice"},
        "family": {"method": "fdr_bh", "alpha": FAMILY_ALPHA,
                   "scope": "all retained non-stimulation contacts per pair and configuration; finite p_joint"},
        "seed_policy": {"namespace": SEED_NAMESPACE,
                        "algorithm": "SHA256 UTF-8 namespace|subject|pair|channel; first 8 bytes big-endian; mask to 63 bits",
                        "configuration_sharing": "same channel seed in both configurations; overrides config random_state",
                        "channel_seeds": {channel: seed_for(subject, pair, channel)
                                          for channel in names if channel not in pair.split("-")}},
        "code_sha256": code_fingerprint(), "dependencies": dependency_fingerprint(),
    }


def receipt_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".receipt.json")


def validate_cache(out: Path, fingerprint: dict) -> None:
    """A receipt is a commit marker; incomplete/legacy caches are never reused."""
    receipt = receipt_path(out)
    try:
        if not out.is_file() or not receipt.is_file():
            raise ValueError("missing prediction CSV or integrity receipt (including legacy caches)")
        saved = json.loads(receipt.read_text())
        expected_sha = hashlib.sha256(_json_bytes(fingerprint)).hexdigest()
        if saved["schema_version"] != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported receipt schema")
        if (saved["fingerprint_sha256"] != expected_sha or
                _json_bytes(saved["fingerprint"]) != _json_bytes(fingerprint)):
            raise ValueError("epoch, settings, code, dependencies, or seed policy changed")
        if saved["output_sha256"] != _sha256(out):
            raise ValueError("prediction CSV integrity hash mismatch")
        cached = pd.read_csv(out)
        if list(cached.columns) != PREDICTION_COLUMNS or len(cached) != saved["records"]:
            raise ValueError("prediction CSV schema or record count mismatch")
        channels = fingerprint["seed_policy"]["channel_seeds"]
        expected_keys = {(fingerprint["subject"], fingerprint["stimpair"], channel, name)
                         for name in fingerprint["configurations"] for channel in channels}
        keys = list(cached[["subject", "stimpair", "channel", "configuration"]].itertuples(index=False, name=None))
        if len(keys) != len(expected_keys) or set(keys) != expected_keys:
            raise ValueError("prediction CSV has missing, duplicate, or unexpected records")
        if len(cached) and (set(cached.epoch_sha256) != {fingerprint["epoch_sha256"]} or
                           any(row.random_seed != channels[row.channel] for row in cached.itertuples())):
            raise ValueError("prediction CSV epoch or random seeds differ from receipt")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Unusable prediction cache at {out}: {exc}; use a fresh output directory") from exc


def write_predictions(out: Path, records: list[dict], fingerprint: dict) -> None:
    """Stage both files on the destination filesystem, then publish receipt last.

    Each replacement is atomic. A crash between replacements leaves an
    uncommitted CSV that validate_cache rejects rather than silently trusts.
    """
    staged = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=out.parent,
                                         prefix="." + out.name + ".", suffix=".tmp", delete=False) as stream:
            csv_temp = Path(stream.name)
            staged.append(csv_temp)
            pd.DataFrame(records, columns=PREDICTION_COLUMNS).to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {"schema_version": CACHE_SCHEMA_VERSION, "fingerprint": fingerprint,
                   "fingerprint_sha256": hashlib.sha256(_json_bytes(fingerprint)).hexdigest(),
                   "output_sha256": _sha256(csv_temp), "records": len(records)}
        with tempfile.NamedTemporaryFile(mode="wb", dir=out.parent, prefix="." + out.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            receipt_temp = Path(stream.name)
            staged.append(receipt_temp)
            stream.write(_json_bytes(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        csv_temp.replace(out)
        receipt_temp.replace(receipt_path(out))
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def seed_for(subject: str, pair: str, channel: str) -> int:
    value = f"{SEED_NAMESPACE}|{subject}|{pair}|{channel}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def load_epochs(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str, str, dict]:
    with np.load(path, allow_pickle=False) as saved:
        values = np.asarray(saved["values"] if "values" in saved else saved["trials"], dtype=np.float64)
        times = np.asarray(saved["times"], dtype=float)
        names = np.asarray(saved["channels"]).astype(str)
        subject, pair = str(saved["subject"]), str(saved["stimpair"])
        metadata = json.loads(str(saved["metadata"] if "metadata" in saved else saved["metadata_json"]))
        keep = np.ones(len(names), dtype=bool)
        if "channel_types" in saved:
            keep &= np.char.upper(saved["channel_types"].astype(str)) == "ECOG"
        if "channel_status" in saved:
            keep &= np.char.lower(saved["channel_status"].astype(str)) == "good"
        if "channel_units" in saved:
            units = saved["channel_units"].astype(str)
            if any(u not in {"µV", "uV", "microvolts"} for u in units[keep]):
                raise ValueError("Retained ECoG channels must be in microvolts")
        names, values = names[keep].tolist(), values[:, :, keep]
    if values.shape[1:] != (len(times), len(names)):
        raise ValueError("Expected trial × time × channel data")
    # Match the released preprocessing's half-open, rounded sample slice.
    # Applying this to already centered data only removes a constant offset.
    fs = 1 / float(np.median(np.diff(times)))
    b0 = round((NORMALIZATION_BASELINE[0] - times[0]) * fs)
    b1 = round((NORMALIZATION_BASELINE[1] - times[0]) * fs)
    if b0 < 0 or b1 > len(times) or b0 >= b1:
        raise ValueError("Normalization baseline is unavailable")
    values -= np.nanmedian(values[:, b0:b1, :], axis=1, keepdims=True)
    return values, times, names, subject, pair, metadata


def predict_file(path: Path, destination: Path) -> dict:
    started = time.perf_counter()
    source_sha = _sha256(path)
    values, times, names, subject, pair, metadata = load_epochs(path)
    out = destination / subject / (pair.replace("/", "_") + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    fingerprint = prediction_fingerprint(source_sha, subject, pair, names, configs)
    if _sha256(path) != source_sha:
        raise ValueError(f"Epoch file changed while loading: {path}")
    if out.exists() or receipt_path(out).exists():
        validate_cache(out, fingerprint)
        return {"subject": subject, "stimpair": pair, "cached": True}
    stim = set(pair.split("-"))
    if len(stim) != 2 or not stim.issubset(names):
        return {"subject": subject, "stimpair": pair, "excluded": "Stimulation pair includes a bad or non-ECoG contact"}
    records = []
    for window_name, config in configs.items():
        family = []
        for index, channel in enumerate(names):
            if channel in stim:
                continue
            seed = seed_for(subject, pair, channel)
            channel_started = time.perf_counter()
            result = run_crp_energy_array(values[:, :, index], times, channel=channel,
                                          config=replace(config, random_state=seed))
            family.append({"subject": subject, "site": "Mayo" if subject.startswith("MAYO") else "UMCU",
                           "stimpair": pair, "channel": channel, "configuration": window_name,
                           "n_trials_total": result.n_trials_total, "n_trials_clean": result.n_trials_clean,
                           "response_start_s": result.response_window[0] if result.n_response_samples else np.nan,
                           "response_stop_s": result.response_window[1] if result.n_response_samples else np.nan,
                           "baseline_start_s": result.baseline_window[0] if result.n_baseline_samples else np.nan,
                           "baseline_stop_s": result.baseline_window[1] if result.n_baseline_samples else np.nan,
                           "n_response_samples": result.n_response_samples,
                           "n_baseline_samples": result.n_baseline_samples,
                           "clean_trial_indices": json.dumps(result.clean_trial_indices.tolist(), separators=(",", ":")),
                           "p_reproducibility": result.p_crp, "p_energy": result.p_energy,
                           "p_joint": result.p_joint, "rms_ratio_db": result.rms_ratio_db,
                           "reproducibility_statistic": result.crp_statistic,
                           "energy_statistic": result.energy_statistic,
                           "component_classification": result.classification,
                           "qc_status": result.qc_status, "notes": result.notes,
                           "projection_exact": result.reproducibility_test_exact,
                           "energy_exact": result.energy_test_exact,
                           "random_seed": seed, "epoch_sha256": source_sha,
                           "elapsed_seconds": time.perf_counter() - channel_started})
        if family:
            adjusted, calls = fdr_bh([r["p_joint"] for r in family], alpha=FAMILY_ALPHA)
            size = int(np.isfinite([r["p_joint"] for r in family]).sum())
            for row, q, call in zip(family, adjusted, calls):
                row.update(q_joint=float(q), detected=bool(call), evaluable=bool(np.isfinite(q)), family_size=size)
            records.extend(family)
    if _sha256(path) != source_sha or code_fingerprint() != fingerprint["code_sha256"]:
        raise ValueError("Epoch or inference source changed during prediction; no output saved")
    write_predictions(out, records, fingerprint)
    return {"subject": subject, "stimpair": pair, "records": len(records),
            "seconds": time.perf_counter() - started, "source_epoch_sha256": source_sha}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epochs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--subject", action="append")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    files = sorted(args.epochs.glob("*/*.npz"))
    if args.subject:
        files = [p for p in files if p.parent.name in args.subject]
    if not files:
        parser.error("No matching epoch files found")
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / ("run_" + "_".join(args.subject or ["all"]) + ".jsonl")
    with ProcessPoolExecutor(max_workers=args.workers) as pool, log.open("a") as stream:
        pending = [pool.submit(predict_file, p, args.output) for p in files]
        for count, future in enumerate(as_completed(pending), 1):
            result = future.result()
            stream.write(json.dumps(result) + "\n"); stream.flush()
            if count % 10 == 0 or count == len(files):
                print(f"Completed {count}/{len(files)} stimulation pairs", flush=True)
