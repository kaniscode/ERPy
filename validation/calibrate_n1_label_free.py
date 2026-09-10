#!/usr/bin/env python3
"""Execute the frozen 480-family, physical-unit label-free N1 calibration.

No clinical annotations, classifiers or configurable detector thresholds enter
this experiment. Pure-null family errors and phenotype challenges are reported
separately. The JSON protocol bytes must match the acknowledged design.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_SHA256 = "0b5b0e5851eedd81b04e8f046388d656fa1922b2767c129725a27d3e03b3d9bb"
PRIMARY_PROTOCOL_SHA256 = "7b8170a51b118078c5567592b3cd1be12ef70db008cd0b02d422a629df201d21"
INFERENCE_NAMESPACE = "erpy-label-free-N1-calibration-v1"
NOISE_NAMESPACE = "erpy-label-free-N1-calibration-noise-v1"
MASTER_SEED = 20260909
CHANNELS = tuple(f"C{i}" for i in range(8))
SCENARIOS = ("early_negative", "early_positive", "late_negative", "residual_artifact")
METHODS = ("primary", "ablation")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def seed_from_material(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class FamilySpec:
    scenario: str
    n_trials: int
    noise_sd_uv: int
    amplitude_uv: int
    replicate: int

    @property
    def family_id(self) -> str:
        return (f"{self.scenario}_n{self.n_trials:02d}_sd{self.noise_sd_uv:03d}"
                f"_amp{self.amplitude_uv:03d}_rep{self.replicate:03d}")

    @property
    def seed_fields(self) -> str:
        return (f"{MASTER_SEED}|{self.scenario}|{self.n_trials}|{self.noise_sd_uv}"
                f"|{self.amplitude_uv}|{self.replicate}")

    @property
    def noise_seed_material(self) -> str:
        return f"{NOISE_NAMESPACE}|{self.seed_fields}"

    def inference_seed_material(self, channel: str) -> str:
        if channel not in CHANNELS:
            raise ValueError("Unknown calibration channel")
        return f"{INFERENCE_NAMESPACE}|{self.seed_fields}|{channel}"


def family_specs() -> list[FamilySpec]:
    """All cells, with fixed zero-based replicate indexing and stable order."""
    result = [FamilySpec("pure_null", n, sd, 0, rep)
              for n in (8, 12, 24) for sd in (25, 100) for rep in range(40)]
    result += [FamilySpec(scenario, n, sd, amp, rep)
               for scenario in SCENARIOS for n in (8, 12, 24)
               for sd in (25, 100) for amp in (100, 250) for rep in range(5)]
    return result


def signal_waveform(spec: FamilySpec, times: np.ndarray) -> np.ndarray:
    """Protocol amplitudes are peak microvolts, never RMS/SNR rescalings."""
    t = np.asarray(times, dtype=float)
    if spec.scenario == "pure_null":
        return np.zeros_like(t)
    if spec.scenario in ("early_negative", "early_positive"):
        sign = -1 if spec.scenario == "early_negative" else 1
        return sign * spec.amplitude_uv * np.exp(-.5 * ((t - .050) / .012) ** 2)
    if spec.scenario == "late_negative":
        return -spec.amplitude_uv * np.exp(-.5 * ((t - .180) / .020) ** 2)
    if spec.scenario == "residual_artifact":
        result = np.zeros_like(t)
        result[t >= 0] = -spec.amplitude_uv * np.exp(-t[t >= 0] / .025)
        return result
    raise ValueError("Unknown fixed scenario")


def generate_family(spec: FamilySpec) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stationary AR(1) components; noise SD refers to the marginal trial SD.

    PCG64 draws one (trial,time,component) array: component0 is shared and
    components1–8 are independent. Initial samples have stationary N(0,1)
    variance. Later innovations have variance 1-rho**2, avoiding burn-in.
    """
    times = np.arange(-500, 176, dtype=float) / 500
    seed = seed_from_material(spec.noise_seed_material)
    rng = np.random.Generator(np.random.PCG64(seed))
    ar = rng.standard_normal((spec.n_trials, len(times), 9))
    innovation_sd = math.sqrt(1 - .6 ** 2)
    for sample in range(1, len(times)):
        ar[:, sample, :] = .6 * ar[:, sample - 1, :] + innovation_sd * ar[:, sample, :]
    noise = spec.noise_sd_uv * (math.sqrt(.75) * ar[:, :, 1:] + math.sqrt(.25) * ar[:, :, :1])
    trials = noise.copy()
    if spec.scenario != "pure_null":
        trials[:, :, :2] += signal_waveform(spec, times)[None, :, None]
    metadata = {
        "family_id": spec.family_id, "spec": asdict(spec),
        "noise_seed": str(seed), "noise_seed_material": spec.noise_seed_material,
        "rng_bit_generator": "PCG64", "channel_names": list(CHANNELS),
        "inference_seeds": {c: str(seed_from_material(spec.inference_seed_material(c))) for c in CHANNELS},
        "inference_seed_materials": {c: spec.inference_seed_material(c) for c in CHANNELS},
        "trial_array_shape": list(trials.shape), "trial_array_sha256": array_hash(trials),
        "noise_array_sha256": array_hash(noise), "times_sha256": array_hash(times),
        "voltage_unit": "uV",
    }
    return trials, times, metadata


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def read_protocols(protocol_path: Path, primary_path: Path) -> dict:
    if file_hash(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("Synthetic protocol does not match the acknowledged SHA256")
    if file_hash(primary_path) != PRIMARY_PROTOCOL_SHA256:
        raise ValueError("Primary protocol does not match the acknowledged SHA256")
    protocols = {"synthetic": json.loads(protocol_path.read_text()),
                 "primary": json.loads(primary_path.read_text())}
    for relative in ("ERPy/n1.py", "ERPy/crp_energy.py", "ERPy/inference.py"):
        if file_hash(ROOT / relative) != protocols["primary"]["frozen_inputs"][relative]:
            raise ValueError(f"Original frozen inference/feature source changed: {relative}")
    return protocols


def source_hashes() -> dict:
    return {str(p.relative_to(ROOT)): file_hash(p) for p in (
        Path(__file__).resolve(), ROOT / "ERPy/n1_detection.py", ROOT / "ERPy/n1.py",
        ROOT / "ERPy/crp_energy.py", ROOT / "ERPy/inference.py")}


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def validate_result(result: dict, metadata: dict) -> None:
    """Reject incomplete output, changed identities or invalid probabilities."""
    if result["family_size"] != 8 or len(result["contacts"]) != 8:
        raise ValueError("Synthetic finite family must retain all eight contacts")
    for channel, row in zip(CHANNELS, result["contacts"]):
        if row.get("channel") != channel:
            raise ValueError("Detector changed channel identity or order")
        expected_seed = int(metadata["inference_seeds"][channel])
        if type(row.get("random_seed")) is not int or row["random_seed"] != expected_seed:
            raise ValueError("Detector did not preserve the exact integer seed")
        if row["provenance"]["family_input_sha256"] != metadata["trial_array_sha256"]:
            raise ValueError("Detector input provenance differs from generated array")
        for field in ("q_screened", "p_screened", "ablation_q_screened", "ablation_p_screened", "p_joint"):
            value = row.get(field)
            if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid finite-family probability: {field}")
        for field in ("detected", "ablation_detected"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"Missing explicit detector call: {field}")
        if row["p_screened"] < row["ablation_p_screened"] or row["ablation_p_screened"] < row["p_joint"]:
            raise ValueError("Primary/ablation screening decreased a joint p value")
        if row["detected"] and not row["ablation_detected"]:
            raise ValueError("Primary call is not a subset of the sole ablation")


def run_family(spec: FamilySpec, binding: dict, cache_dir: Path) -> dict:
    """One family per deterministic cache; safe to resume only exact inputs."""
    from ERPy.n1_detection import detect_n1_family
    path = cache_dir / f"{spec.family_id}.json"
    trials, times, metadata = generate_family(spec)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["binding"] != binding or cached["input"] != metadata:
            raise ValueError(f"Cached source/input mismatch: {path.name}")
        validate_result(cached["result"], metadata)
        return cached
    start = time.perf_counter()
    result = detect_n1_family(
        trials, times, list(CHANNELS), stimulation_contacts=(),
        random_seeds=[int(metadata["inference_seeds"][c]) for c in CHANNELS],
        family_id=spec.family_id, voltage_unit="uV")
    validate_result(result, metadata)
    record = {"binding": binding, "input": metadata, "result": json_ready(result),
              "elapsed_seconds": time.perf_counter() - start}
    write_json(path, record)
    return record


def wilson_interval(successes: int, total: int) -> dict:
    """Two-sided 95% Wilson score interval, including exact zero/all cases."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= positive total")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "total": total, "rate": rate,
            "ci_low": max(0., center - half), "ci_high": min(1., center + half),
            "confidence": .95, "interval": "Wilson binomial"}


def result_frames(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    contacts, families = [], []
    for record in records:
        meta = record["input"]; spec = meta["spec"]
        family = dict(spec, family_id=meta["family_id"], family_size=record["result"]["family_size"],
                      noise_seed=meta["noise_seed"], trial_array_sha256=meta["trial_array_sha256"])
        family_rows = []
        for index, row in enumerate(record["result"]["contacts"]):
            channel = row.get("channel", row.get("channel_name"))
            if channel != CHANNELS[index]:
                raise ValueError("Detector must preserve the supplied channel order and identity")
            target = spec["scenario"] != "pure_null" and index < 2
            out = dict(spec, family_id=meta["family_id"], channel=channel,
                       is_injected_target=target, is_null_contact=not target,
                       synthetic_negative_n1=target and spec["scenario"] == "early_negative",
                       random_seed=meta["inference_seeds"][channel],
                       trial_array_sha256=meta["trial_array_sha256"])
            for key, value in row.items():
                if key not in ("channel", "channel_name", "random_seed"):
                    out[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            contacts.append(out); family_rows.append(out)
        for method, field in (("primary", "detected"), ("ablation", "ablation_detected")):
            family[f"{method}_calls"] = sum(bool(r[field]) for r in family_rows)
            family[f"{method}_any_call"] = any(bool(r[field]) for r in family_rows)
            family[f"{method}_target_calls"] = sum(bool(r[field]) and r["is_injected_target"] for r in family_rows)
            family[f"{method}_null_contact_calls"] = sum(bool(r[field]) and r["is_null_contact"] for r in family_rows)
            family[f"{method}_any_target_call"] = any(bool(r[field]) and r["is_injected_target"] for r in family_rows)
            family[f"{method}_any_null_contact_call"] = any(bool(r[field]) and r["is_null_contact"] for r in family_rows)
        family["target_contacts"] = sum(r["is_injected_target"] for r in family_rows)
        family["null_contacts"] = sum(r["is_null_contact"] for r in family_rows)
        families.append(family)
    return pd.DataFrame(contacts), pd.DataFrame(families)


def summarize(families: pd.DataFrame) -> dict:
    null = families[families.scenario == "pure_null"]
    challenge = families[families.scenario != "pure_null"]
    def null_summary(group):
        return {method: wilson_interval(int(group[f"{method}_any_call"].sum()), len(group))
                for method in METHODS}
    pure = {"overall": null_summary(null), "cells": []}
    for (n, sd), group in null.groupby(["n_trials", "noise_sd_uv"], sort=True):
        pure["cells"].append({"n_trials": int(n), "noise_sd_uv": int(sd), **null_summary(group)})
    cells = []
    for keys, group in challenge.groupby(["scenario", "n_trials", "noise_sd_uv", "amplitude_uv"], sort=True):
        cell = dict(zip(("scenario", "n_trials", "noise_sd_uv", "amplitude_uv"), keys))
        cell["families"] = len(group)
        cell["targets_are_negative_n1"] = keys[0] == "early_negative"
        for method in METHODS:
            target_n = int(group.target_contacts.sum()); null_n = int(group.null_contacts.sum())
            target_calls = int(group[f"{method}_target_calls"].sum())
            null_calls = int(group[f"{method}_null_contact_calls"].sum())
            cell[method] = {
                "target_calls": target_calls, "target_contacts": target_n,
                "target_call_rate": target_calls / target_n,
                "null_contact_calls": null_calls, "null_contacts": null_n,
                "null_contact_call_rate": null_calls / null_n,
                "all_contact_calls": int(group[f"{method}_calls"].sum()),
                "all_contacts": int(group.family_size.sum()),
                "families_with_any_call": int(group[f"{method}_any_call"].sum()),
                "families_with_any_target_call": int(group[f"{method}_any_target_call"].sum()),
                "families_with_any_null_contact_call": int(group[f"{method}_any_null_contact_call"].sum()),
            }
        cells.append(cell)
    return json_ready({
        "pure_null_family_any_call": pure,
        "phenotype_challenge_cells": cells,
        "interpretation": {
            "pure_null": "Family any-call rate across independent all-noise families; Wilson95% intervals use families, not contacts, as binomial units.",
            "phenotypes": "Only early_negative targets have synthetic negative N1. Other injected targets are phenotype challenges, not uniformly original-joint-null cases. Contact call rates are descriptive; correlated contacts are not treated as independent binomial observations.",
            "scope": "Fixed stationary AR(1) physical-unit simulation, not universal N1-specific calibration or fresh clinical validation.",
        }})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--performance-check", action="store_true",
                        help="Execute only the first fixed pure-null family as an explicitly incomplete runtime check.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    protocols = read_protocols(args.protocol, args.primary_protocol)
    binding = {"synthetic_protocol_sha256": PROTOCOL_SHA256,
               "primary_protocol_sha256": PRIMARY_PROTOCOL_SHA256,
               "source_hashes": source_hashes(), "runtime": runtime_identity()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "families"; cache_dir.mkdir(exist_ok=True)
    specs = family_specs()[:1] if args.performance_check else family_specs()
    write_json(args.output_dir / "execution_plan.json", {
        "binding": binding, "protocols": protocols,
        "families": [dict(asdict(s), family_id=s.family_id,
                          noise_seed=str(seed_from_material(s.noise_seed_material)),
                          noise_seed_material=s.noise_seed_material,
                          inference_seeds={c: str(seed_from_material(s.inference_seed_material(c))) for c in CHANNELS},
                          inference_seed_materials={c: s.inference_seed_material(c) for c in CHANNELS})
                     for s in specs],
        "complete_design": not args.performance_check, "workers": args.workers,
        "rng": {"bit_generator": "PCG64", "noise_namespace": NOISE_NAMESPACE,
                "inference_namespace": INFERENCE_NAMESPACE,
                "replicates": "zero-based", "channel_names": list(CHANNELS),
                "seed_integer_format": "decimal integers without leading zeros in seed materials; exact seeds stored as decimal text",
                "AR_initialization": "stationary N(0,1) initial sample; rho=.6, innovation SD=sqrt(1-rho^2)",
                "noise_draw_order": "one standard-normal array (trial,time,9); component0 shared,1–8 independent; time recursion in increasing sample order"}})
    records = []
    started = time.perf_counter()
    if args.workers == 1:
        for spec in specs:
            records.append(run_family(spec, binding, cache_dir))
            if len(records) % 20 == 0:
                print(f"Completed {len(records)}/{len(specs)} fixed families", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_family, s, binding, cache_dir) for s in specs]
            for future in futures:
                records.append(future.result())
                if len(records) % 20 == 0:
                    print(f"Completed {len(records)}/{len(specs)} fixed families", flush=True)
    if source_hashes() != binding["source_hashes"]:
        raise ValueError("Scientific execution source changed during the run; do not publish mixed outputs")
    contacts, families = result_frames(records)
    contacts.to_csv(args.output_dir / "contact_results.csv", index=False)
    families.to_csv(args.output_dir / "family_results.csv", index=False)
    if not args.performance_check:
        if len(families) != 480 or len(contacts) != 3840:
            raise ValueError("Incomplete frozen calibration")
        write_json(args.output_dir / "summary.json", summarize(families))
    files = sorted(f for f in args.output_dir.rglob("*") if f.is_file() and f.name != "manifest.json")
    write_json(args.output_dir / "manifest.json", {
        "schema_version": 1, "binding": binding, "complete_design": not args.performance_check,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family_count": len(families), "contact_count": len(contacts),
        "elapsed_seconds": time.perf_counter() - started, "workers": args.workers,
        "runtime": runtime_identity(),
        "files": {str(f.relative_to(args.output_dir)): file_hash(f) for f in files}})
    print(f"Saved {'complete calibration' if not args.performance_check else 'incomplete runtime check'}: {args.output_dir}")


if __name__ == "__main__":
    main()
