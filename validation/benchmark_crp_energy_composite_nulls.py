#!/usr/bin/env python3
"""Exact stress test of the two one-component-null CRP-energy branches.

This benchmark contains generated data exclusively. Each independently seeded
four-contact family contains one target on a boundary branch of the joint null
and three stationary-noise references. Twelve trials keep both component
randomization tests exact. The released family-level detector API performs the
same per-contact seed derivation, eligibility checks, maximum-p conjunction,
and Benjamini-Hochberg adjustment used in ordinary analyses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

import ERPy as ep
from validation.benchmark_crp_energy import (
    BenchmarkConfig,
    _ar1_noise,
    _epochs_from_array,
    _time_axis,
    response_waveform,
    wilson_interval,
)


SCHEMA_VERSION = "1.0.0"
PROJECTION_ALT_ENERGY_NULL = "projection_alternative_energy_null"
ENERGY_ALT_PROJECTION_NULL = "energy_alternative_projection_null"
ARMS = (PROJECTION_ALT_ENERGY_NULL, ENERGY_ALT_PROJECTION_NULL)


@dataclass(frozen=True)
class CompositeNullConfig:
    """Resolved design for the focused composite-null stress test."""

    master_seed: int = 20_260_901
    n_replicates: int = 1_000
    n_trials: int = 12
    n_family_channels: int = 4
    sfreq: float = 250.0
    tmin: float = -0.36
    tmax: float = 0.34
    baseline_window: tuple[float, float] = (-0.32, -0.016)
    response_window: tuple[float, float] = (0.016, 0.32)
    artifact_interval: tuple[float, float] = (0.0, 0.016)
    noise_sd_uv: float = 10.0
    ar1_phi: float = 0.85
    trial_scale_log_sd: float = 0.18
    energy_null_log_ratio_sd: float = 0.20
    projection_shape_noise_fraction: float = 0.15
    energy_alternative_amplitude_multiplier: float = 2.0
    rms_epsilon: float = 1e-12
    alpha: float = 0.05
    n_permutations: int = 5_000
    max_exact_trials: int = 16
    max_exact_reproducibility_trials: int = 12
    confidence: float = 0.95

    def validate(self) -> None:
        """Reject settings that would weaken the intended exact benchmark."""

        if int(self.n_replicates) < 1:
            raise ValueError("n_replicates must be positive")
        if int(self.n_trials) != 12:
            raise ValueError("the release stress test requires exactly 12 trials")
        if int(self.n_family_channels) != 4:
            raise ValueError("the release stress test requires four contacts")
        if int(self.max_exact_trials) < int(self.n_trials):
            raise ValueError("the energy test must enumerate exact assignments")
        if int(self.max_exact_reproducibility_trials) < int(self.n_trials):
            raise ValueError("the projection test must enumerate exact assignments")
        if self.noise_sd_uv <= 0 or not 0 <= self.ar1_phi < 1:
            raise ValueError("noise settings are invalid")
        if self.energy_null_log_ratio_sd <= 0:
            raise ValueError("energy-null log-ratio scale must be positive")
        if self.projection_shape_noise_fraction < 0:
            raise ValueError("shape-noise fraction must be nonnegative")
        if self.energy_alternative_amplitude_multiplier <= 0:
            raise ValueError("energy-alternative amplitude must be positive")
        if self.rms_epsilon <= 0:
            raise ValueError("RMS epsilon must be positive")
        if not 0 < self.alpha < 1 or not 0 < self.confidence < 1:
            raise ValueError("alpha and confidence must lie between zero and one")

    def record(self) -> dict[str, Any]:
        """Return a complete JSON-compatible design record."""

        result = asdict(self)
        result.update(
            {
                "schema_version": SCHEMA_VERSION,
                "arms": list(ARMS),
                "testing_family": (
                    "one composite-null target plus three both-null references"
                ),
                "primary_decision": (
                    "BH-adjusted q_joint <= alpha, where "
                    "p_joint=max(p_R,p_E)"
                ),
                "exact_projection_assignments": 2 ** (self.n_trials - 1),
                "exact_energy_assignments": 2 ** self.n_trials,
            }
        )
        return result


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2, axis=1))


def _scenario_seed(config: CompositeNullConfig, arm: str, replicate: int) -> int:
    arm_code = ARMS.index(str(arm)) + 1
    sequence = np.random.SeedSequence(
        [int(config.master_seed), int(arm_code), int(replicate)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def simulate_family(
    config: CompositeNullConfig,
    *,
    arm: str,
    replicate: int,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Generate one four-contact family on a declared composite-null branch."""

    config.validate()
    if arm not in ARMS:
        raise ValueError(f"unknown composite-null arm: {arm!r}")
    seed = _scenario_seed(config, arm, replicate)
    root = np.random.SeedSequence(seed)
    simulation_stream, detector_stream = root.spawn(2)
    rng = np.random.default_rng(simulation_stream)
    detector_seed = int(detector_stream.generate_state(1, dtype=np.uint32)[0])
    base = BenchmarkConfig(
        master_seed=int(config.master_seed),
        n_replicates=1,
        trial_counts=(int(config.n_trials),),
        snr_values=(0.0,),
        artifact_fractions=(0.0,),
        n_family_channels=int(config.n_family_channels),
        sfreq=float(config.sfreq),
        tmin=float(config.tmin),
        tmax=float(config.tmax),
        baseline_window=tuple(config.baseline_window),
        response_window=tuple(config.response_window),
        artifact_interval=tuple(config.artifact_interval),
        noise_sd_uv=float(config.noise_sd_uv),
        ar1_phi=float(config.ar1_phi),
        trial_scale_log_sd=float(config.trial_scale_log_sd),
        min_clean_trials=int(config.n_trials),
        n_permutations=int(config.n_permutations),
        max_exact_trials=int(config.max_exact_trials),
        max_exact_reproducibility_trials=int(
            config.max_exact_reproducibility_trials
        ),
    )
    times = _time_axis(base)
    channels = ["target", "reference_01", "reference_02", "reference_03"]
    values = _ar1_noise(
        rng,
        (int(config.n_trials), len(times), int(config.n_family_channels)),
        phi=float(config.ar1_phi),
        scale=float(config.noise_sd_uv),
    )
    scales = rng.lognormal(
        mean=-0.5 * float(config.trial_scale_log_sd) ** 2,
        sigma=float(config.trial_scale_log_sd),
        size=(int(config.n_trials), 1, int(config.n_family_channels)),
    )
    values *= scales

    response_indices = np.flatnonzero(
        (times >= float(config.response_window[0]))
        & (times <= float(config.response_window[1]))
    )
    baseline_candidates = np.flatnonzero(
        (times >= float(config.baseline_window[0]))
        & (times <= float(config.baseline_window[1]))
    )
    if not response_indices.size or len(baseline_candidates) < len(response_indices):
        raise ValueError("matched benchmark windows are unavailable")
    baseline_indices = baseline_candidates[-len(response_indices) :]
    template = response_waveform(times[response_indices])
    template = template - float(np.mean(template))
    template /= float(np.sqrt(np.mean(template * template)))

    if arm == PROJECTION_ALT_ENERGY_NULL:
        baseline = values[:, baseline_indices, 0]
        baseline_residual = baseline - np.mean(baseline, axis=1, keepdims=True)
        baseline_rms = _rms(baseline_residual)
        response_noise = values[:, response_indices, 0]
        response_noise -= np.mean(response_noise, axis=1, keepdims=True)
        response_noise_scale = _rms(response_noise)
        response_noise /= np.maximum(response_noise_scale[:, None], 1e-12)
        shapes = (
            template[None, :]
            + float(config.projection_shape_noise_fraction) * response_noise
        )
        shapes -= np.mean(shapes, axis=1, keepdims=True)
        shapes /= np.maximum(_rms(shapes)[:, None], 1e-12)
        energy_log_ratio = rng.normal(
            0.0,
            float(config.energy_null_log_ratio_sd),
            size=int(config.n_trials),
        )
        eps = float(config.rms_epsilon)
        desired_response_rms = (
            (baseline_rms + eps) * np.exp(energy_log_ratio) - eps
        )
        if np.any(desired_response_rms <= 0.0):
            raise RuntimeError("energy-null response RMS must remain positive")
        values[:, response_indices, 0] = (
            np.mean(baseline, axis=1, keepdims=True)
            + desired_response_rms[:, None] * shapes
        )
        realized_response = values[:, response_indices, 0].copy()
        realized_response -= np.mean(realized_response, axis=1, keepdims=True)
        realized_log_ratio = (
            np.log(_rms(realized_response) + eps)
            - np.log(baseline_rms + eps)
        )
        if not np.allclose(
            realized_log_ratio,
            energy_log_ratio,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("energy-null log-RMS construction drifted")
    else:
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=int(config.n_trials),
        )
        amplitude = (
            float(config.energy_alternative_amplitude_multiplier)
            * float(config.noise_sd_uv)
        )
        values[:, response_indices, 0] += (
            signs[:, None] * amplitude * template[None, :]
        )
    return values, times, channels, detector_seed


def run_family(
    config: CompositeNullConfig,
    arm: str,
    replicate: int,
) -> dict[str, Any]:
    """Generate and evaluate one family through the released detector API."""

    values, times, channels, detector_seed = simulate_family(
        config,
        arm=arm,
        replicate=replicate,
    )
    base = BenchmarkConfig(
        n_replicates=1,
        trial_counts=(int(config.n_trials),),
        snr_values=(0.0,),
        artifact_fractions=(0.0,),
        n_family_channels=int(config.n_family_channels),
        sfreq=float(config.sfreq),
        tmin=float(config.tmin),
        tmax=float(config.tmax),
        baseline_window=tuple(config.baseline_window),
        response_window=tuple(config.response_window),
        artifact_interval=tuple(config.artifact_interval),
    )
    epochs = _epochs_from_array(values, times, channels, base)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        detections = epochs.detect_erp_all(
            methods=["crp_energy"],
            min_consensus=1,
            baseline_window=tuple(config.baseline_window),
            response_window=tuple(config.response_window),
            crp_energy={
                "alpha": float(config.alpha),
                "correction": "fdr_bh",
                "min_clean_trials": int(config.n_trials),
                "n_permutations": int(config.n_permutations),
                "max_exact_trials": int(config.max_exact_trials),
                "max_exact_reproducibility_trials": int(
                    config.max_exact_reproducibility_trials
                ),
                "canonical_energy_cv": False,
                "random_state": int(detector_seed),
                "artifact_interval": tuple(config.artifact_interval),
                "eps": float(config.rms_epsilon),
                "return_arrays": False,
            },
        )
    rows = detections[detections["method"].eq("crp_energy")].set_index(
        "channel"
    )
    if set(rows.index.astype(str)) != set(channels):
        raise RuntimeError("detector output does not contain the complete family")
    target = rows.loc["target"]
    p_r = float(target["p_crp"])
    p_e = float(target["p_energy"])
    p_joint = float(target["p_joint"])
    q_joint = float(target["q_joint"])
    if not np.isclose(p_joint, max(p_r, p_e), rtol=0.0, atol=1e-15):
        raise RuntimeError("joint p-value identity failed")
    if not bool(target["reproducibility_test_exact"]):
        raise RuntimeError("projection test was not exact")
    if not bool(target["energy_test_exact"]):
        raise RuntimeError("energy test was not exact")
    return {
        "scenario_id": f"{arm}__r{int(replicate):04d}",
        "arm": str(arm),
        "replicate": int(replicate),
        "scenario_seed": int(_scenario_seed(config, arm, replicate)),
        "detector_random_state": int(detector_seed),
        "n_trials": int(config.n_trials),
        "n_family_contacts": int(config.n_family_channels),
        "p_R": p_r,
        "p_E": p_e,
        "p_joint": p_joint,
        "q_joint": q_joint,
        "p_R_call": bool(p_r <= float(config.alpha)),
        "p_E_call": bool(p_e <= float(config.alpha)),
        "raw_joint_call": bool(p_joint <= float(config.alpha)),
        "adjusted_target_call": bool(target["significant"]),
        "any_adjusted_family_call": bool(rows["significant"].astype(bool).any()),
        "testing_family_size": int(target["testing_family_size"]),
        "projection_test_exact": bool(target["reproducibility_test_exact"]),
        "projection_randomizations": int(target["reproducibility_n_randomizations"]),
        "energy_test_exact": bool(target["energy_test_exact"]),
        "energy_randomizations": int(target["energy_n_permutations"]),
    }


def _run_payload(payload: tuple[CompositeNullConfig, str, int]) -> dict[str, Any]:
    return run_family(*payload)


def summarize(
    results: pd.DataFrame,
    *,
    confidence: float,
) -> pd.DataFrame:
    """Summarize target component and family-adjusted call rates by arm."""

    rows: list[dict[str, Any]] = []
    for arm, group in results.groupby("arm", sort=False):
        record: dict[str, Any] = {"arm": str(arm), "n_families": int(len(group))}
        for field in (
            "p_R_call",
            "p_E_call",
            "raw_joint_call",
            "adjusted_target_call",
            "any_adjusted_family_call",
        ):
            count = int(group[field].astype(bool).sum())
            low, high = wilson_interval(
                count,
                len(group),
                confidence=float(confidence),
            )
            record[f"{field}_count"] = count
            record[f"{field}_rate"] = count / len(group)
            record[f"{field}_ci_low"] = low
            record[f"{field}_ci_high"] = high
        rows.append(record)
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _report(config: CompositeNullConfig, summary: pd.DataFrame) -> str:
    lines = [
        "# CRP-energy composite-null stress test",
        "",
        "This directory contains generated data exclusively. The test evaluates",
        "the two one-component-null branches of the joint decision with 1,000",
        "independently seeded four-contact families per branch and 12 trials per",
        "contact. Both component randomization tests use exact enumeration.",
        "",
        "| Composite-null branch | pR ≤ .05 | pE ≤ .05 | max(pR,pE) ≤ .05 | Adjusted target | Any adjusted family call |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        PROJECTION_ALT_ENERGY_NULL: "Projection alternative / energy null",
        ENERGY_ALT_PROJECTION_NULL: "Energy alternative / projection null",
    }
    for row in summary.itertuples(index=False):
        values = []
        for field in (
            "p_R_call",
            "p_E_call",
            "raw_joint_call",
            "adjusted_target_call",
            "any_adjusted_family_call",
        ):
            count = int(getattr(row, f"{field}_count"))
            rate = float(getattr(row, f"{field}_rate"))
            low = float(getattr(row, f"{field}_ci_low"))
            high = float(getattr(row, f"{field}_ci_high"))
            values.append(
                f"{count}/{int(row.n_families)} ({100*rate:.1f}%; "
                f"95% CI {100*low:.1f}–{100*high:.1f}%)"
            )
        lines.append(f"| {labels[str(row.arm)]} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "The projection-alternative branch sets the response RMS relative to",
            "its matched baseline with a centrally symmetric log ratio while",
            "preserving a common waveform. The energy-alternative branch adds a",
            "randomly signed biphasic waveform whose sign symmetry satisfies the",
            "projection null while increasing response energy. The three reference",
            "contacts contain stationary independent Gaussian AR(1) noise.",
            "",
            "The stress test evaluates calibration only under these declared",
            "boundary constructions. It complements the broader both-null and",
            "both-alternative factorial in the parent benchmark directory.",
            "",
            f"ERPy compatibility identifier: `{ep.CHECKPOINT_COMPATIBILITY_ID}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def execute(
    config: CompositeNullConfig,
    *,
    output_dir: Path,
    jobs: int = 1,
) -> dict[str, Any]:
    """Run the complete deterministic stress test and write auditable outputs."""

    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        (config, arm, replicate)
        for arm in ARMS
        for replicate in range(int(config.n_replicates))
    ]
    if int(jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
            records = list(executor.map(_run_payload, payloads, chunksize=4))
    else:
        records = [_run_payload(payload) for payload in payloads]
    results = pd.DataFrame(records).sort_values(
        ["arm", "replicate"],
        kind="stable",
    )
    summary = summarize(results, confidence=float(config.confidence))
    if not results["projection_test_exact"].astype(bool).all():
        raise RuntimeError("a projection test was not exact")
    if not results["energy_test_exact"].astype(bool).all():
        raise RuntimeError("an energy test was not exact")
    if not results["testing_family_size"].eq(int(config.n_family_channels)).all():
        raise RuntimeError("a family did not contain four eligible contacts")

    results_path = output_dir / "composite_null_results.csv.gz"
    summary_path = output_dir / "composite_null_summary.csv"
    summary_json_path = output_dir / "composite_null_summary.json"
    config_path = output_dir / "composite_null_config.json"
    provenance_path = output_dir / "provenance.json"
    report_path = output_dir / "README.md"
    results.to_csv(
        results_path,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    _write_json(summary_json_path, summary.to_dict(orient="records"))
    _write_json(config_path, config.record())
    script_path = Path(__file__).resolve()
    _write_json(
        provenance_path,
        {
            "schema_version": SCHEMA_VERSION,
            "data_origin": "fully synthetic; no participant or clinical data",
            "erpy_compatibility_id": ep.CHECKPOINT_COMPATIBILITY_ID,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "script_sha256": _sha256(script_path),
            "detector_version": ep.CRP_ENERGY_DETECTOR_VERSION,
        },
    )
    report_path.write_text(_report(config, summary), encoding="utf-8")
    assets = (
        results_path,
        summary_path,
        summary_json_path,
        config_path,
        provenance_path,
        report_path,
    )
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in assets),
        encoding="utf-8",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "n_families": int(len(results)),
        "summary": summary.to_dict(orient="records"),
        "checksums": str(checksum_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release_artifacts/benchmark/composite_null_stress_test"),
    )
    parser.add_argument("--replicates", type=int, default=1_000)
    parser.add_argument("--master-seed", type=int, default=20_260_901)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = CompositeNullConfig(
        master_seed=int(args.master_seed),
        n_replicates=8 if args.quick else int(args.replicates),
    )
    result = execute(config, output_dir=args.output_dir, jobs=int(args.jobs))
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
