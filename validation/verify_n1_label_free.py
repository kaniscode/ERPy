#!/usr/bin/env python3
"""Verify and replay the curated public fixed-rule N1 evidence.

The default replay recomputes label-blind calls from pinned early inference and
morphology, then runs the separate human-reference scorer. It does not reload
clinical voltage recordings or fit classifiers. Calibration input arrays and
table aggregates are always checked; --full-calibration also reruns inference.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.calibrate_n1_label_free import (
    CHANNELS, PRIMARY_PROTOCOL_SHA256, PROTOCOL_SHA256, family_specs,
    generate_family, json_ready, read_protocols, seed_from_material,
    summarize, validate_result)

DEFAULT_EVIDENCE = ROOT / "validation/n1_label_free"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(base: Path, relative: str) -> Path:
    path = base / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Manifest path escapes its declared root: {relative}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or linked evidence file: {relative}")
    return path


def check_artifacts(base: Path, files: dict) -> int:
    for relative, identity in files.items():
        path = checked_path(base, relative)
        if path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
            raise ValueError(f"Evidence identity mismatch: {relative}")
    actual = {p.relative_to(base).as_posix() for p in base.rglob("*")
              if p.is_file() and p.name not in ("artifact_manifest.json", ".DS_Store")}
    if actual != set(files):
        raise ValueError("Evidence directory has missing or unlisted files")
    return len(files)


def check_repository_bindings(bindings: dict) -> int:
    for relative, expected in bindings.items():
        if sha(checked_path(ROOT, relative)) != expected:
            raise ValueError(f"Repository source/input identity mismatch: {relative}")
    return len(bindings)


def check_embedded_receipts(evidence: Path) -> dict:
    protocol_dir = evidence / "protocols"
    read_protocols(protocol_dir / "SYNTHETIC_CALIBRATION_PROTOCOL.json",
                   protocol_dir / "LABEL_FREE_N1_PROTOCOL.json")
    ack = json.loads((protocol_dir / "protocol_acknowledgment.json").read_text())
    if (ack["primary_protocol_sha256"] != PRIMARY_PROTOCOL_SHA256
            or ack["calibration_protocol_sha256"] != PROTOCOL_SHA256
            or ack["new_calls_before_acknowledgment"]):
        raise ValueError("Protocol acknowledgment mismatch")
    prediction = json.loads((evidence / "empirical_predictions/prediction_manifest.json").read_text())
    scoring = json.loads((evidence / "empirical_scoring/summary.json").read_text())
    score_manifest = json.loads((evidence / "empirical_scoring/manifest.json").read_text())
    if (prediction["human_reference_columns_read"] or prediction["archived_or_model_call_columns_read"]
            or scoring["algorithm_reference_columns_used"]):
        raise ValueError("Prediction/reference separation contract failed")
    for item in (prediction, scoring, score_manifest):
        if item["protocol_sha256"] != PRIMARY_PROTOCOL_SHA256:
            raise ValueError("Empirical protocol identity mismatch")
    for folder, manifest in (("empirical_predictions", prediction), ("empirical_scoring", score_manifest)):
        for name, expected in manifest["outputs"].items():
            if sha(checked_path(evidence / folder, name)) != expected:
                raise ValueError(f"Embedded output receipt mismatch: {folder}/{name}")
    for receipt in (prediction, scoring):
        check_repository_bindings(receipt["source_hashes"])
    if scoring["input_hashes"]["prediction_manifest"] != sha(evidence / "empirical_predictions/prediction_manifest.json"):
        raise ValueError("Scoring does not bind the included prediction manifest")
    nominal = json.loads((evidence / "nominal_stage/summary.json").read_text())
    nominal_manifest = json.loads((evidence / "nominal_stage/manifest.json").read_text())
    if (nominal["primary_protocol_sha256"] != PRIMARY_PROTOCOL_SHA256
            or not nominal["primary_method_unchanged"] or nominal["screen_is_fdr_adjusted"]
            or nominal_manifest["role"] != "post_result_nominal_screen_diagnostic"
            or nominal["input_scoring_manifest_sha256"] != sha(evidence / "empirical_scoring/manifest.json")):
        raise ValueError("Post-result nominal-stage provenance/role changed")
    check_repository_bindings(nominal["source_hashes"])
    for name, expected in nominal_manifest["outputs"].items():
        if sha(checked_path(evidence / "nominal_stage", name)) != expected:
            raise ValueError("Post-result nominal-stage output changed")
    return {"prediction_records": prediction["records"], "families": prediction["families"],
            "finite_inference": prediction["finite_inference"],
            "paired_scoring_records": scoring["paired_records"],
            "label_blind_prediction_and_human_only_reference_contract": True}


def verify_calibration(evidence: Path) -> dict:
    base = evidence / "calibration"
    curation = json.loads((base / "curation_manifest.json").read_text())
    plan = json.loads((base / "execution_plan.json").read_text())
    if not plan["complete_design"] or len(plan["families"]) != 480:
        raise ValueError("Incomplete calibration plan")
    check_repository_bindings(plan["binding"]["source_hashes"])
    for key, filename in (("primary", "LABEL_FREE_N1_PROTOCOL.json"),
                          ("synthetic", "SYNTHETIC_CALIBRATION_PROTOCOL.json")):
        if plan["protocols"][key] != json.loads((evidence / "protocols" / filename).read_text()):
            raise ValueError("Calibration plan and copied protocol differ")
    raw_csv = gzip.decompress((base / "contact_results.csv.gz").read_bytes())
    if hashlib.sha256(raw_csv).hexdigest() != curation["contact_table"]["original_uncompressed_sha256"]:
        raise ValueError("Compressed public contact table does not preserve the source bytes")
    contacts = pd.read_csv(io.BytesIO(raw_csv), dtype={"random_seed": str}, float_precision="round_trip")
    families = pd.read_csv(base / "family_results.csv", dtype={"noise_seed": str}, float_precision="round_trip")
    if len(contacts) != 3840 or len(families) != 480 or families.family_id.nunique() != 480:
        raise ValueError("Frozen calibration membership changed")
    grouped = {key: group for key, group in contacts.groupby("family_id", sort=False)}
    family_index = families.set_index("family_id")
    for spec, planned in zip(family_specs(), plan["families"]):
        trials, times, metadata = generate_family(spec)
        expected_plan = dict(asdict(spec), family_id=spec.family_id,
            noise_seed=metadata["noise_seed"], noise_seed_material=metadata["noise_seed_material"],
            inference_seeds=metadata["inference_seeds"], inference_seed_materials=metadata["inference_seed_materials"])
        if planned != expected_plan:
            raise ValueError("Fixed RNG seed plan changed")
        family = family_index.loc[spec.family_id]
        group = grouped[spec.family_id]
        if (list(group.channel) != list(CHANNELS) or family.family_size != 8
                or family.noise_seed != metadata["noise_seed"]
                or family.trial_array_sha256 != metadata["trial_array_sha256"]
                or not group.trial_array_sha256.eq(metadata["trial_array_sha256"]).all()):
            raise ValueError("Calibration input or family identity changed")
        rows = []
        for row in group.to_dict("records"):
            if row["random_seed"] != metadata["inference_seeds"][row["channel"]]:
                raise ValueError("Exact calibration inference seed changed")
            row["random_seed"] = int(row["random_seed"])
            row["provenance"] = json.loads(row["provenance"])
            rows.append(row)
        validate_result({"family_size": 8, "contacts": rows}, metadata)
        for key, value in asdict(spec).items():
            if family[key] != value or not group[key].eq(value).all():
                raise ValueError("Calibration cell allocation changed")
        target = np.array([spec.scenario != "pure_null" and i < 2 for i in range(8)])
        truth = target & (spec.scenario == "early_negative")
        if (not np.array_equal(group.is_injected_target, target)
                or not np.array_equal(group.is_null_contact, ~target)
                or not np.array_equal(group.synthetic_negative_n1, truth)):
            raise ValueError("Phenotype ground truth was reclassified")
        if family.target_contacts != target.sum() or family.null_contacts != (~target).sum():
            raise ValueError("Incorrect contact denominators")
        for method, field in (("primary", "detected"), ("ablation", "ablation_detected")):
            call = group[field].to_numpy(bool)
            expected = {"calls": int(call.sum()), "any_call": bool(call.any()),
                        "target_calls": int((call & target).sum()),
                        "null_contact_calls": int((call & ~target).sum()),
                        "any_target_call": bool((call & target).any()),
                        "any_null_contact_call": bool((call & ~target).any())}
            if any(family[f"{method}_{k}"] != v for k, v in expected.items()):
                raise ValueError("Family aggregate differs from the complete contact table")
    if summarize(families) != json.loads((base / "summary.json").read_text()):
        raise ValueError("Calibration aggregate/Wilson summary does not reproduce")
    return {"families": 480, "contacts": 3840, "synthetic_arrays_regenerated": 480,
            "exact_noise_and_inference_seeds_verified": True,
            "contact_to_family_to_summary_reproduced": True,
            "new_inference_performed": False}


def compare_secondary_metrics(evidence: Path) -> dict:
    old = pd.read_csv(ROOT / "validation/frozen_results/n1/nested/metrics.csv", float_precision="round_trip")
    current = pd.read_csv(evidence / "empirical_scoring/all_candidate_metrics.csv", float_precision="round_trip")
    methods = ["archived_ER_detect", "N1_logistic_morphology", "N1_logistic_hybrid"]
    old = old[old.method.isin(methods)].sort_values(["method", "metric"]).reset_index(drop=True)
    current = current[current.method.isin(methods)].sort_values(["method", "metric"]).reset_index(drop=True)
    if len(old) != 18 or len(current) != 18:
        raise ValueError("Saved comparator metric membership changed")
    pd.testing.assert_frame_equal(old[["method", "metric", "n_records", "n_subjects", "calls"]],
                                  current[["method", "metric", "n_records", "n_subjects", "calls"]])
    difference = np.abs(old[["estimate", "low", "high"]].to_numpy() - current[["estimate", "low", "high"]].to_numpy())
    if not np.all(difference <= 1e-12):
        raise ValueError("Unchanged saved comparator estimates/intervals differ")
    return {"metric_rows": 18, "estimate_and_interval_values": 54,
            "maximum_absolute_difference": float(difference.max()),
            "methods": methods, "tolerance": 1e-12}


def compare_output_files(expected: Path, actual: Path, names: list[str]) -> int:
    for name in names:
        if (expected / name).read_bytes() != (actual / name).read_bytes():
            raise ValueError(f"Deterministic replay bytes differ: {name}")
    return len(names)


def replay_empirical(evidence: Path, temporary: Path) -> dict:
    from validation.predict_n1_label_free import execute as predict
    from validation.score_n1_label_free import execute as score
    from validation.summarize_n1_nominal_stage import execute as describe_nominal
    protocol = evidence / "protocols/LABEL_FREE_N1_PROTOCOL.json"
    predictions, scoring = temporary / "predictions", temporary / "scoring"
    with redirect_stdout(io.StringIO()):
        predict(predictions, protocol)
    pred_names = sorted(p.name for p in (evidence / "empirical_predictions").iterdir() if p.is_file())
    n_prediction = compare_output_files(evidence / "empirical_predictions", predictions, pred_names)
    # Reference access happens only after completed label-blind output checks.
    with redirect_stdout(io.StringIO()):
        score(predictions, scoring, protocol)
    score_names = sorted(p.name for p in (evidence / "empirical_scoring").iterdir() if p.is_file())
    n_scoring = compare_output_files(evidence / "empirical_scoring", scoring, score_names)
    nominal = temporary / "nominal_stage"
    with redirect_stdout(io.StringIO()):
        describe_nominal(scoring, nominal)
    nominal_names = sorted(p.name for p in (evidence / "nominal_stage").iterdir() if p.is_file())
    n_nominal = compare_output_files(evidence / "nominal_stage", nominal, nominal_names)
    return {"label_blind_prediction_replayed": True, "separate_human_reference_scorer_replayed": True,
            "prediction_files_byte_identical": n_prediction, "scoring_files_byte_identical": n_scoring,
            "post_result_nominal_stage_replayed": True, "nominal_stage_files_byte_identical": n_nominal,
            "raw_clinical_voltage_reprocessed": False, "models_retrained": False}


def replay_full_calibration(evidence: Path, temporary: Path, workers: int) -> dict:
    from validation.calibrate_n1_label_free import main as calibrate
    output = temporary / "calibration"
    calibrate(["--protocol", str(evidence / "protocols/SYNTHETIC_CALIBRATION_PROTOCOL.json"),
               "--primary-protocol", str(evidence / "protocols/LABEL_FREE_N1_PROTOCOL.json"),
               "--output-dir", str(output), "--workers", str(workers)])
    for name in ("family_results.csv", "summary.json"):
        if (output / name).read_bytes() != (evidence / "calibration" / name).read_bytes():
            raise ValueError(f"Full calibration replay differs: {name}")
    if (output / "contact_results.csv").read_bytes() != gzip.decompress((evidence / "calibration/contact_results.csv.gz").read_bytes()):
        raise ValueError("Full calibration contact outcomes differ")
    return {"families_rerun": 480, "contacts_rerun": 3840,
            "contact_family_and_summary_bytes_identical": True}


def verify(evidence=DEFAULT_EVIDENCE, *, hashes_only=False, full_calibration=False, workers=4) -> dict:
    evidence = Path(evidence).resolve()
    manifest_path = evidence / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result = {"status": "passed", "artifact_manifest_sha256": sha(manifest_path),
              "evidence_files": check_artifacts(evidence, manifest["files"]),
              "source_files": check_repository_bindings(manifest["source_hashes"]),
              "external_frozen_inputs": check_repository_bindings(manifest["external_inputs"])}
    result["empirical_contract"] = check_embedded_receipts(evidence)
    result["saved_comparators"] = compare_secondary_metrics(evidence)
    result["calibration"] = verify_calibration(evidence)
    if not hashes_only or full_calibration:
        with tempfile.TemporaryDirectory(prefix="erpy-n1-label-free-verification-") as temp:
            if not hashes_only:
                result["empirical_replay"] = replay_empirical(evidence, Path(temp))
            if full_calibration:
                result["full_calibration_replay"] = replay_full_calibration(evidence, Path(temp), workers)
    result["scope"] = "Scientific evidence verification only; not a publication, upload, human author approval or manuscript visual-QA receipt."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--hashes-only", action="store_true", help="Skip empirical prediction/scoring replay; retain hashes, sources and calibration input/aggregate checks.")
    parser.add_argument("--full-calibration", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Optional receipt outside the immutable evidence folder")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output and (args.output.resolve().is_relative_to(args.evidence.resolve()) or args.output.exists()):
        parser.error("Use a new receipt path outside the immutable evidence folder")
    report = json.dumps(json_ready(verify(args.evidence, hashes_only=args.hashes_only,
                                        full_calibration=args.full_calibration, workers=args.workers)), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)
