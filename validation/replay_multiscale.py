#!/usr/bin/env python3
"""Replay both frozen bank stages in an isolated local source/output folder.

No scientific implementation is modified. Exact historical source snapshots
are overlaid only in the new replay workspace; the installed package and
frozen evidence remain unchanged. The small public baseline fixtures are the
only recorded samples needed; no network or protected recording is accessed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from verify_multiscale_artifacts import verify


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "validation" / "frozen_results" / "multiscale"


def replay(output: Path, workers: int) -> None:
    verify(EVIDENCE)
    output = output.resolve()
    if output.exists() or output.is_relative_to(EVIDENCE):
        raise ValueError("Use a new replay directory outside the frozen evidence")
    if workers < 1:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True)
    code = output / "source_code"
    results = output / "results"
    results.mkdir()
    shutil.copytree(ROOT / "ERPy", code / "ERPy",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    (code / "validation").mkdir(parents=True)

    def overlay(folder):
        for source in folder.rglob("*.py"):
            relative = source.relative_to(folder)
            if relative.parts[0] == "tests":
                continue
            target = code / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    # Retain the same shared inference/noise-helper implementation in both stages.
    overlay(EVIDENCE / "weighted_execution_code")
    overlay(EVIDENCE / "execution_code")
    for name in ("analysis_plan.json", "analysis_plan.sha256", "analysis_plan.md",
                 "analysis_plan_initial.json"):
        shutil.copyfile(EVIDENCE / name, results / name)
    shutil.copytree(EVIDENCE / "weighted_amendment", results / "weighted_amendment")
    shutil.copytree(EVIDENCE / "fixtures", results / "fixtures")
    fixture_path = results / "fixtures" / "fixture_manifest.json"
    fixtures = json.loads(fixture_path.read_text())
    for item in fixtures["fixtures"]:
        item["fixture_file"] = str(results / item["fixture_file"])
    fixture_path.write_text(json.dumps(fixtures, indent=2) + "\n")
    environment = dict(os.environ, PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def execute(arguments):
        subprocess.run([sys.executable, *arguments], cwd=code, env=environment, check=True)

    common = ["-m", "validation.benchmark_multiscale", "--workers", str(workers)]
    execute(common + ["--output-dir", str(results)])
    execute(common + ["--output-dir", str(results / "real_baseline"), "--real-baseline"])
    overlay(EVIDENCE / "weighted_execution_code")
    helper = results / "derive_weighted_development.py"
    shutil.copyfile(EVIDENCE / helper.name, helper)
    execute([str(helper)])
    confirmation = results / "weighted_confirmation"
    confirmation.mkdir()
    for name in ("analysis_plan.json", "analysis_plan.sha256"):
        shutil.copyfile(EVIDENCE / "weighted_confirmation" / name, confirmation / name)
    execute(common + ["--output-dir", str(confirmation), "--weighted-confirmation"])
    audit = results / "audit_and_report.py"
    shutil.copyfile(EVIDENCE / audit.name, audit)
    execute([str(audit)])
    # The retained report script includes the original test-run count. This
    # replay executes its numerical checks, not the historical pytest run.
    receipt_path = results / "verification.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["historical_focused_tests_passed"] = receipt.pop("focused_tests_passed")
    receipt["replay_scope"] = "All numerical experiments and arithmetic checks rerun; historical pytest count not rerun here."
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    report_path = results / "RESULTS_AND_DECISION.md"
    report_path.write_text(
        "> Reproduced numerical report. The focused-test count below belongs to the "
        "original recorded test run; this replay reruns numerical experiments and "
        "arithmetic checks only.\n\n" + report_path.read_text()
    )
    print("Replay complete; results and local execution receipts are in", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    replay(args.output, args.workers)
