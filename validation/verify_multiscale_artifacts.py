#!/usr/bin/env python3
"""Verify the fixed-bank development evidence without rerunning inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    expected = set()
    for item in manifest["files"]:
        relative = item["path"]
        path = root / relative
        if relative in expected or not path.resolve().is_relative_to(root):
            raise ValueError(f"Invalid artifact path: {relative}")
        expected.add(relative)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked artifact: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
            raise ValueError(f"Artifact identity mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "artifact_manifest.json"}
    if actual != expected:
        raise ValueError("Missing or unlisted multiscale evidence files")
    receipts = json.loads((root / "PACKAGING_PROVENANCE.json").read_text())
    originals = {}
    for receipt in receipts["files"]:
        digest = hashlib.sha256((root / receipt["path"]).read_bytes()).hexdigest()
        if digest != receipt["public_sha256"]:
            raise ValueError(f"Public export receipt mismatch: {receipt['path']}")
        if not receipt["changes"] and digest != receipt["original_sha256"]:
            raise ValueError(f"An exact-copy artifact changed: {receipt['path']}")
        originals[receipt["path"]] = receipt["original_sha256"]

    def original_matches(relative, digest):
        relative = Path(relative).as_posix()
        if originals.get(relative) != digest:
            raise ValueError(f"Historical input/output identity mismatch: {relative}")

    referenced = 0
    for stage in ("", "real_baseline", "weighted_confirmation"):
        run = json.loads((root / stage / "run_manifest.json").read_text())
        for name, digest in run["outputs"].items():
            if name == ".DS_Store" or name.endswith(".log"):
                continue  # Explicitly documented non-scientific omissions.
            original_matches(Path(stage) / name, digest)
            referenced += 1
        plan = "analysis_plan.json" if stage != "weighted_confirmation" else stage + "/analysis_plan.json"
        original_matches(plan, run["analysis_plan_sha256"])
        source = "weighted_execution_code" if stage == "weighted_confirmation" else "execution_code"
        for name, digest in run["code_hashes"].items():
            relative = name.removeprefix("<repository>/")
            original_matches(Path(source) / relative, digest)
            referenced += 1
    for stage in ("weighted_development", "weighted_real_baseline"):
        run = json.loads((root / stage / "derivation_manifest.json").read_text())
        original_matches("weighted_amendment/analysis_plan.json", run["amendment_sha256"])
        for name, digest in run["inputs"].items():
            original_matches(name, digest)
            referenced += 1
        for name, digest in run["outputs"].items():
            original_matches(Path(stage) / name, digest)
            referenced += 1
    return {"status": "verified", "files": len(expected),
            "export_receipts": len(receipts["files"]),
            "historical_input_output_references": referenced}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=
                        Path(__file__).resolve().parent / "frozen_results" / "multiscale")
    print(json.dumps(verify(parser.parse_args().root), indent=2))
