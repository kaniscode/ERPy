#!/usr/bin/env python3
"""Verify the public N1 evidence and its shared frozen inference input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    paths = set()
    for group in ("files", "external_inputs"):
        for item in manifest[group]:
            relative = item["path"]
            if relative in paths:
                raise ValueError(f"Duplicate artifact identity: {relative}")
            paths.add(relative)
            path = root / relative
            # External input is the adjacent, already public frozen evaluation.
            if not path.resolve().is_relative_to(root.parent):
                raise ValueError(f"Artifact escapes the frozen-results folder: {relative}")
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or linked artifact: {relative}")
            with path.open("rb") as stream:
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
                digest = hasher.hexdigest()
            if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
                raise ValueError(f"Artifact identity mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "artifact_manifest.json"}
    expected = {item["path"] for item in manifest["files"]}
    if actual != expected:
        raise ValueError("N1 evidence contains missing or unlisted files")
    return {"status": "verified", "files": len(manifest["files"]),
            "external_inputs": len(manifest["external_inputs"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "frozen_results" / "n1")
    print(json.dumps(verify(parser.parse_args().root), indent=2))
