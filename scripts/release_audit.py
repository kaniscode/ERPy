"""Fail a release candidate on privacy, packaging, or notebook hygiene issues."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
# This one licensed public, derived rater-level CSV supports exact validation
# reproduction. A changed payload or another large file still fails the audit.
LARGE_PUBLIC_ARTIFACTS = {
    "validation/frozen_results/erdetect/rater_label_joins.csv.gz": (
        13_262_477,
        "916c76fc6935913a2eb6a589797a2745092488d1ba5893da7f6bc17ef19dc8fc",
    ),
}
ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "ERPy",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "SECURITY.md",
    "config.example.yaml",
    "docs",
    "notebooks",
    "pyproject.toml",
    "requirements.txt",
    "scripts",
    "setup.py",
    "tests",
    "validation",
}
FORBIDDEN_PATH_PARTS = {
    ".DS_Store",
    ".clau" + "de",
    ".virtual_documents",
    "build",
    "dist",
    "ERPy.egg-info",
    "writing_phase",
    "release_artifacts",
    "rcs_manifest.py",
}
FORBIDDEN_TEXT = {
    "local user path": re.compile(
        r"(?:/(?:Users|home|userdata)/[A-Za-z0-9._-]+/|"
        r"[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\\\r\n]+\\)"
    ),
    "private source identifier": re.compile(r"\bRCS0[4-7]\b", re.IGNORECASE),
    "credential marker": re.compile(
        r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})"
    ),
    "generation provenance marker": re.compile(
        "(?:open" + "ai|chat" + "gpt|gpt[- ]?[0-9]|gem" + "ini|anth" +
        "ropic|clau" + "de|copi" + "lot|co" + "dex|ai[- ]gener" + "ated)",
        re.IGNORECASE,
    ),
    "stale public version": re.compile("\\b0" + r"\.5\.0\b"),
}
TEXT_SUFFIXES = {
    ".cff", ".csv", ".ini", ".ipynb", ".json", ".md", ".py", ".toml",
    ".tsv", ".txt", ".yaml", ".yml",
}
CURATED_NOTEBOOKS = {
    "00_quickstart.ipynb", "01_waveform_visualizations.ipynb",
    "02_detection_qc_and_crp.ipynb", "03_spectral_visualizations.ipynb",
    "04_network_visualizations.ipynb", "05_brain_and_interactive_visualizations.ipynb",
    "06_exporting_figures.ipynb", "07_public_ds003708_recipe.ipynb",
    "08_n1_development.ipynb",
}


def _source_digest(notebook: dict) -> str:
    sources = [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest()


def notebook_text(notebook: dict) -> str:
    """Scan all readable content, excluding opaque base64 image payloads."""
    cleaned = json.loads(json.dumps(notebook))
    for cell in cleaned.get("cells", []):
        for output in cell.get("outputs", []):
            output.get("data", {}).pop("image/png", None)
    return json.dumps(cleaned, ensure_ascii=False)


def repository_files(*, staged: bool) -> list[Path]:
    if staged:
        command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    else:
        command = ["git", "ls-files", "-co", "--exclude-standard"]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    return [ROOT / line for line in output.splitlines() if line.strip()]


def notebook_issues(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid notebook: {exc}"]
    curated = path.parent == ROOT / "notebooks" / "examples" and path.name in CURATED_NOTEBOOKS
    provenance = notebook.get("metadata", {}).get("erpy_execution")
    executed = curated and isinstance(provenance, dict)
    allowed_metadata = {"kernelspec", "language_info"}
    if curated:
        allowed_metadata.add("erpy_execution")
    unexpected = sorted(set(notebook.get("metadata", {})) - allowed_metadata)
    if unexpected:
        issues.append(f"unexpected notebook metadata keys: {unexpected}")
    output_count = 0
    code_count = 0
    for cell in notebook.get("cells", []):
        output_count += len(cell.get("outputs", []))
        if cell.get("attachments"):
            issues.append("contains embedded attachments")
        if cell.get("metadata"):
            issues.append("contains cell metadata (including timing or machine state)")
        if cell.get("cell_type") == "code":
            code_count += 1
            if executed:
                if cell.get("execution_count") != code_count:
                    issues.append("execution counts are incomplete or out of order")
            elif cell.get("execution_count") is not None:
                issues.append("contains execution counts without curated execution provenance")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                issues.append("contains a cell execution error")
            if output.get("metadata"):
                issues.append("contains output metadata")
            for mime, payload in output.get("data", {}).items():
                if mime not in {"text/plain", "text/html", "image/png"}:
                    issues.append(f"unsupported stored output type: {mime}")
                if mime == "image/png":
                    try:
                        decoded = base64.b64decode("".join(payload), validate=True)
                        if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                            raise ValueError("invalid PNG signature")
                    except (ValueError, TypeError):
                        issues.append("invalid stored PNG output")
    if output_count and not executed:
        issues.append(f"contains {output_count} stored outputs")
    if executed:
        expected_source = (
            "deidentified CNS/ACC–PAG cohort recording" if path.name.startswith("00_")
            else "public OpenNeuro ds003708" if path.name.startswith("07_")
            else "public OpenNeuro ds004774 derived N1 features" if path.name.startswith("08_")
            else "deterministic synthetic"
        )
        if provenance.get("status") != "completed" or provenance.get("data_source") != expected_source:
            issues.append("invalid curated execution provenance")
        if provenance.get("source_sha256") != _source_digest(notebook):
            issues.append("notebook source changed after execution")
        if not provenance.get("package_versions") or not provenance.get("python_version"):
            issues.append("missing execution environment versions")
        if not output_count:
            issues.append("executed example has no stored results")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="scan only staged additions/modifications")
    args = parser.parse_args()
    problems: list[str] = []
    files = repository_files(staged=args.staged)
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] not in ALLOWED_TOP_LEVEL:
            problems.append(f"{relative}: unexpected top-level release path")
            continue
        parts = set(relative.parts)
        forbidden_parts = sorted(parts.intersection(FORBIDDEN_PATH_PARTS))
        if forbidden_parts:
            problems.append(f"{relative}: forbidden path component {forbidden_parts}")
            continue
        if not path.is_file():
            continue
        if path.is_symlink():
            problems.append(f"{relative}: symbolic links are not allowed in the public snapshot")
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            expected = LARGE_PUBLIC_ARTIFACTS.get(relative.as_posix())
            actual = (size, hashlib.sha256(path.read_bytes()).hexdigest()) if expected else None
            if expected is None or actual != expected:
                problems.append(f"{relative}: {size:,} bytes exceeds 10 MiB without an exact public-artifact exception")
        if path.suffix == ".ipynb":
            problems.extend(f"{relative}: {item}" for item in notebook_issues(path))
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{relative}: expected text but is not valid UTF-8")
            continue
        if path.suffix == ".ipynb":
            try:
                text = notebook_text(json.loads(text))
            except json.JSONDecodeError:
                continue  # Reported by notebook_issues above.
        for label, pattern in FORBIDDEN_TEXT.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{relative}:{line}: {label}")
    required = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "docs" / "GETTING_STARTED.md",
        ROOT / "docs" / "METHODS.md",
        ROOT / "docs" / "API_REFERENCE.md",
        ROOT / "docs" / "VISUALIZATION_GALLERY.md",
    ]
    for path in required:
        if not path.is_file():
            problems.append(f"missing required release file: {path.relative_to(ROOT)}")
    if problems:
        print("Release audit failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"Release audit passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
