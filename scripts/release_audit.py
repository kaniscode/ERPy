"""Fail a release candidate on privacy, packaging, or notebook hygiene issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
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
    allowed_metadata = {"kernelspec", "language_info"}
    unexpected = sorted(set(notebook.get("metadata", {})) - allowed_metadata)
    if unexpected:
        issues.append(f"unexpected notebook metadata keys: {unexpected}")
    output_count = 0
    for cell in notebook.get("cells", []):
        output_count += len(cell.get("outputs", []))
        if cell.get("attachments"):
            issues.append("contains embedded attachments")
        if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
            issues.append("contains execution counts")
            break
    if output_count:
        issues.append(f"contains {output_count} stored outputs")
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
            problems.append(f"{relative}: {size:,} bytes exceeds 10 MiB")
        if path.suffix == ".ipynb":
            problems.extend(f"{relative}: {item}" for item in notebook_issues(path))
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{relative}: expected text but is not valid UTF-8")
            continue
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
