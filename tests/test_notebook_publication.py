"""Regression checks for the curated notebook publication boundary."""

import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "release_audit", Path(__file__).parents[1] / "scripts" / "release_audit.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def published(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "notebooks" / "examples" / "00_quickstart.ipynb"
    path.parent.mkdir(parents=True)
    notebook = {
        "metadata": {},
        "cells": [{"cell_type": "code", "source": ["print('synthetic')"],
                   "metadata": {}, "execution_count": 1,
                   "outputs": [{"output_type": "stream", "name": "stdout", "text": ["synthetic\n"]}]}],
    }
    notebook["metadata"]["erpy_execution"] = {
        "status": "completed", "data_source": "deidentified CNS/ACC–PAG cohort recording",
        "source_sha256": audit._source_digest(notebook),
        "python_version": "3.12", "package_versions": {"erpy-neuro": "1.0.0"},
    }
    return path, notebook


def issues(path, notebook):
    path.write_text(json.dumps(notebook))
    return audit.notebook_issues(path)


def test_curated_executed_notebook_is_allowed(published):
    assert issues(*published) == []


def test_unreviewed_outputs_remain_forbidden(published):
    path, notebook = published
    path = path.with_name("unreviewed.ipynb")
    assert any("stored outputs" in issue for issue in issues(path, notebook))


@pytest.mark.parametrize("change, expected", [
    ("source", "source changed"), ("count", "incomplete"),
    ("error", "execution error"), ("timing", "cell metadata"),
    ("data_source", "invalid curated execution provenance"),
    ("javascript", "unsupported stored output"),
    ("png", "invalid stored PNG"),
])
def test_publication_rejects_incomplete_or_unsafe_results(published, change, expected):
    path, notebook = published
    cell = notebook["cells"][0]
    if change == "source":
        cell["source"] = ["print('changed')"]
    elif change == "count":
        cell["execution_count"] = None
    elif change == "error":
        cell["outputs"].append({"output_type": "error", "ename": "ValueError"})
    elif change == "timing":
        cell["metadata"] = {"execution": {"iopub.execute_input": "timestamp"}}
    elif change == "data_source":
        notebook["metadata"]["erpy_execution"]["data_source"] = "unknown"
    elif change == "javascript":
        cell["outputs"].append({"output_type": "display_data", "data": {"application/javascript": "alert(1)"}})
    elif change == "png":
        cell["outputs"].append({"output_type": "display_data", "data": {"image/png": "bm90IHBuZw=="}})
    assert any(expected in issue for issue in issues(path, notebook))


def test_privacy_scan_keeps_readable_outputs_and_omits_binary_payloads(published):
    _, notebook = published
    notebook["cells"][0]["outputs"].append({
        "output_type": "display_data",
        "data": {"image/png": "opaque-binary-payload", "text/html": "<table>review this text</table>"},
    })
    text = audit.notebook_text(notebook)
    assert "opaque-binary-payload" not in text
    assert "review this text" in text
    assert "print('synthetic')" in text


def test_cohort_loader_requires_authorized_input_without_substitution(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "notebooks" / "examples"))
    from _cohort import load_cohort_example
    monkeypatch.delenv("ERPY_COHORT_EXAMPLE_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="Set ERPY_COHORT_EXAMPLE_DIR"):
        load_cohort_example()


def test_cohort_loader_rejects_a_different_recording(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "notebooks" / "examples"))
    from _cohort import load_cohort_example
    (tmp_path / "all_trials.csv.gz").write_bytes(b"not the documented recording")
    monkeypatch.setenv("ERPY_COHORT_EXAMPLE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="checksum does not match"):
        load_cohort_example()
