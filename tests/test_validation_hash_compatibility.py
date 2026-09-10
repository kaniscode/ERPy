"""Validation provenance hashing must work on every supported Python version."""
import hashlib
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("module_name,helper_name", [
    ("predict_erdetect_epochs", "_sha256"),
    ("extract_n1_features", "sha"),
    ("optimize_n1", "sha"),
])
@pytest.mark.parametrize("payload", [b"", bytes(range(256)) * 8193],
                         ids=["empty", "binary_multiple_blocks"])
def test_sha256_without_python311_file_digest(tmp_path, monkeypatch, module_name,
                                             helper_name, payload):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "validation"))
    helper = getattr(importlib.import_module(module_name), helper_name)
    path = tmp_path / "input.bin"
    path.write_bytes(payload)
    assert helper(path) == hashlib.sha256(payload).hexdigest()
