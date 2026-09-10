"""Evidence verification rejects tampering and unsafe manifest identities."""
import hashlib

import pytest

from validation.verify_n1_label_free import check_artifacts, checked_path, compare_output_files


def test_manifest_byte_identity_and_unlisted_payloads(tmp_path):
    p = tmp_path / "data.csv"; p.write_bytes(b"value\n1\n")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}
    assert check_artifacts(tmp_path, files) == 1
    p.write_bytes(b"value\n2\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        check_artifacts(tmp_path, files)
    p.write_bytes(b"value\n1\n"); (tmp_path / "extra.csv").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="unlisted"):
        check_artifacts(tmp_path, files)


def test_manifest_cannot_escape_root_or_use_symlinks(tmp_path):
    base = tmp_path / "evidence"; base.mkdir()
    outside = tmp_path / "outside.csv"; outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="escapes"):
        checked_path(base, "../outside.csv")
    (base / "linked.csv").symlink_to(outside)
    with pytest.raises(ValueError):
        checked_path(base, "linked.csv")


def test_replay_checks_bytes_even_for_structurally_similar_tables(tmp_path):
    a = tmp_path / "expected"; b = tmp_path / "replay"; a.mkdir(); b.mkdir()
    for directory in (a, b):
        (directory / "data.csv").write_bytes(b"p\n0.01\n")
    assert compare_output_files(a, b, ["data.csv"]) == 1
    (b / "data.csv").write_bytes(b"p\n0.02\n")
    with pytest.raises(ValueError, match="replay bytes differ"):
        compare_output_files(a, b, ["data.csv"])
