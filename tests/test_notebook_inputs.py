"""Input checks for user notebooks."""
from pathlib import Path
import pytest


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
