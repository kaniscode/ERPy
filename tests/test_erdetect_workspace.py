"""Reproduction stages and workspace preparation preserve evidence boundaries."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "validation" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_erdetect_workspace")
runner = load_script("run_erdetect_validation")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    base = tmp_path / "bundle"
    metadata = base / "erdetect_reference/source_metadata"
    metadata.mkdir(parents=True)
    (metadata / "inventory.json").write_text('{"pinned": true}\n')
    electrodes = metadata / "dataset/sub-MAYO01/ses-ieeg01/ieeg/electrodes.tsv"
    electrodes.parent.mkdir(parents=True)
    electrodes.write_text("name\tx\ty\tz\nA\t1\t2\t3\n")
    archive = base / "frozen_results/erdetect/prediction_files.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as stream:
        content = b"subject,channel\nMAYO01,A\n"
        entry = tarfile.TarInfo("predictions/MAYO01/A-B.csv")
        entry.size = len(content)
        stream.addfile(entry, io.BytesIO(content))
    paths = [p for p in base.rglob("*") if p.is_file()]
    manifest = [{"path": str(p.relative_to(base)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                 "bytes": p.stat().st_size} for p in paths]
    (base / "erdetect_reference/artifact_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(prepare, "HERE", base)
    return base, tmp_path / "workspace"


def test_prepare_preserves_exact_frozen_predictions_without_fabricating_cache_receipts(bundle):
    base, destination = bundle
    prepare.initialize(destination, frozen_predictions=True)
    prediction = destination / "predictions/MAYO01/A-B.csv"
    assert prediction.read_bytes() == b"subject,channel\nMAYO01,A\n"
    assert not prediction.with_suffix(".csv.receipt.json").exists()
    source = base / "erdetect_reference/source_metadata/dataset/sub-MAYO01/ses-ieeg01/ieeg/electrodes.tsv"
    assert (destination / "data/ds004774/sub-MAYO01/ses-ieeg01/ieeg/electrodes.tsv").read_bytes() == source.read_bytes()
    mtime = prediction.stat().st_mtime_ns
    prepare.initialize(destination, frozen_predictions=True)
    assert prediction.stat().st_mtime_ns == mtime


def test_prepare_late_prediction_conflict_does_not_copy_any_metadata(bundle):
    _, destination = bundle
    conflict = destination / "predictions/MAYO01/A-B.csv"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"a different detector run\n")
    with pytest.raises(ValueError, match="Existing workspace content differs"):
        prepare.initialize(destination, frozen_predictions=True)
    assert not (destination / "source_metadata").exists()
    assert not (destination / "data").exists()
    assert conflict.read_bytes() == b"a different detector run\n"


@pytest.mark.parametrize("change", ["listed", "unlisted"])
def test_prepare_rejects_changed_or_unmanifested_evidence_before_writing(bundle, change):
    base, destination = bundle
    filename = "inventory.json" if change == "listed" else "unexpected.json"
    (base / "erdetect_reference/source_metadata" / filename).write_text("{}")
    with pytest.raises(ValueError, match="missing or changed|unmanifested or changed"):
        prepare.initialize(destination)
    assert not destination.exists()


def test_all_stages_use_matching_score_input_and_separate_extraction_runtime(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, check: launched.append(command))
    extraction_python = tmp_path / "extraction/bin/python"
    runner.main(["--root", str(tmp_path), "--stage", "all", "--extraction-python", str(extraction_python)])
    scripts = [Path(command[1]).name for command in launched]
    assert scripts.index("score_erdetect_validation.py") < scripts.index("audit_annotation_coverage.py")
    coverage = next(command for command in launched if command[1].endswith("audit_annotation_coverage.py"))
    score = next(command for command in launched if command[1].endswith("score_erdetect_validation.py"))
    score_dir = Path(score[score.index("--output") + 1])
    assert Path(coverage[coverage.index("--records") + 1]) == score_dir / "record_predictions_and_labels.csv"
    for command in launched:
        if Path(command[1]).name in {"erdetect_mayo_data.py", "erdetect_umcu_data.py"}:
            assert command[0] == str(extraction_python)
        else:
            assert command[0] == runner.sys.executable


@pytest.mark.parametrize("stages", [["--stage", "predict"], ["--stage", "all"]])
def test_frozen_scoring_mode_cannot_start_detector_rerun(tmp_path, monkeypatch, stages):
    launched = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: launched.append(args))
    with pytest.raises(SystemExit):
        runner.main(["--root", str(tmp_path), "--frozen-predictions", *stages])
    assert not launched
