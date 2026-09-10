"""Cache provenance must gate reuse without changing scientific predictions."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("erdetect_predictor", ROOT / "validation/predict_erdetect_epochs.py")
predictor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predictor)


@pytest.fixture
def prediction(tmp_path):
    epoch = tmp_path / "epochs.npz"
    times = np.arange(-200, 71) / 200.
    values = np.random.default_rng(5).normal(size=(8, len(times), 4))
    values[:, :, 2] += 4 * np.exp(-((times - .05) / .025) ** 2)
    np.savez(epoch, values=values, times=times, channels=["A", "B", "C", "D"],
             subject="MAYO_TEST", stimpair="A-B", metadata="{}")
    destination = tmp_path / "predictions"
    result = predictor.predict_file(epoch, destination)
    assert result["records"] == 4
    out = destination / "MAYO_TEST/A-B.csv"
    return epoch, destination, out


def no_inference(*args, **kwargs):
    pytest.fail("A cache acceptance/rejection must not recompute predictions")


def test_valid_receipt_reuses_exact_csv_without_inference(prediction, monkeypatch):
    epoch, destination, out = prediction
    before = out.read_bytes()
    receipt = json.loads(predictor.receipt_path(out).read_text())
    assert receipt["output_sha256"] == hashlib.sha256(before).hexdigest()
    configs = receipt["fingerprint"]["configurations"]
    assert configs["early_10_90ms"]["max_exact_reproducibility_trials"] == 12
    assert receipt["fingerprint"]["dependencies"]["versions"]["numpy"] == np.__version__
    assert "ERPy/crp_energy.py" in receipt["fingerprint"]["code_sha256"]
    monkeypatch.setattr(predictor, "run_crp_energy_array", no_inference)
    assert predictor.predict_file(epoch, destination)["cached"] is True
    assert out.read_bytes() == before


@pytest.mark.parametrize("missing", ["receipt", "csv"])
def test_missing_receipt_or_csv_is_rejected(prediction, monkeypatch, missing):
    epoch, destination, out = prediction
    (out if missing == "csv" else predictor.receipt_path(out)).unlink()
    monkeypatch.setattr(predictor, "run_crp_energy_array", no_inference)
    with pytest.raises(ValueError, match="missing prediction CSV or integrity receipt"):
        predictor.predict_file(epoch, destination)


@pytest.mark.parametrize("change", ["epoch", "window", "default_setting", "code", "dependency", "seed"])
def test_changed_inputs_or_runtime_reject_cache(prediction, monkeypatch, change):
    epoch, destination, out = prediction
    before = out.read_bytes()
    if change == "epoch":
        with np.load(epoch, allow_pickle=False) as saved:
            content = {key: saved[key] for key in saved.files}
        content["values"][0, 0, 0] += 1
        np.savez(epoch, **content)
    elif change == "window":
        monkeypatch.setitem(predictor.WINDOWS, "early_10_90ms", ((.011, .09), (.0, .009)))
    elif change == "default_setting":
        original = predictor.configurations
        monkeypatch.setattr(predictor, "configurations", lambda: {
            name: replace(config, eps=1e-10) for name, config in original().items()})
    elif change == "code":
        changed = predictor.code_fingerprint()
        changed["ERPy/inference.py"] = "0" * 64
        monkeypatch.setattr(predictor, "code_fingerprint", lambda: changed)
    elif change == "dependency":
        changed = predictor.dependency_fingerprint()
        changed["versions"]["numpy"] = "different-build"
        monkeypatch.setattr(predictor, "dependency_fingerprint", lambda: changed)
    else:
        monkeypatch.setattr(predictor, "SEED_NAMESPACE", "different-seed-policy")
    monkeypatch.setattr(predictor, "run_crp_energy_array", no_inference)
    with pytest.raises(ValueError, match="epoch, settings, code, dependencies, or seed policy changed"):
        predictor.predict_file(epoch, destination)
    assert out.read_bytes() == before


@pytest.mark.parametrize("corruption", ["number", "missing_row", "missing_column", "truncated"])
def test_corrupt_csv_is_rejected_even_when_epoch_and_configuration_names_remain(prediction, monkeypatch, corruption):
    epoch, destination, out = prediction
    frame = pd.read_csv(out)
    if corruption == "number":
        frame.loc[0, "p_joint"] = .123456
    elif corruption == "missing_row":
        frame = frame.iloc[:-1]
    elif corruption == "missing_column":
        frame = frame.drop(columns="p_joint")
    if corruption == "truncated":
        out.write_bytes(out.read_bytes()[:-20])
    else:
        frame.to_csv(out, index=False)
    monkeypatch.setattr(predictor, "run_crp_energy_array", no_inference)
    with pytest.raises(ValueError, match="CSV integrity hash mismatch"):
        predictor.predict_file(epoch, destination)


def test_incomplete_record_grid_is_rejected_even_with_updated_csv_hash(prediction):
    epoch, destination, out = prediction
    pd.read_csv(out).iloc[:-1].to_csv(out, index=False)
    receipt_path = predictor.receipt_path(out)
    receipt = json.loads(receipt_path.read_text())
    receipt.update(output_sha256=predictor._sha256(out), records=3)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected records"):
        predictor.predict_file(epoch, destination)


@pytest.mark.parametrize("receipt_text", ["{broken", "{}", "[]"])
def test_malformed_receipt_is_rejected(prediction, receipt_text):
    epoch, destination, out = prediction
    predictor.receipt_path(out).write_text(receipt_text)
    with pytest.raises(ValueError, match="Unusable prediction cache"):
        predictor.predict_file(epoch, destination)


def test_interrupted_receipt_publication_leaves_no_reusable_cache(prediction, monkeypatch, tmp_path):
    epoch, _, source = prediction
    receipt = json.loads(predictor.receipt_path(source).read_text())
    out = tmp_path / "interrupted.csv"
    original_replace = Path.replace

    def fail_receipt_replace(path, target):
        if Path(target) == predictor.receipt_path(out):
            raise OSError("simulated interruption")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_receipt_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        predictor.write_predictions(out, pd.read_csv(source).to_dict("records"), receipt["fingerprint"])
    assert out.is_file()
    assert not predictor.receipt_path(out).exists()
    assert not list(tmp_path.glob(".interrupted.csv.*.tmp"))
    with pytest.raises(ValueError, match="missing prediction CSV or integrity receipt"):
        predictor.validate_cache(out, receipt["fingerprint"])


def test_source_fingerprint_hashes_actual_package_content():
    hashes = predictor.code_fingerprint()
    for name in ("ERPy/crp.py", "ERPy/crp_energy.py", "ERPy/inference.py", "validation/predict_erdetect_epochs.py"):
        assert hashes[name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_prediction_export_preserves_realized_windows_and_clean_trials(tmp_path):
    times = np.arange(-200, 71) / 200.0
    values = np.random.default_rng(6061).normal(size=(9, len(times), 3))
    values[0, times == 0.05, 2] = np.nan
    epoch = tmp_path / "support.npz"
    np.savez(epoch, values=values, times=times, channels=["A", "B", "C"],
             subject="MAYO_TEST", stimpair="A-B", metadata="{}")
    output = tmp_path / "predictions"
    predictor.predict_file(epoch, output)
    frame = pd.read_csv(output / "MAYO_TEST/A-B.csv").set_index("configuration")

    for config_name, count, start, stop, base_start in [
        ("early_10_90ms", 17, 0.01, 0.09, -0.18),
        ("broad_15_300ms", 58, 0.015, 0.3, -0.385),
    ]:
        row = frame.loc[config_name]
        assert row.n_response_samples == row.n_baseline_samples == count
        assert row.response_start_s == pytest.approx(start)
        assert row.response_stop_s == pytest.approx(stop)
        assert row.baseline_start_s == pytest.approx(base_start)
        assert row.baseline_stop_s == pytest.approx(-0.1)
        assert row.n_trials_total == 9 and row.n_trials_clean == 8
        assert json.loads(row.clean_trial_indices) == list(range(1, 9))


def test_old_export_schema_cannot_be_reused_as_endpoint_complete(prediction, monkeypatch):
    epoch, destination, out = prediction
    receipt_path = predictor.receipt_path(out)
    receipt = json.loads(receipt_path.read_text())
    receipt["schema_version"] = 1
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(predictor, "run_crp_energy_array", no_inference)
    with pytest.raises(ValueError, match="unsupported receipt schema"):
        predictor.predict_file(epoch, destination)
