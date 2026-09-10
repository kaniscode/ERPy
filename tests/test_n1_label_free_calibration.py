"""Check the frozen synthetic design, physical noise scale and error units."""
from collections import Counter
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from validation.calibrate_n1_label_free import (
    CHANNELS, FamilySpec, SCENARIOS, family_specs, generate_family,
    read_protocols, result_frames, seed_from_material, signal_waveform,
    summarize, validate_result, wilson_interval)


def test_exact_balanced_allocation_and_unique_random_streams():
    specs = family_specs()
    assert len(specs) == len({s.family_id for s in specs}) == 480
    assert Counter(s.scenario for s in specs) == {"pure_null": 240, **dict.fromkeys(SCENARIOS, 60)}
    cells = Counter((s.scenario, s.n_trials, s.noise_sd_uv, s.amplitude_uv) for s in specs)
    assert len(cells) == 54
    assert all(n == (40 if k[0] == "pure_null" else 5) for k, n in cells.items())
    material = [s.noise_seed_material for s in specs]
    material += [s.inference_seed_material(c) for s in specs for c in CHANNELS]
    assert len({seed_from_material(m) for m in material}) == 4320
    expected = "erpy-label-free-N1-calibration-v1|20260909|pure_null|8|25|0|0|C0"
    assert specs[0].inference_seed_material("C0") == expected
    assert seed_from_material(expected) == int.from_bytes(hashlib.sha256(expected.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def test_noise_is_stationary_at_declared_marginal_sd_and_correlations():
    # A large synthetic trial collection checks the distribution, not outcomes.
    spec = FamilySpec("pure_null", 2000, 100, 0, 19)
    x, t, metadata = generate_family(spec)
    assert x.shape == (2000, 676, 8)
    np.testing.assert_array_equal(t, np.arange(-500, 176) / 500)
    assert abs(x.std() / 100 - 1) < .01
    assert abs(x[:, 0, :].std() / 100 - 1) < .025
    assert abs(np.corrcoef(x[:, :-1, 0].ravel(), x[:, 1:, 0].ravel())[0, 1] - .6) < .01
    assert abs(np.corrcoef(x[:, :, 0].ravel(), x[:, :, 1].ravel())[0, 1] - .25) < .015
    assert metadata["rng_bit_generator"] == "PCG64"


@pytest.mark.parametrize("scenario,latency,sign", [
    ("early_negative", .05, -1), ("early_positive", .05, 1), ("late_negative", .18, -1)])
def test_fixed_waveform_peak_polarity_and_physical_amplitude(scenario, latency, sign):
    t = np.arange(-500, 176) / 500
    spec = FamilySpec(scenario, 8, 25, 250, 0)
    wave = signal_waveform(spec, t)
    index = np.argmax(abs(wave))
    assert t[index] == latency
    assert wave[index] == sign * 250


def test_artifact_starts_at_zero_and_is_not_an_interior_gaussian_peak():
    t = np.arange(-500, 176) / 500
    wave = signal_waveform(FamilySpec("residual_artifact", 8, 25, 100, 0), t)
    assert np.all(wave[t < 0] == 0)
    assert wave[t == 0].item() == -100
    assert np.all(np.diff(wave[t >= 0]) > 0)


def test_generator_replay_is_identical_and_family_rng_is_independent():
    spec = FamilySpec("early_negative", 8, 25, 100, 1)
    a, t, ma = generate_family(spec); b, u, mb = generate_family(spec)
    np.testing.assert_array_equal(a, b)
    assert ma == mb
    other, _, mc = generate_family(replace(spec, replicate=2))
    assert ma["noise_seed"] != mc["noise_seed"]
    assert not np.array_equal(a, other)
    assert isinstance(ma["noise_seed"], str)
    assert all(isinstance(s, str) for s in ma["inference_seeds"].values())


def test_protocol_tampering_rejected_before_generation(tmp_path):
    path = tmp_path / "protocol.json"; path.write_text("{}")
    with pytest.raises(ValueError, match="Synthetic protocol"):
        read_protocols(path, path)


def test_wilson_zero_and_all_calls_keep_correct_family_denominator():
    zero = wilson_interval(0, 240)
    assert zero["total"] == 240 and zero["rate"] == 0
    assert zero["ci_low"] == pytest.approx(0, abs=1e-15)
    assert zero["ci_high"] == pytest.approx(.015753919941558822)
    all_calls = wilson_interval(240, 240)
    assert all_calls["ci_high"] == 1
    assert all_calls["ci_low"] == pytest.approx(1 - zero["ci_high"])


def test_summary_separates_pure_null_families_and_phenotype_targets():
    records = []
    for scenario in ("pure_null", "early_negative", "early_positive"):
        spec = FamilySpec(scenario, 8, 25, 0 if scenario == "pure_null" else 250, 0)
        _, _, metadata = generate_family(spec)
        rows = [{"channel": c, "detected": i < 2, "ablation_detected": i < 3}
                for i, c in enumerate(CHANNELS)]
        records.append({"input": metadata, "result": {"family_size": 8, "contacts": rows}})
    contacts, families = result_frames(records)
    assert len(contacts) == 24
    assert contacts.synthetic_negative_n1.sum() == 2
    assert contacts.is_null_contact.sum() == 20
    assert contacts.random_seed.map(type).eq(str).all()
    summary = summarize(families)
    assert summary["pure_null_family_any_call"]["overall"]["primary"]["total"] == 1
    assert summary["pure_null_family_any_call"]["overall"]["primary"]["successes"] == 1
    cells = {c["scenario"]: c for c in summary["phenotype_challenge_cells"]}
    assert cells["early_negative"]["targets_are_negative_n1"]
    assert not cells["early_positive"]["targets_are_negative_n1"]
    assert cells["early_positive"]["primary"]["target_call_rate"] == 1
    assert cells["early_negative"]["ablation"]["null_contact_calls"] == 1


def test_contact_identity_cannot_silently_change_during_export():
    _, _, metadata = generate_family(FamilySpec("pure_null", 8, 25, 0, 0))
    rows = [{"channel": c, "detected": False, "ablation_detected": False} for c in reversed(CHANNELS)]
    with pytest.raises(ValueError, match="channel order"):
        result_frames([{"input": metadata, "result": {"family_size": 8, "contacts": rows}}])


@pytest.mark.parametrize("corruption", ["float_seed", "input_hash", "nonfinite_p", "call_subset"])
def test_detector_contract_rejects_changed_seed_source_or_invalid_output(corruption):
    _, _, metadata = generate_family(FamilySpec("pure_null", 8, 25, 0, 0))
    rows = [{"channel": c, "random_seed": int(metadata["inference_seeds"][c]),
             "provenance": {"family_input_sha256": metadata["trial_array_sha256"]},
             "p_joint": .5, "p_screened": 1., "q_screened": 1.,
             "ablation_p_screened": .5, "ablation_q_screened": 1.,
             "detected": False, "ablation_detected": False} for c in CHANNELS]
    result = {"family_size": 8, "contacts": rows}
    validate_result(result, metadata)
    if corruption == "float_seed":
        rows[0]["random_seed"] = float(rows[0]["random_seed"])
    elif corruption == "input_hash":
        rows[0]["provenance"]["family_input_sha256"] = "changed"
    elif corruption == "nonfinite_p":
        rows[0]["p_screened"] = float("nan")
    else:
        rows[0]["detected"] = True
    with pytest.raises(ValueError):
        validate_result(result, metadata)
