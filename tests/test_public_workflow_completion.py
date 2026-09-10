"""Independent decision-join and source-integrity checks for the public demo."""
import numpy as np
import pandas as pd
import pytest

from validation.public_workflow_completion import complete_contact_table, independent_bh, _verified_file


def frame():
    return pd.DataFrame(dict(channel=["A", "B", "C"], method=["crp_energy"]*3,
        qc_status=["pass"]*3, p_joint=[.01, .02, .2], q_joint=[.03, .03, .2], testing_family_size=[3]*3,
        primary_significant=[True, True, False], artifact_qc_pass=[False, True, True], primary_qc_pass=[False, True, False],
        bad_response_fraction=[.3, 0., 0.], hard_artifact_fraction=[0.]*3, n_clean_response=[10]*3,
        artifact_boundary_peak_fail=[False]*3))


def test_final_intersection_preserves_bh_family_and_accounts_for_every_source():
    output = complete_contact_table(frame(), ["A", "B", "C", "D", "LTG1", "EKG"], ["A", "B", "C", "LTG1"])
    assert output.channel.tolist() == ["A", "B", "C", "D", "LTG1", "EKG"]
    assert output.bh_tested.sum() == 3
    assert output.bh_rejected.sum() == 2
    assert output.final_positive.sum() == 1
    assert output.loc[output.final_positive, "channel"].tolist() == ["B"]
    assert output.loc[output.channel.eq("A"), "q_joint"].iloc[0] == .03  # Gate rejection does not rerun BH.
    assert not output.loc[output.channel.isin(["D", "LTG1", "EKG"]), "qc_gate_available"].any()


def test_bh_ignores_unavailable_values_and_retains_tied_probabilities():
    q, rejected = independent_bh(np.array([.01, np.nan, .02, .2]))
    np.testing.assert_allclose(q, [.03, np.nan, .03, .2], equal_nan=True)
    assert rejected.tolist() == [True, False, True, False]


@pytest.mark.parametrize("corruption", ["q", "denominator", "gate", "intersection", "duplicate", "missing_retained", "missing_field", "stimulation_test"])
def test_inconsistent_decisions_fail_closed(corruption):
    value = frame()
    source = ["A", "B", "C", "LTG1"]
    retained = ["A", "B", "C"]
    if corruption == "q": value.loc[0, "q_joint"] = .02
    elif corruption == "denominator": value.loc[0, "testing_family_size"] = 2
    elif corruption == "gate": value.loc[0, "artifact_qc_pass"] = True
    elif corruption == "intersection": value.loc[0, "primary_qc_pass"] = True
    elif corruption == "duplicate": value = pd.concat([value, value.iloc[[0]]])
    elif corruption == "missing_retained": value = value.iloc[:2]
    elif corruption == "missing_field": value = value.drop(columns="hard_artifact_fraction")
    elif corruption == "stimulation_test": value.loc[0, "channel"] = "LTG1"; retained = ["LTG1", "B", "C"]
    with pytest.raises(ValueError):
        complete_contact_table(value, source, retained)


def test_cached_source_requires_both_size_and_digest(tmp_path):
    import hashlib
    path = tmp_path / "public_source"
    data = b"verified public input"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert _verified_file(path, len(data), digest)
    with pytest.raises(ValueError): _verified_file(path, len(data)+1, digest)
    with pytest.raises(ValueError): _verified_file(path, len(data), "0"*64)


def test_native_clean_indices_bind_to_source_event_samples():
    from validation.public_workflow_completion import clean_trial_membership
    native = pd.DataFrame([dict(channel="A", method="crp_energy", clean_trial_indices=[0, 2], n_trials_clean=2, n_trials_total=3)])
    events = pd.DataFrame(dict(onset=[1., 2., 3.], sample_start=[2048, 4096, 6144]), index=[7, 8, 9])
    out = clean_trial_membership(native, [0, 1, 2], events)
    assert out.used_for_joint_inference.tolist() == [True, False, True]
    assert out.source_event_row.tolist() == [7, 8, 9]
    assert out.source_sample.tolist() == [2048, 4096, 6144]
    native.at[0, "clean_trial_indices"] = [0, 0]
    with pytest.raises(ValueError): clean_trial_membership(native, [0, 1, 2], events)
