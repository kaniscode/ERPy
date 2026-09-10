"""Scientific contracts for external-label joins and clustered scoring."""
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from validation.score_erdetect_validation import (
    KEY, COUNTS, aggregate_reference, apply_pair_manifest, binary_annotation,
    canonical_pair, cohort_frames, confusion_contributions, join_predictions, labeled_matrix, load_annotations,
    load_archived, metrics_from_counts, minimum_stimulation_distance,
    require_unique, score_reference, subject_bootstrap,
)


def test_mat_labels_are_joined_by_canonical_labels_not_positions(tmp_path):
    first = {"channels": ["c2", "c1"], "stimpairs": [" a2-a1 ", "B1-B2"],
             "annotations": np.array([[1, -1], [2, np.nan]])}
    second = {"channels": ["C1", "C2"], "stimpairs": ["b2-b1", "A1-A2"],
              "annotations": np.array([[0, 0], [-1, 1]])}
    for rater, data in [("raterA", first), ("raterB", second)]:
        savemat(tmp_path / f"sub-MAYO01_test_annot_{rater}.mat", {"annot": data})
    labels, _ = load_annotations(tmp_path)
    pooled = aggregate_reference(labels).set_index(["stimpair", "channel"])
    assert pooled.loc[("A1-A2", "C2"), "reference_positive"] == 1
    assert pooled.loc[("A1-A2", "C1"), "reference_positive"] == 0
    assert pooled.loc[("A1-A2", "C1"), "strict_consensus"]
    assert pooled.loc[("B1-B2", "C1"), "n_raters"] == 1
    assert not pooled.loc[("B1-B2", "C1"), "strict_consensus"]
    assert ("B1-B2", "C2") not in pooled.index
    assert labels.reference_positive.isna().sum() == 3


def test_extra_umcu21_rater_retained_with_historical_subset_flag(tmp_path):
    savemat(tmp_path / "sub-UMCU21_test_annot_SB.mat", {"annot": {
        "channels": ["C1"], "stimpairs": ["A1-A2"], "annotations": [[1]]}})
    labels, files = load_annotations(tmp_path)
    assert len(files) == 1 and labels.reference_positive.iloc[0] == 1
    assert not labels.in_published_25_file_list.iloc[0]


def test_excluded_and_unknown_codes_never_become_negative_labels():
    np.testing.assert_equal(binary_annotation([1, 0, 2, -1, np.nan]), [1, 0, 0, np.nan, np.nan])
    with pytest.raises(ValueError, match="Unknown"):
        binary_annotation([3])


def test_transposed_matrix_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="channel-by-pair"):
        labeled_matrix(np.zeros((3, 2)), 2, 3)
    assert labeled_matrix([1, 2, 3], 1, 3).shape == (1, 3)


def test_archived_nan_is_no_call_but_absent_record_is_not_created(tmp_path):
    path = tmp_path / "sub-MAYO01_test"
    path.mkdir()
    savemat(path / "erdetect_data.mat", {"channel_labels": ["C1", "C2"],
        "stimpair_labels": ["A2-A1"], "neg_peak_amplitudes": [[-20], [np.nan]], "config": {"threshold": 3.4}})
    rows, _, _ = load_archived(tmp_path)
    assert rows.archived_record_present.all()
    assert rows.archived_detected.tolist() == [True, False]
    assert len(rows) == 2


def test_duplicate_reversed_pairs_fail_before_multiplying_join_rows():
    assert canonical_pair("a2 - A1") == "A1-A2"
    frame = pd.DataFrame({"subject": ["MAYO01"] * 2, "stimpair": ["A1-A2"] * 2, "channel": ["C1"] * 2})
    with pytest.raises(ValueError, match="Duplicate"):
        require_unique(frame, KEY, "test")


def test_fractional_reference_weights_each_record_once():
    labels = pd.DataFrame({"subject": ["MAYO01"] * 4, "stimpair": ["A1-A2"] * 4,
                          "channel": ["C1", "C2", "C2", "C2"], "rater": ["A", "A", "B", "C"],
                          "reference_positive": [1, 1, 0, 0]})
    pooled = aggregate_reference(labels)
    pooled["site"] = "Mayo"
    contributions = confusion_contributions(pooled, [True, False], [True, True], True)
    totals = contributions[list(COUNTS)].sum().to_numpy()
    np.testing.assert_allclose(totals, [1, 1/3, 2/3, 0, 2, 2])
    assert metrics_from_counts(totals)["accuracy"] == pytest.approx(5/6)
    assert not pooled.strict_consensus.any()  # one rater and a three-rater disagreement


def test_missingness_policies_change_confusion_without_hiding_coverage():
    frame = pd.DataFrame({"subject": ["MAYO01"] * 3, "site": ["Mayo"] * 3, "reference_positive": [1., 1., 0.]})
    available = confusion_contributions(frame, [True, False, False], [True, False, True], True)
    policy = confusion_contributions(frame, [True, False, False], [True, False, True], False)
    a, b = [metrics_from_counts(x[list(COUNTS)].sum().to_numpy()) for x in [available, policy]]
    assert a["sensitivity"] == 1 and b["sensitivity"] == .5
    assert a["coverage"] == b["coverage"] == pytest.approx(2/3)


def test_distance_is_to_either_contact_and_12mm_boundary_is_retained():
    coordinates = {"A1": np.array([0., 0, 0]), "A2": np.array([100., 0, 0]),
                   "C1": np.array([99., 0, 0]), "C2": np.array([12., 0, 0])}
    assert minimum_stimulation_distance("C1", "A1-A2", coordinates) == 1
    assert minimum_stimulation_distance("C2", "A1-A2", coordinates) == 12
    assert np.isnan(minimum_stimulation_distance("MISSING", "A1-A2", coordinates))
    assert np.isnan(minimum_stimulation_distance("C1", "A1-A2", None))


def _joined_fixture():
    universe = pd.DataFrame({"subject": ["MAYO01", "MAYO01"], "stimpair": ["A1-A2"] * 2,
        "channel": ["C1", "C2"], "site": ["Mayo"] * 2, "source_eligible": [True, True]})
    prediction = pd.DataFrame([dict(subject="MAYO01", stimpair="A1-A2", channel="C1", configuration=c,
                                   prediction_present=True, evaluable=True, detected=False, q_joint=.08)
                               for c in ["early_10_90ms", "broad_15_300ms"]])
    archive = universe[KEY].copy()
    archive["archived_record_present"], archive["archived_detected"] = True, False
    return join_predictions(universe, prediction, archive)


def test_label_join_retains_q_and_incomplete_processing_is_not_a_final_no_call():
    joined = _joined_fixture()
    assert joined.loc[joined.channel.eq("C1"), "q_joint"].eq(.08).all()
    assert joined.loc[joined.channel.eq("C2"), "q_joint"].isna().all()
    with pytest.raises(ValueError, match="Incomplete predictions"):
        apply_pair_manifest(joined)
    preview = apply_pair_manifest(joined, allow_incomplete=True)
    assert preview.loc[preview.channel.eq("C2"), "prediction_status"].eq("incomplete_processing_no_prediction").all()


def test_documented_raw_loss_is_separate_from_historical_trial_mismatch():
    joined = _joined_fixture()
    manifest = pd.DataFrame([dict(subject="MAYO01", stimpair="A2-A1", actual_missing_raw=True,
                                  archived_pair_comparable=False, reason="truncated source")])
    result = apply_pair_manifest(joined, manifest)
    assert not result.archived_pair_comparable.any()
    assert result.loc[result.channel.eq("C2"), "prediction_status"].eq("missing_raw_no_prediction").all()
    assert result.loc[result.channel.eq("C1"), "prediction_status"].eq("evaluable").all()


def test_whole_subject_bootstrap_is_not_an_independent_record_bootstrap():
    frame = pd.DataFrame({"subject": ["MAYO01"] * 100 + ["MAYO02"] * 100,
                          "site": ["Mayo"] * 200, "reference_positive": [1.] * 200})
    contributions = confusion_contributions(frame, [True] * 100 + [False] * 100, [True] * 200, True)
    ci = subject_bootstrap(contributions, 2000, 20260907)
    assert ci["sensitivity"]["low"] == 0 and ci["sensitivity"]["high"] == 1
    assert ci == subject_bootstrap(contributions, 2000, 20260907)
    assert ci["specificity"]["valid_resamples"] == 0


def test_site_stratification_and_single_subject_limits():
    frame = pd.DataFrame({"subject": ["MAYO01", "UMCU20"], "site": ["Mayo", "UMCU"], "reference_positive": [1., 1.]})
    c = confusion_contributions(frame, [True, False], [True, True], True)
    ci = subject_bootstrap(c)
    assert ci["sensitivity"]["low"] == ci["sensitivity"]["high"] == .5
    one = subject_bootstrap(c.iloc[:1])
    assert np.isnan(one["sensitivity"]["low"])
    assert one["sensitivity"]["valid_resamples"] == 0


def test_incomparable_trial_pair_excluded_only_from_paired_metrics():
    frame = pd.DataFrame({"subject": ["MAYO01"] * 2, "site": ["Mayo"] * 2,
        "configuration": ["early_10_90ms"] * 2, "reference_positive": [1., 0.],
        "evaluable": [True, True], "detected": [True, False], "archived_record_present": [True, True],
        "archived_pair_comparable": [True, False], "archived_detected": [True, True]})
    result = pd.DataFrame(score_reference(frame, {}, 20, 1))
    overall = result.loc[result.level.eq("overall") & result.missingness_policy.eq("available_case")]
    assert overall.loc[overall.comparison.eq("erpy_all_source_eligible"), "n_scored_records"].iloc[0] == 2
    paired = overall.loc[overall.comparison.eq("paired_archived_intersection")]
    assert paired.n_scored_records.eq(1).all()
    assert set(paired.method) == {"ERPy", "archived_ER_detect"}


def test_paired_bootstrap_keeps_subjects_without_evaluable_records_together():
    frame = pd.DataFrame({"subject": ["MAYO01", "MAYO02"], "site": ["Mayo"] * 2,
        "configuration": ["early_10_90ms"] * 2, "reference_positive": [1., 1.],
        "evaluable": [True, False], "detected": [True, False], "archived_record_present": [True, True],
        "archived_pair_comparable": [True, True], "archived_detected": [True, True]})
    result = pd.DataFrame(score_reference(frame, {}, 400, 7))
    paired = result.loc[result.level.eq("overall") & result.missingness_policy.eq("available_case")
                        & result.comparison.eq("paired_archived_intersection")]
    assert paired.n_subjects.eq(2).all()
    assert paired.n_subjects_with_scored_records.eq(1).all()
    assert paired.sensitivity_valid_resamples.nunique() == 1
    assert 0 < paired.sensitivity_valid_resamples.iloc[0] < 400


def test_development_overlap_is_a_separate_provenance_cohort():
    frame = pd.DataFrame({"subject": ["MAYO01", "MAYO02", "UMCU20"], "reference_positive": [0, 1, 0]})
    scopes = {scope: set(group.subject) for scope, group in cohort_frames(frame)}
    assert scopes["independent_external_excluding_development_overlap"] == {"MAYO02", "UMCU20"}
    assert scopes["all_recovered_replication"] == {"MAYO01", "MAYO02", "UMCU20"}
    assert scopes["development_overlap_only"] == {"MAYO01"}
    frame.reference_positive = 1 - frame.reference_positive
    assert scopes == {scope: set(group.subject) for scope, group in cohort_frames(frame)}
