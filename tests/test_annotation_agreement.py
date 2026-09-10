"""Checks for descriptive agreement estimands and participant-level resampling."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'validation/analyze_annotation_agreement.py'
spec = importlib.util.spec_from_file_location('annotation_agreement', SOURCE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_fractional_rater_votes_are_not_majority_votes():
    # Four equally weighted records: y=(1, 1/2, 0, 1/2), calls=(1,1,0,0).
    # The mixed records contribute half to each reference class.
    metrics = analysis.measures([1.5, .5, 1.5, .5, 4, 4])
    assert metrics['agreement'] == .75
    assert metrics['kappa'] == .5
    assert metrics['n1_positive_call_rate'] == .75
    assert metrics['n1_negative_call_rate'] == .25
    assert metrics['n1_call_rate_difference'] == .5


def test_whole_subject_resampling_keeps_site_and_count_vectors_together():
    index = pd.MultiIndex.from_tuples([('Mayo', 'A'), ('UMCU', 'B')], names=['site','subject'])
    vectors = pd.DataFrame([[2, 7], [5, 11]], index=index, columns=['calls','records'])
    # Exactly one subject per site means every stratified draw is identical.
    draws = analysis.boot_counts(vectors)
    np.testing.assert_array_equal(draws, np.tile([7,18], (analysis.RESAMPLES,1)))


def test_paired_cluster_draws_are_deterministic_and_shared():
    index = pd.MultiIndex.from_tuples([('Mayo','A'),('Mayo','B'),('UMCU','C')], names=['site','subject'])
    vectors = pd.DataFrame([[1,2],[3,7],[8,10]], index=index)
    a = analysis.boot_counts(vectors)
    b = analysis.boot_counts(vectors * 2)
    np.testing.assert_array_equal(b, 2*a)
    np.testing.assert_array_equal(analysis.boot_counts(vectors), a)


def test_undefined_intervals_remain_unavailable():
    result = analysis.interval([np.nan, np.inf, -np.inf])
    assert result == {'low':None, 'high':None, 'valid_resamples':0}
    result = analysis.interval([np.nan, 1., 3.])
    assert result['valid_resamples'] == 2
    assert result['low'] == pytest.approx(1.05)
    assert result['high'] == pytest.approx(2.95)


def test_verifier_rejects_changed_analysis_source_before_replay(tmp_path):
    folder = tmp_path/'validation/annotation_agreement'
    folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps({'script_sha256':'0'*64}))
    with pytest.raises(AssertionError, match='Analysis source changed'):
        analysis.verify(tmp_path)
