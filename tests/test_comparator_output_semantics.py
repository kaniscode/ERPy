"""Comparator labels/availability must not masquerade as primary components."""
import io

import numpy as np
import pandas as pd
import pytest

from ERPy.epochs import Epochs
from ERPy.waveform_qc import audit_waveforms


def _fixture():
    times = np.arange(-250, 151) / 500.0
    values = np.sin(40 * np.pi * times[None, :] + np.arange(8)[:, None])
    values += 30 * np.exp(-((times - 0.05) ** 2) / (2 * 0.012 ** 2))
    frame = pd.DataFrame(
        values.reshape(-1, 1),
        index=pd.MultiIndex.from_product([range(8), times], names=['epoch', 'time']),
        columns=['response'],
    )
    return Epochs(frame, sfreq=500.0, tmin=-0.5, tmax=0.3, baseline=(-0.5, -0.03))


def _audit(epochs, detections):
    return audit_waveforms(
        epochs, detections, pd.DataFrame(), response_window=(0.01, 0.3),
        min_clean_responses=8,
    ).summary.iloc[0]


def test_default_components_are_positive_while_unrun_comparators_are_missing():
    epochs = _fixture()
    detections = epochs.detect_erp_all(methods=['crp_energy'])
    row = detections.iloc[0]
    assert row['primary_joint_p_value'] == pytest.approx(0.0078125)
    assert row['primary_joint_q_value'] == pytest.approx(0.0078125)
    assert row['primary_significant']
    assert row['primary_reproducibility_pass'] and row['primary_energy_pass']
    audit = _audit(epochs, detections)
    assert audit['primary_detector_pass'] and audit['primary_qc_pass']
    for record in [row, audit]:
        for name, alias in [('crp', 'primary_shape_pass'), ('kundu', 'primary_magnitude_pass')]:
            assert pd.isna(record[f'comparator_{name}_pass'])
            assert pd.isna(record[alias])
            assert not record[f'comparator_{name}_available']
            assert record[f'comparator_{name}_availability_reason'] == 'not_run'
    assert 'comparator_crp_kundu_disagreement' not in audit['review_reasons']
    assert 'primary_shape_magnitude_disagreement' not in audit['review_reasons']


@pytest.mark.parametrize('method,available,missing', [
    ('crp_significance', 'crp', 'kundu'),
    ('kundu_rolston', 'kundu', 'crp'),
])
def test_one_comparator_run_is_not_reported_as_a_disagreement(method, available, missing):
    epochs = _fixture()
    detections = epochs.detect_erp_all(methods=[method])
    for record in [detections.iloc[0], _audit(epochs, detections)]:
        assert record[f'comparator_{available}_available']
        assert pd.notna(record[f'comparator_{available}_pass'])
        assert not record[f'comparator_{missing}_available']
        assert pd.isna(record[f'comparator_{missing}_pass'])
    assert 'comparator_crp_kundu_disagreement' not in _audit(epochs, detections)['review_reasons']


def test_actual_comparator_disagreement_is_explicit_and_does_not_change_primary():
    epochs = _fixture()
    detections = epochs.detect_erp_all(methods=['crp_energy', 'crp_significance', 'kundu_rolston'])
    assert detections['primary_significant'].all()
    assert detections['primary_joint_p_value'].eq(0.0078125).all()
    assert detections['comparator_crp_pass'].all()
    assert not detections['comparator_kundu_pass'].any()
    assert not detections['shape_magnitude_significant'].any()
    result = _audit(epochs, detections)
    assert result['primary_qc_pass']
    assert 'comparator_crp_kundu_disagreement' in result['review_reasons'].split(';')
    assert 'primary_shape_magnitude_disagreement' not in result['review_reasons']
    assert result['primary_shape_pass'] == result['comparator_crp_pass']
    assert result['primary_magnitude_pass'] == result['comparator_kundu_pass']


@pytest.mark.parametrize('failure', ['kundu_filter', 'kundu_window', 'crp_failure'])
def test_failed_comparator_evidence_is_unavailable_not_negative(monkeypatch, failure):
    epochs = _fixture()
    options = {}
    missing = 'kundu'
    if failure == 'kundu_filter':
        def fail_filter(*args, **kwargs):
            raise ValueError('controlled filter failure')
        monkeypatch.setattr('ERPy.erp_detection._kundu_envelope', fail_filter)
    elif failure == 'kundu_window':
        options['kundu_rolston'] = {'response_window': (0.9, 1.0)}
    else:
        missing = 'crp'
        def fail_crp(*args, **kwargs):
            raise ValueError('controlled CRP failure')
        monkeypatch.setattr('ERPy.erp_detection.run_crp_array', fail_crp)
    detections = epochs.detect_erp_all(methods=['crp_significance', 'kundu_rolston'], **options)
    for record in [detections.iloc[0], _audit(epochs, detections)]:
        assert not record[f'comparator_{missing}_available']
        assert pd.isna(record[f'comparator_{missing}_pass'])
        assert record[f'comparator_{missing}_availability_reason']
    assert 'comparator_crp_kundu_disagreement' not in _audit(epochs, detections)['review_reasons']


@pytest.mark.parametrize('metadata', ['false', 'missing'])
def test_audit_does_not_infer_availability_from_a_positive_call(metadata):
    epochs = _fixture()
    detections = epochs.detect_erp_all(methods=['crp_significance', 'kundu_rolston'])
    if metadata == 'false':
        mask = detections['method'].eq('crp_significance')
        detections.loc[mask, 'method_available'] = False
        detections.loc[mask, 'availability_reason'] = 'source unavailable'
    else:
        detections = detections.drop(columns='method_available')
    result = _audit(epochs, detections)
    assert pd.isna(result['comparator_crp_pass'])
    assert not result['comparator_crp_available']
    assert 'comparator_crp_kundu_disagreement' not in result['review_reasons']


def test_csv_roundtrip_preserves_unavailable_comparator_call():
    epochs = _fixture()
    detections = epochs.detect_erp_all(methods=['crp_energy'])
    restored = pd.read_csv(io.StringIO(detections.to_csv(index=False)))
    for name in ['crp', 'kundu']:
        assert restored[f'comparator_{name}_pass'].isna().all()
        assert not restored[f'comparator_{name}_available'].any()
        assert restored[f'comparator_{name}_availability_reason'].eq('not_run').all()
    assert _audit(epochs, restored)['primary_qc_pass']
