"""Fixed, label-free negative-N1 screening with full-family joint inference.

The p/q values concern the original projection-energy union null, not the
truth of a physiological N1 label. BH dependence assumptions must apply to
the screened p vector. No fitted model or annotation is an input.
"""
from dataclasses import asdict
import hashlib

import numpy as np

from .crp_energy import run_crp_energy_array
from .inference import fdr_bh
from .n1 import (N1FeatureConfig, N1_FEATURE_VERSION, extract_n1_features,
                 feature_config_sha256, n1_inference_config)


RULE_VERSION = "1.0.0"
AMPLITUDE_FACTOR = 3.4
BASELINE_SD_FLOOR_UV = 50.0
FAMILY_ALPHA = 0.05


def morphology_gate(features, *, baseline_sd_floor_uv=BASELINE_SD_FLOOR_UV):
    """Apply the fixed primary gate or its sole baseline-only ablation.

    This is a descriptive amplitude rule, not an N1-specific p value. The
    family-array API constructs the features itself and binds their source.
    """
    if baseline_sd_floor_uv not in (0.0, BASELINE_SD_FLOOR_UV):
        raise ValueError("Use the fixed 50 uV primary floor or 0 uV ablation")
    if (features.get('voltage_unit') != 'uV'
            or features.get('feature_version') != N1_FEATURE_VERSION
            or features.get('feature_config_sha256') != feature_config_sha256(N1FeatureConfig())):
        raise ValueError("Features differ from the fixed N1 window/version/microvolt contract")
    if not isinstance(features.get('feature_available'), (bool, np.bool_)):
        raise ValueError("Feature availability must be an explicit boolean")
    if not features['feature_available']:
        return {'available': False, 'passed': False, 'threshold_uv': None,
                'reason': 'morphology_unavailable'}
    if not isinstance(features.get('negative_peak_present'), (bool, np.bool_)):
        raise ValueError("Peak availability must be an explicit boolean")
    amplitude = float(features['negative_peak_uv'])
    noise = float(features['baseline_std_uv'])
    if not np.isfinite([amplitude, noise]).all() or amplitude < 0 or noise <= 0:
        raise ValueError("Peak amplitude and baseline SD must be finite in microvolts")
    threshold = AMPLITUDE_FACTOR * max(noise, baseline_sd_floor_uv)
    present = bool(features['negative_peak_present'])
    passed = present and amplitude >= threshold
    return {'available': True, 'passed': passed, 'threshold_uv': threshold,
            'reason': 'passed' if passed else ('amplitude_gate_failed' if present
                                              else 'no_negative_interior_peak')}


def _apply_complete_family(records):
    """Internal common arithmetic after array or pinned-record validation.

    Every original-finite contact remains in the correction, including a
    contact without morphology or subsequent scoring/annotation eligibility.
    This helper does not establish provenance of caller-created records.
    """
    rows = [dict(row) for row in records]
    p = np.array([float(row['p_joint']) for row in rows])
    finite = np.isfinite(p)
    if np.any((p[finite] < 0) | (p[finite] > 1)):
        raise ValueError("Finite joint p values must lie in [0, 1]")
    original_q, original_call = fdr_bh(p, alpha=FAMILY_ALPHA)
    for prefix, floor in [('', BASELINE_SD_FLOOR_UV), ('ablation_', 0.0)]:
        gates = [morphology_gate(row['features'], baseline_sd_floor_uv=floor) for row in rows]
        passed = np.array([g['passed'] for g in gates], dtype=bool)
        screened = np.where(finite, np.where(passed, p, 1.0), np.nan)
        q, calls = fdr_bh(screened, alpha=FAMILY_ALPHA)
        if np.any(q[finite] + 1e-14 < original_q[finite]) or np.any(calls & ~original_call):
            raise AssertionError("Screened-family monotonicity violated")
        for i, (row, gate) in enumerate(zip(rows, gates)):
            row.update({prefix+'morphology_gate': gate['passed'],
                        prefix+'threshold_uv': gate['threshold_uv'],
                        prefix+'p_screened': float(screened[i]),
                        prefix+'q_screened': float(q[i]),
                        prefix+'detected': bool(calls[i] and gate['passed'] and gate['available']),
                        prefix+'available': bool(finite[i] and gate['available'])})
            if not finite[i]:
                stage = 'inference_unavailable'
            elif not gate['available']:
                stage = 'morphology_unavailable'
            elif not gate['passed']:
                stage = gate['reason']
            elif p[i] > FAMILY_ALPHA:
                stage = 'gate_pass_joint_p_above_0.05'
            elif not calls[i]:
                stage = 'gate_pass_joint_p_at_most_0.05_but_screened_q_above_0.05'
            else:
                stage = 'final_pass'
            row[prefix+'decision_stage'] = stage
    for i, row in enumerate(rows):
        row.update(family_size=int(finite.sum()), original_q_joint=float(original_q[i]),
                   original_detected=bool(original_call[i]), rule_version=RULE_VERSION)
    return rows


def _array_hash(value):
    a = np.ascontiguousarray(value, dtype='<f8')
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def detect_n1_family(trials, times, channel_names, *, stimulation_contacts=(),
                     random_seeds=None, family_id, voltage_unit='uV'):
    """Detect fixed early negative N1 in a complete declared contact family.

    Supply trial x time x channel values in microvolts, uniform times in
    seconds, and all retained good ECoG channels for one stimulation pair.
    ``stimulation_contacts`` are excluded before inference; other contacts
    must not be selected by morphology, annotations, spatial scoring rules,
    or another algorithm. The caller establishes acquisition completeness.

    Both scientific paths are computed from the same arrays. Separate actual
    finite-trial masks and realized inference windows are retained. There is
    no precomputed-inference dictionary or learned model to mix accidentally.
    Explicit ``random_seeds`` supplies one exact nonnegative integer per input
    channel; otherwise all channels use the documented seed42. Returned
    contacts include the fixed50uV-floor primary and sole0-floor ablation.
    """
    if voltage_unit != 'uV':
        raise ValueError("Input voltage_unit must be uV (microvolts)")
    if not isinstance(family_id, str) or not family_id.strip():
        raise ValueError("A nonempty declared family_id is required")
    x, t = np.asarray(trials, float), np.asarray(times, float)
    names = [str(name).strip().upper() for name in channel_names]
    if (x.ndim != 3 or t.ndim != 1 or x.shape[1:] != (len(t), len(names))
            or not names or any(not name for name in names) or len(set(names)) != len(names)):
        raise ValueError("Expected trial x time x unique named channels")
    stim = {str(name).strip().upper() for name in stimulation_contacts}
    if not stim.issubset(names):
        raise ValueError("Stimulation contacts must belong to the declared input family")
    seeds = [42] * len(names) if random_seeds is None else list(random_seeds)
    if len(seeds) != len(names):
        raise ValueError("Provide one exact random seed per input channel")
    configs = [n1_inference_config(seed) for seed in seeds]
    # Feature extraction validates every required window, shape and sample grid
    # before deriving indices; no baseline clipping or implicit unit conversion.
    descriptors = [extract_n1_features(x[:, :, i], t) for i in range(len(names))]
    fs = 1.0 / float(np.median(np.diff(t)))
    cfg = N1FeatureConfig()
    windows = [np.arange(*(int(round((v-t[0])*fs)) for v in window))
               for window in (cfg.response_window, cfg.baseline_window, cfg.normalization_window)]
    source_hash, times_hash = _array_hash(x), _array_hash(t)
    rows = []
    for index, name in enumerate(names):
        if name in stim:
            continue
        contact = x[:, :, index]
        feature_clean = np.flatnonzero(np.isfinite(contact[:, np.unique(np.concatenate(windows))]).all(axis=1))
        normalized = contact - np.nanmedian(contact[:, windows[2]], axis=1, keepdims=True)
        result = run_crp_energy_array(normalized, t, channel=name, config=configs[index])
        provenance = dict(kind='same_array_execution', family_input_sha256=source_hash,
            contact_input_sha256=_array_hash(contact), times_sha256=times_hash,
            voltage_unit=voltage_unit, n_trials_total=len(contact),
            feature_clean_trial_indices=feature_clean.tolist(),
            inference_clean_trial_indices=result.clean_trial_indices.tolist(),
            realized_response_window_s=list(result.response_window),
            realized_baseline_window_s=list(result.baseline_window),
            n_response_samples=result.n_response_samples, n_baseline_samples=result.n_baseline_samples,
            feature_config=asdict(cfg), inference_config=asdict(configs[index]))
        rows.append(dict(channel=name, family_id=family_id, features=descriptors[index],
            p_reproducibility=float(result.p_crp), p_energy=float(result.p_energy),
            p_joint=float(result.p_joint), rms_ratio_db=float(result.rms_ratio_db),
            n_trials_total=result.n_trials_total, n_trials_clean=result.n_trials_clean,
            random_seed=configs[index].random_state, provenance=provenance))
    contacts = _apply_complete_family(rows)
    return dict(rule_version=RULE_VERSION, family_id=family_id,
                declared_channels=names, stimulation_contacts=sorted(stim),
                family_size=int(sum(np.isfinite(row['p_joint']) for row in contacts)),
                contacts=contacts,
                inference_scope='original projection-energy union null; not N1-truth FDR',
                multiplicity='BH alpha0.05 on screened p values in the complete original finite family; dependence assumptions required')
