"""Descriptive negative-N1 morphology, separate from response inference.

These features do not supply a p value or inherit the general detector's FDR
properties. They target the cortical-surface negative N1 annotation endpoint.
No annotation, participant identifier or archived detector call is an input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

N1_FEATURE_VERSION = "0.1.0-development"

MORPHOLOGY_FEATURES = (
    "log_negative_peak_z", "log_negative_prominence_z", "peak_latency_fraction",
    "log_peak_width_ms", "negative_area_fraction", "trial_negative_fraction",
    "trial_cosine", "log_preceding_positive_z", "log_response_rms_z",
    "log_baseline_std_uv", "log_negative_peak_uv", "log_n_feature_trials",
)
INFERENCE_FEATURES = ("negative_log10_p_reproducibility", "negative_log10_p_energy", "rms_ratio_db")


def n1_feature_vector(features: dict, inference: dict | None = None) -> np.ndarray:
    """Fixed transformations; centering/scaling belongs to training data only."""
    if not features.get("feature_available", False):
        raise ValueError("N1 features unavailable")
    if features.get('feature_version') != N1_FEATURE_VERSION or features.get('feature_config_sha256') != feature_config_sha256(N1FeatureConfig()) or features.get('voltage_unit') != 'uV':
        raise ValueError("Feature version, settings or voltage units do not match the trained model contract")
    nonnegative=['negative_peak_z','negative_prominence_z','peak_width_ms','preceding_positive_z',
                 'response_rms_z','baseline_std_uv','negative_peak_uv']
    for key in nonnegative:
        if not np.isfinite(float(features[key])) or float(features[key])<0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ['negative_area_fraction','trial_negative_fraction']:
        if not np.isfinite(float(features[key])) or not 0<=float(features[key])<=1:
            raise ValueError(f"{key} must be in [0,1]")
    if not np.isfinite(float(features['trial_cosine'])) or not -1-1e-12<=float(features['trial_cosine'])<=1+1e-12:
        raise ValueError("trial_cosine must be in [-1,1]")
    n=float(features['n_feature_trials'])
    if not np.isfinite(n) or n!=int(n) or n<N1FeatureConfig().min_trials:
        raise ValueError("n_feature_trials is below the feature contract minimum")
    latency=float(features['peak_latency_ms'])
    if not np.isfinite(latency) or not 9<=latency<=91:
        raise ValueError("Peak latency lies outside the declared N1 feature interval")
    if float(features['baseline_std_uv'])<=0:
        raise ValueError("Mean baseline SD must be positive")
    log_positive = lambda value: np.log1p(np.clip(float(value), 0, 1e6))
    vector = [log_positive(features['negative_peak_z']),
              log_positive(features['negative_prominence_z']),
              float(features['peak_latency_ms'])/100,
              log_positive(features['peak_width_ms']),
              float(features['negative_area_fraction']), float(features['trial_negative_fraction']),
              float(features['trial_cosine']), log_positive(features['preceding_positive_z']),
              log_positive(features['response_rms_z']), log_positive(features['baseline_std_uv']),
              log_positive(features['negative_peak_uv']), log_positive(features['n_feature_trials'])]
    if inference is not None:
        for key in ['p_reproducibility','p_energy']:
            if not np.isfinite(float(inference[key])) or not 0<=float(inference[key])<=1:
                raise ValueError(f"{key} must be finite and in [0,1]")
        if not np.isfinite(float(inference['rms_ratio_db'])):
            raise ValueError("rms_ratio_db must be finite")
        vector += [-np.log10(np.clip(float(inference[key]),1e-12,1))
                   for key in ['p_reproducibility','p_energy']]
        vector += [np.clip(float(inference['rms_ratio_db']),-100,100)]
    if not np.isfinite(vector).all():
        raise ValueError("N1 predictor inputs must be finite")
    return np.asarray(vector,float)


@dataclass(frozen=True)
class N1MorphologyModel:
    """Portable development classifier; a morphology score, not a p value.

    Training provenance and applicability must accompany each saved model.
    A positive result does not assert general response significance or FDR
    control. Negative N1 models must not be relabeled as validated sEEG models.
    """
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    threshold: float
    provenance: dict
    feature_version: str = N1_FEATURE_VERSION
    expected_feature_config_sha256: str = field(default_factory=lambda: feature_config_sha256(N1FeatureConfig()))
    voltage_unit: str = "uV"

    def __post_init__(self):
        if self.feature_version != N1_FEATURE_VERSION:
            raise ValueError("Unsupported N1 feature version")
        if self.expected_feature_config_sha256 != feature_config_sha256(N1FeatureConfig()) or self.voltage_unit != 'uV':
            raise ValueError("Saved model feature configuration or units differ from this implementation")
        expected = (MORPHOLOGY_FEATURES, MORPHOLOGY_FEATURES+INFERENCE_FEATURES)
        if tuple(self.feature_names) not in expected:
            raise ValueError("Unknown or reordered N1 feature schema")
        n = len(self.feature_names)
        if any(len(v)!=n for v in (self.mean,self.scale,self.coefficients)):
            raise ValueError("Model dimensions disagree with feature schema")
        numeric = [*self.mean,*self.scale,*self.coefficients,self.intercept,self.threshold]
        if not np.isfinite(numeric).all() or min(self.scale)<=0 or not 0<=self.threshold<=1:
            raise ValueError("Invalid model parameters")

    def predict(self, features: dict, inference: N1InferenceEvidence | None = None) -> dict:
        """Predict from morphology and, for hybrid models, checked evidence.

        Use :func:`prepare_n1_hybrid_inputs` for new arrays. Bare inference
        dictionaries are rejected: their window and trial origin are unknown.
        The public validation adapter supports hash-verified historical rows.
        """
        if not features.get('feature_available',False):
            return {'available':False,'score':None,'detected':False,
                    'reason':features.get('feature_reason','unavailable'),
                    'measured_peak_latency_ms':None,'peak_latency_imputed':False}
        hybrid = len(self.feature_names)>len(MORPHOLOGY_FEATURES)
        if hybrid and not isinstance(inference, N1InferenceEvidence):
            raise ValueError("Hybrid model requires checked early-window N1InferenceEvidence; use prepare_n1_hybrid_inputs")
        values = inference.validated_values(features) if hybrid else None
        vector = n1_feature_vector(features, values)
        logit = float(np.dot((vector-np.asarray(self.mean))/np.asarray(self.scale),self.coefficients)+self.intercept)
        score = float(np.exp(-np.logaddexp(0,-logit)))
        # Historical feature exports contain the peak-presence flag but not
        # the newer reporting fields. Derive timing from that original flag.
        peak_present = bool(features.get('negative_peak_present', False))
        return {'available':True,'score':score,'detected':bool(score>=self.threshold),
                'reason':'development_morphology_classifier','threshold':self.threshold,
                'measured_peak_latency_ms':float(features['peak_latency_ms']) if peak_present else None,
                'peak_latency_imputed':not peak_present}

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self),indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,path):
        values=json.loads(Path(path).read_text())
        if not {'expected_feature_config_sha256','voltage_unit','feature_version'}.issubset(values):
            raise ValueError("Saved model lacks an explicit feature contract")
        return cls(**values)


@dataclass(frozen=True)
class N1FeatureConfig:
    response_window: tuple[float, float] = (0.010, 0.090)
    baseline_window: tuple[float, float] = (-1.0, -0.1)
    normalization_window: tuple[float, float] = (-0.5, -0.02)
    min_trials: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


def feature_config_sha256(config: N1FeatureConfig) -> str:
    contract=dict(settings=config.to_dict(),voltage_unit='uV',feature_version=N1_FEATURE_VERSION)
    return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def extract_n1_features(trials, times, config: N1FeatureConfig | None = None) -> dict:
    """Describe the strongest interior negative peak on the native time grid.

    Trials have shape trial × time and voltage units must be microvolts.
    Baseline normalization uses each trial's median in the declared half-open
    rounded sample interval. Baseline noise is the sample SD of the trial mean.
    Prominence and half-prominence width are restricted to the response window.
    No smoothing, rereferencing, label-dependent window or trial rejection is
    used. Nonfinite trials are excluded on all required samples. An absent
    negative local peak is an evaluable no-peak result, not missing data.
    ``peak_latency_ms`` retains the model's midpoint imputation when no peak
    exists. Use ``measured_peak_latency_ms`` for reported timing; it is None
    without a measured peak, and ``peak_latency_imputed`` marks imputation.
    """
    cfg = config or N1FeatureConfig()
    x, t = np.asarray(trials, float), np.asarray(times, float)
    if x.ndim != 2 or t.ndim != 1 or x.shape[1] != len(t) or len(t) < 3:
        raise ValueError("Expected trial × time values and matching time vector")
    dt = np.diff(t)
    if not np.isfinite(t).all() or np.any(dt <= 0) or not np.allclose(dt, np.median(dt), rtol=1e-5, atol=1e-10):
        raise ValueError("Times must be finite, increasing and uniformly sampled")
    if isinstance(cfg.min_trials, bool) or int(cfg.min_trials) != cfg.min_trials or cfg.min_trials < 2:
        raise ValueError("min_trials must be an integer of at least two")
    fs = 1.0 / np.median(dt)
    windows = []
    for window in (cfg.response_window, cfg.baseline_window, cfg.normalization_window):
        if len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]:
            raise ValueError("Feature windows must contain two ordered finite endpoints")
        lo, hi = (int(round((v-t[0])*fs)) for v in window)
        if lo < 0 or hi > len(t) or hi-lo < 3:
            raise ValueError("Feature window unavailable or contains fewer than three samples")
        windows.append(np.arange(lo, hi))
    response, baseline, normalization = windows
    if cfg.response_window[0] <= 0 or max(cfg.baseline_window[1], cfg.normalization_window[1]) >= 0:
        raise ValueError("Response must follow stimulation and baselines must precede it")
    clean = np.isfinite(x[:, np.unique(np.concatenate(windows))]).all(axis=1)
    result = {"feature_version": N1_FEATURE_VERSION, "feature_available": False,
              "feature_config_sha256": feature_config_sha256(cfg), "voltage_unit": "uV",
              "feature_reason": "insufficient_finite_trials", "n_feature_trials": int(clean.sum()),
              "negative_peak_present": False, "measured_peak_latency_ms": None,
              "peak_latency_imputed": False}
    if clean.sum() < cfg.min_trials:
        return result
    x = x[clean].copy()
    x -= np.median(x[:, normalization], axis=1, keepdims=True)
    average = x.mean(axis=0)
    noise = float(np.std(average[baseline], ddof=1))
    required_scale=float(np.abs(average[np.unique(np.concatenate(windows))]).max())
    if not np.isfinite(noise) or noise <= np.finfo(float).eps * max(required_scale, 1.0):
        result["feature_reason"] = "degenerate_mean_baseline"
        return result
    y = average[response]
    trial_response = x[:, response]
    leave_out = (trial_response.sum(axis=0) - trial_response) / (len(x)-1)
    denominator = np.linalg.norm(trial_response, axis=1) * np.linalg.norm(leave_out, axis=1)
    cosines = np.divide(np.sum(trial_response*leave_out, axis=1), denominator,
                        out=np.zeros(len(x)), where=denominator > 0)
    negative_area = float(np.maximum(-y, 0).sum())
    absolute_area = float(np.abs(y).sum())
    result.update(feature_available=True, feature_reason="ok", baseline_std_uv=noise,
                  negative_peak_uv=0.0, negative_peak_z=0.0,
                  negative_prominence_uv=0.0, negative_prominence_z=0.0,
                  peak_latency_ms=1000*float(np.mean(cfg.response_window)), peak_width_ms=0.0,
                  peak_latency_imputed=True,
                  negative_area_fraction=negative_area/absolute_area if absolute_area else 0.0,
                  trial_negative_fraction=0.0, trial_cosine=float(np.mean(cosines)),
                  preceding_positive_z=0.0, response_rms_z=float(np.sqrt(np.mean(y*y))/noise),
                  realized_response_start_ms=float(t[response[0]]*1000),
                  realized_response_end_ms=float(t[response[-1]]*1000))
    peaks, _ = find_peaks(-y)
    peaks = peaks[y[peaks] < 0]
    if not len(peaks):
        result["feature_reason"] = "no_negative_interior_peak"
        return result
    peak = int(peaks[np.argmin(y[peaks])])
    prominence_data = peak_prominences(-y, np.array([peak]))
    prominence = float(prominence_data[0][0])
    width = float(peak_widths(-y, np.array([peak]), rel_height=.5,
                              prominence_data=prominence_data)[0][0] / fs * 1000)
    neighborhood = np.abs(response-response[peak]) <= .002*fs+1e-8
    amplitude = float(-y[peak])
    result.update(negative_peak_present=True, negative_peak_uv=amplitude,
                  negative_peak_z=amplitude/noise, negative_prominence_uv=prominence,
                  negative_prominence_z=prominence/noise, peak_latency_ms=float(t[response[peak]]*1000),
                  measured_peak_latency_ms=float(t[response[peak]]*1000), peak_latency_imputed=False,
                  peak_width_ms=width,
                  trial_negative_fraction=float(np.mean(np.mean(trial_response[:, neighborhood], axis=1) < 0)),
                  preceding_positive_z=float(max(np.max(y[:peak], initial=0.0), 0.0)/noise))
    return result


def n1_inference_config(random_state: int = 42):
    """The fixed inference settings used to train the development hybrid.

    Response 10–90 ms, baseline −1 to −0.1 s, artifact exclusion 0–9 ms.
    Randomization uses 5,000 draws and exact caps of 12 projection/16 energy
    trials. The seed is explicit; no general-interface defaults are inferred.
    """
    from .crp_energy import CRPEnergyConfig
    config = CRPEnergyConfig(
        response_window=(.010, .090), baseline_window=(-1., -.1),
        artifact_interval=(.0, .009), min_clean_trials=8,
        n_permutations=5000, max_exact_trials=16,
        max_exact_reproducibility_trials=12, canonical_energy_cv=False,
        random_state=random_state, alpha=.05, correction='fdr_bh',
        projection_start_s=.010, projection_step_s=.005, eps=1e-12,
    )
    config.validate()
    if random_state is None:
        raise ValueError("N1 inference requires an explicit random seed")
    return replace(config, random_state=int(random_state))


def _n1_feature_identity(features: dict) -> str:
    # Bind model inputs, timing interpretation and source/trial provenance.
    # Annotation labels and participant IDs never enter the numerical vector.
    vector = n1_feature_vector(features)
    provenance = features.get('_n1_input_provenance')
    if not isinstance(provenance, dict):
        raise ValueError("Missing N1 input/trial provenance; use a checked input constructor")
    material = dict(vector=vector.tolist(),
                    negative_peak_present=bool(features.get('negative_peak_present', False)),
                    provenance=provenance)
    return hashlib.sha256(json.dumps(material, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class N1InferenceEvidence:
    """Early-window evidence bound to one morphology input and trial source.

    Obtain this through ``prepare_n1_hybrid_inputs`` or the public frozen-record
    adapter, not by relabeling an arbitrary inference dictionary. Provenance is
    a reproducibility contract, not cryptographic authentication of a caller.
    """
    feature_sha256: str
    configuration_json: str
    provenance_json: str
    p_reproducibility: float
    p_energy: float
    rms_ratio_db: float

    def __post_init__(self):
        config = json.loads(self.configuration_json)
        expected = asdict(n1_inference_config(config.get('random_state')))
        if json.dumps(config, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("Inference configuration differs from the trained early-window N1 contract")
        if not isinstance(json.loads(self.provenance_json), dict):
            raise ValueError("N1 inference provenance must be an object")

    def validated_values(self, features: dict) -> dict:
        self.__post_init__()
        if self.feature_sha256 != _n1_feature_identity(features):
            raise ValueError("N1 morphology and inference input/epoch/trial identity mismatch")
        if json.loads(self.provenance_json) != features['_n1_input_provenance']:
            raise ValueError("N1 morphology and inference provenance mismatch")
        values = {name: getattr(self, name) for name in
                  ('p_reproducibility', 'p_energy', 'rms_ratio_db')}
        # Reuse the established domain checks without changing transformations.
        n1_feature_vector(features, values)
        return values

    @classmethod
    def _bind(cls, features: dict, values: dict, config, provenance: dict):
        """Internal constructor used after array execution or frozen-row checks."""
        bound = dict(features, _n1_input_provenance=provenance)
        evidence = cls(_n1_feature_identity(bound),
                       json.dumps(asdict(config), sort_keys=True, allow_nan=False),
                       json.dumps(provenance, sort_keys=True, allow_nan=False),
                       *(float(values[k]) for k in
                         ('p_reproducibility', 'p_energy', 'rms_ratio_db')))
        evidence.validated_values(bound)
        return bound, evidence


def prepare_n1_hybrid_inputs(trials, times, *, random_state: int = 42):
    """Construct matching morphology and early inference from the same arrays.

    Input is trial × time, in microvolts, with uniformly sampled seconds.
    Both paths use the declared per-trial median normalization over the rounded
    half-open [−0.5, −0.02) s slice. Their finite-trial masks are retained
    separately: morphology uses its complete declared windows; inference uses
    its effective response and sample-matched baseline. They need not coincide.
    Returns ``(features, evidence)`` for ``model.predict(*inputs)``; unavailable
    morphology returns ``(features, None)`` and produces a no-call.
    """
    from .crp_energy import run_crp_energy_array
    x, t = np.asarray(trials, dtype=float), np.asarray(times, dtype=float)
    features = extract_n1_features(x, t)  # Validates shape, grid and windows.
    config = n1_inference_config(random_state)
    if not features['feature_available']:
        return features, None
    fs = 1 / float(np.median(np.diff(t)))
    cfg = N1FeatureConfig()
    windows = [np.arange(*(int(round((v-t[0])*fs)) for v in window))
               for window in (cfg.response_window, cfg.baseline_window, cfg.normalization_window)]
    feature_clean = np.flatnonzero(np.isfinite(x[:, np.unique(np.concatenate(windows))]).all(axis=1))
    normalized = x - np.nanmedian(x[:, windows[2]], axis=1, keepdims=True)
    result = run_crp_energy_array(normalized, t, config=config)
    values = dict(p_reproducibility=result.p_crp, p_energy=result.p_energy,
                  rms_ratio_db=result.rms_ratio_db)
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Required early-window inference is unavailable for these trials")
    def array_hash(value):
        a = np.ascontiguousarray(value, dtype='<f8')
        return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()
    provenance = dict(
        kind='same_array_execution', input_sha256=array_hash(x), times_sha256=array_hash(t),
        voltage_unit='uV', normalization_window_s=list(cfg.normalization_window),
        normalization='per-trial nanmedian; half-open rounded sample slice',
        n_trials_total=len(x), feature_clean_trial_indices=feature_clean.tolist(),
        inference_clean_trial_indices=result.clean_trial_indices.tolist(),
        realized_response_window_s=list(result.response_window),
        realized_baseline_window_s=list(result.baseline_window),
        n_response_samples=result.n_response_samples, n_baseline_samples=result.n_baseline_samples,
    )
    return N1InferenceEvidence._bind(features, values, config, provenance)
