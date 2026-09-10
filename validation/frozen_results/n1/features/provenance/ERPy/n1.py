"""Descriptive negative-N1 morphology, separate from response inference.

These features do not supply a p value or inherit the general detector's FDR
properties. They target the cortical-surface negative N1 annotation endpoint.
No annotation, participant identifier or archived detector call is an input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    def predict(self, features: dict, inference: dict | None = None) -> dict:
        if not features.get('feature_available',False):
            return {'available':False,'score':None,'detected':False,'reason':features.get('feature_reason','unavailable')}
        hybrid = len(self.feature_names)>len(MORPHOLOGY_FEATURES)
        if hybrid and inference is None:
            raise ValueError("Hybrid model requires general inference features")
        vector = n1_feature_vector(features,inference if hybrid else None)
        logit = float(np.dot((vector-np.asarray(self.mean))/np.asarray(self.scale),self.coefficients)+self.intercept)
        score = float(np.exp(-np.logaddexp(0,-logit)))
        return {'available':True,'score':score,'detected':bool(score>=self.threshold),
                'reason':'development_morphology_classifier','threshold':self.threshold}

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
              "negative_peak_present": False}
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
                  peak_width_ms=width,
                  trial_negative_fraction=float(np.mean(np.mean(trial_response[:, neighborhood], axis=1) < 0)),
                  preceding_positive_z=float(max(np.max(y[:peak], initial=0.0), 0.0)/noise))
    return result
