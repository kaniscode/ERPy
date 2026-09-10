"""Fixed-window reproducibility-energy inference with descriptive CRP effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any

import numpy as np

from .crp import CRPConfig, CRPError, run_crp_array


CRP_ENERGY_DETECTOR_VERSION = "1.0.0"


def _validated_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Return an integer setting after rejecting fractional/nonfinite values."""

    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(numeric) or numeric != float(integer):
        raise ValueError(f"{name} must be an integer")
    if integer < int(minimum) or (
        maximum is not None and integer > int(maximum)
    ):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}")
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return integer


def _validated_window(
    value: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[float, float]:
    """Return a finite, ordered two-element time window."""

    try:
        if len(value) != 2:
            raise ValueError
        start, stop = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError(f"{name} must contain exactly two finite values") from exc
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError(f"{name} must contain exactly two finite values")
    if start > stop or (start == stop and not allow_empty):
        relation = "start at or before" if allow_empty else "start before"
        raise ValueError(f"{name} must {relation} its stop")
    return start, stop


def _validated_random_state(
    value: Any,
    *,
    allow_generator: bool,
) -> int | np.random.Generator | None:
    """Validate an optional nonnegative integer seed or allowed generator."""

    if value is None:
        return None
    if allow_generator and isinstance(value, np.random.Generator):
        return value
    return _validated_integer(
        value,
        name="random_state",
        minimum=0,
    )


@dataclass(frozen=True)
class CRPEnergyConfig:
    """Configuration for the CRP-energy conjunction.

    The response and baseline windows are matched sample-for-sample. Samples
    in ``artifact_interval`` are excluded before either component is fitted.
    The inferential reproducibility window is the complete effective response
    window; CRP duration selection is retained only for descriptive effects.
    """

    response_window: tuple[float, float] = (0.015, 1.0)
    baseline_window: tuple[float, float] = (-1.0, -0.015)
    alpha: float = 0.05
    correction: str = "fdr_bh"
    min_clean_trials: int = 8
    n_permutations: int = 5_000
    max_exact_trials: int = 16
    max_exact_reproducibility_trials: int = 12
    canonical_energy_cv: bool = True
    random_state: int | None = 42
    artifact_interval: tuple[float, float] = (0.0, 0.015)
    projection_start_s: float = 0.010
    projection_step_s: float = 0.005
    eps: float = 1e-12

    def validate(self) -> None:
        """Validate scientific windows and finite inference settings.

        Validation is explicit because silently coercing an invalid window,
        seed, or randomization count into an ``insufficient_data`` result would
        make configuration errors indistinguishable from unsuitable data.
        """

        response_start, response_stop = _validated_window(
            self.response_window,
            name="response_window",
        )
        _, baseline_stop = _validated_window(
            self.baseline_window,
            name="baseline_window",
        )
        _, artifact_stop = _validated_window(
            self.artifact_interval,
            name="artifact_interval",
            allow_empty=True,
        )
        effective_response_start = max(response_start, artifact_stop)
        if effective_response_start >= response_stop:
            raise ValueError(
                "artifact_interval must leave a nonempty response_window"
            )
        if baseline_stop >= effective_response_start:
            raise ValueError(
                "baseline_window must end before the effective response_window"
            )
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and between zero and one")
        if str(self.correction).strip().lower() not in {"fdr_bh", "none"}:
            raise ValueError("correction must be 'fdr_bh' or 'none'")
        _validated_integer(
            self.min_clean_trials,
            name="min_clean_trials",
            minimum=2,
        )
        _validated_integer(
            self.n_permutations,
            name="n_permutations",
            minimum=1,
        )
        _validated_integer(
            self.max_exact_trials,
            name="max_exact_trials",
            minimum=0,
            maximum=18,
        )
        _validated_integer(
            self.max_exact_reproducibility_trials,
            name="max_exact_reproducibility_trials",
            minimum=0,
            maximum=18,
        )
        for name, value in (
            ("projection_start_s", self.projection_start_s),
            ("projection_step_s", self.projection_step_s),
            ("eps", self.eps),
        ):
            numeric = float(value)
            if not np.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        _validated_random_state(self.random_state, allow_generator=False)


@dataclass
class CRPEnergyResult:
    """One channel's unadjusted reproducibility-energy conjunction result.

    For backward compatibility, ``p_crp`` and ``crp_statistic`` store the
    fixed-window reproducibility value :math:`p_R` and statistic :math:`T_R`.
    They are distinct from the standalone, selected-duration CRP comparator.
    The canonical waveform, response duration, coefficients, SNR, and
    explained fraction remain descriptive CRP quantities.
    """

    channel: str
    significant: bool
    classification: str
    p_crp: float
    p_energy: float
    p_joint: float
    q_joint: float
    crp_significant: bool
    energy_significant: bool
    crp_statistic: float
    energy_statistic: float
    rms_response: float
    rms_baseline: float
    rms_ratio_db: float
    canonical_energy: float
    canonical_energy_fraction: float
    response_duration: float
    n_trials_total: int
    n_trials_clean: int
    canonical_waveform: np.ndarray
    canonical_waveform_times: np.ndarray
    trial_coefficients: np.ndarray
    clean_trial_indices: np.ndarray
    crp_explained_variance: float
    crp_snr: float
    reproducibility_test_exact: bool
    reproducibility_n_randomizations: int
    energy_test_exact: bool
    energy_n_permutations: int
    response_window: tuple[float, float]
    baseline_window: tuple[float, float]
    qc_status: str
    parameters: str
    detector_version: str = CRP_ENERGY_DETECTOR_VERSION
    notes: str = ""
    n_response_samples: int = 0
    n_baseline_samples: int = 0

    def to_record(self, *, include_arrays: bool = True) -> dict[str, Any]:
        """Return a detector-table record with optionally compact array fields."""

        record = asdict(self)
        for key in (
            "canonical_waveform",
            "canonical_waveform_times",
            "trial_coefficients",
            "clean_trial_indices",
        ):
            value = np.asarray(record[key])
            record[key] = value.tolist() if include_arrays else None
        return record


@dataclass(frozen=True)
class FixedWindowReproducibilityResult:
    """Result of the trial-level fixed-window reproducibility randomization."""

    statistic: float
    p_value: float
    exact: bool
    n_randomizations: int


def _ordered_projection_matrix(
    response: np.ndarray,
    *,
    sfreq: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return ordered semi-normalized projections with a zero diagonal."""

    values = np.asarray(response, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(
            "response must be a finite trial-by-time array with at least two trials and samples"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("response must contain only finite values")
    sfreq = float(sfreq)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive and finite")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= eps):
        raise ValueError("response contains a degenerate zero-energy trial")
    normalized = values / norms[:, None]
    projections = normalized @ values.T / math.sqrt(sfreq)
    np.fill_diagonal(projections, 0.0)
    return projections


def _unique_quadratic_sign_patterns(n_trials: int) -> np.ndarray:
    """Enumerate sign patterns after fixing the globally redundant first sign."""

    n_trials = int(n_trials)
    assignments = np.arange(1 << (n_trials - 1), dtype=np.uint64)
    bit_positions = np.arange(n_trials - 1, dtype=np.uint64)
    trailing = 2.0 * (
        (assignments[:, None] >> bit_positions[None, :]) & 1
    ).astype(float) - 1.0
    return np.c_[np.ones(len(assignments), dtype=float), trailing]


def fixed_window_reproducibility_sign_flip_test(
    response: np.ndarray,
    *,
    sfreq: float,
    n_permutations: int = 5_000,
    max_exact_trials: int = 12,
    random_state: int | np.random.Generator | None = 42,
    eps: float = 1e-12,
) -> FixedWindowReproducibilityResult:
    """Test fixed-window trial reproducibility by whole-trial sign flips.

    The statistic is the mean of all ordered, semi-normalized cross-trial
    projections. Whole-trial Rademacher signs preserve every trial norm and
    account for the dependence among projections that share trials. The
    null requires joint invariance of the response vectors under independent
    whole-trial sign changes; independent centrally symmetric trials suffice.
    This condition must hold after preprocessing and trial selection. The
    response window must be fixed before this function is called; a
    data-selected CRP duration must not define this inferential array.

    For exact enumeration, the observed all-positive assignment is omitted
    from the matrix calculation and counted explicitly. This guarantees the
    exact randomization floor even when floating-point summation order makes
    the separately calculated observed statistic differ by roundoff.
    """

    n_draws_requested = _validated_integer(
        n_permutations,
        name="n_permutations",
        minimum=1,
    )
    max_exact_trials = _validated_integer(
        max_exact_trials,
        name="max_exact_trials",
        minimum=0,
        maximum=18,
    )
    validated_random_state = _validated_random_state(
        random_state,
        allow_generator=True,
    )
    projections = _ordered_projection_matrix(
        response,
        sfreq=sfreq,
        eps=eps,
    )
    n_trials = int(projections.shape[0])
    denominator = float(n_trials * (n_trials - 1))
    observed = float(np.sum(projections) / denominator)
    if n_trials <= max_exact_trials:
        total_assignments = 1 << (n_trials - 1)
        signs = _unique_quadratic_sign_patterns(n_trials)[:-1]
        null_statistics = np.einsum(
            "bi,ij,bj->b",
            signs,
            projections,
            signs,
            optimize=True,
        ) / denominator
        comparison_scale = max(
            abs(observed),
            float(np.max(np.abs(null_statistics), initial=0.0)),
            np.finfo(float).tiny,
        )
        tolerance = 64.0 * np.finfo(float).eps * comparison_scale
        extreme_count = int(
            np.sum(null_statistics >= observed - tolerance)
        )
        p_value = float((1 + extreme_count) / total_assignments)
        return FixedWindowReproducibilityResult(
            statistic=observed,
            p_value=p_value,
            exact=True,
            n_randomizations=int(total_assignments - 1),
        )

    n_draws = max(n_draws_requested, 5_000)
    rng = (
        validated_random_state
        if isinstance(validated_random_state, np.random.Generator)
        else np.random.default_rng(validated_random_state)
    )
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(n_draws, n_trials),
        replace=True,
    )
    signs[:, 0] = 1.0
    null_statistics = np.einsum(
        "bi,ij,bj->b",
        signs,
        projections,
        signs,
        optimize=True,
    ) / denominator
    comparison_scale = max(
        abs(observed),
        float(np.max(np.abs(null_statistics), initial=0.0)),
        np.finfo(float).tiny,
    )
    tolerance = 64.0 * np.finfo(float).eps * comparison_scale
    extreme_count = int(
        np.sum(null_statistics >= observed - tolerance)
    )
    return FixedWindowReproducibilityResult(
        statistic=observed,
        p_value=float((1 + extreme_count) / (n_draws + 1)),
        exact=False,
        n_randomizations=n_draws,
    )


def paired_log_rms_sign_flip_test(
    log_rms_differences: np.ndarray,
    *,
    n_permutations: int = 5_000,
    max_exact_trials: int = 16,
    random_state: int | np.random.Generator | None = 42,
) -> tuple[float, float, bool, int]:
    """Upper-tail sign-flip test of paired log RMS differences.

    Null validity requires joint invariance under coordinate-wise sign
    changes, as supplied by independent centrally symmetric differences.
    It is not an unrestricted finite-sample test of every distribution with
    a nonpositive mean difference.

    Exact enumeration omits the observed all-positive assignment and adds it
    back through the standard plus-one formula. This is algebraically the
    exact randomization p-value while sharing the Monte Carlo implementation's
    finite-sample form.
    """

    n_draws_requested = _validated_integer(
        n_permutations,
        name="n_permutations",
        minimum=1,
    )
    max_exact_trials = _validated_integer(
        max_exact_trials,
        name="max_exact_trials",
        minimum=0,
        maximum=18,
    )
    validated_random_state = _validated_random_state(
        random_state,
        allow_generator=True,
    )
    differences = np.asarray(log_rms_differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not differences.size:
        return np.nan, np.nan, False, 0
    observed = float(np.mean(differences))
    n_trials = int(len(differences))
    tolerance = 1e-15
    if n_trials <= max_exact_trials:
        total_assignments = 1 << n_trials
        assignments = np.arange(total_assignments - 1, dtype=np.uint64)
        bit_positions = np.arange(n_trials, dtype=np.uint64)
        signs = 2.0 * (
            (assignments[:, None] >> bit_positions[None, :]) & 1
        ).astype(float) - 1.0
        null_statistics = signs @ differences / float(n_trials)
        extreme_count = int(
            np.sum(null_statistics >= observed - tolerance)
        )
        p_value = float((1 + extreme_count) / total_assignments)
        return observed, p_value, True, int(total_assignments - 1)

    n_draws = max(n_draws_requested, 5_000)
    rng = (
        validated_random_state
        if isinstance(validated_random_state, np.random.Generator)
        else np.random.default_rng(validated_random_state)
    )
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(n_draws, n_trials),
        replace=True,
    )
    null_statistics = signs @ differences / float(n_trials)
    extreme_count = int(np.sum(null_statistics >= observed - tolerance))
    p_value = float((1 + extreme_count) / (n_draws + 1))
    return observed, p_value, False, n_draws


def run_crp_energy_array(
    x: np.ndarray,
    times: np.ndarray,
    *,
    channel: str = "",
    config: CRPEnergyConfig | None = None,
) -> CRPEnergyResult:
    """Evaluate fixed-window reproducibility and paired energy for one channel.

    Whole-trial sign flips supply the reproducibility component over the full
    declared response window. Matched response and baseline segments are each
    demeaned separately before paired log-RMS sign flips. A descriptive CRP
    model is fitted independently of the inferential window selection. An
    unavailable descriptive model does not discard valid component p-values.
    Epochs must cover the effective response window within half a sample;
    truncated windows return an unavailable result rather than a shorter test.
    """

    config = config or CRPEnergyConfig()
    config.validate()
    values = np.asarray(x, dtype=float)
    time_values = np.asarray(times, dtype=float).squeeze()
    if values.ndim != 2 or time_values.ndim != 1:
        raise ValueError("x must be trial by time and times must be one-dimensional")
    if values.shape[1] != len(time_values):
        raise ValueError("x and times must have matching time dimensions")
    if (
        len(time_values) < 2
        or not np.all(np.isfinite(time_values))
        or not np.all(np.diff(time_values) > 0)
    ):
        raise ValueError("times must be finite and strictly increasing")
    time_steps = np.diff(time_values)
    median_step = float(np.median(time_steps))
    if not np.allclose(
        time_steps,
        median_step,
        rtol=1e-6,
        atol=max(1e-12, abs(median_step) * 1e-9),
    ):
        raise ValueError("times must be uniformly sampled")

    response_start = max(
        float(config.response_window[0]),
        float(config.artifact_interval[1]),
    )
    response_stop = float(config.response_window[1])
    baseline_start, baseline_stop = map(float, config.baseline_window)
    parameter_json = _parameter_json(config)
    coverage_tolerance = median_step / 2.0 + 1e-12
    if (
        time_values[0] > response_start + coverage_tolerance
        or time_values[-1] < response_stop - coverage_tolerance
    ):
        return _empty_result(
            channel=channel,
            n_trials_total=values.shape[0],
            response_window=(response_start, response_stop),
            baseline_window=(baseline_start, baseline_stop),
            parameters=parameter_json,
            notes=(
                "Declared response window is unavailable: epoch times must "
                "cover both effective endpoints within half a sample"
            ),
        )
    response_indices = np.flatnonzero(
        (time_values >= response_start) & (time_values <= response_stop)
    )
    baseline_candidates = np.flatnonzero(
        (time_values >= baseline_start) & (time_values <= baseline_stop)
    )
    if not response_indices.size or len(baseline_candidates) < len(response_indices):
        return _empty_result(
            channel=channel,
            n_trials_total=values.shape[0],
            response_window=(response_start, response_stop),
            baseline_window=(baseline_start, baseline_stop),
            parameters=parameter_json,
            notes=(
                "Matched response and baseline windows are unavailable; "
                "the baseline must contain at least as many samples as the response"
            ),
        )
    baseline_indices = baseline_candidates[-len(response_indices) :]
    matched_response_window = (
        float(time_values[response_indices[0]]),
        float(time_values[response_indices[-1]]),
    )
    matched_baseline_window = (
        float(time_values[baseline_indices[0]]),
        float(time_values[baseline_indices[-1]]),
    )
    clean = np.all(
        np.isfinite(values[:, np.r_[baseline_indices, response_indices]]),
        axis=1,
    )
    clean_indices = np.flatnonzero(clean)
    if len(clean_indices) < int(config.min_clean_trials):
        return _empty_result(
            channel=channel,
            n_trials_total=values.shape[0],
            n_trials_clean=len(clean_indices),
            clean_trial_indices=clean_indices,
            n_response_samples=len(response_indices),
            n_baseline_samples=len(baseline_indices),
            response_window=matched_response_window,
            baseline_window=matched_baseline_window,
            parameters=parameter_json,
            notes=(
                "Only {} clean matched trials remain; {} are required".format(
                    len(clean_indices),
                    int(config.min_clean_trials),
                )
            ),
        )

    clean_values = values[clean_indices].copy()
    baseline_means = np.mean(clean_values[:, baseline_indices], axis=1)
    centered = clean_values - baseline_means[:, None]
    root_random_state = _validated_random_state(
        config.random_state,
        allow_generator=False,
    )
    component_seed = np.random.SeedSequence(root_random_state)
    reproducibility_seed, energy_seed = component_seed.spawn(2)
    crp_config = CRPConfig(
        response_window=matched_response_window,
        baseline_window=matched_baseline_window,
        min_trials=int(config.min_clean_trials),
        alpha=float(config.alpha),
        projection_start_s=float(config.projection_start_s),
        projection_step_s=float(config.projection_step_s),
        permutation_n=0,
        random_state=root_random_state,
        eps=float(config.eps),
    )
    crp_result = None
    crp_error = ""
    try:
        crp_result = run_crp_array(
            centered,
            time_values,
            channel=str(channel),
            config=crp_config,
        )
    except CRPError as exc:
        crp_error = str(exc)

    reproducibility_result = None
    reproducibility_error = ""
    try:
        reproducibility_result = fixed_window_reproducibility_sign_flip_test(
            centered[:, response_indices],
            sfreq=float(1.0 / np.median(np.diff(time_values))),
            n_permutations=int(config.n_permutations),
            max_exact_trials=int(
                config.max_exact_reproducibility_trials
            ),
            random_state=np.random.default_rng(reproducibility_seed),
            eps=float(config.eps),
        )
    except ValueError as exc:
        reproducibility_error = str(exc)

    response_segment = clean_values[:, response_indices]
    baseline_segment = clean_values[:, baseline_indices]
    response_residual = response_segment - np.mean(
        response_segment,
        axis=1,
        keepdims=True,
    )
    baseline_residual = baseline_segment - np.mean(
        baseline_segment,
        axis=1,
        keepdims=True,
    )
    response_rms = _row_rms(response_residual)
    baseline_rms = _row_rms(baseline_residual)
    eps = max(float(config.eps), np.finfo(float).tiny)
    log_differences = np.log(response_rms + eps) - np.log(baseline_rms + eps)
    energy_statistic, p_energy, exact, n_permutations = (
        paired_log_rms_sign_flip_test(
            log_differences,
            n_permutations=int(config.n_permutations),
            max_exact_trials=int(config.max_exact_trials),
            random_state=np.random.default_rng(energy_seed),
        )
    )
    rms_response = float(np.exp(np.mean(np.log(response_rms + eps))))
    rms_baseline = float(np.exp(np.mean(np.log(baseline_rms + eps))))
    rms_ratio_db = float(20.0 / np.log(10.0) * energy_statistic)

    if reproducibility_result is None:
        p_crp = np.nan
        crp_statistic = np.nan
        qc_status = "reproducibility_unavailable"
    else:
        p_crp = float(reproducibility_result.p_value)
        crp_statistic = float(reproducibility_result.statistic)
        qc_status = "pass"

    if crp_result is None:
        duration = np.nan
        waveform = np.array([], dtype=float)
        waveform_times = np.array([], dtype=float)
        coefficients = np.array([], dtype=float)
        explained = np.nan
        snr = np.nan
        canonical_energy = np.nan
        canonical_fraction = np.nan
    else:
        duration = float(crp_result.response_duration_s)
        waveform = np.asarray(crp_result.canonical_waveform, dtype=float)
        waveform_times = np.asarray(crp_result.times, dtype=float)
        coefficients = np.asarray(crp_result.projections, dtype=float)
        explained = _finite_median(crp_result.explained_variance)
        snr = _finite_median(crp_result.snr)
        canonical_energy, canonical_fraction = _canonical_energy_effects(
            centered[:, response_indices],
            response_samples=len(waveform),
            cross_validated=bool(config.canonical_energy_cv),
            eps=eps,
        )

    crp_pass = bool(np.isfinite(p_crp) and p_crp <= float(config.alpha))
    energy_pass = bool(
        np.isfinite(p_energy) and p_energy <= float(config.alpha)
    )
    p_joint = (
        float(max(p_crp, p_energy))
        if np.isfinite(p_crp) and np.isfinite(p_energy)
        else np.nan
    )
    classification = (
        _classification(crp_pass, energy_pass)
        if np.isfinite(p_joint)
        else "insufficient_data"
    )
    return CRPEnergyResult(
        channel=str(channel),
        significant=bool(
            qc_status == "pass"
            and np.isfinite(p_joint)
            and p_joint <= float(config.alpha)
        ),
        classification=classification,
        p_crp=p_crp,
        p_energy=float(p_energy),
        p_joint=p_joint,
        q_joint=np.nan,
        crp_significant=crp_pass,
        energy_significant=energy_pass,
        crp_statistic=crp_statistic,
        energy_statistic=float(energy_statistic),
        rms_response=rms_response,
        rms_baseline=rms_baseline,
        rms_ratio_db=rms_ratio_db,
        canonical_energy=canonical_energy,
        canonical_energy_fraction=canonical_fraction,
        response_duration=duration,
        n_trials_total=int(values.shape[0]),
        n_trials_clean=int(len(clean_indices)),
        canonical_waveform=waveform,
        canonical_waveform_times=waveform_times,
        trial_coefficients=coefficients,
        clean_trial_indices=clean_indices,
        crp_explained_variance=explained,
        crp_snr=snr,
        reproducibility_test_exact=bool(
            reproducibility_result.exact
            if reproducibility_result is not None
            else False
        ),
        reproducibility_n_randomizations=int(
            reproducibility_result.n_randomizations
            if reproducibility_result is not None
            else 0
        ),
        energy_test_exact=bool(exact),
        energy_n_permutations=int(n_permutations),
        response_window=matched_response_window,
        baseline_window=matched_baseline_window,
        qc_status=qc_status,
        n_response_samples=len(response_indices),
        n_baseline_samples=len(baseline_indices),
        parameters=parameter_json,
        notes="; ".join(
            note
            for note in (crp_error, reproducibility_error)
            if note
        ),
    )


def _canonical_energy_effects(
    response: np.ndarray,
    *,
    response_samples: int,
    cross_validated: bool,
    eps: float,
) -> tuple[float, float]:
    values = np.asarray(response, dtype=float)[:, : int(response_samples)]
    n_trials, n_times = values.shape
    if n_trials < 2 or n_times < 2:
        return np.nan, np.nan
    reconstructed_energy = 0.0
    observed_energy = 0.0
    if cross_validated:
        for held_out in range(n_trials):
            training = np.delete(values, held_out, axis=0)
            _, singular_values, right_vectors = np.linalg.svd(
                training,
                full_matrices=False,
            )
            if not singular_values.size or singular_values[0] <= eps:
                # A held-out fold without a defined training direction does
                # not supply the canonical vector required by the estimand.
                return np.nan, np.nan
            canonical = right_vectors[0]
            if float(np.dot(canonical, np.mean(training, axis=0))) < 0:
                canonical = -canonical
            trial = values[held_out]
            coefficient = float(np.dot(trial, canonical))
            reconstructed_energy += coefficient * coefficient
            observed_energy += float(np.dot(trial, trial))
    else:
        _, singular_values, right_vectors = np.linalg.svd(
            values,
            full_matrices=False,
        )
        if not singular_values.size or singular_values[0] <= eps:
            return np.nan, np.nan
        coefficients = values @ right_vectors[0]
        reconstructed_energy = float(np.sum(coefficients * coefficients))
        observed_energy = float(np.sum(values * values))
    canonical_energy = reconstructed_energy / float(n_trials * n_times)
    canonical_fraction = reconstructed_energy / max(observed_energy, eps)
    return float(canonical_energy), float(canonical_fraction)


def _row_rms(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(array * array, axis=1))


def _finite_median(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else np.nan


def _classification(crp_pass: bool, energy_pass: bool) -> str:
    if crp_pass and energy_pass:
        return "reproducible_energetic"
    if crp_pass:
        return "reproducible_low_energy"
    if energy_pass:
        return "energetic_inconsistent"
    return "no_response"


def _parameter_json(config: CRPEnergyConfig) -> str:
    parameters = asdict(config)
    return json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_scalar,
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported parameter value {type(value).__name__}")


def _empty_result(
    *,
    channel: str,
    n_trials_total: int,
    response_window: tuple[float, float],
    baseline_window: tuple[float, float],
    parameters: str,
    notes: str,
    n_trials_clean: int = 0,
    clean_trial_indices: np.ndarray | None = None,
    n_response_samples: int = 0,
    n_baseline_samples: int = 0,
) -> CRPEnergyResult:
    return CRPEnergyResult(
        channel=str(channel),
        significant=False,
        classification="insufficient_data",
        p_crp=np.nan,
        p_energy=np.nan,
        p_joint=np.nan,
        q_joint=np.nan,
        crp_significant=False,
        energy_significant=False,
        crp_statistic=np.nan,
        energy_statistic=np.nan,
        rms_response=np.nan,
        rms_baseline=np.nan,
        rms_ratio_db=np.nan,
        canonical_energy=np.nan,
        canonical_energy_fraction=np.nan,
        response_duration=np.nan,
        n_trials_total=int(n_trials_total),
        n_trials_clean=int(n_trials_clean),
        canonical_waveform=np.array([], dtype=float),
        canonical_waveform_times=np.array([], dtype=float),
        trial_coefficients=np.array([], dtype=float),
        clean_trial_indices=(
            np.asarray(clean_trial_indices, dtype=int)
            if clean_trial_indices is not None
            else np.array([], dtype=int)
        ),
        crp_explained_variance=np.nan,
        crp_snr=np.nan,
        reproducibility_test_exact=False,
        reproducibility_n_randomizations=0,
        energy_test_exact=False,
        energy_n_permutations=0,
        response_window=response_window,
        baseline_window=baseline_window,
        qc_status="insufficient_data",
        n_response_samples=int(n_response_samples),
        n_baseline_samples=int(n_baseline_samples),
        parameters=parameters,
        notes=str(notes),
    )
