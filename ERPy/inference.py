"""Small-sample inference helpers for nested electrophysiology data."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignFlipResult:
    """Result of an exact or Monte Carlo one-sample sign-flip test."""

    statistic: float
    p_value: float
    alternative: str
    n_observations: int
    n_permutations: int
    exact: bool


@dataclass(frozen=True)
class BootstrapInterval:
    """Patient-balanced hierarchical bootstrap estimate and interval."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    n_observations: int
    n_top_level_clusters: int


def fdr_bh(
    p_values: Iterable[float],
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust p-values with the Benjamini-Hochberg FDR procedure.

    Nonfinite inputs remain nonfinite in the adjusted array and are never
    rejected. The returned arrays preserve the input order.
    """

    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    rejected = np.zeros(values.shape, dtype=bool)
    finite_index = np.flatnonzero(np.isfinite(values))
    if not len(finite_index):
        return adjusted, rejected

    finite_values = np.clip(values[finite_index], 0.0, 1.0)
    order = np.argsort(finite_values, kind="stable")
    ranked = finite_values[order]
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    ranked_adjusted = ranked * float(len(ranked)) / ranks
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)

    original_order_adjusted = np.empty_like(ranked_adjusted)
    original_order_adjusted[order] = ranked_adjusted
    adjusted[finite_index] = original_order_adjusted
    rejected[finite_index] = original_order_adjusted <= float(alpha)
    return adjusted, rejected


def exact_sign_flip_test(
    values: Iterable[float],
    *,
    null_value: float = 0.0,
    alternative: str = "two-sided",
    statistic: str | Callable[[np.ndarray], float] = "mean",
    max_exact_observations: int = 20,
    n_resamples: int = 100_000,
    random_state: int | np.random.Generator | None = 0,
) -> SignFlipResult:
    """Test a paired or one-sample contrast by flipping observation signs.

    The test is exact when the number of finite observations does not exceed
    ``max_exact_observations``. For larger samples, deterministic Monte Carlo
    resampling is used by default. Monte Carlo signs are sampled uniformly
    with replacement; the observed assignment is additionally counted with
    the plus-one correction, so a sampled p-value cannot be zero.
    """

    alternative = str(alternative).lower()
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    stat_func = _statistic_function(statistic)
    centered = np.asarray(list(values), dtype=float) - float(null_value)
    centered = centered[np.isfinite(centered)]
    if not len(centered):
        return SignFlipResult(
            statistic=np.nan,
            p_value=np.nan,
            alternative=alternative,
            n_observations=0,
            n_permutations=0,
            exact=True,
        )

    observed = float(stat_func(centered))
    exact = len(centered) <= int(max_exact_observations)
    if exact:
        signs = np.asarray(
            list(product((-1.0, 1.0), repeat=len(centered))),
            dtype=float,
        )
    else:
        rng = _as_generator(random_state)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(max(int(n_resamples), 1), len(centered)),
            replace=True,
        )
    null_statistics = np.apply_along_axis(
        stat_func,
        1,
        signs * centered[None, :],
    )
    if alternative == "two-sided":
        extreme = np.abs(null_statistics) >= abs(observed) - 1e-15
    elif alternative == "greater":
        extreme = null_statistics >= observed - 1e-15
    else:
        extreme = null_statistics <= observed + 1e-15
    if exact:
        p_value = float(np.mean(extreme))
    else:
        p_value = float((1 + np.count_nonzero(extreme)) / (1 + len(extreme)))
    return SignFlipResult(
        statistic=observed,
        p_value=p_value,
        alternative=alternative,
        n_observations=int(len(centered)),
        n_permutations=int(len(null_statistics)),
        exact=bool(exact),
    )


def hierarchical_bootstrap_interval(
    data: pd.DataFrame,
    *,
    value_col: str,
    levels: Sequence[str],
    statistic: str | Callable[[np.ndarray], float] = "median",
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    random_state: int | np.random.Generator | None = 0,
) -> BootstrapInterval:
    """Estimate an equal-top-cluster interval for nested observations.

    The first level is treated as the biological replicate. Each bootstrap
    draw resamples those clusters, then recursively resamples lower-level
    clusters and observations. A statistic is computed within every selected
    top-level cluster and then across those cluster statistics, preventing
    patients with more acquisitions or contacts from dominating the estimate.
    """

    frame = pd.DataFrame(data).copy()
    if value_col not in frame.columns:
        raise ValueError(f"data is missing value column {value_col!r}")
    levels = [str(level) for level in levels]
    if not levels:
        raise ValueError("levels must include at least one cluster column")
    missing = [level for level in levels if level not in frame.columns]
    if missing:
        raise ValueError(f"data is missing hierarchy columns: {missing}")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be at least 1")

    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame[np.isfinite(frame[value_col])].copy()
    top_values = list(frame[levels[0]].drop_duplicates())
    if frame.empty or not top_values:
        return BootstrapInterval(
            estimate=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            confidence=float(confidence),
            n_resamples=int(n_resamples),
            n_observations=0,
            n_top_level_clusters=0,
        )

    stat_func = _statistic_function(statistic)
    observed_top = [
        float(stat_func(group[value_col].to_numpy(dtype=float)))
        for _, group in frame.groupby(levels[0], dropna=False, sort=False)
    ]
    estimate = float(stat_func(np.asarray(observed_top, dtype=float)))
    rng = _as_generator(random_state)
    grouped = {
        key: group.copy()
        for key, group in frame.groupby(levels[0], dropna=False, sort=False)
    }
    top_keys = list(grouped)
    can_vectorize = (
        isinstance(statistic, str)
        and str(statistic).lower() in {"mean", "median"}
        and len(levels) == 2
        and not frame.duplicated(levels).any()
    )
    if can_vectorize:
        draws = _vectorized_two_level_draws(
            grouped,
            value_col=value_col,
            lower_level=levels[1],
            statistic=str(statistic).lower(),
            n_resamples=int(n_resamples),
            rng=rng,
        )
    else:
        draws = np.empty(int(n_resamples), dtype=float)
        for draw_index in range(int(n_resamples)):
            selected = rng.integers(
                0,
                len(top_keys),
                size=len(top_keys),
            )
            top_statistics = []
            for selected_index in selected:
                group = grouped[top_keys[int(selected_index)]]
                sampled_values = _resample_nested_values(
                    group,
                    value_col=value_col,
                    levels=levels[1:],
                    rng=rng,
                )
                top_statistics.append(float(stat_func(sampled_values)))
            draws[draw_index] = float(
                stat_func(np.asarray(top_statistics, dtype=float))
            )

    tail = (1.0 - float(confidence)) / 2.0
    ci_low, ci_high = np.nanquantile(draws, [tail, 1.0 - tail])
    return BootstrapInterval(
        estimate=estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=float(confidence),
        n_resamples=int(n_resamples),
        n_observations=int(len(frame)),
        n_top_level_clusters=int(len(top_keys)),
    )


def _vectorized_two_level_draws(
    grouped: dict[object, pd.DataFrame],
    *,
    value_col: str,
    lower_level: str,
    statistic: str,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorize the common patient-by-acquisition bootstrap structure."""

    groups = list(grouped.values())
    n_top = len(groups)
    selected_top = rng.integers(
        0,
        n_top,
        size=(int(n_resamples), n_top),
    )
    top_statistics = np.empty(
        (int(n_resamples), n_top),
        dtype=float,
    )
    reducer = np.nanmean if statistic == "mean" else np.nanmedian
    for group_index, group in enumerate(groups):
        ordered = group.sort_values(lower_level, kind="stable")
        values = ordered[value_col].to_numpy(dtype=float)
        selected_values = rng.integers(
            0,
            len(values),
            size=(int(n_resamples), n_top, len(values)),
        )
        candidate_statistics = reducer(
            values[selected_values],
            axis=2,
        )
        selected_mask = selected_top == int(group_index)
        top_statistics[selected_mask] = candidate_statistics[selected_mask]
    return reducer(top_statistics, axis=1)


def _resample_nested_values(
    frame: pd.DataFrame,
    *,
    value_col: str,
    levels: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    if not levels:
        values = frame[value_col].to_numpy(dtype=float)
        selected = rng.integers(0, len(values), size=len(values))
        return values[selected]

    grouped = [
        group
        for _, group in frame.groupby(levels[0], dropna=False, sort=False)
    ]
    selected = rng.integers(0, len(grouped), size=len(grouped))
    return np.concatenate(
        [
            _resample_nested_values(
                grouped[int(index)],
                value_col=value_col,
                levels=levels[1:],
                rng=rng,
            )
            for index in selected
        ]
    )


def _statistic_function(
    statistic: str | Callable[[np.ndarray], float],
) -> Callable[[np.ndarray], float]:
    if callable(statistic):
        return statistic
    name = str(statistic).lower()
    if name == "mean":
        return lambda values: float(np.nanmean(values))
    if name == "median":
        return lambda values: float(np.nanmedian(values))
    raise ValueError("statistic must be 'mean', 'median', or a callable")


def _as_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
