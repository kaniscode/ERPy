import numpy as np
import pandas as pd

from ERPy.inference import (
    exact_sign_flip_test,
    fdr_bh,
    hierarchical_bootstrap_interval,
)


def test_fdr_bh_preserves_order_and_nonfinite_values():
    adjusted, rejected = fdr_bh([0.01, np.nan, 0.04, 0.20])

    np.testing.assert_allclose(
        adjusted[[0, 2, 3]],
        [0.03, 0.06, 0.20],
    )
    assert np.isnan(adjusted[1])
    assert rejected.tolist() == [True, False, False, False]


def test_exact_sign_flip_reports_small_sample_resolution():
    result = exact_sign_flip_test([1.0, 2.0, 3.0, 4.0])

    assert result.exact
    assert result.n_observations == 4
    assert result.n_permutations == 16
    assert result.p_value == 0.125


def test_hierarchical_bootstrap_balances_top_level_clusters():
    frame = pd.DataFrame(
        {
            "patient": ["P1"] * 100 + ["P2"] * 2,
            "session": ["A"] * 100 + ["B"] * 2,
            "acquisition": list(range(100)) + [0, 1],
            "value": [0.0] * 100 + [10.0, 10.0],
        }
    )
    result = hierarchical_bootstrap_interval(
        frame,
        value_col="value",
        levels=["patient", "session", "acquisition"],
        statistic="mean",
        n_resamples=200,
        random_state=7,
    )

    assert result.estimate == 5.0
    assert result.n_top_level_clusters == 2
    assert result.ci_low <= result.estimate <= result.ci_high


def test_two_level_hierarchical_bootstrap_is_deterministic():
    frame = pd.DataFrame(
        {
            "patient": ["P1"] * 4 + ["P2"] * 3 + ["P3"] * 2,
            "acquisition": list(range(4)) + list(range(3)) + list(range(2)),
            "value": [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 4.0, 6.0],
        }
    )
    first = hierarchical_bootstrap_interval(
        frame,
        value_col="value",
        levels=["patient", "acquisition"],
        statistic="median",
        n_resamples=500,
        random_state=29,
    )
    second = hierarchical_bootstrap_interval(
        frame,
        value_col="value",
        levels=["patient", "acquisition"],
        statistic="median",
        n_resamples=500,
        random_state=29,
    )

    assert first == second
    assert first.estimate == 5.0
    assert first.ci_low <= first.estimate <= first.ci_high
