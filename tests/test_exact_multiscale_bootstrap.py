"""Independent tiny-sample exhaustive checks of the empirical-bootstrap law."""
from collections import Counter
from fractions import Fraction
from itertools import product

import pandas as pd
import pytest

from validation.exact_multiscale_bootstrap import exact_stratified_distribution, paired_target_calls


def brute_force(strata):
    possibilities = [list(product(values, repeat=len(values))) for values in strata]
    counts = Counter(sum(sum(sample) for sample in draw) for draw in product(*possibilities))
    denominator = sum(counts.values())
    return {k: Fraction(v, denominator) for k, v in counts.items()}


@pytest.mark.parametrize("strata", [[[-1, 0], [0, 1]], [[-1, -1, 1], [0]], [[1, 1], [-1], [0, 0]]])
def test_exact_law_matches_all_bootstrap_index_assignments(strata):
    expected = brute_force(strata)
    actual = exact_stratified_distribution(strata)
    assert sum(actual.coefficients) == actual.denominator
    for count in range(min(expected) - 1, max(expected) + 2):
        assert actual.cdf(count) == sum(value for k, value in expected.items() if k <= count)
    for probability in (Fraction(1, 40), Fraction(1, 2), Fraction(39, 40)):
        q = actual.quantile(probability)
        assert actual.cdf(q - 1) < probability <= actual.cdf(q)


def test_fixed_strata_are_not_pooled_resampling():
    law = exact_stratified_distribution([[-1, -1], [1, 1]])
    assert law.quantile(Fraction(1, 40)) == law.quantile(Fraction(39, 40)) == 0
    assert law.total_samples == 4


@pytest.mark.parametrize("strata", [[], [[]], [[-2, 0]], [[0.5, 1]], [[[0, 1]]], [[True, False]]])
def test_rejects_invalid_delta_contract(strata):
    with pytest.raises(ValueError):
        exact_stratified_distribution(strata)


def paired_frame():
    return pd.DataFrame([dict(scenario="power", family_id="family_a", n_trials=8, morphology="early", polarity=-1, snr=.5,
        method=method, channel="target", call=call, available=True, is_injected_target=True)
        for method, call in [("candidate", True), ("reference", False)]])


def test_pairing_retains_signed_disagreement():
    wide = paired_target_calls(paired_frame(), ["candidate", "reference"])
    assert (wide.candidate - wide.reference).tolist() == [1]


@pytest.mark.parametrize("corruption", ["duplicate", "missing_method", "stratum_mismatch", "unavailable", "non_boolean", "wrong_target", "missing_field"])
def test_rejects_unpaired_or_ambiguous_inputs(corruption):
    frame = paired_frame()
    if corruption == "duplicate": frame = pd.concat([frame, frame.iloc[[0]]])
    elif corruption == "missing_method": frame = frame.iloc[[0]]
    elif corruption == "stratum_mismatch": frame.loc[1, "n_trials"] = 12
    elif corruption == "unavailable": frame.loc[1, "available"] = False
    elif corruption == "non_boolean": frame["call"] = ["True", "False"]
    elif corruption == "wrong_target": frame.loc[1, "is_injected_target"] = False
    elif corruption == "missing_field": frame = frame.drop(columns="snr")
    with pytest.raises(ValueError):
        paired_target_calls(frame, ["candidate", "reference"])
