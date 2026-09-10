"""Exact empirical-bootstrap quantiles for the frozen multiscale confirmation.

This computes the declared stratified paired-family bootstrap distribution without
Monte Carlo error. It does not make its percentile interval an exact frequentist
confidence interval. No detector, model, family selection or old result is changed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import reduce
import hashlib
import json
from math import gcd
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "validation/frozen_results/multiscale"
STRATA = ["n_trials", "morphology", "polarity", "snr"]
CANDIDATES = ("multiscale", "weighted_multiscale")
COMPARISONS = ("early", "broad", "matched_broad")
METHODS = (*COMPARISONS, "late", *CANDIDATES)


@dataclass(frozen=True)
class CountDistribution:
    """Integer coefficients for P(S=offset+j)=coefficients[j]/denominator."""

    offset: int
    coefficients: tuple[int, ...]
    denominator: int
    total_samples: int

    def cdf(self, count: int) -> Fraction:
        stop = min(max(count - self.offset + 1, 0), len(self.coefficients))
        return Fraction(sum(self.coefficients[:stop]), self.denominator)

    def quantile(self, probability: Fraction) -> int:
        """Infimum of integer s such that F(s)>=probability (inverse CDF)."""
        if not isinstance(probability, Fraction) or not 0 < probability <= 1:
            raise ValueError("probability must be a Fraction in (0, 1]")
        cumulative = 0
        for j, coefficient in enumerate(self.coefficients):
            cumulative += coefficient
            if cumulative * probability.denominator >= self.denominator * probability.numerator:
                return self.offset + j
        raise AssertionError("distribution is not normalized")


def exact_stratified_distribution(strata: Sequence[Sequence[int]]) -> CountDistribution:
    """Convolve exact categorical laws, drawing each stratum's original size.

    If stratum h has m_h paired deltas in {-1,0,1}, its bootstrap count law is
    (a_h*z^-1+b_h+c_h*z)^m_h/m_h^m_h. Independent strata multiply these laws.
    Integer arithmetic is used for every probability and quantile comparison.
    """
    coefficients = [1]
    denominator, offset, total_samples = 1, 0, 0
    if len(strata) == 0:
        raise ValueError("at least one nonempty stratum is required")
    for values in strata:
        delta = np.asarray(values)
        if delta.ndim != 1 or not len(delta) or delta.dtype.kind not in "iu":
            raise ValueError("each stratum must be a nonempty one-dimensional integer vector")
        if not np.isin(delta, [-1, 0, 1]).all():
            raise ValueError("paired deltas must belong to {-1, 0, 1}")
        size = len(delta)
        total_samples += size
        counts = [int(np.count_nonzero(delta == k)) for k in (-1, 0, 1)]
        divisor = reduce(gcd, counts)
        counts = [count // divisor for count in counts]
        draw_denominator = size // divisor
        nonzero = [j for j, count in enumerate(counts) if count]
        first, last = nonzero[0], nonzero[-1]
        factor = counts[first:last + 1]
        offset += (first - 1) * size
        if len(factor) == 1:
            continue  # Deterministic draws; factor and denominator are both one.
        for _ in range(size):
            product = [0] * (len(coefficients) + len(factor) - 1)
            for j, value in enumerate(coefficients):
                if value:
                    for k, count in enumerate(factor):
                        if count:
                            product[j + k] += value * count
            coefficients = product
            denominator *= draw_denominator
    if sum(coefficients) != denominator:
        raise AssertionError("exact probability mass does not sum to one")
    return CountDistribution(offset, tuple(coefficients), denominator, total_samples)


def paired_target_calls(frame: pd.DataFrame, methods: Sequence[str] = METHODS) -> pd.DataFrame:
    """Reject duplicate, unpaired, unavailable or inconsistently labeled targets."""
    required = {"scenario", "family_id", *STRATA, "method", "channel", "call", "available", "is_injected_target"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    power = frame.loc[frame.scenario.eq("power")].copy()
    if power.empty or power[list(required)].isna().any().any():
        raise ValueError("power rows must be present with complete required fields")
    for name in ("call", "available", "is_injected_target"):
        if not power[name].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f"{name} must contain actual Boolean values")
    if not power.is_injected_target.eq(power.channel.eq("target")).all():
        raise ValueError("target channel and injected-target flag disagree")
    targets = power.loc[power.is_injected_target]
    if targets.duplicated(["family_id", "method"]).any():
        raise ValueError("duplicate target family/method records")
    if not targets.available.all():
        raise ValueError("this confirmation comparison requires available target calls")
    if (targets.groupby("family_id")[STRATA].nunique(dropna=False) != 1).any().any():
        raise ValueError("a family's stratum metadata differ across methods")
    wanted = set(methods)
    if len(wanted) != len(methods) or len(wanted) < 2:
        raise ValueError("methods must name at least two distinct methods")
    if not targets.groupby("family_id").method.agg(set).map(lambda x: x == wanted).all():
        raise ValueError("each family must have exactly the requested paired methods")
    n = pd.to_numeric(targets.n_trials, errors="coerce")
    if not (np.isfinite(n) & (n > 0) & (n == np.floor(n))).all():
        raise ValueError("trial counts must be positive integers")
    return targets.pivot(index=["family_id", *STRATA], columns="method", values="call").astype(int)


def confirmation_intervals(frame: pd.DataFrame, saved: pd.DataFrame) -> pd.DataFrame:
    """All 24 declared comparisons; also reproduce their original 2,000 draws."""
    wide_all = paired_target_calls(frame)
    sizes = wide_all.groupby(level=STRATA).size()
    expected = pd.MultiIndex.from_product([[8, 12, 24], ["early", "narrow", "late", "biphasic"], [-1, 1], [.5, 1., 2.]], names=STRATA)
    if len(wide_all) != 1440 or not sizes.index.sort_values().equals(expected.sort_values()) or not sizes.eq(20).all():
        raise ValueError("confirmation must contain the complete 72-cell grid with 20 target families per cell")
    saved = saved.copy()
    saved["n_trials"] = saved.n_trials.astype(str)
    key = ["candidate", "n_trials", "comparison"]
    if len(saved) != 24 or saved.duplicated(key).any():
        raise ValueError("saved comparison table must contain 24 unique records")
    saved = saved.set_index(key)
    rng = np.random.default_rng(20260909)
    rows = []
    for candidate in CANDIDATES:
        for n in ("all", 8, 12, 24):
            wide = wide_all if n == "all" else wide_all.loc[wide_all.index.get_level_values("n_trials") == n]
            groups = wide.groupby(level=STRATA).indices
            for comparison in COMPARISONS:
                delta = wide[candidate] - wide[comparison]
                strata = [delta.iloc[indices].to_numpy() for indices in groups.values()]
                distribution = exact_stratified_distribution(strata)
                lo, hi = [distribution.quantile(q) for q in (Fraction(1, 40), Fraction(39, 40))]
                monte_carlo = np.zeros(2000)
                for values in strata:
                    monte_carlo += values[rng.integers(0, len(values), size=(2000, len(values)))].sum(axis=1)
                monte_carlo /= len(wide)
                mc_lo, mc_hi = np.quantile(monte_carlo, [.025, .975])
                old = saved.loc[(candidate, str(n), comparison)]
                observed = float(delta.mean())
                if not np.allclose([old.ci_low, old.ci_high, old.paired_difference], [mc_lo, mc_hi, observed], atol=2e-15, rtol=0):
                    raise ValueError("saved summary does not match the declared 2,000-draw implementation")
                nonnegative = 1 - distribution.cdf(-1)
                rows.append(dict(candidate=candidate, n_trials=str(n), comparison=comparison,
                    families=len(wide), strata=len(strata), families_per_stratum=20,
                    candidate_recovery=float(wide[candidate].mean()), reference_recovery=float(wide[comparison].mean()),
                    candidate_only=int((delta == 1).sum()), reference_only=int((delta == -1).sum()),
                    paired_difference=observed, exact_ci_low_count=lo, exact_ci_high_count=hi,
                    exact_ci_low=lo / len(wide), exact_ci_high=hi / len(wide),
                    exact_ci_low_pp=100 * lo / len(wide), exact_ci_high_pp=100 * hi / len(wide),
                    cdf_before_lower=float(distribution.cdf(lo - 1)), cdf_at_lower=float(distribution.cdf(lo)),
                    cdf_before_upper=float(distribution.cdf(hi - 1)), cdf_at_upper=float(distribution.cdf(hi)),
                    probability_bootstrap_difference_nonnegative=float(nonnegative),
                    saved_2000_ci_low=float(old.ci_low), saved_2000_ci_high=float(old.ci_high),
                    saved_2000_ci_low_pp=100 * float(old.ci_low), saved_2000_ci_high_pp=100 * float(old.ci_high),
                    saved_2000_reproduced=True,
                    probability_at_most_49_nonnegative_in_2000_draws=float(binom.cdf(49, 2000, float(nonnegative))),
                    probability_at_most_50_nonnegative_in_2000_draws=float(binom.cdf(50, 2000, float(nonnegative)))))
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text())
    for record in manifest["artifacts"]:
        artifact = path.parent / record["path"]
        if not artifact.is_file() or artifact.stat().st_size != record["bytes"] or sha256(artifact) != record["sha256"]:
            raise ValueError(f"artifact verification failed: {record['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/multiscale_interval_precision")
    parser.add_argument("--verify", action="store_true", help="verify the existing evidence manifest without recomputing")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify:
        verify_manifest(output / "manifest.json")
        print("Exact-bootstrap evidence hashes and sizes verified.")
        return
    inputs = [FROZEN / "weighted_confirmation/method_results.csv.gz",
              FROZEN / "weighted_confirmation/paired_recovery.csv",
              FROZEN / "weighted_confirmation/analysis_plan.json",
              FROZEN / "weighted_execution_code/validation/benchmark_multiscale.py"]
    results = confirmation_intervals(pd.read_csv(inputs[0]), pd.read_csv(inputs[1]))
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "exact_confirmation_intervals.csv"
    results.to_csv(result_path, index=False)
    artifacts = []
    for path in [*inputs, Path(__file__).resolve(), ROOT / "tests/test_exact_multiscale_bootstrap.py", ROOT / "validation/multiscale_interval_precision/README.md", result_path]:
        if not path.is_file():
            raise ValueError(f"required provenance artifact is missing: {path.name}")
        artifacts.append(dict(path=Path(os.path.relpath(path, output)).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Precision supplement to the unchanged frozen 2,000-resample confirmation summaries",
        bootstrap_law="Independent empirical resampling of 20 paired target-family deltas within each n_trials/morphology/polarity/snr cell; retain fixed cell sizes and pool by family count",
        arithmetic="Exact integer polynomial convolution; rational inverse-CDF comparisons",
        quantile_definition="Q(p)=min{s in integer support: F(s)>=p}; p=1/40 and 39/40; divide count quantiles by the number of families",
        interpretation="Exact distribution of the specified empirical bootstrap conditional on the observed fixed grid; not exact frequentist or clinical-population coverage",
        compared_rows=24, all_saved_2000_intervals_reproduced=True,
        frozen_inputs_modified=False, artifacts=artifacts)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_manifest(output / "manifest.json")
    selected = results[(results.candidate == "weighted_multiscale") & (results.comparison == "matched_broad")]
    print(selected[["n_trials", "paired_difference", "exact_ci_low_pp", "exact_ci_high_pp", "saved_2000_ci_low_pp", "saved_2000_ci_high_pp"]].to_string(index=False))
    print("All 24 frozen 2,000-resample intervals reproduced; new exact-distribution results saved and verified.")


if __name__ == "__main__":
    main()
