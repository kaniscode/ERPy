# Multiscale confirmation: bootstrap numerical precision

This supplements the unchanged frozen confirmation outputs. It computes the exact distribution of the **same empirical stratified paired-family bootstrap** used for all 24 confirmation comparisons. No new families, detector outputs, windows, weights, outcomes or selection rules are introduced. The original 2,000-resample summaries and their code remain in `../frozen_results/multiscale/weighted_confirmation/` and `../frozen_results/multiscale/weighted_execution_code/`.

Within each trial-count × morphology × polarity × SNR cell, 20 paired target-call differences belong to {-1, 0, 1}. Twenty draws with replacement from that empirical cell give its count distribution. Independent cell distributions are convolved using integer polynomial coefficients. The pooled result divides the total count by the number of families (1,440 overall; 480 at one trial count). The 95% percentile endpoints are the inverse-CDF quantiles `Q(p) = min{s: F(s) >= p}` at exactly `p = 1/40` and `39/40`. Integer/rational comparisons avoid Monte Carlo and floating-point quantile error.

“Exact” describes calculation of this conditional empirical-bootstrap distribution. It does **not** promise exact frequentist coverage, population representativeness, or clinical validity. Strata and empirical probabilities remain estimated from the fixed simulation grid.

## Reporting correction

For weighted minus matched broad recovery, the pooled point difference remains −15/1,440 = **−1.0417 percentage points** (74.5139% versus 75.5556% recovery). The exact empirical-bootstrap percentile interval is **[−2.0833, 0.0000] percentage points**. The old 2,000-resample interval [−2.0833, −0.0694] is a reproducible finite-resample approximation, not a changed bootstrap design or detector implementation error. The exact distribution puts 2.5368796% probability on a nonnegative bootstrap difference, so the old upper endpoint's exclusion of zero is not stable to bootstrap numerical precision.

| Trial count | Exact percentile interval (percentage points) | Original 2,000-resample interval |
|---|---:|---:|
| All | −2.0833 to 0.0000 | −2.0833 to −0.0694 |
| 8 | −6.2500 to −2.7083 | −6.0469 to −2.7083 |
| 12 | −1.8750 to 1.8750 | −1.8750 to 1.8750 |
| 24 | −0.4167 to 2.9167 | −0.6250 to 3.1250 |

The full CSV gives all 24 comparisons, integer endpoints, CDF values around each endpoint, and both old and exact intervals. The program also reproduces **every** original interval using the original seed and loop order. The overall decision is unchanged: no general recovery improvement was established and the original single-window detector remains the default. Avoid interpreting the old slightly negative pooled upper endpoint as stable evidence of a strictly negative effect. These exact intervals support reporting precision; they do not change the original experimental plan.

The two final CSV columns give binomial probabilities of at most 49 or at most 50 nonnegative differences among 2,000 bootstrap draws. These bound the probability of a strictly negative upper endpoint under NumPy's original linear percentile interpolation: its 97.5th percentile interpolates zero-based order statistics 1,949 and 1,950. With 50 nonnegative draws, the sign also depends on the adjacent values, so neither diagnostic alone is labeled as that exact probability.

## Reproduction and verification

From the repository root, using the scientific validation environment (validated with Python 3.12.5):

```bash
python validation/exact_multiscale_bootstrap.py
python validation/exact_multiscale_bootstrap.py --verify
python -m pytest tests/test_exact_multiscale_bootstrap.py
```

The default run reads only the frozen source tables and writes this separate precision directory. Its manifest records SHA-256 and size for the exact utility, its tests, the original bootstrap source snapshot, the frozen method table, the old paired summary, the frozen plan, this README and the new CSV. It contains relative paths and requires no private recordings. The source table holds generated-data decisions only.
