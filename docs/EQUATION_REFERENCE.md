# Equation and implementation reference

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

This reference maps the numerical definitions to the implementation. It does
not certify clinical accuracy, preprocessing/QC selection validity or the
dependence assumptions required by multiple-testing procedures.

The [workflow follow-up](WORKFLOW_CONCORDANCE.md) documents the subsequent
full replay of both saved primary benchmarks and two availability/QC fixes.

## Equation-to-function mapping

| Numerical definition | Implementation | Audit result |
| --- | --- | --- |
| trial baseline centering | `crp.run_crp_array`; `crp_energy.run_crp_energy_array` | CRP uses its configured baseline. Primary inference recenters on the latest sample-matched baseline, after selecting trials finite over both matched windows. |
| Declared effective response window | `crp_energy.run_crp_energy_array` | Epoch endpoints must cover the requested interval within half a sample. Truncated inputs receive unavailable p-values and no randomizations; complete-input numerical tests are unchanged. See `validation/RESPONSE_WINDOW_SUPPORT.md`. |
| SEM | `crp._sem` | Sample standard deviation (`ddof=1`) divided by square root of finite trial count. |
| RMS | `crp_energy._row_rms` | Square root of mean squared sample voltage. |
| ordered projections and mean | `crp_energy._ordered_projection_matrix`; `fixed_window_reproducibility_sign_flip_test` | Trial-by-time input; rows normalized by their L2 norms; divide by `sqrt(sfreq)`; zero diagonal; sum divided by `n(n-1)`. Rejects response norms at or below `eps`. |
| projection sign randomization | `fixed_window_reproducibility_sign_flip_test`; `_unique_quadratic_sign_patterns` | Exact default through 12 trials: first sign fixed positive, `2**(n-1)-1` non-observed patterns plus observed identity. Monte Carlo: at least 5,000 uniform sign draws with replacement, first sign fixed positive, plus-one correction. |
| descriptive duration grid and maximizing mean projection | `crp._projection_sample_counts`; `_mean_projection_profile` | Python nearest-even rounding, minimum two samples, positive step, full endpoint appended. Profile norm-floor inconsistency corrected below. |
| canonical vector, coefficients, SNR, explained fraction | `crp.run_crp_array` | First left singular vector of time-by-trial response, sign aligned with mean; `alpha_prime = alpha/sqrt(q)`; residual subtraction; `SNR = alpha/sqrt(max(sum(residual**2), eps))`; explained fraction uses squared energies.  |
| matched energy and dB | `crp_energy.run_crp_energy_array`; `paired_log_rms_sign_flip_test` | Both windows separately demeaned; log RMS differences use additive `eps`; mean difference transformed by `20/log(10)`. Output RMS summaries are geometric means of stabilized trial RMS. Exact default through 16 trials has denominator `2**n`; Monte Carlo uses at least 5,000 draws and plus-one. |
| intersection–union p-value | `crp_energy.run_crp_energy_array` | Maximum of finite component p-values. Descriptive model failure no longer discards a valid component or joint p-value. A missing component remains insufficient data. |
| BH | `inference.fdr_bh` | Stable finite-value sort; multiply by family size/rank; reverse cumulative minimum; clamp at 1; restore input order; reject at `q <= alpha`. Nonfinite values excluded. No algorithm change. |
| seeds | `erp_detection._stable_channel_random_state`; `crp_energy.run_crp_energy_array` | High-level wrapper derives channel seed from root and exact label using SHA-256; core spawns separate NumPy `SeedSequence` children for components. Direct array calls take the supplied seed; their channel argument is only a result label. Existing order-invariance tests pass. |
| conditional leave-one-trial-out canonical energy | `crp_energy._canonical_energy_effects` | Held-out projection squares divided by `n*q`, and by total observed energy for the fraction. Undefined training directions now return missing effects. Full-data duration selection remains descriptive. |

The primary interface requires finite, strictly increasing, uniformly spaced
times and infers sampling frequency from their median increment. Windows are
in seconds, endpoint comparisons are inclusive, and realized first/last sample
times are stored. The response start is the later of the declared start and
artifact-interval stop; the baseline must have enough samples to match it.

Adding any trial-specific constant to every sample cancels in the primary
matched-baseline recentering and in both separately demeaned energy segments.
Consequently, an additive post-artifact anchor cannot directly alter these
test inputs in exact arithmetic. It may still affect upstream response-based
QC and the displayed waveform, which are outside this core audit.

## Confirmed corrections and numerical consequences

1. **General sign-flip Monte Carlo p-values.**
   `inference.exact_sign_flip_test` used `b/M` for sampled nulls. It now uses
   `(b+1)/(M+1)`, counting the observed assignment explicitly. With 21 positive
   observations and nine sampled signs, the reproduced case changed from 0
   to 0.1. Exact enumeration is unchanged. This helper is separate from the
   primary CRP-energy helpers, which already had the correction.
2. **Descriptive projection normalization.**
   `_mean_projection_profile` previously used
   `sqrt(max(sum(x**2), eps))`; the direct ordered-projection helper uses
   `max(sqrt(sum(x**2)), eps)`. The profile now uses the same norm floor as
   the direct helper. This can change profile values and selected durations
   when a prefix norm lies between `eps` and `sqrt(eps)`. No SNR radical was
   changed: its denominator is a residual-energy floor under a square root
  .
3. **Independence of inference from descriptive availability.**
   An eight-trial balanced-polarity fixture with a zero early prefix had a
   valid full-window projection p-value of 1. Its selected-duration canonical
   fit was degenerate; the old joint wrapper discarded the projection and
   joint p-values as missing. The revised wrapper retains both as 1, leaves
   descriptive quantities missing with an explanatory note, and keeps the
   contact eligible for BH. This can increase family size and alter other
   contacts' adjusted values when such failures occur. Missing primary
   components now receive `insufficient_data`, instead of implying that a
   valid component threshold was simply not crossed.
4. **Undefined leave-one-out directions.**
   A fold whose training set has no nondegenerate singular direction used
   to be skipped; its held-out energy was also omitted from the denominator.
   The helper now returns missing canonical energy and fraction when such a
   required direction is undefined. A reproduced one-nonzero-trial case
   changed from `(0, 0)` to `(NaN, NaN)`. It does not create a new primary
   inference rule.

These changes do not justify editing previously reported clinical or
simulation numbers without re-running the associated inputs, seeds, and
family rules. Original numerical runs retain their recorded provenance.

## Numerical conventions and remaining boundaries

- Exact projection identity is explicitly counted. Comparisons use a
  tolerance of `64 * machine_epsilon * comparison_scale`, where the scale is
  the largest absolute observed or generated statistic (with a tiny floor).
  Energy and the generic sign-flip helper use absolute tolerance `1e-15`.
  These implement inclusion of numerical ties; they are not effect-size
  thresholds.
- `eps` is a numerical floor whose physical units depend on its use: voltage
  for RMS addition/norm thresholds; squared voltage for energy denominators.
  Scale invariance holds away from those floors and numerical ties. Primary
  projection statistics scale linearly with voltage and inversely with
  `sqrt(sfreq)` while their p-values remain unchanged by either positive
  common scaling factor.
- The BH helper clips finite out-of-range inputs into `[0, 1]`; the valid
  component p-values audited here are already in that range. This audit did
  not redesign that existing input policy or the multiple-testing family.
- Sign-flip validity depends on group invariance after selection; low p-values
  do not validate that assumption. BH's inferential guarantee separately
  depends on appropriate p-value dependence. A QC-filtered rejection set does
  not automatically inherit an unfiltered FDR guarantee.

## Verification

The subsequent N1 timing-report extension retains the original feature vector,
midpoint imputation and saved models. Current exports distinguish measured
latency from imputation. Its [current-source replay](../validation/n1_timing_reporting/README.md)
checks all 64,096 held-out model predictions against saved results, with no
changed calls. The [multiscale precision supplement](../validation/multiscale_interval_precision/README.md)
computes exact quantiles of the declared empirical bootstrap while preserving
the old simulation tables and 2,000-resample summaries. These are reporting
and numerical-precision corrections, not new physiological validation.

Command run from the public audit checkout, using the available Anaconda
Python 3.12.5 environment:

```text
python -m pytest \
  tests/test_equation_contract.py \
  tests/test_inference.py \
  tests/test_published_detectors.py \
  tests/test_synthetic_benchmark.py \
  tests/test_crp_energy_validation.py
```

The initial focused run passed **61 tests**. New equation-level regressions
cover Monte Carlo identity and ties, direct/profile norm agreement, descriptive
failure without lost inference, missing component classification, undefined
and defined canonical-energy folds, correct SNR/coefficient radicals, exact
energy enumeration, primary Monte Carlo resolution, projection unit scaling,
and sample-grid rounding. Existing tests additionally check the quadratic
projection test against complete enumeration, independence from duration-grid
choice, separate energy demeaning, channel seed stability, and the benchmark
bookkeeping. These checks are software verification, not new performance
experiments.
