# Optional tests across several response windows

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

The optional method tests several predefined response intervals and adjusts for
that search. It is called **multiscale** in the API. This experiment did **not**
establish a general recovery gain.
ERPy retains its existing single-window default. The equal-weight candidate,
one declared change to the window weights, all development results and the fresh
confirmation remain available in the [saved evidence](../validation/frozen_results/multiscale/README.md).
These are internal simulation and semisynthetic experiments, not clinical
validation or formal preregistration.

## How the multiple-window test works

The declared windows are 10–90, 90–300 and 10–300 ms. Within each window the
existing projection–energy conjunction is retained:

$$p_w=\max(p_{R,w},p_{E,w}).$$

The equal-weight window set reports `min(1, 3 * min(p_w))`. The weighted version uses
fixed positive weights (0.1, 0.1, 0.8) and reports

$$p_{\mathrm{bank}}=\min\{1,\min_w(p_w/a_w)\}.$$

The global null requires at least one component null in **every** declared
window. Bonferroni does not require cross-window independence, but it does
require valid window p values under the component assumptions. The windows
share a complete-case trial mask across their available support. Unavailable
windows contribute p=1; neither the number of windows nor their weights is
reduced after seeing availability. Selection must preserve the assumed null
invariance; these calculations do not certify response-derived QC selection.

BH is applied once across the declared non-stimulation contact family. It
retains its cross-contact independence/positive-dependence conditions; this
experiment is not a general FDR guarantee for arbitrary contact dependence.
An unavailable contact contributes p=1 with its availability flag retained.

At eight trials, the exact projection minimum is 1/128. For an **isolated**
discovery among four contacts, the equal-weight adjusted floor is 12/128 =
0.09375. Assigning 0.8 to the broad window permits a floor of
4/(128 × 0.8) = 0.0390625. These are isolated-discovery bounds: additional small
contact p values can change BH ranks, so they do not forbid every eight-trial
bank discovery. The weighting restores possible broad-window detection while
retaining a search penalty.

This combines ERPy's existing randomization/energy conjunction with standard
Bonferroni weighting. CRP supplies the projection representation
([Miller et al., 2023](https://doi.org/10.1371/journal.pcbi.1011105)); ER-detect also
offers fixed-window CRP detection and selectable polarity
([Van den Boom et al., 2025](https://doi.org/10.1016/j.jneumeth.2025.110389)).
Neither projection detection nor Bonferroni is claimed as new.

## Declared sequence and negative result

The initial 2,000 independent generated families include 1,440 signal families
across three trial counts, three signal levels, four shapes, two polarities and
20 replicates per cell; 240 stationary-noise families; and 320 component-null
stress families. Every method uses the same four-contact family and reused
window evidence. The equal-weight development bank recovered 0/480 targets at
eight trials, 365/480 at twelve trials and 421/480 at twenty-four trials.

The single weighted amendment was recorded **after** observing equal-weight
outcomes and **before** weighted outcomes. Development recombination uses
the same saved window p values. A separate 2,000-family confirmation uses a
fixed new namespace and disjoint seeds. No further window/weight search was
performed. The complete initial plan, feasibility amendment and weighted
amendment are preserved.

| Fresh confirmation | Matched broad recovery | Weighted-window recovery | Paired difference, percentage points (95% interval) |
| --- | ---: | ---: | ---: |
| 8 trials; 480 target families | 64.38% | 60.00% | −4.38 (−6.25, −2.71) |
| 12 trials; 480 target families | 74.58% | 74.58% | 0.00 (−1.88, 1.88) |
| 24 trials; 480 target families | 87.71% | 88.96% | 1.25 (−0.42, 2.92) |
| All 1,440 target families | 75.56% | 74.51% | −1.04 (−2.08, 0.00) |

Intervals are exact discrete quantiles of the empirical paired-family bootstrap
within the fixed simulation cells. They describe this design, not uncertainty in
a clinical population or exact frequentist coverage. The [interval table and
reproduction utility](../validation/multiscale_interval_precision/README.md)
also retain the original 2,000-draw approximations unchanged. The positive 24-trial development
difference did not have an interval excluding zero in confirmation and should
not be selected as a general improvement. The original broad 15–300 ms and
matched broad 10–300 ms references are separately retained in all tables.

The weighted confirmation had no family with a false call in 240 stationary
noise families or either set of 160 global component-null stress families.
These finite empirical checks do not prove calibration under arbitrary
physiology. The energy-null ramp stress uses exchangeable positive baseline/
response slopes; the projection-null converse uses independently signed
templates with centrally symmetric noise. Their nulls hold jointly across
the bank; no mixed-window alternative is mislabeled as globally null.

## Limits of the two recorded-baseline examples

The recorded-baseline experiment adds known simulated signals to measured
prestimulus noise (a semisynthetic design). It uses one annotation-blind, lexicographically
selected non-stimulation good ECoG contact per site, excluding MAYO01: MAYO02
and UMCU20. An infeasible ≥24-native-trial request was amended before outcomes
to ten native finite prestimulus trials per fixture. No trials were repeated
to claim larger native sample sizes. The 960 scenarios use n=8/10 selected
without replacement, disjoint prestimulus blocks resampled to 500 Hz, block
swaps and whole-trial sign randomization before injection.

The same two finite sources are reused across contacts and scenarios. These
are conditional semisynthetic characterizations, not 960 independent clinical
recordings or natural-noise false-positive estimates. Weighted recovery was
52.19% versus matched broad 53.75% at n=8, and 63.75% versus 67.19% at n=10.
All positive/negative morphology cells, null results, seeds, availability and
failures are retained. No production family failed or emitted a recorded
warning; all tested window/contact evaluations were available.

## API and reproducible execution

This multiple-window method is an explicit experimental API. Predeclare windows, weights, trial
mask and contact family before inspecting outcomes. For trial × time × contact
data, use the family operation so contact adjustment is included:

```python
from ERPy.multiscale import MultiScaleConfig, run_multiscale_family

config = MultiScaleConfig(weights=(0.1, 0.1, 0.8))
results = run_multiscale_family(
    trials_uv, times_seconds, channels=non_stimulation_contacts, config=config,
)
records = [result.to_record() for result in results]
```

`weights=None` selects the equal-weight experimental bank. The standalone
single-contact function's significance flag is unadjusted; use the family
function for family decisions. Keep unavailable processing separate from
observed physiological absence. No default facade route is replaced.

Install ERPy and run from the repository root. The small CC0 fixtures are
included, so this complete replay needs no network or governed recordings:

```bash
python -m pip install -e .
python validation/verify_multiscale_artifacts.py
python validation/replay_multiscale.py --output validation/work/multiscale_replay --workers 3
```

The replay stages exact historical source files in a new workspace, runs
equal-weight development and the two-fixture experiment, overlays the exact
weighted source, recombines development results, runs fresh confirmation, and
rebuilds the recorded arithmetic/report checks. The report's original focused
pytest count is explicitly labeled historical; the replay does not rerun
pytest. The preserved support scripts
only have a machine-specific import path removed. Public metadata edits and
unchanged fixture-array hashes are explicitly receipted; full epoch caches
and progress logs are excluded. The original scientific working files remain
untouched. Shared helper snapshots added at packaging are identified as such,
without inventing historical execution hashes.

The saved runs used Python 3.12.5 and NumPy 1.26.4, with one BLAS thread per
worker; SciPy 1.13.1 and pandas 2.2.2 are the matching replay environment.
Execution timestamps, timing columns and local provenance paths will differ.
Compare p/q values, calls and summary statistics by their declared identities,
rather than requiring runtime receipts to be byte-identical. Use a new output
folder on each run. `validation/work` is ignored by Git.
