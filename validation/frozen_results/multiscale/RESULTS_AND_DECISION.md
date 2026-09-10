# Optional multiscale comparison: final bounded evaluation

**Decision: retain the existing single-window default.** Neither the original equal-weight bank nor the single amended fixed-weight bank establishes a general recovery improvement over the matched broad window. Both remain optional experimental APIs. No further window or weight search was performed.

The original plan preceded all unweighted outcomes. The explicit weighted amendment (early 0.1, late 0.1, broad 0.8) was written after seeing the unweighted outcome, before any weighted outcomes. Development recombination is separated from one fresh 2,000-family confirmation. Its independent family seeds have no overlap with development. This is an internal prespecified analysis and an openly declared later amendment, not a registered protocol.

## Fresh confirmation: recovery of the injected target

Each row pools the fixed four-shape, two-polarity, three-signal-level grid, 20 independent families per cell. Values are percentages; intervals are stratified paired-family bootstrap 95% intervals for this grid. They do not describe clinical-population uncertainty.

| Trials | Families | Early | Original broad 15–300 ms | Matched broad 10–300 ms | Equal-weight bank | Weighted bank | Weighted minus matched broad (pp, 95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 480 | 48.54 | 64.38 | 64.38 | 0.21 | 60.00 | -4.38 (-6.05, -2.71) |
| 12 | 480 | 59.17 | 74.38 | 74.58 | 74.38 | 74.58 | +0.00 (-1.88, +1.88) |
| 24 | 480 | 66.67 | 87.71 | 87.71 | 87.71 | 88.96 | +1.25 (-0.62, +3.12) |

Across all 1,440 confirmation target families, weighted recovery was 74.51% versus 75.56% for matched broad: -1.04 percentage points (95% CI -2.08 to -0.07). The equal-weight development bank recovered 0/480 eight-trial targets, 365/480 twelve-trial targets and 421/480 twenty-four-trial targets; those results remain intact. Weighted development showed a positive 24-trial point difference with a conditional interval above zero, but its fresh confirmation interval includes zero. Do not select the development finding as evidence of general improvement.

At eight trials, an isolated four-contact discovery has equal-weight adjusted resolution 12/128 = 0.09375. The amended broad weight permits 4/(128×0.8) = 0.0390625, while retaining penalties for the other windows. This restores the possibility of broad-window detection but still reduces recovery versus a single broad test. Unavailable windows contribute p=1 and never redistribute or reduce the declared multiplicity.

## Global-null checks in fresh confirmation

Rates below count families with at least one false call; Wilson intervals use independent generated families. They are empirical checks under the stated generators, not proof of calibration under arbitrary physiology or response-dependent selection.

| Construction | Families | Matched broad any false call | Equal-weight any false call | Weighted any false call (95% CI) |
|---|---:|---:|---:|---:|
| noise_only | 240 | 0 | 0 | 0/240 (0.00%; 0.00–1.58%) |
| projection_alt_energy_null | 160 | 0 | 0 | 0/160 (0.00%; 0.00–2.34%) |
| energy_alt_projection_null | 160 | 0 | 0 | 0/160 (0.00%; 0.00–2.34%) |

The ramp stress uses independent exchangeable positive slopes for baseline and response, giving joint sign symmetry of all matched-window log-RMS differences after separate demeaning. The converse uses independently signed response templates with centrally symmetric noise, giving the whole-trial projection null jointly over windows. The global multiscale null is the intersection of window-specific component-union nulls. Bonferroni requires valid component/window p-values but no cross-window independence; across-contact BH retains its own conditions. No mixed-window construction is mislabeled as globally null.

## Public baseline-injection characterization

The 960 source-based families use two annotation-blind prestimulus fixtures: MAYO02/LG1-LG2/LAS1 and UMCU20/HF14-HF15/F01, ten native trials each. The initial ≥24-trial criterion proved infeasible from source metadata and was amended before outcomes to ten native trials, with n=8/10 selected without replacement. Disjoint prestimulus blocks were resampled to 500 Hz, swapped and whole-trial-sign randomized before injection. The same finite source recordings are reused across contacts and scenarios; these are conditional semisynthetic experiments, not 960 independent clinical recordings. Their imposed projection null is not a natural-noise false-positive estimate.

| Native trials used | Families with injection | Matched broad recovery | Equal-weight recovery | Weighted recovery |
|---|---:|---:|---:|---:|
| 8 | 320 | 53.75% | 0.00% | 52.19% |
| 10 | 320 | 67.19% | 69.38% | 63.75% |

Complete positive/negative and morphology-specific cells, including zero rates, are retained in each `all_cells.csv`; all contact/window p-values, seeds, availability and realized intervals remain in compressed row-level results. No family failed, and no production warning was recorded. All tested contact/window evaluations were available.

## Verification, timing and attribution

31 focused tests passed. They cover exact core concordance, whole-record polarity reflection including Monte Carlo paths, common clean-trial masks, missing-window conservatism, fixed weights, bank/contact ordering, malformed inputs and both eight-trial resolution bounds. Two deliberate identical-trial tests emit the existing descriptive t-statistic precision-loss warning; the randomization calculations and all production runs are unaffected. This count is a targeted subset, not a claimed total project suite.

The audit independently recomputes BH arithmetic from saved contact p-values, verifies all scientific output hashes, checks complete grids and disjoint confirmation seeds, and exactly replays deterministic raw-data scenarios from every scenario/trial-count combination. The original run manifests accidentally included progress-log hashes while logs were still growing; they remain preserved, and the amended runner freezes only explicit scientific artifacts. This bookkeeping repair did not alter prior numerical results. Exact execution code snapshots are retained.

CRP supplies the semi-normalized projection representation (Miller et al., 2023, https://doi.org/10.1371/journal.pcbi.1011105). ER-detect also includes fixed-window CRP projection t-threshold detection, Python/BIDS workflow and selectable polarity (van Boom et al., 2025, https://doi.org/10.1016/j.jneumeth.2025.110389). ERPy combines its existing whole-trial sign-randomized projection and separately demeaned log-RMS conjunction with standard Bonferroni weighting. This is not a new statistical theorem, a first projection detector, or evidence of clinical utility.

- Original 2,000 generated families: 61.09 s for family evaluation, 3 workers, Python 3.12.5, NumPy 1.26.4, macOS-26.2-arm64-arm-64bit, single BLAS thread per worker. Measured workload timing excludes summary/report generation and is not a scalability benchmark.
- 960 source-based families: 54.94 s for family evaluation, 2 workers, Python 3.12.5, NumPy 1.26.4, macOS-26.2-arm64-arm-64bit, single BLAS thread per worker. Measured workload timing excludes summary/report generation and is not a scalability benchmark.
- Fresh 2,000-family confirmation: 60.28 s for family evaluation, 3 workers, Python 3.12.5, NumPy 1.26.4, macOS-26.2-arm64-arm-64bit, single BLAS thread per worker. Measured workload timing excludes summary/report generation and is not a scalability benchmark.
