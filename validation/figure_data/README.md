# Packaged figure inputs

This directory contains frozen synthetic benchmark decisions/summaries and
selected QC aggregate counts used to render Figures 1, 3 and 5. It contains no
private clinical recording, trial epoch, participant/contact locator or raw
cohort data. Synthetic family rows are generated simulations.

The QC counts reproduce the published figure-level accounting. In particular,
the clinical QC-eligible and non-stimulation sets each contain 143 contacts but
intersect in 142; the BH-call count is not asserted to be an independently
joined post-QC count. Missing public QC joins remain null.

Hashes are in `source_manifest.json`. Figure 2/4 frozen display SVGs are not
included: their scope and required authorized inputs are described in
`docs/FIGURE_REPRODUCTION.md`. The raw clinical cohort is not redistributed.

## Frozen synthetic conventions

The nominal 250 Hz response and matched-baseline settings are 16–320 ms and
−320 to −16 ms. The frozen floating-point time grid and literal comparisons
actually select 76 samples in each window: 16–316 ms and −320 to −20 ms.
Reproducing a 77-sample inclusive window would change this benchmark. Preserve
the published settings and grid when replaying the frozen outputs.

The factorial generator RMS-normalizes its three-Gaussian waveform. The
component-null stress generator additionally subtracts the template mean and
normalizes the demeaned template to unit RMS on the selected response samples.
Its random-polarity injection therefore has RMS 20 microvolts; the bare
three-Gaussian expression must not be substituted without this normalization.

In `family_calibration.csv`, the legacy `total_false_call_rate` is the mean of
within-family fractions among evaluable null contacts, not a single ratio of
pooled false calls to pooled evaluable null contacts. Family sizes can differ.
For example, the all-null standalone projection result is 0.1565972 by that
legacy field, whereas 278/1,800 pooled evaluable contacts is 0.1544444.

Legacy recovery/evaluability interval columns in the frozen tables preserve
historical output. The manuscript uses `stratified_recovery.json` for recovery
intervals: 20,000 resamples within the fixed design cells. Fixed evaluability
is reported as a design proportion, without an inferential confidence interval.
