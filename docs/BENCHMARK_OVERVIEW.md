# Benchmarks and public-data checks: an overview

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

This page explains what the synthetic benchmark, the public-data examples and
the annotation comparisons each establish, and how to run the public-data
checks. Full results and commands are in the [validation guide](VALIDATION.md).

## How the synthetic benchmark relates to the real-data examples

The benchmark in
[`validation/benchmark_crp_energy.py`](../validation/benchmark_crp_energy.py)
characterizes the joint projection–energy decision under one declared synthetic
model. It contains generated values exclusively and creates 2,400 independent
simulated stimulation acquisitions by
varying trial count, response-to-noise ratio, and the fraction of target trials
randomly masked after upstream QC. Each acquisition contains stationary colored noise in
four channels. A known, RMS-normalized biphasic response with small amplitude
and latency variation is added only to one designated target channel; the
other three channels remain known nulls.

In the benchmark, a **testing family** means the four simulated channels
processed together in one acquisition. It is the small synthetic counterpart
of the recording contacts considered for one multiple-testing correction in a
real stimulation acquisition; only detector-eligible contacts with finite
joint p-values enter the Benjamini-Hochberg adjustment. The adjusted set has
three contacts when random masking leaves the target with too few usable trials
and four contacts otherwise (600 and 1,800 of the 2,400 families,
respectively). Here, “family” has no kinship or cross-dataset meaning. At zero response-to-noise ratio, all four
simulated channels are null. At nonzero ratios, the target contains the
injected response and the three references remain null.

Known synthetic truth makes false calls, evaluability, and response detection
measurable under these specific conditions. The model uses stationary,
independent Gaussian AR(1) noise, one biphasic morphology, four-channel
families, and signal-independent random target-trial removal. The removal step
tests detector behavior after missingness; it does not evaluate the accuracy
of artifact identification. The public `ds003708` example shows that ERPy can
read and analyze an independently hosted real recording, while a governed
deidentified example shows the same end-to-end workflow on local data. The
three sources serve distinct roles and are reported separately.

The four-channel family is smaller than the 30- and 143-contact real examples.
For `n <= 12`, the exact reproducibility component has minimum attainable
`p_R = 2**(-(n - 1))`; with eight clean trials an isolated BH discovery at
alpha 0.05 is arithmetically possible only through six contacts. With 5,000
Monte Carlo draws, the minimum estimated p-value is `1 / 5001`, making an
isolated discovery arithmetically possible through 250 contacts. The exact
energy component has minimum `p_E = 2**(-n)`, so the exact conjunction floor
is governed by `p_R`. These are resolution bounds rather than power
guarantees. Plan trial counts, family size, and randomization budget together.
The exact implementation omits the observed all-positive assignment from the
matrix calculation, counts it explicitly through a plus-one numerator, and
uses a scale-aware floating-point tolerance for numerically tied statistics.

The saved target-level component comparison uses the primary row's own component values.
Among 360 evaluable noise-only targets, `p_R <= 0.05`, `p_E <= 0.05`, and
`max(p_R, p_E) <= 0.05` occurred 19, 21, and 1 times, respectively; no joint
call survived family adjustment. Among 1,440 evaluable injected targets, the
corresponding recovery counts were 1,423, 1,302, and 1,298 before 1,205 joint
calls survived adjustment. The separately named `crp_only_fdr` and
`paired_rms_fdr` benchmark rows are source-informed comparator
implementations, rather than these primary component tests.

A focused companion benchmark in
[`validation/benchmark_crp_energy_composite_nulls.py`](../validation/benchmark_crp_energy_composite_nulls.py)
tests both one-component-null branches with 1,000 independently seeded
four-contact families per branch and 12 trials per contact. Both tests use
exact enumeration. In the projection-alternative/energy-null branch, the raw
joint and adjusted target call rates were 5.6% and 1.9%; in the
energy-alternative/projection-null branch, they were 5.1% and 1.0%. Complete
results, Wilson intervals, provenance, and checksums are retained in
`release_artifacts/benchmark/composite_null_stress_test/` when the command is
run.

The standalone selected-duration CRP p-value is explicitly exploratory: its
duration optimization and shared-trial projection t-test were anti-calibrated
under the all-null simulation. The paired-Wilcoxon RMS field now separately
demeans its matched segments, correcting baseline-centering leakage observed
during release review. The reader-facing RMS binary field continues to share
the peak-z rule with the peak-amplitude comparator and supplies no additional
independent vote.

## Public-data portability targets

The validation script in [`validation/validate_public_spes.py`](../validation/validate_public_spes.py) is designed for small, reproducible OpenNeuro slices:

| Dataset | Why it matters |
| --- | --- |
| `ds003708` | Mayo one-patient iEEG SPES dataset used around basis profile curve work; open BrainVision signal derivative. |
| `ds004080` | 74-subject CCEP ECoG SPES dataset across age 4-51; open raw BrainVision signals. |

From a source checkout, install the validation dependencies once and then run:

```bash
python -m pip install -e ".[validation]"
python validation/validate_public_spes.py --dataset ds003708 --min-events 8 --max-events 8
python validation/validate_public_spes.py --dataset ds004080 --min-events 8 --max-events 8
```

Eight artifact-clean trials are required by the primary CRP-energy detector.
Smaller slices can still test download, event, epoch, and QC interoperability,
but their primary detector result is correctly reported as insufficient data.

Generate high quality visualizations from a larger real-data sample spanning
multiple stimulation sites:

```bash
python validation/validate_public_spes.py \
  --dataset all \
  --sites-per-dataset 3 \
  --max-events 10 \
  --max-channels 48 \
  --figures
```

These high quality visualizations include waveform panels with
post-artifact zero-anchor verification, response-ranked grids,
anatomy-organized grids, response maps, exploratory within-stimulation metric
z-score plots, CRP canonical response views, response matrices, network
summaries, and time-resolved brain views.

Large datasets are intentionally sampled by stimulation event window; ERPy should not require downloading a full multi-GB BIDS run to prove that open raw signal loading, event creation, epoching, QC, and detection work.

## Detector–annotation agreement

The [agreement guide](ANNOTATION_AGREEMENT.md) explains the current broad-response and focused N1 comparisons. The broad detector is compared with released negative-N1 ratings using equal weight per contact/pair record and participant-level uncertainty. Its four recorded examples retain their original q values and source hashes. N1-negative ratings are not a complete reference for every evoked response.

The retained aggregate tables include counts, paired differences and participant-level uncertainty. Their manifest identifies the fixed inputs and outputs.

## See also

- [Validation (complete results)](VALIDATION.md)
- [External expert-label evaluation](EXTERNAL_VALIDATION.md)
- [Detector–annotation agreement](ANNOTATION_AGREEMENT.md)
