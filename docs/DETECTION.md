# Response detection

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

ERPy reports several published detectors side by side and one recommended
primary result, the CRP-energy conjunction. This guide explains how to run
them and how to read the output columns. The mathematical definitions are in
[METHODS.md](METHODS.md).

## Running detectors

```python
detections = clean_epochs.detect()          # primary detector: one row per channel
everything = clean_epochs.detect_all()      # every detector, long form, plus the primary columns
keller = clean_epochs.detect("keller_zscore")
```

`ep.ALL_METHODS` lists the available detectors: `kundu_rolston`,
`keller_zscore`, `crp_significance`, `crp_energy`, `crowther_gamma`,
`peak_amplitude`, and `rms_response`. `ep.DETECTOR_DEFINITIONS` and
`ep.DETECTOR_REFERENCES` give each one's definition and source publication.
Run detection on artifact-cleaned epochs (see
[preprocessing and quality control](PREPROCESSING_AND_QC.md)); the one-call
`patient.analyze(...)` does this for you.

## High-gamma input

The standard `blank_filt` stream is 0.1–50 Hz, so its SIGNI/high-gamma row is
reported as unavailable rather than negative. To evaluate 70–170-Hz gamma,
pass a separately artifact-masked, passband-preserving object through
`wideband_epochs=`. The legacy `consensus` field counts only
methods whose required inputs are available.

When the epochs are not explicitly wideband, `crowther_gamma` remains in the
long-form table with `method_available=False`, an availability reason, and no
scientific call. Pass `wideband_epochs=clean_wideband_epochs` only when that
object truly retains 70–170 Hz.

## The primary detector

The recommended detector is `crp_energy`. It tests two necessary components on
the same artifact-cleaned trials. Amplitude-weighted waveform reproducibility
uses whole-trial sign flips of an ordered, semi-normalized cross-projection
statistic over the full response window declared before inference; the
CRP-selected duration remains descriptive. Excess energy
uses a one-sided paired sign-flip test of the mean log RMS ratio after the
matched response and baseline segments are each demeaned separately. Their
intersection-union value is \(p_{\mathrm{joint}}=\max(p_R,p_E)\); `q_joint`
applies Benjamini-Hochberg adjustment over the eligible recording-channel
family. The historical result fields `p_crp` and `p_energy` store \(p_R\) and
\(p_E\), respectively. Both contacts defining the bipolar
stimulation pair and channels with insufficient matched clean trials are
outside that family. Their unadjusted detector quantities remain auditable,
but stimulation contacts receive `qc_status="stimulation_contact_excluded"`,
no `q_joint`, and cannot enter network or travelling-wave outputs. In the
long-form detection table, the `primary_significant` column records the
BH-adjusted statistical decision before the separate response-artifact gate,
whereas `primary_qc_pass` records the combined decision. By contrast,
`AnalysisResult.primary_significant` is a convenience property that already
returns only primary rows passing artifact QC and is the recommended final
result for a one-call analysis. CRP-only, energy-only, the former
`shape_magnitude_significant` CRP-Kundu rule, and count-based `consensus_ch`
remain available as explicitly labeled descriptive comparisons.

## Reading the output columns

Detection and waveform-audit tables distinguish the default components from
standalone comparator calls:

| Field | Meaning |
| --- | --- |
| `primary_reproducibility_pass`, `primary_energy_pass` | Unadjusted component decisions within the default `crp_energy` method. |
| `primary_significant` (detection table), `primary_detector_pass` (audit) | Joint BH decision before the separate contact-QC gate. |
| `primary_qc_pass` (audit) | Final intersection of the joint decision and contact-QC eligibility. |
| `comparator_crp_pass` | Standalone `crp_significance` comparator, using its own adjusted result. |
| `comparator_kundu_pass` | Standalone `kundu_rolston` comparator's published-rule call. |

Each comparator has a matching `comparator_*_available` boolean and
`comparator_*_availability_reason`. Calls use nullable booleans: missing
(`pd.NA`, empty in CSV) means not run or unavailable, not a negative result.
Reasons distinguish `not_run`, missing availability metadata, unavailable
method input, and missing required comparator evidence. Availability requires
explicit method metadata and finite applicable quantities; a boolean call alone
is insufficient. Only two available comparator calls can produce the audit
reason `comparator_crp_kundu_disagreement`; this review diagnostic does not
change detector decisions or contact eligibility.

`primary_shape_pass` and `primary_magnitude_pass` are deprecated compatibility
aliases for `comparator_crp_pass` and `comparator_kundu_pass`, respectively,
including their missing states. They do not represent the default detector's
components. Code using these aliases as ordinary booleans should migrate to the
comparator names and inspect availability first. The historical
`shape_magnitude_significant`/`shape_magnitude_detector_pass` conjunction and
consensus fields retain their existing selection semantics.

The primary result retains the historical field names `p_crp` and
`crp_statistic`; in version 1.0.0 they contain the fixed-window
reproducibility value \(p_R\) and statistic \(T_R\). The standalone
`crp_significance` comparator keeps its own selected-duration extraction-test
quantities.

## Detector options

```python
result = ep.detect(
    clean_epochs,
    method="crp_energy",
    response_window=(0.015, 1.0),
    baseline_window=(-1.0, -0.015),
    alpha=0.05,
    correction="fdr_bh",
    min_clean_trials=8,
    n_permutations=5000,
    canonical_energy_cv=True,
    random_state=42,
)
```

The result retains four unadjusted component-test states:
`reproducible_energetic`, `reproducible_low_energy`,
`energetic_inconsistent`, and `no_response`. They are determined by
`p_crp <= 0.05` and `p_energy <= 0.05`; the final call still uses `q_joint`
and artifact eligibility. Canonical projection energy and its fraction are
computed leave-one-trial-out at the duration selected from all clean trials
and are reported as effect sizes. Too few clean trials yield
`insufficient_data` instead of an exception.

`validation/validate_crp_energy_real_baseline.py` injects the four diagnostic
response patterns into matched prestimulus noise from a saved epoch file. It
writes only derived detector quantities and can be run with `--strict` to make
class recovery and the joint-p identity executable release checks.

## Adjudicating one channel

To adjudicate one channel with the exact source-native quantities, use the
six-panel detector diagnostic. It shows the mean response and Keller/Hays
windows, Kundu envelope and thresholds, the descriptive CRP canonical
reconstruction and data-driven duration, the fixed-window reproducibility
test, the paired CRP-energy RMS comparison, and all detector decisions. Pass
the same artifact-masked standard and wideband
epochs used for detection when reproducing a saved result.

`ERPy.DETECTOR_QUANTITY_COLUMNS` lists the source-native output columns for
each detector. Hays N1 quantities use explicit `hays_n1_*` aliases even though
their 10-50-ms measurement window is shared with the Keller implementation.

```python
fig = clean_epochs.plot.detectors(
    "CONTACT_B1",
    detections=detections,
    wideband_epochs=clean_wideband_epochs,
)
```

## Canonical response parameterization (CRP)

CRP analysis exposes both the learned canonical response curve and the time-resolved canonical weight across the epoch, plus comparison metrics for stimulation-site contrasts:

```python
from ERPy.crp import CRPConfig, compare_crp_across_stim_sites, run_crp_all
from ERPy.viz.crp import plot_crp_site_comparison, plot_crp_summary

crp_config = CRPConfig(
    response_window=(0.015, 0.30),
    baseline_window=(-0.30, -0.05),
    permutation_n=200,
)
crp_table = run_crp_all(epochs, config=crp_config)
plot_crp_summary(epochs, "CONTACT_B1", config=crp_config)

comparison = compare_crp_across_stim_sites({"STIM_A_1_2": crp_table})
plot_crp_site_comparison(comparison, metric="within_stim_z")
```

`compare_crp_across_stim_sites` reports within-stimulation-site z-scored canonical weights, within-channel baseline-normalized CRP expression, reconstructed response RMS in microvolts, and optional permutation/null p-values. The more general `zscore_metric_within_stim` helper works with any response metric column, including peak amplitude, RMS, canonical weight, reconstructed CRP RMS, or user-defined metrics.

## Label-free negative-N1 detection

`ERPy.n1_detection.detect_n1_family` is the proposed N1 method. It requires no
training labels or fitted model: an interior negative peak in the fixed early
window must exceed 3.4 times the mean-baseline sample SD, with a 50 µV SD floor.
The original early projection–energy joint p becomes 1 when this gate fails,
then BH is recomputed over the complete original finite contact family.
Unannotated contacts remain in that family. The general polarity-invariant
detector keeps its existing defaults.

On the previously examined paired public cohort, the fixed rule had 54.80%
sensitivity, 98.50% specificity and 84.60% PPV. Sensitivity was lower than
archived ER-detect; paired intervals did not establish higher specificity or
PPV. Its p/q values concern the original response conjunction, not N1 truth.
See the [label-free guide](N1_LABEL_FREE.md) for the checked array API,
complete-family public example, uncertainty, synthetic checks and limitations.

## Secondary supervised N1 development models

`ERPy.n1` retains separate supervised negative-N1 ECoG annotation scores and decisions;
the default polarity-invariant detector and its p/q values remain unchanged.
The morphology arm includes trial-consistency descriptors; the hybrid adds
reproducibility, energy and RMS inference features. Training uses fixed C=1,
soft expert-vote targets with total weight one per record, train-only scaling,
and nested leave-one-participant-out threshold selection. Its score is not a
p value or q value, and the fitted population does not validate sEEG use.
Hybrid prediction requires checked early-window evidence. Use
`ERPy.n1.prepare_n1_hybrid_inputs(trials_uv, times_seconds)` to construct both
paths from matching arrays and retain their separate finite-trial masks; bare
inference dictionaries are rejected. The guide also provides a checksum-verified
saved-reference adapter with explicit historical trial-provenance limits.

The comparison uses archived negative-N1 outputs from ER-detect. ER-detect also
supports other polarities and detection methods, including CRP similarity across
trials; the ERPy hybrid is a separate classifier and does not use those calls as
predictors. See the [method comparison](N1_DEVELOPMENT.md#how-this-relates-to-er-detect)
and [ER-detect paper](https://doi.org/10.1016/j.jneumeth.2025.110389).

This is corrected post hoc development after the original external evaluation,
with all candidates and the identifier-join correction disclosed. Saved reference
inputs and models are checksum-pinned. The [N1 development guide](N1_DEVELOPMENT.md)
explains the model, limitations, API and complete reproduction commands.
Runtime prediction uses the existing NumPy/SciPy dependencies; install
`python -m pip install -e ".[n1-training]"` for optional scikit-learn training.
The N1 extraction, training and secondary-summary scripts require Python
3.11+ and were validated on 3.12.5. Portable model prediction retains the
package's broader declared Python range and does not require scikit-learn.

The [optional multiple-window experiment](MULTISCALE_DEVELOPMENT.md) tests
several response intervals with a correction for searching across them. Neither
equal weights nor one declared set of unequal weights established a general
improvement in a separate confirmation simulation. The single-window default
remains. Plans, source versions, result tables and two small public baseline
samples are retained so the experiment can be reproduced.

## See also

- [Methods and mathematical definitions](METHODS.md)
- [Equation and implementation reference](EQUATION_REFERENCE.md)
- [Validation](VALIDATION.md)
- [Label-free N1 detection](N1_LABEL_FREE.md)
- [Secondary supervised N1 models](N1_DEVELOPMENT.md)
