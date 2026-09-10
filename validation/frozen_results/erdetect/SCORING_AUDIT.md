# Frozen external-label scoring audit

This report records the original completed scoring run. The post-review
coverage addendum below makes its eligibility denominator explicit; the
original predictions and conditional performance estimates are unchanged.

The final run used all 26 released annotation files and completed predictions
from 14 participants. The primary independent cohort excludes MAYO01, which
overlaps the earlier public worked example. It contains four Mayo and nine
UMCU participants. All-14 replication and MAYO01-only results are retained.
The overlap amendment followed the Mayo pilot scoring and preceded UMCU
scoring; no detector settings changed. This timing is recorded in summary.json.

Each labeled channel/pair contributes total reference weight one. Its positive
weight is the fraction of available raters marking N1. Thus the confusion
counts below target a randomly chosen available rater within each record;
they are not majority-vote labels or an average of participant metrics.

## Primary 13-participant results

There are 33,694 source- and site-eligible labeled records, of which 32,121 are evaluable
(95.33%). Eligibility applies the published 12-mm exclusion to either
stimulation contact at Mayo; UMCU has no released coordinates and has no
spatial exclusion. Overall results combine these different site rules.

| ERPy configuration, available case | Sensitivity (95% CI) | Specificity (95% CI) | Balanced accuracy | PPV | NPV | Kappa |
| --- | --- | --- | --- | --- | --- | --- |
| Early 10–90 ms | 77.40% (62.86–89.27) | 89.59% (83.10–93.98) | 83.49% | 52.78% | 96.35% | 0.559 |
| Broad 15–300 ms | 69.79% (55.20–82.20) | 88.84% (81.63–93.61) | 79.31% | 48.45% | 95.14% | 0.494 |

Intervals resample whole participants within site 2,000 times, seed 20260907.
All nested records and raters remain together. All 2,000 draws were defined
for each primary overall metric. These intervals describe uncertainty in a
small convenience cohort, not population representativeness.

On the 32,048 paired available records with comparable archived trial sets,
early ERPy sensitivity/specificity are 77.41%/89.57%; archived ER-detect values
are 67.61%/97.64%. Their balanced accuracies are 83.49% and 82.63%, their PPVs
52.72% and 81.17%, and their kappas 0.559 and 0.702. This is a descriptive
sensitivity/specificity tradeoff, not evidence of superiority. The historical
comparator is a frozen-output comparison, not a source-package rerun.

The early ERPy site results differ: sensitivity/specificity are 61.49%/96.47%
for the four independent Mayo participants and 78.03%/87.66% for nine UMCU
participants. Subject-specific rows and all uncertainty estimates remain in
metrics.csv. Pooled figures weight records, not subjects equally.

The independent primary early confusion weights are TP 3248.5, FN 948.666667,
TN 25017.333333, FP 2906.5. An independent reconstruction from individual
rater rows with weights 1/n_available_raters reproduced all four within 1e-9.

## Missingness, provenance, and interpretation

Final scoring passed the processing-completeness guard. Incomplete computation
was not counted as a no-call. The separate unavailable-as-no-call policy
includes detector unavailability and explicitly documented missing raw data.
The UMCU25 truncated-source exclusions are supplied by the pair manifest.
Its partially recovered F52-F53 pair remains in ERPy-versus-rater scoring but
is excluded from the paired archived comparison because the trial sets differ.

The full 14-participant replication analysis has 34,255 evaluable records out
of 35,889 eligible labeled records (95.45%). Across that cohort, each detector
configuration has 977 detector-unavailable records and 657 labeled records
with documented missing raw data. The all-14 numbers are a replication view,
not a wholly independent external validation.

These are concordance metrics for negative early N1 annotations. Both ERPy
configurations accept either polarity, and the broad window accepts later
responses. Discordance with an N1-negative annotation is not proof that no
biological evoked response exists. Annotation 2 (P1) remains N1-negative and
has its own secondary discordance table; it was never relabeled N1-positive.

No BH values were recomputed after reference-label or spatial exclusions.
Missing labels and stimulation-contact codes were excluded, never mapped to
negative references. Strict consensus requires at least two unanimous raters.
Interrater agreement uses only shared valid records after source and
site-specific spatial eligibility, independently of prediction availability.

## Verification and outputs

All 16 scorer tests pass. They cover label orientation and canonical joins,
NaN/-1/P1 handling, unequal rater counts, strict consensus, unavailable/no-call
policies, incomplete-processing rejection, documented raw loss and historical
trial mismatch, either-contact geometry, whole-subject/site bootstrap behavior,
and separation of the development-overlap cohort.

- summary.json: complete interpretation, cohort timing, software/scorer hash,
  frozen comparator configurations, input checksums, and overall metrics.
- metrics.csv: pooled, individual-rater, strict-consensus and secondary
  historical-25-file views; participant, site and overall rows; three cohorts.
- record_predictions_and_labels.csv and rater_label_joins.csv: complete joins,
  original raw annotation codes, unchanged q-values, eligibility and provenance.
- record_exclusion_denominators.csv and annotation_exclusion_denominators.csv:
  omitted-record and label denominators.
- availability_by_reference.csv: coverage and reference-positive/negative
  weights among available and unavailable records.
- annotation_category_discordance.csv: distinct annotation0/N1/annotation2
  outcomes, with literal rater counts and record-normalized weights.
- interrater_agreement.csv: shared-record pairwise agreement and kappa.

At the original scoring audit, the scorer and tests were local changes awaiting
integration; the summary preserves that checkout and scorer-file identity.
No code was pushed by the scoring audit.

## Post-review coverage addendum

The primary 13-participant frame contains 44,795 unique channel/pair records
with a valid reference. A declared mutually exclusive hierarchy excludes
4,559 contact/stimulation-ineligible records, 5,901 without matching released
source events, 73 with fewer than five good source events, and 568 failing
site spatial eligibility. This leaves the original 33,694 denominator.
Within it, 657 lack raw input because of documented truncation, 914 have fewer
than eight retained trials, and two have a degenerate zero-energy response
trial; 32,121 are evaluable.

The 5,901 absent-source records are otherwise eligible and each has only one
available rater. They cover 75 additional pairs in UMCU20, UMCU25 and UMCU62.
No verified raw-source mapping was recovered for them. A secondary expanded
coverage frame adds those records: 32,121/39,595 = 81.12%, compared with the
source/site-conditional 95.33%. The expanded frame describes source coverage,
not measured detector accuracy. Missing inputs never become observed negative
calls. The [full participant flow](coverage_audit/annotation_source_flow.csv)
and [absent-source pair inventory](coverage_audit/source_event_absent_pairs.csv)
retain exact counts and reference composition; the compressed source joins
reproduce them exactly using `audit_annotation_coverage.py`.

The later [cache-safeguard verification](cache_safeguard_verification.json)
records additional code tests and a fresh public-pair numerical reproduction.
It does not replace or retrospectively relabel the original frozen prediction
provenance.
