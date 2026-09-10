# Broad-response detection and N1 annotation agreement

The broad-response analysis compares saved ERPy 15–300 ms detector calls with released human negative-N1 ratings in public OpenNeuro ds004774. Its analysis and examples remain unchanged. The N1 analysis separately compares the [fixed label-free N1 detector](N1_LABEL_FREE.md) with archived ER-detect; the trained morphology and hybrid models remain secondary comparisons. Historical calls, models and prediction records are retained.

These are different endpoints. The ER-detect paper defines manual N1 labels by a negative peak at 10–90 ms. ERPy's default response window tests reproducible, elevated response energy without selecting a polarity or requiring an N1 peak. A discordant broad call does not by itself establish either a biological false response or an annotation error. [ER-detect Methods 2.5](https://pmc.ncbi.nlm.nih.gov/articles/PMC12267860/).

## Interpreting broad-response agreement

There are 32,121 evaluable contact/pair records from 13 participants, excluding the previously used development participant MAYO01. Each record contributes total weight one, divided equally among its available raters. If half the raters label N1, half that record contributes to each N1 reference category. The matrix therefore contains fractional record equivalents; it does not round mixed ratings to a majority vote.

Agreement is 86.35% (nominal 95% interval 81.47–89.86%); kappa is 0.494 (0.449–0.524). Grouped call rates distinguish unanimous N1-positive, mixed, unanimous N1-negative and single-rater patterns. Their intervals resample whole participants within site, keeping each participant's contacts, pairs and ratings together (2,000 draws; seed 20260907). A paired contrast uses the same draws for both groups. These intervals describe the saved predictions and this convenience cohort; they do not include model-fitting or source-selection uncertainty.

Code 2 identifies an observed P1 label in the released annotation files. Some raters used only codes 0/1, and the published comparison recodes P1 as negative for N1. It would be incorrect to convert these data into a complete reference for “any evoked response.” The optional P1 count is a partial annotation description. [Released comparison code](https://github.com/MaxvandenBoom/Paper_VandenBoom_ERDetect/blob/ebbbdc8dc6436cd16df8d54e9e6320c0cbe3150a/functions/ccep_annot_matchManualAndAuto.m#L109-L142).

The broad external run uses a 15–300 ms response, 0–15 ms excluded artifact interval, −1 to −0.1 s candidate baseline and at least eight finite matched trials. It keeps the last baseline samples matching the response sample count. Trial normalization uses a median over the rounded half-open −0.5 to −0.02 s slice. Thus “default” identifies the response-window detector; the external-data preprocessing is explicitly configured. The call is the saved BH decision across retained non-stimulation contacts, with no additional acquisition-wide contact-QC gate in this benchmark.

The four examples were selected before viewing their waveforms: in each unanimous binary rating/call group with at least two raters, take the record closest to the median log10(q), breaking ties by subject, pair and channel. Their source-epoch hashes match the saved predictions. Mean and SEM use the same normalized finite matched trials. Examples illustrate agreement and disagreement; their detector q values are not tests of annotation correctness. The displayed tail after 300 ms is outside the tested response window.

## Reading the retained analysis

The full contact matrix and individual-rater records remain in
`validation/frozen_results/erdetect`. The aggregate agreement tables and
`validation/annotation_agreement/manifest.json` identify their fixed inputs and
outputs. `source_traces` contains the four selected public-data summaries and
reconstruction metadata with dataset-relative paths. Historical source hashes
identify the executed analysis; the summaries can be inspected directly as CSV.

The N1 paired intervals are nominal conditional intervals for the fixed label-free calls and archived comparator on the same participants. An interval including zero does not demonstrate equivalence or noninferiority. The cohort had already been examined before the fixed rule was proposed, and no threshold was selected on its new scores. The previously retained six classifier intervals in `annotation_agreement/supervised_paired_intervals.csv` remain valid secondary supervised evidence; they are not the primary label-free N1 comparison data.
