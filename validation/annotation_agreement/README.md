# Broad-detector agreement analysis

This directory contains a reproducible descriptive analysis of the saved public ds004774 15–300 ms detector calls and released negative-N1 annotations. It preserves the original prediction records and settings. See [the user guide](../../docs/ANNOTATION_AGREEMENT.md) for endpoint definitions and interpretation.

Run `python validation/analyze_annotation_agreement.py --verify` from the repository root. Verification recomputes all fourteen retained outputs in a temporary directory, checks exact bytes and validates original input hashes. Use `--output PATH` to produce a new copy, and optionally `--epochs PATH` to reconstruct the selected traces from hash-matching public NPZ caches. The default path uses the preserved trace summaries, so no raw data download is required.

- `crossclassification.csv`: four fractional record-equivalent cells, proportions and participant-cluster intervals.
- `summary.json`: settings, denominators, estimates and interval definitions.
- `annotation_group_rates.csv` and `subject_annotation_group_rates.csv`: five explicit rating patterns, separating single-rater records.
- `three_annotation_group_rates.csv` and its subject counterpart: an alternative collapse with single-rater counts stated.
- `annotation_group_contrasts.csv`: shared-draw paired differences in call rates; nominal intervals, not adjusted significance tests.
- `subject_metrics.csv`: participant-level counts and agreement estimates.
- `record_classification.csv.gz`: all 32,121 available primary records with original p/q, trial counts and annotation votes.
- `observed_annotation_codes_by_rater.csv`: the incomplete P1-label coverage by rater/site.
- `exemplar_selection.csv`, `exemplar_traces.csv.gz`, `exemplar_source_provenance.json`: four examples selected by a fixed q-rank rule before waveform inspection, with matched epoch hashes and trial indices.
- `figure7_paired_intervals.csv`: six retained secondary supervised N1 comparison intervals, without new fits or resampling. These historical classifier comparisons are not the current primary label-free Figure7 data; see `../n1_label_free/empirical_scoring/` and the separately reported `../n1_label_free/nominal_stage/`.
- `source_traces`: pinned summaries and metadata needed to reproduce the figure without redistributing full source recordings.

All waveform summaries derive from the public CC0 ds004774 data. No private cohort data or copied GPL implementation is included. The displayed mean/SEM summaries and detector q values support illustrations, not a claim of physiological ground truth. Source- and site-eligible coverage is 95.33%; the expanded missing-source feasibility denominator gives 81.12%.
