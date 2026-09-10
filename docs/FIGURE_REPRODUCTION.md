# Reproducing manuscript figures

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

The figure scripts draw saved results. They do not rerun detection or change
performance estimates. Run the commands below from the repository root using
Python 3.12 and `validation/requirements-figures.txt`.

The repository includes the inputs for Figures 1, 3, 5 and 6:

| Figure | Input and scope |
|---|---|
| 1 | Workflow diagram in `plot_manuscript_validation_figures.py`; no recordings. |
| 3 | `figure_data/qc_aggregate_summary.json`: selected figure-level QC counts and contact-family accounting. |
| 5 | `figure_data/synthetic_benchmark/`: saved simulated-family decisions and summaries; `stratified_recovery.json` supplies the within-design-cell bootstrap intervals. |
| 6 | `frozen_results/erdetect/metrics.csv`: saved public-dataset scoring aggregates. |

`validation/figure_data/source_manifest.json` records the packaged figure-data
hashes. The Figure 5 script checks its recorded source hashes before
export. Its confidence intervals can be regenerated without signals:

```bash
python validation/reanalyze_factorial_recovery.py \
  validation/figure_data/synthetic_benchmark/scenario_results.csv.gz \
  /tmp/recovery_check.json
```

Generate the standard journal figures in a chosen output directory:

```bash
python validation/plot_manuscript_validation_figures.py --output-dir outputs/figures
python validation/plot_qc_aggregate.py --output outputs/figures/fig03_artifact_qc
python validation/plot_erdetect_validation.py \
  --metrics validation/frozen_results/erdetect/metrics.csv \
  --output outputs/figures/fig06_external_validation
```

The journal exporters default to the four reproducible figures above and use
packaged inputs; neither executes a private editorial script:

```bash
python validation/export_frontiers_figures.py --base-dir outputs/figures
python validation/export_neuroinformatics_figures.py --base-dir outputs/figures
```

Frontiers exports use at least 2-point visible strokes. Full-size plain labels
replace reduced mathematical glyphs in its Figures 1 and 4. Neuroinformatics
exports use lowercase panel letters. The chosen width remains 180 mm; the
Neuroinformatics guide's production-format-dependent width options must be
confirmed before any later resizing. All exports include PDF, SVG, PNG and
600-dpi RGB TIFF; captions/alt text are generated for Figure 6. The standard
Figure 1/5 script additionally writes captions, alt text and a source manifest.

## Saved display inputs for Figures 2 and 4

Figures 2 and 4 are explicitly scoped to restyling the saved manuscript
figure data. They cannot be regenerated from the included aggregate counts.
They include previously plotted clinical waveforms, a trial image and
contact-level summaries; their original raw clinical recording and epochs are
not redistributed in this repository. A saved display SVG is a derived
publication figure, not a replacement for raw cohort data or a detector replay.

To include these figures, obtain the authorized manuscript SVGs and place
`fig02_selected_local_example.svg` and
`fig04_crp_energy_local_evidence.svg` in the directory passed to `--base-dir`.
Then explicitly request all figures:

```bash
python validation/export_frontiers_figures.py --base-dir outputs/figures --figures 1 2 3 4 5 6
python validation/export_neuroinformatics_figures.py --base-dir outputs/figures --figures 1 2 3 4 5 6
```

The scripts fail clearly if a requested saved SVG is absent. They preserve its
data paths, positions and embedded image. Frontiers adjusts strokes, removes
two obscuring white star outlines and uses equivalent plain mathematical
labels; Neuroinformatics changes panel-letter case. These are display edits,
not recovered raw-data analyses. Both exporters normalize the legacy Figure 4
SVG metadata title to "Projection–energy"; the stable file stem and source
detector key retain their historical spelling. Figure 3 uses only packaged aggregate counts;
no private participant/contact locator is included in its input JSON.

## Figure 6 coverage and display checks

The two sensitivity axes retain their 100% domain and grid boundary while
omitting that boundary's tick label, preventing confusion with an adjacent
specificity-axis tick. Neither the plotted estimates nor their intervals or
axis limits change. `--lowercase-panels` and `--frontiers` select the native
journal variants without modifying the source metrics.

Coverage of 32,121/33,694 (95.33%) is conditional on source and site eligibility
in the 13-participant independent cohort. The source-availability audit adds
5,901 otherwise eligible labeled records without matching source events;
expanded coverage is 32,121/39,595 (81.12%). This feasibility denominator does
not turn unavailable records into observed negative detections or alter
measured sensitivity/specificity. The audit source is
`validation/frozen_results/erdetect/coverage_audit/annotation_source_flow.json`.

Before submission, render the exported PDFs at their intended physical size
and inspect labels, symbols, legends and edges. The code records font/extent
checks, but a completed export alone is not a visual-quality certification.
The recorded local export verification is
`validation/figure_data/export_validation.json`; it includes the generator
hashes, exact Figure 6 plotted-data equality and the completed visual checks.
