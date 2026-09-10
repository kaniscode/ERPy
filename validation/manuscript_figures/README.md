# Current manuscript figures 6, 7 and S1

Figure 6 compares ERPy's broad 15–300 ms detector calls with released negative-N1 annotations. Its matrix gives fractional record-equivalent agreement counts, and its rate panel separates five annotation patterns, with nominal participant-stratified confidence intervals. Four recorded mean/SEM traces illustrate agreement and disagreement and retain their original detector q values. N1-negative ratings do not establish absence of all evoked responses. The underlying calls and labels are unchanged; the new descriptive agreement analysis and trace selection are explicit.

Figure 7 compares archived ER-detect with the fixed label-free N1 detector and its explicitly post-result nominal screening diagnostic on the same 32,048 records. The primary requires BH adjustment over the full original finite family; the nominal gate-plus-joint-p screen does not. Three operating-point panels and two paired-difference panels retain the primary sensitivity loss. Filled interval markers indicate exclusion of zero; the intervals are nominal and conditional, without a multiple-comparison claim. Inclusion of zero does not demonstrate equivalence. Supervised models and their earlier plotted evidence remain secondary and are preserved separately.

Figure S1 displays all 240 injection families from the fixed 480-family calibration, including all 48 phenotype/amplitude/noise/trial-count cells for the primary and sole no-floor ablation. Cell target-call rates are descriptive: each cell contains five independent families and ten correlated target contacts. The separate 240 pure-noise families and their saved family-level Wilson interval remain in the retained tables and summary. No new simulations, detector calls, tuning or outcome selection are introduced.

JNM and Frontiers artwork is 180 mm wide; Neuroinformatics artwork is 174 mm wide with lowercase panel labels. Ordinary text is at least 8 pt at the specified widths. Frontiers visible strokes are at least 2 pt. PNG and TIFF are 600 dpi; PDF/SVG retain vector content. Retain the native width when embedding. Captions, alt text, plotted tables and verification records accompany each figure.

## Reproduction

Install the scientific development dependencies, then run from the repository root:

```bash
python validation/analyze_annotation_agreement.py --verify
python validation/plot_annotation_agreement.py --analysis validation/annotation_agreement --output /path/to/new-figure6/jnm --journal jnm
python validation/verify_n1_label_free.py
python validation/plot_n1_label_free.py --data-dir validation/n1_label_free/nominal_stage --output-dir /path/to/new-figure7 --journal all
python validation/plot_n1_stress_test.py --data-dir validation/n1_label_free/calibration --output-dir /path/to/new-figureS1 --journal all
```

The Figure 6 recipe requires a new output directory. Repeat it with `--journal frontiers` or `--journal neuroinformatics` and a separate destination for that version. Figures 7 and S1 use the explicitly supplied verified data directory and produces one folder per selected journal. Rendering requires matplotlib, numpy, pandas and Pillow. A new renderer/environment may change font metrics or image bytes and requires a new visual check.

The [agreement guide](../../docs/ANNOTATION_AGREEMENT.md) explains endpoint limits and optional reconstruction from matching public NPZ caches. All four retained source-trace hashes match the original predictions. No private cohort recordings or GPL implementation is redistributed. `manifest.json` binds the current publication files and their exact source tables/recipes; visual receipts identify the individually inspected source artwork and verify exact copied bytes. Earlier artwork in `validation/frozen_results` and the prior Git revision remains intact.
