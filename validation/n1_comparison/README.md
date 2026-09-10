# Secondary supervised negative-N1 comparison

These retained tables compare archived ER-detect, N1 morphology and N1 hybrid on the same32,048 records from13 participants. They support the earlier supervised development display, preserved as secondary evidence. Current primary Figure7 instead uses the fixed label-free detector and its nominal diagnostic stage in `../n1_label_free/nominal_stage/`. No fitted model, threshold or saved supervised outcome changed.

`data/operating_points.csv` has nine operating-point estimates and their saved intervals. `data/paired_differences.csv` has six hybrid-minus-comparator differences in proportion units; `paired_intervals_percentage_points.csv` gives the same values in percentage points. The two precision–recall files are byte-identical copies of the saved curves. `source_manifest.json` identifies the original result files and exact table hashes.

The paired intervals are nominal conditional participant-cluster bootstrap intervals for fixed participant-held-out development predictions. They omit fitting uncertainty and are not multiplicity-adjusted. Filled plot markers mean that the saved interval excludes zero; open markers mean that it includes zero. All three contrasts with archived ER-detect include zero. Against morphology, specificity and PPV exclude zero. These results do not establish external validity, clinical superiority, equivalence or noninferiority.

Run from the repository root:

```bash
python validation/plot_n1_comparison.py --output-dir /path/to/new-n1-figures
```

Use `--variant jnm`, `--variant frontiers` or `--variant neuroinformatics` for one journal. This retained secondary recipe requires matplotlib, numpy, pandas and Pillow and checks the pinned input tables before rendering. The [current manuscript figure directory](../manuscript_figures/README.md) documents the label-free primary display. Earlier supervised artwork is preserved in the prior Git revision and `frozen_results`.
