# Secondary supervised negative-N1 comparison

These retained tables compare archived ER-detect, N1 morphology and N1 hybrid on the same 32,048 records from 13 participants as secondary development evidence. The fixed label-free detector and its nominal diagnostic stage are documented in `../n1_label_free/nominal_stage/`.

`data/operating_points.csv` has nine operating-point estimates and their saved intervals. `data/paired_differences.csv` has six hybrid-minus-comparator differences in proportion units; `paired_intervals_percentage_points.csv` gives the same values in percentage points. The two precision–recall files are byte-identical copies of the saved curves. `source_manifest.json` identifies the original result files and exact table hashes.

The paired intervals are nominal conditional participant-cluster bootstrap intervals for fixed participant-held-out development predictions. They omit fitting uncertainty and are not multiplicity-adjusted. All three contrasts with archived ER-detect include zero. Against morphology, specificity and PPV exclude zero. These results do not establish external validity, clinical superiority, equivalence or noninferiority.
