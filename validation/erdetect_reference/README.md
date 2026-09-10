# Public reference inputs and provenance

The reference inputs come from **Automatic Evoked Response Detection
(ER-Detect) dataset**, OpenNeuro **ds004774 version 1.0.0**, DOI
[10.18112/openneuro.ds004774.v1.0.0](https://doi.org/10.18112/openneuro.ds004774.v1.0.0).
The pinned metadata Git commit is
`582149a14a29c29f765d71ed61ba376047cd6686`. Its preserved
`source_metadata/dataset/dataset_description.json` declares **CC0** and retains
the creators, acknowledgments, funding and source ethics statement. Cite the
dataset and [Van den Boom et al.'s ER-detect paper](https://doi.org/10.1016/j.jneumeth.2025.110389)
when using these inputs.

This folder contains released dataset metadata and annotation matrices plus
derived inventory, overlap and reconstruction receipts. Recordings and the
large archived waveform/comparator MAT files are retrieved separately by the
documented scripts. Public source files are retained as bytes; the derived
UMCU completion manifest uses portable relative paths. The complete file
checksum manifest is `artifact_manifest.json`, with paths relative to the
parent `validation` folder. It also covers the packaged frozen evaluation
outputs. `prepare_erdetect_workspace.py` verifies the manifest before creating
a rerun workspace.

The validation-folder Git attributes preserve evidence bytes across platform
line-ending settings. Do not normalize whitespace in checksum-pinned source
files or frozen outputs. The release audit permits one 12.65 MiB compressed
rater-join table only when its exact path, byte size and SHA-256 match the
declared public-artifact exception; other large files still fail.

The ERPy validation scripts independently read these released inputs; this
folder does not redistribute ER-detect program source. The recorded historical
comparison uses the actual archived negative-N1 calls, rather than executing
or approximating a new ER-detect package run. ER-detect's broader capabilities
are described in its paper.

No governed CNS/ACC–PAG raw recordings are included here. The lead notebook's
external governed example has separate access requirements and is not part of
this public evaluation cohort.
