# Frozen optional multiscale development evidence

**Result: retain ERPy's original single-window default.** Both the failed
equal-weight development bank and the single fixed-weight amendment are
preserved. The fresh confirmation did not establish an overall improvement.
See [the complete recorded decision](RESULTS_AND_DECISION.md) and the
[model and reproduction guide](../../../docs/MULTISCALE_DEVELOPMENT.md).

| Location | Evidence |
| --- | --- |
| Root plans and result tables | Original 2,000-family equal-weight development; initial and amended pre-outcome plans remain exact. |
| `real_baseline` | 960 conditional semisynthetic families using two public prestimulus fixtures. |
| `weighted_amendment` | One declared amendment after equal-weight outcomes, before weighted outcomes: early/late/broad weights 0.1/0.1/0.8. |
| `weighted_development`, `weighted_real_baseline` | Recombination of retained window p values; no new trials or core inference. |
| `weighted_confirmation` | A fresh 2,000-family confirmation with disjoint seeds; all outcomes retained. |
| `execution_code`, `weighted_execution_code` | Exact original equal-weight and final weighted execution source snapshots. Shared helper/test snapshots are identified separately in the packaging receipts. |
| `fixtures` | Only two necessary, small CC0 prestimulus fixtures (10 native trials each); no full epoch caches or governed cohort signals. |

`artifact_manifest.json` binds every public file by byte size and SHA-256.
`PACKAGING_PROVENANCE.json` records original and public hashes for each imported
artifact. Plans, tables and scientific implementation snapshots are exact
copies. Local path strings in selected JSON records were replaced with portable
identities. The two fixture NPZs have only their metadata path text changed;
their numerical arrays, dtypes and dimensions are unchanged and individually
hashed. Their original archive hashes remain in the public fixture manifest.
The two support scripts have one local `sys.path` insertion removed; their
scientific statements are unchanged.

Historical manifests retain their original referenced hashes. Where public
metadata differs, its original hash is checked against the packaging receipt,
not falsely asserted to equal the transformed public file. The first manifests
also recorded growing progress logs and desktop metadata: those unnecessary
files are omitted, and the recorded caveat is retained. The public artifact
manifest is the complete verification contract for this export.

The fixtures derive from [CC0 OpenNeuro ds004774 v1.0.0](https://doi.org/10.18112/openneuro.ds004774.v1.0.0).
Source attribution and license metadata are retained in the
[public reference folder](../../erdetect_reference/README.md); cite the dataset
and [Van den Boom et al., 2025](https://doi.org/10.1016/j.jneumeth.2025.110389).
No ER-detect program source is redistributed. ERPy code remains MIT licensed.

From the repository root:

```bash
python validation/verify_multiscale_artifacts.py
python validation/replay_multiscale.py --output validation/work/multiscale_replay --workers 3
```

The replay uses an isolated copy of the source snapshots and a new output
folder. Frozen artifacts, the installed default and original working sources
remain unchanged. Runtime/timestamps and local provenance paths differ in a
replay; compare scientific rows by scenario/method/contact identity.
