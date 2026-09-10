# Frozen optional multiscale development evidence

**Result: retain ERPy's original single-window default.** Both the failed
equal-weight development bank and the single fixed-weight amendment are
preserved. The fresh confirmation did not establish an overall improvement.
See [the complete recorded decision](RESULTS_AND_DECISION.md) and the
[model and evidence guide](../../../docs/MULTISCALE_DEVELOPMENT.md).

| Location | Evidence |
| --- | --- |
| Root plans and result tables | Original 2,000-family equal-weight development; initial and amended pre-outcome plans remain exact. |
| `real_baseline` | 960 conditional semisynthetic families using two public prestimulus fixtures. |
| `weighted_amendment` | One declared amendment after equal-weight outcomes, before weighted outcomes: early/late/broad weights 0.1/0.1/0.8. |
| `weighted_development`, `weighted_real_baseline` | Recombination of retained window p values; no new trials or core inference. |
| `weighted_confirmation` | A fresh 2,000-family confirmation with disjoint seeds; all outcomes retained. |
| `fixtures` | Only two necessary, small CC0 prestimulus fixtures (10 native trials each); no full epoch caches or governed cohort signals. |

`artifact_manifest.json` binds the packaged evidence by byte size and SHA-256.
Plans, tables and fixtures retain their numerical contents. Historical source
hashes identify the original runs; only the current library implementation is
distributed. The two small fixture archives retain their numerical arrays,
dtypes and dimensions and the source identities in the fixture manifest.

The fixtures derive from [CC0 OpenNeuro ds004774 v1.0.0](https://doi.org/10.18112/openneuro.ds004774.v1.0.0).
Source attribution and license metadata are retained in the
[public reference folder](../../erdetect_reference/README.md); cite the dataset
and [Van den Boom et al., 2025](https://doi.org/10.1016/j.jneumeth.2025.110389).
No ER-detect program source is redistributed. ERPy code remains MIT licensed.
