# Changes

## 1.1.0rc1 — analysis release candidate (prepared locally)

- Distinguish measured N1 timing from the unchanged model's midpoint imputation in current feature and prediction exports.
- Supplement the frozen multiscale confirmation with exact empirical-bootstrap quantiles, removing Monte Carlo endpoint variation without changing predictions or the default-detector decision.
- Provide revised journal-sized validation figures showing both source-coverage denominators and the morphology-model operating point beside the hybrid.
- Complete an all-channel public acquisition demonstration with separate BH, contact-quality and final decisions, exact trial membership and sampling provenance; preserve the earlier subset example.
- Correct fixed-window projection/energy inference and retained family/QC accounting; preserve original benchmark provenance and document the matching validation results.
- Return unavailable inference when an epoch does not cover the declared effective response window within half a sample; retain the requested window and skip testing instead of silently shortening it.
- Preserve realized response/baseline endpoints, sample counts and clean-trial indices in new external prediction exports. Reject older cache schemas instead of presenting historical compact exports as complete provenance records.
- Validate automatic epoch reuse against current events, source contents, effective metadata/settings and implementation identity. Legacy or unverifiable automatic caches are recomputed.
- Add an optional, explicitly developmental negative-N1 classifier with participant-separated evaluation, portable model parameters and a distinct interpretation from general response significance.
- Add experimental multiscale projection–energy variants and report their power costs. The original general detector remains the default; the tested variants did not improve overall recovery in confirmation.
- Provide executed real-cohort/public-data examples, pinned public source identities, external-validation reproduction artifacts, and publication figure recipes.

This candidate has not been published by the assistant. The final public commit and any archived release identifier must be recorded by the author. Historical outputs retain the implementation identities under which they were generated; the version number alone does not identify those immutable artifacts.

## 1.0.0

The preceding public release is retained in Git history. It does not contain all analysis corrections or the development modes above.
