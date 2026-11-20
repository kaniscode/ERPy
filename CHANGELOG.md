# Changes

## Unreleased

Code organization and interface cleanup. Numerical results, cache file names,
stored metadata and the compatibility checkpoint are unchanged, and every
earlier public name remains available.

- Fix `Patient.epoch()`/`Patient.analyze()` failing with "An epoch input changed during content verification" on cloud-synchronized folders, where the first read of a placeholder file changes its status. An unstable read is retried before the input is rejected.
- Shorter names: `Epochs.detect()`, `Epochs.detect_all()`, `Epochs.mean()`/`std()`/`sem()`, `Epochs.shape`, `Recording.detect_events()`, `Patient.session_ids`, `Pipeline.resolve_pipeline()`, and snake_case `DataLoader` attributes (`rawdata_path`, `all_raw_meta`, ...). See the table in [docs/README.md](docs/README.md#names-changed-in-11).
- Readable `repr` for `Patient`, `DataLoader`, `Epochs` and `AnalysisResult`.
- The standard preprocessing pipelines are listed in `ERPy.STANDARD_PIPELINES`.
- Internal restructuring: metadata storage (`_metadata.py`), analysis results (`results.py`), shared defaults (`defaults.py`), single-sourced version (`_version.py`); consistent formatting and linting with ruff.
- Updating the package invalidates automatic epoch-cache reuse once: the first `epoch()`/`analyze()` call per acquisition recomputes its epochs.
- The README is now a short overview; its detailed sections moved to guides under `docs/` (installation, workflow, data formats, events, preprocessing and QC, detection, spectral, visualization, networks).

## 1.1.0rc1 — release candidate

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


## 1.0.0

The preceding public release is retained in Git history. It does not contain all current numerical corrections or the development modes above.
