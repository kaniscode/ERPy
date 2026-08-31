# Changelog

## 1.0.0 - 2026-09-01

- Published the first stable, clinician-oriented ERPy release.
- Added response-level artifact quality control, post-artifact anchoring,
  spectral and connectivity analyses, travelling-wave inference, and the
  synthetically evaluated fixed-window reproducibility-energy conjunction.
- Added native ERPy CSV, BIDS/iEEG, and NWB input paths with auditable event,
  preprocessing, epoch, detection, and figure outputs.
- Made detector availability explicit: SIGNI/high gamma requires a declared
  passband-preserving input, and unavailable methods are excluded from legacy
  comparator counts rather than encoded as negative responses.
- Added exact stress tests for both one-component-null boundaries of the joint
  criterion, with family-level outputs, confidence intervals, and locked
  checksums.
- Counted the observed whole-trial sign assignment explicitly in exact
  reproducibility tests and added scale-aware tie handling, guaranteeing the
  finite exact p-value floor under floating-point arithmetic.
- Constructed the energy-null stress branch on the detector's exact
  epsilon-stabilized log-RMS scale and verified the intended paired differences
  during generation.
- Separately demeaned matched baseline and response segments for the
  exploratory paired-RMS comparison, and marked the selected-duration
  standalone CRP p-value as exploratory after null-calibration review.
- Added a complete public API reference, a Python-newcomer installation guide,
  troubleshooting guidance, and compact executable visualization notebooks.
- Separated private cohort-specific manifests and server utilities from the
  public package and added release-time privacy and metadata checks.
- Renamed the installable distribution to `erpy-neuro` because the unrelated
  `erpy` name is already registered on the Python Package Index. The Python
  import remains `import ERPy as ep`.

## Pre-release development

Versions below 1.0.0 were internal development snapshots rather than official
public releases. Their capabilities were consolidated, retested, documented,
and privacy-audited for 1.0.0; no support commitment is made for those snapshot
formats or APIs.
