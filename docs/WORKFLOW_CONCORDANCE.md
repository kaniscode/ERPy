# Workflow concordance follow-up

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

This follow-up fixes two demonstrated availability/QC failures and checks
whether the earlier primary-inference corrections change the saved synthetic
benchmark. Original release artifacts were read without being replaced.

## Corrected behavior

`waveform_qc.audit_waveforms` now requires finite stored and recomputed peak
amplitude and latency before accepting feature agreement. Previously, a
comparison with `NaN` returned false; the subsequent `fillna(True)` did not
repair that Boolean result. An infinite recomputed amplitude also produced an
infinite tolerance. Such features now set `feature_mismatch`, exclude the row
from `analysis_eligible`, and retain the existing
`recomputed_metric_mismatch` review reason. Tolerances for finite values are
unchanged.

`erp_detection._detect_signi` now returns the existing unavailable-method
record when SIGNI calculation raises an exception. The record includes the
failure reason, missing method-specific quantities, and
`method_available=False`. It is excluded from the available-method count.
Previously, the default record marked the method available with a negative
call, making a failed calculation look like an evaluated negative result.
Successful SIGNI calculations are unchanged. This correction does not extend
the existing sample-count coverage check into a full-window endpoint check.

## Public-interface regression

An eight-trial, two-contact example isolates the earlier correction separating
primary inference from descriptive-model availability. Both contacts have a
zero early response prefix. One has a coherent nonzero late response; the
other has the same response with four positive and four negative trial signs.
The latter has a valid full-window projection test with `p_R=1`, although its
selected-duration descriptive fit is degenerate.

Replacing only the primary array wrapper with its pre-correction source gives
the following before/after result through `Epochs.detect_erp_all`:

| Quantity | Before | Corrected |
| --- | ---: | ---: |
| BH family size | 1 | 2 |
| Balanced-contact joint p-value | missing | 1 |
| Balanced-contact QC status | `crp_unavailable` | `pass` |
| Coherent-contact adjusted p-value | 0.0078125 | 0.015625 |

The coherent contact is called in both cases; the balanced contact is called
in neither. Missing descriptive quantities remain missing. This demonstrates
a real effect on another contact's adjusted value when an otherwise valid
contact was previously discarded.

## Saved benchmark replay

The revised public primary API was run on every saved factorial scenario
using the original configuration, generator, exact channel labels, and seeds.
The replay contained **2,400 families and 9,600 contact rows**:

- All calls, evaluability decisions, retained trial counts, and root/effective
  seeds matched the archive.
- Every primary raw and adjusted p-value matched within `1e-14`; the maximum
  absolute difference was `1.11e-16`, consistent with CSV round-trip rounding.
- All 600 exclusions had fewer than eight retained trials. All 9,000 evaluable
  contacts had finite descriptive durations; the newly corrected descriptive
  failure path was absent from this benchmark.
- The primary counts remained 1,205 detections among 1,440 evaluable injected
  targets, zero families with a call among 480 all-null families, and two
  families with a false reference call among 1,920 mixed families.

The **2,000-family exact composite-null benchmark** was also fully replayed.
Every archived field matched, including component, joint, and adjusted
p-values with zero numerical difference. The projection-alternative,
energy-null arm retained 19 adjusted target calls and 19 families with any
adjusted call per 1,000 families. The energy-alternative, projection-null arm
retained 10 adjusted target calls and 11 families with any adjusted call per
1,000 families.

These primary benchmark counts therefore do **not** need replacement because
of the audited changes. The replay did not rerun the exploratory comparators,
clinical recordings, waveform QC, or a successful/failed clinical SIGNI run.
Those outputs retain their original provenance and are not certified by this
primary benchmark comparison. Software concordance does not establish
exchangeability after QC or BH validity under arbitrary dependence.

Saved input checksums (SHA-256):

| Input | SHA-256 |
| --- | --- |
| Factorial configuration | `81972f5396d3120c31de477aa0c22ed6ffdcd369d22645153ae7ae7b28836c9d` |
| Factorial detector table | `de3ea035a335991e8dfe951415654958a194eaba716587e24d4038e916b3f1d8` |
| Composite-null configuration | `9e35607b87bf8f82489f6700d3d9420a06483dd1a4d75b7cc0861b413268a11c` |
| Composite-null result table | `e4e0965bcf189a73c5a5d64bf67c6fc30f9a08f761631ba5bc6b2933620e7420` |

## Verification

The new workflow test file has 16 parameterized cases. Before these two fixes,
nine cases failed and seven passed. All 16 pass after the fixes. Coverage
includes stored/recomputed amplitude and latency with `NaN` or either infinity,
SIGNI sampling-rate/coverage/filter failures through the public interface, and
the descriptive-failure BH-family example above. Existing waveform tests
retain the finite-feature agreement and successful eligibility checks.

The following combined check passed **86 tests** under Python 3.12.5,
NumPy 1.26.4, and SciPy 1.13.1:

```text
python -m pytest tests/test_workflow_concordance.py tests/test_waveform_qc.py \
  tests/test_equation_contract.py tests/test_inference.py \
  tests/test_published_detectors.py tests/test_synthetic_benchmark.py \
  tests/test_crp_energy_validation.py
```
