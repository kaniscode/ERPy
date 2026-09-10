# Secondary supervised negative-N1 development classifiers

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

The proposed N1 method is the [fixed label-free detector](N1_LABEL_FREE.md).
The fitted models below are retained as secondary development comparisons;
they require labels when training, although deployment uses only signals and
saved model parameters. Notebook08 remains their executed public-data example.

`ERPy.n1` provides a separate annotation classifier for early **negative N1
responses in cortical-surface ECoG**. It is optional; default ERPy detection
remains polarity-invariant and its p/q values and decisions are unchanged.
A fitted N1 score is not a p value, q value, FDR estimate, proof of connectivity,
or validated sEEG interpretation.

The public development evidence comes from [OpenNeuro ds004774 v1.0.0](https://doi.org/10.18112/openneuro.ds004774.v1.0.0),
released under CC0, and the annotations accompanying
[Van den Boom et al., 2025](https://doi.org/10.1016/j.jneumeth.2025.110389).
The [original external evaluation](EXTERNAL_VALIDATION.md) preceded this
extension. Nested participant-held-out results therefore describe **post hoc
development cross-validation**, not an untouched external validation. An
additional independent cohort is needed to establish generalization.

## How this relates to ER-detect

ER-detect supports selectable response polarity and several methods, including
baseline-relative amplitude, CRP similarity across trials and waveform-based
detection. Our comparison uses its **archived negative-N1 calls** from the
released dataset; it does not rerun or retune ER-detect. See the
[ER-detect paper](https://doi.org/10.1016/j.jneumeth.2025.110389) and
[configuration reference](https://github.com/MultimodalNeuroimagingLab/erdetect/wiki/Configuration).

ERPy's hybrid is a separate logistic classifier. It adds three ERPy inference
features to the same morphology model, rather than using ER-detect's calls as
inputs. The paired results below describe the benefit of those added features
within this development cohort. They do not establish superiority on a new
cohort or validate an N1 interpretation of the default polarity-invariant test.

## Inputs, features and score

`extract_n1_features(trials, times)` accepts one contact's trial-by-time array
in **microvolts** and uniformly sampled times in seconds. The trained input
contract requires at least eight finite trials across the required windows:
10–90 ms response, −1 to −0.1 s baseline, and −0.5 to −0.02 s trial-median
normalization. Windows use rounded sample indices and half-open slicing;
realized endpoints are returned. Arbitrary settings or voltage units are not
interchangeable with the fitted models.

The 12-feature morphology arm includes negative peak amplitude, prominence,
latency, half-prominence width, negative area fraction, the fraction of negative
trials near the selected peak, leave-one-trial-out cosine consistency, preceding
positive amplitude, response RMS, baseline SD, absolute amplitude and trial
count, with the fixed transformations in `n1_feature_vector`. Noise-normalized
amplitudes use the baseline SD of the mean waveform. The negative-trial fraction
is descriptive and uses a peak chosen from the same trials. An absent interior
negative peak has declared zero peak features; the logistic model does not
impose a separate hard peak gate. Feature unavailability remains explicit.

The model input `peak_latency_ms` uses the response-window midpoint (50 ms)
when no interior negative peak exists. Current extraction and prediction
outputs separately provide `measured_peak_latency_ms`, which is `None` without
a measured peak, and `peak_latency_imputed`, which explicitly flags model
imputation. Report timing from the measured field, not the imputed model input.
This additional reporting metadata does not change feature vectors, saved
models, scores or calls; historical exports retain their original fields.
Among the 5,324 no-peak records in the paired development cohort, 76 receive
hybrid-positive calls with 15.5 positive reference weight (PPV 20.39%).
Classification in this subgroup does not provide a measured N1 latency.

The hybrid arm adds **inference features**: −log10 of the early-window reproducibility
and energy p values and the RMS ratio in dB. It does not add participant/site
identity, labels, coordinates or archived calls as predictors. Both arms
already contain trial-consistency descriptors. The feature schema, version,
window-contract hash and voltage unit are checked when building a feature
vector, loading a model and predicting.

For transformed features x, train-only mean μ and scale s, the portable model is

$$z=b+\sum_j\beta_j(x_j-\mu_j)/s_j,\qquad S=1/(1+e^{-z}).$$

The decision is `S >= model.threshold`. S is a logistic annotation score in
[0,1], trained against the probability of a uniformly selected available
expert's positive vote within a record. It is not established as calibrated
in new populations. Saved calibration tables and Brier summaries describe the
development predictions only.

## Fitting and held-out evaluation

The [recorded protocol](../validation/N1_OPTIMIZATION_PROTOCOL.md) fixes C=1
L2-regularized logistic regression, train-only standardization and the feature
schemas. Fractional positive/negative training weights sum to one per record,
so records with more raters receive no extra total weight. All records from a
participant remain together. For each outer leave-one-participant-out fold,
inner folds hold out one of the remaining participants to select the threshold.
It maximizes PPV while meeting archived sensitivity and specificity on inner
records. Declared fallbacks prioritize sensitivity when both constraints are
infeasible. Outer outcomes never select coefficients, scaling or thresholds.
The rule candidates, fixed-amplitude ablation and both logistic arms are all
reported; cross-site training/transport is secondary.

| Paired development records (32,048; 13 participants) | Sensitivity | Specificity | PPV |
| --- | ---: | ---: | ---: |
| Early-window ERPy | 77.41% | 89.57% | 52.72% |
| Archived negative-N1 ER-detect | 67.61% | 97.64% | 81.17% |
| Morphology logistic | 67.28% | 97.48% | 80.07% |
| Hybrid logistic | 67.39% | 97.96% | 83.25% |

This is an operating tradeoff on a fixed reference prevalence. It does not
establish superiority or equivalence. All candidates, paired differences,
participant counts, precision–recall curves, calibration and transport outputs
are retained in [the saved evidence](../validation/frozen_results/n1/README.md).
Intervals use 2,000 whole-participant resamples within site, conditional on the
fixed cross-validated predictions; they do not include all fitting uncertainty.
MAYO01 is excluded because it overlaps the earlier worked example. Original
source availability, eligibility and missing-record limitations still apply.

A preliminary join failed to canonicalize undirected stimulation pairs and
falsely omitted 1,670 available feature records. The entire procedure was
rerun after a deterministic identifier correction, with unchanged definitions
and hyperparameters. Its preliminary outcomes had been viewed; the
[correction receipt](../validation/frozen_results/n1/CORRECTION_NOTE.json)
records this explicitly. Only the corrected output set is packaged.

## Use a portable model

Portable runtime prediction uses ERPy's NumPy/SciPy dependencies and retains
the package's declared Python 3.9+ range; it does not need scikit-learn.
The extraction, training and secondary-summary scripts require **Python 3.11+**
because they use `hashlib.file_digest`; this workflow was validated on Python
3.12.5. Training and the training-script-based secondary summaries additionally
need scikit-learn.
The executed [notebook 08](../notebooks/examples/08_n1_development.ipynb) verifies
input/model hashes, joins an actual public record by canonical identifiers and
epoch hash, checks participant exclusion, matches its saved held-out prediction,
and round-trips `save`/`load`. It needs no raw download or scikit-learn.

```python
from ERPy.n1 import N1MorphologyModel, prepare_n1_hybrid_inputs

# One contact: trial × time in microvolts, sample times in seconds.
model = N1MorphologyModel.load("model.json")
features, evidence = prepare_n1_hybrid_inputs(trials_uv, times_seconds,
                                             random_state=42)
result = model.predict(features, evidence)
model.save("model_copy.json")  # Retains weights, threshold and training provenance.
```

Hybrid models require `N1InferenceEvidence` constructed by this helper; bare
inference dictionaries are rejected. The trained inference configuration is a
10–90 ms response, −1000 to −100 ms candidate baseline, 0–9 ms artifact interval,
eight-trial minimum, 5,000 randomizations, and exact caps of 12 projection and
16 energy trials. Both paths use trial-median normalization over the rounded
half-open [−500, −20) ms slice. The helper binds the numerical model inputs to
the same array/time hashes and retains each method's actual clean-trial indices
and inference sample windows. The masks can differ because their declared
windows differ; equality of those masks is not required. A nondefault random
seed must be supplied explicitly and is retained without integer rounding.

For the immutable public example, `validation.n1_frozen_inputs.FrozenN1Inputs`
verifies the exact feature/inference table bytes and original resolved settings,
then verifies separate participant/pair/contact identities, epoch hashes,
recorded trial counts and exact seed strings before binding a selected row.
It also rejects broad-window values relabeled as early evidence. This archival
path does **not** reconstruct unarchived clean-trial indices: both index fields
remain `None`, and the provenance states the source-epoch/policy-level limit.
Use the array helper for new recordings. `n1_feature_vector` is only the fixed
numerical transformation used by training and contribution plots; a vector or
an arbitrary three-number dictionary is not a checked hybrid prediction input.
Morphology-only models can still consume `extract_n1_features` output directly.

Inspect `result['available']` before interpreting a decision. A result with
unavailable features is a no-call, not an observed physiological negative.
Files named `heldout_SUBJECT` excluded that participant; `transport_test_SITE`
excluded that site. Files named `development_model` fit all 13 participants and
are development artifacts for further testing, never held-out tests of those
participants. The API validates numerical contracts; the caller must also
check training provenance and applicability to the intended population.

## Inspect the retained evidence

The saved feature table, participant-held-out models, predictions and matched
annotation table are distributed with their source and protocol identities.
The model API above loads and applies these models; the executed notebook 08
shows an example using a participant-held-out model. Author fitting and replay
automation are not part of the library API. The saved run used Python 3.12.5,
NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.2 and scikit-learn 1.4.2.

Do not use a development model's training participants to claim held-out
performance. New populations require separate evaluation.
