# N1 morphology optimization: development protocol

Recorded 2026-09-08 UTC, before extracting the new morphology features or evaluating candidate N1 rules/models. This is a post hoc development extension requested after the original external-validation results were examined. The original detector settings, predictions, coverage audit and external comparison remain frozen.

## Target and evidence boundary

The target is agreement with a uniformly selected available expert's negative N1 annotation for a labeled electrode/stimulation pair. It is not proof of connectivity or clinical utility. The general projection–energy conjunction continues to report its existing p values and family-adjusted q values. A separate N1 phenotype decision must not inherit their inferential or FDR guarantees.

Development excludes MAYO01, which overlaps the earlier worked example. All channels, stimulation pairs and raters from one participant stay in the same fold. The primary comparison retains the original paired available-case frame (13 participants, 32,048 records); new feature unavailability is reported and treated as no call on that frame. A common-feature intersection is secondary. No missing source records are invented or silently dropped. Original source-coverage tables remain applicable.

Because aggregate results and benchmark labels have already informed this extension, patient-held-out predictions are cross-validation of development, not a new untouched external validation. Results on an additional independent cohort would still be needed to establish transportability of a selected model.

## Candidates fixed before evaluation

1. Frozen early ERPy and archived negative-peak ER-detect calls, reproduced on identical records.
2. A transparent N1 morphology rule using a negative local peak, its amplitude relative to mean-waveform baseline noise, prominence, duration and trial consistency. Evaluate a small declared threshold grid, with all selection confined to training participants. Include a fixed literature-motivated 3.4-baseline-SD amplitude rule as an ablation, explicitly a new implementation rather than an ER-detect rerun.
3. A regularized logistic model of predefined N1 morphology features, with fixed regularization unless a subsequent recorded protocol amendment is justified before viewing its outer-fold scores.
4. The same model augmented with the frozen general detector's reproducibility, energy and RMS features. This tests whether ERPy's trial information improves morphology classification. Archived calls, expert labels, participant/site identity, electrode names and coordinates are never prediction features.

Feature extraction reads epochs only and never reads annotations or archived predictions. It records exact inputs, code and settings. Peak morphology is descriptive: no selected-window p value is presented as fixed-window inference. Changes to feature definitions after an outcome is viewed require an explicit amendment and retain the prior results.

## Selection and evaluation

For each outer held-out participant, fit all data transformations and model coefficients on the other participants only. Use participant-separated inner out-of-fold predictions to choose an operating threshold. Before outcome evaluation, statistical design review strengthened the operating objective to maximum PPV while meeting BOTH archived sensitivity and specificity on the inner records. On a fixed reference prevalence, meeting both also meets comparator PPV. If both constraints are infeasible, maximize PPV among candidates meeting sensitivity and record the specificity failure; if sensitivity itself is unreachable, maximize sensitivity then PPV. Remaining ties prefer specificity, sensitivity and finally the earliest declared rule/highest tied threshold. No outer outcomes enter threshold selection. Models use soft expert-vote targets, with total weight one per record rather than one per rater; no outcome oversampling.

The exact rule grid is amplitude in baseline SD units {2,3.4,5,7}, prominence in baseline SD units {0,1,2}, minimum half-prominence width {0,4} ms, minimum negative-trial fraction {0,0.8}, absolute amplitude floor {0,30,60,100} microvolts and a separate requirement for the frozen IUT/BH call {false,true}: 384 combinations, split into two prespecified 192-rule families. Each rule family's selection uses training participants only and is evaluated on its held-out participant. Logistic models use fixed C=1 L2 regularization, train-only feature standardization and the fixed morphology-only or morphology-plus-inference feature schema. There is no model selection using outer outcomes. The runnable protocol saves this grid and exact input/code hashes before reading outcome data.

The transparent rules explicitly require an interior negative peak. The learned models classify expert N1 annotation from the complete predefined feature vector; they do not add a separate hard peak gate. An absent interior negative peak is represented by the declared zero peak features and remains an observed record. This distinction is fixed before model evaluation and will be described in the report. The proportion of negative trials around the pooled-mean-selected peak is a descriptive consistency feature, not held-out trial validation; the cosine feature uses leave-one-trial-out mean templates.

Report sensitivity, specificity, PPV, NPV, balanced accuracy, kappa, precision–recall curves, prevalence, coverage and call counts for all declared candidates. Report paired differences against both original ERPy and archived ER-detect, participant-level variation and whole-participant resampling within site. Bootstrap intervals based on fixed cross-validated predictions are conditional on the fitted procedure and do not capture every source of training/model-selection uncertainty; no unqualified equivalence or noninferiority claim follows from overlapping intervals.

Use Mayo-to-UMCU and UMCU-to-Mayo training/transport experiments as additional checks, with threshold selection confined to the training site. These small and imbalanced site samples are descriptive. Report all candidates rather than selecting the best method on outer-fold results and calling that selection independently validated.

## Release

Expose morphology scores and decisions separately from general response inference. Save model coefficients, feature schema, training provenance, threshold policy and intended population. Keep original examples and validation findings intact; any selected model is explicitly a development model. All changes, results, figures and manuscript claims receive subsequent fresh independent adversarial review. Publication remains local and under the user's account.
