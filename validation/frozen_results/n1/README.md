# Frozen negative-N1 development evidence

These tables and models derive from the CC0 public OpenNeuro **ds004774 v1.0.0**
recordings and expert annotations. They contain no governed cohort recordings
and no redistributed ER-detect program source. Dataset attribution and pinned source metadata
are in [the public reference folder](../../erdetect_reference/README.md).

This is **corrected post hoc development**, after the original external
evaluation had been examined. Participant-held-out development results are
not a new independent external validation. The original polarity-invariant
detector, p values, q values and frozen external results are unchanged.

| Folder | Contents and analysis scope |
| --- | --- |
| `features` | 36,016 contact/pair rows from 596 cached public epoch files; 34,928 feature-available. Includes MAYO01 extraction rows, which are excluded from primary development. Extraction source identities and the earlier protocol are recorded. |
| `nested` | 32,048 paired available records from 13 participants; all have features after identifier correction. Predictions, all candidate summaries, 26 participant-held-out logistic models, two development models, four site-transport models, selection receipts and source/protocol identities. |
| `secondary` | Frozen held-out models/rules applied without refitting to the 33,694 source- and site-eligible records (32,121 evaluable), plus paired hybrid-minus-morphology summaries. Processing no-call is an operational policy, not observed physiological absence. |

`artifact_manifest.json` binds the packaged evidence by size and SHA-256.
Historical source hashes identify the executed analysis, while protocol
revisions retain their original identities. The current library is the
distributed implementation.

`CORRECTION_NOTE.json` preserves the invalidation receipt for the preliminary
training run. Undirected stimulation-pair identifiers were not canonicalized
before the original feature join, falsely omitting 1,670 available records.
The complete nested and site-transport procedure was rerun after canonicalizing
pairs with the frozen scorer's rule and rejecting duplicate identities. Feature
definitions, hyperparameters, objective and cohort were retained. Preliminary
outcomes had been viewed before the correction. Invalid outputs are excluded
from this public evidence bundle; their original local audit copy is preserved.

See [N1 development: model, API and evidence](../../../docs/N1_DEVELOPMENT.md)
and the executed [public record example](../../../notebooks/examples/08_n1_development.ipynb).
Development models fit all 13 participants and must not be used to claim
held-out performance on them; the example uses a participant-held-out model.
