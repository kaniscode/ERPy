# Fixed label-free N1 evidence

This package preserves the acknowledged fixed rule, its sole baseline-only
ablation, complete-family predictions, separate human-reference scoring and
480-family synthetic calibration. The clinical cohort had already been examined;
this is post hoc development, with no new fitting or outcome-based threshold
selection. Saved supervised predictions remain secondary comparisons.

The primary gate requires an interior negative peak at least
`3.4 * max(baseline SD, 50 microvolts)`. It screens the existing early-window
projection–energy conjunction before BH adjustment over the complete original
finite contact family. The sole ablation removes the 50-microvolt SD floor.
The p/q interpretation concerns the original response conjunction, subject to
its assumptions; this is not a guarantee of physiological N1-specific FDR.

## Contents

- `protocols/`: byte-identical primary/calibration protocols and acknowledgment.
  Their original pending-status fields are preserved; the separate dated
  acknowledgment records acceptance before new calls.
- `empirical_predictions/`: 36,016 retained contacts in 577 families, including
  34,926 contacts with finite early inference. Prediction reads an explicit
  identity/inference/morphology column whitelist, without human votes or
  archived/model calls.
- `empirical_scoring/`: the separately reconstructed human-vote reference,
  32,048-record/13-participant paired analysis, all retained candidates, shared
  participant-bootstrap intervals, coverage and decision stages. Human votes
  define the reference; ER-detect and saved models are comparators.
- `nominal_stage/`: an explicitly post-result description of the already
  recorded gate-plus-unadjusted-joint-p stage. It is not FDR-adjusted and does
  not replace the acknowledged family-adjusted primary rule. Its three-method
  display tables label each role explicitly.
- `calibration/`: all 3,840 contact outcomes and morphology/provenance fields,
  480 family aggregates, the complete exact-seed execution plan and summaries.
  Pure-noise family any-call intervals use 240 families as the units. Positive,
  late and artifact injections are separate phenotype challenges.
- `provenance/`: pre-call family coverage and archived-rule provenance. These
  are historical records, not fresh validation or scientific approval.
- `artifact_manifest.json`: hashes of included files, execution sources and
  adjacent frozen inputs. Scientific evidence and publication/UI receipts are
  separate.

Calibration contact CSV bytes are preserved inside deterministic gzip. The
curation receipt records their original uncompressed hash and the original
internal run-manifest hash. The 480 per-family JSON files duplicate the outcomes
and seed information in these tables; they remain internally preserved and are
not listed as included public files. Per-family wall-clock durations are omitted.
Historical third-party code hashes identify provenance; no GPL source snapshot
is vendored here. ER-detect's source is available from its
[original repository](https://github.com/MultimodalNeuroimagingLab/erdetect).

## Reading the evidence

The retained protocols, per-contact and family tables, calibration aggregates
and human-reference scoring can be inspected directly. Their recorded hashes
identify the fixed inputs and executed sources. Synthetic arrays were generated
in microvolts from the specified protocol and require no human data. Historical
execution identities do not imply that author workflow scripts are distributed.
The recorded numerical environment is Python 3.12.5, NumPy 1.26.4, SciPy 1.13.1
and pandas 2.2.2. Use the documented library API for new recordings.
