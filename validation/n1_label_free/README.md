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
  are historical records, not fresh validation or manuscript approval.
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

## Verify and reproduce

Run from the repository root in an environment with the project dependencies.
Use one BLAS thread for the recorded deterministic replay:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python validation/verify_n1_label_free.py
```

The default verifier checks every included hash and source binding, regenerates
all 480 synthetic input arrays and exact seeds, reconstructs calibration
aggregates, and checks the unchanged archived/supervised metric rows. It then
replays label-blind prediction in a temporary directory, verifies its bytes,
and only afterwards invokes the separate human-reference scorer and compares
all deterministic CSV/JSON outputs. It does not reload clinical voltage
recordings or retrain models. The final replay step reproduces the separate
post-result nominal-stage description from completed human-reference scoring.
`--hashes-only` skips empirical replay while still
checking calibration arrays and aggregates. `--output receipt.json` saves a new
receipt outside this immutable evidence folder.

To additionally rerun all 480 families through the detector and compare the
complete contact/family/summary results:

```sh
python validation/verify_n1_label_free.py --full-calibration --workers 4
```

To retain a complete new calibration run, including its per-family JSON files:

```sh
python validation/calibrate_n1_label_free.py \
  --protocol validation/n1_label_free/protocols/SYNTHETIC_CALIBRATION_PROTOCOL.json \
  --primary-protocol validation/n1_label_free/protocols/LABEL_FREE_N1_PROTOCOL.json \
  --output-dir calibration_replay --workers 4
```

Prediction and scoring can also be run separately into new directories:

```sh
python validation/predict_n1_label_free.py \
  --protocol validation/n1_label_free/protocols/LABEL_FREE_N1_PROTOCOL.json \
  --output prediction_replay
python validation/score_n1_label_free.py \
  --protocol validation/n1_label_free/protocols/LABEL_FREE_N1_PROTOCOL.json \
  --predictions prediction_replay --output scoring_replay
python validation/summarize_n1_nominal_stage.py \
  --scoring scoring_replay --output nominal_stage_replay
```

The recorded numerical environment is Python 3.12.5, NumPy 1.26.4,
SciPy 1.13.1 and pandas 2.2.2. Exact-byte replay can reveal numerical or
serialization changes across dependency versions; it does not silently accept
different outputs. No source recordings or annotation downloads are needed
for the pinned-table replay when the repository's adjacent frozen inputs are
present. Synthetic arrays are generated in microvolts from the acknowledged
protocol and require no human data.
