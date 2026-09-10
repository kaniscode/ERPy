# Complete public acquisition workflow

This is a new current-source demonstration of the complete contact-decision contract on public OpenNeuro **ds003708, sub-01 / ses-ieeg01, stimulation LTG1–LTG2**. It accounts for all **89 header channels and all 17 good events** for that pair. EKG is explicitly excluded as an auxiliary channel; stimulation contacts are excluded from the testing family. The prior 32-channel public example and every historical result remain unchanged. The expanded family is a separate demonstration, not a reinterpretation of its old counts.

`contact_decisions.csv` reports every source channel, preprocessing/availability disposition, finite BH-family membership, BH rejection, separate contact-gate availability and outcome, and the final intersection. Empty gate values mean no gate result was available for that source channel; they are not treated as observed clean responses. `summary.json` gives the cardinalities. `acquisition_detections_qc.csv` and its companion native exports retain the underlying component statistics, trial/artifact summaries, boundary flags and deterministic channel-specific random seeds. `contact_clean_trials.csv` binds the detector's own clean-trial indices to original event rows and source samples; `epoch_sample_grid.csv` retains the exact realized times. The source header is 2,048 Hz; the native timestamp-derived epoch rate is also reported explicitly. Independent verification reconstructed BH and the gate, requiring exact agreement with the current `Recording.analyze` final decision. BH was not rerun after the gate.

The declared processing follows the existing public recipe: EKG exclusion; persistent-channel screening at robust z threshold 8; 4-ms artifact interpolation; 60-Hz harmonic notch; 0.5–80-Hz third-order bandpass; −500 to 600-ms epochs; −500 to −30-ms baseline; and the current post-artifact common-anchor procedure. Trial-contact artifact rejection uses z threshold 3 and `nan_response`. The primary projection–energy detector uses its current defaults, including root seed 42 and 5,000 Monte Carlo permutations when exact enumeration is unavailable. Its requested 10–300-ms response window is clipped by its declared 0–15-ms artifact interval; the realized grid and effective parameters are exported. The contact gate requires bad-response fraction < 0.25, hard-artifact fraction < 0.10, at least 8 clean responses, and no excluded boundary peak. These are fixed workflow settings, not tuned from the new outputs.

This completes the software audit chain; it does not estimate clinical sensitivity/specificity or establish unconditional FDR control for a preprocessing- and QC-selected family. A single acquisition also does not validate calibration across dependent channels, natural artifacts, or other recordings.

## Retained public evidence

The source metadata, processing settings, contact decisions and native exports
are retained as fixed evidence. `provenance.json` binds the public source object
versions, range, input hashes, effective API settings and runtime; `manifest.json`
identifies the executed source and exported evidence. Source hashes are
historical execution identities. The public library supports new analyses via
`Recording.analyze` and the documented OpenNeuro recipe.

When reading the compact CSV in pandas, preserve seed fields as strings:
`pd.read_csv("contact_decisions.csv", dtype={"random_state_root": "string", "random_state_effective": "string"})`.
Blank seed fields identify excluded source channels.
