# Label-free negative-N1 detection

`ERPy.n1_detection` provides a fixed negative-N1 detector for cortical-surface
ECoG. It requires no human labels, fitted coefficients, threshold search or
model file. It combines a preset negative-peak criterion with ERPy's early
projection–energy conjunction. The general polarity-invariant detector remains
available with its existing defaults. The [supervised N1 models](N1_DEVELOPMENT.md)
are separate secondary development comparisons.

## Fixed method

The strongest strictly interior negative local peak must occur in the native
rounded half-open [10,90) ms feature window and satisfy

$$A\geq3.4\max(s,50\ \mu\mathrm V),$$

where A is the positive magnitude of the negative mean-waveform peak, and s is
the sample SD (ddof=1) of its mean-waveform baseline over [−1000,−100) ms.
Each trial is centered by its median over [−500,−20) ms before averaging.
These feature windows use rounded sample indices and half-open slices. At
least eight trials must be finite over all required feature samples. No
smoothing, extra prominence cutoff or label-dependent peak search is applied.
The absolute minimum amplitude is therefore 170 µV.

The two existing tests use the fixed 10–90 ms response and −1000 to −100 ms
candidate baseline, with 0–9 ms artifact exclusion. Inference retains its
existing **inclusive comparisons on native sample times** and takes the latest
baseline samples matching the realized response sample count. These samples
need not be identical to the feature slices. Reproducibility and separately
demeaned RMS energy use their original randomization tests; the joint value is
`max(p_reproducibility, p_energy)`. There are 5,000 randomizations, with exact
caps of 12 projection and 16 energy trials. Actual windows, seeds and the two
finite-trial masks are recorded separately.

For every contact in the original finite inference family, replace joint p by
1 if the morphology gate fails, then apply BH at 0.05 to the **complete** vector.
Finite contacts without available morphology remain in this family with p=1
and a separate unavailable flag. Nonfinite original inference remains
unavailable. Include all retained non-stimulation contacts, even those without
annotations or outside subsequent scoring eligibility rules. Do not correct
only the contacts that pass the gate or have human labels.

The sole declared ablation removes the 50 µV floor while retaining every other
setting and the complete family. Both results are returned; the primary is
fixed in advance and is not selected after evaluating its performance.

## Use on recorded data

```python
from ERPy.n1_detection import detect_n1_family

# Complete retained good-ECoG family for one stimulation pair.
# trials_uv has dimensions trial × time × channel; time_seconds is uniform.
result = detect_n1_family(
    trials_uv, time_seconds, channel_names,
    stimulation_contacts=["A1", "A2"],
    family_id="recording/stimulation-pair",
    random_seeds=channel_seeds,  # One exact nonnegative integer per input channel.
)
for contact in result["contacts"]:
    print(contact["channel"], contact["available"],
          contact["detected"], contact["q_screened"], contact["decision_stage"])
```

Input voltages must be microvolts (`voltage_unit="uV"`). The API computes both
paths from the same arrays and retains their hashes; arbitrary precomputed
inference dictionaries are not accepted. The caller supplies every retained
good ECoG contact in the acquisition family; the API cannot infer omitted
channels. Stimulation contacts are removed explicitly. If `random_seeds` is
omitted, the documented seed 42 is used for each input channel.

Each contact contains original component/joint p values, `original_q_joint`,
`p_screened`, `q_screened`, `morphology_gate`, `threshold_uv`, `available`,
`detected`, `decision_stage`, and detailed features/provenance. Fields prefixed
`ablation_` describe the fixed baseline-only ablation. A no-peak record with
available inference is an observed gate failure; an unavailable record must
not be described as evidence of physiological absence. The public
`morphology_gate` helper validates feature units, version and window-contract
hash and returns a descriptive gate only; family inference requires the
complete array API or the separately verified archived-record workflow.

## Evidence and limits

The 50 µV floor was hard-coded in [historical ER-detect source](https://github.com/MultimodalNeuroimagingLab/erdetect/blob/47f58d161537a735ebd400f9dfe3db57b65a17d4/erdetect/core/detection.py#L186-L193)
before becoming configurable. This is a literature/software-anchored threshold,
not an exact ER-detect reimplementation: the latter uses a different peak
finder, a 9–90 ms detection window, population baseline SD and a saturation
rejection. ERPy deliberately retains its existing feature definitions. No
GPL implementation is incorporated.

In the previously examined public ds004774 cohort, the fixed rule was evaluated
against **human N1 votes**, with total weight one per contact/pair record. On
the paired 32,048-record, 13-participant frame, sensitivity was 54.80%
(95% interval 45.81–62.83), specificity 98.50% (97.47–99.35), and PPV 84.60%
(80.51–90.22). Archived ER-detect had 67.61%, 97.64%, and 81.17%, respectively.
The paired sensitivity difference was −12.81 percentage points
(−25.07 to −3.76); specificity and PPV difference intervals included zero.
These results do not establish comparable sensitivity or superiority.
Intervals use 2,000 whole-participant resamples within site, conditional on
the saved predictions. The rule fits no labels, but this is a post hoc
fixed-rule evaluation on an already examined cohort.

An explicitly post-result diagnostic also describes the intermediate
`available and morphology_gate and p_joint <= 0.05` screen. It uses the same
fixed gate, with no BH selection. Its sensitivity was 65.05%, specificity 97.82%
and PPV 81.73%; all three paired difference intervals versus archived ER-detect
included zero. This does not establish equivalence. The screen is not an
FDR-adjusted alternative primary detector. The prespecified family-adjusted
decision above remains the primary result. Displaying both stages makes the
different per-record and family decision standards explicit: BH removed 620
screen-positive records, containing 429 human N1-positive record equivalents.
The separate `nominal_stage` evidence records this diagnostic's post-result
status and preserves the original fixed-rule assessment.

In the separately specified 480-family synthetic check, neither method called
any of 240 pure-noise families (Wilson 95% interval 0–1.58%). The primary called
51/120 early-negative targets and 1/120 residual-artifact targets; the
baseline-only ablation called 82/120 and 55/120. The 170 µV minimum necessarily
excluded the 100 µV target condition. Positive, late and artifact challenges
are not all nulls of the original projection–energy test; these finite
simulations do not establish general physiological specificity.

The [calibration tables](../validation/n1_label_free/calibration/) retain every injection-calibration cell. The saved pure-noise family interval remains in the retained summary and tables. Contact-level rates are descriptive because the two targets within each family share noise; they are not ten independent Bernoulli trials per cell.

The screened p value is never smaller than original joint p, so marginal
validity under the **original projection–energy union null** is retained when
that original p is valid. This does not make p/q an N1-truth probability or
establish N1-specific FDR. BH's dependence assumptions must apply to the
screened p vector; marginal validity alone does not prove those assumptions.
The amplitude/latency gate is an operational phenotype criterion. Positive,
late, artifactual or mixed physiology can still complicate interpretation.

## Inspect the retained public result

The [retained evidence](../validation/n1_label_free/README.md) contains the fixed
protocols, all family-level calls, human-reference scoring, decision-stage counts,
the sole ablation and secondary comparisons, plus synthetic calibration. All
577 original families and 34,926 finite early p values are retained; 50 finite
contacts have no available human rater and 639 are ineligible for later scoring,
but remain in BH. The CSV tables can be inspected directly.

Use the array API above for a new raw recording with its declared settings and
complete family. Notebook 08 demonstrates the secondary supervised model and is
not a label-free example.
