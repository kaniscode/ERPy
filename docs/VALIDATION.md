# ERPy Validation

ERPy validation is intentionally split into three tiers.

From a source checkout, install the test and public-data dependencies before
running this page's commands:

```bash
python -m pip install -e ".[dev,validation]"
```

The optional brain and surface figures additionally require the `viz` and
`surface` extras described in the visualization gallery.

## Tier 1: Unit And Fixture Tests

```bash
python -m pytest
```

The tests cover:

- timestamp and binary event conversion;
- stimulation-artifact event detection;
- bad-channel rejection;
- response-artifact rejection;
- preprocessing and epoching;
- HDF5 epoch caching;
- native BIDS/BrainVision import using a small synthetic BIDS fixture;
- native NWB import using a small PyNWB fixture when `pynwb` is installed.

## Tier 1.5: Reproducible Synthetic CRP-Energy Benchmark

Run the declared synthetic operating-characteristic benchmark from the
repository root:

```bash
python validation/benchmark_crp_energy.py
```

This benchmark asks a controlled question that real recordings cannot answer
by themselves: when the presence or absence of a response is known exactly,
how does ERPy's joint CRP-energy rule behave as trial count, signal strength,
and random target-trial masking change? It is a generated method test reported
separately from the public and governed real-data workflow examples.

The default design uses 40 independently seeded replicates for every
combination of trial count (8, 12, 24, or 36), nominal response-to-noise ratio
(0, 0.5, 1, 2, or 4), and random target-trial masking level (0%, 25%, or 50%). The
4 × 5 × 3 × 40 design therefore contains 2,400 simulated acquisitions.

Each simulated acquisition contains four channels processed together. In the
benchmark code and output tables, that four-channel group is called a
**testing family**. It represents the contacts considered for one
multiple-testing correction in one stimulation acquisition; it does not mean a
family of participants or a group of datasets. Only detector-eligible contacts
with finite joint p-values enter the Benjamini-Hochberg adjustment. The
adjusted set contains three contacts when random target-trial masking leaves the
target with too few usable trials and four contacts otherwise (600 and 1,800
of the 2,400 families, respectively). One channel is designated as the target,
and the other three are null references. At response-to-noise ratio 0, all
four simulated channels are known nulls. At nonzero ratios, only the target
receives a known response.

The generator begins with independently sampled stationary Gaussian AR(1)
colored noise. Trial amplitudes vary modestly. When a response is present, it
adds an RMS-normalized biphasic waveform to the target with small positive
amplitude variation and latency jitter. To represent trials already rejected
by upstream QC, the chosen fraction of target trials is set to missing in the
matched baseline and response windows. The same public detector API used for
real data then analyzes every simulated acquisition.

The synthetic tier measures observed calls, evaluability, and response
detection under this declared model. The public OpenNeuro tier demonstrates
ingestion and end-to-end operation on an independently hosted real recording.
A governed deidentified example can demonstrate the same workflow on local
data. The synthetic design uses stationary independent Gaussian AR(1) noise,
one biphasic morphology, four-channel families, and signal-independent random
target-trial removal. It therefore characterizes those conditions rather than
artifact-identification accuracy, larger-family multiplicity, or universal
detector performance.

### Declared CRP-Energy Rule

Let \(x_i\) be clean trial \(i\)'s baseline-mean-centered samples in the fixed,
predeclared response window and let \(f_s\) be the sampling frequency. For
\(i \ne j\), define the ordered semi-normalized cross-projection

$$
P_{ij}=\frac{x_i^\mathsf{T}x_j}{\lVert x_i\rVert_2\sqrt{f_s}},
\qquad P_{ii}=0,
$$

and the full-window reproducibility statistic

$$
T_R=\frac{1}{n(n-1)}\sum_{i\ne j}P_{ij}.
$$

For a trial-level sign vector \(s\), the randomized statistic is
\(T_R(s)=s^\mathsf{T}Ps/[n(n-1)]\). The one-sided \(p_R\) is the probability
that \(T_R(s)\ge T_R\). With at most 12 clean trials, ERPy enumerates all
\(2^{n-1}\) unique patterns after fixing one globally redundant sign. With
more trials, it draws 5,000 deterministic Rademacher patterns and uses the
standard plus-one Monte Carlo p-value. The response window is fixed before
inference; the CRP-selected duration is retained only as a descriptive output.

For matched baseline and response segments \(B_i\) and \(R_i\), define
separately demeaned segments \(\widetilde B_i=B_i-\overline B_i\) and
\(\widetilde R_i=R_i-\overline R_i\), then

$$
d_i=\log\{\operatorname{RMS}(\widetilde R_i)+\epsilon\}
   -\log\{\operatorname{RMS}(\widetilde B_i)+\epsilon\}.
$$

The one-sided \(p_E\) is a paired sign-flip test of
\(\operatorname{mean}(d_i)>0\), enumerated exactly through 16 clean trials and
otherwise evaluated with 5,000 deterministic draws and a plus-one p-value.
The intersection-union result is \(p_{\mathrm{joint}}=\max(p_R,p_E)\).
Benjamini-Hochberg adjustment is then applied across the detector-eligible
contacts with finite joint p-values in that four-channel simulation (three or
four contacts, as described above).
Calling that adjustment FDR-controlling additionally requires valid component
p-values and independent or PRDS-dependent null hypotheses; the benchmark's
channels are independent by construction.

### Frozen v1.0.0 Benchmark Results

The table below is the deterministic result of ERPy v1.0.0 compatibility
checkpoint `415a636abd5b9a28`. This fixed software-behavior label is shared by
all equivalent installations; it is not a release-record or dataset label.
Wilson intervals are used for binary rates. Intervals for mean null-call rates
and false-discovery proportions resample complete four-channel testing groups.

| Outcome | Estimate (count) | 95% interval |
| --- | ---: | ---: |
| All-null family: probability of any adjusted call | 0.000 (0/480) | 0.000–0.008 |
| Evaluable all-null target: adjusted call fraction | 0.000 (0/360) | 0.000–0.011 |
| Generated all-null target: unconditional adjusted call fraction | 0.000 (0/480) | 0.000–0.008 |
| Evaluable all-null contacts: adjusted call fraction | 0.000 (0/1,800) | Not estimated* |
| Generated all-null contacts: unconditional adjusted call fraction | 0.000 (0/1,920) | Not estimated* |
| Signal-present groups: null-reference call rate | 0.00035 (2/5,760) | 0.00000–0.00087 |
| Signal-present groups: probability of any null-reference call | 0.00104 (2/1,920) | 0.00029–0.00379 |
| Signal-present groups: mean false-discovery proportion | 0.00052 | 0.00000–0.00130 |
| Balanced-grid unconditional target recovery | 0.628 (1,205/1,920) | 0.606–0.649 |
| Target evaluability | 0.750 (1,440/1,920) | 0.730–0.769 |
| Recovery conditional on evaluability | 0.837 (1,205/1,440) | 0.817–0.855 |

The balanced-grid recovery rate averages equally over the declared signal and
masking grid and counts non-evaluable targets as negative. With no target-trial
masking, recovery was 0.400 at SNR 0.5, 0.944 at SNR 1, and 1.000 at SNR 2 and
4. Cell-level tables preserve the full trial-count, SNR, and masking-burden
dependence.

The unconditional generated-target and generated-contact denominators count a
non-evaluable hypothesis as having no call. The evaluable rows restrict the
denominator to hypotheses that actually entered testing. Family-level
any-call behavior is the primary calibration summary because multiplicity is
applied within each generated family.

\*A percentile bootstrap cannot infer unseen events from an all-zero sample.
The family-level any-call Wilson interval provides the primary uncertainty
summary for these clustered contact outcomes.

### Primary-component ablation

The target rows retain the primary detector's own component p-values, which
allows a direct ablation on the same evaluable targets:

| Target truth | Projection `p_R <= 0.05` | Energy `p_E <= 0.05` | Raw conjunction | BH-adjusted conjunction |
| --- | ---: | ---: | ---: | ---: |
| Noise only | 19/360 (5.3%) | 21/360 (5.8%) | 1/360 (0.28%) | 0/360 (0%) |
| Injected response | 1,423/1,440 (98.8%) | 1,302/1,440 (90.4%) | 1,298/1,440 (90.1%) | 1,205/1,440 (83.7%) |

This ablation shows the null-call–recovery trade-off of requiring both
components under the both-components-null and both-components-alternative
conditions represented here.

### Exact one-component-null stress test

The companion command evaluates the two boundary branches of the composite
null with 1,000 independently seeded four-contact families per branch, 12
trials, no masking, and exact enumeration for both component tests:

```bash
python validation/benchmark_crp_energy_composite_nulls.py \
  --output-dir release_artifacts/benchmark/composite_null_stress_test \
  --jobs 4
```

| Composite-null branch | `p_R <= 0.05` | `p_E <= 0.05` | Raw joint | Adjusted target | Any adjusted family call |
| --- | ---: | ---: | ---: | ---: | ---: |
| Projection alternative / energy null | 1,000/1,000 | 56/1,000 (5.6%) | 56/1,000 (5.6%) | 19/1,000 (1.9%) | 19/1,000 (1.9%) |
| Energy alternative / projection null | 51/1,000 (5.1%) | 1,000/1,000 | 51/1,000 (5.1%) | 10/1,000 (1.0%) | 11/1,000 (1.1%) |

In the energy-null branch, response RMS is set to
`(baseline_rms + eps) * exp(log_ratio) - eps`, so the detector's
epsilon-stabilized paired log-RMS difference equals the sampled centrally
symmetric `log_ratio`; a common waveform supplies the projection alternative.
In the projection-null branch, a randomly signed
biphasic waveform increases response energy while preserving central sign
symmetry. These simulations sample one declared construction on each
one-component-null branch and exercise the maximum-p decision there.

### Family size and p-value resolution

For an exact reproducibility test with `n <= 12`, the minimum attainable value
is `2**(-(n - 1))`. An isolated BH discovery at alpha 0.05 approximately
requires `m * p_min <= 0.05`; eight clean trials therefore support such a
discovery only through `m = 6`. With 5,000 Monte Carlo draws the minimum value
is `1 / 5001`, supporting an isolated discovery through `m = 250`. The
four-contact benchmark does not measure the larger multiplicity burden of the
30- and 143-contact real examples, so investigators should plan family size,
clean-trial count, and randomization budget together.

The generated, git-ignored `release_artifacts/benchmark/` directory retains raw
scenario-, channel-, and family-level decisions; cell and aggregate estimates;
the complete resolved configuration; code and environment provenance; and
SHA-256 checksums. Detector provenance records the declared root seed and the
exact order-invariant per-channel seed as decimal strings, together with their
SHA-256 derivation rule. Rerunning the command above reconstructs the directory
locally.

For a fast installation check, use:

```bash
python validation/benchmark_crp_energy.py --quick
```

CRP-energy, CRP-only, and paired-RMS benchmark calls receive
Benjamini-Hochberg adjustment across the simulated family. Kundu-Rolston and
N1-z retain unadjusted operational thresholds. These are implementation-level
comparators under a shared synthetic design, not exact replications of cited
publications and not substitutes for prospectively labeled clinical
validation.

Their null behavior also supports a descriptive interpretation. The
standalone selected-duration CRP implementation called 58/360 evaluable
noise-only targets and 278/1,800 evaluable contacts in all-null families; 195
of 480 families contained a call. Duration optimization and a t-test over
projections that share trials make this p-value field explicitly exploratory.
After separately demeaning each paired segment to remove baseline-centering
leakage, the final paired-Wilcoxon implementation called 3/480 noise-only
targets and 18/1,920 all-null contacts; 18/480 families contained a call. These
rows are distinct from the primary detector's `p_R` and `p_E` component tests
shown in the ablation table.

## Tier 2: Public Open Raw SPES Data

The script below downloads only byte ranges around real BIDS stimulation events from public OpenNeuro BrainVision signal files.

```bash
python validation/validate_public_spes.py --dataset ds003708 --min-events 8 --max-events 8
python validation/validate_public_spes.py --dataset ds004080 --min-events 8 --max-events 8
```

The primary CRP-energy detector requires at least eight artifact-clean trials.
A smaller slice remains useful for ingestion and epoching checks, but its
primary result is expected to be `insufficient_data`.

For a richer demonstration using more events, more stimulation sites, and summary figures:

```bash
python validation/validate_public_spes.py \
  --dataset all \
  --sites-per-dataset 3 \
  --max-events 10 \
  --max-channels 48 \
  --figures
```

Demonstrated datasets:

| Dataset | Signal source | Execution scope |
| --- | --- | --- |
| `ds003708` | Open BrainVision signal derivative | BIDS events -> raw signal window -> ERPy CSV -> epoch -> QC -> detection |
| `ds004080` | Open raw BrainVision signal | BIDS events -> raw signal window -> ERPy CSV -> epoch -> QC -> detection |

This tier demonstrates execution on real open stimulation timing and signal
files while avoiding full multi-GB downloads.

When `--figures` is enabled, each dataset/stimulation-pair workspace includes:

- summary waveform panel;
- spectral summary panel (PSD, ERSP, and inter-trial phase coherence) and an alpha-band inter-trial PLV connectivity matrix;
- response-ranked waveform grid;
- anatomy-organized waveform grid when electrode labels are available;
- response map;
- waveform panels with post-artifact zero-anchor verification, keeping stimulation onset at time 0;
- within-stimulation-site z-score plot for selected response metrics;
- CRP score map;
- CRP canonical response curve, canonical-weight timecourse, trial projections, and baseline-normalized expression panel;
- stimulation-source by recording-channel response matrix;
- CRP cross-stimulation comparison heatmaps;
- MNI glass-brain response network when Nilearn is installed, with electrode coordinates audited against the MNI152 brain mask;
- fsaverage cortical-surface response network with exact in-brain MNI electrode positions;
- max-over-epoch evoked-response brain network using peak edge magnitude, peak cumulative node magnitude, and peak active significance-method count;
- separate black/gold diamond markers for stimulation sources across static and animated brain-network views;
- dynamic hub and authority timecourses, node x time heatmaps, glass-brain node metric maps, and fsaverage cortical node metric maps;
- MNI coordinate QC network;
- Kamada-Kawai topology graph;
- interactive cortical-surface connectome;
- cortical-surface evoked-response animation over post-stimulus time.
- aggregate Plotly fsaverage brain animation showing evoked responses over time across the sampled stimulation sources.

## Tier 3: Lab-Specific Cohort Validation

For a lab cohort, run the same workflow on your actual data:

```python
patient = ep.Patient("EXAMPLE_PATIENT")
summary = patient.warmup(
    session_ids=["EXAMPLE_SESSION"],
    pipeline="blank_filt",
    event_method="auto",
)
```

Recommended cohort outputs:

- cache inventory table;
- event detection summary per stimulation pair;
- bad-channel and artifact-response QC tables;
- per-method detection table;
- method agreement matrix;
- response map, response-ranked grids, anatomy-organized grids, brain graph animations, and representative summary figures.

## NWB portability checks

The release does not use a separate experimental NWB notebook as evidence of
stimulation-analysis validity. To check an NWB file, use the documented
`Patient.from_nwb(...)` workflow in [BIDS_OPENNEURO.md](BIDS_OPENNEURO.md).
Provide explicit `stim_times` when the NWB trials table does not contain an
electrical-stimulation-site column. A cognitive-task NWB file without usable
stimulation timing is outside the stimulation-timing evaluation scope; ERPy does not
fabricate stimulation events from generic task columns.
