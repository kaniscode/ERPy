# Methods and mathematical definitions

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

This document specifies the analysis quantities implemented in ERPy 1.0.0.
Defaults are stated where they define the released workflow; every result table
also records detector parameters. Time is in seconds relative to stimulation
onset and voltage is in microvolts unless the input metadata states otherwise.

## Notation and epoch summary

Let \(x_{ict}\) denote sample \(t\) from trial \(i\) and recording contact
\(c\). For a baseline sample set \(B\), each trial is centered independently:

\[
\tilde{x}_{ict}=x_{ict}-\frac{1}{|B_{ic}|}\sum_{s\in B_{ic}}x_{ics},
\]

where \(B_{ic}\) contains finite samples only. Unless overridden, the general
detection baseline is \([-0.50,-0.03]\) s and the response window is
\([0.01,0.30]\) s. Detector-specific windows below take precedence.

With the clean-trial set \(C_{ct}\), the artifact-aware mean and standard error
are

\[
\bar{x}_{ct}=\frac{1}{n_{ct}}\sum_{i\in C_{ct}}\tilde{x}_{ict},\qquad
\operatorname{SEM}_{ct}=\frac{s_{ct}}{\sqrt{n_{ct}}}.
\]

Counts may vary across time/contact when only selected responses are masked.
An empty or ineligible set is reported as missing rather than as zero.

Operational component windows are 10–50 ms for the early A1/N1 component and
50–250 ms for the later A2/N2 component. These are analysis conventions, not
claims that physiological components have universal boundaries. For a window
\(W\), ERPy reports peak magnitude \(\max_{t\in W}|\bar{x}_{ct}|\), its signed
value and latency, root-mean-square (RMS)

\[
\operatorname{RMS}_{ic}(W)=
\sqrt{\frac{1}{|W|}\sum_{t\in W}\tilde{x}_{ict}^{2}},
\]

and trapezoidal area under the curve when requested.

## Preprocessing and referencing

The standard `blank_filt` pipeline blanks the configured stimulation interval,
applies notch and band-pass filters, and applies a robustly screened
common-average reference. It excludes stimulation contacts and contacts already
declared bad, but it does not automatically run persistent-channel detection.
Use `blank_filt_reject` or an explicit `reject_bad_channels` step to add
automatic persistent-channel screening. The named `blank_dec_*` variants also
change sampling rate. Steps and parameters are stored in cache metadata.
Filters use zero-phase forward/backward second-order sections where the
operation permits it.

For common-average referencing, the sample-wise cross-contact median and robust
scale first define an eligible reference set \(G_t\). The reference is then the
arithmetic mean over that screened set,

\[
r_t=\frac{1}{|G_t|}\sum_{g\in G_t}x_{gt},
\qquad x'_{ct}=x_{ct}-r_t,
\]

when enough screened contacts remain; otherwise the reference falls back to the
cross-contact median. Stimulation contacts, predeclared bad contacts, nonfinite
channels, and sample-wise amplitude outliers are excluded from \(G_t\). Bipolar
referencing subtracts adjacent contacts on the same lead; Laplacian referencing
subtracts a distance-weighted mean of immediate lead neighbors. Any montage
change alters channel identity and is recorded explicitly.

## Post-artifact anchor

Time zero remains stimulation onset. A separate amplitude anchor may be used to
avoid referencing peaks to the stimulation spike. ERPy forms a robust onset
envelope \(e_t\): the maximum absolute voltage across one or two reference
contacts, or the 95th percentile across larger reference sets. From the
prestimulus envelope,

\[
z_t=\frac{e_t-\operatorname{median}(e_B)}
{1.4826\,\operatorname{MAD}(e_B)}.
\]

After light smoothing, the detector locates the largest near-onset artifact and
chooses the first post-peak time at which both \(z_t\le2\) and the absolute
samplewise slope is at most two robust baseline units for at least 2 ms. Search
begins no earlier than 3 ms and ends at 20 ms by default. If no sustained run
exists, the first amplitude-return sample or the quietest post-peak sample is
reported as a documented fallback. A short common post-anchor window is then
subtracted per trial. The anchor time, peak time, scores, channels, and fallback
status are retained for audit.

## Continuous-channel quality control

Persistent-channel screening is an optional preprocessing step. When enabled,
ERPy computes standard deviation, peak-to-peak
range, missing-sample fraction, median absolute voltage, and mean correlation
with the other contacts. Feature \(f_c\) receives a robust cohort score

\[
z^{\mathrm{rob}}_c=
\frac{f_c-\operatorname{median}(f)}
{1.4826\,\operatorname{MAD}(f)}.
\]

If MAD is zero, the finite standard deviation is used; a unit denominator is
the final degenerate fallback. A channel is flagged when any configured hard
rule fires: excessive missingness, flatness, robust amplitude/offset outlier,
or excessive anticorrelation. The report records every reason rather than only
a binary label.

## Trial-by-contact artifact quality control

Within response window \(W\), each trial/contact receives the following
features:

- missing fraction, \(1-|F|/|W|\), for finite-sample set \(F\);
- absolute peak, \(\max_{t\in F}|x_t|\);
- peak-to-peak range, \(\max(x_F)-\min(x_F)\);
- maximum absolute gradient, \(\max_t|\nabla x_t|\);
- repeated-value occupancy, the largest rounded-value count divided by \(|F|\);
- longest near-constant run and its fraction of \(|F|\);
- extreme-value dwell and, when acquisition limits are supplied, true rail
  dwell within the configured rail tolerance;
- late-response roughness ratio, defined as the mean absolute sample gradient
  from 50–500 ms divided by the corresponding mean from 0–50 ms (with the
  available epoch end used when shorter). The serialized
  `late_high_frequency_ratio` field name is retained for compatibility, but
  this feature is not a spectral-energy or high-gamma measure.

Peak, range, gradient, and late-response roughness features receive robust z-scores
within contact. Defaults flag missing fraction above 0.02, a repeated plateau
above the persistence/occupancy rule, known-rail dwell above 0.02, robust peak,
range, or gradient z-score at least 6, an extreme peak/range z-score at least
12, or a late-response roughness-ratio z-score at least 6. Absolute acquisition limits
are applied only when supplied. The trial/contact decision is the logical OR
of triggered rules, and every reason is preserved. Channel burden is

\[
\pi_c=\frac{\#\{i:\text{trial/contact }(i,c)\text{ flagged}\}}
{\#\{i:\text{trial }i\text{ evaluated at }c\}}.
\]

Hard clipping/saturation reasons are reported separately. Downstream primary
inference requires the configured minimum number of clean matched trials and
excludes stimulation contacts by default.

## Primary CRP-energy detector

The released primary decision requires both amplitude-weighted waveform
reproducibility and poststimulus excess energy. Their maximum p-value forms an
intersection–union test; complementary detector fields remain separate.

### Fixed-window trial reproducibility

For clean baseline-centered trial vectors \(x_i\) over the full response window
declared before inference, ERPy forms every ordered semi-normalized
cross-projection. Because only the first trial norm appears in each ordered
term, the resulting statistic measures amplitude-weighted waveform alignment
rather than pure shape correlation:

\[
P_{ij}=\frac{\langle x_i,x_j\rangle}
{\|x_i\|_2\sqrt{f_s}},\quad i\ne j,\qquad P_{ii}=0.
\]

The reproducibility statistic and its sign-randomized form are

\[
T_R=\frac{\mathbf{1}^{\mathsf T}P\mathbf{1}}{n(n-1)},
\qquad
T_R(s)=\frac{s^{\mathsf T}Ps}{n(n-1)},
\]

where \(s\in\{-1,+1\}^n\) reverses whole trials and therefore preserves every
trial norm and the shared-trial dependence among projections. The one-sided
component value is \(p_R=\Pr_s\{T_R(s)\ge T_R\}\). For at most 12 trials,
ERPy evaluates the \(2^{n-1}-1\) non-observed sign patterns after fixing
\(s_1=+1\), counts the observed all-positive assignment explicitly, and uses
denominator \(2^{n-1}\). A scale-aware floating-point tolerance treats
numerically tied statistics as ties. Larger samples use at least 5,000 seeded Rademacher assignments and
the Monte Carlo plus-one correction

\[
\widehat p_R=\frac{1+\sum_{b=1}^{N_R}
\mathbf{1}\{T_R(s_b)\ge T_R\}}{1+N_R}.
\]

This test assumes independent trials and
central sign symmetry of the full response waveform under the null. A
data-selected duration cannot define its inferential window.

### Descriptive canonical response parametrization

For a candidate duration \(T\), arrange response data as
\(V_T\in\mathbb{R}^{T\times n}\). Candidate durations begin at 10 ms and
advance in 5-ms steps. ERPy chooses the duration maximizing the mean ordered
off-diagonal semi-normalized projection. Singular-value decomposition of
\(V_T\) supplies unit canonical curve \(c\), oriented toward the mean response;
trial coefficient \(\alpha_i=c^\top v_i\). Residual and explained-energy
quantities are

\[
r_i=v_i-c\alpha_i,\qquad
\operatorname{EV}_i=1-\frac{\|r_i\|_2^2}{\|v_i\|_2^2}.
\]

The selected duration, canonical curve, coefficients, SNR, and explained
fraction are descriptive and do not supply \(p_R\).

The standalone `crp_significance` comparator uses a different inferential
quantity. At the data-selected duration, it takes one projection from each
reciprocal trial pair with the published balanced-index construction and
applies a one-sided one-sample t-test against zero. Its unadjusted extraction
value is stored as `p_value`; `crp_q_value` is its Benjamini-Hochberg-adjusted
value across channels. This selected-duration extraction test is retained for
source-aligned comparison with the published CRP method. It is not the
fixed-window whole-trial sign-flip test \(p_R\), and it is not used to define
the primary CRP-energy conjunction.

### Matched excess energy

The response begins after the excluded 0–15 ms artifact interval by default.
ERPy pairs each response sample with the same number of immediately available
prestimulus baseline samples. For clean trial \(i\), let
\(R_i^\circ=R_i-\overline{R}_i\mathbf{1}\) and
\(B_i^\circ=B_i-\overline{B}_i\mathbf{1}\), so the matched segments are
demeaned separately. Then

\[
d_i=\log\{\operatorname{RMS}(R_i^\circ)+\epsilon\}
-\log\{\operatorname{RMS}(B_i^\circ)+\epsilon\}.
\]

The statistic is \(\bar d\). Its one-sided randomization p-value \(p_E\) is
computed by all sign assignments for at most 16 trials and by at least 5,000
seeded random assignments otherwise. The reported RMS ratio is

\[
20\log_{10}
\left(\frac{\operatorname{GM}\{\operatorname{RMS}(R_i^\circ)+\epsilon\}}
{\operatorname{GM}\{\operatorname{RMS}(B_i^\circ)+\epsilon\}}\right),
\]

which equals \(20\bar d/\ln(10)\).

The energy randomization requires the trial differences to be independent
across trials and jointly invariant under coordinate-wise sign changes under
the null; independent, centrally symmetric differences are sufficient.
Within-trial exchangeability of the separately demeaned matched segments can
supply marginal symmetry but does not by itself establish across-trial
independence. Separate demeaning removes sustained DC-only shifts by
construction.

The channel-level conjunction p-value is

\[
p_{\mathrm{joint}}=\max(p_R,p_E).
\]

With component nulls \(H_R\) and \(H_E\), the joint scientific null is
\(H_R\cup H_E\). The maximum is therefore an intersection–union p-value and
requires no independence assumption between the two components, provided that
each component p-value is valid under its own null.

For API compatibility, the primary-result fields `p_crp` and `crp_statistic`
store \(p_R\) and \(T_R\), respectively. They are not the standalone CRP
comparator's selected-duration t-test fields.

Benjamini-Hochberg adjustment is applied across the eligible recording
contacts with finite joint p-values from one acquisition and stimulation pair.
Its false-discovery-rate interpretation additionally requires valid p-values
and independence, positive regression dependence on a subset (PRDS), or
another dependence structure appropriate to the chosen correction across null
contacts. A primary response requires clean-trial
eligibility, non-stimulation-contact eligibility, and
\(q_{\mathrm{joint}}\le0.05\). The two unadjusted component decisions define
four diagnostic test classes: reproducible/energetic,
reproducible/low-energy, energetic/inconsistent, and neither. They use
\(p_R\leq0.05\) and \(p_E\leq0.05\), whereas the final result uses `q_joint`
and separately retained artifact eligibility. `insufficient_data` is kept
distinct.

With a declared root seed, a versioned SHA-256 derivation produces a distinct,
order-invariant 63-bit seed from each exact channel label. NumPy `SeedSequence`
then creates separate child streams for reproducibility and energy. Mixed-method
tables serialize root and effective seeds as canonical decimal strings so
63-bit integers cannot be rounded through a floating-point column; the strict
per-channel parameter JSON retains the exact effective integer. A missing root
seed is recorded as nondeterministic.

Canonical projection energy is evaluated with leave-one-trial-out canonical
curves by default, conditional on the duration selected from all clean trials.
For held-out trial \(v_i\), the remaining training trials define \(c_{-i}\),
and reconstructed energy is \((v_i^\top c_{-i})^2\). Summing these terms and
dividing by trial and sample counts gives the conditional leave-one-trial-out
canonical projection energy; division by total observed energy gives its
fraction.

## Comparator detectors

These methods expose source-aligned quantities for descriptive comparison. The
legacy `consensus` field records whether at least a configured count (two by
default) of the *available* standalone comparators is positive; primary
inference uses the joint CRP-energy row. ERPy retains `method_available`,
`availability_reason`, and `n_methods_available` so a missing required input
remains distinguishable from a negative call. The reader-facing
`rms_response` binary field and `peak_amplitude` share the same 10–50-ms
peak-z rule, so their joint positivity supplies one decision rule in duplicate
rather than independent corroboration.

- **Keller z-score.** The maximum absolute mean-waveform deviation in 10–50 ms
  (A1/N1) and 50–250 ms (A2/N2) is divided by the prestimulus standard
  deviation. The default decision requires the larger of the A1/N1 and A2/N2
  z-scores to be at least 6 together with the released polarity/artifact rule.
- **Kundu/Rolston envelope.** After robust common-median rereferencing, a 10-Hz
  high-pass is squared and 10-Hz low-pass smoothed, then square-rooted. The
  median trial envelope must remain above three times its −100 to −5 ms
  baseline for at least 15 ms in 5–100 ms and its poststimulus median must
  exceed 30 µV.
- **CRP significance.** Uses the data-selected-duration, balanced-projection
  one-sample extraction test described above. Its channelwise adjusted value
  is `crp_q_value`. Duration optimization reuses the analyzed response and the
  t-test treats projections that share trials as observations; the resulting
  p-value was anti-calibrated under the release all-null simulation. ERPy marks
  the field `exploratory_uncalibrated_selected_duration_shared_trial_t_test`.
  The field is distinct from the primary detector's fixed-window whole-trial
  sign-flip value \(p_R\).
- **SIGNI/Crowther high gamma.** The 0–5 ms artifact is interpolated, the
  polarity-specific evoked mean is removed, and the 70–170 Hz Hilbert envelope
  is formed. Response SNR is the total 10–100 ms variance divided by the mean
  variance in six 15-ms bins. Trialwise circular-shift/time-reversal
  randomization forms a log-null distribution; the released decision uses the
  channel-wise Bonferroni-adjusted parametric tail probability. This method is
  unavailable unless a passband-preserving `wideband_epochs` object is passed
  explicitly or the caller explicitly declares a verified wideband primary
  input. A high sampling rate does not restore frequencies removed by a
  low-pass filter; standard 0.1–50-Hz `blank_filt` epochs cannot support this
  comparison.
- **Peak amplitude.** Reports signed and absolute A1/N1 peak, latency, and
  baseline-normalized z-score; the decision uses \(z\ge6\).
- **Matched-window RMS.** Reports paired 10–100 ms response and equally sized
  prestimulus RMS values after demeaning each segment separately, together with
  their ratio, robust z-score, exploratory one-sided Wilcoxon test, and
  BH-adjusted value. Separate segment demeaning prevents trial baseline
  centering from suppressing the comparator baseline RMS. To remain
  source-aligned, its binary detection field uses the 10–50 ms A1/N1
  \(z\ge6\) rule; RMS is the magnitude quantity.

For two binary method sets \(A\) and \(B\), agreement may be summarized with
the Jaccard index \(J(A,B)=|A\cap B|/|A\cup B|\). Empty unions are reported
explicitly rather than interpreted as perfect agreement.

## Spectral and connectivity quantities

Welch power spectral density uses windowed, overlapping segments. Morlet
time-frequency power is \(|(x*w_f)(t)|^2\); ERSP applies the selected baseline
transform. Inter-trial phase coherence is

\[
\operatorname{ITPC}(f,t)=
\left|\frac{1}{n}\sum_i e^{j\phi_i(f,t)}\right|.
\]

Pairwise phase-locking value is

\[
\operatorname{PLV}_{xy}(t)=
\left|\frac{1}{n}\sum_i e^{j(\phi_{xi}(t)-\phi_{yi}(t))}\right|.
\]

ERPy also reports magnitude-squared coherence, imaginary coherence,
histogram-based mutual information, regression-based Granger scores,
phase-amplitude modulation index, and n:m phase-phase coupling. Because these
are optional descriptive analyses, their parameters and window definitions
must accompany any reported result.

## Graph construction

A directed edge connects stimulation source \(s\) to response contact \(c\)
when a declared detector/QC rule passes; the edge weight is an explicitly
named response metric. ERPy never silently treats absent, ineligible, and
nonsignificant edges as interchangeable. Dynamic node strength, in/out degree,
hub/authority scores, reciprocity, and target-distribution metrics are evaluated
on each time-resolved adjacency matrix. Coordinates are used only after label
resolution and coordinate-frame validation.

## Reproducibility requirements

An analysis record should state the ERPy version and compatibility checkpoint,
input file hashes or governed source IDs, preprocessing sequence, reference,
windows, artifact thresholds, clean-trial counts, detector parameters,
multiple-testing family, random seed/draw count, and any manual exclusions.
The compatibility checkpoint is a fixed software-behavior label shared by all
equivalent installations; it is not a dataset identifier.

An optional `ERPY-RR-<UUID>` release-record label is different. It is a random,
author-assigned local audit label that can point, through a separately governed
mapping, to an exact approved derivative. ERPy does not upload data under that
label, and `AnalysisResult.save(...)` does not generate or register one. Saved
tables and JSON metadata remain in the requested local output directory.
Protected provenance and any release-record mapping belong in the governed
research environment, not the public repository.

## Primary references

- Benjamini Y, Hochberg Y. Controlling the false discovery rate. *Journal of
  the Royal Statistical Society B*. 1995;57:289–300.
- Crowther LJ et al. A quantitative method for evaluating cortical responses
  to electrical stimulation. *Journal of Neuroscience Methods*. 2019.
  doi:10.1016/j.jneumeth.2018.09.034.
- Keller CJ et al. Corticocortical evoked potentials reveal projectors and
  integrators in human brain networks. *Journal of Neuroscience*. 2014.
  doi:10.1523/JNEUROSCI.4289-13.2014.
- Kundu B et al. A systematic exploration of parameters affecting evoked
  intracranial potentials. *Brain Stimulation*. 2020.
  doi:10.1016/j.brs.2020.06.002.
- Miller KJ et al. Canonical response parameterization for electrical
  stimulation. *PLOS Computational Biology*. 2023.
  doi:10.1371/journal.pcbi.1011105.
