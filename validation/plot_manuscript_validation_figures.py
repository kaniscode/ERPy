#!/usr/bin/env python3
"""Replot manuscript Figures 1 and 5 from frozen aggregate benchmark outputs.

This script never reruns a benchmark or reads patient-level source recordings.
Dimensions and typography are specified at the final 180-mm publication width.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd

BLUE = "#24659A"
TEAL = "#198477"
ORANGE = "#BC652A"
PURPLE = "#79529B"
INK = "#172C3D"
MUTED = "#4D5D68"
GRID = "#D7E0E5"
PALE = "#F1F6F9"
W = 180.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.65,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "mathtext.fontset": "dejavusans",
})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canvas(height):
    fig = plt.figure(figsize=(W / 25.4, height / 25.4), dpi=160)
    fig._height_mm = height
    return fig


def text(fig, x, y, value, **kwargs):
    defaults = dict(ha="left", va="center", fontsize=8.5, linespacing=1.28)
    defaults.update(kwargs)
    return fig.text(x / W, y / fig._height_mm, value, **defaults)


def panel(fig, x, y, value):
    return text(fig, x, y, value, fontsize=9.2, fontweight="bold")


def axes(fig, x, y, w, h):
    ax = fig.add_axes([x / W, y / fig._height_mm, w / W, h / fig._height_mm])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.4, width=0.6, pad=2)
    ax.set_axisbelow(True)
    return ax


def box(fig, x, y, w, h, *, face=PALE, edge=GRID):
    shape = FancyBboxPatch((x / W, y / fig._height_mm), w / W,
                          h / fig._height_mm, boxstyle="round,pad=0,rounding_size=0.007",
                          transform=fig.transFigure, linewidth=0.8,
                          facecolor=face, edgecolor=edge, zorder=0)
    fig.add_artist(shape)
    return shape


def arrow(fig, start, end, **kwargs):
    defaults = dict(arrowstyle="-|>", mutation_scale=9, linewidth=1, color=MUTED,
                    shrinkA=0, shrinkB=0)
    defaults.update(kwargs)
    artist = FancyArrowPatch((start[0] / W, start[1] / fig._height_mm),
                            (end[0] / W, end[1] / fig._height_mm),
                            transform=fig.transFigure, **defaults)
    fig.add_artist(artist)


def workflow():
    fig = canvas(159)
    # The two inference paths use fixed windows and precede the contact-level gate.
    box(fig, 4, 141, 121, 14)
    text(fig, 64.5, 150.5, "Signals and events", ha="center", fontsize=10, fontweight="bold")
    text(fig, 64.5, 145.1, "Declared reference, processing settings and stimulation times", ha="center")

    box(fig, 4, 112, 121, 23)
    text(fig, 64.5, 129.9, "Time-aligned epochs and trial-contact eligibility", ha="center", fontsize=9.5, fontweight="bold")
    text(fig, 64.5, 124.2, "Trial × time × contact; trial-contact artifact masks", ha="center")
    text(fig, 64.5, 118.5, "Minimum clean trials; predeclared response and baseline windows", ha="center")
    arrow(fig, (64.5, 141), (64.5, 136))

    for x, title, lines in [
        (4, "Projection test", ["Cross-trial projections", "Whole-trial sign flips", "$p_R$ in the fixed response window"]),
        (67, "Matched energy test", ["Response and baseline RMS", "Paired log-RMS sign flips", "$p_E$ in the fixed windows"]),
    ]:
        box(fig, x, 76, 58, 29, face="#ECF4F8", edge="#A9C3D6")
        text(fig, x + 29, 99.8, title, fontsize=9.5, fontweight="bold", ha="center")
        for y, line in zip([93.5, 87.7, 81.9], lines):
            text(fig, x + 29, y, line, ha="center")
    arrow(fig, (43, 112), (33, 106))
    arrow(fig, (86, 112), (96, 106))

    box(fig, 4, 54, 121, 15, face="#EAF3F0", edge="#A8C9C0")
    text(fig, 64.5, 64.2, "Intersection-union test", ha="center", fontsize=9.5, fontweight="bold")
    text(fig, 64.5, 58.3, r"$p_{\mathrm{joint}} = \max(p_R,\,p_E)$", ha="center", fontsize=10)
    arrow(fig, (33, 76), (52, 70))
    arrow(fig, (96, 76), (77, 70))

    box(fig, 4, 29, 121, 19)
    text(fig, 64.5, 43, "Benjamini-Hochberg adjustment", ha="center", fontsize=9.5, fontweight="bold")
    text(fig, 64.5, 37.5, "Finite joint p-values in the eligible non-stimulation family", ha="center")
    text(fig, 64.5, 32.4, r"Adjusted detector call: $q_{\mathrm{joint}} \leq 0.05$", ha="center")
    arrow(fig, (64.5, 54), (64.5, 49))

    box(fig, 4, 3, 121, 20, face="#FFF5E9", edge="#D7B890")
    text(fig, 64.5, 17.8, "Separate contact-level QC gate", ha="center", fontsize=9.5, fontweight="bold")
    text(fig, 64.5, 12.3, "Final call = adjusted detector call AND declared contact eligibility", ha="center")
    text(fig, 64.5, 6.9, "Persistent-contact, artifact and boundary checks", ha="center")
    arrow(fig, (64.5, 29), (64.5, 24))

    # Descriptive estimates branch from the epochs; they do not choose test windows.
    box(fig, 135, 79, 41, 52, face="#F5F3F8", edge="#CCC0D8")
    text(fig, 155.5, 124.5, "Descriptive outputs", ha="center", fontsize=9.2, fontweight="bold")
    text(fig, 155.5, 115.2, "RMS ratio\nPeak amplitude\nand latency", ha="center")
    text(fig, 155.5, 96, "Selected-duration\nCRP measurements", ha="center")
    text(fig, 155.5, 84.5, "No window selection\nfor these tests", ha="center", fontsize=8)
    arrow(fig, (125, 123), (134, 123))

    box(fig, 135, 3, 41, 61)
    text(fig, 155.5, 57.4, "Retained record", ha="center", fontsize=9.2, fontweight="bold")
    text(fig, 155.5, 45.5, "Component p-values\nJoint p-value and q\nDetector and final calls", ha="center", fontsize=8)
    text(fig, 155.5, 28.3, "Trial masks\nContact QC reasons\nAnalysis windows", ha="center")
    text(fig, 155.5, 11.3, "Source provenance\nand parameters", ha="center")
    arrow(fig, (155.5, 79), (155.5, 65))
    arrow(fig, (125, 13), (134, 13))
    return fig


def synthetic(bench, recovery):
    fig = canvas(178)
    cell = pd.read_csv(bench / "cell_metrics.csv")
    cell = cell.loc[(cell.detector == "crp_energy") & (cell.truth == "injected")]
    family = pd.read_csv(bench / "family_calibration.csv")
    family = family.loc[family.detector == "crp_energy"].set_index("family_type")
    stress = pd.read_csv(bench / "composite_null_stress_test/composite_null_summary.csv").set_index("arm")
    assert (int(cell.n_scenarios.sum()), int(cell.n_evaluable.sum()), int(cell.n_calls_evaluable.sum())) == (1920, 1440, 1205)
    assert (recovery["injected_families"], recovery["evaluable_targets"], recovery["detected_targets"]) == (1920, 1440, 1205)

    panel(fig, 3, 173, "A  Factorial benchmark (2,400 families)")
    ax = axes(fig, 18, 143, 60, 23)
    # Exact zero-jitter benchmark shape, normalized on the response sample grid.
    config = json.loads((bench / "benchmark_config.json").read_text())
    step = 1.0 / config["sfreq"]
    n_samples = int(round((config["tmax"] - config["tmin"]) / step)) + 1
    all_times = config["tmin"] + np.arange(n_samples, dtype=float) * step
    t = all_times[(all_times >= config["response_window"][0]) &
                  (all_times <= config["response_window"][1])]
    waveform = (-np.exp(-0.5 * ((t - .042) / .010) ** 2)
                + .72 * np.exp(-0.5 * ((t - .105) / .022) ** 2)
                - .24 * np.exp(-0.5 * ((t - .205) / .040) ** 2))
    waveform /= np.sqrt(np.mean(waveform ** 2))
    ax.axhline(0, color=GRID, lw=.7)
    ax.plot(t * 1000, waveform, color=BLUE, lw=1.4)
    ax.set(xlim=(0, 330), xticks=[0, 100, 200, 300], yticks=[-2, 0, 2],
           xlabel="Time after stimulation (ms)", ylabel="Normalized\ntemplate")
    text(fig, 48, 129.7, "4 contacts; 40 replicates per design cell", ha="center")

    panel(fig, 94, 173, "B  Fixed-design evaluability")
    ax = axes(fig, 112, 143, 45, 23)
    grouped = cell.groupby("artifact_fraction")[["n_scenarios", "n_evaluable"]].sum()
    yy = np.array([2, 1, 0])
    ax.barh(yy, 100 * grouped.n_evaluable / grouped.n_scenarios,
            height=.52, color=[BLUE, "#6599BC", "#A8C4D7"])
    ax.set(xlim=(0, 103), xticks=[0, 50, 100], ylim=(-.65, 2.65),
           yticks=yy, yticklabels=["0%", "25%", "50%"], xlabel="Target evaluability (%)")
    ax.set_ylabel("Masked trials", labelpad=5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for y, row in zip(yy, grouped.itertuples()):
        # Outside the plotting rectangle, in a reserved count column.
        y_mm = 143 + 23 * (y + .65) / 3.3
        text(fig, 160, y_mm, f"{row.n_evaluable:.0f}/{row.n_scenarios:.0f}", fontsize=8)
    text(fig, 136, 129.7, "Overall: 75.0% (1,440/1,920); no CI", ha="center")

    panel(fig, 3, 119, "C  Recovery among evaluable targets")
    ax = axes(fig, 18, 81, 60, 31)
    styles = [(BLUE, "o", "-"), (ORANGE, "s", "--"), (TEAL, "^", "-."), (PURPLE, "D", ":")]
    for trials, (color, marker, line) in zip([8, 12, 24, 36], styles):
        row = cell.loc[cell.n_trials_total == trials].groupby("snr_nominal")[["n_evaluable", "n_calls_evaluable"]].sum()
        ax.plot(row.index, 100 * row.n_calls_evaluable / row.n_evaluable,
                marker=marker, markersize=3.5, markerfacecolor="white", linestyle=line,
                lw=1.1, color=color, label=f"{trials} trials")
    ax.set_xscale("log", base=2)
    ax.set(xlim=(.45, 4.5), xticks=[.5, 1, 2, 4], xticklabels=["0.5", "1", "2", "4"],
           ylim=(-3, 106), yticks=[0, 50, 100], xlabel="Nominal SNR", ylabel="Recovery (%)")
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.legend(loc="upper center", bbox_to_anchor=(48 / W, 69 / 178),
              bbox_transform=fig.transFigure, ncol=2, frameon=True, fancybox=False,
              edgecolor=GRID, framealpha=1, handlelength=1.8, borderpad=.35,
              columnspacing=1.1, labelspacing=.45)

    panel(fig, 94, 119, "D  Recovery and stratified uncertainty")
    cards = [(97, "Evaluable targets", "conditional_recovery", "conditional_95_percentile_interval", 1440),
             (76, "All injected targets", "overall_recovery", "overall_95_percentile_interval", 1920)]
    for y, label, rate_key, ci_key, denominator in cards:
        box(fig, 94, y, 83, 17)
        text(fig, 98, y + 12.5, label, fontsize=8.5, fontweight="bold")
        text(fig, 98, y + 5.0, f"{100 * recovery[rate_key]:.1f}%", fontsize=13, color=BLUE, fontweight="bold")
        low, high = 100 * np.asarray(recovery[ci_key])
        text(fig, 119, y + 5.1, f"1,205/{denominator:,}; 95% CI {low:.1f}–{high:.1f}%", fontsize=8)
    text(fig, 135.5, 69.1, "Whole families resampled within design cells", ha="center", fontsize=8)

    panel(fig, 3, 54, "E  Any null call: factorial families")
    ax = axes(fig, 29, 16, 49, 28)
    for y, key in [(1, "all_null"), (0, "mixed_signal")]:
        row = family.loc[key]
        p = 100 * row.probability_any_null_call
        lo = 100 * row.probability_any_null_call_ci_low
        hi = 100 * row.probability_any_null_call_ci_high
        ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o", ms=4,
                    color=BLUE, elinewidth=1.2, capsize=2)
    ax.set(xlim=(-.035, 1.03), xticks=[0, .5, 1], ylim=(-.6, 1.6), yticks=[1, 0],
           yticklabels=["All-null\n0/480", "Mixed\n2/1,920"], xlabel="Families with any null call (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color=GRID, lw=.6)

    panel(fig, 94, 54, "F  Composite-null targets (2 × 1,000)")
    ax = axes(fig, 123, 16, 52, 28)
    for y, arm in [(1, "projection_alternative_energy_null"), (0, "energy_alternative_projection_null")]:
        row = stress.loc[arm]
        for offset, field, color, marker, label in [
            (.12, "raw_joint_call", ORANGE, "o", "Raw"),
            (-.12, "adjusted_target_call", BLUE, "s", "After BH")]:
            p = 100 * row[field + "_rate"]
            lo, hi = 100 * row[field + "_ci_low"], 100 * row[field + "_ci_high"]
            ax.errorbar(p, y + offset, xerr=[[p - lo], [hi - p]], fmt=marker,
                        ms=3.7, color=color, mfc="white" if field == "raw_joint_call" else color,
                        elinewidth=1.0, capsize=2, label=label if y == 1 else None)
    ax.set(xlim=(0, 8), xticks=[0, 2, 4, 6, 8], ylim=(-.6, 1.6), yticks=[1, 0],
           yticklabels=["Energy null\n5.6% → 1.9%", "Projection null\n5.1% → 1.0%"], xlabel="Target false calls (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.legend(loc="lower center", bbox_to_anchor=(147 / W, 45.4 / 178),
              bbox_transform=fig.transFigure, ncol=2, frameon=True, fancybox=False,
              edgecolor=GRID, framealpha=1, handlelength=1.2, borderpad=.35, columnspacing=1)
    return fig


def layout_audit(fig):
    """Report actual drawn text extents at final size; do not silently hide errors."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    items = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        # Matplotlib keeps invisible alternate tick-label artists in its tree.
        bbox = artist.get_window_extent(renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        items.append((artist, bbox))
    clipped = [a.get_text() for a, b in items if not
               (b.x0 >= bounds.x0 and b.y0 >= bounds.y0 and b.x1 <= bounds.x1 and b.y1 <= bounds.y1)]
    collisions = []
    for (a, b), (c, d) in itertools.combinations(items, 2):
        overlap_w, overlap_h = min(b.x1, d.x1) - max(b.x0, d.x0), min(b.y1, d.y1) - max(b.y0, d.y0)
        if overlap_w > .5 and overlap_h > .5:
            collisions.append([a.get_text(), c.get_text()])
    return {"width_mm": W, "height_mm": fig._height_mm,
            "minimum_plain_text_font_pt": min(a.get_fontsize() for a, _ in items),
            "text_artist_count": len(items), "clipped_text": clipped,
            "text_collisions": collisions}


CAPTIONS = {
    "fig01_workflow_revised": {
        "caption": "Figure 1. ERPy analysis workflow. Signals and stimulation events are converted to aligned epochs, with trial-contact eligibility and analysis windows recorded explicitly. The primary detector evaluates cross-trial projection reproducibility and response-versus-baseline energy in fixed, predeclared windows. Their component p-values are combined by an intersection-union rule, p_joint = max(p_R, p_E), and finite joint p-values are adjusted with Benjamini-Hochberg across the eligible non-stimulation contact family. An adjusted detector call at q_joint ≤ 0.05 precedes a separate, declared contact-level quality-control gate; both calls and exclusion reasons are retained. Descriptive amplitude, latency, RMS-ratio and selected-duration CRP measurements branch from the epochs and do not select windows for these primary tests. The diagram states the implemented sequence, not a universal false-discovery-rate guarantee: interpretation depends on component-test calibration, the testing family, dependence and any subsequent gate.",
        "alt_text": "Workflow with a main inference path and a separate descriptive-output branch. Signals and events lead to aligned epochs and trial-contact masks. Two parallel fixed-window tests, projection reproducibility and matched response-baseline energy, feed the maximum-p intersection-union test. Benjamini-Hochberg adjustment uses finite joint p-values for eligible non-stimulation contacts. A separate contact-level QC gate follows the adjusted detector call. Descriptive estimates and both calls, masks, windows, reasons and provenance enter a retained record."
    },
    "fig05_synthetic_validation_revised": {
        "caption": "Figure 5. Frozen synthetic benchmark and composite-null stress tests. (A) The zero-jitter, RMS-normalized injected response template used in the factorial benchmark. The 2,400 four-contact families comprise 1,920 injected-target families and 480 all-null families; injected cells cross 8, 12, 24 or 36 trials, nominal SNR 0.5, 1, 2 or 4, and masked-trial fractions 0, 0.25 or 0.50, with 40 independently generated families per cell. (B) Target evaluability by masked fraction under the minimum-eight-clean-trial rule. Overall evaluability is fixed by the design at 1,440/1,920 (75.0%); no sampling interval is assigned. Masking is supplied to the detector and does not evaluate artifact-recognition accuracy. (C) Conditional target recovery, pooling evaluable targets over mask fractions for each trial count and SNR. (D) Recovery among evaluable targets is 1,205/1,440 (83.7%; 95% interval 82.4–84.9%); recovery across all injected targets is 1,205/1,920 (62.8%; 61.8–63.7%). Percentile intervals use 20,000 resamples of whole families independently within the fixed design cells (seed 20260907), preserving design weights. (E) At least one null-contact call occurred in 0/480 all-null families and 2/1,920 mixed families. (F) In separate 1,000-family stress arms, the designated target is energy-null with a projection alternative, or projection-null with an energy alternative. Raw maximum-p target false-call rates were 5.6% and 5.1%, respectively; after family adjustment they were 1.9% and 1.0%. Error bars in E–F are 95% Wilson intervals from the frozen summaries. Target calls in F differ from the event of any call in the whole family. These results describe the stated simulations, not clinical sensitivity or a general guarantee of error control.",
        "alt_text": "Six panels summarize 2,400 factorial families and 2,000 composite-null stress families. A shows the injected negative-positive-negative template. B shows target evaluability of 100%, 75% and 50% when 0%, 25% and 50% of trials are masked; overall evaluability is 75% without a confidence interval. C shows recovery increasing with nominal SNR and trial count. D reports conditional recovery 83.7%, interval 82.4–84.9%, and overall recovery 62.8%, interval 61.8–63.7%. E reports any-null-call counts 0/480 all-null and 2/1,920 mixed families. F reports raw and adjusted target false calls: energy-null 5.6% and 1.9%, projection-null 5.1% and 1.0%."
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = Path(__file__).parent / "figure_data"
    parser.add_argument("--benchmark", type=Path, default=data / "synthetic_benchmark")
    parser.add_argument("--recovery", type=Path, default=data / "stratified_recovery.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        "scenario_results.csv.gz": "cc832ab691db83d961b1607f7a726e80bbc7645b3c6fddb0e5d5a4019349ad53",
        "cell_metrics.csv": "6698c01e9601f8250988d896fbdb838155ddf66d5d96d962171711b058670084",
        "family_calibration.csv": "90671f0e3ccef5fe7afc8521bcbb254a4cf47d1bc77d37c8ec5e1c70c4fe97ff",
        "benchmark_config.json": "81972f5396d3120c31de477aa0c22ed6ffdcd369d22645153ae7ae7b28836c9d",
        "composite_null_stress_test/composite_null_summary.csv": "859d99e3575e237a4af3bc4f6957df39d1ea09c74d1d1a7598b8f9bcede7eb26",
    }
    provenance = {}
    for name, expected_hash in expected.items():
        observed = sha256(args.benchmark / name)
        if observed != expected_hash:
            raise ValueError(f"Frozen source hash mismatch: {name}")
        provenance[name] = observed
    stress_name = "composite_null_stress_test/composite_null_summary.csv"
    provenance[stress_name] = sha256(args.benchmark / stress_name)
    recovery = json.loads(args.recovery.read_text())
    assert recovery["input_sha256"] == expected["scenario_results.csv.gz"]
    provenance[args.recovery.name] = sha256(args.recovery)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, fig in [("fig01_workflow_revised", workflow()),
                      ("fig05_synthetic_validation_revised", synthetic(args.benchmark, recovery))]:
        report = layout_audit(fig)
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(args.output_dir / f"{name}.{extension}", dpi=450)
        reports[name] = report
        plt.close(fig)
    (args.output_dir / "revised_figure_captions.md").write_text("\n\n".join(
        f"## {name}\n\n{entry['caption']}\n\nAlt text: {entry['alt_text']}"
        for name, entry in CAPTIONS.items()) + "\n")
    (args.output_dir / "revised_figure_alt_text.json").write_text(json.dumps(
        {name: item["alt_text"] for name, item in CAPTIONS.items()}, indent=2) + "\n")
    manifest = {"script_sha256": sha256(Path(__file__)), "source_sha256": provenance,
                "layout_audit": reports, "png_dpi": 450,
                "benchmark_replayed": False, "private_recordings_accessed": False}
    (args.output_dir / "revised_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    if any(r["clipped_text"] or r["text_collisions"] or r["minimum_plain_text_font_pt"] < 8 for r in reports.values()):
        raise SystemExit("Layout audit requires review; inspect exported figures before delivery.")


if __name__ == "__main__":
    main()
