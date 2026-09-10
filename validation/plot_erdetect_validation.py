#!/usr/bin/env python3
"""Export the independent-cohort external validation figure at 180 mm width."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

SCOPE = "independent_external_excluding_development_overlap"
SUBJECTS = ("MAYO02", "MAYO03", "MAYO04", "MAYO05", "UMCU20", "UMCU21",
            "UMCU22", "UMCU23", "UMCU25", "UMCU26", "UMCU59", "UMCU62", "UMCU67")
EARLY = "#007F79"
BROAD = "#5275A5"
ARCHIVED = "#4D4D4D"
TEXT = "#222222"
GRID = "#E3E7E9"


def select_rows(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics.loc[metrics.cohort_scope.eq(SCOPE)
                           & metrics.reference_view.eq("equal_record_available_rater")
                           & metrics.missingness_policy.eq("available_case")]
    pooled = selected.loc[selected.level.eq("overall")
                          & selected.comparison.eq("paired_archived_intersection")]
    rows = []
    for method, configuration, label in [
        ("ERPy", "early_10_90ms", "ERPy early\n10-90 ms"),
        ("ERPy", "broad_15_300ms", "ERPy broad\n15-300 ms"),
        ("archived_ER_detect", "early_10_90ms", "Archived ER-detect\n9-90 ms"),
    ]:
        match = pooled.loc[pooled.method.eq(method) & pooled.configuration.eq(configuration)]
        if len(match) != 1:
            raise ValueError("Expected one unique pooled row for each matched method")
        row = match.iloc[0].copy()
        if row.n_subjects != 13 or not np.isclose(row.n_scored_records, 32048, rtol=0, atol=1e-8):
            raise ValueError("Matched comparison differs from the frozen independent cohort")
        row["plot_label"] = label
        rows.append(row)
    archived = pooled.loc[pooled.method.eq("archived_ER_detect")]
    fields = ["sensitivity", "sensitivity_low", "sensitivity_high", "specificity", "specificity_low", "specificity_high"]
    if len(archived) != 2 or not np.allclose(archived.iloc[0][fields].astype(float), archived.iloc[1][fields].astype(float)):
        raise ValueError("Archived comparator differs across the two paired cohorts")
    subjects = selected.loc[selected.level.eq("subject") & selected.method.eq("ERPy")
                            & selected.comparison.eq("erpy_all_source_eligible")].copy()
    if len(subjects) != 26 or set(subjects.group) != set(SUBJECTS):
        raise ValueError("Participant panel must contain exactly the 13 independent participants")
    for _, group in subjects.groupby("configuration"):
        if not np.allclose([group.n_scored_records.sum(), group.eligible.sum()], [32121, 33694], rtol=0, atol=1e-8):
            raise ValueError("All-source denominators differ from the frozen scoring result")
    return pd.DataFrame(rows).reset_index(drop=True), subjects


def clean_axis(ax):
    for name in ["top", "right", "left"]:
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color("#9BA4A9")
    ax.spines["bottom"].set_linewidth(.6)
    ax.tick_params(axis="both", length=0, pad=4, colors=TEXT)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.55)


def figure_caption(*, lowercase_panels: bool = False) -> tuple[str, str]:
    a, b = ("a", "b") if lowercase_panels else ("A", "B")
    caption = (
        "**Figure 6. External agreement with early negative-response annotations in 13 independent participants.**\n"
        f"**({a})** Sensitivity and specificity for early-window and broad-window ERPy and frozen archived ER-detect outputs "
        "on the same 32,048 channel–stimulation-pair records with available ERPy results and comparable archived trial sets. "
        "Points are pooled estimates; horizontal bars are 95% percentile intervals from 2,000 whole-participant bootstrap "
        "resamples within site (four Mayo and nine UMCU participants; seed 20260907). Archived outputs are a historical comparison, "
        "not a rerun of the source package. "
        f"**({b})** Participant-level ERPy estimates on all 32,121 evaluable records, including records outside the paired historical "
        "comparison. Circles indicate 10–90 ms and squares 15–300 ms. The right column reports evaluable and source/site-eligible "
        "labeled records; coverage is identical between configurations. Conditional coverage is 32,121/33,694 (95.33%) after source "
        "and site eligibility. An additional 5,901 otherwise eligible labeled records lack matching source events. Including these "
        "records expands the coverage denominator to 39,595 and gives 32,121/39,595 (81.12%; Table 2). This expanded denominator "
        "describes feasibility, not measured detector accuracy; missing source records are not assigned negative detections. "
        "Participant points are descriptive estimates without individual confidence intervals. Each channel–pair record has total "
        "reference weight one, shared equally among its available raters; annotation 2 (positive P1) is negative for the N1 endpoint. "
        "Mayo records within 12 mm of either stimulation contact are excluded. UMCU coordinates are unavailable, so its records have "
        "no spatial exclusion. MAYO01 is excluded because it overlaps the earlier development example. ERPy accepts either polarity; "
        "the broad configuration also admits later responses. Discordance with an N1-negative annotation does not establish absence "
        "of a biological evoked response. The paired estimates describe a sensitivity–specificity tradeoff, not a superiority test.\n"
    )
    alt = (
        f"Two-panel external validation figure. Panel {a} compares 32,048 matched records across 13 independent participants. "
        "Early ERPy has approximately 77% sensitivity and 90% specificity, broad ERPy 70% and 89%, and archived ER-detect 68% and 98%. "
        f"Participant-bootstrap intervals are wider for ERPy. Panel {b} shows participant-level estimates and evaluable/source-and-site-eligible "
        "counts. Conditional coverage is 32,121 of 33,694 records, or 95.33%. Adding 5,901 otherwise eligible records without matching source "
        "events gives expanded coverage of 32,121 of 39,595, or 81.12%; these unavailable records do not alter measured sensitivity or specificity.\n"
    )
    return caption, alt


def plot(metrics_path: Path, destination: Path, *, frontiers: bool = False,
         lowercase_panels: bool = False, width_mm: float = 180.) -> dict:
    if width_mm not in (174., 180.):
        raise ValueError('Publication width must be 174 or 180 mm')
    metrics = pd.read_csv(metrics_path)
    paired, subjects = select_rows(metrics)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": TEXT,
        "axes.labelcolor": TEXT, "axes.titlesize": 9., "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    height_mm = 162. * width_mm / 180.
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    panel_a, panel_b = ("a", "b") if lowercase_panels else ("A", "B")
    fig.text(.045, .957, f"{panel_a}   Matched comparison · 32,048 records", fontsize=10., weight="bold")
    fig.text(.045, .918, "13 independent participants; 95% CIs from bootstrap resampling within site", fontsize=8.5)
    colors = [EARLY, BROAD, ARCHIVED]
    marker = ["o", "s", "D"]
    y = np.array([2., 1., 0.])
    for index, (metric, bounds, ticks, left) in enumerate([
        ("sensitivity", (50, 100), np.arange(50, 101, 10), .275),
        ("specificity", (80, 100), np.arange(80, 101, 5), .67),
    ]):
        ax = fig.add_axes([left, .717, .275, .162])
        clean_axis(ax)
        for i, row in paired.iterrows():
            point, low, high = 100 * row[metric], 100 * row[metric + "_low"], 100 * row[metric + "_high"]
            ax.errorbar(point, y[i], xerr=[[point-low], [high-point]], color=colors[i],
                        fmt=marker[i], markersize=5., markeredgewidth=.8,
                        linewidth=1.25, capsize=3., capthick=.8, zorder=3)
        ax.set(xlim=bounds, ylim=(-.5, 2.5), xticks=ticks, yticks=y)
        ax.set_title(metric.capitalize() + " (%)", pad=8, weight="bold")
        ax.set_yticklabels(paired.plot_label if index == 0 else [""] * 3)
        if index == 0:
            ax.tick_params(axis="y", pad=12)
            # Keep the grid and 100% domain boundary; omit its redundant label
            # so it cannot read as one number with the adjacent specificity axis.
            ax.set_xticklabels([str(t) if t != 100 else "" for t in ticks])

    fig.text(.045, .615, f"{panel_b}   ERPy across participants · 32,121 all-source available records", fontsize=10., weight="bold")
    legend = [Line2D([], [], marker="o", color=EARLY, linestyle="none", markersize=4.5,
                     label="Early (10-90 ms)"),
              Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=BROAD,
                     color=BROAD, linestyle="none", markersize=4.5, label="Broad (15-300 ms)")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.226, .595), ncol=2,
               frameon=False, fontsize=8.5, handletextpad=.5, columnspacing=2.2)
    bottoms, heights = .112, .403
    sensitivity_ax = fig.add_axes([.235, bottoms, .21, heights])
    specificity_ax = fig.add_axes([.49, bottoms, .205, heights])
    coverage_ax = fig.add_axes([.735, bottoms, .23, heights])
    positions = dict(zip(SUBJECTS, np.r_[np.arange(12.8, 8.8, -1), np.arange(8.2, -.8, -1)]))
    for ax, metric, bounds, ticks in [
        (sensitivity_ax, "sensitivity", (0, 100), [0, 25, 50, 75, 100]),
        (specificity_ax, "specificity", (50, 100), [50, 75, 100]),
    ]:
        clean_axis(ax)
        ax.set(xlim=bounds, ylim=(-.6, 13.45), xticks=ticks, yticks=list(positions.values()))
        ax.set_title(metric.capitalize() + " (%)", pad=10, weight="bold")
        if metric == "sensitivity":
            ax.set_xticklabels([str(t) if t != 100 else "" for t in ticks])
        for subject in SUBJECTS:
            dodge = .23 if frontiers else .17
            for configuration, offset, color, shape, fill in [
                ("early_10_90ms", dodge, EARLY, "o", EARLY),
                ("broad_15_300ms", -dodge, BROAD, "s", "white"),
            ]:
                row = subjects.loc[subjects.group.eq(subject) & subjects.configuration.eq(configuration)].iloc[0]
                ax.plot(100 * row[metric], positions[subject] + offset, marker=shape,
                        markersize=5.5 if frontiers else 4.1, markerfacecolor=fill, markeredgecolor=color,
                        markeredgewidth=.9, linestyle="none", zorder=3, clip_on=False)
        ax.set_yticklabels(SUBJECTS if metric == "sensitivity" else [""] * 13)
        ax.tick_params(axis="y", pad=10)
        ax.axhline(9., color="#AAB2B6", linewidth=.6, zorder=1)

    coverage_ax.set(xlim=(0, 1), ylim=(-.6, 13.45))
    coverage_ax.axis("off")
    coverage_ax.set_title("Evaluable / eligible", pad=10, weight="bold")
    for subject in SUBJECTS:
        rows = subjects.loc[subjects.group.eq(subject)]
        if rows.coverage.nunique() != 1 or rows.n_scored_records.nunique() != 1 or rows.eligible.nunique() != 1:
            raise ValueError("Coverage is not shared between configurations; redesign the panel")
        row = rows.iloc[0]
        coverage_ax.text(.02, positions[subject], f"{int(round(row.n_scored_records)):,} / {int(round(row.eligible)):,}",
                         ha="left", va="center", fontsize=8.5)
        coverage_ax.text(.99, positions[subject], f"{100 * row.coverage:.1f}%", ha="right", va="center", fontsize=8.5)
    coverage_ax.axhline(9., color="#AAB2B6", linewidth=.6)
    for label, center in [("Mayo", 11.3), ("UMCU", 4.2)]:
        fig.text(.06, bottoms + heights * (center + .6) / 14.05, label,
                 rotation=90, va="center", ha="center", fontsize=9., weight="bold", color="#56646B")
    fig.text(.235, .062, "Coverage: 95.33% conditional; 81.12% including absent-source records", fontsize=8.5)
    fig.text(.045, .024, "N1-label concordance; ERPy accepts either polarity. Mayo: ≥12 mm; UMCU: no spatial exclusion.", fontsize=8.)

    if frontiers:
        # Apply the journal's >=2 pt requirement to data, auxiliary lines,
        # marker outlines and legend symbols; text remains >=8 pt.
        for artist in fig.findobj(matplotlib.lines.Line2D):
            artist.set_linewidth(max(2., artist.get_linewidth()))
            artist.set_markeredgewidth(max(2., artist.get_markeredgewidth()))
        for artist in fig.findobj(matplotlib.collections.Collection):
            artist.set_linewidth(np.maximum(2., artist.get_linewidth()))
        for ax in fig.axes:
            for spine in ax.spines.values():
                if spine.get_visible(): spine.set_linewidth(2.)

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Preserve exact journal width: never use bbox_inches='tight', which would
    # resize the page around the artist bounds.
    for suffix in ["pdf", "svg"]:
        fig.savefig(destination.with_suffix("." + suffix), facecolor="white")
    fig.savefig(destination.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(destination.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(destination.with_suffix(".tiff")) as source:
        rgb = source.convert("RGB")
    rgb.save(destination.with_suffix(".tiff"), compression="tiff_lzw", dpi=(600, 600))
    fig.savefig(destination.with_name(destination.name + "_publication_scale.png"), dpi=150, facecolor="white")
    fig.canvas.draw()
    canvas = fig.canvas.get_renderer()
    outside = []
    all_text = [artist for artist in fig.findobj(matplotlib.text.Text) if artist.get_visible() and artist.get_text()]
    for artist in all_text:
        box = artist.get_window_extent(canvas)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            outside.append(artist.get_text())
    qa = {"width_mm": width_mm, "height_mm": height_mm, "minimum_visible_font_pt": min(a.get_fontsize() for a in all_text),
          "style": "Frontiers minimum2pt strokes" if frontiers else "standard journal style",
          "outside_canvas_text": outside, "n_participants": 13, "n_paired_available_records": 32048,
          "n_all_source_available_records": 32121, "n_all_source_eligible_labeled_records": 33694,
          "coverage_scope": "Conditional on source and site eligibility",
          "source_event_absent_otherwise_eligible_records": 5901,
          "expanded_coverage_denominator": 39595,
          "expanded_coverage": 32121 / 39595,
          "suppressed_boundary_tick_labels": {"pooled_sensitivity": 100, "participant_sensitivity": 100},
          "ci": "95% percentile,2000whole-subject resamples stratified by site; seed20260907",
          "source_metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
          "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "visual_inspection": "Pending inspection of the rendered publication-scale PNG."}
    if outside or qa["minimum_visible_font_pt"] < 8.:
        raise RuntimeError(f"Figure layout fails size/extent checks: {qa}")
    columns = ["cohort_scope", "configuration", "method", "comparison", "level", "group", "n_subjects", "n_scored_records", "eligible",
               "sensitivity", "sensitivity_low", "sensitivity_high", "specificity", "specificity_low", "specificity_high", "coverage"]
    plotted = pd.concat([paired.assign(panel=panel_a), subjects.assign(panel=panel_b)], ignore_index=True)
    plotted[["panel", *columns]].to_csv(destination.with_name(destination.name + "_data.csv"), index=False)
    destination.with_name(destination.name + "_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    caption, alt = figure_caption(lowercase_panels=lowercase_panels)
    destination.with_name(destination.name + "_caption.md").write_text(caption)
    destination.with_name(destination.name + "_alt.txt").write_text(alt)
    plt.close(fig)
    return qa


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output stem, without an extension")
    parser.add_argument("--frontiers", action="store_true", help="Use >=2 pt visible strokes for Frontiers")
    parser.add_argument("--lowercase-panels", action="store_true", help="Use lowercase a/b for Neuroinformatics")
    parser.add_argument("--width-mm", type=float, choices=[174., 180.], default=180.)
    args = parser.parse_args()
    print(json.dumps(plot(args.metrics, args.output, frontiers=args.frontiers,
                          lowercase_panels=args.lowercase_panels, width_mm=args.width_mm), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
