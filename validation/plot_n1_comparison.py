#!/usr/bin/env python3
"""Render the focused N1 comparison from accompanying saved summary data.

Run with Python, matplotlib, numpy, pandas and Pillow. No model fitting,
resampling or hypothesis testing occurs here. Outputs are deterministic apart
from container metadata; all displayed values come from the supplied CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image


METHODS = ["archived_ER_detect", "N1_logistic_morphology", "N1_logistic_hybrid"]
LABELS = ["ER-detect archive", "N1 morphology", "N1 hybrid"]
COLORS = dict(zip(METHODS, ["#4b5158", "#b27722", "#007d73"]))
METRICS = ["sensitivity", "specificity", "ppv"]
METRIC_LABELS = ["Sensitivity", "Specificity", "PPV"]
VARIANTS = {"jnm": (180.0, False), "frontiers": (180.0, False),
            "neuroinformatics": (174.0, True)}
CURVE_STYLES = {"N1_logistic_morphology": (0, (4, 2)), "N1_logistic_hybrid": "solid"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel_label(fig, x, y, letter, lowercase):
    return fig.text(x, y, letter.lower() if lowercase else letter,
                    fontsize=10, fontweight="bold", va="bottom")


def load_data(data_dir):
    source_manifest = json.loads((data_dir.parent / "source_manifest.json").read_text())
    for name, expected in source_manifest["data_files"].items():
        if digest(data_dir / name) != expected:
            raise ValueError("Pinned plotting data changed: " + name)
    repo = Path(__file__).resolve().parents[1]
    for item in source_manifest["source_files"]:
        if digest(repo / item["repository_relative_path"]) != item["sha256"]:
            raise ValueError("Original N1 evidence changed: " + item["repository_relative_path"])
    points = pd.read_csv(data_dir / "operating_points.csv")
    paired = pd.read_csv(data_dir / "paired_differences.csv")
    assert len(points) == 9 and set(points.method) == set(METHODS)
    assert len(paired) == 6 and set(paired.method) == {METHODS[2]}
    assert set(paired.reference) == set(METHODS[:2])
    assert set(points.metric) == set(paired.metric) == set(METRICS)
    assert (points.n_records == 32048).all() and (points.n_subjects == 13).all()
    for row in paired.itertuples():
        left = points[(points.method == row.method) & (points.metric == row.metric)].estimate.item()
        right = points[(points.method == row.reference) & (points.metric == row.metric)].estimate.item()
        assert abs(left - right - row.difference) < 1e-12
        assert bool(row.ci_excludes_zero) == (row.low > 0 or row.high < 0)
    curves = {m: pd.read_csv(data_dir / f"{m}_precision_recall.csv.gz")
              for m in METHODS[1:]}
    return points, paired, curves


def render(variant, output_dir, data_dir):
    width_mm, lowercase = VARIANTS[variant]
    height_mm = 163.0 * width_mm / 180.0
    output_dir.mkdir(parents=True, exist_ok=True)
    points, paired, curves = load_data(data_dir)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 2, "xtick.major.width": 2, "ytick.major.width": 2,
        "lines.linewidth": 2, "lines.markeredgewidth": 2,
        "savefig.facecolor": "white", "svg.hashsalt": "ERPy-focused-N1-20260908",
    })
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.text(.045, .977, "Negative-N1 comparison", fontsize=10,
             fontweight="bold", va="top")
    fig.text(.965, .975, "32,048 records · 13 participants", fontsize=8,
             ha="right", va="top")
    axes = []
    top_specs = [
        ("sensitivity", "Sensitivity (%)", (60, 75), [60, 65, 70, 75]),
        ("specificity", "Specificity (%)", (95, 100), [95, 97.5, 100]),
        ("ppv", "Positive predictive value (%)", (70, 91), [70, 80, 90]),
    ]
    for col, (metric, title, limits, ticks) in enumerate(top_specs):
        left = .195 + col * .257
        ax = fig.add_axes([left, .743, .223, .173])
        axes.append(ax)
        for i, method in enumerate(METHODS):
            row = points[(points.method == method) & (points.metric == metric)].iloc[0]
            estimate, low, high = 100 * row[["estimate", "low", "high"]].to_numpy(float)
            y = 2 - i
            ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]],
                        fmt="o", color=COLORS[method], ms=4.6, capsize=3,
                        capthick=2, elinewidth=2, zorder=3)
            ax.annotate(f"{estimate:.2f}", (estimate, y), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=COLORS[method])
        ax.set(xlim=limits, ylim=(-.5, 2.8), xticks=ticks,
               yticks=[2, 1, 0], xlabel=title)
        ax.set_yticklabels(LABELS if col == 0 else [])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        ax.tick_params(axis="y", length=0, pad=7)
        ax.xaxis.labelpad = 6
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="#e7eaec", linewidth=2)
        ax.set_axisbelow(True)
        panel_label(fig, left - .024, .928, "ABC"[col], lowercase)

    # Curves use pooled participant-held-out scores. The archive supplies one
    # binary operating point, not a score sweep or an interpolated curve.
    pr = fig.add_axes([.102, .203, .346, .385])
    axes.append(pr)
    for method in METHODS[1:]:
        curve = curves[method]
        pr.step(100 * curve.recall, 100 * curve.precision, where="post",
                color=COLORS[method], linestyle=CURVE_STYLES[method], linewidth=2, zorder=3)
    archive = points[points.method == METHODS[0]].set_index("metric")
    pr.plot(100 * archive.at["sensitivity", "estimate"],
            100 * archive.at["ppv", "estimate"], marker="D", linestyle="none",
            color=COLORS[METHODS[0]], ms=5, markeredgewidth=2, zorder=5)
    pr.set(xlim=(0, 100), ylim=(0, 103), xticks=[0, 25, 50, 75, 100],
           yticks=[0, 25, 50, 75, 100], xlabel="Sensitivity (recall, %)",
           ylabel="Positive predictive value (%)")
    pr.xaxis.labelpad = 6
    pr.yaxis.labelpad = 6
    pr.grid(color="#e7eaec", linewidth=2)
    pr.set_axisbelow(True)
    panel_label(fig, .054, .622, "D", lowercase)
    fig.text(.103, .624, "Precision–recall", fontsize=9, va="bottom")
    pr_handles = [Line2D([], [], color=COLORS[m], lw=2, linestyle=CURVE_STYLES[m],
                        label=LABELS[METHODS.index(m)]) for m in METHODS[1:]]
    pr_handles.append(Line2D([], [], color=COLORS[METHODS[0]], marker="D",
                             linestyle="none", markersize=5, markeredgewidth=2,
                             label="ER-detect archive (point)"))
    fig.legend(handles=pr_handles, loc="upper left", bbox_to_anchor=(.102, .135),
               frameon=False, handlelength=2.0, handletextpad=.7,
               labelspacing=.6, borderaxespad=0)

    # Six paired differences share one percentage-point axis and one zero line.
    # Marker fill conveys literal interval exclusion; it does not encode p.
    delta = fig.add_axes([.631, .203, .334, .385])
    axes.append(delta)
    row_y = [5.6, 4.6, 3.6, 1.6, .6, -.4]
    for group, reference in enumerate(METHODS[:2]):
        for i, metric in enumerate(METRICS):
            row = paired[(paired.reference == reference) & (paired.metric == metric)].iloc[0]
            estimate, low, high = 100 * row[["difference", "low", "high"]].to_numpy(float)
            excludes = bool(row.ci_excludes_zero)
            color = COLORS[METHODS[2]] if excludes else "#4b5158"
            delta.errorbar(estimate, row_y[3 * group + i],
                           xerr=[[estimate - low], [high - estimate]],
                           fmt="o", color=color, mfc=color if excludes else "white",
                           mec=color, ms=5, capsize=3, capthick=2,
                           elinewidth=2, markeredgewidth=2, zorder=3)
    delta.set(xlim=(-6.5, 11.5), ylim=(-1.1, 7.0), xticks=[-5, 0, 5, 10],
              yticks=row_y, yticklabels=METRIC_LABELS * 2,
              xlabel="Hybrid − comparator (percentage points)")
    delta.xaxis.labelpad = 6
    delta.tick_params(axis="y", length=0, pad=6)
    delta.spines["left"].set_visible(False)
    delta.grid(axis="x", color="#e7eaec", linewidth=2)
    delta.axvline(0, color="#737b82", lw=2, linestyle=(0, (3, 3)), zorder=2)
    delta.set_axisbelow(True)
    delta.text(.01, 6.45, "Hybrid − ER-detect archive",
               transform=delta.get_yaxis_transform(), fontsize=8,
               fontweight="bold", va="center",
               bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    delta.text(.01, 2.45, "Hybrid − N1 morphology",
               transform=delta.get_yaxis_transform(), fontsize=8,
               fontweight="bold", va="center",
               bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    panel_label(fig, .548, .622, "E", lowercase)
    fig.text(.598, .624, "Paired differences", fontsize=9, va="bottom")
    interval_handles = [
        Line2D([], [], color=COLORS[METHODS[2]], marker="o", linestyle="none",
               markersize=5, markeredgewidth=2, label="95% CI excludes 0"),
        Line2D([], [], color="#4b5158", marker="o", markerfacecolor="white",
               linestyle="none", markersize=5, markeredgewidth=2,
               label="95% CI includes 0"),
    ]
    fig.legend(handles=interval_handles, loc="upper left", bbox_to_anchor=(.59, .135),
               frameon=False, handlelength=1.5, handletextpad=.7,
               labelspacing=.6, borderaxespad=0)
    fig.text(.965, .025, "Nominal conditional 95% CIs; no multiplicity adjustment.",
             fontsize=8, ha="right", va="bottom")

    # Guard the canvas boundary and mandatory publication specifications. The
    # signed visual receipt remains a separate, necessary inspection artifact.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    visible_fonts = []
    text_boxes = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        b = artist.get_window_extent(renderer)
        text_boxes.append((artist.get_text(), b))
        visible_fonts.append(artist.get_fontsize())
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.x1 + .5 or b.y1 > fig.bbox.y1 + .5:
            outside.append(artist.get_text())
    assert not outside, outside
    text_overlaps = []
    for i, (left_text, left_box) in enumerate(text_boxes):
        for right_text, right_box in text_boxes[i + 1:]:
            width = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            height = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if width > .5 and height > .5:
                text_overlaps.append([left_text, right_text])
    assert not text_overlaps, text_overlaps
    assert min(visible_fonts) >= 8
    stem = output_dir / "fig07_n1_development"
    for ext in ["svg", "pdf", "png"]:
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        if ext == "svg":
            metadata = {"Date": None}
        fig.savefig(stem.with_suffix("." + ext), dpi=600, metadata=metadata)
    with Image.open(stem.with_suffix(".png")) as png:
        png.convert("RGB").save(stem.with_suffix(".tiff"),
                                compression="tiff_lzw", dpi=(600, 600))
        pixels = list(png.size)
    plt.close(fig)
    abc, d, e = ("a–c", "d", "e") if lowercase else ("A–C", "D", "E")
    caption = (
        "Negative-N1 comparison on 32,048 records from 13 participants. "
        f"({abc}) Operating-point sensitivity, specificity and positive predictive value "
        "(PPV), with nominal 95% participant-cluster bootstrap intervals. "
        f"({d}) Precision–recall curves from pooled participant-held-out morphology and "
        "hybrid scores; the diamond is the archived ER-detect operating point. "
        "Curves sweep scores, whereas classifier operating points use fold-specific thresholds. "
        f"({e}) Paired hybrid-minus-comparator differences; filled markers identify intervals "
        "excluding zero. Intervals condition on the saved development predictions and are "
        "not multiplicity-adjusted. The cohort informed development, so these results do "
        "not establish external validity or equivalence."
    )
    alt = (
        "Five-panel negative-N1 comparison using 32,048 records from 13 participants. "
        "Three upper panels show operating-point estimates and 95% intervals for archived "
        "ER-detect, N1 morphology and N1 hybrid. Their sensitivity is 67.61%, 67.28% and "
        "67.39%; specificity is 97.64%, 97.48% and 97.96%; positive predictive value is "
        "81.17%, 80.07% and 83.25%, respectively. At lower left are the dashed morphology "
        "and solid hybrid precision–recall curves and the archived ER-detect diamond. At lower right, six "
        "paired hybrid-minus-comparator differences have a dashed zero line. All three "
        "intervals against archived ER-detect include zero. Against morphology, the "
        "specificity interval, 0.07 to 1.03 percentage points, and PPV interval, 1.16 to "
        "4.65 points, exclude zero and have filled markers; the sensitivity interval "
        "includes zero and has an open marker. All intervals are nominal conditional "
        "bootstrap intervals without multiplicity adjustment."
    )
    (output_dir / "fig07_n1_development_caption.txt").write_text(caption + "\n")
    (output_dir / "fig07_n1_development_alt.txt").write_text(alt + "\n")
    manifest = {
        "variant": variant, "width_mm": width_mm, "height_mm": height_mm,
        "minimum_font_pt": min(visible_fonts), "minimum_visible_stroke_pt": 2,
        "raster_dpi": 600, "raster_pixel_dimensions": pixels,
        "lowercase_panels": lowercase, "outside_text": outside,
        "precision_recall_line_styles": {"N1_logistic_morphology": "dashed", "N1_logistic_hybrid": "solid"},
        "overlapping_text_bounding_boxes": text_overlaps,
        "methods": METHODS, "operating_point_rows": 9,
        "paired_difference_rows": 6, "paired_intervals_excluding_zero": 2,
        "inference": "Nominal conditional participant-cluster bootstrap 95% intervals for fixed participant-held-out development predictions; no multiplicity adjustment, p-value stars, refitting, new statistical test, or equivalence claim.",
        "generator_sha256": digest(Path(__file__)),
        "data_files": {p.name: digest(p) for p in sorted(data_dir.iterdir()) if p.is_file()},
        "files": {p.name: digest(p) for p in sorted(output_dir.iterdir())
                  if p.is_file() and p.name.startswith("fig07_n1_development")
                  and p.suffix in {".svg", ".pdf", ".png", ".tiff", ".txt"}},
        "visual_inspection_required": True,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=base / "n1_comparison" / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["all", *VARIANTS], default="all")
    args = parser.parse_args()
    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        m = render(variant, args.output_dir / variant, args.data_dir)
        print(json.dumps({"variant": variant, "outside_text": m["outside_text"],
                          "width_mm": m["width_mm"], "pixels": m["raster_pixel_dimensions"]}))


if __name__ == "__main__":
    main()
