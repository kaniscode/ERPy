#!/usr/bin/env python3
"""Render the fixed label-free N1 comparison from recorded evaluation tables.

No fitting, threshold selection, prediction or resampling occurs in this file.
The supervised comparison is retained in the accompanying secondary tables.
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
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

METHODS = ["archived_ER_detect", "N1_nominal_screen", "label_free_N1"]
LABELS = ["ER-detect archive", "ERPy N1 screen", "ERPy N1 detection"]
COLORS = {"archived_ER_detect": "#555d63", "N1_nominal_screen": "#a36b16",
          "label_free_N1": "#007d73"}
METRICS = ["sensitivity", "specificity", "ppv"]
METRIC_LABELS = ["Sensitivity", "Specificity", "PPV"]
VARIANTS = {"jnm": (180.0, False), "frontiers": (180.0, False),
            "neuroinformatics": (174.0, True)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data(data_dir):
    points = pd.read_csv(data_dir / "operating_points.csv")
    points = points.loc[points.method.isin(METHODS) & points.metric.isin(METRICS)].copy()
    paired = pd.read_csv(data_dir / "paired_differences.csv")
    paired = paired.loc[paired.method.isin(METHODS[1:])
                        & paired.reference.eq("archived_ER_detect")
                        & paired.metric.isin(METRICS)].copy()
    if len(points) != 9 or points.duplicated(["method", "metric"]).any():
        raise ValueError("Expected nine unique operating points for archive and both N1 stages")
    if len(paired) != 6 or paired.duplicated(["method", "metric"]).any():
        raise ValueError("Expected six unique paired differences")
    if not ((points.n_records == 32048).all() and (points.n_subjects == 13).all()):
        raise ValueError("The declared matched evaluation frame changed")
    if not np.isfinite(points[["estimate", "low", "high"]]).all().all():
        raise ValueError("All plotted operating-point quantities must be finite")
    if not (points.low.between(0, 1).all() and points.high.between(0, 1).all()):
        raise ValueError("Probability intervals outside [0,1]")
    for row in paired.itertuples():
        left = points.loc[(points.method == row.method) & (points.metric == row.metric), "estimate"].item()
        right = points.loc[(points.method == row.reference) & (points.metric == row.metric), "estimate"].item()
        if abs(left - right - row.difference) > 1e-12:
            raise ValueError("Paired difference does not equal the unrounded point difference")
        expected = row.low > 0 or row.high < 0
        actual = str(row.ci_excludes_zero).lower() == "true"
        if actual != expected:
            raise ValueError("Interval-exclusion flag disagrees with interval bounds")
    return points, paired


def panel_label(fig, x, y, letter, lowercase):
    return fig.text(x, y, letter.lower() if lowercase else letter,
                    fontsize=10, fontweight="bold", va="bottom")


def render(variant, output_dir, data_dir):
    width_mm, lowercase = VARIANTS[variant]
    height_mm = 163.0 * width_mm / 180.0
    points, paired = read_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 2, "xtick.major.width": 2, "ytick.major.width": 2,
        "lines.linewidth": 2, "lines.markeredgewidth": 2,
        "savefig.facecolor": "white", "svg.hashsalt": "ERPy-label-free-negative-N1",
    })
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.text(.045, .976, "Label-free N1 screening and detection", fontsize=10,
             fontweight="bold", va="top")
    fig.text(.965, .970, "32,048 records · 13 participants", fontsize=8,
             ha="right", va="top")
    axes = []
    for col, metric in enumerate(METRICS):
        left = .202 + col * .253
        ax = fig.add_axes([left, .715, .221, .192])
        axes.append(ax)
        for i, method in enumerate(METHODS):
            row = points.loc[(points.method == method) & (points.metric == metric)].iloc[0]
            estimate, low, high = 100 * row[["estimate", "low", "high"]].to_numpy(float)
            y = 2 - i
            # Draw the interval directly: percentile intervals need not contain
            # the original estimate, so do not assume positive error lengths.
            ax.hlines(y, low, high, color=COLORS[method], linewidth=2, zorder=3)
            ax.vlines([low, high], y-.075, y+.075, color=COLORS[method], linewidth=2, zorder=3)
            ax.plot(estimate, y, marker="o", color=COLORS[method], ms=4.6,
                    linestyle="none", zorder=4)
            ax.annotate(f"{estimate:.2f}", (estimate, y), xytext=(0, 7),
                        textcoords="offset points", ha="right" if estimate >= 90 else "center",
                        va="bottom", fontsize=8, color=COLORS[method])
        # The explicitly labeled specificity scale resolves narrow intervals.
        # Unlabeled right padding preserves complete marker circles near 100%.
        limits = (94, 100.4) if metric == "specificity" else (0, 104)
        ticks = [94, 96, 98, 100] if metric == "specificity" else [0, 25, 50, 75, 100]
        ax.set(xlim=limits, ylim=(-.55, 2.8), xticks=ticks,
               yticks=[2, 1, 0], xlabel=METRIC_LABELS[col] + " (%)")
        ax.set_yticklabels(LABELS if col == 0 else [])
        ax.tick_params(axis="y", length=0, pad=7)
        ax.xaxis.labelpad = 6
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="#e7eaec", linewidth=2)
        ax.set_axisbelow(True)
        panel_label(fig, left-.027, .920, "ABC"[col], lowercase)

    bounds = 100 * paired[["low", "high"]].to_numpy(float)
    lo, hi = min(0.0, float(bounds.min())), max(0.0, float(bounds.max()))
    span = max(hi-lo, 4.0)
    for method, bottom, panel, heading in [
            ("N1_nominal_screen", .414, "D", "Nominal N1 screen − ER-detect archive"),
            ("label_free_N1", .151, "E", "Family-adjusted N1 detection − ER-detect archive")]:
        height = .154
        forest = fig.add_axes([.202, bottom, .522, height])
        axes.append(forest)
        forest.set_xlim(lo-.12*span, hi+.12*span)
        forest.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        for i, metric in enumerate(METRICS):
            row = paired.loc[paired.method.eq(method) & paired.metric.eq(metric)].iloc[0]
            estimate, low, high = 100 * row[["difference", "low", "high"]].to_numpy(float)
            excludes = low > 0 or high < 0
            y = 2-i
            color = COLORS[method] if excludes else "#555d63"
            forest.hlines(y, low, high, color=color, linewidth=2, zorder=3)
            forest.vlines([low, high], y-.09, y+.09, color=color, linewidth=2, zorder=3)
            forest.plot(estimate, y, marker="o", linestyle="none", color=color,
                        mfc=color if excludes else "white", mec=color, ms=5, zorder=4)
            fig.text(.758, bottom+height*(y+.65)/3.3,
                     f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}]", fontsize=8,
                     ha="left", va="center")
        forest.set(ylim=(-.65, 2.65), yticks=[2, 1, 0], yticklabels=METRIC_LABELS)
        if method == "label_free_N1":
            forest.set_xlabel("Difference (percentage points)")
            forest.xaxis.labelpad = 6
        forest.tick_params(axis="y", length=0, pad=7)
        forest.spines["left"].set_visible(False)
        forest.grid(axis="x", color="#e7eaec", linewidth=2)
        forest.set_axisbelow(True)
        forest.axvline(0, color="#737b82", linewidth=2, linestyle=(0, (3, 3)), zorder=2)
        panel_label(fig, .175, bottom+height+.035, panel, lowercase)
        fig.text(.202, bottom+height+.035, heading, fontsize=9, va="bottom")
        fig.text(.758, bottom+height+.011, "Difference [95% CI]", fontsize=8, va="bottom")
    handles = [
        Line2D([], [], color=COLORS["label_free_N1"], marker="o", linestyle="none",
               ms=5, label="CI excludes zero"),
        Line2D([], [], color="#555d63", mfc="white", marker="o", linestyle="none",
               ms=5, label="CI includes zero"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.193, .025),
               ncol=2, frameon=False, handletextpad=.6, columnspacing=1.8,
               borderaxespad=0)
    fig.text(.966, .042, "Paired, nominal 95% intervals", fontsize=8,
             ha="right", va="center")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_bounds = fig.bbox
    marker_bounds = []
    for ax in axes:
        for line in ax.lines:
            if line.get_marker() != "o":
                continue
            box = line.get_window_extent(renderer)
            inside = (box.x0 >= ax.bbox.x0 and box.x1 <= ax.bbox.x1
                      and box.y0 >= ax.bbox.y0 and box.y1 <= ax.bbox.y1)
            marker_bounds.append({"bounds_px": list(box.bounds), "inside_axes": bool(inside)})
            if line.get_clip_on() and not inside:
                raise ValueError("A point marker extends outside its clipping axes")
    text_bounds = []
    for obj in fig.findobj(matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        box = obj.get_window_extent(renderer)
        # Tick labels outside the axes' visible range are not painted.
        if obj.axes and box.width == 0:
            continue
        text_bounds.append({"text": obj.get_text(), "font_size_pt": obj.get_fontsize(),
                            "bounds_px": list(box.bounds),
                            "within_canvas": bool(box.x0 >= -1 and box.y0 >= -1
                                                  and box.x1 <= canvas_bounds.width+1
                                                  and box.y1 <= canvas_bounds.height+1)})
    stem = output_dir / "fig07_label_free_n1"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "ERPy", "CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    with Image.open(stem.with_suffix(".png")) as im:
        im.convert("RGB").save(stem.with_suffix(".tiff"), dpi=(600,600), compression="tiff_lzw")
        pixels = im.size
    fig.savefig(output_dir / "publication_scale.png", dpi=160)
    plt.close(fig)
    info = {"journal": variant, "width_mm": width_mm, "height_mm": height_mm,
            "dpi": 600, "pixel_dimensions": list(pixels),
            "minimum_text_pt": min(r["font_size_pt"] for r in text_bounds),
            "panel_labels": "abcde" if lowercase else "ABCDE",
            "source_tables": {p.name: digest(p) for p in [data_dir/"operating_points.csv", data_dir/"paired_differences.csv"]},
            "generator_sha256": digest(__file__),
            "assets": {ext: {"name": stem.with_suffix("."+ext).name, "sha256": digest(stem.with_suffix("."+ext))}
                       for ext in ["svg", "pdf", "png", "tiff"]},
            "text_bounds": text_bounds,
            "marker_bounds": marker_bounds,
            "visual_qa_status": "pending_direct_inspection"}
    (output_dir / "figure_manifest.json").write_text(json.dumps(info, indent=2)+"\n")
    return {"journal": variant, "output": str(output_dir), "pixels": pixels,
            "outside_canvas": [r["text"] for r in text_bounds if not r["within_canvas"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--journal", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()
    variants = VARIANTS if args.journal == "all" else [args.journal]
    for journal in variants:
        print(json.dumps(render(journal, args.output_dir / journal, args.data_dir)))
