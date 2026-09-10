#!/usr/bin/env python3
"""Draw Figure 3 from packaged aggregate counts, without private recordings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

DEFAULT_DATA = Path(__file__).parent / "figure_data/qc_aggregate_summary.json"


def figure(data_path: Path = DEFAULT_DATA):
    data = json.loads(data_path.read_text())
    style = {"font.family": "DejaVu Sans", "font.size": 8.5,
             "axes.labelsize": 8.5, "axes.titlesize": 9,
             "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
             "svg.fonttype": "none", "pdf.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(style)
        fig = plt.figure(figsize=(180 / 25.4, 172 / 25.4))
        fig._height_mm = 172
        grid = fig.add_gridspec(2, 2, left=.24, right=.975, bottom=.09, top=.91,
                               wspace=.28, hspace=.80, height_ratios=[1, 1.16])
        axs = [fig.add_subplot(g) for g in grid]
        labels = ["Amplifier / motion", "Sharp transient", "Late roughness", "Extreme amplitude"]
        colors = ["#D55E00", "#E69F00", "#56B4E9", "#9B67A8"]
        keys = ["amplifier_or_motion_artifact", "sharp_transient",
                "late_high_frequency_outlier", "extreme_amplitude_outlier"]
        for ax, letter, title in [(axs[0], "A", "Clinical acquisition"),
                                  (axs[1], "B", "Public demonstration")]:
            rows = {r["reason"]: r for r in data["flag_counts"] if r["panel"] == letter}
            counts = [rows[key]["contacts"] for key in keys]
            denominators = {row["denominator"] for row in rows.values()}
            if len(denominators) != 1:
                raise ValueError("QC flag denominator differs within a panel")
            total = denominators.pop()
            y = np.arange(4)
            ax.barh(y, counts, color=colors, height=.57)
            ax.set_yticks(y, labels if letter == "A" else [""] * 4)
            ax.invert_yaxis()
            ax.set_xlim(0, max(counts) * 1.22)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
            ax.set_xlabel("Contacts with ≥1 flag")
            ax.set_title(f"{letter}  {title}", loc="left", pad=23, fontweight="bold")
            ax.text(0, 1.025, f"{total} contacts", transform=ax.transAxes, va="bottom")
            ax.grid(axis="x", alpha=.22)
            ax.set_axisbelow(True)
            for i, value in enumerate(counts):
                ax.text(value + max(counts) * .025, i, str(value), va="center", fontsize=8.5)
        ax = axs[2]
        classes = ["Both", "Reproducibility only", "Energy only", "Neither"]
        y = np.arange(4)
        for source, offset, color, label in [("clinical", -.17, "#0072B2", "Clinical"),
                                              ("public", .17, "#009E73", "Public")]:
            values = np.asarray(data["classes"][source])
            denominator = data["accounting"][source]["non_stimulation"]
            if values.sum() != denominator:
                raise ValueError("Component classes do not partition their contact family")
            ax.barh(y + offset, values / denominator * 100, height=.30, color=color, label=label)
        ax.set_yticks(y, classes)
        ax.invert_yaxis()
        ax.set_xlim(0, 80)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlabel("Non-stimulation contacts (%)")
        ax.set_title("C  Unadjusted component calls", loc="left", pad=25, fontweight="bold")
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", bbox_to_anchor=(-.01, 1.0), ncol=2, frameon=False,
                  handlelength=1.0, columnspacing=.8, borderaxespad=0)
        ax = axs[3]
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("D  Accounting and call scope", loc="left", pad=25, fontweight="bold")
        lines = [("", "Clinical", "Public")]
        for label, key in [("Loaded", "loaded"), ("Non-stim.", "non_stimulation"),
                           ("QC eligible", "qc_eligible"), ("QC ∩ non-stim.", "qc_and_non_stimulation"),
                           ("BH calls", "bh_calls")]:
            values = [data["accounting"][source][key] for source in ["clinical", "public"]]
            lines.append((label, *["—" if value is None else value for value in values]))
        for j, line in enumerate(lines):
            for x, value, align in zip([-.10, .60, .98], line, ["left", "right", "right"]):
                ax.text(x, 1 - j * .125, str(value), transform=ax.transAxes, ha=align,
                        va="center", fontsize=8.5, fontweight="bold" if j == 0 else "normal")
        ax.text(-.10, .15, "QC and non-stimulation sets differ.\nBH counts precede an independent\nacquisition-wide QC join.",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5, linespacing=1.35)
        fig.canvas.draw()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True, help="Output stem")
    args = parser.parse_args()
    fig = figure(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png", "tiff"]:
        path = args.output.with_suffix("." + suffix)
        fig.savefig(path, dpi=600, facecolor="white")
        if suffix == "tiff":
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            rgb.save(path, dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)


if __name__ == "__main__":
    main()
