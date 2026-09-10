#!/usr/bin/env python3
"""Create lowercase-panel Neuroinformatics artwork from the revised figures.

Keep 180-mm width pending confirmation of the journal's production format.
Only panel-letter case changes; plot values, data geometry and base styles stay.
Also add 600-dpi TIFFs for base Figures 1 and 5, without replacing other base files.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lower_panels(fig):
    changes = []
    for artist in fig.findobj(Text):
        original = artist.get_text()
        if re.match(r"^[A-F]\s{2,}", original):
            revised = original[0].lower() + original[1:]
            artist.set_text(revised)
            changes.append([original, revised])
    return changes


def tiff(fig, path):
    fig.savefig(path, dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    rgb.save(path, dpi=(600, 600), compression="tiff_lzw")


def export(fig, stem, builder):
    report = builder.layout_audit(fig)
    if report["clipped_text"] or report["text_collisions"]:
        raise ValueError(f"Figure requires layout review: {report}")
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(stem.with_suffix("." + extension), dpi=600, facecolor="white")
    tiff(fig, stem.with_suffix(".tiff"))
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = Path(__file__).parent / "figure_data"
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=data / "synthetic_benchmark")
    parser.add_argument("--recovery", type=Path, default=data / "stratified_recovery.json")
    parser.add_argument("--qc-data", type=Path, default=data / "qc_aggregate_summary.json")
    parser.add_argument("--metrics", type=Path, default=Path(__file__).parent / "frozen_results/erdetect/metrics.csv")
    parser.add_argument("--figures", type=int, nargs="+", choices=range(1, 7), default=[1, 3, 5, 6],
                        help="Figures 2/4 additionally require their frozen base SVGs in --base-dir")
    args = parser.parse_args()
    out = args.base_dir / "neuroinformatics"
    out.mkdir(exist_ok=True, parents=True)
    directory = Path(__file__).parent
    frontiers = load_module("frontiers_helpers", directory / "export_frontiers_figures.py")
    builder = load_module("native_figures", directory / "plot_manuscript_validation_figures.py")
    recovery = json.loads(args.recovery.read_text())
    reports = {}
    for number in [n for n in args.figures if n != 6]:
        stem = frontiers.STEMS[number]
        if number in [2, 4]:
            if not (args.base_dir / (stem + ".svg")).is_file():
                raise FileNotFoundError(f"Figure {number} requires its frozen display SVG in --base-dir; raw cohort reconstruction is outside this exporter.")
            tree = ET.parse(args.base_dir / (stem + ".svg"))
            changes = []
            for element in tree.getroot().iter("{http://www.w3.org/2000/svg}text"):
                if element.text and element.text.strip() in {"A", "B", "C"}:
                    original = element.text
                    element.text = original.lower()
                    changes.append([original, element.text])
            if len(changes) != 3:
                raise ValueError(f"Expected exactly three panel letters in Figure {number}: {changes}")
            metadata_changes = frontiers.normalize_figure_metadata(tree.getroot())
            tree.write(out / (stem + ".svg"), encoding="utf-8", xml_declaration=True)
            frontiers.cairo_exports(out / (stem + ".svg"), out / stem)
            reports[str(number)] = {"panel_changes": changes, "metadata_title_changes": metadata_changes,
                                    "geometry": "Source SVG plotted geometry and image unchanged"}
            continue
        fig = (builder.workflow() if number == 1 else
               frontiers.qc_figure(args.qc_data) if number == 3 else
               builder.synthetic(args.benchmark, recovery))
        if number in [1, 5]:
            tiff(fig, args.base_dir / (stem + ".tiff"))
        changes = lower_panels(fig)
        reports[str(number)] = {"panel_changes": changes, "layout": export(fig, out / stem, builder)}
    if 6 in args.figures:
        external = load_module("external_figure", directory / "plot_erdetect_validation.py")
        report = external.plot(args.metrics, out / frontiers.STEMS[6], lowercase_panels=True)
        reports["6"] = {"panel_changes": [["A", "a"], ["B", "b"]], "native_plotter_report": report}
    for number, report in reports.items():
        stem = frontiers.STEMS[int(number)]
        report["files"] = {suffix: {"name": stem + "." + suffix,
                           "sha256": hashlib.sha256((out / (stem + "." + suffix)).read_bytes()).hexdigest()}
                           for suffix in ["png", "pdf", "svg", "tiff"]}
    (out / "neuroinformatics_export_manifest.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "width_mm": 180, "tiff_dpi": 600, "figures": reports,
        "format_note": "The guide gives conditional 174-mm and 119-mm formats without identifying this journal's format. Width remains 180 mm as authorized pending production-format confirmation.",
        "typography_note": "The plain-text floor is 8 pt; conventional mathematical subscripts/superscripts retain smaller sizes in these base-style variants.",
        "visual_qa": "Pending final PDF inspection",
    }, indent=2) + "\n")
    print(json.dumps({k: v["panel_changes"] for k, v in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
