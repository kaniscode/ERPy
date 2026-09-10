#!/usr/bin/env python3
"""Export journal-specific figures with physical strokes of at least 2 points.

Figures 1/3/5/6 are rendered natively from packaged aggregate inputs.
Figures 2/4 require authorized frozen display SVGs. Their plotted geometry and
embedded images are preserved; documented stroke, mathematical-label and
metadata-title edits are display changes, not a raw-data replay. Equivalent font
shorthand is expanded in the in-memory Cairo input for reliable font weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from xml.etree import ElementTree as etree
from PIL import Image

STEMS = {
    1: "fig01_workflow_revised", 2: "fig02_selected_local_example",
    3: "fig03_artifact_qc", 4: "fig04_crp_energy_local_evidence",
    5: "fig05_synthetic_validation_revised", 6: "fig06_external_validation",
}
MIN_PT = 2.000001
for prefix, uri in [("", "http://www.w3.org/2000/svg"),
                    ("xlink", "http://www.w3.org/1999/xlink"),
                    ("dc", "http://purl.org/dc/elements/1.1/"),
                    ("cc", "http://creativecommons.org/ns#"),
                    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")]:
    etree.register_namespace(prefix, uri)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_figure_metadata(root):
    """Use the primary detector name in legacy frozen SVG metadata only."""
    changes = []
    for element in root.iter():
        if (isinstance(element.tag, str) and element.tag.endswith("}title")
                and element.text and "CRP–energy" in element.text):
            original = element.text
            element.text = original.replace("CRP–energy", "Projection–energy")
            changes.append([original, element.text])
    return changes


def properties(value):
    return {
        key.strip(): val.strip() for part in value.split(";") if ":" in part
        for key, val in [part.split(":", 1)]}


def physical_points(value):
    number, unit = re.fullmatch(r"([\d.]+)(pt|mm|cm|in|px)?", value).groups()
    return float(number) * {"pt": 1, "mm": 72 / 25.4, "cm": 72 / 2.54,
                            "in": 72, "px": .75, None: .75}[unit]


def point_scale(root):
    viewport = list(map(float, root.attrib["viewBox"].split()))
    scale_x = physical_points(root.attrib["width"]) / viewport[2]
    scale_y = physical_points(root.attrib["height"]) / viewport[3]
    if not np.isclose(scale_x, scale_y, rtol=1e-7):
        raise ValueError("Non-isotropic root scaling requires a separate stroke audit")
    return min(scale_x, scale_y)


def without_stroke_width(root, exclude_stroke_color=False):
    """Compare every SVG node, geometry/text attribute and embedded image."""
    result = []
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        attributes = dict(element.attrib)
        attributes.pop("stroke-width", None)
        if exclude_stroke_color:
            attributes.pop("stroke", None)
        if "style" in attributes:
            style = properties(attributes["style"])
            style.pop("stroke-width", None)
            if exclude_stroke_color:
                style.pop("stroke", None)
            attributes["style"] = tuple(sorted(style.items()))
        result.append((element.tag, tuple(sorted(attributes.items())), element.text, element.tail))
    return result


def stroke_audit(path):
    root = etree.parse(str(path)).getroot()
    scale = point_scale(root)
    widths = []
    for element in root.iter():
        style = properties(element.get("style", ""))
        raw = style.get("stroke-width", element.get("stroke-width"))
        if raw is not None and float(raw) > 0:
            widths.append(float(raw) * scale)
    return {"width_mm": physical_points(root.get("width")) * 25.4 / 72,
            "height_mm": physical_points(root.get("height")) * 25.4 / 72,
            "explicit_positive_stroke_count": len(widths),
            "minimum_explicit_stroke_pt": min(widths) if widths else None,
            "strokes_below_2pt": sum(x < 2 - 1e-6 for x in widths)}


def restyle_svg(source, destination):
    tree = etree.parse(str(source))
    root = tree.getroot()
    before = without_stroke_width(root)
    before_all_strokes = without_stroke_width(root, exclude_stroke_color=True)
    scale = point_scale(root)
    floor = MIN_PT / scale
    changes = 0
    parents = {child: node for node in root.iter() for child in node}
    for element in root.iter():
        style = properties(element.get("style", ""))
        raw = style.get("stroke-width", element.get("stroke-width"))
        stroke = style.get("stroke", element.get("stroke"))
        # All artwork strokes in these Matplotlib SVGs are declared inline.
        # Explicit positive widths also cover markers defined under <defs>.
        if raw is None and stroke and stroke != "none":
            raw = "1"
        if raw is None or float(raw) <= 0:
            continue
        # Stroke-bearing geometry is unscaled in the source figures; reject
        # unexpected scaling rather than silently underestimate physical width.
        nodes = [element]
        while nodes[-1] in parents:
            nodes.append(parents[nodes[-1]])
        for node in nodes:
            transform = node.get("transform", "")
            if "scale" in transform or "matrix" in transform:
                raise ValueError(f"Scaled stroke needs instance-specific audit: {transform}")
        value = max(float(raw), floor)
        if float(raw) < value:
            changes += 1
        if "stroke-width" in element.attrib:
            element.set("stroke-width", f"{value:.9f}")
        else:
            style["stroke-width"] = f"{value:.9f}"
            element.set("style", "; ".join(f"{k}: {v}" for k, v in style.items()))
    if before != without_stroke_width(root):
        raise AssertionError("A non-stroke-width SVG attribute changed")
    tree.write(str(destination), encoding="utf-8", xml_declaration=True)
    # A 2-point white border obscures the small gold selected-contact stars.
    # Removing the border preserves their exact filled paths and coordinates.
    removed_outlines = 0
    if source.stem == STEMS[4]:
        identifiers = {node.get("id"): node for node in root.iter() if node.get("id")}
        for element in root.iter():
            style = properties(element.get("style", ""))
            if style.get("fill", "").lower() in {"#e69f00", "#e69f00ff"} and style.get("stroke", "").lower() in {"#ffffff", "white", "#fff"}:
                style["stroke"] = "none"
                element.set("style", "; ".join(f"{k}: {v}" for k, v in style.items()))
                removed_outlines += 1
                reference = element.get("{http://www.w3.org/1999/xlink}href", "")
                if reference.startswith("#") and reference[1:] in identifiers:
                    definition = identifiers[reference[1:]]
                    definition_style = properties(definition.get("style", ""))
                    definition_style["stroke"] = "none"
                    definition.set("style", "; ".join(f"{k}: {v}" for k, v in definition_style.items()))
        tree.write(str(destination), encoding="utf-8", xml_declaration=True)
    if before_all_strokes != without_stroke_width(root, exclude_stroke_color=True):
        raise AssertionError("Geometry, text, images or a non-stroke style changed")
    metadata_changes = normalize_figure_metadata(root)
    tree.write(str(destination), encoding="utf-8", xml_declaration=True)
    return {"source_svg_sha256": digest(source), "stroke_declarations_changed": changes,
            "gold_marker_white_outlines_removed": removed_outlines,
            "metadata_title_changes": metadata_changes,
            "non_stroke_attributes_and_text_identical_before_documented_label_edits": True}


def cairo_input(path):
    """Expand CSS font shorthand without changing its intended typography."""
    root = etree.parse(str(path)).getroot()
    for element in root.iter():
        style = properties(element.get("style", ""))
        shorthand = style.get("font")
        if not shorthand:
            continue
        match = re.fullmatch(r"(?:(.*?)\s+)?([\d.]+(?:px|pt))\s+(.+)", shorthand)
        if not match:
            raise ValueError(f"Unsupported font shorthand: {shorthand}")
        prefix, size, family = match.groups()
        style["font-size"], style["font-family"] = size, family
        for item in (prefix or "").split():
            if item in {"italic", "oblique", "normal"}:
                style["font-style"] = item
            elif item.isdigit() or item in {"bold", "bolder", "lighter"}:
                style["font-weight"] = item
            else:
                raise ValueError(f"Unsupported font modifier: {item}")
        del style["font"]
        element.set("style", "; ".join(f"{k}: {v}" for k, v in style.items()))
    return etree.tostring(root)


def cairo_exports(svg, stem):
    import cairosvg
    rendered = cairo_input(svg)
    png_data = cairosvg.svg2png(bytestring=rendered, dpi=600, background_color="white")
    with Image.open(io.BytesIO(png_data)) as source:
        image = source.convert("RGB")
        image.save(stem.with_suffix(".png"), dpi=(600, 600))
        image.save(stem.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    cairosvg.svg2pdf(bytestring=rendered, write_to=str(stem.with_suffix(".pdf")))


def plain_math_svg(path):
    """Keep the literal 8-pt floor by replacing small math glyphs with prose."""
    tree = etree.parse(path)
    root = tree.getroot()
    parents = {child: node for node in root.iter() for child in node}
    replacements = []
    for element in root.iter("{http://www.w3.org/2000/svg}text"):
        spans = list(element)
        if not spans:
            continue
        sizes = [float(re.search(r"[\d.]+", properties(e.get("style", "")).get("font-size", "99")).group()) for e in spans]
        if min(sizes) >= 8.5:
            continue
        transform = parents[element].get("transform", "")
        contents = "".join(e.text or "" for e in spans).replace("\xa0", " ")
        if transform.startswith("translate(53.28 62."):
            label = "Joint p = max(projection p, energy p)"
        elif transform.startswith("translate(319.620506 62."):
            label = "Joint q = BH-adjusted joint p"
        elif contents.startswith("10"):
            exponent = contents[2:].replace("−", "-")
            label = "1" if exponent == "0" else "1e" + exponent
        elif contents.startswith("Energy"):
            label = "Energy p"
        elif contents.startswith("Reproducibility"):
            label = "Reproducibility p"
        else:
            raise ValueError(f"Unrecognized small math glyph group: {contents}")
        for child in spans:
            element.remove(child)
        element.text = label
        element.set("x", "0")
        element.set("y", "0")
        element.set("style", "font-size: 8.5px; font-family: Arial, Helvetica, sans-serif; fill: #17242f")
        replacements.append(label)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return replacements


def qc_figure(data_path=None):
    """Use the portable aggregate-only builder; no editorial code execution."""
    spec = importlib.util.spec_from_file_location("qc_aggregate", Path(__file__).with_name("plot_qc_aggregate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.figure(data_path or module.DEFAULT_DATA)


def native_exports(number, builder, benchmark, recovery, stem, qc_data=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.collections import Collection
    if number == 3:
        fig = qc_figure(qc_data)
    else:
        fig = builder.workflow() if number == 1 else builder.synthetic(benchmark, recovery)
    if number == 1:
        from matplotlib.text import Text
        replacements = {
            "$p_R$ in the fixed response window": "Projection p: fixed response window",
            "$p_E$ in the fixed windows": "Energy p: fixed windows",
            r"$p_{\mathrm{joint}} = \max(p_R,\,p_E)$": "Joint p = max(projection p, energy p)",
            r"Adjusted detector call: $q_{\mathrm{joint}} \leq 0.05$": "Adjusted detector call: joint q ≤ 0.05",
        }
        for artist in fig.findobj(Text):
            if artist.get_text() in replacements:
                artist.set_text(replacements[artist.get_text()])
    fig.canvas.draw()
    for artist in fig.findobj():
        if isinstance(artist, Line2D):
            artist.set_linewidth(max(MIN_PT, artist.get_linewidth()))
            artist.set_markeredgewidth(max(MIN_PT, artist.get_markeredgewidth()))
        elif isinstance(artist, Patch):
            if artist.get_linewidth() > 0:
                artist.set_linewidth(max(MIN_PT, artist.get_linewidth()))
        elif isinstance(artist, Collection):
            values = np.asarray(artist.get_linewidths())
            artist.set_linewidths(np.where(values > 0, np.maximum(values, MIN_PT), values))
    report = builder.layout_audit(fig)
    if report["clipped_text"] or report["text_collisions"]:
        raise AssertionError("Native Frontiers export introduced a text collision")
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(stem.with_suffix("." + suffix), dpi=600)
    with Image.open(stem.with_suffix(".png")) as source:
        source.convert("RGB").save(stem.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return {"native_matplotlib": True, "text_audit": report,
            "data_source": "Unchanged frozen aggregate benchmark outputs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    data = Path(__file__).parent / "figure_data"
    parser.add_argument("--benchmark", type=Path, default=data / "synthetic_benchmark")
    parser.add_argument("--recovery", type=Path, default=data / "stratified_recovery.json")
    parser.add_argument("--qc-data", type=Path, default=data / "qc_aggregate_summary.json")
    parser.add_argument("--metrics", type=Path, default=Path(__file__).parent / "frozen_results/erdetect/metrics.csv")
    parser.add_argument("--figures", type=int, nargs="+", choices=range(1, 7), default=[1, 3, 5, 6],
                        help="Figures 2/4 additionally require their frozen base SVGs in --base-dir")
    args = parser.parse_args()
    destination = args.base_dir / "frontiers"
    destination.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("plot_manuscript_validation_figures.py")
    spec = importlib.util.spec_from_file_location("native_figure_builder", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    recovery = json.loads(args.recovery.read_text())
    reports = {}
    for number in args.figures:
        stem = destination / STEMS[number]
        if number == 6:
            spec = importlib.util.spec_from_file_location("external_figure", Path(__file__).with_name("plot_erdetect_validation.py"))
            external = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(external)
            report = {"native_plotter_report": external.plot(args.metrics, stem, frontiers=True)}
        elif number in [1, 3, 5]:
            report = native_exports(number, builder, args.benchmark, recovery, stem, args.qc_data)
        else:
            if not (args.base_dir / (STEMS[number] + ".svg")).is_file():
                raise FileNotFoundError(f"Figure {number} requires its frozen display SVG in --base-dir; raw cohort reconstruction is outside this exporter.")
            report = restyle_svg(args.base_dir / (STEMS[number] + ".svg"), stem.with_suffix(".svg"))
            if number == 4:
                report["same_size_plain_math_labels"] = plain_math_svg(stem.with_suffix(".svg"))
            cairo_exports(stem.with_suffix(".svg"), stem)
        report["physical_stroke_audit"] = stroke_audit(stem.with_suffix(".svg"))
        if report["physical_stroke_audit"]["strokes_below_2pt"]:
            raise AssertionError(f"Figure {number} retains a thin explicit stroke")
        report["files"] = {suffix: {"name": stem.name + "." + suffix,
                                   "sha256": digest(stem.with_suffix("." + suffix))}
                           for suffix in ["svg", "pdf", "png", "tiff"]}
        reports[str(number)] = report
    manifest = {"script_sha256": digest(Path(__file__)), "minimum_stroke_pt": 2,
                "raster_dpi": 600, "figure_reports": reports,
                "visual_qa": "Pending independent inspection of exported PDF renders"}
    (destination / "frontiers_stroke_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v["physical_stroke_audit"] for k, v in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
