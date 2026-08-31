"""Generate the public ERPy API reference from signatures and docstrings."""

from __future__ import annotations

import argparse
from dataclasses import is_dataclass
import inspect
from pathlib import Path
import re
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ERPy
import ERPy.viz as viz


OUTPUT = ROOT / "docs" / "API_REFERENCE.md"


def clean_doc(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    # ``@dataclass`` synthesizes a signature-like ``__doc__`` when a class has
    # no explicit description. Do not mistake that signature for documentation.
    if (
        inspect.isclass(obj)
        and is_dataclass(obj)
        and doc.startswith(f"{obj.__name__}(")
    ):
        return ""
    return doc


def signature(obj, name: str) -> str:
    try:
        rendered = str(inspect.signature(obj))
    except (TypeError, ValueError):
        rendered = ""
    return f"{name}{rendered}"


def annotation_text(annotation) -> str:
    if annotation is inspect.Signature.empty:
        return "not specified"
    if isinstance(annotation, str):
        return annotation.replace("|", "\\|")
    return getattr(annotation, "__name__", repr(annotation)).replace("|", "\\|")


def signature_details(obj) -> list[str]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    parameters = [
        parameter
        for parameter in sig.parameters.values()
        if parameter.name not in {"self", "cls"}
    ]
    lines: list[str] = []
    if parameters:
        lines.extend(
            [
                "",
                "**Parameters**",
                "",
                "| Name | Type | Default |",
                "| --- | --- | --- |",
            ]
        )
        for parameter in parameters:
            if parameter.default is inspect.Signature.empty:
                default = "required"
            else:
                default = f"`{parameter.default!r}`".replace("|", "\\|")
            lines.append(
                f"| `{parameter.name}` | {annotation_text(parameter.annotation)} | {default} |"
            )
    if sig.return_annotation is not inspect.Signature.empty:
        lines.extend(
            ["", f"**Returns:** {annotation_text(sig.return_annotation)}."]
        )
    return lines


def public_members(cls):
    for name, value in inspect.getmembers_static(cls):
        if name.startswith("_"):
            continue
        if isinstance(value, property):
            yield name, value, value.fget
        elif isinstance(value, (staticmethod, classmethod)):
            yield name, value, value.__func__
        elif inspect.isfunction(value) or inspect.ismethoddescriptor(value):
            yield name, value, value


def render_doc(doc: str) -> list[str]:
    if not doc:
        return []
    normalized = re.sub(r":math:`([^`]+)`", r"`\1`", textwrap.dedent(doc))
    return [line.rstrip() for line in normalized.splitlines()]


def render_class(name: str, cls, missing: list[str]) -> list[str]:
    lines = [f"### `{name}`", "", f"```python\n{signature(cls, name)}\n```", ""]
    doc = clean_doc(cls)
    if not doc:
        missing.append(name)
    lines.extend(render_doc(doc) or ["No public description supplied."])
    lines.extend(signature_details(cls))
    members = list(public_members(cls))
    if members:
        lines.extend(["", "#### Public members", ""])
    for member_name, descriptor, target in members:
        qualified = f"{name}.{member_name}"
        if isinstance(descriptor, property):
            display = f"{member_name}  # property"
        else:
            display = signature(target, member_name)
        lines.extend([f"##### `{qualified}`", "", f"```python\n{display}\n```", ""])
        doc = clean_doc(target)
        if not doc:
            missing.append(qualified)
        lines.extend(render_doc(doc) or ["No public description supplied."])
        lines.extend(signature_details(target))
        lines.append("")
    return lines


def render_function(name: str, function, missing: list[str]) -> list[str]:
    lines = [f"### `{name}`", "", f"```python\n{signature(function, name)}\n```", ""]
    doc = clean_doc(function)
    if not doc:
        missing.append(name)
    lines.extend(render_doc(doc) or ["No public description supplied."])
    lines.extend(signature_details(function))
    lines.append("")
    return lines


def render_namespace(title: str, module, names: list[str], missing: list[str]) -> list[str]:
    classes = []
    functions = []
    constants = []
    for name in names:
        obj = getattr(module, name)
        if inspect.isclass(obj):
            classes.append((name, obj))
        elif callable(obj):
            functions.append((name, obj))
        else:
            constants.append((name, obj))

    lines = [f"## {title}", ""]
    if classes:
        lines.extend(["### Classes", ""])
        for name, obj in classes:
            lines.extend(render_class(name, obj, missing))
    if functions:
        lines.extend(["### Functions", ""])
        for name, obj in functions:
            lines.extend(render_function(name, obj, missing))
    if constants:
        lines.extend(["### Constants", ""])
        for name, value in constants:
            value_type = type(value).__name__
            lines.append(f"- `{name}` — Public `{value_type}` package constant.")
        lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    missing: list[str] = []
    lines = [
        "# ERPy 1.0.0 API reference",
        "",
        "This reference is generated from the released public signatures and docstrings.",
        "It documents the stable `ERPy` import surface and every exported visualization.",
        "Scientific definitions and defaults are explained in [METHODS.md](METHODS.md).",
        "",
    ]
    lines.extend(render_namespace("Core API (`ERPy`)", ERPy, list(ERPy.__all__), missing))
    lines.extend(render_namespace("Visualization API (`ERPy.viz`)", viz, list(viz.__all__), missing))
    text = "\n".join(lines).rstrip() + "\n"
    # ``Path.write_text(..., newline=...)`` requires Python 3.10. Path.open
    # keeps deterministic LF output while remaining compatible with Python 3.9.
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    if missing:
        print("Undocumented public callables:")
        for name in sorted(set(missing)):
            print(f"- {name}")
        if not args.allow_missing:
            return 1
    print(f"Wrote {OUTPUT} ({len(text):,} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
