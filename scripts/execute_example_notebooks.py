"""Execute and retain reviewed synthetic/public example outputs for GitHub.

Install ``.[viz,validation]`` plus nbclient, nbformat, ipykernel and kaleido.
Kaleido also needs a local Chrome/Chromium installation. Run from any directory.
Large surface/FreeSurfer recipes remain explicitly opt-in in notebook 05.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat

from build_example_notebooks import NOTEBOOKS, OUT, ROOT, assign_deterministic_cell_ids


def source_digest(notebook) -> str:
    sources = [(cell.cell_type, cell.source) for cell in notebook.cells]
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest()


def prepare_for_publication(notebook, filename: str) -> None:
    """Remove machine/timing metadata while retaining results and provenance."""
    replacements = sorted(
        [(str(ROOT), "<repository>"), (sys.prefix, "<python-environment>"),
         (sys.base_prefix, "<python-runtime>"), (str(Path.home()), "<home>")],
        key=lambda pair: len(pair[0]), reverse=True,
    )

    def clean_text(value):
        if isinstance(value, str):
            for old, new in replacements:
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [clean_text(item) for item in value]
        return value

    for cell in notebook.cells:
        cell.metadata = {}
        for output in cell.get("outputs", []):
            if "metadata" in output:
                output.metadata = {}
            if "text" in output:
                output.text = clean_text(output.text)
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/"):
                    output.data[mime] = clean_text(value)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()},
        "erpy_execution": {
            "status": "completed",
            "data_source": (
                "deidentified CNS/ACC–PAG cohort recording" if filename.startswith("00_")
                else "public OpenNeuro ds003708" if filename.startswith("07_")
                else "public OpenNeuro ds004774 derived N1 features" if filename.startswith("08_")
                else "deterministic synthetic"
            ),
            "source_sha256": source_digest(notebook),
            "python_version": platform.python_version(),
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("erpy-neuro", "numpy", "pandas", "scipy", "matplotlib", "nilearn", "plotly", "nbclient")
            },
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", metavar="NOTEBOOK", help="example filenames; omit to run all nine")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.notebooks) - NOTEBOOKS.keys())
    if unknown:
        parser.error("unknown example notebook(s): " + ", ".join(unknown))
    return args


def main() -> None:
    args = parse_args()
    selected = args.notebooks or sorted(NOTEBOOKS)
    with TemporaryDirectory(prefix="erpy-example-kernel-") as kernel_root:
        kernel_dir = Path(kernel_root) / "erpy-examples"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "ERPy example execution", "language": "python",
        }))
        for filename in selected:
            notebook = NOTEBOOKS[filename]
            assign_deterministic_cell_ids(filename, notebook)
            manager = KernelManager(
                kernel_name="erpy-examples",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[kernel_root]),
            )
            print(f"Executing {filename}", flush=True)
            client = NotebookClient(
                notebook, km=manager, timeout=args.timeout, allow_errors=False,
                record_timing=False, resources={"metadata": {"path": str(ROOT)}},
            )
            # A fresh kernel for each notebook; no source/result is reused.
            client.execute(cleanup_kc=True)
            prepare_for_publication(notebook, filename)
            nbformat.validate(notebook)
            nbformat.write(notebook, OUT / filename)
            counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"]
            images = sum("image/png" in output.get("data", {}) for cell in notebook.cells for output in cell.get("outputs", []))
            print(f"Completed {filename}: {len(counts)} code cells, {images} inline PNGs", flush=True)


if __name__ == "__main__":
    main()
