"""Save analysis figures into a patient analysis directory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.axes
import matplotlib.figure


def save_analysis_figure(
    fig: Union[matplotlib.figure.Figure, matplotlib.axes.Axes],
    name: str,
    patient_path: Optional[str] = None,
    subdir: str = "",
    formats: Iterable[str] = ("png", "pdf"),
) -> list[str]:
    """Write a figure to ``{patient_path}/analysis/{subdir}/{name}.{fmt}``."""

    if isinstance(fig, matplotlib.axes.Axes):
        fig = fig.figure
    if patient_path is None:
        return []
    root = Path(patient_path) / "analysis"
    if subdir:
        root = root / subdir
    root.mkdir(parents=True, exist_ok=True)
    base = name.strip().replace(" ", "_")
    outputs: list[str] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = root / f"{base}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(str(path))
    return outputs

