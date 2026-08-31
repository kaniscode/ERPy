"""Simple electrode anatomy plotting helpers with optional Nilearn support."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from matplotlib import colormaps, colors

from .theme import apply_erpy_theme


def _require_nilearn():
    try:
        from nilearn import plotting  # noqa: F401
    except ImportError as exc:
        raise ImportError("nilearn is required for anatomy plots. Install with: pip install nilearn") from exc


def plot_electrode_mni(
    coords: np.ndarray,
    values: Optional[np.ndarray] = None,
    node_size: Union[float, np.ndarray] = 50.0,
    title: Optional[str] = None,
    display_mode: str = "ortho",
    **kwargs,
):
    """Plot electrode locations in MNI space using ``nilearn.plot_connectome``."""

    _require_nilearn()
    from nilearn import plotting

    apply_erpy_theme()
    coords = np.asarray(coords, dtype=float)
    adj = np.zeros((coords.shape[0], coords.shape[0]))
    node_color = kwargs.pop("node_color", None)
    cmap = kwargs.pop("cmap", "viridis")
    if node_color is None and values is not None:
        scalar = np.asarray(values, dtype=float)
        finite = scalar[np.isfinite(scalar)]
        if finite.size:
            lower, upper = float(np.min(finite)), float(np.max(finite))
            if upper <= lower:
                upper = lower + 1.0
            normalizer = colors.Normalize(vmin=lower, vmax=upper)
            node_color = colormaps.get_cmap(cmap)(normalizer(scalar))
        else:
            node_color = "auto"
    elif node_color is None:
        node_color = "auto"
    return plotting.plot_connectome(
        adj,
        coords,
        node_color=node_color,
        node_size=node_size,
        display_mode=display_mode,
        colorbar=False,
        title=title,
        **kwargs,
    )


def plot_electrodes_from_metadata(
    elec_df: pd.DataFrame,
    value_col: Optional[str] = None,
    mni_cols: tuple[str, str, str] = ("mni_x", "mni_y", "mni_z"),
    **kwargs,
):
    """Plot electrodes using coordinate columns from electrode metadata."""

    missing = [c for c in mni_cols if c not in elec_df.columns]
    if missing:
        raise ValueError(f"elec_df is missing MNI coordinate columns: {missing}")
    coords = elec_df[list(mni_cols)].to_numpy(dtype=float)
    values = elec_df[value_col].to_numpy(dtype=float) if value_col else None
    return plot_electrode_mni(coords, values=values, **kwargs)


def plot_erps_on_brain(
    elec_df: pd.DataFrame,
    metric: pd.Series,
    elec_label_col: str = "elec_label",
    mni_cols: tuple[str, str, str] = ("mni_x", "mni_y", "mni_z"),
    **kwargs,
):
    """Color electrodes by a per-contact scalar metric."""

    missing = [c for c in (elec_label_col, *mni_cols) if c not in elec_df.columns]
    if missing:
        raise ValueError(f"elec_df is missing required columns: {missing}")
    labels = elec_df[elec_label_col].astype(str)
    values = labels.map(metric)
    if values.isna().all():
        raise ValueError("No overlap between metric index and electrode metadata labels.")
    coords = elec_df[list(mni_cols)].to_numpy(dtype=float)
    return plot_electrode_mni(coords, values=values.to_numpy(dtype=float), **kwargs)
