from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_metric_within_stim(
    data: pd.DataFrame,
    metric: str,
    *,
    stim_col: str = "stim_pair",
    channel_col: str | None = None,
    absolute: bool = False,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Z-score a response metric across recording contacts within each stim site.

    This is an exploratory ranking transform: for each stimulation site, ERPy
    normalizes the selected response metric across recording electrodes so that
    unusually high contacts for that stimulation site are easy to inspect.
    """

    if data.empty:
        return data.copy()
    if stim_col not in data.columns:
        raise ValueError(f"Input table must contain a '{stim_col}' column")
    if metric not in data.columns:
        raise ValueError(f"Input table must contain metric column {metric!r}")
    if channel_col is None:
        if "channel" in data.columns:
            channel_col = "channel"
        elif "record_elec" in data.columns:
            channel_col = "record_elec"
        else:
            raise ValueError("Provide channel_col or include 'channel'/'record_elec' in the table")
    if channel_col not in data.columns:
        raise ValueError(f"Input table must contain a '{channel_col}' column")

    out = data.copy()
    values = pd.to_numeric(out[metric], errors="coerce")
    if absolute:
        values = values.abs()
    value_col = f"_{metric}_for_z"
    out[value_col] = values

    def group_z(series: pd.Series) -> pd.Series:
        mean = series.mean(skipna=True)
        std = series.std(skipna=True, ddof=1)
        if not np.isfinite(std) or std == 0:
            return pd.Series(np.nan, index=series.index)
        return (series - mean) / std

    z_col = output_col or f"{metric}_within_stim_z"
    out[z_col] = out.groupby(stim_col, group_keys=False)[value_col].transform(group_z)
    out[f"{z_col}_rank"] = out.groupby(stim_col)[z_col].rank(ascending=False, method="dense")
    out = out.drop(columns=[value_col])
    return out.sort_values([stim_col, f"{z_col}_rank", channel_col])
