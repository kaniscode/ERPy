from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_graph_metric_timecourse(
    metric_table: pd.DataFrame,
    *,
    node: str,
    metric: str = "hub",
    ax=None,
    color: str = "#2563eb",
    title: str | None = None,
):
    """Plot how one node-level graph metric changes through the stim epoch."""

    ax = ax or plt.subplots(figsize=(6.6, 3.4))[1]
    if metric_table.empty or not {"node", "metric", "value", "time_ms"}.issubset(metric_table.columns):
        ax.text(0.5, 0.5, "No graph metric data", ha="center", va="center")
        ax.axis("off")
        return ax
    data = metric_table[
        (metric_table["node"].astype(str) == str(node))
        & (metric_table["metric"].astype(str) == str(metric))
    ].copy()
    if data.empty:
        ax.text(0.5, 0.5, f"No {metric} data for {node}", ha="center", va="center")
        ax.axis("off")
        return ax
    data = data.sort_values("time_ms")
    ax.plot(data["time_ms"], pd.to_numeric(data["value"], errors="coerce"), color=color, lw=2.0)
    ax.axvline(0, color="0.2", ls="--", lw=0.9)
    ax.axhline(0, color="0.82", lw=0.8)
    ax.set_xlabel("Time after stimulation (ms)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"{node} {metric.replace('_', ' ')} over time")
    return ax


def plot_graph_metric_heatmap(
    metric_table: pd.DataFrame,
    *,
    metric: str = "hub",
    top_n_nodes: int = 24,
    ax=None,
    title: str | None = None,
):
    """Plot node graph metrics as a node x time heatmap."""

    ax = ax or plt.subplots(figsize=(8.4, 5.0))[1]
    if metric_table.empty or not {"node", "metric", "value", "time_ms"}.issubset(metric_table.columns):
        ax.text(0.5, 0.5, "No graph metric data", ha="center", va="center")
        ax.axis("off")
        return ax
    data = metric_table[metric_table["metric"].astype(str) == str(metric)].copy()
    if data.empty:
        ax.text(0.5, 0.5, f"No {metric} data", ha="center", va="center")
        ax.axis("off")
        return ax
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    node_order = (
        data.groupby("node")["value"]
        .max()
        .sort_values(ascending=False)
        .head(top_n_nodes)
        .index.astype(str)
        .tolist()
    )
    matrix = data[data["node"].astype(str).isin(node_order)].pivot_table(
        index="node",
        columns="time_ms",
        values="value",
        aggfunc="max",
    )
    matrix = matrix.reindex(index=node_order)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    step = max(1, len(matrix.columns) // 8)
    ticks = np.arange(0, len(matrix.columns), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{float(matrix.columns[i]):.0f}" for i in ticks], rotation=0, fontsize=8)
    ax.set_xlabel("Time after stimulation (ms)")
    ax.set_ylabel("Node")
    ax.set_title(title or f"{metric.replace('_', ' ').title()} over time")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label(metric.replace("_", " "))
    return ax
