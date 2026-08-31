from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd

from ..baseline import baseline_center_frame


STIM_NODE_COLOR = "#111827"
STIM_NODE_ACCENT = "#fbbf24"
RESPONSE_NODE_COLOR = "#0f766e"
RESPONSE_NODE_COLOR_ALT = "#14b8a6"
STIM_NODE_STATIC_SCALE = 0.86
STIM_NODE_PLOTLY_SIZE = 10
STIM_NODE_AGGREGATE_PLOTLY_SIZE = 11
STIM_NODE_GLASS_SCALE = 0.92


def edges_from_detections(
    detections: pd.DataFrame,
    stim_pair: str | None = None,
    elec_meta: pd.DataFrame | None = None,
    metric: str = "peak_amplitude_uv",
    channel_col: str = "channel",
    consensus_only: bool = True,
    significance_column: str | None = None,
    qc_only: bool = True,
    exclude_stimulation_contacts: bool = True,
    identity_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert detection rows into a tidy directed edge table.

    The result is intentionally plain pandas so users can pass it to NetworkX,
    Nilearn, Plotly, BCT-style graph packages, or their own statistics code.
    """

    data = detections.copy()
    if consensus_only:
        selected_significance = significance_column or "consensus_ch"
        if significance_column is not None and selected_significance not in data.columns:
            raise ValueError(
                f"detections must contain significance column {selected_significance!r}"
            )
        if selected_significance in data.columns:
            data = data[_boolean_mask(data[selected_significance])]
    if qc_only:
        for qc_col in ("qc_pass", "artifact_qc_pass"):
            if qc_col in data.columns:
                data = data[_boolean_mask(data[qc_col])]
    if stim_pair is not None and "stim_pair" not in data.columns:
        data["stim_pair"] = stim_pair
    if "stim_pair" not in data.columns:
        raise ValueError("Provide stim_pair or include a 'stim_pair' column in detections")
    if metric not in data.columns:
        raise ValueError(f"detections must contain metric column {metric!r}")
    if channel_col not in data.columns:
        raise ValueError(f"detections must contain channel column {channel_col!r}")

    identity_candidates = list(
        identity_cols
        or ("patient_id", "session_id", "acquisition_id", "raw_file")
    )
    detection_identity = [col for col in identity_candidates if col in data.columns]
    dedup_cols = [*detection_identity, "stim_pair", channel_col]
    if "method" in data.columns:
        data = data.sort_values("method", kind="stable")
    data = data.drop_duplicates(dedup_cols, keep="first")
    if data.empty:
        return pd.DataFrame()

    metadata = pd.DataFrame(elec_meta).copy() if elec_meta is not None else pd.DataFrame()
    metadata_identity = [
        col
        for col in ("patient_id", "session_id")
        if col in data.columns and col in metadata.columns
    ]
    passthrough_cols = [
        col
        for col in (
            "patient_id",
            "session_id",
            "acquisition_id",
            "raw_file",
            "stim_start",
            "stim_stop",
            "stim_anat",
        )
        if col in data.columns
    ]
    rows = []
    grouped = (
        data.groupby(metadata_identity, dropna=False, sort=False)
        if metadata_identity
        else [((), data)]
    )
    for group_values, frame in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        sub_meta = metadata
        for col, value in zip(metadata_identity, group_values):
            sub_meta = sub_meta[
                sub_meta[col].fillna("").astype(str).eq(
                    "" if pd.isna(value) else str(value)
                )
            ]
        region_map, coord_map, available_labels = _electrode_metadata_maps(
            sub_meta,
            context=dict(zip(metadata_identity, group_values)),
        )

        for _, row in frame.iterrows():
            sp = str(row["stim_pair"])
            stim_elec = resolve_electrode_label(
                stim_pair_to_source_electrode(sp),
                available_labels,
            )
            record_elec = resolve_electrode_label(
                str(row[channel_col]),
                available_labels,
            )
            stimulation_contacts = {
                electrode_label_identity(
                    resolve_electrode_label(contact, available_labels)
                )
                for contact in stim_pair_to_contact_electrodes(sp)
            }
            if (
                exclude_stimulation_contacts
                and electrode_label_identity(record_elec) in stimulation_contacts
            ):
                continue
            metric_value = pd.to_numeric(
                pd.Series([row[metric]]),
                errors="coerce",
            ).iloc[0]
            out = {
                **{col: row[col] for col in passthrough_cols},
                "stim_pair": sp,
                "stim_elec": stim_elec,
                "record_elec": record_elec,
                "source": stim_elec,
                "target": record_elec,
                "weight": float(metric_value) if pd.notna(metric_value) else 0.0,
                metric: metric_value,
                "stim_region": region_map.get(stim_elec, ""),
                "record_region": region_map.get(record_elec, ""),
            }
            for prefix, elec in (("stim", stim_elec), ("record", record_elec)):
                coords = coord_map.get(elec, {})
                for axis, col in zip(
                    ("x", "y", "z"),
                    ("mni_x", "mni_y", "mni_z"),
                ):
                    out[f"{prefix}_mni_{axis}"] = coords.get(col, np.nan)
            rows.append(out)
    return pd.DataFrame(rows)


def _boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def _electrode_metadata_maps(
    elec_meta: pd.DataFrame,
    *,
    context: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]], list[str]]:
    if elec_meta.empty or "elec_label" not in elec_meta.columns:
        return {}, {}, []

    metadata = elec_meta.copy()
    metadata["elec_label"] = metadata["elec_label"].astype(str)
    value_cols = [
        col
        for col in ("anat_label", "mni_x", "mni_y", "mni_z")
        if col in metadata.columns
    ]
    conflicts = []
    for label, frame in metadata.groupby("elec_label", sort=False):
        for col in value_cols:
            values = frame[col].dropna()
            if col == "anat_label":
                values = values.astype(str).str.strip()
                values = values[values.ne("")]
            if values.nunique(dropna=True) > 1:
                conflicts.append(f"{label}:{col}")
    if conflicts:
        location = (
            ", ".join(f"{key}={value!r}" for key, value in (context or {}).items())
            or "unscoped metadata"
        )
        raise ValueError(
            "Conflicting electrode metadata within "
            f"{location}: {', '.join(conflicts[:8])}"
        )

    metadata = metadata.drop_duplicates("elec_label", keep="first")
    available = metadata["elec_label"].tolist()
    region_map = (
        metadata.set_index("elec_label")["anat_label"].fillna("").to_dict()
        if "anat_label" in metadata.columns
        else {}
    )
    coord_cols = [
        col for col in ("mni_x", "mni_y", "mni_z") if col in metadata.columns
    ]
    coord_map = (
        metadata.set_index("elec_label")[coord_cols].to_dict("index")
        if len(coord_cols) == 3
        else {}
    )
    return region_map, coord_map, available


def stim_pair_to_contact_electrodes(stim_pair: str) -> tuple[str, ...]:
    """Return both contact labels represented by a stimulation-pair label."""

    parts = str(stim_pair).split("_")
    if len(parts) >= 3:
        lead = "_".join(parts[:-2])
        contacts = []
        for value in parts[-2:]:
            try:
                number = float(value)
                suffix = str(int(number)) if number.is_integer() else str(value)
            except ValueError:
                suffix = str(value)
            contacts.append(f"{lead}{suffix}")
        return tuple(contacts)
    return (str(stim_pair),)


def stim_pair_to_source_electrode(stim_pair: str, contact: str = "positive") -> str:
    contacts = stim_pair_to_contact_electrodes(stim_pair)
    index = 0 if contact == "positive" else min(1, len(contacts) - 1)
    return contacts[index]


def electrode_label_identity(label: str) -> str:
    """Return a reference- and zero-padding-invariant contact identity."""

    text = re.sub(
        r"(?:[-_ ]?REF(?:ERENCE)?)$",
        "",
        str(label).strip(),
        flags=re.IGNORECASE,
    )
    compact = re.sub(r"[^A-Z0-9]", "", text.upper())
    return "".join(
        str(int(part)) if part.isdigit() else part
        for part in re_split_label(compact)
    )


def _label_key(label: str) -> str:
    return electrode_label_identity(label)


def re_split_label(label: str) -> list[str]:
    return [part for part in re.split(r"(\d+)", str(label)) if part]


def resolve_electrode_label(label: str, available_labels: list[str] | None = None) -> str:
    """Match compact ERPy labels such as `PT1` to coordinate labels such as `PT01`."""

    if not available_labels:
        return str(label)
    label = str(label)
    if label in available_labels:
        return label
    target = _label_key(label)
    for candidate in available_labels:
        if _label_key(candidate) == target:
            return str(candidate)
    return label


def adjacency_from_edges(
    edges: pd.DataFrame,
    source_col: str = "stim_elec",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    agg: str = "mean",
) -> pd.DataFrame:
    """Pivot a directed edge table into a square electrode adjacency matrix."""

    if edges.empty:
        return pd.DataFrame()
    table = edges.pivot_table(index=source_col, columns=target_col, values=weight_col, aggfunc=agg, fill_value=0.0)
    nodes = sorted(set(table.index).union(table.columns))
    return table.reindex(index=nodes, columns=nodes, fill_value=0.0)


def natural_label_key(label: str) -> tuple:
    parts = re_split_label(label)
    out = []
    for part in parts:
        out.append((0, int(part)) if part.isdigit() else (1, part.upper()))
    return tuple(out)


def order_nodes_by_metadata(
    nodes: list[str],
    elec_meta: pd.DataFrame | None = None,
    elec_col: str = "elec_label",
    group_cols=("hemisphere", "group"),
    coord_cols=("mni_x", "mni_y", "mni_z"),
) -> list[str]:
    """Order electrodes by shaft metadata, natural contact label, then coordinates."""

    nodes = [str(n) for n in nodes]
    if elec_meta is None or elec_meta.empty or elec_col not in elec_meta.columns:
        return sorted(nodes, key=natural_label_key)
    meta = elec_meta.copy()
    meta[elec_col] = meta[elec_col].astype(str)
    available = meta[elec_col].tolist()
    label_map = {node: resolve_electrode_label(node, available) for node in nodes}
    meta = meta.drop_duplicates(elec_col).set_index(elec_col)

    def key(node: str):
        resolved = label_map[node]
        if resolved not in meta.index:
            empty_groups = tuple("" for _ in group_cols)
            empty_coords = tuple(np.inf for _ in coord_cols)
            return (*empty_groups, *natural_label_key(node), *empty_coords)
        row = meta.loc[resolved]
        groups = tuple(str(row.get(col, "")) for col in group_cols)
        coords = []
        for col in coord_cols:
            coords.append(pd.to_numeric(row.get(col, np.nan), errors="coerce"))
        coords = tuple(float(v) if pd.notna(v) else np.inf for v in coords)
        return (*groups, *natural_label_key(resolved), *coords)

    return sorted(nodes, key=key)


def ordered_adjacency_from_edges(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame | None = None,
    source_col: str = "stim_elec",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    agg: str = "mean",
) -> pd.DataFrame:
    """Build an adjacency matrix ordered by electrode metadata and label."""

    matrix = adjacency_from_edges(edges, source_col=source_col, target_col=target_col, weight_col=weight_col, agg=agg)
    if matrix.empty:
        return matrix
    ordered = order_nodes_by_metadata(list(matrix.index), elec_meta=elec_meta)
    return matrix.reindex(index=ordered, columns=ordered, fill_value=0.0)


def order_sources_by_metadata(
    sources: list[str],
    elec_meta: pd.DataFrame | None = None,
) -> list[str]:
    """Order stimulation sources by their first contact, then by pair label."""

    sources = [str(source) for source in sources]
    if not sources:
        return []
    contact_by_source = {
        source: stim_pair_to_source_electrode(source) if "_" in source else source
        for source in sources
    }
    contacts = list(dict.fromkeys(contact_by_source.values()))
    ordered_contacts = order_nodes_by_metadata(contacts, elec_meta=elec_meta)
    rank = {contact: idx for idx, contact in enumerate(ordered_contacts)}
    return sorted(
        sources,
        key=lambda source: (
            rank.get(contact_by_source[source], np.inf),
            *natural_label_key(source),
        ),
    )


def response_matrix_from_edges(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame | None = None,
    source_col: str = "stim_pair",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    agg: str = "mean",
) -> pd.DataFrame:
    """Return stimulation-source by recording-channel response weights.

    Unlike a graph adjacency matrix, this table keeps stimulation sources and
    recording contacts on separate axes. That is the view most users want for
    CCEP/SPES response fields, especially when comparing multiple stimulation
    sites in the same subject or public dataset.
    """

    if edges.empty:
        return pd.DataFrame()
    table = edges.pivot_table(
        index=source_col,
        columns=target_col,
        values=weight_col,
        aggfunc=agg,
        fill_value=0.0,
    )
    rows = order_sources_by_metadata(list(table.index), elec_meta=elec_meta)
    columns = order_nodes_by_metadata(list(table.columns), elec_meta=elec_meta)
    return table.reindex(index=rows, columns=columns, fill_value=0.0)


def graph_from_matrix(matrix: pd.DataFrame, threshold: float = 0.0, directed: bool = True) -> nx.Graph:
    """Convert finite supra-threshold matrix entries to weighted graph edges."""

    graph = nx.DiGraph() if directed else nx.Graph()
    for src in matrix.index:
        for dst in matrix.columns:
            weight = float(matrix.loc[src, dst])
            if np.isfinite(weight) and abs(weight) > threshold:
                graph.add_edge(src, dst, weight=weight)
    return graph


def graph_from_edges(
    edges: pd.DataFrame,
    source_col: str = "stim_elec",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    directed: bool = True,
    threshold: float = 0.0,
) -> nx.Graph:
    """Convert a response-edge table to a weighted directed or undirected graph."""

    graph = nx.DiGraph() if directed else nx.Graph()
    for _, row in edges.iterrows():
        weight = float(row.get(weight_col, 1.0))
        if not np.isfinite(weight) or abs(weight) <= threshold:
            continue
        attrs = row.to_dict()
        attrs["weight"] = weight
        graph.add_edge(row[source_col], row[target_col], **attrs)
    return graph


def summarize_graph(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "node": node,
                "degree": graph.degree(node),
                "in_degree": graph.in_degree(node) if graph.is_directed() else graph.degree(node),
                "out_degree": graph.out_degree(node) if graph.is_directed() else graph.degree(node),
            }
        )
    return pd.DataFrame(rows)


def export_graph_summary(graph: nx.Graph, path: str | Path) -> Path:
    path = Path(path)
    summarize_graph(graph).to_csv(path, index=False)
    return path


def _network_layout(
    graph: nx.Graph,
    layout: str = "kamada_kawai",
    seed: int = 8,
    weight_attr: str = "weight",
) -> dict:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        node = next(iter(graph.nodes))
        return {node: np.array([0.0, 0.0])}
    if layout == "kamada_kawai":
        layout_graph = graph.copy()
        for _, _, data in layout_graph.edges(data=True):
            weight = abs(float(data.get(weight_attr, 1.0)))
            data["_layout_distance"] = 1.0 / (weight + 1e-9)
        try:
            return nx.kamada_kawai_layout(layout_graph, weight="_layout_distance")
        except Exception:
            return nx.spring_layout(graph, seed=seed, weight=weight_attr)
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed, weight=weight_attr)
    if layout == "spectral":
        return nx.spectral_layout(graph, weight=weight_attr)
    if layout == "shell":
        return nx.shell_layout(graph)
    if layout == "circular":
        return nx.circular_layout(graph)
    raise ValueError("layout must be one of 'kamada_kawai', 'spring', 'spectral', 'shell', or 'circular'")


def plot_network(
    graph: nx.Graph,
    ax=None,
    weight_attr: str = "weight",
    layout: str = "kamada_kawai",
    seed: int = 8,
    title: str | None = None,
):
    """Plot a weighted response graph with strength-scaled nodes and edges."""

    ax = ax or plt.subplots(figsize=(6, 5))[1]
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No response edges", ha="center", va="center")
        ax.axis("off")
        return ax
    pos = _network_layout(graph, layout=layout, seed=seed, weight_attr=weight_attr)
    weights = np.array([abs(d.get(weight_attr, 1.0)) for _, _, d in graph.edges(data=True)])
    graph_vmax = _robust_abs_vmax(weights)
    widths = 0.85 + 3.00 * np.clip(weights / graph_vmax, 0.0, 1.0)
    def node_strength(node: str) -> float:
        incident = list(graph.edges(node, data=True))
        if graph.is_directed():
            incident.extend(list(graph.in_edges(node, data=True)))
        return sum(abs(float(data.get(weight_attr, 1.0))) for *_, data in incident)

    strength = {node: node_strength(node) for node in graph.nodes}
    max_strength = max(strength.values()) if strength else 1.0
    node_sizes = {node: 300 + 560 * strength.get(node, 0.0) / (max_strength if max_strength else 1.0) for node in graph.nodes}
    source_nodes = [node for node in graph.nodes if graph.is_directed() and graph.out_degree(node) > 0]
    response_nodes = [node for node in graph.nodes if node not in source_nodes]
    if response_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=response_nodes,
            ax=ax,
            node_color=RESPONSE_NODE_COLOR,
            edgecolors="white",
            linewidths=1.2,
            node_size=[node_sizes[node] for node in response_nodes],
        )
    if source_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=source_nodes,
            ax=ax,
            node_color=STIM_NODE_COLOR,
            edgecolors=STIM_NODE_ACCENT,
            linewidths=1.8,
            node_size=[node_sizes[node] * STIM_NODE_STATIC_SCALE for node in source_nodes],
            node_shape="D",
        )
    edge_collection = nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=widths,
        alpha=0.72,
        arrows=graph.is_directed(),
        arrowstyle="-|>",
        arrowsize=9,
        edge_color=weights if len(weights) else "#64748b",
        edge_cmap=plt.get_cmap("plasma"),
        edge_vmin=float(weights.min()) if len(weights) else None,
        edge_vmax=graph_vmax if len(weights) else None,
    )
    if edge_collection is not None and not isinstance(edge_collection, list):
        edge_collection.set_zorder(1)
    label_nodes = set(graph.nodes) if graph.number_of_nodes() <= 24 else set(
        sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=True)[:24]
    )
    nx.draw_networkx_labels(
        graph.subgraph(label_nodes),
        {node: pos[node] for node in label_nodes},
        ax=ax,
        font_size=8,
        font_color="#111827",
        bbox={"boxstyle": "round,pad=0.13", "fc": "white", "ec": "none", "alpha": 0.72},
    )
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax


def plot_adjacency_heatmap(matrix: pd.DataFrame, ax=None, cmap: str = "magma", title: str = "Connectivity matrix"):
    """Plot a labeled stimulation-by-recording adjacency heat map."""

    ax = ax or plt.subplots(figsize=(6, 5))[1]
    data = matrix.to_numpy(dtype=float) if not matrix.empty else np.zeros((1, 1))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Recording channel")
    ax.set_ylabel("Stimulation source")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    return ax


def plot_ordered_adjacency_heatmap(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame | None = None,
    ax=None,
    cmap: str = "magma",
    title: str = "Ordered response adjacency",
    **kwargs,
):
    """Build and plot an anatomically ordered electrode adjacency matrix."""

    matrix = ordered_adjacency_from_edges(edges, elec_meta=elec_meta, **kwargs)
    return plot_adjacency_heatmap(matrix, ax=ax, cmap=cmap, title=title)


def plot_response_matrix_heatmap(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame | None = None,
    ax=None,
    cmap: str = "magma",
    title: str = "Stimulation response matrix",
    **kwargs,
):
    """Build and plot a stimulation-source by recording-channel response matrix."""

    matrix = response_matrix_from_edges(edges, elec_meta=elec_meta, **kwargs)
    return plot_adjacency_heatmap(matrix, ax=ax, cmap=cmap, title=title)


def region_centroid_coords(
    elec_meta: pd.DataFrame,
    region_col: str = "anat_label",
    mni_cols=("mni_x", "mni_y", "mni_z"),
) -> pd.DataFrame:
    cols = [region_col, *mni_cols]
    data = elec_meta[cols].dropna(subset=[region_col]).copy()
    for col in mni_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.groupby(region_col)[list(mni_cols)].mean().dropna().reset_index()


def flag_same_region_edges(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame,
    stim_col: str = "stim_elec",
    record_col: str = "record_elec",
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
) -> pd.Series:
    mapping = elec_meta.set_index(elec_col)[region_col].to_dict()
    return edges.apply(lambda r: mapping.get(r[stim_col]) == mapping.get(r[record_col]), axis=1)


def electrode_region_mapping_table(
    elec_meta: pd.DataFrame,
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    mni_cols=("mni_x", "mni_y", "mni_z"),
) -> pd.DataFrame:
    cols = [c for c in [elec_col, region_col, *mni_cols, "patient_id", "session_id"] if c in elec_meta.columns]
    return elec_meta[cols].drop_duplicates().sort_values(cols[:1]).reset_index(drop=True)


def validate_mni_coordinates(
    elec_meta: pd.DataFrame,
    elec_col: str = "elec_label",
    mni_cols=("mni_x", "mni_y", "mni_z"),
    margin_mm: float = 4.0,
) -> pd.DataFrame:
    """Flag whether electrode coordinates fall inside the MNI152 brain mask."""

    cols = [c for c in [elec_col, *mni_cols] if c in elec_meta.columns]
    missing = [c for c in (elec_col, *mni_cols) if c not in elec_meta.columns]
    if missing:
        raise ValueError(f"electrode metadata is missing required coordinate columns: {missing}")
    out = elec_meta[cols].drop_duplicates(elec_col).copy()
    for col in mni_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    valid_rows = out.dropna(subset=list(mni_cols)).copy()
    inside = _inside_mni152_brain_mask(valid_rows[list(mni_cols)].to_numpy(dtype=float), margin_mm=margin_mm)
    out["inside_mni152_brain"] = False
    out.loc[valid_rows.index, "inside_mni152_brain"] = inside
    return out.reset_index(drop=True)


def node_coordinates_from_metadata(
    elec_meta: pd.DataFrame,
    nodes: list[str] | None = None,
    elec_col: str = "elec_label",
    coord_cols=("mni_x", "mni_y", "mni_z"),
    region_col: str = "anat_label",
) -> pd.DataFrame:
    """Return electrode coordinates in a predictable plotting schema."""

    if elec_meta is None or elec_meta.empty:
        return pd.DataFrame(columns=["node", "x", "y", "z", "region"])
    missing = [c for c in (elec_col, *coord_cols) if c not in elec_meta.columns]
    if missing:
        return pd.DataFrame(columns=["node", "x", "y", "z", "region"])
    data = elec_meta.drop_duplicates(elec_col).copy()
    if nodes is not None:
        data = data[data[elec_col].isin(nodes)]
    for col in coord_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=list(coord_cols))
    out = pd.DataFrame(
        {
            "node": data[elec_col].astype(str),
            "x": data[coord_cols[0]].astype(float),
            "y": data[coord_cols[1]].astype(float),
            "z": data[coord_cols[2]].astype(float),
            "region": data[region_col].astype(str) if region_col in data.columns else "",
        }
    )
    return out.reset_index(drop=True)


@lru_cache(maxsize=8)
def _mni152_brain_mask_data(resolution: int = 2, threshold: float = 0.2, margin_mm: float = 4.0):
    try:
        from nilearn import datasets
        from scipy import ndimage
    except Exception as exc:
        raise ImportError("MNI brain-mask validation requires the optional 'nilearn' and 'scipy' dependencies") from exc

    img = datasets.load_mni152_brain_mask(resolution=resolution, threshold=threshold)
    mask = np.asarray(img.get_fdata() > 0, dtype=bool)
    if margin_mm > 0:
        voxel_sizes = np.sqrt((np.asarray(img.affine[:3, :3], dtype=float) ** 2).sum(axis=0))
        iterations = int(np.ceil(float(margin_mm) / float(np.nanmin(voxel_sizes))))
        if iterations > 0:
            mask = ndimage.binary_dilation(mask, iterations=iterations)
    return mask, np.linalg.inv(np.asarray(img.affine, dtype=float))


def _inside_mni152_brain_mask(
    xyz: np.ndarray,
    resolution: int = 2,
    threshold: float = 0.2,
    margin_mm: float = 4.0,
) -> np.ndarray:
    mask, inv_affine = _mni152_brain_mask_data(resolution=resolution, threshold=threshold, margin_mm=margin_mm)
    xyz = np.asarray(xyz, dtype=float)
    vox = np.c_[xyz, np.ones(len(xyz))] @ inv_affine.T
    ijk = np.round(vox[:, :3]).astype(int)
    bounds = np.array(mask.shape)
    in_bounds = np.all((ijk >= 0) & (ijk < bounds), axis=1)
    inside = np.zeros(len(xyz), dtype=bool)
    valid = ijk[in_bounds]
    if len(valid):
        inside[in_bounds] = mask[valid[:, 0], valid[:, 1], valid[:, 2]]
    return inside


def _filter_mni_brain_coordinates(
    coords: pd.DataFrame,
    margin_mm: float = 4.0,
    outside: str = "drop",
    context: str = "MNI brain plot",
) -> pd.DataFrame:
    """Keep exact MNI coordinates, but remove/report points outside the template brain."""

    if coords.empty or outside == "ignore":
        return coords
    if outside not in {"drop", "warn", "raise", "ignore"}:
        raise ValueError("outside_mni must be one of 'drop', 'warn', 'raise', or 'ignore'")
    try:
        inside = _inside_mni152_brain_mask(coords[["x", "y", "z"]].to_numpy(dtype=float), margin_mm=margin_mm)
    except ImportError:
        warnings.warn(
            f"{context}: could not validate MNI coordinates because Nilearn/SciPy is unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return coords
    outside_nodes = coords.loc[~inside, "node"].astype(str).tolist()
    if not outside_nodes:
        return coords
    message = (
        f"{context}: excluded {len(outside_nodes)} electrode coordinate(s) outside the MNI152 brain mask: "
        + ", ".join(outside_nodes[:12])
        + ("..." if len(outside_nodes) > 12 else "")
    )
    if outside == "raise":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    if outside == "drop":
        return coords.loc[inside].reset_index(drop=True)
    return coords


def _color_from_cmap(value: float, vmin: float, vmax: float, cmap: str = "viridis") -> str:
    cmap_obj = plt.get_cmap(cmap)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        scaled = 0.75
    else:
        scaled = (value - vmin) / (vmax - vmin)
    return mcolors.to_hex(cmap_obj(float(np.clip(scaled, 0, 1))))


def _plotly_signed_amplitude_colorscale() -> list[list[float | str]]:
    return [
        [0.0, "#053061"],
        [0.25, "#4393c3"],
        [0.5, "#f7f7f7"],
        [0.75, "#d6604d"],
        [1.0, "#67001f"],
    ]


def _plotly_significance_count_colorscale() -> list[list[float | str]]:
    return [
        [0.0, "#e0f2fe"],
        [0.35, "#38bdf8"],
        [0.70, "#facc15"],
        [1.0, "#dc2626"],
    ]


def _robust_abs_vmax(values, percentile: float = 90.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > 0]
    if arr.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(arr, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(arr))
    return vmax if vmax > 0 else 1.0


def _positive_scale(value: float, vmax: float) -> float:
    if not np.isfinite(vmax) or vmax <= 0:
        return 0.0
    return float(np.clip(abs(value) / vmax, 0.0, 1.0))


def _template_brain_surface(n_theta: int = 72, n_phi: int = 36):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0.04 * np.pi, 0.96 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    surfaces = []
    for sign in (-1, 1):
        cx = sign * 31.0
        rx, ry, rz = 39.0, 58.0, 44.0
        taper = 1.0 - 0.08 * np.cos(theta_grid) ** 2 + 0.04 * np.sin(2 * phi_grid)
        x = cx + sign * rx * np.sin(phi_grid) * np.cos(theta_grid) * taper
        y = -20.0 + ry * np.sin(phi_grid) * np.sin(theta_grid) * (0.94 + 0.06 * np.cos(phi_grid))
        z = 8.0 + rz * np.cos(phi_grid) * (0.92 + 0.08 * np.sin(theta_grid) ** 2)
        surfaces.append((x, y, z))
    return surfaces


def _surface_attr(surface: str, hemi: str) -> str:
    surface = {"inflated": "infl"}.get(surface, surface)
    return f"{surface}_{'left' if hemi == 'lh' else 'right'}"


@lru_cache(maxsize=16)
def _load_fsaverage_surfaces(mesh: str = "fsaverage5", surface: str = "pial"):
    """Load template cortical meshes from Nilearn's packaged fsaverage surfaces."""

    try:
        from nilearn import datasets, surface as nilearn_surface
    except Exception as exc:
        raise ImportError("fsaverage cortical rendering requires the optional 'nilearn' dependency") from exc
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    out = []
    for hemi in ("lh", "rh"):
        attr = _surface_attr(surface, hemi)
        if attr not in fsaverage:
            raise ValueError(f"Unknown fsaverage surface {surface!r}; expected one of pial, white, infl, or inflated")
        surf_mesh = nilearn_surface.load_surf_mesh(fsaverage[attr])
        out.append((np.asarray(surf_mesh.coordinates, dtype=float), np.asarray(surf_mesh.faces, dtype=int), hemi))
    return tuple(out)


def _plotly_fsaverage_surface_traces(
    go,
    mesh: str = "fsaverage5",
    surface: str = "pial",
    opacity: float = 0.18,
    stride: int = 1,
):
    traces = []
    for coords, faces, hemi in _load_fsaverage_surfaces(mesh=mesh, surface=surface):
        faces = faces[:: max(1, int(stride))]
        traces.append(
            go.Mesh3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#d7d1c7",
                opacity=opacity,
                flatshading=False,
                lighting={"ambient": 0.72, "diffuse": 0.82, "specular": 0.18, "roughness": 0.58},
                lightposition={"x": 0, "y": -90, "z": 120},
                hoverinfo="skip",
                name=f"fsaverage {hemi}",
                showlegend=False,
            )
        )
    return traces


def _plotly_template_brain_surface_traces(
    go,
    opacity: float = 0.18,
    mesh: str = "fsaverage5",
    surface: str = "pial",
    fallback_shell: bool = True,
):
    try:
        return _plotly_fsaverage_surface_traces(go, mesh=mesh, surface=surface, opacity=opacity, stride=2)
    except Exception:
        if not fallback_shell:
            raise
    traces = []
    for x, y, z in _template_brain_surface(n_theta=48, n_phi=26):
        traces.append(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=np.zeros_like(x),
                colorscale=[[0.0, "#d8d2c7"], [1.0, "#d8d2c7"]],
                showscale=False,
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
                lighting={"ambient": 0.72, "diffuse": 0.78, "specular": 0.18, "roughness": 0.66},
                contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
            )
        )
    return traces


def _edge_subset_with_coords(
    edges: pd.DataFrame,
    coords: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str,
    top_n: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty or coords.empty:
        return edges.iloc[0:0].copy(), coords.iloc[0:0].copy()
    coord_nodes = set(coords["node"])
    data = edges[
        edges[source_col].astype(str).isin(coord_nodes)
        & edges[target_col].astype(str).isin(coord_nodes)
    ].copy()
    if data.empty:
        return data, coords.iloc[0:0].copy()
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce").fillna(0.0)
    data["_abs_weight"] = data[weight_col].abs()
    data = data.sort_values("_abs_weight", ascending=False)
    if top_n is not None:
        data = data.head(top_n)
    nodes = sorted(set(data[source_col].astype(str)).union(data[target_col].astype(str)))
    coords = coords[coords["node"].isin(nodes)].copy()
    return data, coords


def plot_coordinate_network(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame,
    source_col: str = "stim_elec",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    coord_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    projection: tuple[str, str] = ("mni_x", "mni_y"),
    top_n: int | None = 60,
    ax=None,
    title: str = "Coordinate response network",
    cmap: str = "plasma",
):
    """Plot a data-aware 2D network using electrode coordinates instead of a spring layout."""

    ax = ax or plt.subplots(figsize=(7.2, 6.0))[1]
    if edges.empty:
        ax.text(0.5, 0.5, "No response edges", ha="center", va="center")
        ax.axis("off")
        return ax
    all_nodes = sorted(set(edges[source_col].astype(str)).union(edges[target_col].astype(str)))
    coords = node_coordinates_from_metadata(
        elec_meta,
        nodes=all_nodes,
        elec_col=elec_col,
        coord_cols=coord_cols,
        region_col=region_col,
    )
    data, coords = _edge_subset_with_coords(edges, coords, source_col, target_col, weight_col, top_n)
    if data.empty or coords.empty:
        graph = graph_from_edges(edges, source_col=source_col, target_col=target_col, weight_col=weight_col)
        return plot_network(graph, ax=ax, weight_attr=weight_col)

    axis_map = {"mni_x": "x", "mni_y": "y", "mni_z": "z", "x": "x", "y": "y", "z": "z"}
    px, py = axis_map.get(projection[0], "x"), axis_map.get(projection[1], "y")
    coord_map = coords.set_index("node")[[px, py, "region"]].to_dict("index")
    weights = pd.to_numeric(data[weight_col], errors="coerce").fillna(0.0).abs()
    vmax = _robust_abs_vmax(weights)
    vmin = 0.0

    for _, edge in data.iterrows():
        src = str(edge[source_col])
        dst = str(edge[target_col])
        if src not in coord_map or dst not in coord_map:
            continue
        weight = abs(float(edge[weight_col]))
        x0, y0 = coord_map[src][px], coord_map[src][py]
        x1, y1 = coord_map[dst][px], coord_map[dst][py]
        color = _color_from_cmap(weight, vmin, vmax, cmap)
        rad = 0.08 if x1 >= x0 else -0.08
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=8,
            lw=0.80 + 3.40 * _positive_scale(weight, vmax),
            color=color,
            alpha=0.76,
            zorder=1,
        )
        ax.add_patch(arrow)

    strength = (
        pd.concat(
            [
                data[[source_col, weight_col]].rename(columns={source_col: "node"}),
                data[[target_col, weight_col]].rename(columns={target_col: "node"}),
            ]
        )
        .assign(abs_weight=lambda d: pd.to_numeric(d[weight_col], errors="coerce").fillna(0.0).abs())
        .groupby("node")["abs_weight"]
        .sum()
    )
    coords["strength"] = coords["node"].map(strength).fillna(0.0)
    node_sizes = 28 + 230 * coords["strength"] / (coords["strength"].max() if coords["strength"].max() else 1.0)
    stim_nodes = set(data[source_col].astype(str))
    response_coords = coords[~coords["node"].astype(str).isin(stim_nodes)]
    stim_coords = coords[coords["node"].astype(str).isin(stim_nodes)]
    if not response_coords.empty:
        ax.scatter(
            response_coords[px],
            response_coords[py],
            s=node_sizes[response_coords.index],
            c=RESPONSE_NODE_COLOR,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
            label="Recording response",
        )
    if not stim_coords.empty:
        ax.scatter(
            stim_coords[px],
            stim_coords[py],
            s=node_sizes[stim_coords.index] * STIM_NODE_STATIC_SCALE,
            c=STIM_NODE_COLOR,
            marker="D",
            edgecolors=STIM_NODE_ACCENT,
            linewidths=1.5,
            zorder=5,
            label="Stimulation source",
        )

    stim_nodes = set(data[source_col].astype(str))
    strong_targets = [
        node
        for node in coords.sort_values("strength", ascending=False)["node"].astype(str)
        if node not in stim_nodes
    ][:7]
    label_order = [*sorted(stim_nodes, key=natural_label_key), *strong_targets]
    offsets = [(8, 8), (10, -10), (-10, 9), (-12, -9), (0, 13), (13, 0), (-13, 0)]
    span = max(
        float(coords[px].max() - coords[px].min()) if len(coords) else 1.0,
        float(coords[py].max() - coords[py].min()) if len(coords) else 1.0,
        1.0,
    )
    min_dist = span * 0.055
    placed: list[tuple[float, float]] = []
    label_frame = coords.set_index("node")
    stim_points = [
        (float(label_frame.loc[node][px]), float(label_frame.loc[node][py]))
        for node in stim_nodes
        if node in label_frame.index
    ]
    for label_i, node in enumerate(label_order):
        if node not in label_frame.index:
            continue
        row = label_frame.loc[node]
        x, y = float(row[px]), float(row[py])
        if node not in stim_nodes:
            if any(np.hypot(x - sx, y - sy) < min_dist * 1.6 for sx, sy in stim_points):
                continue
            if any(np.hypot(x - px0, y - py0) < min_dist for px0, py0 in placed):
                continue
        placed.append((x, y))
        ax.annotate(
            node,
            xy=(x, y),
            xytext=offsets[label_i % len(offsets)],
            textcoords="offset points",
            fontsize=7.3,
            ha="center",
            va="center",
            color="#111827",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.72},
            zorder=4,
        )

    ax.set_xlabel(f"{projection[0]} (mm)" if projection[0].startswith("mni_") else projection[0])
    ax.set_ylabel(f"{projection[1]} (mm)" if projection[1].startswith("mni_") else projection[1])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_template_brain_network(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame,
    source_col: str = "stim_elec",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    coord_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    top_n: int | None = 80,
    views: tuple[str, ...] = ("left_lateral", "dorsal"),
    title: str = "Template brain response network",
    cmap: str = "plasma",
    node_size_values: dict[str, float] | pd.Series | None = None,
    node_color_values: dict[str, float] | pd.Series | None = None,
    node_color_label: str = "Active significance metrics",
    node_cmap: str = "YlOrRd",
    mesh: str = "fsaverage5",
    surface: str = "pial",
    fallback_shell: bool = True,
    validate_mni: bool = True,
    mni_mask_margin_mm: float = 4.0,
    outside_mni: str = "drop",
):
    """Render response edges and electrodes on a template cortical surface.

    ERPy first tries Nilearn's packaged fsaverage cortical meshes, which are
    derived from FreeSurfer surfaces and work well for MNI-coordinate public
    datasets. If Nilearn is unavailable, the function can fall back to the
    lightweight translucent shell. For patient-specific surgical figures, use
    `plot_surface_connectome` with that subject's FreeSurfer reconstruction.
    """

    all_nodes = sorted(set(edges.get(source_col, pd.Series(dtype=str)).astype(str)).union(edges.get(target_col, pd.Series(dtype=str)).astype(str)))
    coords = node_coordinates_from_metadata(
        elec_meta,
        nodes=all_nodes,
        elec_col=elec_col,
        coord_cols=coord_cols,
        region_col=region_col,
    )
    if validate_mni:
        coords = _filter_mni_brain_coordinates(
            coords,
            margin_mm=mni_mask_margin_mm,
            outside=outside_mni,
            context="Template cortical-surface network",
        )
    data, coords = _edge_subset_with_coords(edges, coords, source_col, target_col, weight_col, top_n)
    if data.empty or coords.empty:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.text2D(0.5, 0.5, "No response edges with MNI coordinates", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    n_views = len(views)
    fig = plt.figure(figsize=(6.7 * n_views, 6.1), layout="constrained")
    axes = [fig.add_subplot(1, n_views, i + 1, projection="3d") for i in range(n_views)]
    fig.suptitle(title, fontsize=16, y=0.99)
    cortical_surfaces = None
    shell_surfaces = None
    try:
        cortical_surfaces = _load_fsaverage_surfaces(mesh=mesh, surface=surface)
    except Exception:
        if not fallback_shell:
            raise
        shell_surfaces = _template_brain_surface()
    coord_map = coords.set_index("node")[["x", "y", "z", "region"]].to_dict("index")
    data = data.copy()
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce").fillna(0.0)
    weights = data[weight_col].abs()
    vmin = 0.0
    vmax = _robust_abs_vmax(weights)
    strength = (
        pd.concat(
            [
                data[[source_col, weight_col]].rename(columns={source_col: "node"}),
                data[[target_col, weight_col]].rename(columns={target_col: "node"}),
            ]
        )
        .assign(abs_weight=lambda d: pd.to_numeric(d[weight_col], errors="coerce").fillna(0.0).abs())
        .groupby("node")["abs_weight"]
        .sum()
    )
    coords["strength"] = coords["node"].map(strength).fillna(0.0)
    if node_size_values is not None:
        size_map = dict(pd.Series(node_size_values, dtype=float))
        coords["strength"] = coords["node"].astype(str).map(size_map).fillna(0.0)
    if node_color_values is not None:
        color_map = dict(pd.Series(node_color_values, dtype=float))
        coords["node_color_value"] = coords["node"].astype(str).map(color_map).fillna(0.0)
    stim_nodes = set(data[source_col].astype(str))

    def apply_view(ax, view: str):
        if view == "left_lateral":
            ax.view_init(elev=7, azim=180)
            ax.set_title("Left lateral")
        elif view == "right_lateral":
            ax.view_init(elev=7, azim=0)
            ax.set_title("Right lateral")
        elif view == "dorsal":
            ax.view_init(elev=82, azim=-90)
            ax.set_title("Dorsal")
        elif view == "anterior":
            ax.view_init(elev=8, azim=-90)
            ax.set_title("Anterior")
        else:
            ax.view_init(elev=12, azim=-65)
            ax.set_title(view.replace("_", " ").title())

    for ax, view in zip(axes, views):
        if cortical_surfaces is not None:
            for surf_coords, faces, _ in cortical_surfaces:
                plot_faces = faces[::2]
                ax.plot_trisurf(
                    surf_coords[:, 0],
                    surf_coords[:, 1],
                    surf_coords[:, 2],
                    triangles=plot_faces,
                    color="#d8d2c7",
                    alpha=0.20,
                    linewidth=0,
                    shade=True,
                    antialiased=False,
                    zorder=0,
                )
        elif shell_surfaces is not None:
            for x, y, z in shell_surfaces:
                ax.plot_surface(x, y, z, rstride=2, cstride=2, color="#d8d2c7", alpha=0.13, linewidth=0, shade=True)
        for _, edge in data.iterrows():
            src = str(edge[source_col])
            dst = str(edge[target_col])
            if src not in coord_map or dst not in coord_map:
                continue
            src_xyz = coord_map[src]
            dst_xyz = coord_map[dst]
            weight = abs(float(edge[weight_col]))
            edge_scale = _positive_scale(weight, vmax)
            ax.plot(
                [src_xyz["x"], dst_xyz["x"]],
                [src_xyz["y"], dst_xyz["y"]],
                [src_xyz["z"], dst_xyz["z"]],
                color=_color_from_cmap(weight, vmin, vmax, cmap),
                linewidth=0.85 + 3.80 * edge_scale,
                alpha=0.82,
                zorder=2,
            )
        coords["_node_size"] = 24 + 155 * coords["strength"] / (coords["strength"].max() if coords["strength"].max() else 1.0)
        response_coords = coords[~coords["node"].astype(str).isin(stim_nodes)]
        stim_coords = coords[coords["node"].astype(str).isin(stim_nodes)]
        if node_color_values is not None:
            node_colors = response_coords["node_color_value"].to_numpy(dtype=float) if not response_coords.empty else np.array([], dtype=float)
            if not response_coords.empty:
                scatter = ax.scatter(
                    response_coords["x"],
                    response_coords["y"],
                    response_coords["z"],
                    s=response_coords["_node_size"],
                    c=node_colors,
                    cmap=node_cmap,
                    vmin=0,
                    vmax=max(1.0, float(np.nanmax(node_colors)) if np.isfinite(node_colors).any() else 1.0),
                    edgecolors="white",
                    linewidths=0.75,
                    depthshade=False,
                    zorder=4,
                )
        else:
            if not response_coords.empty:
                scatter = ax.scatter(
                    response_coords["x"],
                    response_coords["y"],
                    response_coords["z"],
                    s=response_coords["_node_size"],
                    c=RESPONSE_NODE_COLOR,
                    edgecolors="white",
                    linewidths=0.75,
                    depthshade=False,
                    zorder=4,
                )
        if not stim_coords.empty:
            ax.scatter(
                stim_coords["x"],
                stim_coords["y"],
                stim_coords["z"],
                s=stim_coords["_node_size"] * STIM_NODE_STATIC_SCALE,
                c=STIM_NODE_COLOR,
                marker="D",
                edgecolors=STIM_NODE_ACCENT,
                linewidths=1.15,
                depthshade=False,
                zorder=6,
            )
        label_nodes = sorted(stim_nodes, key=natural_label_key)
        label_nodes.extend(
            [
                node
                for node in coords.sort_values("strength", ascending=False)["node"].astype(str).head(5)
                if node not in stim_nodes
            ]
        )
        for node in label_nodes:
            if node not in coord_map:
                continue
            xyz = coord_map[node]
            ax.text(xyz["x"], xyz["y"], xyz["z"], f" {node}", fontsize=7.4, color="#111827", zorder=5)
        if cortical_surfaces is not None:
            all_surface_coords = np.vstack([surf_coords for surf_coords, _, _ in cortical_surfaces])
            all_plot_coords = np.vstack([all_surface_coords, coords[["x", "y", "z"]].to_numpy(dtype=float)])
            mins = np.nanmin(all_plot_coords, axis=0)
            maxs = np.nanmax(all_plot_coords, axis=0)
            pad = np.maximum((maxs - mins) * 0.08, 5)
            ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
            ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
            ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
        else:
            ax.set_xlim(-92, 92)
            ax.set_ylim(-112, 78)
            ax.set_zlim(-68, 82)
        ax.set_box_aspect((184, 190, 150))
        ax.set_axis_off()
        apply_view(ax, view)
    if node_color_values is not None and scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, fraction=0.020, pad=0.01)
        cbar.set_label(node_color_label)
    return fig


def _node_metric_values(
    metric_table: pd.DataFrame | pd.Series | dict[str, float],
    metric: str = "hub",
    summary: str = "max",
    time_s: float | None = None,
    node_col: str = "node",
    value_col: str = "value",
    metric_col: str = "metric",
    time_col: str = "time_s",
) -> pd.Series:
    if isinstance(metric_table, pd.Series):
        return pd.to_numeric(metric_table, errors="coerce").dropna().sort_values(ascending=False)
    if isinstance(metric_table, dict):
        return pd.Series(metric_table, dtype=float).dropna().sort_values(ascending=False)
    data = metric_table.copy()
    if data.empty:
        return pd.Series(dtype=float)
    if metric_col in data.columns:
        data = data[data[metric_col].astype(str) == str(metric)]
    if time_s is not None and time_col in data.columns and not data.empty:
        nearest = data[time_col].iloc[np.argmin(np.abs(pd.to_numeric(data[time_col], errors="coerce") - time_s))]
        data = data[pd.to_numeric(data[time_col], errors="coerce") == nearest]
    if data.empty or node_col not in data.columns or value_col not in data.columns:
        return pd.Series(dtype=float)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    grouped = data.groupby(node_col)[value_col]
    if summary == "mean":
        values = grouped.mean()
    elif summary == "last":
        values = grouped.last()
    elif summary == "min":
        values = grouped.min()
    elif summary == "absmax":
        values = grouped.apply(lambda vals: vals.iloc[np.nanargmax(np.abs(vals.to_numpy(dtype=float)))] if vals.notna().any() else np.nan)
    else:
        values = grouped.max()
    return values.dropna().sort_values(ascending=False)


def plot_node_metric_template_brain(
    metric_table: pd.DataFrame | pd.Series | dict[str, float],
    elec_meta: pd.DataFrame,
    metric: str = "hub",
    summary: str = "max",
    time_s: float | None = None,
    stimulation_nodes: list[str] | set[str] | tuple[str, ...] | None = None,
    coord_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    views: tuple[str, ...] = ("left_lateral", "dorsal"),
    title: str | None = None,
    cmap: str = "magma",
    mesh: str = "fsaverage5",
    surface: str = "pial",
    fallback_shell: bool = True,
    validate_mni: bool = True,
    mni_mask_margin_mm: float = 4.0,
    outside_mni: str = "drop",
    top_n_labels: int = 8,
):
    """Render a dynamic graph metric summary on an fsaverage cortical surface."""

    values = _node_metric_values(metric_table, metric=metric, summary=summary, time_s=time_s)
    coords = node_coordinates_from_metadata(
        elec_meta,
        nodes=values.index.astype(str).tolist(),
        elec_col=elec_col,
        coord_cols=coord_cols,
        region_col=region_col,
    )
    if validate_mni:
        coords = _filter_mni_brain_coordinates(
            coords,
            margin_mm=mni_mask_margin_mm,
            outside=outside_mni,
            context="Template cortical-surface node metric",
        )
    coords["value"] = coords["node"].astype(str).map(values).fillna(0.0)
    coords = coords[coords["value"].notna()].copy()
    if coords.empty:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.text2D(0.5, 0.5, "No node metric values with MNI coordinates", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    n_views = len(views)
    fig = plt.figure(figsize=(6.7 * n_views, 6.1), layout="constrained")
    axes = [fig.add_subplot(1, n_views, i + 1, projection="3d") for i in range(n_views)]
    summary_label = f"{summary} {metric.replace('_', ' ')}" if time_s is None else f"{metric.replace('_', ' ')} at {time_s * 1000:.0f} ms"
    fig.suptitle(title or f"Node graph metric brain map: {summary_label}", fontsize=16, y=0.99)
    cortical_surfaces = None
    shell_surfaces = None
    try:
        cortical_surfaces = _load_fsaverage_surfaces(mesh=mesh, surface=surface)
    except Exception:
        if not fallback_shell:
            raise
        shell_surfaces = _template_brain_surface()

    vmax = float(np.nanpercentile(coords["value"].to_numpy(dtype=float), 98)) if np.isfinite(coords["value"]).any() else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    coords["_node_size"] = 26 + 190 * np.sqrt(np.clip(coords["value"].to_numpy(dtype=float) / vmax, 0.0, 1.0))
    stim_nodes = set(str(node) for node in (stimulation_nodes or []))

    def apply_view(ax, view: str):
        if view == "left_lateral":
            ax.view_init(elev=7, azim=180)
            ax.set_title("Left lateral")
        elif view == "right_lateral":
            ax.view_init(elev=7, azim=0)
            ax.set_title("Right lateral")
        elif view == "dorsal":
            ax.view_init(elev=82, azim=-90)
            ax.set_title("Dorsal")
        elif view == "anterior":
            ax.view_init(elev=8, azim=-90)
            ax.set_title("Anterior")
        else:
            ax.view_init(elev=12, azim=-65)
            ax.set_title(view.replace("_", " ").title())

    scatter = None
    for ax, view in zip(axes, views):
        if cortical_surfaces is not None:
            for surf_coords, faces, _ in cortical_surfaces:
                ax.plot_trisurf(
                    surf_coords[:, 0],
                    surf_coords[:, 1],
                    surf_coords[:, 2],
                    triangles=faces[::2],
                    color="#d8d2c7",
                    alpha=0.20,
                    linewidth=0,
                    shade=True,
                    antialiased=False,
                    zorder=0,
                )
        elif shell_surfaces is not None:
            for x, y, z in shell_surfaces:
                ax.plot_surface(x, y, z, rstride=2, cstride=2, color="#d8d2c7", alpha=0.13, linewidth=0, shade=True)
        response_coords = coords[~coords["node"].astype(str).isin(stim_nodes)]
        stim_coords = coords[coords["node"].astype(str).isin(stim_nodes)]
        if not response_coords.empty:
            scatter = ax.scatter(
                response_coords["x"],
                response_coords["y"],
                response_coords["z"],
                s=response_coords["_node_size"],
                c=response_coords["value"],
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                edgecolors="white",
                linewidths=0.85,
                depthshade=False,
                zorder=4,
            )
        if not stim_coords.empty:
            ax.scatter(
                stim_coords["x"],
                stim_coords["y"],
                stim_coords["z"],
                s=stim_coords["_node_size"] * STIM_NODE_STATIC_SCALE,
                c=STIM_NODE_COLOR,
                marker="D",
                edgecolors=STIM_NODE_ACCENT,
                linewidths=1.15,
                depthshade=False,
                zorder=6,
            )
        label_frame = pd.concat(
            [
                coords[coords["node"].astype(str).isin(stim_nodes)],
                coords[~coords["node"].astype(str).isin(stim_nodes)].sort_values("value", ascending=False).head(top_n_labels),
            ]
        ).drop_duplicates("node")
        for _, row in label_frame.iterrows():
            label = f" {row['node']}"
            if str(row["node"]) in stim_nodes:
                label += f" ({float(row['value']):.2g})"
            ax.text(row["x"], row["y"], row["z"], label, fontsize=7.4, color="#111827", zorder=5)
        if cortical_surfaces is not None:
            all_surface_coords = np.vstack([surf_coords for surf_coords, _, _ in cortical_surfaces])
            all_plot_coords = np.vstack([all_surface_coords, coords[["x", "y", "z"]].to_numpy(dtype=float)])
            mins = np.nanmin(all_plot_coords, axis=0)
            maxs = np.nanmax(all_plot_coords, axis=0)
            pad = np.maximum((maxs - mins) * 0.08, 5)
            ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
            ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
            ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
        else:
            ax.set_xlim(-92, 92)
            ax.set_ylim(-112, 78)
            ax.set_zlim(-68, 82)
        ax.set_box_aspect((184, 190, 150))
        ax.set_axis_off()
        apply_view(ax, view)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, fraction=0.020, pad=0.01)
        cbar.set_label(summary_label)
    return fig


def plot_interactive_connectome(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame,
    source_col: str = "stim_elec",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    coord_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    top_n: int | None = 80,
    title: str = "ERPy stimulation-response connectome",
    output_html: str | Path | None = None,
    cmap: str = "plasma",
    show_brain_shell: bool = True,
    mesh: str = "fsaverage5",
    surface: str = "pial",
    fallback_shell: bool = True,
    validate_mni: bool = True,
    mni_mask_margin_mm: float = 4.0,
    outside_mni: str = "drop",
):
    """Interactive 3D connectome on a template cortical surface.

    This is the most portable brain-network view: it needs electrode coordinates
    but not a subject MRI, FreeSurfer reconstruction, or graphical backend.
    When Nilearn is installed, the brain is rendered with fsaverage cortical
    meshes; otherwise ERPy falls back to a lightweight translucent shell.
    """

    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise ImportError("Interactive connectome plotting requires the optional 'plotly' dependency") from exc

    all_nodes = sorted(set(edges.get(source_col, pd.Series(dtype=str)).astype(str)).union(edges.get(target_col, pd.Series(dtype=str)).astype(str)))
    coords = node_coordinates_from_metadata(
        elec_meta,
        nodes=all_nodes,
        elec_col=elec_col,
        coord_cols=coord_cols,
        region_col=region_col,
    )
    if validate_mni:
        coords = _filter_mni_brain_coordinates(
            coords,
            margin_mm=mni_mask_margin_mm,
            outside=outside_mni,
            context="Interactive cortical-surface connectome",
        )
    data, coords = _edge_subset_with_coords(edges, coords, source_col, target_col, weight_col, top_n)
    if data.empty or coords.empty:
        raise ValueError("No edges have matching electrode coordinates")

    coord_map = coords.set_index("node")[["x", "y", "z", "region"]].to_dict("index")
    weights = data[weight_col].astype(float).to_numpy()
    abs_weights = np.abs(weights)
    vmin = 0.0
    vmax = _robust_abs_vmax(abs_weights)
    traces = (
        _plotly_template_brain_surface_traces(
            go,
            mesh=mesh,
            surface=surface,
            fallback_shell=fallback_shell,
        )
        if show_brain_shell
        else []
    )
    for _, edge in data.iterrows():
        src = str(edge[source_col])
        dst = str(edge[target_col])
        src_xyz = coord_map[src]
        dst_xyz = coord_map[dst]
        weight = float(edge[weight_col])
        edge_scale = _positive_scale(weight, vmax)
        width = 1.20 + 5.00 * edge_scale
        traces.append(
            go.Scatter3d(
                x=[src_xyz["x"], dst_xyz["x"]],
                y=[src_xyz["y"], dst_xyz["y"]],
                z=[src_xyz["z"], dst_xyz["z"]],
                mode="lines",
                line={
                    "color": _color_from_cmap(abs(weight), vmin, vmax, cmap),
                    "width": width,
                },
                opacity=0.84,
                hoverinfo="text",
                text=f"{src} -> {dst}<br>{weight_col}: {weight:.3g}",
                showlegend=False,
            )
        )

    strength = (
        pd.concat(
            [
                data[[source_col, weight_col]].rename(columns={source_col: "node"}),
                data[[target_col, weight_col]].rename(columns={target_col: "node"}),
            ]
        )
        .assign(abs_weight=lambda d: d[weight_col].abs())
        .groupby("node")["abs_weight"]
        .sum()
    )
    coords["strength"] = coords["node"].map(strength).fillna(0.0)
    coords["_marker_size"] = 4 + 13 * coords["strength"] / (coords["strength"].max() if coords["strength"].max() else 1.0)
    stim_nodes = set(data[source_col].astype(str))
    response_coords = coords[~coords["node"].astype(str).isin(stim_nodes)]
    stim_coords = coords[coords["node"].astype(str).isin(stim_nodes)]
    if not response_coords.empty:
        traces.append(
            go.Scatter3d(
                x=response_coords["x"],
                y=response_coords["y"],
                z=response_coords["z"],
                mode="markers+text",
                text=response_coords["node"],
                textposition="top center",
                marker={
                    "size": response_coords["_marker_size"],
                    "color": response_coords["strength"],
                    "colorscale": "Viridis",
                    "line": {"color": "white", "width": 1.0},
                    "opacity": 0.94,
                    "colorbar": {"title": "Node strength"},
                },
                hovertext=response_coords.apply(lambda r: f"{r['node']}<br>{r['region']}<br>strength: {r['strength']:.3g}", axis=1),
                hoverinfo="text",
                name="Recording responses",
            )
        )
    if not stim_coords.empty:
        traces.append(
            go.Scatter3d(
                x=stim_coords["x"],
                y=stim_coords["y"],
                z=stim_coords["z"],
                mode="markers+text",
                text=stim_coords["node"],
                textposition="bottom center",
                marker={
                    "size": stim_coords["_marker_size"] * STIM_NODE_STATIC_SCALE,
                    "symbol": "diamond",
                    "color": STIM_NODE_COLOR,
                    "line": {"color": STIM_NODE_ACCENT, "width": 1.8},
                    "opacity": 0.98,
                },
                hovertext=stim_coords.apply(lambda r: f"{r['node']}<br>stimulation source<br>{r['region']}<br>strength: {r['strength']:.3g}", axis=1),
                hoverinfo="text",
                name="Stimulation sources",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        width=980,
        height=760,
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        scene={
            "xaxis": {"title": "MNI x", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "yaxis": {"title": "MNI y", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "zaxis": {"title": "MNI z", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "aspectmode": "data",
        },
        font={"family": "Arial, sans-serif", "size": 12},
    )
    if output_html is not None:
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)
    return fig


def plot_evoked_response_graph(
    epochs,
    elec_meta: pd.DataFrame,
    stim_pair: str | None = None,
    stim_elec: str | None = None,
    detections: pd.DataFrame | None = None,
    channels: list[str] | None = None,
    metric: str = "peak_amplitude_uv",
    consensus_only: bool = True,
    response_window: tuple[float, float] = (0.0, 0.35),
    frame_step_ms: float = 8.0,
    top_n: int = 24,
    coord_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    title: str = "Evoked response graph over time",
    output_html: str | Path | None = None,
    cmap: str = "RdBu_r",
    show_brain_shell: bool = True,
    mesh: str = "fsaverage5",
    surface: str = "pial",
    fallback_shell: bool = True,
    validate_mni: bool = True,
    mni_mask_margin_mm: float = 4.0,
    outside_mni: str = "drop",
):
    """Animate mean evoked response amplitudes on a template cortical surface.

    Node color and size follow the mean response waveform at each post-stimulus
    time point. Edges run from the stimulation contact to recording contacts and
    brighten as response magnitude increases. Static fsaverage cortical meshes
    are kept outside the animation frames so the HTML remains tractable.
    """

    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise ImportError("Evoked response graph plotting requires the optional 'plotly' dependency") from exc
    if stim_elec is None:
        if stim_pair is None:
            raise ValueError("Provide stim_pair or stim_elec")
        stim_elec = stim_pair_to_source_electrode(stim_pair)

    mean = baseline_center_frame(
        epochs.get_mean_waveform(),
        getattr(epochs, "baseline", None),
    )
    times = mean.index.to_numpy(dtype=float)
    response_mask = (times >= response_window[0]) & (times <= response_window[1])
    if not response_mask.any():
        raise ValueError("response_window does not overlap epoch times")

    available_labels = elec_meta[elec_col].astype(str).tolist() if elec_col in elec_meta.columns else []
    stim_node = resolve_electrode_label(stim_elec, available_labels)
    if channels is None:
        if detections is not None and not detections.empty and "channel" in detections.columns:
            ranked = detections.copy()
            if consensus_only and "consensus_ch" in ranked.columns:
                ranked = ranked[ranked["consensus_ch"]]
            if metric in ranked.columns:
                ranked = ranked.drop_duplicates("channel").sort_values(metric, ascending=False)
            channels = ranked["channel"].astype(str).tolist()
        if not channels:
            response_mag = mean.loc[response_mask].abs().max(axis=0).sort_values(ascending=False)
            channels = response_mag.index.astype(str).tolist()
    channels = [ch for ch in channels if ch in mean.columns and ch != stim_node]
    channel_nodes = {ch: resolve_electrode_label(ch, available_labels) for ch in channels}
    node_names = [stim_node, *channel_nodes.values()]
    coords = node_coordinates_from_metadata(
        elec_meta,
        nodes=node_names,
        elec_col=elec_col,
        coord_cols=coord_cols,
        region_col=region_col,
    )
    if validate_mni:
        coords = _filter_mni_brain_coordinates(
            coords,
            margin_mm=mni_mask_margin_mm,
            outside=outside_mni,
            context="Evoked response brain graph",
        )
    coord_nodes = set(coords["node"])
    selected = [(ch, node) for ch, node in channel_nodes.items() if node in coord_nodes and node != stim_node]
    if top_n is not None:
        selected = selected[:top_n]
    if stim_node not in coord_nodes or not selected:
        raise ValueError("No selected response channels have matching electrode coordinates")

    frame_stride = max(1, int(round((frame_step_ms / 1000.0) * float(epochs.sfreq))))
    frame_indices = np.flatnonzero(response_mask)[::frame_stride]
    if frame_indices.size == 0:
        frame_indices = np.flatnonzero(response_mask)[:1]
    selected_channels = [ch for ch, _ in selected]
    values = mean[selected_channels].to_numpy(dtype=float)
    scale = float(np.nanpercentile(np.abs(values[response_mask]), 98))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    coord_map = coords.set_index("node")[["x", "y", "z", "region"]].to_dict("index")

    def edge_trace(channel: str, node: str, time_i: int):
        src = coord_map[stim_node]
        dst = coord_map[node]
        value = float(mean.iloc[time_i][channel])
        mag = abs(value)
        response_scale = _positive_scale(mag, scale)
        width = 0.90 + 4.80 * response_scale
        return go.Scatter3d(
            x=[src["x"], dst["x"]],
            y=[src["y"], dst["y"]],
            z=[src["z"], dst["z"]],
            mode="lines",
            line={"color": _color_from_cmap(value, -scale, scale, cmap), "width": width},
            opacity=float(np.clip(0.24 + 0.68 * response_scale, 0.24, 0.90)),
            hoverinfo="text",
            text=f"{stim_node} -> {node}<br>{channel}: {value:.3g}",
            showlegend=False,
        )

    def response_node_trace(time_i: int):
        node_values = {node: float(mean.iloc[time_i][ch]) for ch, node in selected}
        plot_nodes = [node for _, node in selected]
        amplitudes = np.array([node_values[n] for n in plot_nodes], dtype=float)
        sizes = 6 + 17 * np.clip(np.abs(amplitudes) / scale, 0.0, 1.0)
        text = [
            f"{node}<br>{coord_map[node]['region']}<br>{amp:.3g}"
            for node, amp in zip(plot_nodes, amplitudes)
        ]
        return go.Scatter3d(
            x=[coord_map[node]["x"] for node in plot_nodes],
            y=[coord_map[node]["y"] for node in plot_nodes],
            z=[coord_map[node]["z"] for node in plot_nodes],
            mode="markers+text",
            text=plot_nodes,
            textposition="top center",
            marker={
                "size": sizes,
                "color": amplitudes,
                "colorscale": _plotly_signed_amplitude_colorscale(),
                "cmin": -scale,
                "cmax": scale,
                "line": {"color": "white", "width": 1.0},
                "opacity": 0.95,
                "colorbar": {"title": "uV"},
            },
            hovertext=text,
            hoverinfo="text",
            showlegend=False,
        )

    def source_node_trace():
        src = coord_map[stim_node]
        return go.Scatter3d(
            x=[src["x"]],
            y=[src["y"]],
            z=[src["z"]],
            mode="markers+text",
            text=[stim_node],
            textposition="bottom center",
            marker={
                "size": STIM_NODE_PLOTLY_SIZE,
                "symbol": "diamond",
                "color": STIM_NODE_COLOR,
                "line": {"color": STIM_NODE_ACCENT, "width": 1.8},
                "opacity": 0.98,
            },
            hovertext=[f"{stim_node}<br>stimulation source<br>{coord_map[stim_node]['region']}"],
            hoverinfo="text",
            name="Stimulation source",
            showlegend=False,
        )

    brain_traces = (
        _plotly_template_brain_surface_traces(
            go,
            mesh=mesh,
            surface=surface,
            fallback_shell=fallback_shell,
        )
        if show_brain_shell
        else []
    )

    def dynamic_data(time_i: int):
        return [*[edge_trace(ch, node, time_i) for ch, node in selected], response_node_trace(time_i), source_node_trace()]

    def frame_data(time_i: int):
        return [*brain_traces, *dynamic_data(time_i)]

    initial_i = int(frame_indices[0])
    dynamic_trace_indices = list(range(len(brain_traces), len(frame_data(initial_i))))
    frames = [
        go.Frame(
            data=dynamic_data(int(time_i)),
            traces=dynamic_trace_indices,
            name=f"{times[time_i] * 1000:.0f} ms",
            layout={"title": {"text": f"{title} ({times[time_i] * 1000:.0f} ms)"}},
        )
        for time_i in frame_indices
    ]
    fig = go.Figure(data=frame_data(initial_i), frames=frames)
    slider_steps = [
        {
            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": frame.name,
            "method": "animate",
        }
        for frame in frames
    ]
    fig.update_layout(
        title={"text": f"{title} ({times[initial_i] * 1000:.0f} ms)", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        width=1000,
        height=780,
        margin={"l": 0, "r": 0, "t": 58, "b": 0},
        scene={
            "xaxis": {"title": "MNI x", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "yaxis": {"title": "MNI y", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "zaxis": {"title": "MNI z", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[{"active": 0, "steps": slider_steps, "x": 0.12, "y": 0.02, "len": 0.78}],
        font={"family": "Arial, sans-serif", "size": 12},
    )
    if output_html is not None:
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)
    return fig


def plot_aggregate_evoked_response_graph(
    runs: list[dict],
    elec_meta: pd.DataFrame,
    metric: str = "peak_amplitude_uv",
    response_metric: str = "voltage",
    response_metric_frame_key: str = "response_metric_frame",
    metric_label: str | None = None,
    consensus_only: bool = True,
    response_window: tuple[float, float] = (0.0, 0.35),
    frame_step_ms: float = 12.0,
    top_n_per_stim: int = 18,
    top_n_total: int | None = 90,
    coord_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    title: str = "Aggregate evoked response graph over time",
    output_html: str | Path | None = None,
    cmap: str = "plasma",
    show_brain_shell: bool = True,
    mesh: str = "fsaverage5",
    surface: str = "pial",
    fallback_shell: bool = True,
    validate_mni: bool = True,
    mni_mask_margin_mm: float = 4.0,
    outside_mni: str = "drop",
    significance_time_gate: float = 0.20,
):
    """Animate evoked responses from multiple stimulation sites on one brain.

    Each run dictionary should contain ``epochs`` and ``stim_pair``. Optional
    keys include ``detections``, ``channels``, and ``response_metric_frame``.
    By default, edges use absolute mean evoked voltage in uV. To animate another
    response metric such as CRP canonical weight or z-scored CRP expression,
    pass a time-indexed DataFrame in ``response_metric_frame`` with recording
    contacts as columns and set ``metric_label`` accordingly.
    """

    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise ImportError("Aggregate evoked response graph plotting requires the optional 'plotly' dependency") from exc

    if not runs:
        raise ValueError("runs must contain at least one stimulation run")
    available_labels = elec_meta[elec_col].astype(str).tolist() if elec_col in elec_meta.columns else []
    prepared = []
    all_nodes: set[str] = set()
    all_values = []
    edge_specs = []
    metric_label = metric_label or ("Mean evoked voltage (uV)" if response_metric == "voltage" else response_metric.replace("_", " "))

    def metric_frame_for_run(mean: pd.DataFrame, run: dict) -> pd.DataFrame:
        frame = run.get(response_metric_frame_key)
        if frame is None:
            return mean
        frame = pd.DataFrame(frame).copy()
        frame.index = pd.to_numeric(frame.index, errors="coerce")
        frame = frame.loc[np.isfinite(frame.index)]
        frame = frame.sort_index()
        if frame.empty:
            return mean
        target_times = mean.index.to_numpy(dtype=float)
        data: dict[str, np.ndarray] = {}
        for channel in mean.columns:
            if channel not in frame.columns:
                continue
            series = pd.to_numeric(frame[channel], errors="coerce").dropna()
            if series.empty:
                continue
            data[channel] = np.interp(
                target_times,
                series.index.to_numpy(dtype=float),
                series.to_numpy(dtype=float),
                left=np.nan,
                right=np.nan,
            )
        return pd.DataFrame(data, index=mean.index)

    def significance_counts(detections: pd.DataFrame | None) -> dict[str, int]:
        if not isinstance(detections, pd.DataFrame) or detections.empty or "channel" not in detections.columns:
            return {}
        data = detections.copy()
        if "significant" in data.columns:
            significant = data["significant"].astype(bool)
        elif {"p_value", "threshold"}.issubset(data.columns):
            significant = pd.to_numeric(data["p_value"], errors="coerce") < pd.to_numeric(data["threshold"], errors="coerce")
        elif "consensus_ch" in data.columns:
            significant = data["consensus_ch"].astype(bool)
        else:
            return {}
        data = data.loc[significant]
        if data.empty:
            return {}
        if "method" in data.columns:
            counts = data.groupby("channel")["method"].nunique()
        else:
            counts = data.groupby("channel").size()
        return {str(channel): int(count) for channel, count in counts.items()}

    for run_i, run in enumerate(runs):
        epochs = run.get("epochs")
        if epochs is None:
            raise ValueError("Each run must include an 'epochs' object")
        stim_pair = run.get("stim_pair") or getattr(epochs, "metadata", {}).get("stim_pair")
        if stim_pair is None:
            raise ValueError("Each run must include 'stim_pair' or epochs.metadata['stim_pair']")
        stim_node = resolve_electrode_label(stim_pair_to_source_electrode(str(stim_pair)), available_labels)
        mean = baseline_center_frame(
            epochs.get_mean_waveform(),
            getattr(epochs, "baseline", None),
        )
        metric_mean = metric_frame_for_run(mean, run)
        times = mean.index.to_numpy(dtype=float)
        response_mask = (times >= response_window[0]) & (times <= response_window[1])
        if not response_mask.any():
            continue

        channels = run.get("channels")
        detections = run.get("detections")
        method_counts = significance_counts(detections)
        if channels is None:
            if isinstance(detections, pd.DataFrame) and not detections.empty and "channel" in detections.columns:
                ranked = detections.copy()
                if consensus_only and "consensus_ch" in ranked.columns:
                    ranked = ranked[ranked["consensus_ch"]]
                if metric in ranked.columns and not ranked.empty:
                    ranked["_abs_metric"] = pd.to_numeric(ranked[metric], errors="coerce").abs()
                    ranked = ranked.drop_duplicates("channel").sort_values("_abs_metric", ascending=False)
                channels = ranked["channel"].astype(str).tolist()
            if not channels:
                response_mag = mean.loc[response_mask].abs().max(axis=0).sort_values(ascending=False)
                channels = response_mag.index.astype(str).tolist()

        selected = []
        for ch in channels:
            if ch not in mean.columns or ch not in metric_mean.columns:
                continue
            node = resolve_electrode_label(str(ch), available_labels)
            if node == stim_node:
                continue
            peak = float(metric_mean.loc[response_mask, ch].abs().max())
            selected.append({"channel": str(ch), "node": node, "peak": peak, "sig_count": method_counts.get(str(ch), 0)})
        selected = sorted(selected, key=lambda item: item["peak"], reverse=True)[:top_n_per_stim]
        if not selected:
            continue
        prepared.append(
            {
                "run_i": run_i,
                "stim_pair": str(stim_pair),
                "stim_node": stim_node,
                "mean": mean,
                "metric": metric_mean,
                "times": times,
                "selected": selected,
            }
        )
        all_nodes.add(stim_node)
        all_nodes.update(item["node"] for item in selected)
        all_values.append(metric_mean.loc[response_mask, [item["channel"] for item in selected]].to_numpy(dtype=float))

    if not prepared:
        raise ValueError("No runs had response-window data and selected response channels")

    coords = node_coordinates_from_metadata(
        elec_meta,
        nodes=sorted(all_nodes),
        elec_col=elec_col,
        coord_cols=coord_cols,
        region_col=region_col,
    )
    if validate_mni:
        coords = _filter_mni_brain_coordinates(
            coords,
            margin_mm=mni_mask_margin_mm,
            outside=outside_mni,
            context="Aggregate evoked response brain graph",
        )
    coord_nodes = set(coords["node"])
    for info in prepared:
        info["selected"] = [item for item in info["selected"] if item["node"] in coord_nodes and info["stim_node"] in coord_nodes]
        for item in info["selected"]:
            edge_specs.append(
                {
                    "run_i": info["run_i"],
                    "stim_pair": info["stim_pair"],
                    "stim_node": info["stim_node"],
                    "channel": item["channel"],
                    "node": item["node"],
                    "peak": item["peak"],
                    "sig_count": item.get("sig_count", 0),
                }
            )
    if top_n_total is not None:
        edge_specs = sorted(edge_specs, key=lambda item: item["peak"], reverse=True)[:top_n_total]
    if not edge_specs:
        raise ValueError("No selected aggregate response channels have matching electrode coordinates")

    prepared_by_run = {info["run_i"]: info for info in prepared}
    coord_map = coords.set_index("node")[["x", "y", "z", "region"]].to_dict("index")
    value_arrays = [arr for arr in all_values if arr.size]
    scale = float(np.nanpercentile(np.abs(np.concatenate([arr.ravel() for arr in value_arrays])), 98)) if value_arrays else 1.0
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    frame_times = np.arange(response_window[0], response_window[1] + frame_step_ms / 1000.0, frame_step_ms / 1000.0)
    frame_times = frame_times[np.isfinite(frame_times)]
    if frame_times.size == 0:
        frame_times = np.array([response_window[0]], dtype=float)

    def value_at(edge: dict, time_s: float) -> float:
        info = prepared_by_run[edge["run_i"]]
        idx = int(np.argmin(np.abs(info["times"] - time_s)))
        return float(info["metric"].iloc[idx][edge["channel"]])

    def edge_trace(edge: dict, time_s: float):
        src = coord_map[edge["stim_node"]]
        dst = coord_map[edge["node"]]
        value = value_at(edge, time_s)
        magnitude = abs(value)
        response_scale = _positive_scale(magnitude, scale)
        return go.Scatter3d(
            x=[src["x"], dst["x"]],
            y=[src["y"], dst["y"]],
            z=[src["z"], dst["z"]],
            mode="lines",
            line={"color": _color_from_cmap(magnitude, 0.0, scale, cmap), "width": 1.60 + 7.40 * response_scale},
            opacity=float(np.clip(0.28 + 0.70 * response_scale, 0.28, 0.98)),
            hoverinfo="text",
            text=(
                f"{edge['stim_pair']}: {edge['stim_node']} -> {edge['node']}"
                f"<br>{edge['channel']} {metric_label}: {value:.3g}"
                f"<br>magnitude: {magnitude:.3g}"
                f"<br>active significance methods: {edge.get('sig_count', 0)}"
            ),
            showlegend=False,
        )

    source_nodes = sorted({edge["stim_node"] for edge in edge_specs})
    response_nodes = sorted({edge["node"] for edge in edge_specs if edge["node"] not in source_nodes})
    cumulative_values = []
    significance_values = []
    for time_s in frame_times:
        for node in response_nodes:
            node_edges = [edge for edge in edge_specs if edge["node"] == node]
            cumulative = float(np.nansum([abs(value_at(edge, float(time_s))) for edge in node_edges]))
            active_count = int(
                np.nansum(
                    [
                        edge.get("sig_count", 0)
                        if abs(value_at(edge, float(time_s))) >= significance_time_gate * max(float(edge.get("peak", 0.0)), 1e-12)
                        else 0
                        for edge in node_edges
                    ]
                )
            )
            cumulative_values.append(cumulative)
            significance_values.append(active_count)
    node_scale = _robust_abs_vmax(cumulative_values, percentile=98.0)
    max_sig_count = max([1, *[int(value) for value in significance_values if np.isfinite(value)]])

    def response_node_trace(time_s: float):
        node_values = {}
        node_sig_counts = {}
        for node in response_nodes:
            node_edges = [edge for edge in edge_specs if edge["node"] == node]
            vals = [abs(value_at(edge, time_s)) for edge in node_edges]
            node_values[node] = float(np.nansum(vals)) if vals else 0.0
            node_sig_counts[node] = int(
                np.nansum(
                    [
                        edge.get("sig_count", 0)
                        if abs(value_at(edge, time_s)) >= significance_time_gate * max(float(edge.get("peak", 0.0)), 1e-12)
                        else 0
                        for edge in node_edges
                    ]
                )
            )
        cumulative = np.array([node_values[node] for node in response_nodes], dtype=float)
        sig_counts = np.array([node_sig_counts[node] for node in response_nodes], dtype=float)
        sizes = 5 + 19 * np.sqrt(np.clip(cumulative / node_scale, 0.0, 1.0))
        return go.Scatter3d(
            x=[coord_map[node]["x"] for node in response_nodes],
            y=[coord_map[node]["y"] for node in response_nodes],
            z=[coord_map[node]["z"] for node in response_nodes],
            mode="markers+text",
            text=response_nodes,
            textposition="top center",
            marker={
                "size": sizes,
                "color": sig_counts,
                "colorscale": _plotly_significance_count_colorscale(),
                "cmin": 0,
                "cmax": max_sig_count,
                "line": {"color": "white", "width": 1.0},
                "opacity": 0.95,
                "colorbar": {"title": "active metrics"},
            },
            hovertext=[
                (
                    f"{node}<br>{coord_map[node]['region']}"
                    f"<br>cumulative {metric_label}: {node_values[node]:.3g}"
                    f"<br>active significance methods: {node_sig_counts[node]}"
                )
                for node in response_nodes
            ],
            hoverinfo="text",
            showlegend=False,
        )

    def source_node_trace():
        return go.Scatter3d(
            x=[coord_map[node]["x"] for node in source_nodes],
            y=[coord_map[node]["y"] for node in source_nodes],
            z=[coord_map[node]["z"] for node in source_nodes],
            mode="markers+text",
            text=source_nodes,
            textposition="bottom center",
            marker={
                "size": STIM_NODE_AGGREGATE_PLOTLY_SIZE,
                "symbol": "diamond",
                "color": STIM_NODE_COLOR,
                "line": {"color": STIM_NODE_ACCENT, "width": 1.9},
                "opacity": 0.99,
            },
            hovertext=[f"{node}<br>stimulation source<br>{coord_map[node]['region']}" for node in source_nodes],
            hoverinfo="text",
            name="Stimulation sources",
            showlegend=False,
        )

    brain_traces = (
        _plotly_template_brain_surface_traces(
            go,
            mesh=mesh,
            surface=surface,
            fallback_shell=fallback_shell,
        )
        if show_brain_shell
        else []
    )

    def dynamic_data(time_s: float):
        return [*[edge_trace(edge, time_s) for edge in edge_specs], response_node_trace(time_s), source_node_trace()]

    initial_t = float(frame_times[0])
    initial_data = [*brain_traces, *dynamic_data(initial_t)]
    dynamic_trace_indices = list(range(len(brain_traces), len(initial_data)))
    frames = [
        go.Frame(
            data=dynamic_data(float(time_s)),
            traces=dynamic_trace_indices,
            name=f"{time_s * 1000:.0f} ms",
            layout={"title": {"text": f"{title} ({time_s * 1000:.0f} ms)"}},
        )
        for time_s in frame_times
    ]
    fig = go.Figure(data=initial_data, frames=frames)
    slider_steps = [
        {
            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": frame.name,
            "method": "animate",
        }
        for frame in frames
    ]
    fig.update_layout(
        title={"text": f"{title} ({initial_t * 1000:.0f} ms)", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        width=1000,
        height=780,
        margin={"l": 0, "r": 0, "t": 58, "b": 0},
        scene={
            "xaxis": {"title": "MNI x", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "yaxis": {"title": "MNI y", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "zaxis": {"title": "MNI z", "showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[{"active": 0, "steps": slider_steps, "x": 0.12, "y": 0.02, "len": 0.78}],
        font={"family": "Arial, sans-serif", "size": 12},
    )
    if output_html is not None:
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)
    return fig


def plot_surface_connectome(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame,
    subjects_dir: str | Path,
    subject: str = "fsaverage",
    surface: str = "pial",
    source_col: str = "stim_elec",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    coord_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    top_n: int | None = 80,
    output_png: str | Path | None = None,
    off_screen: bool = True,
    title: str = "ERPy surface connectome",
    cmap: str = "plasma",
    edge_radius: float = 0.18,
    edge_radius_scale: float = 1.20,
    edge_opacity: float = 0.82,
):
    """Render a high-fidelity surface connectome with PyVista/MNE.

    The electrode coordinates must be in the same coordinate frame as the
    FreeSurfer surface. For subject-specific figures, pass that subject's
    `subjects_dir` and coordinates from the same reconstruction.
    """

    try:
        import pyvista as pv
        import mne
    except Exception as exc:
        raise ImportError("Surface connectome plotting requires optional dependencies: mne and pyvista") from exc

    all_nodes = sorted(set(edges.get(source_col, pd.Series(dtype=str)).astype(str)).union(edges.get(target_col, pd.Series(dtype=str)).astype(str)))
    coords = node_coordinates_from_metadata(
        elec_meta,
        nodes=all_nodes,
        elec_col=elec_col,
        coord_cols=coord_cols,
        region_col=region_col,
    )
    data, coords = _edge_subset_with_coords(edges, coords, source_col, target_col, weight_col, top_n)
    if data.empty or coords.empty:
        raise ValueError("No edges have matching electrode coordinates")

    subjects_dir = Path(subjects_dir).expanduser()
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1800, 1400))
    plotter.set_background("white")
    for hemi, color in (("lh", "#d8d0c4"), ("rh", "#e5ded2")):
        surf_path = subjects_dir / subject / "surf" / f"{hemi}.{surface}"
        if not surf_path.exists():
            continue
        vertices, faces = mne.read_surface(str(surf_path), verbose=False)
        face_block = np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel()
        mesh = pv.PolyData(vertices, face_block)
        plotter.add_mesh(mesh, color=color, opacity=0.24, smooth_shading=True, specular=0.18, roughness=0.52)
    if len(plotter.renderer.actors) == 0:
        raise FileNotFoundError(f"No FreeSurfer surfaces found under {subjects_dir / subject / 'surf'}")

    coord_map = coords.set_index("node")[["x", "y", "z", "region"]].to_dict("index")
    weights = data[weight_col].astype(float).abs().to_numpy()
    vmax = _robust_abs_vmax(weights)
    vmin = 0.0
    for _, edge in data.iterrows():
        src = str(edge[source_col])
        dst = str(edge[target_col])
        src_xyz = np.array([coord_map[src]["x"], coord_map[src]["y"], coord_map[src]["z"]], dtype=float)
        dst_xyz = np.array([coord_map[dst]["x"], coord_map[dst]["y"], coord_map[dst]["z"]], dtype=float)
        weight = abs(float(edge[weight_col]))
        tube = pv.Line(src_xyz, dst_xyz).tube(radius=edge_radius + edge_radius_scale * _positive_scale(weight, vmax), n_sides=18)
        plotter.add_mesh(tube, color=_color_from_cmap(weight, vmin, vmax, cmap), opacity=edge_opacity, smooth_shading=True)

    stim_nodes = set(data[source_col].astype(str))
    response_points = coords[~coords["node"].astype(str).isin(stim_nodes)][["x", "y", "z"]].to_numpy(dtype=float)
    source_points = coords[coords["node"].astype(str).isin(stim_nodes)][["x", "y", "z"]].to_numpy(dtype=float)
    if len(response_points):
        plotter.add_points(
            pv.PolyData(response_points),
            color=RESPONSE_NODE_COLOR,
            point_size=14,
            render_points_as_spheres=True,
            specular=0.45,
        )
    if len(source_points):
        plotter.add_points(
            pv.PolyData(source_points),
            color=STIM_NODE_ACCENT,
            point_size=15,
            render_points_as_spheres=True,
            specular=0.52,
        )
    if len(coords) <= 30:
        points = coords[["x", "y", "z"]].to_numpy(dtype=float)
        plotter.add_point_labels(points, coords["node"].tolist(), font_size=12, text_color="#111827", shape=None)
    plotter.add_text(title, position="upper_left", color="#111827", font_size=14)
    plotter.camera_position = "xy"
    plotter.camera.zoom(1.22)
    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(output_png), auto_close=True)
    return plotter


def _connectome(edges, elec_meta, stim_col, record_col, weight_col, region_col, mni_cols, title, display_mode, edge_cmap):
    try:
        from nilearn import plotting
    except Exception as exc:
        raise ImportError("Connectome plotting requires the optional 'nilearn' dependency") from exc
    coords = region_centroid_coords(elec_meta, region_col=region_col, mni_cols=mni_cols)
    region_order = sorted(set(edges[stim_col]).union(edges[record_col]).intersection(coords[region_col]))
    if not region_order:
        raise ValueError("No edge regions have MNI coordinates")
    coord_map = coords.set_index(region_col)[list(mni_cols)].loc[region_order]
    matrix = pd.DataFrame(0.0, index=region_order, columns=region_order)
    for _, row in edges.iterrows():
        if row[stim_col] in matrix.index and row[record_col] in matrix.columns:
            matrix.loc[row[stim_col], row[record_col]] = float(row[weight_col])
    source_regions = set(edges[stim_col].astype(str))
    node_colors = [STIM_NODE_COLOR if region in source_regions else RESPONSE_NODE_COLOR_ALT for region in region_order]
    display = plotting.plot_connectome(
        matrix.to_numpy(),
        coord_map.to_numpy(),
        node_color=node_colors,
        edge_cmap=edge_cmap,
        title=title,
        display_mode=display_mode,
    )
    overlay_regions = [region for region in region_order if region in source_regions]
    if overlay_regions:
        display.add_markers(coord_map.loc[overlay_regions].to_numpy(), marker_color=STIM_NODE_ACCENT, marker_size=28)
    return display, region_order


def plot_region_connectome(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame,
    stim_col: str = "stim_region",
    record_col: str = "record_region",
    weight_col: str = "weight",
    mni_cols=("mni_x", "mni_y", "mni_z"),
    region_col: str = "anat_label",
    metric_label: str | None = None,
    title: str = "Region connectome",
    edge_cmap: str = "plasma",
    display_mode: str = "lyrz",
):
    """Plot region-level directed response edges on an MNI connectome view."""

    return _connectome(edges, elec_meta, stim_col, record_col, weight_col, region_col, mni_cols, title, display_mode, edge_cmap)


def plot_electrode_connectome(
    adjacency: pd.DataFrame,
    elec_meta: pd.DataFrame,
    mni_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    metric_label: str | None = None,
    title: str = "Electrode connectome",
    edge_cmap: str = "plasma",
    display_mode: str = "lyrz",
):
    """Plot an electrode adjacency matrix at registered MNI coordinates."""

    try:
        from nilearn import plotting
    except Exception as exc:
        raise ImportError("Connectome plotting requires the optional 'nilearn' dependency") from exc
    meta = elec_meta.dropna(subset=[elec_col, *mni_cols]).set_index(elec_col)
    nodes = [n for n in sorted(set(adjacency.index).union(adjacency.columns)) if n in meta.index]
    matrix = adjacency.reindex(index=nodes, columns=nodes, fill_value=0.0).to_numpy()
    coords = meta.loc[nodes, list(mni_cols)].astype(float).to_numpy()
    return plotting.plot_connectome(matrix, coords, title=title, edge_cmap=edge_cmap, display_mode=display_mode)


def plot_node_metric_glass_brain(
    metric_table: pd.DataFrame | pd.Series | dict[str, float],
    elec_meta: pd.DataFrame,
    metric: str = "hub",
    summary: str = "max",
    time_s: float | None = None,
    stimulation_nodes: list[str] | set[str] | tuple[str, ...] | None = None,
    mni_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    title: str | None = None,
    cmap: str = "magma",
    display_mode: str = "lyrz",
    node_size: float = 42.0,
    brain_alpha: float = 0.72,
    validate_mni: bool = True,
    mni_mask_margin_mm: float = 4.0,
    outside_mni: str = "drop",
):
    """Visualize a node graph metric summary on Nilearn's MNI glass brain."""

    try:
        from nilearn import plotting
    except Exception as exc:
        raise ImportError("Glass-brain node metric plotting requires the optional 'nilearn' dependency") from exc
    values = _node_metric_values(metric_table, metric=metric, summary=summary, time_s=time_s)
    if values.empty:
        raise ValueError("No node metric values to plot")
    meta = elec_meta.dropna(subset=[elec_col, *mni_cols]).copy()
    if meta.empty:
        raise ValueError("electrode metadata does not contain MNI coordinates")
    meta[elec_col] = meta[elec_col].astype(str)
    meta = meta.drop_duplicates(elec_col).set_index(elec_col)
    nodes = order_nodes_by_metadata([node for node in values.index.astype(str) if node in meta.index], elec_meta=elec_meta, elec_col=elec_col)
    if validate_mni and nodes:
        coord_frame = pd.DataFrame(
            {
                "node": nodes,
                "x": meta.loc[nodes, mni_cols[0]].astype(float).to_numpy(),
                "y": meta.loc[nodes, mni_cols[1]].astype(float).to_numpy(),
                "z": meta.loc[nodes, mni_cols[2]].astype(float).to_numpy(),
            }
        )
        coord_frame = _filter_mni_brain_coordinates(
            coord_frame,
            margin_mm=mni_mask_margin_mm,
            outside=outside_mni,
            context="Nilearn glass-brain node metric",
        )
        nodes = coord_frame["node"].astype(str).tolist()
    if not nodes:
        raise ValueError("No node metric values have matching in-brain MNI coordinates")
    stim_nodes = set(str(node) for node in (stimulation_nodes or []))
    response_nodes = [node for node in nodes if node not in stim_nodes]
    source_nodes = [node for node in nodes if node in stim_nodes]
    coords = meta.loc[nodes, list(mni_cols)].astype(float).to_numpy()
    node_values = values.reindex(nodes).fillna(0.0).to_numpy(dtype=float)
    vmax = float(np.nanpercentile(node_values, 98)) if np.isfinite(node_values).any() else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    summary_label = f"{summary} {metric.replace('_', ' ')}" if time_s is None else f"{metric.replace('_', ' ')} at {time_s * 1000:.0f} ms"
    if response_nodes:
        response_coords = meta.loc[response_nodes, list(mni_cols)].astype(float).to_numpy()
        response_values = values.reindex(response_nodes).fillna(0.0).to_numpy(dtype=float)
        response_sizes = node_size * (0.55 + 1.30 * np.sqrt(np.clip(response_values / vmax, 0.0, 1.0)))
        display = plotting.plot_markers(
            response_values,
            response_coords,
            node_size=response_sizes,
            node_cmap=cmap,
            node_vmin=0,
            node_vmax=vmax,
            alpha=brain_alpha,
            title=title or f"Node metric glass brain: {summary_label}",
            display_mode=display_mode,
            colorbar=True,
        )
    else:
        source_coords = meta.loc[source_nodes, list(mni_cols)].astype(float).to_numpy()
        display = plotting.plot_markers(
            np.zeros(len(source_nodes), dtype=float),
            source_coords,
            node_size=node_size,
            node_cmap="Greys",
            node_vmin=0,
            node_vmax=1,
            alpha=brain_alpha,
            title=title or f"Node metric glass brain: {summary_label}",
            display_mode=display_mode,
            colorbar=False,
        )
    if source_nodes:
        source_coords = meta.loc[source_nodes, list(mni_cols)].astype(float).to_numpy()
        source_values = values.reindex(source_nodes).fillna(0.0).to_numpy(dtype=float)
        source_sizes = node_size * (0.78 + 0.88 * np.sqrt(np.clip(source_values / vmax, 0.0, 1.0)))
        display.add_markers(source_coords, marker_color=STIM_NODE_ACCENT, marker_size=source_sizes)
    return display


def plot_glass_brain_network(
    edges: pd.DataFrame,
    elec_meta: pd.DataFrame,
    source_col: str = "stim_elec",
    target_col: str = "record_elec",
    weight_col: str = "weight",
    mni_cols=("mni_x", "mni_y", "mni_z"),
    elec_col: str = "elec_label",
    title: str | None = "Glass brain response network",
    edge_cmap: str = "plasma",
    display_mode: str = "lyrz",
    edge_linewidth: float = 1.45,
    edge_alpha: float = 0.82,
    edge_threshold: str | float | None = None,
    node_size: float = 42.0,
    directed: bool = False,
    brain_alpha: float = 0.72,
    validate_mni: bool = True,
    mni_mask_margin_mm: float = 4.0,
    outside_mni: str = "drop",
):
    """Plot an edge table on Nilearn's MNI glass-brain display.

    The default is a symmetric edge display because Nilearn's directed arrows
    become visually heavy in dense intracranial stimulation networks. Source
    electrodes remain highlighted, and ``directed=True`` is available when
    arrow geometry is explicitly desired.
    """

    try:
        from nilearn import plotting
    except Exception as exc:
        raise ImportError("Glass-brain plotting requires the optional 'nilearn' dependency") from exc
    if edges.empty:
        raise ValueError("edges is empty")
    meta = elec_meta.dropna(subset=[elec_col, *mni_cols]).copy()
    if meta.empty:
        raise ValueError("electrode metadata does not contain MNI coordinates")
    meta[elec_col] = meta[elec_col].astype(str)
    meta = meta.drop_duplicates(elec_col).set_index(elec_col)
    nodes = order_nodes_by_metadata(
        sorted(set(edges[source_col].astype(str)).union(edges[target_col].astype(str))),
        elec_meta=elec_meta,
        elec_col=elec_col,
    )
    nodes = [node for node in nodes if node in meta.index]
    if not nodes:
        raise ValueError("No edge nodes have matching MNI coordinates")
    if validate_mni:
        coord_frame = pd.DataFrame(
            {
                "node": nodes,
                "x": meta.loc[nodes, mni_cols[0]].astype(float).to_numpy(),
                "y": meta.loc[nodes, mni_cols[1]].astype(float).to_numpy(),
                "z": meta.loc[nodes, mni_cols[2]].astype(float).to_numpy(),
            }
        )
        coord_frame = _filter_mni_brain_coordinates(
            coord_frame,
            margin_mm=mni_mask_margin_mm,
            outside=outside_mni,
            context="Nilearn glass-brain network",
        )
        nodes = coord_frame["node"].astype(str).tolist()
        if not nodes:
            raise ValueError("No edge nodes remain inside the MNI152 brain mask")
    matrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for _, row in edges.iterrows():
        src = str(row[source_col])
        dst = str(row[target_col])
        if src in matrix.index and dst in matrix.columns:
            value = float(row.get(weight_col, 1.0))
            if directed:
                matrix.loc[src, dst] = value
            else:
                current = matrix.loc[src, dst]
                if abs(value) >= abs(current):
                    matrix.loc[src, dst] = value
                    matrix.loc[dst, src] = value
    coords = meta.loc[nodes, list(mni_cols)].astype(float).to_numpy()
    source_nodes = set(edges[source_col].astype(str))
    node_colors = [STIM_NODE_COLOR if node in source_nodes else RESPONSE_NODE_COLOR_ALT for node in nodes]
    edge_vmax = _robust_abs_vmax(matrix.to_numpy())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'adjacency_matrix' is not symmetric.*", category=UserWarning)
        display = plotting.plot_connectome(
            matrix.to_numpy(),
            coords,
            node_color=node_colors,
            node_size=node_size,
            edge_cmap=edge_cmap,
            edge_vmin=0.0,
            edge_vmax=edge_vmax if edge_vmax > 0 else 1.0,
            edge_threshold=edge_threshold,
            alpha=brain_alpha,
            edge_kwargs={"linewidth": edge_linewidth, "alpha": edge_alpha},
            title=title,
            display_mode=display_mode,
        )
    overlay_nodes = [node for node in nodes if node in source_nodes]
    if overlay_nodes:
        source_coords = meta.loc[overlay_nodes, list(mni_cols)].astype(float).to_numpy()
        display.add_markers(source_coords, marker_color=STIM_NODE_ACCENT, marker_size=node_size * STIM_NODE_GLASS_SCALE)
    return display
