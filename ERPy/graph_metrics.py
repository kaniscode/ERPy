from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .baseline import baseline_center_frame
from .viz.networks import (
    electrode_label_identity,
    resolve_electrode_label,
    stim_pair_to_contact_electrodes,
    stim_pair_to_source_electrode,
)


DEFAULT_GRAPH_METRICS = (
    "in_strength",
    "out_strength",
    "total_strength",
    "hub",
    "authority",
)


def opportunity_conditioned_reciprocity(
    edges: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    response_col: str = "responding",
    group_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Estimate reciprocity only where both directions were sampled.

    The denominator is the number of bidirectionally tested unordered node
    pairs with at least one detected response. This prevents an unavailable
    reverse stimulation direction from being interpreted as a negative edge.
    Duplicate directed opportunities are collapsed with an ``any`` response
    rule before pair-level counting.
    """

    group_cols = list(group_cols or [])
    result_columns = [
        *group_cols,
        "bidirectionally_tested_pairs",
        "pairs_with_any_response",
        "bidirectional_pairs",
        "unidirectional_pairs",
        "reciprocity",
    ]
    if edges.empty:
        return pd.DataFrame(columns=result_columns)
    needed = [source_col, target_col, response_col, *group_cols]
    missing = [column for column in needed if column not in edges.columns]
    if missing:
        raise ValueError(f"edges is missing required columns: {missing}")

    data = edges[needed].copy()
    data[source_col] = data[source_col].astype(str)
    data[target_col] = data[target_col].astype(str)
    data = data[data[source_col] != data[target_col]]
    responses = data[response_col]
    if not pd.api.types.is_bool_dtype(responses):
        responses = (
            responses.fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
    data[response_col] = responses.fillna(False).astype(bool)
    if data.empty:
        return pd.DataFrame(columns=result_columns)

    collapsed = (
        data.groupby(
            [*group_cols, source_col, target_col],
            dropna=False,
            sort=False,
        )[response_col]
        .any()
        .reset_index()
    )
    grouped = (
        collapsed.groupby(group_cols, dropna=False, sort=False)
        if group_cols
        else [((), collapsed)]
    )
    rows = []
    for group_values, frame in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        opportunity_directions = set(
            zip(frame[source_col], frame[target_col])
        )
        response_directions = set(
            zip(
                frame.loc[frame[response_col], source_col],
                frame.loc[frame[response_col], target_col],
            )
        )
        tested_pairs = {
            tuple(sorted((source, target)))
            for source, target in opportunity_directions
            if (target, source) in opportunity_directions
        }
        any_response = sum(
            (left, right) in response_directions
            or (right, left) in response_directions
            for left, right in tested_pairs
        )
        bidirectional = sum(
            (left, right) in response_directions
            and (right, left) in response_directions
            for left, right in tested_pairs
        )
        row = {
            column: value
            for column, value in zip(group_cols, group_values)
        }
        row.update(
            {
                "bidirectionally_tested_pairs": int(len(tested_pairs)),
                "pairs_with_any_response": int(any_response),
                "bidirectional_pairs": int(bidirectional),
                "unidirectional_pairs": int(any_response - bidirectional),
                "reciprocity": (
                    float(bidirectional / any_response)
                    if any_response
                    else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=result_columns)


def bidirectional_edge_summary(
    edges: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    group_cols: Iterable[str] | None = None,
    min_weight: float = 0.0,
    agg: str = "mean",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Summarize reciprocal directed edges and directional asymmetry.

    Rows are unordered node pairs within optional grouping columns. ``a_to_b``
    and ``b_to_a`` are aggregated directed weights; ``asymmetry_log2`` is
    positive when the first listed direction is stronger and negative when the
    reverse direction is stronger.
    """

    group_cols = list(group_cols or [])
    result_columns = [
        *group_cols,
        "node_a",
        "node_b",
        "a_to_b",
        "b_to_a",
        "mean_weight",
        "min_direction_weight",
        "asymmetry_log2",
        "dominant_direction",
    ]
    if edges.empty:
        return pd.DataFrame(columns=result_columns)
    needed = [source_col, target_col, weight_col, *group_cols]
    missing = [col for col in needed if col not in edges.columns]
    if missing:
        raise ValueError(f"edges is missing required columns: {missing}")
    data = edges[needed].copy()
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
    data = data[np.isfinite(data[weight_col]) & (data[weight_col].abs() > float(min_weight))]
    data[source_col] = data[source_col].astype(str)
    data[target_col] = data[target_col].astype(str)
    data = data[data[source_col] != data[target_col]]
    if data.empty:
        return pd.DataFrame(columns=result_columns)
    directed = (
        data.groupby([*group_cols, source_col, target_col], dropna=False)[weight_col]
        .agg(agg)
        .reset_index()
    )
    lookup = {
        tuple(row[col] for col in [*group_cols, source_col, target_col]): float(row[weight_col])
        for _, row in directed.iterrows()
    }
    rows = []
    for group_values, frame in directed.groupby(group_cols, dropna=False) if group_cols else [((), directed)]:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        nodes = sorted(set(frame[source_col]).union(frame[target_col]))
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                key_ab = (*group_values, a, b)
                key_ba = (*group_values, b, a)
                w_ab = lookup.get(key_ab, 0.0)
                w_ba = lookup.get(key_ba, 0.0)
                if w_ab <= min_weight or w_ba <= min_weight:
                    continue
                asym = float(np.log2((abs(w_ab) + eps) / (abs(w_ba) + eps)))
                row = {col: value for col, value in zip(group_cols, group_values)}
                row.update(
                    {
                        "node_a": a,
                        "node_b": b,
                        "a_to_b": w_ab,
                        "b_to_a": w_ba,
                        "mean_weight": float(np.mean([abs(w_ab), abs(w_ba)])),
                        "min_direction_weight": float(min(abs(w_ab), abs(w_ba))),
                        "asymmetry_log2": asym,
                        "dominant_direction": f"{a}->{b}" if asym >= 0 else f"{b}->{a}",
                    }
                )
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=result_columns)
    return (
        pd.DataFrame(rows, columns=result_columns)
        .sort_values(
            [*group_cols, "mean_weight"],
            ascending=[True] * len(group_cols) + [False],
        )
        .reset_index(drop=True)
    )


def _weighted_module_cartography(
    graph: nx.DiGraph,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return participation and within-module strength z for graph nodes."""

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, attributes in graph.edges(data=True):
        weight = float(attributes.get("weight", 0.0))
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    if undirected.number_of_edges() == 0:
        zeros = {str(node): 0.0 for node in undirected.nodes}
        return zeros, zeros.copy()

    communities = list(
        nx.community.greedy_modularity_communities(
            undirected,
            weight="weight",
        )
    )
    membership = {
        str(node): community_index
        for community_index, community in enumerate(communities)
        for node in community
    }
    participation: dict[str, float] = {}
    within_strength: dict[str, float] = {}
    for node in undirected.nodes:
        module_strengths: dict[int, float] = {}
        for neighbor, attributes in undirected[node].items():
            module = membership[str(neighbor)]
            module_strengths[module] = module_strengths.get(
                module, 0.0
            ) + float(attributes.get("weight", 0.0))
        total_strength = float(sum(module_strengths.values()))
        participation[str(node)] = (
            float(
                1.0
                - sum(
                    (module_strength / total_strength) ** 2
                    for module_strength in module_strengths.values()
                )
            )
            if total_strength > 0
            else 0.0
        )
        within_strength[str(node)] = float(
            module_strengths.get(membership[str(node)], 0.0)
        )

    within_module_z: dict[str, float] = {}
    for community in communities:
        nodes = [str(node) for node in community]
        values = np.asarray(
            [within_strength[node] for node in nodes],
            dtype=float,
        )
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=0))
        for node in nodes:
            within_module_z[node] = (
                float((within_strength[node] - mean) / standard_deviation)
                if standard_deviation > 1e-12
                else 0.0
            )
    return participation, within_module_z


def weighted_directed_node_metrics(
    edges: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    group_cols: Iterable[str] | None = None,
    min_weight: float = 0.0,
    agg: str = "median",
) -> pd.DataFrame:
    """Compute interpretable weighted metrics for directed network nodes.

    Edge weights must be nonnegative connection strengths. Shortest-path
    measures use their reciprocal as distance, while local reaching centrality
    receives the original strength because its definition already treats larger
    weights as shorter, stronger connections.
    """

    group_cols = list(group_cols or [])
    result_columns = [
        *group_cols,
        "node",
        "in_degree",
        "out_degree",
        "in_strength",
        "out_strength",
        "total_strength",
        "in_strength_share",
        "out_strength_share",
        "sender_receiver_index",
        "pagerank",
        "betweenness",
        "out_closeness",
        "clustering",
        "module_participation",
        "within_module_z",
        "local_reaching_centrality",
    ]
    if edges.empty:
        return pd.DataFrame(columns=result_columns)
    needed = [source_col, target_col, weight_col, *group_cols]
    missing = [column for column in needed if column not in edges.columns]
    if missing:
        raise ValueError(f"edges is missing required columns: {missing}")

    data = edges[needed].copy()
    data[source_col] = data[source_col].astype(str)
    data[target_col] = data[target_col].astype(str)
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
    data = data[
        np.isfinite(data[weight_col])
        & data[weight_col].gt(float(min_weight))
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=result_columns)
    directed = (
        data.groupby(
            [*group_cols, source_col, target_col],
            dropna=False,
            sort=True,
        )[weight_col]
        .agg(agg)
        .reset_index()
    )
    grouped = (
        directed.groupby(group_cols, dropna=False, sort=True)
        if group_cols
        else [((), directed)]
    )
    rows: list[dict[str, float | str]] = []
    for group_values, frame in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        graph = nx.DiGraph()
        nodes = sorted(
            set(frame[source_col].astype(str)).union(
                frame[target_col].astype(str)
            )
        )
        graph.add_nodes_from(nodes)
        for edge in frame.sort_values([source_col, target_col]).itertuples(
            index=False
        ):
            source = str(getattr(edge, source_col))
            target = str(getattr(edge, target_col))
            weight = float(getattr(edge, weight_col))
            graph.add_edge(
                source,
                target,
                weight=weight,
                distance=1.0 / max(weight, 1e-12),
            )

        pagerank = nx.pagerank(graph, weight="weight")
        betweenness = nx.betweenness_centrality(
            graph,
            weight="distance",
            normalized=True,
        )
        out_closeness = nx.closeness_centrality(
            graph.reverse(copy=False),
            distance="distance",
        )
        clustering = nx.clustering(graph.to_undirected(), weight="weight")
        participation, within_module_z = _weighted_module_cartography(graph)
        graph_strength = float(graph.size(weight="weight"))
        for node in nodes:
            in_strength = float(graph.in_degree(node, weight="weight"))
            out_strength = float(graph.out_degree(node, weight="weight"))
            total_strength = in_strength + out_strength
            row: dict[str, float | str] = {
                column: value
                for column, value in zip(group_cols, group_values)
            }
            row.update(
                {
                    "node": str(node),
                    "in_degree": float(graph.in_degree(node)),
                    "out_degree": float(graph.out_degree(node)),
                    "in_strength": in_strength,
                    "out_strength": out_strength,
                    "total_strength": total_strength,
                    "in_strength_share": (
                        in_strength / graph_strength
                        if graph_strength > 0
                        else np.nan
                    ),
                    "out_strength_share": (
                        out_strength / graph_strength
                        if graph_strength > 0
                        else np.nan
                    ),
                    "sender_receiver_index": (
                        (out_strength - in_strength) / total_strength
                        if total_strength > 0
                        else np.nan
                    ),
                    "pagerank": float(pagerank.get(node, np.nan)),
                    "betweenness": float(betweenness.get(node, np.nan)),
                    "out_closeness": float(out_closeness.get(node, np.nan)),
                    "clustering": float(clustering.get(node, np.nan)),
                    "module_participation": float(
                        participation.get(str(node), np.nan)
                    ),
                    "within_module_z": float(
                        within_module_z.get(str(node), np.nan)
                    ),
                    "local_reaching_centrality": float(
                        nx.local_reaching_centrality(
                            graph,
                            node,
                            weight="weight",
                        )
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=result_columns)


def _jensen_shannon_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return base-2 Jensen-Shannon distance for two probability vectors."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    midpoint = 0.5 * (first + second)

    def divergence(values: np.ndarray) -> float:
        positive = values > 0
        if not positive.any():
            return 0.0
        return float(
            np.sum(
                values[positive]
                * np.log2(values[positive] / midpoint[positive])
            )
        )

    return float(np.sqrt(0.5 * (divergence(first) + divergence(second))))


def dynamic_target_distribution_metrics(
    distributions: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    time_col: str = "time_s",
    target_col: str = "target",
    weight_col: str = "weight",
    scaffold_nodes: Iterable[str] | None = None,
    agg: str = "sum",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify a perturbed node's target distribution through time.

    This is a source-node broadcast summary, not a whole-network community
    analysis. Participation and entropy describe how evenly nonnegative evoked
    weight is distributed across the target nodes sampled for that source.
    Consecutive Jensen-Shannon distance measures target-profile reconfiguration.
    """

    group_cols = list(group_cols)
    time_columns = [*group_cols, time_col]
    time_result_columns = [
        *time_columns,
        "total_weight",
        "active_targets",
        "broadcast_participation",
        "broadcast_entropy",
        "broadcast_effective_targets",
        "scaffold_weight_fraction",
        "reconfiguration_jsd",
    ]
    summary_columns = [
        *group_cols,
        "target_count",
        "broadcast_participation",
        "broadcast_entropy",
        "broadcast_effective_targets",
        "scaffold_weight_fraction",
        "participation_peak",
        "participation_peak_time",
        "participation_variability",
        "reconfiguration_mean_jsd",
        "reconfiguration_peak_jsd",
    ]
    if distributions.empty:
        return (
            pd.DataFrame(columns=time_result_columns),
            pd.DataFrame(columns=summary_columns),
        )
    needed = [*group_cols, time_col, target_col, weight_col]
    missing = [column for column in needed if column not in distributions.columns]
    if missing:
        raise ValueError(
            f"distributions is missing required columns: {missing}"
        )
    data = distributions[needed].copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
    data[target_col] = data[target_col].astype(str)
    data = data[
        np.isfinite(data[time_col]) & np.isfinite(data[weight_col])
    ].copy()
    if (data[weight_col] < 0).any():
        raise ValueError("dynamic target weights must be nonnegative")
    data = (
        data.groupby(
            [*group_cols, time_col, target_col],
            dropna=False,
            sort=True,
        )[weight_col]
        .agg(agg)
        .reset_index()
    )
    scaffold = {str(node) for node in (scaffold_nodes or [])}
    time_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    grouped = data.groupby(group_cols, dropna=False, sort=True)
    for group_values, frame in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        pivot = (
            frame.pivot_table(
                index=time_col,
                columns=target_col,
                values=weight_col,
                aggfunc=agg,
                fill_value=0.0,
            )
            .sort_index()
            .sort_index(axis=1)
        )
        times = pivot.index.to_numpy(dtype=float)
        weights = pivot.to_numpy(dtype=float)
        total_weight = weights.sum(axis=1)
        has_weight = np.isfinite(total_weight) & (total_weight > 0)
        probabilities = np.divide(
            weights,
            total_weight[:, None],
            out=np.zeros_like(weights),
            where=total_weight[:, None] > 0,
        )
        target_count = int(weights.shape[1])
        participation = 1.0 - np.sum(probabilities**2, axis=1)
        if target_count > 1:
            participation /= 1.0 - 1.0 / target_count
        participation[~has_weight] = np.nan
        entropy = -np.sum(
            np.where(
                probabilities > 0,
                probabilities
                * np.log(np.where(probabilities > 0, probabilities, 1.0)),
                0.0,
            ),
            axis=1,
        )
        normalized_entropy = (
            entropy / np.log(target_count)
            if target_count > 1
            else np.zeros_like(entropy)
        )
        effective_targets = np.exp(entropy)
        normalized_entropy[~has_weight] = np.nan
        effective_targets[~has_weight] = np.nan
        scaffold_indices = [
            index
            for index, node in enumerate(pivot.columns.astype(str))
            if node in scaffold
        ]
        scaffold_fraction = (
            np.divide(
                weights[:, scaffold_indices].sum(axis=1),
                total_weight,
                out=np.zeros_like(total_weight),
                where=total_weight > 0,
            )
            if scaffold_indices
            else np.full(len(times), np.nan, dtype=float)
        )
        scaffold_fraction[~has_weight] = np.nan
        reconfiguration = np.full(len(times), np.nan, dtype=float)
        for index in range(1, len(times)):
            if total_weight[index - 1] > 0 and total_weight[index] > 0:
                reconfiguration[index] = _jensen_shannon_distance(
                    probabilities[index - 1],
                    probabilities[index],
                )
        identity = {
            column: value
            for column, value in zip(group_cols, group_values)
        }
        for index, time_value in enumerate(times):
            time_rows.append(
                {
                    **identity,
                    time_col: float(time_value),
                    "total_weight": float(total_weight[index]),
                    "active_targets": int(np.sum(weights[index] > 0)),
                    "broadcast_participation": float(participation[index]),
                    "broadcast_entropy": float(normalized_entropy[index]),
                    "broadcast_effective_targets": float(
                        effective_targets[index]
                    ),
                    "scaffold_weight_fraction": float(
                        scaffold_fraction[index]
                    ),
                    "reconfiguration_jsd": float(reconfiguration[index]),
                }
            )

        finite_weight = has_weight

        def weighted_mean(values: np.ndarray) -> float:
            valid = finite_weight & np.isfinite(values)
            return (
                float(np.average(values[valid], weights=total_weight[valid]))
                if valid.any() and float(total_weight[valid].sum()) > 0
                else np.nan
            )

        finite_reconfiguration = reconfiguration[
            np.isfinite(reconfiguration)
        ]
        finite_participation = np.flatnonzero(
            np.isfinite(participation) & has_weight
        )
        peak_index = (
            int(
                finite_participation[
                    np.argmax(participation[finite_participation])
                ]
            )
            if len(finite_participation)
            else None
        )
        summary_rows.append(
            {
                **identity,
                "target_count": target_count,
                "broadcast_participation": weighted_mean(participation),
                "broadcast_entropy": weighted_mean(normalized_entropy),
                "broadcast_effective_targets": weighted_mean(
                    effective_targets
                ),
                "scaffold_weight_fraction": weighted_mean(
                    scaffold_fraction
                ),
                "participation_peak": (
                    float(participation[peak_index])
                    if peak_index is not None
                    else np.nan
                ),
                "participation_peak_time": (
                    float(times[peak_index])
                    if peak_index is not None
                    else np.nan
                ),
                "participation_variability": float(
                    np.nanstd(participation[has_weight])
                ) if has_weight.any() else np.nan,
                "reconfiguration_mean_jsd": (
                    float(np.mean(finite_reconfiguration))
                    if len(finite_reconfiguration)
                    else np.nan
                ),
                "reconfiguration_peak_jsd": (
                    float(np.max(finite_reconfiguration))
                    if len(finite_reconfiguration)
                    else np.nan
                ),
            }
        )
    return (
        pd.DataFrame(time_rows, columns=time_result_columns),
        pd.DataFrame(summary_rows, columns=summary_columns),
    )


def _normalized_positive(values: dict[str, float], nodes: list[str]) -> dict[str, float]:
    clean = {node: max(float(values.get(node, 0.0)), 0.0) for node in nodes}
    clean = {node: (0.0 if abs(value) <= 1e-12 else value) for node, value in clean.items()}
    total = float(sum(clean.values()))
    if total <= 0 or not np.isfinite(total):
        return {node: 0.0 for node in nodes}
    return {node: value / total for node, value in clean.items()}


def _metric_frame_for_run(mean: pd.DataFrame, run: dict, response_metric_frame_key: str) -> pd.DataFrame:
    frame = run.get(response_metric_frame_key)
    if frame is None:
        return mean
    frame = pd.DataFrame(frame).copy()
    frame.index = pd.to_numeric(frame.index, errors="coerce")
    frame = frame.loc[np.isfinite(frame.index)].sort_index()
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


def evoked_response_edges_over_time(
    runs: list[dict],
    elec_meta: pd.DataFrame | None = None,
    *,
    response_window: tuple[float, float] = (0.0, 0.35),
    frame_step_ms: float = 12.0,
    top_n_per_stim: int = 18,
    top_n_total: int | None = 90,
    detection_metric: str = "peak_amplitude_uv",
    consensus_only: bool = True,
    significance_column: str | None = None,
    response_metric_frame_key: str = "response_metric_frame",
    metric_label: str = "Mean evoked voltage (uV)",
    elec_col: str = "elec_label",
    baseline_window: tuple[float, float] | None = None,
    exclude_stimulation_contacts: bool = True,
) -> pd.DataFrame:
    """Create a dynamic edge table from evoked-response epochs.

    Rows are directed stimulation-source -> recording-contact edges at sampled
    post-stimulus times. ``weight`` is the nonnegative response magnitude used
    for graph metrics; ``signed_value`` preserves the original metric polarity.
    """

    if not runs:
        return pd.DataFrame()
    available_labels = (
        elec_meta[elec_col].astype(str).tolist()
        if elec_meta is not None and not elec_meta.empty and elec_col in elec_meta.columns
        else []
    )
    selected_edges: list[dict] = []
    for run_i, run in enumerate(runs):
        epochs = run.get("epochs")
        if epochs is None:
            raise ValueError("Each run must include an 'epochs' object")
        stim_pair = run.get("stim_pair") or getattr(epochs, "metadata", {}).get("stim_pair")
        if stim_pair is None:
            raise ValueError("Each run must include 'stim_pair' or epochs.metadata['stim_pair']")
        stim_pair = str(stim_pair)
        stim_node = resolve_electrode_label(stim_pair_to_source_electrode(stim_pair), available_labels)
        stimulation_contacts = {
            electrode_label_identity(
                resolve_electrode_label(contact, available_labels)
            )
            for contact in stim_pair_to_contact_electrodes(stim_pair)
        }
        mean = baseline_center_frame(
            epochs.get_mean_waveform(),
            baseline_window
            if baseline_window is not None
            else getattr(epochs, "baseline", None),
        )
        metric_frame = _metric_frame_for_run(mean, run, response_metric_frame_key)
        times = mean.index.to_numpy(dtype=float)
        response_mask = (times >= response_window[0]) & (times <= response_window[1])
        if not response_mask.any():
            continue

        detections = run.get("detections")
        channels = run.get("channels")
        explicit_significance_filter = False
        if channels is None:
            if isinstance(detections, pd.DataFrame) and not detections.empty and "channel" in detections.columns:
                ranked = detections.copy()
                selected_significance = significance_column or "consensus_ch"
                if significance_column is not None and selected_significance not in ranked.columns:
                    raise ValueError(
                        f"detections must contain significance column {selected_significance!r}"
                    )
                if consensus_only and selected_significance in ranked.columns:
                    values = ranked[selected_significance]
                    significance_mask = (
                        values.fillna(False).astype(bool)
                        if pd.api.types.is_bool_dtype(values)
                        else values.fillna(False).astype(str).str.lower().isin(
                            {"true", "1", "yes", "y"}
                        )
                    )
                    if significance_column is not None or significance_mask.any():
                        ranked = ranked[significance_mask]
                        explicit_significance_filter = significance_column is not None
                if detection_metric in ranked.columns and not ranked.empty:
                    ranked["_rank_metric"] = pd.to_numeric(ranked[detection_metric], errors="coerce").abs()
                    ranked = ranked.drop_duplicates("channel").sort_values("_rank_metric", ascending=False)
                channels = ranked["channel"].astype(str).tolist()
            if not channels and not explicit_significance_filter:
                response_mag = metric_frame.loc[response_mask].abs().max(axis=0).sort_values(ascending=False)
                channels = response_mag.index.astype(str).tolist()

        selected = []
        for channel in channels:
            if channel not in metric_frame.columns:
                continue
            target_node = resolve_electrode_label(str(channel), available_labels)
            if (
                exclude_stimulation_contacts
                and electrode_label_identity(target_node) in stimulation_contacts
            ):
                continue
            peak = float(metric_frame.loc[response_mask, channel].abs().max())
            if not np.isfinite(peak):
                continue
            selected.append(
                {
                    "run_i": run_i,
                    "stim_pair": stim_pair,
                    "source": stim_node,
                    "target": target_node,
                    "channel": str(channel),
                    "peak_magnitude": peak,
                    "_metric_frame": metric_frame,
                    "_times": times,
                }
            )
        selected_edges.extend(sorted(selected, key=lambda row: row["peak_magnitude"], reverse=True)[:top_n_per_stim])

    if top_n_total is not None:
        selected_edges = sorted(selected_edges, key=lambda row: row["peak_magnitude"], reverse=True)[:top_n_total]
    if not selected_edges:
        return pd.DataFrame()

    frame_times = np.arange(response_window[0], response_window[1] + frame_step_ms / 1000.0, frame_step_ms / 1000.0)
    frame_times = frame_times[np.isfinite(frame_times)]
    if frame_times.size == 0:
        frame_times = np.array([response_window[0]], dtype=float)

    rows = []
    for time_s in frame_times:
        for edge in selected_edges:
            idx = int(np.argmin(np.abs(edge["_times"] - time_s)))
            signed_value = float(edge["_metric_frame"].iloc[idx][edge["channel"]])
            if not np.isfinite(signed_value):
                continue
            rows.append(
                {
                    "time_s": float(time_s),
                    "time_ms": float(time_s * 1000.0),
                    "stim_pair": edge["stim_pair"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "channel": edge["channel"],
                    "signed_value": signed_value,
                    "weight": abs(signed_value),
                    "peak_magnitude": edge["peak_magnitude"],
                    "metric_label": metric_label,
                }
            )
    return pd.DataFrame(rows)


def compute_dynamic_graph_metrics(
    edge_time_table: pd.DataFrame,
    *,
    metrics: Iterable[str] = DEFAULT_GRAPH_METRICS,
    time_col: str = "time_s",
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """Compute node-level graph metrics for every sampled response time."""

    metrics = tuple(metrics)
    if edge_time_table.empty:
        return pd.DataFrame(columns=[time_col, "time_ms", "node", "metric", "value"])
    data = edge_time_table.copy()
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce").fillna(0.0)
    data = data[data[weight_col] > min_weight]
    all_nodes = sorted(set(edge_time_table[source_col].astype(str)).union(edge_time_table[target_col].astype(str)))
    rows = []
    for time_value, frame in data.groupby(time_col, sort=True):
        graph = nx.DiGraph()
        graph.add_nodes_from(all_nodes)
        for _, edge in frame.iterrows():
            weight = float(edge[weight_col])
            if weight <= min_weight:
                continue
            graph.add_edge(
                str(edge[source_col]),
                str(edge[target_col]),
                weight=weight,
                distance=1.0 / (weight + 1e-12),
            )

        values: dict[str, dict[str, float]] = {metric: {node: 0.0 for node in all_nodes} for metric in metrics}
        if "in_strength" in metrics:
            values["in_strength"] = dict(graph.in_degree(weight="weight"))
        if "out_strength" in metrics:
            values["out_strength"] = dict(graph.out_degree(weight="weight"))
        if "total_strength" in metrics:
            in_strength = dict(graph.in_degree(weight="weight"))
            out_strength = dict(graph.out_degree(weight="weight"))
            values["total_strength"] = {node: float(in_strength.get(node, 0.0) + out_strength.get(node, 0.0)) for node in all_nodes}
        if "in_degree" in metrics:
            values["in_degree"] = dict(graph.in_degree())
        if "out_degree" in metrics:
            values["out_degree"] = dict(graph.out_degree())
        if "pagerank" in metrics and graph.number_of_edges():
            try:
                values["pagerank"] = nx.pagerank(graph, weight="weight")
            except Exception:
                pass
        if "betweenness" in metrics and graph.number_of_edges():
            try:
                values["betweenness"] = nx.betweenness_centrality(graph, weight="distance", normalized=True)
            except Exception:
                pass
        if ({"hub", "authority"} & set(metrics)) and graph.number_of_edges():
            out_strength = dict(graph.out_degree(weight="weight"))
            in_strength = dict(graph.in_degree(weight="weight"))
            try:
                hub, authority = nx.hits(graph, max_iter=1000, normalized=True)
                if "hub" in metrics:
                    values["hub"] = _normalized_positive(hub, all_nodes)
                if "authority" in metrics:
                    values["authority"] = _normalized_positive(authority, all_nodes)
            except Exception:
                if "hub" in metrics:
                    values["hub"] = _normalized_positive(out_strength, all_nodes)
                if "authority" in metrics:
                    values["authority"] = _normalized_positive(in_strength, all_nodes)
            if "hub" in metrics and max(values["hub"].values(), default=0.0) <= 0:
                values["hub"] = _normalized_positive(out_strength, all_nodes)
            if "authority" in metrics and max(values["authority"].values(), default=0.0) <= 0:
                values["authority"] = _normalized_positive(in_strength, all_nodes)

        time_ms = float(frame["time_ms"].iloc[0]) if "time_ms" in frame.columns and not frame.empty else float(time_value) * 1000.0
        for metric in metrics:
            metric_values = values.get(metric, {})
            for node in all_nodes:
                rows.append(
                    {
                        time_col: float(time_value),
                        "time_ms": time_ms,
                        "node": node,
                        "metric": metric,
                        "value": float(metric_values.get(node, 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def evoked_graph_metric_timecourse(
    runs: list[dict],
    elec_meta: pd.DataFrame | None = None,
    *,
    metrics: Iterable[str] = DEFAULT_GRAPH_METRICS,
    **kwargs,
) -> pd.DataFrame:
    """Build dynamic evoked-response edges and return node graph metrics."""

    edges = evoked_response_edges_over_time(runs, elec_meta=elec_meta, **kwargs)
    return compute_dynamic_graph_metrics(edges, metrics=metrics)


def summarize_graph_metric(
    metric_table: pd.DataFrame,
    *,
    metric: str = "hub",
    summary: str = "max",
    node_col: str = "node",
    value_col: str = "value",
) -> pd.Series:
    """Summarize a node metric over time for static brain maps."""

    data = metric_table.copy()
    if "metric" in data.columns:
        data = data[data["metric"].astype(str) == str(metric)]
    if data.empty:
        return pd.Series(dtype=float)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    grouped = data.groupby(node_col)[value_col]
    if summary == "mean":
        return grouped.mean().sort_values(ascending=False)
    if summary == "last":
        return grouped.last().sort_values(ascending=False)
    if summary == "min":
        return grouped.min().sort_values(ascending=False)
    if summary == "absmax":
        return grouped.apply(lambda values: values.iloc[np.nanargmax(np.abs(values.to_numpy(dtype=float)))] if values.notna().any() else np.nan).sort_values(ascending=False)
    return grouped.max().sort_values(ascending=False)
