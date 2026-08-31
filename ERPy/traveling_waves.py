from __future__ import annotations

from collections.abc import Iterable
import re

import numpy as np
import pandas as pd

from .inference import fdr_bh
from .viz.networks import (
    electrode_label_identity,
    resolve_electrode_label,
    stim_pair_to_contact_electrodes,
)


DEFAULT_SCAFFOLD_PATTERN = (
    r"\bACC\b|anterior[\s_-]*cingulat|cingul(?:ate)?(?:[\s_-]*mid)?[\s_-]*ant\b|"
    r"\bACgG\b|brodmann[\s_-]*area[\s_-]*32|\bSGC\b|subgenual|thalam|caudate|putamen|striat|"
    r"\bSFG\b|frontal|pallid|capsule"
)


def scaffold_region_mask(values: pd.Series, pattern: str = DEFAULT_SCAFFOLD_PATTERN) -> pd.Series:
    """Return rows whose anatomical labels belong to the ACC-thalamic-striatal scaffold."""

    return values.fillna("").astype(str).str.contains(pattern, case=False, regex=True, na=False)


def latency_wavefront_table(
    detections: pd.DataFrame,
    elec_meta: pd.DataFrame,
    *,
    latency_col: str = "n1_latency_ms",
    metric_col: str = "peak_amplitude_uv",
    channel_col: str = "channel",
    consensus_col: str = "consensus_ch",
    consensus_only: bool = True,
    patient_col: str = "patient_id",
    session_col: str = "session_id",
    elec_col: str = "elec_label",
    region_col: str = "anat_label",
    coord_cols: tuple[str, str, str] = ("mni_x", "mni_y", "mni_z"),
    exclude_stimulation_contacts: bool = True,
) -> pd.DataFrame:
    """Merge response latencies with recording-contact anatomy and MNI coordinates.

    The returned table is suitable for wavefront plots and
    :func:`estimate_latency_gradient`.
    """

    if detections.empty:
        return pd.DataFrame()
    if elec_meta.empty:
        raise ValueError("elec_meta is empty; MNI coordinates are required for wavefront analysis")
    missing_det = [col for col in (channel_col, latency_col) if col not in detections.columns]
    if missing_det:
        raise ValueError(f"detections is missing required columns: {missing_det}")
    missing_elec = [col for col in (elec_col, *coord_cols) if col not in elec_meta.columns]
    if missing_elec:
        raise ValueError(f"elec_meta is missing required columns: {missing_elec}")

    data = detections.copy()
    if consensus_only and consensus_col in data.columns:
        data = data[_boolean_mask(data[consensus_col])]
    data = _collapse_detector_rows(
        data,
        channel_col=channel_col,
        latency_col=latency_col,
        metric_col=metric_col,
    )
    data[latency_col] = pd.to_numeric(data[latency_col], errors="coerce")
    data = data[np.isfinite(data[latency_col])]
    if metric_col in data.columns:
        data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    if data.empty:
        return pd.DataFrame()

    meta = elec_meta.copy()
    meta[elec_col] = meta[elec_col].astype(str)
    for col in coord_cols:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.dropna(subset=list(coord_cols))
    if meta.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    patient_session_cols = [col for col in (patient_col, session_col) if col in data.columns and col in meta.columns]
    if patient_session_cols:
        grouped = data.groupby(patient_session_cols, dropna=False)
    else:
        grouped = [((), data)]
    for group_values, frame in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        sub_meta = meta
        for col, value in zip(patient_session_cols, group_values):
            sub_meta = sub_meta[sub_meta[col].astype(str) == str(value)]
        if sub_meta.empty:
            sub_meta = meta
        available = sub_meta[elec_col].astype(str).tolist()
        meta_lookup = sub_meta.drop_duplicates(elec_col).set_index(elec_col)
        for _, row in frame.iterrows():
            resolved = resolve_electrode_label(str(row[channel_col]), available)
            if exclude_stimulation_contacts and "stim_pair" in row.index:
                stimulation_contacts = {
                    electrode_label_identity(
                        resolve_electrode_label(contact, available)
                    )
                    for contact in stim_pair_to_contact_electrodes(
                        str(row["stim_pair"])
                    )
                }
                if electrode_label_identity(resolved) in stimulation_contacts:
                    continue
            if resolved not in meta_lookup.index:
                continue
            meta_row = meta_lookup.loc[resolved]
            out = row.to_dict()
            out["record_elec"] = resolved
            out["record_region"] = meta_row.get(region_col, "") if region_col in meta_lookup.columns else ""
            out["latency_ms"] = float(row[latency_col])
            out["metric_abs"] = abs(float(row[metric_col])) if metric_col in row and pd.notna(row[metric_col]) else np.nan
            for axis, col in zip(("x", "y", "z"), coord_cols):
                out[f"mni_{axis}"] = float(meta_row[col])
            out["in_scaffold"] = bool(scaffold_region_mask(pd.Series([out["record_region"]])).iloc[0])
            rows.append(out)
    return pd.DataFrame(rows)


def _boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def _collapse_detector_rows(
    detections: pd.DataFrame,
    *,
    channel_col: str,
    latency_col: str,
    metric_col: str,
) -> pd.DataFrame:
    """Return one deterministic row per physical acquisition-channel response."""

    if detections.empty:
        return detections.copy()
    identity_cols = [
        col
        for col in (
            "patient_id",
            "session_id",
            "stim_pair",
            "acquisition_id",
            "raw_file",
            "stim_start",
        )
        if col in detections.columns
    ]
    keys = [*identity_cols, channel_col]
    data = detections.copy()
    if "method" in data.columns:
        data = data.sort_values("method", kind="stable")
    first = data.drop_duplicates(keys, keep="first").copy()

    grouped = data.groupby(keys, dropna=False, sort=False)
    latency = grouped[latency_col].median().rename("_latency")
    first = first.merge(latency.reset_index(), on=keys, how="left")
    first[latency_col] = first.pop("_latency")
    if metric_col in data.columns:
        metric = grouped[metric_col].median().rename("_metric")
        first = first.merge(metric.reset_index(), on=keys, how="left")
        first[metric_col] = first.pop("_metric")
    first["n_detector_rows"] = first.set_index(keys).index.map(grouped.size())
    if "method" in data.columns:
        methods = grouped["method"].agg(
            lambda values: ";".join(
                sorted(set(values.dropna().astype(str)))
            )
        )
        first["detector_methods"] = first.set_index(keys).index.map(methods)
    return first.reset_index(drop=True)


def estimate_latency_gradient(
    wavefront: pd.DataFrame,
    *,
    group_cols: Iterable[str] | None = None,
    latency_col: str = "latency_ms",
    coord_cols: tuple[str, str, str] = ("mni_x", "mni_y", "mni_z"),
    weight_col: str | None = "metric_abs",
    min_points: int = 4,
) -> pd.DataFrame:
    """Estimate latency gradients and apparent propagation speeds.

    Latency is modeled as ``latency_ms ~ x + y + z`` within each group. The
    gradient norm has units of milliseconds per millimeter; its reciprocal is
    reported as ``apparent_speed_m_per_s`` because 1 mm/ms equals 1 m/s.
    """

    if wavefront.empty:
        return pd.DataFrame()
    group_cols = list(group_cols or [])
    needed = [latency_col, *coord_cols, *group_cols]
    missing = [col for col in needed if col not in wavefront.columns]
    if missing:
        raise ValueError(f"wavefront is missing required columns: {missing}")

    data = wavefront.copy()
    for col in (latency_col, *coord_cols):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[latency_col, *coord_cols])
    if data.empty:
        return pd.DataFrame()

    groups = data.groupby(group_cols, dropna=False) if group_cols else [((), data)]
    rows: list[dict] = []
    for group_values, frame in groups:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {col: value for col, value in zip(group_cols, group_values)}
        row["n_contacts"] = int(len(frame))
        if len(frame) < min_points:
            row.update(_empty_gradient("too_few_contacts"))
            rows.append(row)
            continue
        coords = frame[list(coord_cols)].to_numpy(dtype=float)
        latency = frame[latency_col].to_numpy(dtype=float)
        coords_centered = coords - np.nanmean(coords, axis=0, keepdims=True)
        latency_centered = latency - float(np.nanmean(latency))
        design = np.column_stack([coords_centered, np.ones(len(coords_centered))])
        weights = _regression_weights(frame, weight_col)
        try:
            if weights is not None:
                sw = np.sqrt(weights)[:, None]
                coef, *_ = np.linalg.lstsq(design * sw, latency_centered * sw.ravel(), rcond=None)
            else:
                coef, *_ = np.linalg.lstsq(design, latency_centered, rcond=None)
            gradient = coef[:3]
            fitted = design @ coef
            residual = latency_centered - fitted
            ss_res = float(np.nansum(residual**2))
            ss_tot = float(np.nansum((latency_centered - np.nanmean(latency_centered)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            grad_norm = float(np.linalg.norm(gradient))
            if grad_norm > 0 and np.isfinite(grad_norm):
                unit = gradient / grad_norm
                speed = float(1.0 / grad_norm)
            else:
                unit = np.full(3, np.nan)
                speed = np.nan
            row.update(
                {
                    "gradient_x_ms_per_mm": float(gradient[0]),
                    "gradient_y_ms_per_mm": float(gradient[1]),
                    "gradient_z_ms_per_mm": float(gradient[2]),
                    "gradient_norm_ms_per_mm": grad_norm,
                    "apparent_speed_m_per_s": speed,
                    "propagation_unit_x": float(unit[0]),
                    "propagation_unit_y": float(unit[1]),
                    "propagation_unit_z": float(unit[2]),
                    "latency_r2": float(r2),
                    "status": "ok",
                }
            )
        except np.linalg.LinAlgError:
            row.update(_empty_gradient("singular_fit"))
        rows.append(row)
    return pd.DataFrame(rows)


def latency_gradient_permutation_test(
    wavefront: pd.DataFrame,
    *,
    group_cols: Iterable[str] | None = None,
    latency_col: str = "latency_ms",
    coord_cols: tuple[str, str, str] = ("mni_x", "mni_y", "mni_z"),
    weight_col: str | None = "metric_abs",
    min_points: int = 4,
    n_permutations: int = 2_000,
    fdr_alpha: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> pd.DataFrame:
    """Test spatial latency gradients against a contact-permutation null.

    Latency and optional response weight are permuted together across fixed MNI
    coordinates. This preserves the latency-amplitude relationship while
    breaking spatial organization. The returned ``gradient_p_value`` tests
    whether the observed weighted planar-model R-squared exceeds the null, and
    ``gradient_q_value`` applies Benjamini-Hochberg correction across groups.
    """

    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be at least 1")
    if wavefront.empty:
        return pd.DataFrame()
    group_cols = list(group_cols or [])
    needed = [latency_col, *coord_cols, *group_cols]
    missing = [col for col in needed if col not in wavefront.columns]
    if missing:
        raise ValueError(f"wavefront is missing required columns: {missing}")

    data = wavefront.copy()
    for col in (latency_col, *coord_cols):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[latency_col, *coord_cols])
    groups = data.groupby(group_cols, dropna=False) if group_cols else [((), data)]
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    rows: list[dict] = []
    for group_values, frame in groups:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {col: value for col, value in zip(group_cols, group_values)}
        row["n_contacts"] = int(len(frame))
        if len(frame) < int(min_points):
            row.update(_empty_gradient("too_few_contacts"))
            row.update(
                {
                    "gradient_p_value": np.nan,
                    "gradient_null_r2_median": np.nan,
                    "gradient_null_r2_p95": np.nan,
                    "n_permutations": int(n_permutations),
                }
            )
            rows.append(row)
            continue

        coords = frame[list(coord_cols)].to_numpy(dtype=float)
        latency = frame[latency_col].to_numpy(dtype=float)
        weights = _regression_weights(frame, weight_col)
        observed = _fit_latency_gradient(coords, latency, weights)
        row.update(observed)
        if observed["status"] != "ok":
            row.update(
                {
                    "gradient_p_value": np.nan,
                    "gradient_null_r2_median": np.nan,
                    "gradient_null_r2_p95": np.nan,
                    "n_permutations": int(n_permutations),
                }
            )
            rows.append(row)
            continue

        if weights is None:
            null_r2 = _unweighted_permutation_r2(
                coords,
                latency,
                n_permutations=int(n_permutations),
                rng=rng,
            )
        else:
            null_r2 = np.full(
                int(n_permutations),
                np.nan,
                dtype=float,
            )
            for permutation_index in range(int(n_permutations)):
                order = rng.permutation(len(frame))
                fitted = _fit_latency_gradient(
                    coords,
                    latency[order],
                    weights[order],
                )
                null_r2[permutation_index] = fitted["latency_r2"]
        finite_null = null_r2[np.isfinite(null_r2)]
        if len(finite_null):
            p_value = (
                1.0
                + float(
                    np.sum(
                        finite_null
                        >= float(observed["latency_r2"]) - 1e-15
                    )
                )
            ) / (len(finite_null) + 1.0)
            null_median = float(np.nanmedian(finite_null))
            null_p95 = float(np.nanquantile(finite_null, 0.95))
        else:
            p_value = null_median = null_p95 = np.nan
        row.update(
            {
                "gradient_p_value": p_value,
                "gradient_null_r2_median": null_median,
                "gradient_null_r2_p95": null_p95,
                "n_permutations": int(n_permutations),
            }
        )
        rows.append(row)

    output = pd.DataFrame(rows)
    adjusted, rejected = fdr_bh(
        output.get(
            "gradient_p_value",
            pd.Series(np.nan, index=output.index),
        ),
        alpha=float(fdr_alpha),
    )
    output["gradient_q_value"] = adjusted
    output["gradient_fdr_significant"] = rejected
    output["gradient_fdr_alpha"] = float(fdr_alpha)
    return output


def latency_distance_permutation_test(
    wavefront: pd.DataFrame,
    *,
    group_cols: Iterable[str] | None = None,
    latency_col: str = "latency_ms",
    distance_col: str = "distance_from_stim_mm",
    weight_col: str | None = "metric_abs",
    min_points: int = 4,
    n_permutations: int = 2_000,
    fdr_alpha: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> pd.DataFrame:
    """Test whether response latency is ordered by distance from stimulation.

    Latency is modeled as ``latency_ms ~ distance_from_stim_mm``. A positive
    slope indicates later responses at more distant contacts. The permutation
    null shuffles latency, and optional response weights with it, over the fixed
    contact distances. Euclidean distance is a spatial proxy rather than tract
    length, so the returned speed is explicitly an apparent radial speed.
    """

    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be at least 1")
    if wavefront.empty:
        return pd.DataFrame()
    group_cols = list(group_cols or [])
    needed = [latency_col, distance_col, *group_cols]
    missing = [col for col in needed if col not in wavefront.columns]
    if missing:
        raise ValueError(f"wavefront is missing required columns: {missing}")

    data = wavefront.copy()
    for col in (latency_col, distance_col):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[latency_col, distance_col])
    groups = data.groupby(group_cols, dropna=False) if group_cols else [((), data)]
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    rows: list[dict] = []
    for group_values, frame in groups:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {col: value for col, value in zip(group_cols, group_values)}
        row["n_contacts"] = int(len(frame))
        if len(frame) < int(min_points):
            row.update(_empty_distance_fit("too_few_contacts"))
            row.update(
                {
                    "distance_p_value": np.nan,
                    "distance_null_r2_median": np.nan,
                    "distance_null_r2_p95": np.nan,
                    "n_permutations": int(n_permutations),
                }
            )
            rows.append(row)
            continue

        distance = frame[distance_col].to_numpy(dtype=float)
        latency = frame[latency_col].to_numpy(dtype=float)
        weights = _regression_weights(frame, weight_col)
        observed = _fit_latency_distance(distance, latency, weights)
        row.update(observed)
        if observed["status"] != "ok":
            row.update(
                {
                    "distance_p_value": np.nan,
                    "distance_null_r2_median": np.nan,
                    "distance_null_r2_p95": np.nan,
                    "n_permutations": int(n_permutations),
                }
            )
            rows.append(row)
            continue

        if weights is None:
            null_r2 = _unweighted_distance_permutation_r2(
                distance,
                latency,
                n_permutations=int(n_permutations),
                rng=rng,
            )
        else:
            null_r2 = np.full(int(n_permutations), np.nan, dtype=float)
            for permutation_index in range(int(n_permutations)):
                order = rng.permutation(len(frame))
                fitted = _fit_latency_distance(
                    distance,
                    latency[order],
                    weights[order],
                )
                null_r2[permutation_index] = fitted["distance_r2"]
        finite_null = null_r2[np.isfinite(null_r2)]
        if len(finite_null):
            p_value = (
                1.0
                + float(
                    np.sum(
                        finite_null
                        >= float(observed["distance_r2"]) - 1e-15
                    )
                )
            ) / (len(finite_null) + 1.0)
            null_median = float(np.nanmedian(finite_null))
            null_p95 = float(np.nanquantile(finite_null, 0.95))
        else:
            p_value = null_median = null_p95 = np.nan
        row.update(
            {
                "distance_p_value": p_value,
                "distance_null_r2_median": null_median,
                "distance_null_r2_p95": null_p95,
                "n_permutations": int(n_permutations),
            }
        )
        rows.append(row)

    output = pd.DataFrame(rows)
    adjusted, rejected = fdr_bh(
        output.get(
            "distance_p_value",
            pd.Series(np.nan, index=output.index),
        ),
        alpha=float(fdr_alpha),
    )
    output["distance_q_value"] = adjusted
    output["distance_fdr_significant"] = rejected
    output["distance_fdr_alpha"] = float(fdr_alpha)
    return output


def _unweighted_permutation_r2(
    coords: np.ndarray,
    latency: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    batch_size: int = 512,
) -> np.ndarray:
    """Vectorize fixed-design permutation R-squared calculations."""

    coords_centered = coords - np.nanmean(
        coords,
        axis=0,
        keepdims=True,
    )
    latency_centered = latency - float(np.nanmean(latency))
    design = np.column_stack(
        [coords_centered, np.ones(len(coords_centered))]
    )
    try:
        design_pinv = np.linalg.pinv(design)
    except np.linalg.LinAlgError:
        return np.full(int(n_permutations), np.nan, dtype=float)
    ss_tot = float(np.nansum(latency_centered**2))
    if not np.isfinite(ss_tot) or ss_tot <= 0:
        return np.full(int(n_permutations), np.nan, dtype=float)

    output = np.full(int(n_permutations), np.nan, dtype=float)
    for start in range(0, int(n_permutations), int(batch_size)):
        stop = min(start + int(batch_size), int(n_permutations))
        size = stop - start
        orders = np.argsort(
            rng.random((size, len(latency_centered))),
            axis=1,
        )
        permuted = latency_centered[orders]
        coefficients = permuted @ design_pinv.T
        fitted = coefficients @ design.T
        residual = permuted - fitted
        ss_res = np.sum(residual**2, axis=1)
        output[start:stop] = 1.0 - ss_res / ss_tot
    return output


def _fit_latency_gradient(
    coords: np.ndarray,
    latency: np.ndarray,
    weights: np.ndarray | None,
) -> dict[str, float | str]:
    coords_centered = coords - np.nanmean(coords, axis=0, keepdims=True)
    latency_centered = latency - float(np.nanmean(latency))
    design = np.column_stack([coords_centered, np.ones(len(coords_centered))])
    try:
        if weights is not None:
            sw = np.sqrt(weights)[:, None]
            coef, *_ = np.linalg.lstsq(
                design * sw,
                latency_centered * sw.ravel(),
                rcond=None,
            )
        else:
            coef, *_ = np.linalg.lstsq(
                design,
                latency_centered,
                rcond=None,
            )
    except np.linalg.LinAlgError:
        return _empty_gradient("singular_fit")

    gradient = coef[:3]
    fitted = design @ coef
    residual = latency_centered - fitted
    ss_res = float(np.nansum(residual**2))
    ss_tot = float(
        np.nansum(
            (latency_centered - np.nanmean(latency_centered)) ** 2
        )
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    grad_norm = float(np.linalg.norm(gradient))
    if grad_norm > 0 and np.isfinite(grad_norm):
        unit = gradient / grad_norm
        speed = float(1.0 / grad_norm)
    else:
        unit = np.full(3, np.nan)
        speed = np.nan
    return {
        "gradient_x_ms_per_mm": float(gradient[0]),
        "gradient_y_ms_per_mm": float(gradient[1]),
        "gradient_z_ms_per_mm": float(gradient[2]),
        "gradient_norm_ms_per_mm": grad_norm,
        "apparent_speed_m_per_s": speed,
        "propagation_unit_x": float(unit[0]),
        "propagation_unit_y": float(unit[1]),
        "propagation_unit_z": float(unit[2]),
        "latency_r2": float(r2),
        "status": "ok",
    }


def _unweighted_distance_permutation_r2(
    distance: np.ndarray,
    latency: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    batch_size: int = 512,
) -> np.ndarray:
    distance_centered = distance - float(np.nanmean(distance))
    latency_centered = latency - float(np.nanmean(latency))
    design = np.column_stack(
        [distance_centered, np.ones(len(distance_centered))]
    )
    try:
        design_pinv = np.linalg.pinv(design)
    except np.linalg.LinAlgError:
        return np.full(int(n_permutations), np.nan, dtype=float)
    ss_tot = float(np.nansum(latency_centered**2))
    if not np.isfinite(ss_tot) or ss_tot <= 0:
        return np.full(int(n_permutations), np.nan, dtype=float)

    output = np.full(int(n_permutations), np.nan, dtype=float)
    for start in range(0, int(n_permutations), int(batch_size)):
        stop = min(start + int(batch_size), int(n_permutations))
        size = stop - start
        orders = np.argsort(
            rng.random((size, len(latency_centered))),
            axis=1,
        )
        permuted = latency_centered[orders]
        coefficients = permuted @ design_pinv.T
        fitted = coefficients @ design.T
        residual = permuted - fitted
        output[start:stop] = 1.0 - np.sum(residual**2, axis=1) / ss_tot
    return output


def _fit_latency_distance(
    distance: np.ndarray,
    latency: np.ndarray,
    weights: np.ndarray | None,
) -> dict[str, float | bool | str]:
    distance_centered = distance - float(np.nanmean(distance))
    if not np.isfinite(distance_centered).all() or np.ptp(distance_centered) <= 0:
        return _empty_distance_fit("constant_distance")
    latency_centered = latency - float(np.nanmean(latency))
    design = np.column_stack(
        [distance_centered, np.ones(len(distance_centered))]
    )
    try:
        if weights is not None:
            sw = np.sqrt(weights)[:, None]
            coefficients, *_ = np.linalg.lstsq(
                design * sw,
                latency_centered * sw.ravel(),
                rcond=None,
            )
        else:
            coefficients, *_ = np.linalg.lstsq(
                design,
                latency_centered,
                rcond=None,
            )
    except np.linalg.LinAlgError:
        return _empty_distance_fit("singular_fit")

    slope = float(coefficients[0])
    fitted = design @ coefficients
    residual = latency_centered - fitted
    ss_res = float(np.nansum(residual**2))
    ss_tot = float(np.nansum(latency_centered**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    speed = (
        float(1.0 / abs(slope))
        if np.isfinite(slope) and not np.isclose(slope, 0.0)
        else np.nan
    )
    return {
        "distance_slope_ms_per_mm": slope,
        "apparent_radial_speed_m_per_s": speed,
        "outward_latency_gradient": bool(slope > 0),
        "distance_r2": float(r2),
        "status": "ok",
    }


def _empty_distance_fit(status: str) -> dict[str, float | bool | str]:
    return {
        "distance_slope_ms_per_mm": np.nan,
        "apparent_radial_speed_m_per_s": np.nan,
        "outward_latency_gradient": False,
        "distance_r2": np.nan,
        "status": status,
    }


def _regression_weights(frame: pd.DataFrame, weight_col: str | None) -> np.ndarray | None:
    if not weight_col or weight_col not in frame.columns:
        return None
    weights = pd.to_numeric(frame[weight_col], errors="coerce").to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights), np.abs(weights), 0.0)
    if np.nanmax(weights) <= 0:
        return None
    weights = weights / np.nanmax(weights)
    weights = np.clip(weights, 1e-3, None)
    return weights


def _empty_gradient(status: str) -> dict[str, float | str]:
    return {
        "gradient_x_ms_per_mm": np.nan,
        "gradient_y_ms_per_mm": np.nan,
        "gradient_z_ms_per_mm": np.nan,
        "gradient_norm_ms_per_mm": np.nan,
        "apparent_speed_m_per_s": np.nan,
        "propagation_unit_x": np.nan,
        "propagation_unit_y": np.nan,
        "propagation_unit_z": np.nan,
        "latency_r2": np.nan,
        "status": status,
    }
