from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .baseline import baseline_center_array


DEFAULT_HARD_ARTIFACT_REASONS = frozenset(
    {
        "saturation_or_clipping",
        "plateau_clipping",
        "rail_clipping",
        "absolute_peak",
        "absolute_ptp",
        "extreme_amplitude_outlier",
    }
)


@dataclass
class BadChannelReport:
    """Auditable summary of continuous-recording channel quality control.

    ``table`` is indexed by channel and retains the measured features, Boolean
    decision, and reason string used by ``detect_bad_channels``.
    """

    table: pd.DataFrame

    @property
    def bad_channels(self) -> list[str]:
        """Return channel labels marked bad in the report table."""

        if self.table.empty or "bad_channel" not in self.table.columns:
            return []
        return self.table.index[self.table["bad_channel"]].astype(str).tolist()

    def to_csv(self, path) -> None:
        """Write the full channel-QC table to a CSV file."""

        self.table.to_csv(path)


@dataclass
class ArtifactResponseReport:
    """Trial-by-channel response-artifact quality-control report.

    ``table`` contains one row per evaluated trial and channel, including the
    measured artifact features, the ``bad_response`` decision, and its reason.
    Use ``by_channel`` to summarize burden without discarding the
    trial-level audit trail.
    """

    table: pd.DataFrame

    @property
    def bad_epochs(self) -> list[int]:
        """Return trial identifiers containing at least one bad response."""

        if self.table.empty or "bad_response" not in self.table.columns:
            return []
        return sorted(self.table.loc[self.table["bad_response"], "epoch"].astype(int).unique().tolist())

    @property
    def bad_channels(self) -> list[str]:
        """Return channels whose bad-response fraction exceeds 25 percent."""

        if self.table.empty or "bad_response" not in self.table.columns:
            return []
        summary = self.by_channel()
        return sorted(
            summary.loc[summary["bad_response_fraction"] > 0.25, "channel"].astype(str).tolist()
        )

    def to_csv(self, path) -> None:
        """Write the trial-by-channel artifact report to a CSV file."""

        self.table.to_csv(path, index=False)

    def by_channel(
        self,
        group_cols: Iterable[str] | None = None,
        hard_artifact_reasons: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Summarize response-artifact burden by channel.

        This is the table used by cohort notebooks before building ranked
        response edges. It keeps saturated/rail-clipped responses auditable
        without requiring downstream code to reimplement the trial-level QC
        rules.
        """

        return response_artifact_summary(
            self,
            group_cols=group_cols,
            hard_artifact_reasons=hard_artifact_reasons,
        )


def _robust_z(values: pd.Series) -> pd.Series:
    med = values.median(skipna=True)
    mad = (values - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad == 0:
        std = values.std(skipna=True)
        denom = std if std and np.isfinite(std) else 1.0
    else:
        denom = 1.4826 * mad
    return (values - med) / denom


def detect_bad_channels(
    df: pd.DataFrame,
    zscore_threshold: float = 5.0,
    flat_std: float = 1e-12,
    max_nan_fraction: float = 0.05,
    min_correlation: float | None = -0.2,
) -> BadChannelReport:
    """Detect flat, noisy, NaN-heavy, and anticorrelated channels.

    The rule is deliberately transparent: each feature receives a robust z-score
    and a channel is bad when any hard QC flag fires. The report table records
    the reason string so clinical users can audit the decision.
    """

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return BadChannelReport(pd.DataFrame(columns=["bad_channel", "reason"]))

    std = numeric.std(axis=0, skipna=True)
    ptp = numeric.max(axis=0, skipna=True) - numeric.min(axis=0, skipna=True)
    nan_fraction = numeric.isna().mean(axis=0)
    median_abs = numeric.abs().median(axis=0, skipna=True)
    line = pd.DataFrame(
        {
            "std": std,
            "ptp": ptp,
            "nan_fraction": nan_fraction,
            "median_abs": median_abs,
        }
    )
    line["std_z"] = _robust_z(line["std"]).abs()
    line["ptp_z"] = _robust_z(line["ptp"]).abs()
    line["median_abs_z"] = _robust_z(line["median_abs"]).abs()
    reasons: list[list[str]] = [[] for _ in range(len(line))]

    for i, (_, row) in enumerate(line.iterrows()):
        if row["nan_fraction"] > max_nan_fraction:
            reasons[i].append("nan_fraction")
        if row["std"] <= flat_std or row["ptp"] <= flat_std:
            reasons[i].append("flat")
        if row["std_z"] >= zscore_threshold or row["ptp_z"] >= zscore_threshold:
            reasons[i].append("amplitude_outlier")
        if row["median_abs_z"] >= zscore_threshold:
            reasons[i].append("offset_outlier")

    if min_correlation is not None and numeric.shape[1] > 2 and len(numeric) > 5:
        corr = numeric.corr().replace([np.inf, -np.inf], np.nan)
        mean_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).mean(axis=1)
        line["mean_correlation"] = mean_corr
        for i, ch in enumerate(line.index):
            val = mean_corr.loc[ch]
            if pd.notna(val) and val < min_correlation:
                reasons[i].append("anticorrelated")
    else:
        line["mean_correlation"] = np.nan

    line["reason"] = [";".join(r) for r in reasons]
    line["bad_channel"] = line["reason"].astype(bool)
    return BadChannelReport(line.sort_values(["bad_channel", "std_z"], ascending=[False, False]))


def reject_bad_channels(
    df: pd.DataFrame,
    bad_channels: Iterable[str] | str | None = None,
    method: str = "drop",
    detection_params: dict | None = None,
    exclude: Iterable[str] | None = None,
    return_report: bool = False,
):
    """Reject or repair bad channels.

    Parameters
    ----------
    bad_channels:
        Explicit channel list, ``"auto"`` for data-driven detection, or ``None``.
    method:
        ``"drop"`` removes bad columns, ``"nan"`` keeps them as NaN, and
        ``"interpolate"`` replaces them with the median good-channel signal.
    """

    exclude = set(exclude or [])
    report = None
    if isinstance(bad_channels, str) and bad_channels.lower() == "auto":
        report = detect_bad_channels(df, **(detection_params or {}))
        bad = set(report.bad_channels)
    elif bad_channels is None:
        bad = set()
    else:
        bad = {str(ch) for ch in bad_channels}
    bad -= exclude
    bad &= set(map(str, df.columns))

    out = df.copy()
    if method == "drop":
        out = out.drop(columns=sorted(bad), errors="ignore")
    elif method == "nan":
        for ch in bad:
            out[ch] = np.nan
    elif method == "interpolate":
        good = [c for c in out.select_dtypes(include=[np.number]).columns if c not in bad]
        if not good:
            raise ValueError("Cannot interpolate bad channels without at least one good numeric channel")
        replacement = out[good].median(axis=1)
        for ch in bad:
            out[ch] = replacement
    else:
        raise ValueError("method must be one of: 'drop', 'nan', 'interpolate'")

    out.attrs.update(df.attrs)
    if report is None:
        report = BadChannelReport(
            pd.DataFrame(
                {"bad_channel": [c in bad for c in df.columns], "reason": ["manual" if c in bad else "" for c in df.columns]},
                index=df.columns,
            )
        )
    out.attrs["bad_channels"] = sorted(bad)
    if return_report:
        return out, report
    return out


def detect_artifactual_responses(
    epochs,
    response_window: tuple[float, float] | None = None,
    baseline_window: tuple[float, float] | None = None,
    zscore_threshold: float = 6.0,
    max_nan_fraction: float = 0.02,
    saturation_fraction: float = 0.02,
    plateau_fraction: float = 0.015,
    plateau_min_run_s: float = 0.003,
    plateau_epsilon_uv: float | None = None,
    rail_fraction: float = 0.02,
    rail_epsilon_fraction: float = 0.002,
    rail_epsilon_uv: float | None = None,
    rail_limits_uv: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    absolute_peak_uv: float | None = None,
    absolute_ptp_uv: float | None = None,
    extreme_zscore_threshold: float | None = 12.0,
    ringing_zscore_threshold: float = 6.0,
    late_high_frequency_zscore_threshold: float | None = None,
) -> ArtifactResponseReport:
    """Flag aberrant trial-by-channel responses after epoching.

    This catches problems that are not persistent bad channels: amplifier
    saturations on one trial, plateau or known-rail clipping, extreme
    trial-relative amplitude outliers, late-response roughness-ratio outliers,
    isolated movement/noise bursts, and NaN-heavy extracted windows. The
    roughness ratio compares mean absolute sample gradients from 50–500 ms and
    0–50 ms (bounded by the available epoch); the serialized
    ``late_high_frequency_ratio`` name is retained for compatibility and does
    not denote spectral energy. Local waveform
    minima and maxima are retained as an ``extreme_dwell_fraction`` diagnostic,
    but are not treated as amplifier rails unless explicit ``rail_limits_uv``
    are supplied.
    """

    arr, times, channels = epochs.as_array()
    if baseline_window is None:
        baseline_window = getattr(epochs, "baseline", None)
    referenced = baseline_center_array(arr, times, baseline_window)
    if response_window is None:
        response_window = (max(0.005, epochs.tmin), min(epochs.tmax, 0.5))
    mask = (times >= response_window[0]) & (times <= response_window[1])
    if not mask.any():
        mask = np.ones_like(times, dtype=bool)
    response_duration = float(np.nanmax(times[mask]) - np.nanmin(times[mask])) if mask.any() else 0.0
    min_plateau_run = max(3, int(np.ceil(float(plateau_min_run_s) * float(epochs.sfreq))))
    rows = []
    for ti in range(arr.shape[0]):
        for ci, ch in enumerate(channels):
            x = referenced[ti, mask, ci]
            raw_x = arr[ti, mask, ci]
            full = arr[ti, :, ci]
            finite = np.isfinite(x)
            nan_fraction = 1.0 - float(finite.mean()) if len(finite) else 1.0
            if finite.any():
                xf = x[finite]
                peak_abs = float(np.nanmax(np.abs(xf)))
                ptp = float(np.nanmax(xf) - np.nanmin(xf))
                grad = float(np.nanmax(np.abs(np.gradient(xf))))
                raw_finite = raw_x[np.isfinite(raw_x)]
                rounded = np.round(raw_finite, decimals=9)
                _, counts = np.unique(rounded, return_counts=True)
                sat_frac = (
                    float(counts.max() / len(raw_finite))
                    if len(raw_finite)
                    else 1.0
                )
                plateau_run = _longest_near_constant_run(
                    raw_finite,
                    epsilon_uv=plateau_epsilon_uv,
                )
                plateau_frac = (
                    float(plateau_run / len(raw_finite))
                    if len(raw_finite)
                    else 1.0
                )
                extreme_dwell_frac = _rail_clip_fraction(
                    raw_finite,
                    epsilon_fraction=rail_epsilon_fraction,
                    epsilon_uv=rail_epsilon_uv,
                )
                rail_limits = _channel_rail_limits(rail_limits_uv, str(ch))
                rail_frac = (
                    _rail_clip_fraction(
                        raw_finite,
                        epsilon_fraction=rail_epsilon_fraction,
                        epsilon_uv=rail_epsilon_uv,
                        rail_limits=rail_limits,
                    )
                    if rail_limits is not None
                    else extreme_dwell_frac
                )
                ringing = _late_high_frequency_ratio(full, times, epochs.sfreq)
            else:
                peak_abs = ptp = grad = ringing = np.nan
                sat_frac = plateau_frac = extreme_dwell_frac = rail_frac = 1.0
                plateau_run = 0
                rail_limits = _channel_rail_limits(rail_limits_uv, str(ch))
            rows.append(
                {
                    "epoch": ti,
                    "channel": ch,
                    "peak_abs": peak_abs,
                    "ptp": ptp,
                    "max_gradient": grad,
                    "nan_fraction": nan_fraction,
                    "saturation_fraction": sat_frac,
                    "plateau_fraction": plateau_frac,
                    "plateau_run_samples": int(plateau_run),
                    "plateau_run_s": float(plateau_run / float(epochs.sfreq)) if np.isfinite(epochs.sfreq) and epochs.sfreq else np.nan,
                    "extreme_dwell_fraction": extreme_dwell_frac,
                    "rail_fraction": rail_frac,
                    "rail_limits_known": rail_limits is not None,
                    "ringing_ratio": ringing,
                    "late_high_frequency_ratio": ringing,
                    "response_duration_s": response_duration,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return ArtifactResponseReport(table)
    for col in ("peak_abs", "ptp", "max_gradient"):
        table[f"{col}_z"] = table.groupby("channel")[col].transform(_robust_z)
    table["ringing_ratio_z"] = table.groupby("channel")["ringing_ratio"].transform(_robust_z)
    table["late_high_frequency_ratio_z"] = table["ringing_ratio_z"]
    late_frequency_threshold = (
        float(late_high_frequency_zscore_threshold)
        if late_high_frequency_zscore_threshold is not None
        else float(ringing_zscore_threshold)
    )
    reasons = []
    for _, row in table.iterrows():
        r = []
        n_response_samples = max(
            int(round(float(row["response_duration_s"]) * float(epochs.sfreq))) + 1,
            1,
        )
        plateau_required = max(min_plateau_run, int(np.ceil(float(plateau_fraction) * n_response_samples)))
        plateau_evidence = row["plateau_run_samples"] >= plateau_required
        if row["nan_fraction"] > max_nan_fraction:
            r.append("nan_fraction")
        if row["saturation_fraction"] > saturation_fraction and plateau_evidence:
            r.append("saturation_or_clipping")
        if plateau_evidence:
            r.append("plateau_clipping")
        if bool(row["rail_limits_known"]) and row["rail_fraction"] > rail_fraction:
            r.append("rail_clipping")
        if absolute_peak_uv is not None and np.isfinite(row["peak_abs"]) and row["peak_abs"] >= float(absolute_peak_uv):
            r.append("absolute_peak")
        if absolute_ptp_uv is not None and np.isfinite(row["ptp"]) and row["ptp"] >= float(absolute_ptp_uv):
            r.append("absolute_ptp")
        if extreme_zscore_threshold is not None and (
            row["peak_abs_z"] >= float(extreme_zscore_threshold)
            or row["ptp_z"] >= float(extreme_zscore_threshold)
        ):
            r.append("extreme_amplitude_outlier")
        if row["peak_abs_z"] >= zscore_threshold or row["ptp_z"] >= zscore_threshold:
            r.append("amplifier_or_motion_artifact")
        if row["max_gradient_z"] >= zscore_threshold:
            r.append("sharp_transient")
        if row["late_high_frequency_ratio_z"] >= late_frequency_threshold:
            r.append("late_high_frequency_outlier")
        reasons.append(";".join(r))
    table["reason"] = reasons
    table["bad_response"] = table["reason"].astype(bool)
    table["hard_artifact_response"] = table["reason"].map(
        lambda text: bool(DEFAULT_HARD_ARTIFACT_REASONS.intersection(_split_reasons(text)))
    )
    return ArtifactResponseReport(table.sort_values(["bad_response", "epoch", "channel"], ascending=[False, True, True]))


def response_artifact_summary(
    report: ArtifactResponseReport | pd.DataFrame,
    group_cols: Iterable[str] | None = None,
    hard_artifact_reasons: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return an auditable channel-level response-artifact summary.

    Parameters
    ----------
    report:
        Artifact response report or its table.
    group_cols:
        Optional columns such as ``patient_id``, ``session_id``, or
        ``stim_pair`` to preserve in the grouping.
    """

    table = report.table if isinstance(report, ArtifactResponseReport) else pd.DataFrame(report)
    group_cols = list(group_cols or [])
    hard_reasons = set(DEFAULT_HARD_ARTIFACT_REASONS if hard_artifact_reasons is None else hard_artifact_reasons)
    if table.empty:
        columns = [
            *group_cols,
            "channel",
            "n_epochs",
            "n_bad_response",
            "n_clean_response",
            "bad_response_fraction",
            "n_hard_artifact_response",
            "hard_artifact_fraction",
            "artifact_reasons",
        ]
        return pd.DataFrame(columns=columns)
    missing = [col for col in [*group_cols, "channel", "bad_response"] if col not in table.columns]
    if missing:
        raise ValueError(f"artifact table is missing required columns: {missing}")
    data = table.copy()
    data["bad_response"] = data["bad_response"].fillna(False).astype(bool)
    reason_values = data["reason"] if "reason" in data.columns else pd.Series("", index=data.index)
    data["_hard_artifact_response"] = reason_values.map(
        lambda text: bool(hard_reasons.intersection(_split_reasons(text)))
    )

    def _reason_text(values: pd.Series) -> str:
        reasons: set[str] = set()
        for value in values.dropna().astype(str):
            reasons.update(part for part in value.split(";") if part)
        return ";".join(sorted(reasons))

    summary = (
        data.groupby([*group_cols, "channel"], dropna=False)
        .agg(
            n_epochs=("bad_response", "size"),
            n_bad_response=("bad_response", "sum"),
            n_hard_artifact_response=("_hard_artifact_response", "sum"),
            max_peak_abs=("peak_abs", "max") if "peak_abs" in data.columns else ("bad_response", "size"),
            max_ptp=("ptp", "max") if "ptp" in data.columns else ("bad_response", "size"),
            max_saturation_fraction=("saturation_fraction", "max") if "saturation_fraction" in data.columns else ("bad_response", "size"),
            max_plateau_fraction=("plateau_fraction", "max") if "plateau_fraction" in data.columns else ("bad_response", "size"),
            max_plateau_run_s=("plateau_run_s", "max") if "plateau_run_s" in data.columns else ("bad_response", "size"),
            max_extreme_dwell_fraction=("extreme_dwell_fraction", "max") if "extreme_dwell_fraction" in data.columns else ("bad_response", "size"),
            max_rail_fraction=("rail_fraction", "max") if "rail_fraction" in data.columns else ("bad_response", "size"),
            artifact_reasons=("reason", _reason_text) if "reason" in data.columns else ("bad_response", "size"),
        )
        .reset_index()
    )
    summary["n_clean_response"] = summary["n_epochs"] - summary["n_bad_response"]
    summary["bad_response_fraction"] = summary["n_bad_response"] / summary["n_epochs"].replace(0, np.nan)
    summary["hard_artifact_fraction"] = summary["n_hard_artifact_response"] / summary["n_epochs"].replace(0, np.nan)
    sort_cols = [*group_cols, "bad_response_fraction", "n_bad_response"]
    ascending = [True] * len(group_cols) + [False, False]
    return summary.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def reject_artifactual_responses(
    epochs,
    report: ArtifactResponseReport | None = None,
    mode: str = "drop_epoch",
    max_bad_channel_fraction: float = 0.25,
    max_bad_response_fraction: float = 0.25,
    **detect_kwargs,
):
    """Remove or mask aberrant trial-channel responses from an Epochs object."""

    from .epochs import Epochs

    report = report or detect_artifactual_responses(epochs, **detect_kwargs)
    table = report.table
    if table.empty or not table["bad_response"].any():
        return epochs

    df = epochs.epochs_df.copy()
    if mode == "nan_response":
        for _, row in table[table["bad_response"]].iterrows():
            ep = int(row["epoch"])
            ch = row["channel"]
            if ch in df.columns and ep in df.index.get_level_values("epoch"):
                df.loc[(ep, slice(None)), ch] = np.nan
    elif mode == "drop_epoch":
        bad = table[table["bad_response"]]
        frac = bad.groupby("epoch")["channel"].nunique() / max(len(epochs.channels), 1)
        drop_epochs = frac[frac >= max_bad_channel_fraction].index.astype(int).tolist()
        kept_epochs = [int(ep) for ep in sorted(df.index.get_level_values("epoch").unique()) if int(ep) not in set(drop_epochs)]
        epoch_mapping = {old: new for new, old in enumerate(kept_epochs)}
        df = df.drop(index=drop_epochs, level="epoch", errors="ignore")
        df = _renumber_epoch_index(df)
    elif mode == "drop_channel":
        summary = report.by_channel()
        bad_channels = summary.loc[
            summary["bad_response_fraction"] > float(max_bad_response_fraction),
            "channel",
        ].astype(str)
        df = df.drop(columns=bad_channels.tolist(), errors="ignore")
    else:
        raise ValueError("mode must be 'drop_epoch', 'drop_channel', or 'nan_response'")

    md = dict(epochs.metadata)
    md["artifact_rejection_mode"] = mode
    md["artifact_rejection_n_flagged"] = int(table["bad_response"].sum())
    if mode == "drop_epoch" and md.get("zero_anchor_reports"):
        anchor_reports = []
        for anchor_report in md.get("zero_anchor_reports", []):
            report_epoch = int(anchor_report.get("epoch", -1))
            if report_epoch in epoch_mapping:
                anchor_reports.append({**anchor_report, "epoch": int(epoch_mapping[report_epoch])})
        md["zero_anchor_reports"] = anchor_reports
    return Epochs(
        epochs_df=df,
        sfreq=epochs.sfreq,
        tmin=epochs.tmin,
        tmax=epochs.tmax,
        baseline=epochs.baseline,
        stim_ch=epochs.stim_ch,
        metadata=md,
        zero_time=epochs.zero_time,
    )


def _late_high_frequency_ratio(x: np.ndarray, times: np.ndarray, sfreq: float) -> float:
    finite = np.isfinite(x)
    if finite.sum() < 8:
        return np.nan
    y = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)
    early = (times >= 0.0) & (times <= min(0.05, times.max()))
    late = (times >= 0.05) & (times <= min(0.5, times.max()))
    if not early.any() or not late.any():
        return np.nan
    grad = np.abs(np.gradient(y))
    return float(np.nanmean(grad[late]) / (np.nanmean(grad[early]) + 1e-12))


def _longest_near_constant_run(x: np.ndarray, epsilon_uv: float | None = None) -> int:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0
    if finite.size == 1:
        return 1
    if epsilon_uv is None:
        # Keep this tight: it is intended to catch digitized amplifier rails or
        # interpolation plateaus, not smooth physiological extrema.
        scale = float(np.nanmedian(np.abs(finite - np.nanmedian(finite))))
        epsilon_uv = max(1e-9, scale * 1e-6)
    diffs = np.abs(np.diff(finite)) <= float(epsilon_uv)
    longest = current = 1
    for is_flat in diffs:
        if bool(is_flat):
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return int(max(longest, current))


def _rail_clip_fraction(
    x: np.ndarray,
    *,
    epsilon_fraction: float = 0.002,
    epsilon_uv: float | None = None,
    rail_limits: tuple[float, float] | None = None,
) -> float:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    if rail_limits is None:
        xmin = float(np.nanmin(finite))
        xmax = float(np.nanmax(finite))
    else:
        xmin, xmax = map(float, rail_limits)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
    span = xmax - xmin
    if not np.isfinite(span) or span <= 0:
        return 1.0
    eps = float(epsilon_uv) if epsilon_uv is not None else max(1e-9, span * float(epsilon_fraction))
    near_rail = (finite <= xmin + eps) | (finite >= xmax - eps)
    return float(np.nanmean(near_rail))


def _channel_rail_limits(
    rail_limits_uv: tuple[float, float] | Mapping[str, tuple[float, float]] | None,
    channel: str,
) -> tuple[float, float] | None:
    if rail_limits_uv is None:
        return None
    if isinstance(rail_limits_uv, Mapping):
        limits = rail_limits_uv.get(channel)
        return tuple(map(float, limits)) if limits is not None else None
    return tuple(map(float, rail_limits_uv))


def _split_reasons(value: object) -> set[str]:
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return set()
    return {part for part in str(value).split(";") if part}


def _renumber_epoch_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    epochs = sorted(df.index.get_level_values("epoch").unique())
    mapping = {old: new for new, old in enumerate(epochs)}
    tuples = [(mapping[ep], t) for ep, t in df.index]
    out = df.copy()
    out.index = pd.MultiIndex.from_tuples(tuples, names=df.index.names)
    return out
