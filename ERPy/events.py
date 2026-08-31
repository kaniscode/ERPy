from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import signal

from .utils.jobrunner import JobRunner
from .utils.utils import ensure_dir, infer_sfreq, parse_list, read_time_series_csv, write_time_series_csv


class EventDetectionError(RuntimeError):
    """Raised when no defensible stimulation-event sequence can be detected."""


def _clean_mne_channel_label(label: str) -> str:
    text = str(label).strip()
    text = re.sub(r"^POL\s*", "", text, flags=re.I)
    text = re.sub(r"[-\s_]*Ref$", "", text, flags=re.I)
    text = re.sub(r"\s+", "", text)
    text = text.replace("-", "")
    text = re.sub(r"([A-Za-z]+)[_]?0*([0-9]+)$", lambda m: f"{m.group(1)}{int(m.group(2))}", text)
    return text


def _infer_distance_samples(sfreq: float, stim_freq: float | None = None, min_interval_s: float | None = None) -> int:
    if min_interval_s is not None:
        return max(int(round(float(min_interval_s) * sfreq)), 1)
    if stim_freq is None or float(stim_freq) <= 0:
        return max(int(round(0.05 * sfreq)), 1)
    return max(
        int(round(0.65 * sfreq / float(stim_freq))),
        1,
    )


def _event_df(times: Iterable[Any], labels: Iterable[Any] | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"times": list(times)})
    if labels is not None:
        out["label"] = list(labels)
    if not out.empty:
        if not pd.api.types.is_numeric_dtype(out["times"]):
            parsed = pd.to_datetime(out["times"], errors="coerce")
            if parsed.notna().mean() > 0.8:
                out["times"] = parsed
        out = out.drop_duplicates("times").sort_values("times").reset_index(drop=True)
    return out


def _event_times_to_seconds(values: Iterable[Any] | pd.Series) -> np.ndarray:
    """Convert event times to seconds without assuming a datetime resolution."""

    series = pd.Series(values, copy=False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    seconds = np.full(len(parsed), np.nan, dtype=float)
    valid = parsed.notna()
    if valid.any():
        reference = parsed.loc[valid].iloc[0]
        seconds[valid.to_numpy()] = (
            parsed.loc[valid] - reference
        ).dt.total_seconds().to_numpy(dtype=float)
    return seconds


def _regularize_event_sequence(
    events: pd.DataFrame,
    stim_freq: float | None,
    *,
    tolerance: float = 0.12,
) -> pd.DataFrame:
    """Keep the longest cadence-consistent path through candidate events."""

    output = pd.DataFrame(events).copy()
    input_count = int(len(output))
    frequency = pd.to_numeric(pd.Series([stim_freq]), errors="coerce").iloc[0]
    if input_count < 2 or pd.isna(frequency) or float(frequency) <= 0:
        output["event_count_before_regularization"] = input_count
        output["event_count_removed_by_regularization"] = 0
        output["event_regularization_retained_fraction"] = 1.0
        return output

    seconds = _event_times_to_seconds(output["times"])
    finite_positions = np.flatnonzero(np.isfinite(seconds))
    if len(finite_positions) < 2:
        return output.iloc[finite_positions].copy()

    seconds = seconds[finite_positions]
    order = np.argsort(seconds, kind="stable")
    seconds = seconds[order]
    original_positions = finite_positions[order]
    expected_interval = 1.0 / float(frequency)
    n_events = len(seconds)
    path_lengths = np.ones(n_events, dtype=int)
    path_costs = np.zeros(n_events, dtype=float)
    predecessors = np.full(n_events, -1, dtype=int)

    for stop in range(n_events):
        for start in range(stop):
            delta = float(seconds[stop] - seconds[start])
            pulse_steps = int(max(round(delta / expected_interval), 1))
            per_pulse_interval = delta / pulse_steps
            relative_error = abs(
                per_pulse_interval - expected_interval
            ) / expected_interval
            if relative_error > float(tolerance):
                continue
            candidate_length = int(path_lengths[start] + 1)
            candidate_cost = float(
                path_costs[start] + relative_error**2
            )
            if (
                candidate_length > path_lengths[stop]
                or (
                    candidate_length == path_lengths[stop]
                    and candidate_cost < path_costs[stop]
                )
            ):
                path_lengths[stop] = candidate_length
                path_costs[stop] = candidate_cost
                predecessors[stop] = start

    best_length = int(path_lengths.max())
    endpoints = np.flatnonzero(path_lengths == best_length)
    endpoint = int(endpoints[np.argmin(path_costs[endpoints])])
    selected = []
    while endpoint >= 0:
        selected.append(int(original_positions[endpoint]))
        endpoint = int(predecessors[endpoint])
    selected.reverse()
    output = output.iloc[selected].copy().reset_index(drop=True)
    retained_fraction = float(len(output) / input_count) if input_count else 1.0
    output["event_count_before_regularization"] = input_count
    output["event_count_removed_by_regularization"] = input_count - len(output)
    output["event_regularization_retained_fraction"] = retained_fraction
    output["event_regularization_tolerance"] = float(tolerance)
    return output


def _event_candidate_score(
    events: pd.DataFrame,
    *,
    label: str,
    expected_count: float,
    stim_freq: float | None,
) -> float:
    count_error = (
        abs(len(events) - expected_count) / expected_count
        if expected_count > 0
        else 0.0
    )
    cadence_error = 0.0
    frequency = pd.to_numeric(pd.Series([stim_freq]), errors="coerce").iloc[0]
    if len(events) > 1 and pd.notna(frequency) and float(frequency) > 0:
        values = events["times"]
        if pd.api.types.is_numeric_dtype(values):
            ordered = np.sort(
                pd.to_numeric(values, errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            intervals = np.diff(ordered)
        else:
            ordered = pd.to_datetime(values, errors="coerce").dropna().sort_values()
            intervals = (
                ordered.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
            )
        expected_interval = 1.0 / float(frequency)
        if len(intervals):
            multiples = np.maximum(
                np.rint(intervals / expected_interval),
                1.0,
            )
            cadence_error = float(
                np.mean(
                    np.abs(intervals / multiples - expected_interval)
                    / expected_interval
                )
            )
    source_penalty = {
        "annotations": 0.0,
        "trigger": 0.0,
        "adjacent": 0.02,
        "consensus": 0.025,
        "adjacent_adaptive": 0.04,
        "consensus_adaptive": 0.045,
    }.get(str(label), 0.05)
    retained = pd.to_numeric(
        events.get(
            "event_regularization_retained_fraction",
            pd.Series(1.0, index=events.index),
        ),
        errors="coerce",
    ).median()
    regularization_penalty = (
        max(1.0 - float(retained), 0.0) * 0.05
        if pd.notna(retained)
        else 0.0
    )
    return float(
        count_error
        + 0.25 * cadence_error
        + source_penalty
        + regularization_penalty
    )


def create_events_from_timestamps(
    timestamps: Iterable[Any] | pd.Series | pd.DataFrame,
    label: str | None = None,
    time_column: str = "times",
) -> pd.DataFrame:
    """Create canonical ERPy event annotations from timestamp rows."""

    if isinstance(timestamps, pd.DataFrame):
        if time_column in timestamps.columns:
            times = timestamps[time_column]
        elif "onset" in timestamps.columns:
            times = timestamps["onset"]
        else:
            times = timestamps.iloc[:, 0]
        labels = timestamps["label"] if "label" in timestamps.columns else None
    else:
        times = timestamps
        labels = None
    df = _event_df(times, labels)
    if label is not None:
        df["label"] = label
    return df


def create_events_from_binary_series(
    series: Iterable[Any] | pd.Series | pd.DataFrame,
    start_datetime: Any | None = None,
    end_datetime: Any | None = None,
    sfreq: float | None = None,
    event_value: int | float | str = 1,
    event_column: int | str = 0,
    edge: str = "rising",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a binary marker vector into a datetime-indexed table and events."""

    if isinstance(series, pd.DataFrame):
        marker = series[event_column] if event_column in series.columns else series.iloc[:, int(event_column)]
        base = series.copy()
    else:
        marker = pd.Series(series)
        base = pd.DataFrame({"event": marker})

    if isinstance(base.index, pd.DatetimeIndex):
        index = base.index
    elif start_datetime is not None and end_datetime is not None:
        index = pd.date_range(pd.Timestamp(start_datetime), pd.Timestamp(end_datetime), periods=len(base))
    elif start_datetime is not None and sfreq is not None:
        index = pd.DatetimeIndex(pd.Timestamp(start_datetime) + pd.to_timedelta(np.arange(len(base)) / sfreq, unit="s"))
    else:
        index = pd.Index(np.arange(len(base), dtype=float) / float(sfreq or 1.0), name="time")

    base.index = index
    values = marker.to_numpy()
    active = values == event_value
    if edge == "level":
        hits = np.flatnonzero(active)
    else:
        prev = np.r_[False, active[:-1]]
        rising = active & ~prev
        falling = ~active & prev
        if edge == "rising":
            hits = np.flatnonzero(rising)
        elif edge == "falling":
            hits = np.flatnonzero(falling)
        elif edge == "both":
            hits = np.flatnonzero(rising | falling)
        else:
            raise ValueError("edge must be 'rising', 'falling', 'both', or 'level'")
    return base, _event_df(base.index[hits])


def evoked_to_datetime(
    df: pd.DataFrame,
    start_datetime: Any,
    end_datetime: Any,
    event_column: int | str = 0,
    event_value: int | float | str = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a datetime axis and extract rising binary-event onsets.

    This compatibility helper returns both the timestamp-indexed input frame
    and a one-column ``times`` table containing detected onset timestamps.
    """

    return create_events_from_binary_series(
        df,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        event_column=event_column,
        event_value=event_value,
        edge="rising",
    )


def _parse_stim_pair(stim_pair: str) -> tuple[str, int | None, int | None]:
    parts = str(stim_pair).split("_")
    if len(parts) >= 3:
        try:
            return parts[0], int(parts[1]), int(parts[2])
        except Exception:
            return parts[0], None, None
    match = re.match(r"([A-Za-z]+)(\d+)[_-]?(\d+)", str(stim_pair))
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return str(stim_pair), None, None


def _candidate_channels(df: pd.DataFrame, stim_pair: str, mode: str, electrode_range: int = 4) -> list[str]:
    lead, pos, neg = _parse_stim_pair(stim_pair)
    columns = [str(c) for c in df.select_dtypes(include=[np.number]).columns]
    if pos is None or neg is None:
        return columns
    numbers: set[int]
    if mode == "adjacent":
        numbers = {pos - 1, pos + 1, neg - 1, neg + 1}
    else:
        lo = max(min(pos, neg) - electrode_range, 1)
        hi = max(pos, neg) + electrode_range
        numbers = set(range(lo, hi + 1))
    numbers -= {pos, neg}
    patterns = {
        label.casefold()
        for n in numbers
        for label in (f"{lead}{n}", f"{lead}_{n}")
    }
    picked = [
        col
        for col in columns
        if _clean_mne_channel_label(col).casefold() in patterns
        or col.casefold() in patterns
    ]
    return picked or columns


def _cluster_peak_samples(
    peaks_by_channel: dict[str, np.ndarray],
    margin_samples: int,
    min_channel_consensus: int,
) -> np.ndarray:
    points: list[tuple[int, str]] = []
    for ch, peaks in peaks_by_channel.items():
        points.extend((int(p), ch) for p in peaks)
    if not points:
        return np.array([], dtype=int)
    points.sort(key=lambda x: x[0])
    clusters: list[list[tuple[int, str]]] = []
    current = [points[0]]
    for point in points[1:]:
        if point[0] - current[-1][0] <= margin_samples:
            current.append(point)
        else:
            clusters.append(current)
            current = [point]
    clusters.append(current)
    accepted: list[int] = []
    for cluster in clusters:
        channels = {ch for _, ch in cluster}
        if len(channels) >= min_channel_consensus:
            accepted.append(int(round(np.median([p for p, _ in cluster]))))
    return np.array(sorted(set(accepted)), dtype=int)


def _positions_to_times(df: pd.DataFrame, positions: np.ndarray, start_datetime: Any | None = None) -> pd.Index:
    if isinstance(df.index, pd.DatetimeIndex):
        positions = positions[(positions >= 0) & (positions < len(df))]
        return df.index[positions]
    if start_datetime is not None:
        sfreq = infer_sfreq(df)
        return pd.DatetimeIndex(pd.Timestamp(start_datetime) + pd.to_timedelta(positions / sfreq, unit="s"))
    positions = positions[(positions >= 0) & (positions < len(df))]
    return pd.Index(np.asarray(df.index)[positions])


def _load_signal(
    dataloader,
    session_id: str,
    stim_pair: str,
    raw_source: str,
    min_window_coverage: float,
    *,
    raw_file: str | Path | None = None,
    stim_start: Any | None = None,
) -> pd.DataFrame:
    if raw_source not in {"auto", "csv", "edf", "nwb", "raw"}:
        raise ValueError("raw_source must be 'auto', 'csv', 'edf', 'nwb', or 'raw'")
    if raw_source in {"edf", "nwb"}:
        return dataloader.load_stim_data(
            session_id,
            stim_pair,
            source="raw",
            raw_file=raw_file,
            stim_start=stim_start,
        )
    if raw_source in {"csv", "auto"}:
        try:
            df = dataloader.load_stim_data(
                session_id,
                stim_pair,
                source="csv",
                raw_file=raw_file,
                stim_start=stim_start,
            )
            if raw_source == "auto" and raw_file is None and isinstance(df.index, pd.DatetimeIndex):
                row = dataloader.get_stim_row(session_id, stim_pair)
                start = pd.Timestamp(row["stim_start"])
                stop = pd.Timestamp(row["stim_stop"])
                total = max((stop - start).total_seconds(), 1e-9)
                covered = max((df.index.max() - df.index.min()).total_seconds(), 0)
                if covered / total < min_window_coverage:
                    return dataloader.load_stim_data(
                        session_id,
                        stim_pair,
                        source="raw",
                        raw_file=raw_file,
                        stim_start=stim_start,
                    )
            return df
        except Exception:
            if raw_source == "csv":
                raise
    return dataloader.load_stim_data(
        session_id,
        stim_pair,
        source="raw",
        raw_file=raw_file,
        stim_start=stim_start,
    )


def _timestamp_for_index(value: Any, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if index.tz is None:
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
    elif timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(index.tz)
    else:
        timestamp = timestamp.tz_convert(index.tz)
    return timestamp


def _stim_window_bounds(
    dataloader: Any,
    session_id: str,
    stim_pair: str,
    *,
    stim_start: Any | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    try:
        row = dataloader.get_stim_row(
            session_id,
            stim_pair,
            stim_start=stim_start,
        )
        start = pd.to_datetime(row.get("stim_start"), errors="coerce")
        stop = pd.to_datetime(row.get("stim_stop"), errors="coerce")
    except Exception:
        return None
    if pd.isna(start) or pd.isna(stop) or pd.Timestamp(stop) <= pd.Timestamp(start):
        return None
    return pd.Timestamp(start), pd.Timestamp(stop)


def _restrict_signal_to_stim_window(
    df: pd.DataFrame,
    dataloader: Any,
    session_id: str,
    stim_pair: str,
    *,
    stim_start: Any | None = None,
    margin_s: float = 0.0,
) -> pd.DataFrame:
    bounds = _stim_window_bounds(
        dataloader,
        session_id,
        stim_pair,
        stim_start=stim_start,
    )
    if bounds is None or not isinstance(df.index, pd.DatetimeIndex):
        return df
    start, stop = bounds
    start = _timestamp_for_index(start, df.index) - pd.to_timedelta(float(margin_s), unit="s")
    stop = _timestamp_for_index(stop, df.index) + pd.to_timedelta(float(margin_s), unit="s")
    attrs = dict(df.attrs)
    selected = df.loc[(df.index >= start) & (df.index <= stop)].copy()
    if selected.empty:
        raise ValueError(
            "Stimulation window {} to {} does not overlap signal index {} to {}".format(
                start,
                stop,
                df.index.min(),
                df.index.max(),
            )
        )
    selected.attrs.update(attrs)
    selected.attrs["event_detection_window_start"] = str(start)
    selected.attrs["event_detection_window_stop"] = str(stop)
    return selected


def _restrict_events_to_stim_window(
    events: pd.DataFrame,
    dataloader: Any,
    session_id: str,
    stim_pair: str,
    *,
    stim_start: Any | None = None,
    margin_s: float = 0.0,
) -> pd.DataFrame:
    bounds = _stim_window_bounds(
        dataloader,
        session_id,
        stim_pair,
        stim_start=stim_start,
    )
    if bounds is None or events.empty or "times" not in events.columns:
        return events
    values = pd.to_datetime(events["times"], errors="coerce")
    if values.notna().mean() <= 0.8:
        return events
    start, stop = bounds
    if getattr(values.dt, "tz", None) is None:
        if start.tzinfo is not None:
            start = start.tz_convert(None)
        if stop.tzinfo is not None:
            stop = stop.tz_convert(None)
    else:
        timezone = values.dt.tz
        start = start.tz_localize(timezone) if start.tzinfo is None else start.tz_convert(timezone)
        stop = stop.tz_localize(timezone) if stop.tzinfo is None else stop.tz_convert(timezone)
    start -= pd.to_timedelta(float(margin_s), unit="s")
    stop += pd.to_timedelta(float(margin_s), unit="s")
    selected = events.loc[values.between(start, stop, inclusive="both")].copy()
    return selected.reset_index(drop=True)


def _adaptive_channel_threshold(
    detector: np.ndarray,
    configured_threshold: float,
    *,
    floor_uv: float,
    noise_z: float,
) -> tuple[float, float]:
    finite = detector[np.isfinite(detector)]
    if not len(finite):
        return float(configured_threshold), 0.0
    center = float(np.nanmedian(finite))
    noise_sigma = float(1.4826 * np.nanmedian(np.abs(finite - center)))
    if not np.isfinite(noise_sigma):
        noise_sigma = 0.0
    threshold = max(float(floor_uv), center + float(noise_z) * noise_sigma)
    threshold = min(float(configured_threshold), threshold)
    prominence = max(0.0, float(noise_z) * 0.5 * noise_sigma)
    return float(threshold), float(prominence)


def _detect_amplitude_events(
    df: pd.DataFrame,
    stim_pair: str,
    method: str,
    stim_freq: float | None,
    min_peak_height: float,
    consensus_factor: float = 4.0,
    electrode_range: int = 4,
    min_adjacent_consensus: int = 1,
    time_margin: float = 0.005,
    polarity: str = "abs",
    adaptive_threshold: bool = False,
    adaptive_floor_uv: float = 100.0,
    adaptive_noise_z: float = 12.0,
) -> pd.DataFrame:
    sfreq = float(df.attrs.get("sfreq") or infer_sfreq(df))
    channels = _candidate_channels(df, stim_pair, "adjacent" if method == "adjacent" else "consensus", electrode_range)
    distance = _infer_distance_samples(sfreq, stim_freq)
    margin = max(int(round(float(time_margin) * sfreq)), 1)
    peaks_by_channel: dict[str, np.ndarray] = {}
    thresholds: list[float] = []
    for ch in channels:
        if ch not in df.columns:
            continue
        y = pd.to_numeric(df[ch], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
        if polarity == "positive":
            detector = y
        elif polarity == "negative":
            detector = -y
        else:
            detector = np.abs(y - np.nanmedian(y))
        threshold = float(min_peak_height)
        prominence = None
        if adaptive_threshold:
            threshold, prominence = _adaptive_channel_threshold(
                detector,
                min_peak_height,
                floor_uv=adaptive_floor_uv,
                noise_z=adaptive_noise_z,
            )
        thresholds.append(threshold)
        peaks, _ = signal.find_peaks(
            detector,
            height=threshold,
            prominence=prominence,
            distance=distance,
        )
        if len(peaks):
            peaks_by_channel[ch] = peaks
    if method == "adjacent":
        min_cons = min_adjacent_consensus
    else:
        min_cons = max(int(np.floor(max(len(channels), 1) / float(consensus_factor))), 1)
    positions = _cluster_peak_samples(peaks_by_channel, margin, min_cons)
    out = _event_df(_positions_to_times(df, positions))
    if thresholds:
        out["detection_threshold_uv"] = float(np.median(thresholds))
        out["detection_threshold_min_uv"] = float(np.min(thresholds))
        out["detection_threshold_max_uv"] = float(np.max(thresholds))
    out["detection_channel_count"] = int(len(channels))
    out["detection_adaptive"] = bool(adaptive_threshold)
    return out


def _detect_matched_filter_events(
    df: pd.DataFrame,
    stim_pair: str,
    stim_freq: float | None,
    channel: str | None = None,
    template: np.ndarray | None = None,
    template_window_s: tuple[float, float] = (0.002, 0.002),
    corr_threshold: float = 0.6,
    min_peak_height_for_template: float = 1000.0,
) -> pd.DataFrame:
    sfreq = float(df.attrs.get("sfreq") or infer_sfreq(df))
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return _event_df([])
    if channel is None:
        candidates = _candidate_channels(numeric, stim_pair, "consensus")
        scores = numeric[candidates].abs().quantile(0.99).sort_values(ascending=False)
        channel = str(scores.index[0])
    y = pd.to_numeric(numeric[channel], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    y = y - np.nanmedian(y)

    if template is None:
        distance = _infer_distance_samples(sfreq, stim_freq)
        peaks, _ = signal.find_peaks(np.abs(y), height=min_peak_height_for_template, distance=distance)
        pre = max(int(round(template_window_s[0] * sfreq)), 1)
        post = max(int(round(template_window_s[1] * sfreq)), 1)
        snippets = [y[p - pre : p + post + 1] for p in peaks[:20] if p - pre >= 0 and p + post < len(y)]
        if not snippets:
            return _event_df([])
        template = np.mean(np.vstack(snippets), axis=0)
    template = np.asarray(template, dtype=float)
    template = template - np.nanmean(template)
    denom_t = np.linalg.norm(template) or 1.0
    corr = signal.correlate(y, template, mode="same") / denom_t
    win = max(len(template), 1)
    power = np.sqrt(signal.convolve(y * y, np.ones(win), mode="same"))
    corr = corr / np.maximum(power, 1e-12)
    peaks, _ = signal.find_peaks(corr, height=corr_threshold, distance=_infer_distance_samples(sfreq, stim_freq))
    return _event_df(_positions_to_times(df, peaks))


def _detect_trigger_events(
    df: pd.DataFrame,
    trigger_channels: list[str] | None = None,
    trigger_regex: str | None = None,
    edge: str = "rising",
    threshold: float | None = None,
    min_interval_s: float = 0.001,
    merge_margin_s: float = 0.001,
    min_channel_consensus: int = 1,
) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if trigger_channels:
        channels = [ch for ch in trigger_channels if ch in numeric.columns]
    elif trigger_regex:
        rgx = re.compile(trigger_regex, flags=re.I)
        channels = [ch for ch in numeric.columns if rgx.search(str(ch))]
    else:
        rgx = re.compile(
            r"^(?:TRIG(?:GER)?|TTL|STATUS|EVENT|MARKER|"
            r"STIM(?:ULUS)?|STIMTRIG(?:GER)?|TR)\d*$",
            flags=re.I,
        )
        channels = [
            ch
            for ch in numeric.columns
            if rgx.fullmatch(re.sub(r"[\s._-]+", "", str(ch)))
        ]
    if not channels:
        raise ValueError("No trigger channels matched")
    sfreq = float(df.attrs.get("sfreq") or infer_sfreq(df))
    peaks_by_channel: dict[str, np.ndarray] = {}
    for ch in channels:
        y = numeric[ch].interpolate(limit_direction="both").to_numpy(dtype=float)
        thr = float(threshold) if threshold is not None else (np.nanmin(y) + np.nanmax(y)) / 2.0
        active = y >= thr
        prev = np.r_[False, active[:-1]]
        if edge == "rising":
            hits = np.flatnonzero(active & ~prev)
        elif edge == "falling":
            hits = np.flatnonzero(~active & prev)
        elif edge == "both":
            hits = np.flatnonzero((active & ~prev) | (~active & prev))
        else:
            raise ValueError("edge must be 'rising', 'falling', or 'both'")
        if len(hits):
            min_dist = _infer_distance_samples(sfreq, min_interval_s=min_interval_s)
            kept = [hits[0]]
            for hit in hits[1:]:
                if hit - kept[-1] >= min_dist:
                    kept.append(hit)
            peaks_by_channel[ch] = np.array(kept, dtype=int)
    positions = _cluster_peak_samples(peaks_by_channel, max(int(round(merge_margin_s * sfreq)), 1), min_channel_consensus)
    return _event_df(_positions_to_times(df, positions))


def _auto_event_candidate_is_plausible(
    events: pd.DataFrame,
    dataloader: Any,
    session_id: str,
    stim_pair: str,
    *,
    stim_freq: float | None,
    stim_start: Any | None,
) -> tuple[bool, str]:
    """Reject nonempty auto-detection results that cannot represent the pulse train."""

    if events.empty:
        return False, "empty"
    if "event_regularization_retained_fraction" in events.columns:
        retained = pd.to_numeric(
            events["event_regularization_retained_fraction"],
            errors="coerce",
        ).median()
        if pd.notna(retained) and float(retained) < 0.65:
            return (
                False,
                "cadence_consistency_after_regularization="
                f"{float(retained):.1%} required>=65.0%",
            )
    frequency = pd.to_numeric(pd.Series([stim_freq]), errors="coerce").iloc[0]
    if pd.isna(frequency) or float(frequency) <= 0:
        return True, "accepted_without_frequency"
    frequency = float(frequency)
    expected_interval = 1.0 / frequency

    values = events["times"]
    if pd.api.types.is_numeric_dtype(values):
        ordered = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
        intervals = np.diff(ordered)
    else:
        parsed = pd.to_datetime(values, errors="coerce").dropna().sort_values()
        intervals = parsed.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals):
        median_interval = float(np.median(intervals))
        ratio = median_interval / expected_interval
        if ratio < 0.65 or ratio > 1.5:
            return (
                False,
                "median_interval={:.6g}s expected={:.6g}s".format(
                    median_interval,
                    expected_interval,
                ),
            )
        multiples = np.maximum(np.rint(intervals / expected_interval), 1.0)
        per_pulse_intervals = intervals / multiples
        relative_error = np.abs(per_pulse_intervals - expected_interval) / expected_interval
        regular_fraction = float(np.mean(relative_error <= 0.12))
        if len(intervals) >= 3 and regular_fraction < 0.65:
            return (
                False,
                "cadence_consistency={:.1%} required>=65.0%".format(
                    regular_fraction,
                ),
            )

    try:
        row = dataloader.get_stim_row(
            session_id,
            stim_pair,
            stim_start=stim_start,
        )
        start = pd.Timestamp(row["stim_start"])
        stop = pd.Timestamp(row["stim_stop"])
        duration_s = max((stop - start).total_seconds(), 0.0)
    except Exception:
        duration_s = 0.0
    expected_count = duration_s * frequency if duration_s > 0 else 0.0
    if expected_count >= 3:
        minimum = max(int(math.floor(expected_count * 0.5)), 2)
        maximum = max(int(math.ceil(expected_count * 1.5)) + 1, minimum)
        if not minimum <= len(events) <= maximum:
            return (
                False,
                "count={} expected_about={:.1f} accepted_range={}-{}".format(
                    len(events),
                    expected_count,
                    minimum,
                    maximum,
                ),
            )
    return True, "pulse_train_consistent"


def _detect_annotation_events_from_raw(raw, include_regex: str | None = None, exclude_regex: str | None = None) -> pd.DataFrame:
    include = re.compile(include_regex, flags=re.I) if include_regex else None
    exclude = re.compile(exclude_regex, flags=re.I) if exclude_regex else None
    meas_date = raw.info.get("meas_date")
    base = pd.Timestamp(meas_date).tz_localize(None) if meas_date is not None else pd.Timestamp("1970-01-01")
    rows = []
    labels = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        desc = str(desc)
        if include and not include.search(desc):
            continue
        if exclude and exclude.search(desc):
            continue
        rows.append(base + pd.to_timedelta(float(onset), unit="s"))
        labels.append(desc)
    return _event_df(rows, labels)


def detect_events_from_artifacts(
    dataloader,
    session_id: str,
    stim_pair: str,
    method: str = "consensus",
    stim_freq: float | None = None,
    min_peak_height: float = 1000.0,
    raw_source: str = "auto",
    min_window_coverage: float = 0.95,
    save_events: bool = False,
    raw_file: str | Path | None = None,
    stim_start: Any | None = None,
    signal_data: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Unified event detector for artifact, trigger, annotation, and binary sources.

    ``method="auto"`` searches only the annotated stimulation interval. It first
    tries explicit annotations/triggers and the configured amplitude threshold,
    then uses a robust noise-qualified threshold down to
    ``auto_min_peak_height`` (100 microvolts by default). Every automatic result
    must match the expected pulse count and cadence.
    """

    method = method.lower()
    if method == "auto":
        attempts: list[tuple[str, str, dict[str, Any]]] = []
        suffixes = [suffix.lower() for suffix in Path(str(raw_file or "")).suffixes]
        if not raw_file or any(suffix in {".edf", ".bdf"} for suffix in suffixes):
            attempts.append(("annotations", "annotations", {}))
        attempts.extend(
            [
                ("trigger", "trigger", {}),
                ("adjacent", "adjacent", {}),
                ("consensus", "consensus", {}),
            ]
        )
        adaptive_floor = float(kwargs.get("auto_min_peak_height", 100.0))
        if float(min_peak_height) > adaptive_floor:
            attempts.extend(
                [
                    (
                        "adjacent_adaptive",
                        "adjacent",
                        {
                            "adaptive_threshold": True,
                            "adaptive_floor_uv": adaptive_floor,
                            "adaptive_noise_z": float(kwargs.get("auto_peak_noise_z", 12.0)),
                        },
                    ),
                    (
                        "consensus_adaptive",
                        "consensus",
                        {
                            "adaptive_threshold": True,
                            "adaptive_floor_uv": adaptive_floor,
                            "adaptive_noise_z": float(kwargs.get("auto_peak_noise_z", 12.0)),
                        },
                    ),
                ]
            )
        errors = []
        plausible_candidates: list[
            tuple[float, str, str, pd.DataFrame]
        ] = []
        loaded_signal = signal_data
        for label, candidate, candidate_options in attempts:
            try:
                if candidate != "annotations" and loaded_signal is None:
                    loaded_signal = _load_signal(
                        dataloader,
                        session_id,
                        stim_pair,
                        raw_source,
                        min_window_coverage,
                        raw_file=raw_file,
                        stim_start=stim_start,
                    )
                nested_kwargs = dict(kwargs)
                for key in (
                    "adaptive_threshold",
                    "adaptive_floor_uv",
                    "adaptive_noise_z",
                ):
                    nested_kwargs.pop(key, None)
                nested_kwargs.update(candidate_options)
                candidate_events = detect_events_from_artifacts(
                    dataloader,
                    session_id,
                    stim_pair,
                    method=candidate,
                    stim_freq=stim_freq,
                    min_peak_height=min_peak_height,
                    raw_source=raw_source,
                    save_events=False,
                    raw_file=raw_file,
                    stim_start=stim_start,
                    signal_data=loaded_signal,
                    **nested_kwargs,
                )
                plausible, validation = _auto_event_candidate_is_plausible(
                    candidate_events,
                    dataloader,
                    session_id,
                    stim_pair,
                    stim_freq=stim_freq,
                    stim_start=stim_start,
                )
                if plausible:
                    try:
                        stim_row = dataloader.get_stim_row(
                            session_id,
                            stim_pair,
                            stim_start=stim_start,
                        )
                        duration_s = max(
                            (
                                pd.Timestamp(stim_row["stim_stop"])
                                - pd.Timestamp(stim_row["stim_start"])
                            ).total_seconds(),
                            0.0,
                        )
                    except Exception:
                        duration_s = 0.0
                    expected_count = (
                        duration_s * float(stim_freq)
                        if stim_freq is not None and float(stim_freq) > 0
                        else 0.0
                    )
                    score = _event_candidate_score(
                        candidate_events,
                        label=label,
                        expected_count=expected_count,
                        stim_freq=stim_freq,
                    )
                    plausible_candidates.append(
                        (score, label, validation, candidate_events.copy())
                    )
                    continue
                errors.append("{}: rejected ({})".format(label, validation))
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        if not plausible_candidates:
            raise EventDetectionError(
                "auto event detection failed: " + " | ".join(errors)
            )
        plausible_candidates.sort(key=lambda item: (item[0], item[1]))
        selected_score, selected_label, selected_validation, out = (
            plausible_candidates[0]
        )
        candidate_summary = ";".join(
            "{}:n={},score={:.5f}".format(label, len(frame), score)
            for score, label, _, frame in plausible_candidates
        )
        out["detection_method"] = selected_label
        out["auto_validation"] = selected_validation
        out["auto_candidate_score"] = float(selected_score)
        out["auto_candidates_evaluated"] = int(len(plausible_candidates))
        out["auto_candidate_summary"] = candidate_summary
    elif method == "annotations":
        out = _detect_annotations(dataloader, session_id, stim_pair, raw_file=raw_file, **kwargs)
        out = _restrict_events_to_stim_window(
            out,
            dataloader,
            session_id,
            stim_pair,
            stim_start=stim_start,
        )
    elif method == "trigger":
        df = signal_data if signal_data is not None else _load_signal(
            dataloader,
            session_id,
            stim_pair,
            raw_source,
            min_window_coverage,
            raw_file=raw_file,
            stim_start=stim_start,
        )
        df = _restrict_signal_to_stim_window(
            df,
            dataloader,
            session_id,
            stim_pair,
            stim_start=stim_start,
        )
        out = _detect_trigger_events(
            df,
            trigger_channels=parse_list(kwargs.get("trigger_channels")) if isinstance(kwargs.get("trigger_channels"), str) else kwargs.get("trigger_channels"),
            trigger_regex=kwargs.get("trigger_regex"),
            edge=kwargs.get("edge", "rising"),
            threshold=kwargs.get("threshold"),
            min_interval_s=float(kwargs.get("min_interval_s", 0.001)),
            merge_margin_s=float(kwargs.get("merge_margin_s", 0.001)),
            min_channel_consensus=int(kwargs.get("min_channel_consensus", 1)),
        )
    elif method in {"consensus", "adjacent"}:
        if stim_freq is None:
            try:
                row = dataloader.get_stim_row(session_id, stim_pair)
                stim_freq = float(row["stim_freq"]) if pd.notna(row.get("stim_freq")) else None
            except Exception:
                stim_freq = None
        df = signal_data if signal_data is not None else _load_signal(
            dataloader,
            session_id,
            stim_pair,
            raw_source,
            min_window_coverage,
            raw_file=raw_file,
            stim_start=stim_start,
        )
        df = _restrict_signal_to_stim_window(
            df,
            dataloader,
            session_id,
            stim_pair,
            stim_start=stim_start,
        )
        out = _detect_amplitude_events(
            df,
            stim_pair=stim_pair,
            method=method,
            stim_freq=stim_freq,
            min_peak_height=float(min_peak_height),
            consensus_factor=float(kwargs.get("consensus_factor", 4.0)),
            electrode_range=int(kwargs.get("electrode_range", 4)),
            min_adjacent_consensus=int(kwargs.get("min_adjacent_consensus", 1)),
            time_margin=float(kwargs.get("time_margin", 0.005)),
            polarity=kwargs.get("polarity", "abs"),
            adaptive_threshold=bool(kwargs.get("adaptive_threshold", False)),
            adaptive_floor_uv=float(kwargs.get("adaptive_floor_uv", 100.0)),
            adaptive_noise_z=float(kwargs.get("adaptive_noise_z", 12.0)),
        )
    elif method == "matched_filter":
        if stim_freq is None:
            try:
                row = dataloader.get_stim_row(session_id, stim_pair)
                stim_freq = float(row["stim_freq"]) if pd.notna(row.get("stim_freq")) else None
            except Exception:
                stim_freq = None
        df = signal_data if signal_data is not None else _load_signal(
            dataloader,
            session_id,
            stim_pair,
            raw_source,
            min_window_coverage,
            raw_file=raw_file,
            stim_start=stim_start,
        )
        df = _restrict_signal_to_stim_window(
            df,
            dataloader,
            session_id,
            stim_pair,
            stim_start=stim_start,
        )
        out = _detect_matched_filter_events(
            df,
            stim_pair=stim_pair,
            stim_freq=stim_freq,
            channel=kwargs.get("channel"),
            template=kwargs.get("template"),
            corr_threshold=float(kwargs.get("corr_threshold", 0.6)),
            template_window_s=tuple(kwargs.get("template_window_s", (0.002, 0.002))),
            min_peak_height_for_template=float(kwargs.get("min_peak_height_for_template", min_peak_height)),
        )
    elif method == "binary":
        path = kwargs.get("binary_event_path")
        if path is None:
            raise ValueError("binary method requires binary_event_path")
        binary = read_time_series_csv(path)
        row = dataloader.get_stim_row(session_id, stim_pair)
        _, out = create_events_from_binary_series(
            binary,
            start_datetime=row["stim_start"],
            end_datetime=row["stim_stop"],
            event_column=kwargs.get("event_column", 0),
            event_value=kwargs.get("event_value", 1),
        )
    else:
        raise ValueError(f"Unknown event detection method {method!r}")

    if (
        method != "auto"
        and bool(kwargs.get("regularize_sequence", True))
        and stim_freq is not None
    ):
        out = _regularize_event_sequence(
            out,
            stim_freq,
            tolerance=float(kwargs.get("cadence_tolerance", 0.12)),
        )
    if save_events:
        path = dataloader.get_event_path(session_id, stim_pair)
        ensure_dir(path.parent)
        out.to_csv(path, index=False)
    return out


def _detect_annotations(dataloader, session_id: str, stim_pair: str, **kwargs: Any) -> pd.DataFrame:
    try:
        import mne
    except Exception as exc:
        raise ImportError("Annotation event detection requires the optional 'mne' dependency") from exc
    raw_files = dataloader.get_raw_files(
        session_id,
        stim_pair,
        raw_file=kwargs.get("raw_file"),
    )
    if not raw_files:
        raise FileNotFoundError("No raw EDF/BDF files registered")
    frames = []
    for path in raw_files:
        if Path(path).suffix.lower() not in {".edf", ".bdf"}:
            continue
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        frames.append(
            _detect_annotation_events_from_raw(
                raw,
                include_regex=kwargs.get("include_regex"),
                exclude_regex=kwargs.get("exclude_regex"),
            )
        )
    if not frames:
        raise FileNotFoundError("No EDF/BDF files available for annotation detection")
    return pd.concat(frames, ignore_index=True).drop_duplicates("times").sort_values("times").reset_index(drop=True)


def process_events_for_session(
    dataloader,
    session_id: str,
    method: str = "consensus",
    stim_pairs: Iterable[str] | None = None,
    save_events: bool = True,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Detect and optionally save events for each stimulation pair in a session.

    Returns a mapping from stimulation-pair label to its normalized event table.
    """

    pairs = list(stim_pairs) if stim_pairs is not None else dataloader.stim_pairs(session_id)
    results: dict[str, pd.DataFrame] = {}
    for stim_pair in pairs:
        params = dict(kwargs)
        if method == "binary" and "binary_event_path" in params:
            path = str(params["binary_event_path"]).format(stim_pair=stim_pair, session_id=session_id)
            params["binary_event_path"] = path
        results[stim_pair] = detect_events_from_artifacts(
            dataloader,
            session_id,
            stim_pair,
            method=method,
            save_events=save_events,
            **params,
        )
    return results


@dataclass
class Events:
    """Session-event workflow bound to an ERPy ``DataLoader``.

    The facade creates canonical event tables for selected stimulation pairs
    and exposes the same processing path for local execution or a configured
    job runner. Event-source parameters remain explicit in each method call.
    """

    dataloader: Any

    PYTHON_SCRIPT: str | None = None
    BASH_SCRIPT: str | None = None

    def process(self, session_id: str, method: str = "consensus", stim_pairs: Iterable[str] | None = None, **kwargs: Any):
        """Detect events for the requested session and stimulation pairs."""

        return process_events_for_session(self.dataloader, session_id, method=method, stim_pairs=stim_pairs, **kwargs)

    @classmethod
    def install_standard_scripts(cls) -> None:
        """Reset event detection to ERPy's built-in execution path."""

        return None

    @classmethod
    def use_standard_scripts(cls, bash_type: str = "simple") -> None:
        """Select ERPy's built-in local event-processing hooks."""

        cls.PYTHON_SCRIPT = None
        cls.BASH_SCRIPT = None

    def queue_job(self, session_ids: Iterable[str] | None = None, backend: str | None = None, **kwargs: Any):
        """Process event jobs and return the runner plus execution summaries."""

        # Local execution uses the same code path and returns a summary string.
        runner = JobRunner(backend=backend, job_name=kwargs.get("job_name", "erpy_events"))
        sessions = list(session_ids or self.dataloader.session_ids)
        lines = []
        for session_id in sessions:
            result = self.process(session_id, **{k: v for k, v in kwargs.items() if k not in {"job_name", "queue", "numcpu", "memory"}})
            lines.append(f"{session_id}: {sum(len(v) for v in result.values())} events across {len(result)} stim pairs")
        return runner, "\n".join(lines), ""
