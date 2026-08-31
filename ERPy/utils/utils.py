from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


PATH_KEYS = {
    "rawdata_path",
    "procdata_path",
    "rawdata_meta_path",
    "stim_meta_path",
    "elec_meta_path",
    "event_path",
    "python_path",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir).expanduser() / path
    return path.resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_config_paths(explicit: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("ERPY_CONFIG")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(Path.home() / ".erpy" / "config.yaml")
    candidates.append(_repo_root() / "config.yaml")
    return candidates


def load_config(path: str | Path | None = None, required: bool = True) -> dict[str, Any]:
    """Load ERPy YAML configuration.

    Search order is ``$ERPY_CONFIG``, current working directory, ``~/.erpy``,
    then the editable checkout root. Relative paths are resolved relative to
    the YAML file that defined them.
    """

    for candidate in _candidate_config_paths(path):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            base = candidate.parent
            for key in PATH_KEYS:
                if key in config:
                    resolved = resolve_path(config.get(key), base)
                    config[key] = str(resolved) if resolved is not None else None
            config["_config_path"] = str(candidate.resolve())
            return config
    if required:
        searched = "\n".join(str(p) for p in _candidate_config_paths(path))
        raise FileNotFoundError(f"No ERPy config.yaml found. Searched:\n{searched}")
    return {}


def stable_hash(value: Any, length: int = 10) -> str:
    text = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def parse_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    if default is not None:
        return default
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_list(value: Any, sep: str = ",") -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [part.strip() for part in str(value).split(sep) if part.strip()]


def infer_sfreq(index_or_df: pd.Index | pd.DataFrame, default: float | None = None) -> float:
    index = index_or_df.index if isinstance(index_or_df, pd.DataFrame) else index_or_df
    if hasattr(index, "tz_localize"):
        try:
            diffs = pd.Series(index).diff().dropna().dt.total_seconds().to_numpy()
        except Exception:
            diffs = np.array([])
    else:
        vals = pd.to_numeric(pd.Series(index), errors="coerce").to_numpy()
        diffs = np.diff(vals[np.isfinite(vals)])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        if default is None:
            raise ValueError("Cannot infer sampling frequency from index")
        return float(default)
    return float(1.0 / np.median(diffs))


def nearest_index(index: pd.Index, when: Any) -> int:
    if isinstance(index, pd.DatetimeIndex):
        value = pd.Timestamp(when)
        pos = index.searchsorted(value)
        if pos <= 0:
            return 0
        if pos >= len(index):
            return len(index) - 1
        before = abs(index[pos - 1] - value)
        after = abs(index[pos] - value)
        return int(pos - 1 if before <= after else pos)
    values = pd.to_numeric(pd.Series(index), errors="coerce").to_numpy(dtype=float)
    target = float(when)
    return int(np.nanargmin(np.abs(values - target)))


def seconds_to_index(start: pd.Timestamp, seconds: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(start + pd.to_timedelta(seconds, unit="s"))


def read_time_series_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a signal CSV and use a time-like column as the index when present."""

    path = Path(path)
    df = pd.read_csv(path, nrows=nrows)
    if df.empty:
        return df
    lower = {str(c).lower(): c for c in df.columns}
    time_col = None
    for key in ("times", "time", "timestamp", "datetime", "date"):
        if key in lower:
            time_col = lower[key]
            break
    if time_col is None:
        first = df.columns[0]
        if str(first).lower().startswith("unnamed"):
            time_col = first
    if time_col is not None:
        raw = df.pop(time_col)
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.notna().mean() > 0.8:
            df.index = pd.DatetimeIndex(parsed)
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().mean() > 0.8:
                df.index = numeric.to_numpy(dtype=float)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception:
            pass
    return df


def write_time_series_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    out = df.copy()
    name = "times" if isinstance(out.index, pd.DatetimeIndex) else "time"
    out.insert(0, name, out.index)
    out.to_csv(path, index=False)
    return path


def normalize_columns(df: pd.DataFrame, aliases: dict[str, Iterable[str]]) -> pd.DataFrame:
    """Rename common metadata aliases to canonical names when available."""

    rename: dict[str, str] = {}
    lower = {str(c).lower(): c for c in df.columns}
    for canonical, names in aliases.items():
        if canonical in df.columns:
            continue
        for name in names:
            col = lower.get(name.lower())
            if col is not None:
                rename[col] = canonical
                break
    return df.rename(columns=rename)
