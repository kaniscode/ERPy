from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io import parse_brainvision_header
from .storage import load_continuous_npz, save_continuous_npz
from .utils.utils import (
    ensure_dir,
    infer_sfreq,
    load_config,
    normalize_columns,
    read_time_series_csv,
    resolve_path,
    seconds_to_index,
    write_time_series_csv,
)


RAW_ALIASES = {
    "patient_id": ["subject", "participant_id", "sub"],
    "session_id": ["session", "ses"],
    "raw_file": ["file", "filepath", "file_path", "path", "edf_file", "ieeg_file", "nwb_file", "nwb_path"],
    "start_time": ["recording_start", "start", "meas_date"],
    "stop_time": ["recording_stop", "end", "end_time"],
    "sampling_freq": ["sfreq", "sampling_frequency", "sample_rate", "SamplingFrequency"],
    "value_scale_to_uv": ["scale_to_uv", "signal_scale_to_uv", "voltage_scale_to_uv"],
}

STIM_ALIASES = {
    "patient_id": ["subject", "participant_id", "sub"],
    "session_id": ["session", "ses"],
    "stim_pair": ["stim", "stimulation_pair", "stimulus_pair", "electrical_stimulation_site"],
    "stim_start": ["start", "onset", "start_time", "stim_onset"],
    "stim_stop": ["stop", "end", "end_time", "stim_end"],
    "stim_freq": ["frequency", "stimulation_frequency", "repetition_frequency"],
}

ELEC_ALIASES = {
    "patient_id": ["subject", "participant_id", "sub"],
    "session_id": ["session", "ses"],
    "elec_label": ["name", "channel", "channel_name", "label", "electrode"],
    "anat_label": ["region", "anat", "anatomy", "destrieux_label"],
    "bad_channel": ["bad", "is_bad", "exclude"],
}


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


class DataLoader:
    """Configuration and metadata access for one ERPy patient."""

    def __init__(
        self,
        patient_id: str,
        config_path: str | Path | None = None,
        load_data: bool = False,
        create_tree: bool = True,
        verbose: bool = True,
    ) -> None:
        self.patient_id = str(patient_id)
        self.verbose = verbose
        self.config = load_config(config_path)
        self.CONFIG_PATH = self.config.get("_config_path")
        self.RAWDATA_PATH = str(resolve_path(self.config.get("rawdata_path")) or Path.cwd())
        self.PROCDATA_PATH = str(resolve_path(self.config.get("procdata_path")) or Path.cwd() / "procdata")
        self.RAW_META_PATH = str(resolve_path(self.config.get("rawdata_meta_path")) or "")
        self.STIM_META_PATH = str(resolve_path(self.config.get("stim_meta_path")) or "")
        self.ELEC_META_PATH = str(resolve_path(self.config.get("elec_meta_path")) or "")
        self.EVENT_PATH = str(resolve_path(self.config.get("event_path")) or "")
        self.python_path = self.config.get("python_path")
        self.RAW_CACHE_FORMAT = str(self.config.get("raw_cache_format", "csv")).strip().lower()
        self.RAW_CACHE_DTYPE = str(self.config.get("raw_cache_dtype", "float32")).strip()
        staging = resolve_path(self.config.get("raw_staging_path"))
        self.RAW_STAGING_PATH = Path(staging) if staging is not None else None
        self._stim_memory_cache_key: tuple[str, str, str] | None = None
        self._stim_memory_cache_df: pd.DataFrame | None = None

        self.RAW_META_DF = self._load_metadata(self.RAW_META_PATH, RAW_ALIASES, ["patient_id", "session_id", "raw_file"])
        self.STIM_META_DF = self._load_metadata(
            self.STIM_META_PATH,
            STIM_ALIASES,
            ["patient_id", "session_id", "stim_pair", "stim_start", "stim_stop", "stim_freq"],
        )
        self.ELEC_META_DF = self._load_metadata(
            self.ELEC_META_PATH,
            ELEC_ALIASES,
            ["patient_id", "session_id", "elec_label", "anat_label", "bad_channel"],
        )

        self.raw_meta = self._filter_patient(self.RAW_META_DF)
        self.stim_meta = self._filter_patient(self.STIM_META_DF)
        self.elec_meta = self._filter_patient(self.ELEC_META_DF)

        self.session_ids = self._resolve_session_ids()
        self._validate_patient()
        self.PATIENT_PATH = str(Path(self.PROCDATA_PATH) / self.patient_id)
        self.SESSION_PATHS = {
            sid: str(Path(self.PATIENT_PATH) / str(sid)) for sid in self.session_ids
        }
        if create_tree:
            self.create_session_tree()
        if load_data:
            self.prepare_raw_windows()

    @classmethod
    def from_bids(
        cls,
        bids_root: str | Path,
        subject: str,
        output_root: str | Path | None = None,
        session: str | None = None,
        task: str | None = None,
        patient_id: str | None = None,
        copy_raw: bool = False,
        derivatives: bool = True,
        **kwargs: Any,
    ) -> "DataLoader":
        """Import one BIDS iEEG subject and return its configured data loader.

        ERPy metadata tables and configuration are created under
        ``output_root``; raw recordings remain in place unless ``copy_raw`` is
        enabled.
        """

        from .bids import import_bids_project, normalize_subject

        patient_id = patient_id or normalize_subject(subject)
        output_root = Path(output_root or Path(bids_root).expanduser().resolve() / "erpy_project")
        result = import_bids_project(
            bids_root=bids_root,
            output_root=output_root,
            subject=subject,
            session=session,
            task=task,
            patient_id=patient_id,
            copy_raw=copy_raw,
            derivatives=derivatives,
        )
        return cls(patient_id, config_path=result.config_path, **kwargs)

    @classmethod
    def from_nwb(
        cls,
        nwb_path: str | Path,
        output_root: str | Path | None = None,
        patient_id: str | None = None,
        session_id: str | None = None,
        stim_pair: str | None = None,
        stim_times: Any | None = None,
        stim_duration_s: float = 0.001,
        series_name: str | None = None,
        copy_raw: bool = False,
        **kwargs: Any,
    ) -> "DataLoader":
        """Create a minimal ERPy project around one NWB file.

        Parameters
        ----------
        nwb_path
            Local NWB file.
        output_root
            Folder where ERPy metadata, config, and processed outputs are written.
        patient_id, session_id
            Optional ERPy identifiers. When omitted, ERPy uses the NWB subject
            and session identifiers when available.
        stim_pair, stim_times
            Optional stimulation-site name and event times. Numeric times are
            interpreted as seconds relative to NWB session start; datetimes are
            preserved as absolute event times. If omitted, ERPy tries to infer a
            stimulation table from NWB trials.
        series_name
            Name of the NWB ElectricalSeries to load. If omitted, the first
            ElectricalSeries is used.
        copy_raw
            Copy the NWB into the ERPy project. Defaults to keeping large NWB
            files in place.
        """

        NWBHDF5IO, ElectricalSeries = _require_pynwb()
        nwb_path = Path(nwb_path).expanduser().resolve()
        if not nwb_path.exists():
            raise FileNotFoundError(nwb_path)
        output_root = Path(output_root or nwb_path.parent / "erpy_nwb_project").expanduser().resolve()
        raw_dir = ensure_dir(output_root / "raw")
        proc_dir = ensure_dir(output_root / "procdata")
        meta_dir = ensure_dir(output_root / "metadata")

        with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
            nwbfile = io.read()
            series_path, series = _pick_nwb_electrical_series(nwbfile, ElectricalSeries, series_name)
            patient_id = (patient_id or _nwb_subject_id(nwbfile) or nwb_path.stem).replace("sub-", "")
            session_id = str(session_id or getattr(nwbfile, "session_id", None) or "A").replace("ses-", "")
            session_start = _coerce_timestamp(getattr(nwbfile, "session_start_time", None))
            sfreq = _infer_nwb_sfreq(series)
            elec_meta = _nwb_electrode_metadata(nwbfile, patient_id, session_id, series=series)
            inferred_pair, inferred_times = _extract_nwb_stim_events(nwbfile, stim_pair=stim_pair)
            if stim_times is None and inferred_times is not None:
                stim_pair = stim_pair or inferred_pair
                stim_times = inferred_times

        erpy_raw_file = raw_dir / nwb_path.name if copy_raw else nwb_path
        if copy_raw and not erpy_raw_file.exists():
            shutil.copy2(nwb_path, erpy_raw_file)
        raw_file_value = erpy_raw_file.name if copy_raw else nwb_path.name
        raw_root = raw_dir if copy_raw else nwb_path.parent

        raw_meta = pd.DataFrame(
            [
                {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "raw_file": raw_file_value,
                    "start_time": session_start.isoformat() if session_start is not None else "",
                    "stop_time": "",
                    "sampling_freq": sfreq or "",
                    "nwb_series": series_path,
                    "source_nwb_file": str(nwb_path),
                }
            ]
        )
        stim_meta = _nwb_stim_metadata(
            patient_id=patient_id,
            session_id=session_id,
            stim_pair=stim_pair,
            stim_times=stim_times,
            stim_duration_s=stim_duration_s,
            source_nwb_file=str(nwb_path),
        )

        raw_meta_path = meta_dir / "raw_metadata.csv"
        stim_meta_path = meta_dir / "stim_metadata.csv"
        elec_meta_path = meta_dir / "electrode_metadata.csv"
        raw_meta.to_csv(raw_meta_path, index=False)
        stim_meta.to_csv(stim_meta_path, index=False)
        elec_meta.to_csv(elec_meta_path, index=False)

        config = {
            "rawdata_path": str(raw_root),
            "procdata_path": str(proc_dir),
            "rawdata_meta_path": str(raw_meta_path),
            "stim_meta_path": str(stim_meta_path),
            "elec_meta_path": str(elec_meta_path),
            "session_ids": {patient_id: [session_id]},
        }
        config_path = output_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        if not stim_meta.empty and stim_times is not None:
            event_path = proc_dir / patient_id / session_id / "events" / f"{stim_pair}_events.csv"
            ensure_dir(event_path.parent)
            _nwb_event_table(stim_times).to_csv(event_path, index=False)

        return cls(patient_id, config_path=config_path, **kwargs)

    def _load_metadata(
        self,
        path: str,
        aliases: dict[str, list[str]],
        default_columns: list[str],
    ) -> pd.DataFrame:
        if not path or not Path(path).exists():
            return _empty(default_columns)
        df = pd.read_csv(path, dtype={"patient_id": "string", "session_id": "string"})
        df = normalize_columns(df, aliases)
        for col in default_columns:
            if col not in df.columns:
                df[col] = np.nan
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str).str.replace("^sub-", "", regex=True)
        if "session_id" in df.columns:
            df["session_id"] = df["session_id"].fillna("").astype(str).str.replace("^ses-", "", regex=True)
        return df

    def _filter_patient(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "patient_id" not in df.columns:
            return df.copy()
        pid = self.patient_id.replace("sub-", "")
        return df[df["patient_id"].astype(str).str.replace("^sub-", "", regex=True) == pid].copy()

    def _resolve_session_ids(self) -> list[str]:
        configured = self.config.get("session_ids", {}) or {}
        values = configured.get(self.patient_id) or configured.get(f"sub-{self.patient_id}") or []
        if isinstance(values, str):
            values = [values]
        sessions = [str(v).replace("ses-", "") for v in values]
        for df in (self.raw_meta, self.stim_meta, self.elec_meta):
            if not df.empty and "session_id" in df.columns:
                sessions.extend(str(v).replace("ses-", "") for v in df["session_id"].dropna().unique())
        return sorted({s for s in sessions if s and s.lower() != "nan"})

    def _validate_patient(self) -> None:
        known = set()
        for df in (self.RAW_META_DF, self.STIM_META_DF, self.ELEC_META_DF):
            if not df.empty and "patient_id" in df.columns:
                known.update(df["patient_id"].dropna().astype(str).str.replace("^sub-", "", regex=True))
        if known and self.patient_id.replace("sub-", "") not in known:
            raise ValueError(f"Unknown patient_id {self.patient_id!r}. Known patients: {sorted(known)}")

    def create_session_tree(self) -> None:
        """Create the standard raw, event, epoch, analysis, and log folders."""

        for session_path in self.SESSION_PATHS.values():
            for sub in ("raw", "events", "processed", "epochs", "analysis", "logs"):
                ensure_dir(Path(session_path) / sub)

    def get_session_path(self, session_id: str, kind: str | None = None) -> Path:
        """Return a session directory or one named subdirectory within it."""

        sid = str(session_id).replace("ses-", "")
        if sid not in self.SESSION_PATHS:
            self.SESSION_PATHS[sid] = str(Path(self.PATIENT_PATH) / sid)
        path = Path(self.SESSION_PATHS[sid])
        return path / kind if kind else path

    def get_analysis_path(self, session_id: str | None = None) -> Path:
        """Return and create the patient- or session-level analysis directory."""

        if session_id is None:
            return ensure_dir(Path(self.PATIENT_PATH) / "analysis")
        return ensure_dir(self.get_session_path(session_id, "analysis"))

    def stim_pairs(self, session_id: str | None = None) -> list[str]:
        """Return sorted stimulation-pair labels, optionally for one session."""

        df = self.stim_meta
        if session_id is not None and not df.empty:
            df = df[df["session_id"].astype(str) == str(session_id).replace("ses-", "")]
        if df.empty or "stim_pair" not in df.columns:
            return []
        return sorted(df["stim_pair"].dropna().astype(str).unique())

    def get_stim_row(
        self,
        session_id: str,
        stim_pair: str,
        *,
        stim_start: Any | None = None,
    ) -> pd.Series:
        """Return the metadata row matching a session and stimulation pair.

        When repeated acquisitions exist, ``stim_start`` selects the closest
        registered start time.
        """

        sid = str(session_id).replace("ses-", "")
        mask = (self.stim_meta["session_id"].astype(str) == sid) & (
            self.stim_meta["stim_pair"].astype(str) == str(stim_pair)
        )
        rows = self.stim_meta[mask]
        if rows.empty:
            raise KeyError(f"No stimulation metadata for session={session_id!r}, stim_pair={stim_pair!r}")
        if stim_start is not None and len(rows) > 1 and "stim_start" in rows.columns:
            wanted = pd.Timestamp(stim_start)
            starts = pd.to_datetime(rows["stim_start"], errors="coerce")
            distance = (starts - wanted).abs()
            if distance.notna().any():
                return rows.loc[distance.idxmin()]
        return rows.iloc[0]

    def get_sampling_freq(self, session_id: str | None = None, stim_pair: str | None = None) -> float | None:
        """Resolve the registered sampling frequency for a run or session."""

        if stim_pair is not None and session_id is not None:
            try:
                row = self.get_stim_row(session_id, stim_pair)
                for key in ("sampling_freq", "sfreq"):
                    if key in row and pd.notna(row[key]):
                        return float(row[key])
            except Exception:
                pass
        df = self.raw_meta
        if session_id is not None and not df.empty:
            df = df[df["session_id"].astype(str) == str(session_id).replace("ses-", "")]
        if "sampling_freq" in df.columns and df["sampling_freq"].notna().any():
            return float(df["sampling_freq"].dropna().iloc[0])
        return None

    def get_raw_files(
        self,
        session_id: str | None = None,
        stim_pair: str | None = None,
        *,
        raw_file: str | Path | None = None,
        stim_start: Any | None = None,
    ) -> list[Path]:
        """Resolve raw recording paths for a session or stimulation acquisition.

        ``raw_file`` and ``stim_start`` disambiguate repeated acquisitions when
        the metadata table contains more than one candidate.
        """

        df = self.raw_meta
        if session_id is not None and not df.empty:
            df = df[df["session_id"].astype(str) == str(session_id).replace("ses-", "")]
        if stim_pair is not None and not df.empty and "stim_pair" in df.columns:
            pair_mask = df["stim_pair"].astype(str) == str(stim_pair)
            if pair_mask.any():
                df = df[pair_mask]
        if raw_file is not None and not df.empty and "raw_file" in df.columns:
            wanted = Path(str(raw_file))
            exact = df["raw_file"].astype(str) == str(raw_file)
            by_name = df["raw_file"].astype(str).map(lambda value: Path(value).name == wanted.name)
            selected = exact | by_name
            if not selected.any():
                raise FileNotFoundError(
                    f"Raw acquisition {raw_file!r} is not registered for "
                    f"{self.patient_id}/{session_id}/{stim_pair}"
                )
            df = df[selected]
        if (
            raw_file is None
            and stim_pair is not None
            and not df.empty
            and {"start_time", "stop_time"}.issubset(df.columns)
        ):
            try:
                stim = self.get_stim_row(
                    session_id or df.iloc[0]["session_id"],
                    stim_pair,
                    stim_start=stim_start,
                )
                start = pd.Timestamp(stim["stim_start"])
                stop = pd.Timestamp(stim["stim_stop"])
                starts = pd.to_datetime(df["start_time"], errors="coerce")
                stops = pd.to_datetime(df["stop_time"], errors="coerce")
                # Recording stop times are exclusive. A file whose final
                # interval boundary equals the requested start has no samples
                # in the requested window.
                overlap = (starts <= stop) & (stops > start)
                if overlap.any():
                    df = df[overlap]
            except Exception:
                pass
        files: list[Path] = []
        for value in df.get("raw_file", pd.Series(dtype=str)).dropna().astype(str):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path(self.RAWDATA_PATH) / path
            files.append(path.resolve())
        return files

    def _raw_rows_for_path(self, path: str | Path) -> pd.DataFrame:
        if self.raw_meta.empty or "raw_file" not in self.raw_meta.columns:
            return self.raw_meta.iloc[0:0]
        target = Path(path).expanduser()
        target_name = target.name

        def matches(value: Any) -> bool:
            if pd.isna(value):
                return False
            candidate = Path(str(value)).expanduser()
            if candidate.name == target_name:
                return True
            if not candidate.is_absolute():
                candidate = Path(self.RAWDATA_PATH) / candidate
            try:
                return candidate.resolve() == target.resolve()
            except OSError:
                return False

        return self.raw_meta[self.raw_meta["raw_file"].map(matches)]

    def _value_scale_to_uv_for_path(self, path: str | Path) -> float:
        rows = self._raw_rows_for_path(path)
        if "value_scale_to_uv" in rows.columns:
            values = pd.to_numeric(rows["value_scale_to_uv"], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0])
        return float(self.config.get("raw_value_scale_to_uv", 1.0))

    def _value_scale_to_uv_for_stim(self, session_id: str, stim_pair: str) -> float:
        rows = self.raw_meta
        if not rows.empty:
            rows = rows[rows["session_id"].astype(str) == str(session_id).replace("ses-", "")]
            if "stim_pair" in rows.columns:
                paired = rows[rows["stim_pair"].astype(str) == str(stim_pair)]
                if not paired.empty:
                    rows = paired
        if "value_scale_to_uv" in rows.columns:
            values = pd.to_numeric(rows["value_scale_to_uv"], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0])
        return float(self.config.get("raw_value_scale_to_uv", 1.0))

    @staticmethod
    def _apply_value_scale_to_uv(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        numeric = out.select_dtypes(include=[np.number]).columns
        if float(factor) != 1.0 and len(numeric):
            out.loc[:, numeric] = out.loc[:, numeric] * float(factor)
        out.attrs.update(df.attrs)
        out.attrs["signal_units"] = "uV"
        out.attrs["value_scale_to_uv_applied"] = float(factor)
        return out

    @staticmethod
    def _raw_cache_metadata_path(path: str | Path) -> Path:
        path = Path(path)
        return path.with_suffix(path.suffix + ".meta.json")

    def _write_stim_cache(self, df: pd.DataFrame, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() == ".npz":
            save_continuous_npz(
                df,
                path,
                dtype=self.RAW_CACHE_DTYPE,
                compressed=False,
                metadata={"cache_role": "temporary_full_rate"},
            )
        else:
            path = write_time_series_csv(df, path)
        metadata = {
            "signal_units": str(df.attrs.get("signal_units") or "uV"),
            "value_scale_to_uv": 1.0,
            "sfreq": float(df.attrs["sfreq"]) if df.attrs.get("sfreq") is not None else None,
        }
        metadata_path = self._raw_cache_metadata_path(path)
        temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary, metadata_path)
        return path

    def get_event_path(self, session_id: str, stim_pair: str) -> Path:
        """Return the conventional CSV path for a detected event table."""

        return self.get_session_path(session_id, "events") / f"{stim_pair}_events.csv"

    @staticmethod
    def _raw_cache_discriminator(raw_file: str | Path | None) -> str:
        if raw_file is None:
            return ""
        value = str(raw_file)
        readable = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(value).stem).strip("._")
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        return f"__{readable[:48]}_{digest}"

    def get_raw_csv_path(
        self,
        session_id: str,
        stim_pair: str,
        *,
        raw_file: str | Path | None = None,
    ) -> Path:
        """Return the acquisition-specific path for a text raw-data cache."""

        discriminator = self._raw_cache_discriminator(raw_file)
        return self.get_session_path(session_id, "raw") / f"{stim_pair}_{session_id}{discriminator}.csv"

    def get_raw_cache_path(
        self,
        session_id: str,
        stim_pair: str,
        *,
        raw_file: str | Path | None = None,
    ) -> Path:
        """Return the configured binary or text cache path for an acquisition."""

        if self.RAW_CACHE_FORMAT in {"npz", "binary", "continuous_npz"}:
            discriminator = self._raw_cache_discriminator(raw_file)
            return self.get_session_path(session_id, "raw") / f"{stim_pair}_{session_id}{discriminator}.npz"
        return self.get_raw_csv_path(session_id, stim_pair, raw_file=raw_file)

    def _raw_cache_candidates(
        self,
        session_id: str,
        stim_pair: str,
        *,
        raw_file: str | Path | None,
        allow_legacy_cache: bool,
    ) -> list[Path]:
        candidates = [
            self.get_raw_cache_path(session_id, stim_pair, raw_file=raw_file),
            self.get_raw_csv_path(session_id, stim_pair, raw_file=raw_file),
        ]
        if allow_legacy_cache and raw_file is not None:
            candidates.extend(
                [
                    self.get_raw_cache_path(session_id, stim_pair),
                    self.get_raw_csv_path(session_id, stim_pair),
                ]
            )
        return list(dict.fromkeys(candidates))

    def clear_memory_cache(self) -> None:
        """Discard the currently retained stimulation window from memory."""

        self._stim_memory_cache_key = None
        self._stim_memory_cache_df = None

    def has_stim_cache(
        self,
        session_id: str,
        stim_pair: str,
        *,
        raw_file: str | Path | None = None,
        allow_legacy_cache: bool = False,
    ) -> bool:
        """Return whether a matching on-disk stimulation cache is available."""

        return any(
            path.is_file()
            for path in self._raw_cache_candidates(
                session_id,
                stim_pair,
                raw_file=raw_file,
                allow_legacy_cache=allow_legacy_cache,
            )
        )

    def prefetch_stim_data(
        self,
        session_id: str,
        stim_pair: str,
        *,
        raw_file: str | Path,
        stim_start: Any | None = None,
        stop_event: Any | None = None,
    ) -> list[Path]:
        """Stage registered text exports locally using large sequential reads."""

        staged: list[Path] = []
        for path in self.get_raw_files(
            session_id,
            stim_pair,
            raw_file=raw_file,
            stim_start=stim_start,
        ):
            staged_path, temporary = self._stage_raw_file(path, stop_event=stop_event)
            if temporary is not None:
                staged.append(staged_path)
        return staged

    def remove_stim_cache(
        self,
        session_id: str,
        stim_pair: str,
        *,
        raw_file: str | Path | None = None,
        include_legacy: bool = False,
    ) -> list[Path]:
        """Remove matching cache files and return the paths that were deleted."""

        removed: list[Path] = []
        for path in self._raw_cache_candidates(
            session_id,
            stim_pair,
            raw_file=raw_file,
            allow_legacy_cache=include_legacy,
        ):
            for candidate in (path, self._raw_cache_metadata_path(path)):
                if candidate.exists():
                    candidate.unlink()
                    removed.append(candidate)
        key = self._memory_cache_key(session_id, stim_pair, raw_file)
        if self._stim_memory_cache_key == key:
            self.clear_memory_cache()
        return removed

    @staticmethod
    def _memory_cache_key(
        session_id: str,
        stim_pair: str,
        raw_file: str | Path | None,
    ) -> tuple[str, str, str]:
        return (
            str(session_id).replace("ses-", ""),
            str(stim_pair),
            str(raw_file or ""),
        )

    def _stage_raw_file(
        self,
        path: Path,
        *,
        stop_event: Any | None = None,
    ) -> tuple[Path, Path | None]:
        name = path.name.lower()
        if (
            self.RAW_STAGING_PATH is None
            or not name.endswith((".csv", ".tsv", ".csv.gz", ".tsv.gz"))
            or not path.is_file()
        ):
            return path, None
        stage_dir = self.RAW_STAGING_PATH / self.patient_id
        stage_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
        staged = stage_dir / f"{digest}_{path.name}"
        try:
            source_size = path.stat().st_size
        except OSError:
            return path, None
        if staged.is_file() and staged.stat().st_size == source_size:
            return staged, staged

        partial = staged.with_suffix(staged.suffix + ".partial")
        partial.unlink(missing_ok=True)
        try:
            with path.open("rb") as source, partial.open("wb") as target:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError(f"Staging cancelled for {path}")
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if partial.stat().st_size != source_size:
                raise OSError(
                    f"Staged raw file is incomplete: expected {source_size} bytes, "
                    f"found {partial.stat().st_size}"
                )
            os.replace(partial, staged)
        finally:
            partial.unlink(missing_ok=True)
        return staged, staged

    def bad_channels(self, session_id: str | None = None) -> list[str]:
        """Return electrode labels explicitly marked bad in project metadata."""

        df = self.elec_meta
        if df.empty or "bad_channel" not in df.columns:
            return []
        if session_id is not None and "session_id" in df.columns:
            sid = str(session_id).replace("ses-", "")
            df = df[(df["session_id"].astype(str) == sid) | (df["session_id"].astype(str) == "")]
        bad = df["bad_channel"].astype(str).str.lower().isin({"1", "true", "yes", "bad", "exclude"})
        return df.loc[bad, "elec_label"].dropna().astype(str).tolist()

    def prepare_raw_windows(self, overwrite: bool = False) -> list[Path]:
        """Precompute registered stimulation-window caches for all sessions."""

        outputs: list[Path] = []
        for session_id in self.session_ids:
            for stim_pair in self.stim_pairs(session_id):
                path = self.get_raw_cache_path(session_id, stim_pair)
                if path.exists() and not overwrite:
                    outputs.append(path)
                    continue
                try:
                    df = self.load_stim_data(session_id, stim_pair, source="raw", save_csv=False)
                except Exception:
                    continue
                outputs.append(self._write_stim_cache(df, path))
        return outputs

    def load_stim_data(
        self,
        session_id: str,
        stim_pair: str,
        source: str = "auto",
        save_csv: bool = True,
        nrows: int | None = None,
        raw_file: str | Path | None = None,
        stim_start: Any | None = None,
        allow_legacy_cache: bool = False,
    ) -> pd.DataFrame:
        """Load a stimulation window as a time-indexed DataFrame.

        ``source='auto'`` prefers the local per-acquisition cache and otherwise
        reads a registered raw file. ``raw_file`` disambiguates repeated
        stimulation acquisitions that share a session and electrode pair.
        """

        memory_key = self._memory_cache_key(session_id, stim_pair, raw_file)
        if (
            nrows is None
            and source in {"auto", "csv", "cache"}
            and self._stim_memory_cache_key == memory_key
            and self._stim_memory_cache_df is not None
        ):
            return self._stim_memory_cache_df

        cache_candidates = self._raw_cache_candidates(
            session_id,
            stim_pair,
            raw_file=raw_file,
            allow_legacy_cache=allow_legacy_cache,
        )
        cache_path = next((path for path in cache_candidates if path.is_file()), cache_candidates[0])
        if source in {"auto", "csv", "cache"} and cache_path.exists():
            if cache_path.suffix.lower() == ".npz":
                df = load_continuous_npz(cache_path, nrows=nrows)
            else:
                df = read_time_series_csv(cache_path, nrows=nrows)
            cache_metadata_path = self._raw_cache_metadata_path(cache_path)
            if cache_metadata_path.exists():
                try:
                    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    cache_metadata = {}
                scale = float(cache_metadata.get("value_scale_to_uv", 1.0))
            else:
                # Caches created before unit metadata was introduced contain
                # the same values as their registered raw export.
                scale = self._value_scale_to_uv_for_stim(session_id, stim_pair)
            df = self._apply_value_scale_to_uv(df, scale)
            if "sfreq" not in df.attrs:
                try:
                    df.attrs["sfreq"] = infer_sfreq(df, self.get_sampling_freq(session_id, stim_pair))
                except Exception:
                    pass
            if nrows is None:
                self._stim_memory_cache_key = memory_key
                self._stim_memory_cache_df = df
            return df
        if source in {"csv", "cache"}:
            raise FileNotFoundError(cache_path)

        raw_files = self.get_raw_files(
            session_id,
            stim_pair,
            raw_file=raw_file,
            stim_start=stim_start,
        )
        if not raw_files:
            raise FileNotFoundError(f"No raw files registered for {self.patient_id}/{session_id}/{stim_pair}")
        frames: list[pd.DataFrame] = []
        for path in raw_files:
            staged_path: Path | None = None
            loaded = False
            try:
                load_path, staged_path = self._stage_raw_file(path)
                suffix = load_path.suffix.lower()
                name = load_path.name.lower()
                if suffix in {".csv", ".tsv"} or name.endswith((".csv.gz", ".tsv.gz")):
                    frame = read_time_series_csv(load_path, nrows=nrows)
                    frame = self._apply_value_scale_to_uv(frame, self._value_scale_to_uv_for_path(path))
                elif suffix in {".edf", ".bdf"}:
                    frame = self._load_edf_window(load_path, session_id, stim_pair)
                elif suffix == ".vhdr":
                    frame = self._load_brainvision_window(load_path, session_id, stim_pair)
                elif suffix == ".nwb":
                    frame = self._load_nwb_window(load_path, session_id, stim_pair)
                else:
                    continue
                frames.append(frame)
                loaded = True
            finally:
                if loaded and staged_path is not None:
                    staged_path.unlink(missing_ok=True)
        if not frames:
            raise ValueError(f"Raw files were found but none could be loaded for {stim_pair}")
        applied_scales = {
            float(frame.attrs.get("value_scale_to_uv_applied", 1.0))
            for frame in frames
        }
        if len(applied_scales) != 1:
            raise ValueError(
                "Raw files for one stimulation acquisition use inconsistent "
                f"value_scale_to_uv factors: {sorted(applied_scales)}"
            )
        df = pd.concat(frames).sort_index()
        df = df.loc[:, ~df.columns.duplicated()]
        try:
            df.attrs["sfreq"] = infer_sfreq(df, self.get_sampling_freq(session_id, stim_pair))
        except Exception:
            sf = self.get_sampling_freq(session_id, stim_pair)
            if sf:
                df.attrs["sfreq"] = sf
        df.attrs["signal_units"] = "uV"
        df.attrs["value_scale_to_uv_applied"] = applied_scales.pop()
        if save_csv:
            self._write_stim_cache(df, cache_candidates[0])
        if nrows is None:
            self._stim_memory_cache_key = memory_key
            self._stim_memory_cache_df = df
        return df

    def _load_nwb_window(self, path: Path, session_id: str, stim_pair: str) -> pd.DataFrame:
        NWBHDF5IO, ElectricalSeries = _require_pynwb()
        margin = float(self.config.get("bids_window_margin_s", self.config.get("nwb_window_margin_s", 1.0)))
        stim = self.get_stim_row(session_id, stim_pair)
        series_name = self._nwb_series_name_for_path(path)

        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwbfile = io.read()
            series_path, series = _pick_nwb_electrical_series(nwbfile, ElectricalSeries, series_name)
            sfreq = _infer_nwb_sfreq(series) or self.get_sampling_freq(session_id, stim_pair)
            if sfreq is None:
                raise ValueError(f"Cannot infer sampling frequency for NWB ElectricalSeries {series_path!r}")
            sfreq = float(sfreq)
            time_axis = _nwb_time_axis(series)
            total_samples = int(series.data.shape[time_axis])
            base_time = self._recording_start_for_path(path) or _coerce_timestamp(getattr(nwbfile, "session_start_time", None))
            timestamps = _nwb_timestamps(series)
            if timestamps is not None and len(timestamps) == total_samples:
                start_s, stop_s, index_kind, base_time = self._window_seconds_from_stim_values(
                    stim.get("stim_start"),
                    stim.get("stim_stop"),
                    margin,
                    base_time=base_time,
                )
                start_sample = int(np.searchsorted(timestamps, start_s, side="left"))
                stop_sample = int(np.searchsorted(timestamps, stop_s, side="right"))
                stop_sample = min(stop_sample, total_samples)
            else:
                series_start = _nwb_series_starting_time(series)
                start_sample, stop_sample, index_kind, base_time = self._sample_window_from_stim_values(
                    stim.get("stim_start"),
                    stim.get("stim_stop"),
                    sfreq,
                    total_samples,
                    margin,
                    base_time=base_time,
                    time_offset_s=series_start,
                )
            if stop_sample <= start_sample:
                raise ValueError(
                    f"Stimulation window {stim_pair!r} does not overlap NWB ElectricalSeries {series_path!r}"
                )

            data = _slice_nwb_data(series.data, start_sample, stop_sample, time_axis)
            data = _scale_nwb_electrical_data(series, data)
            labels = _nwb_series_channel_labels(series, data.shape[1])
            if len(labels) != data.shape[1]:
                labels = [f"CH{i + 1}" for i in range(data.shape[1])]
            labels = _unique_labels([_clean_channel_label(ch) for ch in labels])
            index = _nwb_window_index(series, start_sample, stop_sample, sfreq, index_kind, base_time)

        df = pd.DataFrame(data, index=index, columns=labels)
        df.attrs["sfreq"] = sfreq
        df.attrs["source_file"] = str(path)
        df.attrs["nwb_series"] = series_path
        df.attrs["signal_units"] = "uV"
        df.attrs["value_scale_to_uv_applied"] = 1.0
        return df

    def _nwb_series_name_for_path(self, path: Path) -> str | None:
        if self.raw_meta.empty:
            return self.config.get("nwb_series")
        rows = self.raw_meta[self.raw_meta.get("raw_file", pd.Series(dtype=str)).astype(str).map(lambda v: Path(v).name == path.name)]
        if rows.empty:
            return self.config.get("nwb_series")
        row = rows.iloc[0]
        for key in ("nwb_series", "nwb_acquisition", "series_name"):
            if key in row and pd.notna(row[key]) and str(row[key]).strip():
                return str(row[key]).strip()
        return self.config.get("nwb_series")

    def _recording_start_for_path(self, path: Path) -> pd.Timestamp | None:
        if self.raw_meta.empty or "start_time" not in self.raw_meta.columns:
            return None
        rows = self.raw_meta[self.raw_meta.get("raw_file", pd.Series(dtype=str)).astype(str).map(lambda v: Path(v).name == path.name)]
        values = rows["start_time"] if not rows.empty else self.raw_meta["start_time"]
        parsed = pd.to_datetime(values, errors="coerce").dropna()
        if parsed.empty:
            return None
        return _coerce_timestamp(parsed.iloc[0])

    def _load_edf_window(self, path: Path, session_id: str, stim_pair: str) -> pd.DataFrame:
        try:
            import mne
        except Exception as exc:
            raise ImportError("EDF loading requires the optional 'mne' dependency") from exc

        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data().T * 1e6
        channels = [_clean_channel_label(ch) for ch in raw.ch_names]
        meas_date = raw.info.get("meas_date")
        if meas_date is not None:
            start = pd.Timestamp(meas_date).tz_localize(None)
        else:
            row = self.raw_meta[self.raw_meta["raw_file"].astype(str).str.endswith(path.name)]
            start = pd.Timestamp(row.iloc[0]["start_time"]) if not row.empty and pd.notna(row.iloc[0].get("start_time")) else pd.Timestamp("1970-01-01")
        times = seconds_to_index(start, np.arange(data.shape[0]) / sfreq)
        df = pd.DataFrame(data, index=times, columns=channels)
        df.attrs["sfreq"] = sfreq
        df.attrs["signal_units"] = "uV"
        df.attrs["value_scale_to_uv_applied"] = 1.0
        try:
            stim = self.get_stim_row(session_id, stim_pair)
            start_t = pd.Timestamp(stim["stim_start"])
            stop_t = pd.Timestamp(stim["stim_stop"])
            df = df.loc[(df.index >= start_t) & (df.index <= stop_t)]
            df.attrs["sfreq"] = sfreq
        except Exception:
            pass
        return df

    def _load_brainvision_window(self, path: Path, session_id: str, stim_pair: str) -> pd.DataFrame:
        header = parse_brainvision_header(path.read_text(encoding="utf-8", errors="ignore"))
        sfreq = float(header["sfreq"])
        n_channels = int(header["n_channels"])
        channels = list(header["channels"])
        resolutions = np.asarray(header["resolutions"], dtype=float)
        data_file = header.get("data_file") or path.with_suffix(".eeg").name
        eeg_path = path.with_name(str(data_file))
        if not eeg_path.exists():
            raise FileNotFoundError(eeg_path)

        total_samples = eeg_path.stat().st_size // (4 * n_channels)
        row = self.get_stim_row(session_id, stim_pair)
        margin = float(self.config.get("bids_window_margin_s", 1.0))
        start_value = row.get("stim_start")
        stop_value = row.get("stim_stop")
        start_sample, stop_sample, index_kind, base_time = self._sample_window_from_stim_values(
            start_value,
            stop_value,
            sfreq,
            total_samples,
            margin,
        )
        n_samples = stop_sample - start_sample
        offset = start_sample * n_channels * 4
        mm = np.memmap(eeg_path, dtype="<f4", mode="r", offset=offset, shape=(n_samples, n_channels))
        data = np.asarray(mm, dtype=float) * resolutions
        if index_kind == "datetime":
            seconds = np.arange(start_sample, stop_sample) / sfreq
            index = pd.DatetimeIndex(base_time + pd.to_timedelta(seconds, unit="s"))
        else:
            index = pd.Index(np.arange(start_sample, stop_sample) / sfreq, name="time")
        df = pd.DataFrame(data, index=index, columns=channels)
        df.attrs["sfreq"] = sfreq
        df.attrs["source_file"] = str(path)
        df.attrs["signal_units"] = "uV"
        df.attrs["value_scale_to_uv_applied"] = 1.0
        return df

    def _window_seconds_from_stim_values(
        self,
        start_value: Any,
        stop_value: Any,
        margin: float,
        base_time: pd.Timestamp | None = None,
    ) -> tuple[float, float, str, pd.Timestamp | None]:
        start_num = pd.to_numeric(pd.Series([start_value]), errors="coerce").iloc[0]
        stop_num = pd.to_numeric(pd.Series([stop_value]), errors="coerce").iloc[0]
        if pd.notna(start_num) and pd.notna(stop_num):
            return max(float(start_num) - margin, 0.0), float(stop_num) + margin, "seconds", None

        start_dt = pd.to_datetime(start_value, errors="coerce")
        stop_dt = pd.to_datetime(stop_value, errors="coerce")
        if pd.notna(start_dt) and pd.notna(stop_dt):
            base = _coerce_timestamp(base_time)
            if base is None and "start_time" in self.raw_meta.columns and self.raw_meta["start_time"].notna().any():
                parsed = pd.to_datetime(self.raw_meta["start_time"], errors="coerce").dropna()
                if not parsed.empty:
                    base = _coerce_timestamp(parsed.iloc[0])
            base = base if base is not None else _coerce_timestamp(start_dt)
            start_s = max((_coerce_timestamp(start_dt) - base).total_seconds() - margin, 0.0)
            stop_s = (_coerce_timestamp(stop_dt) - base).total_seconds() + margin
            return start_s, stop_s, "datetime", base

        return 0.0, np.inf, "seconds", None

    def _sample_window_from_stim_values(
        self,
        start_value: Any,
        stop_value: Any,
        sfreq: float,
        total_samples: int,
        margin: float,
        base_time: pd.Timestamp | None = None,
        time_offset_s: float = 0.0,
    ) -> tuple[int, int, str, pd.Timestamp | None]:
        start_s, stop_s, index_kind, base = self._window_seconds_from_stim_values(
            start_value,
            stop_value,
            margin,
            base_time=base_time,
        )
        start_s = max(start_s - float(time_offset_s), 0.0)
        stop_s = min(stop_s - float(time_offset_s), total_samples / sfreq)
        if not np.isfinite(stop_s):
            stop_s = total_samples / sfreq
        start_sample = min(max(int(np.floor(start_s * sfreq)), 0), int(total_samples))
        stop_sample = min(max(int(np.ceil(stop_s * sfreq)), 0), int(total_samples))
        return start_sample, stop_sample, index_kind, base


def _clean_channel_label(label: str) -> str:
    text = str(label).strip()
    text = re.sub(r"^POL\s*", "", text, flags=re.I)
    text = re.sub(r"[-\s_]*Ref$", "", text, flags=re.I)
    text = text.replace(" ", "")
    text = re.sub(r"([A-Za-z]+)[_-]?0*([0-9]+)$", lambda m: f"{m.group(1)}{int(m.group(2))}", text)
    return text


def _require_pynwb():
    try:
        from pynwb import NWBHDF5IO
        from pynwb.ecephys import ElectricalSeries
    except Exception as exc:
        raise ImportError("NWB support requires the optional 'pynwb' dependency. Install with: python -m pip install -e '.[nwb]'") from exc
    return NWBHDF5IO, ElectricalSeries


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _nwb_subject_id(nwbfile: Any) -> str | None:
    subject = getattr(nwbfile, "subject", None)
    if subject is not None:
        sid = getattr(subject, "subject_id", None)
        if sid:
            return str(sid)
    return None


def _pick_nwb_electrical_series(nwbfile: Any, electrical_series_cls: Any, series_name: str | None = None) -> tuple[str, Any]:
    candidates = list(_iter_nwb_electrical_series(nwbfile, electrical_series_cls))
    if not candidates:
        raise FileNotFoundError("No NWB ElectricalSeries found in acquisitions, processing modules, or stimulus containers")
    if series_name:
        target = str(series_name).strip()
        for path, series in candidates:
            if target in {path, path.split("/")[-1], getattr(series, "name", "")}:
                return path, series
        available = ", ".join(path for path, _ in candidates[:10])
        raise KeyError(f"NWB ElectricalSeries {series_name!r} not found. Available series include: {available}")
    return candidates[0]


def _iter_nwb_electrical_series(nwbfile: Any, electrical_series_cls: Any):
    def is_series(obj: Any) -> bool:
        return isinstance(obj, electrical_series_cls) or obj.__class__.__name__ == "ElectricalSeries"

    for name, obj in getattr(nwbfile, "acquisition", {}).items():
        if is_series(obj):
            yield f"acquisition/{name}", obj
    for module_name, module in getattr(nwbfile, "processing", {}).items():
        for name, obj in getattr(module, "data_interfaces", {}).items():
            if is_series(obj):
                yield f"processing/{module_name}/{name}", obj
            for subname, subobj in getattr(obj, "electrical_series", {}).items():
                if is_series(subobj):
                    yield f"processing/{module_name}/{name}/{subname}", subobj
    for name, obj in getattr(nwbfile, "stimulus", {}).items():
        if is_series(obj):
            yield f"stimulus/{name}", obj


def _infer_nwb_sfreq(series: Any) -> float | None:
    rate = getattr(series, "rate", None)
    if rate is not None:
        try:
            return float(rate)
        except Exception:
            pass
    timestamps = _nwb_timestamps(series)
    if timestamps is not None and len(timestamps) > 1:
        diffs = np.diff(timestamps)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(1.0 / np.median(diffs))
    return None


def _nwb_timestamps(series: Any) -> np.ndarray | None:
    timestamps = getattr(series, "timestamps", None)
    if timestamps is None:
        return None
    try:
        values = np.asarray(timestamps[:], dtype=float)
    except Exception:
        try:
            values = np.asarray(timestamps, dtype=float)
        except Exception:
            return None
    return values if values.size else None


def _nwb_series_starting_time(series: Any) -> float:
    value = getattr(series, "starting_time", 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _nwb_time_axis(series: Any) -> int:
    shape = tuple(series.data.shape)
    if len(shape) < 2:
        return 0
    n_electrodes = _nwb_series_electrode_count(series)
    if n_electrodes and shape[0] == n_electrodes and shape[1] != n_electrodes:
        return 1
    return 0


def _slice_nwb_data(data_obj: Any, start_sample: int, stop_sample: int, time_axis: int) -> np.ndarray:
    if time_axis == 0:
        data = np.asarray(data_obj[start_sample:stop_sample], dtype=float)
    else:
        data = np.asarray(data_obj[:, start_sample:stop_sample], dtype=float)
        data = np.moveaxis(data, 1, 0)
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], int(np.prod(data.shape[1:])))
    return data


def _scale_nwb_electrical_data(series: Any, data: np.ndarray) -> np.ndarray:
    conversion = getattr(series, "conversion", None)
    if conversion is not None:
        try:
            conversion = float(conversion)
            if np.isfinite(conversion):
                data = data * conversion
        except Exception:
            pass
    unit = str(getattr(series, "unit", "") or "").strip().lower()
    if unit in {"v", "volt", "volts"}:
        data = data * 1e6
    elif unit in {"mv", "millivolt", "millivolts"}:
        data = data * 1e3
    return data


def _nwb_window_index(
    series: Any,
    start_sample: int,
    stop_sample: int,
    sfreq: float,
    index_kind: str,
    base_time: pd.Timestamp | None,
) -> pd.Index:
    timestamps = _nwb_timestamps(series)
    if timestamps is not None and len(timestamps) >= stop_sample:
        seconds = timestamps[start_sample:stop_sample]
    else:
        start = _nwb_series_starting_time(series)
        seconds = start + np.arange(start_sample, stop_sample) / sfreq
    if index_kind == "datetime" and base_time is not None:
        return seconds_to_index(base_time, seconds)
    return pd.Index(seconds, name="time")


def _nwb_series_channel_labels(series: Any, n_channels: int) -> list[str]:
    region = getattr(series, "electrodes", None)
    table_df = _nwb_electrode_region_dataframe(region)
    if table_df is not None and not table_df.empty:
        labels = _labels_from_electrode_dataframe(table_df)
    else:
        labels = []
    if len(labels) < n_channels:
        labels.extend(f"CH{i + 1}" for i in range(len(labels), n_channels))
    return labels[:n_channels]


def _nwb_series_electrode_count(series: Any) -> int | None:
    region = getattr(series, "electrodes", None)
    if region is None:
        return None
    try:
        return int(len(region.data[:]))
    except Exception:
        try:
            return int(len(region))
        except Exception:
            return None


def _nwb_electrode_region_dataframe(region: Any) -> pd.DataFrame | None:
    if region is None:
        return None
    try:
        table = region.table.to_dataframe()
    except Exception:
        return None
    try:
        selected = np.asarray(region.data[:])
    except Exception:
        selected = np.array([])
    if selected.size:
        try:
            if np.issubdtype(selected.dtype, np.integer) and selected.max(initial=-1) < len(table):
                return table.iloc[selected.astype(int)].copy()
            return table.loc[selected].copy()
        except Exception:
            pass
    return table.copy()


def _nwb_electrode_metadata(nwbfile: Any, patient_id: str, session_id: str, series: Any | None = None) -> pd.DataFrame:
    rows = []
    try:
        table = nwbfile.electrodes.to_dataframe()
    except Exception:
        table = pd.DataFrame()
    if table.empty and series is not None:
        region_table = _nwb_electrode_region_dataframe(getattr(series, "electrodes", None))
        table = region_table if region_table is not None else pd.DataFrame()
    if table.empty and series is not None:
        n_channels = _nwb_series_electrode_count(series)
        if n_channels is None:
            shape = tuple(series.data.shape)
            n_channels = int(shape[1]) if len(shape) > 1 else 1
        labels = _nwb_series_channel_labels(series, n_channels)
        return pd.DataFrame(
            [
                {"patient_id": patient_id, "session_id": session_id, "elec_label": label, "anat_label": "", "bad_channel": False}
                for label in labels
            ]
        )
    labels = _labels_from_electrode_dataframe(table)
    for i, (_, row) in enumerate(table.iterrows()):
        label = labels[i] if i < len(labels) else f"E{i + 1}"
        rows.append(
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "elec_label": _clean_channel_label(label),
                "anat_label": row.get("location", row.get("group_name", "")),
                "mni_x": row.get("x", ""),
                "mni_y": row.get("y", ""),
                "mni_z": row.get("z", ""),
                "coordinate_system": row.get("coordinate_system", row.get("reference_frame", "")),
                "bad_channel": False,
            }
        )
    return pd.DataFrame(rows)


def _labels_from_electrode_dataframe(df: pd.DataFrame) -> list[str]:
    for col in ("label", "name", "electrode_label", "channel_name", "origchannel_name"):
        if col in df.columns and df[col].notna().any():
            return df[col].fillna("").astype(str).replace("", np.nan).ffill().fillna("").tolist()
    if df.index.name is not None or len(df.index):
        return [f"E{idx}" if str(idx).isdigit() else str(idx) for idx in df.index]
    return [f"E{i + 1}" for i in range(len(df))]


def _unique_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        base = label or "CH"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def _extract_nwb_stim_events(nwbfile: Any, stim_pair: str | None = None) -> tuple[str | None, pd.Series | None]:
    trials = getattr(nwbfile, "trials", None)
    if trials is None:
        return stim_pair, None
    try:
        df = trials.to_dataframe()
    except Exception:
        return stim_pair, None
    if df.empty:
        return stim_pair, None
    site_col = next(
        (col for col in ("stim_pair", "electrical_stimulation_site", "stimulation_site", "stimulation_pair", "stimulus_pair") if col in df.columns),
        None,
    )
    time_col = "start_time" if "start_time" in df.columns else None
    if time_col is None:
        return stim_pair, None
    work = df.copy()
    if site_col is not None:
        work["_erpy_stim_pair"] = work[site_col].map(_normalize_nwb_stim_pair)
    else:
        stim_like = pd.Series(False, index=work.index)
        for col in work.columns:
            if any(key in str(col).lower() for key in ("electrical", "stimulation", "stim_pair", "pulse", "trigger")):
                stim_like = stim_like | work[col].astype(str).str.contains(
                    "electrical|stimulation|stim_pair|pulse|trigger",
                    case=False,
                    na=False,
                )
        if not stim_like.any():
            return stim_pair, None
        work = work[stim_like].copy()
        work["_erpy_stim_pair"] = stim_pair or "NWB_STIM"
    if stim_pair:
        target = _normalize_nwb_stim_pair(stim_pair)
        work = work[work["_erpy_stim_pair"].astype(str) == target]
    else:
        if work.empty:
            return stim_pair, None
        stim_pair = str(work["_erpy_stim_pair"].mode().iloc[0])
        work = work[work["_erpy_stim_pair"].astype(str) == stim_pair]
    times = pd.to_numeric(work[time_col], errors="coerce").dropna()
    if times.empty:
        return stim_pair, None
    return stim_pair, times.sort_values().reset_index(drop=True)


def _normalize_nwb_stim_pair(value: Any) -> str:
    try:
        from .bids import normalize_stim_pair

        return normalize_stim_pair(str(value))
    except Exception:
        return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "NWB_STIM"


def _nwb_event_table(stim_times: Any) -> pd.DataFrame:
    times = _values_to_series(stim_times)
    numeric = pd.to_numeric(times, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return pd.DataFrame({"times": numeric.dropna().to_numpy(dtype=float)})
    parsed = pd.to_datetime(times, errors="coerce")
    if parsed.notna().mean() > 0.8:
        return pd.DataFrame({"times": parsed.dropna().map(lambda ts: _coerce_timestamp(ts).isoformat())})
    return pd.DataFrame({"times": times.dropna().astype(str)})


def _values_to_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    if isinstance(values, pd.Index):
        return pd.Series(values.to_numpy())
    if isinstance(values, np.ndarray):
        return pd.Series(values.ravel())
    if isinstance(values, (list, tuple, set)):
        return pd.Series(list(values))
    return pd.Series([values])


def _nwb_stim_metadata(
    patient_id: str,
    session_id: str,
    stim_pair: str | None,
    stim_times: Any | None,
    stim_duration_s: float,
    source_nwb_file: str,
) -> pd.DataFrame:
    if not stim_pair or stim_times is None:
        return pd.DataFrame(columns=["patient_id", "session_id", "stim_pair", "stim_start", "stim_stop", "stim_freq"])
    events = _nwb_event_table(stim_times)
    if events.empty:
        return pd.DataFrame(columns=["patient_id", "session_id", "stim_pair", "stim_start", "stim_stop", "stim_freq"])
    values = events["times"]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.8:
        ordered = np.sort(numeric.dropna().to_numpy(dtype=float))
        stim_start: Any = float(ordered[0])
        stim_stop: Any = float(ordered[-1] + stim_duration_s)
        diffs = np.diff(ordered)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        stim_freq: Any = float(1.0 / np.median(diffs)) if diffs.size else ""
    else:
        parsed = pd.to_datetime(values, errors="coerce").dropna().sort_values()
        stim_start = _coerce_timestamp(parsed.iloc[0]).isoformat()
        stim_stop = (_coerce_timestamp(parsed.iloc[-1]) + pd.to_timedelta(stim_duration_s, unit="s")).isoformat()
        diffs = parsed.diff().dropna().dt.total_seconds().to_numpy()
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        stim_freq = float(1.0 / np.median(diffs)) if diffs.size else ""
    return pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "stim_pair": _normalize_nwb_stim_pair(stim_pair),
                "stim_start": stim_start,
                "stim_stop": stim_stop,
                "stim_freq": stim_freq,
                "source_nwb_file": source_nwb_file,
            }
        ]
    )
