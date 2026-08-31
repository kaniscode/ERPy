from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable
import uuid

import numpy as np
import pandas as pd

from .bad_channels import reject_bad_channels
from .artifact_anchor import POST_ARTIFACT_ZERO, detect_post_artifact_anchor
from .epochs import (
    Epochs,
    EpochsRunner,
    UnsafeLegacyMetadataError,
    _read_metadata_node,
    _write_metadata_node,
)
from .utils import preproc
from .utils.jobrunner import JobRunner
from .utils.utils import ensure_dir, infer_sfreq, stable_hash, write_time_series_csv


class UnsafeLegacyProcessedMetadataError(UnsafeLegacyMetadataError):
    """Raised when pickle-backed processed-data metadata are not trusted."""


class Pipeline:
    """Registry-based preprocessing and epoch extraction."""

    PYTHON_SCRIPT: str | None = None
    BASH_SCRIPT: str | None = None

    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.registry = {
            "artifact_blank": self._step_artifact_blank,
            "notch_filter": self._step_notch_filter,
            "bandpass_filter": self._step_bandpass_filter,
            "decimate": self._step_decimate,
            "commonavg_ref": self._step_commonavg_ref,
            "bipolar_ref": self._step_bipolar_ref,
            "laplacian_ref": self._step_laplacian_ref,
            "reject_bad_channels": self._step_reject_bad_channels,
        }
        # Re-reference aliases. Common-average referencing (CAR) is the default
        # montage; bipolar and Laplacian are available alternatives.
        self.registry["car_ref"] = self._step_commonavg_ref

    def get_standard_pipeline(self, name: str = "blank_filt") -> list[tuple[str, dict[str, Any]]]:
        """Return a copy-ready named preprocessing step sequence.

        Available presets combine artifact blanking, line-noise filtering,
        band-pass filtering, optional decimation, and a selected reference.
        """

        if name in {"raw", "none"}:
            return []
        # Standard pipelines apply common-average referencing (CAR) by default,
        # after blanking/filtering, excluding the stimulation contacts from the
        # reference. ``*_noref`` variants and ``raw``/``none`` skip re-referencing;
        # the ``bipolar_ref`` and ``laplacian_ref`` steps offer alternative montages.
        pipelines = {
            "blank_filt": [
                ("artifact_blank", {"width_s": 0.003}),
                ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
                ("bandpass_filter", {"lowcut": 0.1, "highcut": 50.0, "order": 4}),
                ("commonavg_ref", {"method": "robust_mean", "zscore_threshold": 8.0}),
            ],
            "blank_filt_noref": [
                ("artifact_blank", {"width_s": 0.003}),
                ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
                ("bandpass_filter", {"lowcut": 0.1, "highcut": 50.0, "order": 4}),
            ],
            "blank_filt_bipolar": [
                ("artifact_blank", {"width_s": 0.003}),
                ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
                ("bandpass_filter", {"lowcut": 0.1, "highcut": 50.0, "order": 4}),
                ("bipolar_ref", {}),
            ],
            "blank_filt_laplacian": [
                ("artifact_blank", {"width_s": 0.003}),
                ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
                ("bandpass_filter", {"lowcut": 0.1, "highcut": 50.0, "order": 4}),
                ("laplacian_ref", {}),
            ],
            "blank_dec_filt": [
                ("artifact_blank", {"width_s": 0.003}),
                ("decimate", {"fs_new": 500.0}),
                ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
                ("bandpass_filter", {"lowcut": 0.1, "highcut": 50.0, "order": 4}),
                ("commonavg_ref", {"method": "robust_mean", "zscore_threshold": 8.0}),
            ],
            "blank_dec_wide": [
                ("artifact_blank", {"width_s": 0.003}),
                ("decimate", {"fs_new": 500.0}),
                ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
                ("bandpass_filter", {"lowcut": 0.1, "highcut": 180.0, "order": 4}),
                ("commonavg_ref", {"method": "robust_mean", "zscore_threshold": 8.0}),
            ],
            "blank_filt_reject": [
                ("reject_bad_channels", {"bad_channels": "auto", "method": "drop"}),
                ("artifact_blank", {"width_s": 0.003}),
                ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
                ("bandpass_filter", {"lowcut": 0.1, "highcut": 50.0, "order": 4}),
                ("commonavg_ref", {"method": "robust_mean", "zscore_threshold": 8.0}),
            ],
        }
        if name not in pipelines:
            raise KeyError(f"Unknown standard pipeline {name!r}. Available: {sorted(pipelines)}")
        return pipelines[name]

    def build_custom_pipeline(self, steps: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
        """Validate and normalize a user-supplied preprocessing sequence."""

        validated = []
        for step, params in steps:
            if step not in self.registry:
                raise KeyError(f"Unknown pipeline step {step!r}. Available: {sorted(self.registry)}")
            validated.append((step, dict(params or {})))
        return validated

    def pipeline_id(self, pipeline_steps: list[tuple[str, dict[str, Any]]] | None) -> str:
        """Return a stable identifier for a preprocessing step sequence."""

        return stable_hash(pipeline_steps or [], length=10)

    def _processed_path(self, session_id: str, stim_pair: str, pipeline_steps) -> Path:
        pid = self.pipeline_id(pipeline_steps)
        return self.dataloader.get_session_path(session_id, "processed") / f"{stim_pair}_{session_id}_{pid}_processed.h5"

    def _epochs_path(
        self,
        session_id: str,
        stim_pair: str,
        pipeline_steps,
        tmin,
        tmax,
        baseline,
        zero_time=POST_ARTIFACT_ZERO,
        artifact_anchor_params: dict[str, Any] | None = None,
    ) -> Path:
        pid = self.pipeline_id(pipeline_steps)
        tag = stable_hash(
            {
                "tmin": tmin,
                "tmax": tmax,
                "baseline": baseline,
                "zero_time": zero_time,
                "artifact_anchor_params": artifact_anchor_params or {},
            },
            length=8,
        )
        return self.dataloader.get_session_path(session_id, "epochs") / f"{stim_pair}_{session_id}_{pid}_{tag}_epochs.h5"

    def _load_event_info(self, session_id: str, stim_pair: str) -> pd.DataFrame:
        path = self.dataloader.get_event_path(session_id, stim_pair)
        if not path.exists():
            raise FileNotFoundError(f"Missing event file: {path}")
        df = pd.read_csv(path)
        if "times" not in df.columns:
            raise ValueError(f"Event file must contain a 'times' column: {path}")
        return df

    def _event_times(self, session_id: str, stim_pair: str):
        df = self._load_event_info(session_id, stim_pair)
        if pd.api.types.is_numeric_dtype(df["times"]):
            return df["times"].to_numpy()
        parsed = pd.to_datetime(df["times"], errors="coerce")
        return parsed.to_numpy() if parsed.notna().mean() > 0.8 else df["times"].to_numpy()

    def process(
        self,
        pipeline_steps: list[tuple[str, dict[str, Any]]] | None = None,
        session_id: str | None = None,
        stim_pair: str | None = None,
        **kwargs: Any,
    ):
        """Run preprocessing for one acquisition or all matching acquisitions."""

        if session_id is not None and stim_pair is not None:
            return self.process_stim_session(session_id, stim_pair, pipeline_steps, **kwargs)
        results = {}
        sessions = [session_id] if session_id else self.dataloader.session_ids
        for sid in sessions:
            pairs = [stim_pair] if stim_pair else self.dataloader.stim_pairs(sid)
            for sp in pairs:
                results[(sid, sp)] = self.process_stim_session(sid, sp, pipeline_steps, **kwargs)
        return results

    def process_stim_session(
        self,
        session_id: str,
        stim_pair: str,
        pipeline_steps: list[tuple[str, dict[str, Any]]] | None = None,
        input_source: str = "auto",
        load_from_raw: bool = True,
        save_processed: bool = False,
        overwrite: bool = False,
        raw_file: str | Path | None = None,
        stim_start: Any | None = None,
        time_window: tuple[Any, Any] | None = None,
        **context: Any,
    ) -> pd.DataFrame:
        """Load and preprocess one session/stimulation-pair acquisition.

        The returned continuous table records sampling frequency and pipeline
        identity in ``DataFrame.attrs`` and can optionally be cached on disk.
        """

        steps = self.build_custom_pipeline(pipeline_steps or [])
        path = self._processed_path(session_id, stim_pair, steps)
        if path.exists() and not overwrite and not load_from_raw and input_source == "processed":
            return self.load_processed_data(path)
        if input_source == "processed" and path.exists() and not overwrite:
            return self.load_processed_data(path)

        df = self.dataloader.load_stim_data(
            session_id,
            stim_pair,
            source="auto" if input_source == "auto" else input_source,
            raw_file=raw_file,
            stim_start=stim_start,
        )
        if time_window is not None:
            df = _crop_continuous_window(df, *time_window)
        sfreq = float(df.attrs.get("sfreq") or self.dataloader.get_sampling_freq(session_id, stim_pair) or infer_sfreq(df))
        state = {
            "session_id": session_id,
            "stim_pair": stim_pair,
            "sfreq": sfreq,
            "stim_chs": _stim_channels(stim_pair),
            **context,
        }
        for step_name, params in steps:
            df = self.registry[step_name](df, **params, **state)
            sfreq = float(df.attrs.get("sfreq", sfreq))
            state["sfreq"] = sfreq
        df.attrs["sfreq"] = sfreq
        df.attrs["pipeline_id"] = self.pipeline_id(steps)
        if save_processed:
            self.save_processed_data(df, path, pipeline_steps=steps, session_id=session_id, stim_pair=stim_pair)
        return df

    def process_session(self, session_id: str, pipeline_steps=None, stim_pairs: Iterable[str] | None = None, **kwargs: Any):
        """Preprocess selected stimulation pairs within one session."""

        return {
            sp: self.process_stim_session(session_id, sp, pipeline_steps, **kwargs)
            for sp in (list(stim_pairs) if stim_pairs is not None else self.dataloader.stim_pairs(session_id))
        }

    def process_all_sessions(self, pipeline_steps=None, session_ids: Iterable[str] | None = None, **kwargs: Any):
        """Preprocess selected or registered sessions and stimulation pairs."""

        return {
            sid: self.process_session(sid, pipeline_steps, **kwargs)
            for sid in (list(session_ids) if session_ids is not None else self.dataloader.session_ids)
        }

    def epoch_data(
        self,
        df: pd.DataFrame,
        event_times: Iterable[Any],
        tmin: float = -0.5,
        tmax: float = 1.0,
        baseline: tuple[float, float] | None = None,
        zero_time: float | str | None = POST_ARTIFACT_ZERO,
        stim_ch: str | list[str] | None = None,
        artifact_anchor_params: dict[str, Any] | None = None,
    ) -> Epochs:
        """Extract event-locked trials from a continuous processed recording.

        Epochs may use a fixed zero-time sample or ERPy's post-artifact anchor;
        the latter stores per-trial anchor diagnostics in audit metadata.
        """

        events = list(event_times)
        if not events:
            raise ValueError("process_and_epoch requires at least one event")
        sfreq = float(df.attrs.get("sfreq") or infer_sfreq(df))
        n_pre = int(round(abs(tmin) * sfreq))
        n_post = int(round(tmax * sfreq))
        n = n_pre + n_post + 1
        times = np.round((np.arange(n) - n_pre) / sfreq, decimals=9)
        zero_anchor_reports: list[dict[str, Any]] = []
        anchor_params = dict(artifact_anchor_params or {})
        # Width of the post-anchor baseline window (subtracted per trial so the
        # response starts from a clean, artifact-free zero). A short window is far
        # more robust than a single sample on the still-decaying artifact tail.
        post_baseline_s = float(anchor_params.pop("post_baseline_s", 0.005))

        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            event_values = pd.to_datetime(pd.Series(events), errors="coerce")
            positions = idx.searchsorted(event_values)
        else:
            idx_values = pd.to_numeric(pd.Series(df.index), errors="coerce").to_numpy(dtype=float)
            event_values = pd.to_numeric(pd.Series(events), errors="coerce").to_numpy(dtype=float)
            positions = np.searchsorted(idx_values, event_values)

        values = df.select_dtypes(include=[np.number])
        # First pass: extract and baseline-window-correct every epoch.
        chunks: list[tuple[int, pd.DataFrame]] = []
        for epoch_i, pos in enumerate(positions):
            start = int(pos) - n_pre
            stop = int(pos) + n_post + 1
            if start < 0 or stop > len(values):
                continue
            chunk = values.iloc[start:stop].copy()
            if len(chunk) != n:
                continue
            chunk.index = times
            if baseline is not None:
                mask = (times >= baseline[0]) & (times <= baseline[1])
                if mask.any():
                    chunk = chunk - chunk.loc[mask].mean(axis=0)
            chunks.append((int(epoch_i), chunk))
        if not chunks:
            raise ValueError("No full epochs could be extracted; check event times and tmin/tmax")

        # Post-artifact zero anchoring. The anchor is applied at a single COMMON
        # time across trials, using a short windowed baseline, so the trial-
        # averaged waveform is genuinely zero at the anchor (a per-trial, single-
        # sample anchor leaves the average non-zero because trials settle at
        # slightly different times). The common anchor is detected on the trial
        # average, which is robust to per-trial noise and ensures the artifact has
        # settled for the cohort.
        n_win = max(int(round(post_baseline_s * sfreq)), 1)
        common_anchor_idx = -1
        common_anchor_time = float("nan")
        if zero_time is not None:
            if zero_time == POST_ARTIFACT_ZERO:
                for epoch_i, chunk in chunks:
                    rep = detect_post_artifact_anchor(
                        chunk, stim_ch=stim_ch, baseline_window=baseline, **anchor_params
                    )
                    rep = {**rep, "epoch": int(epoch_i), "zero_time_strategy": POST_ARTIFACT_ZERO}
                    zero_anchor_reports.append(rep)
                mean_arr = np.nanmean(np.stack([c.to_numpy(dtype=float) for _, c in chunks]), axis=0)
                mean_chunk = pd.DataFrame(mean_arr, index=times, columns=values.columns)
                mean_rep = detect_post_artifact_anchor(
                    mean_chunk, stim_ch=stim_ch, baseline_window=baseline, **anchor_params
                )
                common_anchor_idx = int(mean_rep.get("anchor_index", -1))
                if not (0 <= common_anchor_idx < n):
                    valid = [int(r.get("anchor_index", -1)) for r in zero_anchor_reports]
                    valid = [i for i in valid if 0 <= i < n]
                    common_anchor_idx = int(np.median(valid)) if valid else int(np.argmin(np.abs(times)))
            else:
                common_anchor_idx = int(np.argmin(np.abs(times - float(zero_time))))
                for epoch_i, _ in chunks:
                    zero_anchor_reports.append(
                        {
                            "epoch": int(epoch_i),
                            "anchor_time_s": float(times[common_anchor_idx]),
                            "anchor_index": int(common_anchor_idx),
                            "artifact_peak_time_s": np.nan,
                            "artifact_peak_score_z": np.nan,
                            "onset_score_z": np.nan,
                            "onset_is_artifact": False,
                            "settled": True,
                            "method": "fixed_zero_time",
                            "reference_channels": "",
                            "zero_time_strategy": "fixed",
                        }
                    )
            common_anchor_time = float(times[common_anchor_idx])
            win_lo = int(common_anchor_idx)
            win_hi = int(min(common_anchor_idx + n_win, n))
            for rep in zero_anchor_reports:
                rep["common_anchor_time_s"] = common_anchor_time
                rep["common_anchor_index"] = int(common_anchor_idx)
                rep["post_baseline_window_s"] = float(n_win / sfreq)
            # Second pass: subtract each trial's mean over the common post-anchor window.
            for _, chunk in chunks:
                window = chunk.iloc[win_lo:win_hi].replace([np.inf, -np.inf], np.nan)
                offset = window.mean(axis=0).fillna(0.0)
                chunk -= offset

        rows = [chunk.to_numpy(dtype=float) for _, chunk in chunks]
        index_parts = [(epoch_i, float(t)) for epoch_i, _ in chunks for t in times]
        data = np.vstack(rows)
        idx = pd.MultiIndex.from_tuples(index_parts, names=["epoch", "time"])
        epochs_df = pd.DataFrame(data, index=idx, columns=list(values.columns))
        return Epochs(
            epochs_df=epochs_df,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            stim_ch=stim_ch,
            metadata={
                "n_events_input": len(events),
                "n_epochs": len(rows),
                "signal_units": df.attrs.get("signal_units"),
                "value_scale_to_uv_applied": df.attrs.get("value_scale_to_uv_applied", 1.0),
                "reference_method": df.attrs.get("reference_method"),
                "reference_channels": df.attrs.get("reference_channels"),
                "reference_zscore_threshold": df.attrs.get("reference_zscore_threshold"),
                "zero_time_strategy": zero_time,
                "zero_anchor_reports": zero_anchor_reports,
                "artifact_anchor_params": anchor_params,
                "common_anchor_time_s": common_anchor_time,
                "post_baseline_window_s": float(n_win / sfreq),
                "post_baseline_samples": int(n_win),
            },
            zero_time=zero_time,
        )

    def process_and_epoch(
        self,
        session_id: str,
        stim_pair: str,
        pipeline_steps: list[tuple[str, dict[str, Any]]] | None = None,
        event_times: Iterable[Any] | None = None,
        tmin: float = -0.5,
        tmax: float = 1.0,
        baseline: tuple[float, float] | None = (-0.5, -0.03),
        zero_time: float | str | None = POST_ARTIFACT_ZERO,
        artifact_anchor_params: dict[str, Any] | None = None,
        processing_margin_s: float | None = None,
        save_processed: bool = False,
        save_epochs: bool = False,
        **kwargs: Any,
    ) -> tuple[Epochs, pd.DataFrame]:
        """Preprocess one acquisition and extract its event-locked epochs."""

        steps = self.build_custom_pipeline(pipeline_steps or [])
        if event_times is None:
            event_times = self._event_times(session_id, stim_pair)
        event_times = list(event_times)
        time_window = _event_processing_window(
            event_times,
            tmin=tmin,
            tmax=tmax,
            margin_s=processing_margin_s,
        )
        processed = self.process_stim_session(
            session_id,
            stim_pair,
            steps,
            save_processed=save_processed,
            event_times=event_times,
            time_window=time_window,
            **kwargs,
        )
        epochs = self.epoch_data(
            processed,
            event_times=event_times,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            zero_time=zero_time,
            stim_ch=_stim_channels(stim_pair),
            artifact_anchor_params=artifact_anchor_params,
        )
        epochs.metadata.update(
            {
                "patient_id": self.dataloader.patient_id,
                "session_id": session_id,
                "stim_pair": stim_pair,
                "pipeline_id": self.pipeline_id(steps),
                "pipeline_steps": steps,
                "zero_time": zero_time,
                "zero_time_strategy": zero_time,
                "artifact_anchor_params": artifact_anchor_params or {},
                "processing_margin_s": processing_margin_s,
            }
        )
        if save_epochs:
            epochs.to_hdf(
                self._epochs_path(
                    session_id,
                    stim_pair,
                    steps,
                    tmin,
                    tmax,
                    baseline,
                    zero_time,
                    artifact_anchor_params=artifact_anchor_params,
                )
            )
        return epochs, processed

    def save_processed_data(self, df: pd.DataFrame, path: str | Path, **metadata: Any) -> Path:
        """Atomically save processed data with deterministic JSON provenance."""

        path = Path(path)
        ensure_dir(path.parent)
        temporary = path.with_name(
            f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix or '.h5'}"
        )
        provenance = {
            "sfreq": df.attrs.get("sfreq"),
            "pipeline_id": df.attrs.get("pipeline_id"),
            **metadata,
        }
        try:
            with pd.HDFStore(temporary, mode="w") as store:
                store.put("data", df, format="table")
            _write_metadata_node(temporary, "data", provenance)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load_processed_data(
        self,
        path: str | Path,
        *,
        allow_unsafe_legacy_pickle: bool = False,
    ) -> pd.DataFrame:
        """Load processed data and safely restore its provenance metadata.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 processed-data file to read.
        allow_unsafe_legacy_pickle : bool, default ``False``
            Permit metadata from a trusted pre-1.0 file. Legacy PyTables
            attributes use pickle and may execute arbitrary code; never enable
            this option for an untrusted or downloaded file.

        Returns
        -------
        pandas.DataFrame
            Processed samples with restored metadata in ``DataFrame.attrs``.
        """

        path = Path(path)
        md = _read_metadata_node(
            path,
            "data",
            allow_unsafe_legacy_pickle=allow_unsafe_legacy_pickle,
            legacy_error_type=UnsafeLegacyProcessedMetadataError,
            metadata_description="ERPy processed-data",
            unsafe_opt_in_example=(
                "Pipeline.load_processed_data(..., "
                "allow_unsafe_legacy_pickle=True)"
            ),
        )
        with pd.HDFStore(path, mode="r") as store:
            df = store["data"]
            if md is None and allow_unsafe_legacy_pickle:
                md = getattr(store.get_storer("data").attrs, "metadata", {}) or {}
        df.attrs.update(md or {})
        return df

    def load_epochs_file(
        self,
        path: str | Path,
        *,
        allow_unsafe_legacy_pickle: bool = False,
    ) -> Epochs:
        """Load a cached epoch file.

        Set ``allow_unsafe_legacy_pickle`` only for a pre-1.0 file that you
        created and trust; legacy pickle metadata can execute arbitrary code.
        """

        return Epochs.from_hdf(
            path,
            allow_unsafe_legacy_pickle=allow_unsafe_legacy_pickle,
        )

    def find_epochs_files(self, session_id: str, stim_pair: str, pipeline_steps=None) -> list[Path]:
        """Return matching epoch caches from newest to oldest."""

        root = self.dataloader.get_session_path(session_id, "epochs")
        if pipeline_steps is None:
            pattern = f"{stim_pair}_{session_id}_*_epochs.h5"
        else:
            pattern = f"{stim_pair}_{session_id}_{self.pipeline_id(pipeline_steps)}_*_epochs.h5"
        return sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    def load_epochs(
        self,
        session_id: str,
        stim_pair: str,
        pipeline_steps=None,
        *,
        allow_unsafe_legacy_pickle: bool = False,
    ) -> Epochs:
        """Load the newest matching epoch cache for an acquisition."""

        files = self.find_epochs_files(session_id, stim_pair, pipeline_steps)
        if not files:
            raise FileNotFoundError(f"No cached epochs for {session_id}/{stim_pair}")
        return self.load_epochs_file(
            files[0],
            allow_unsafe_legacy_pickle=allow_unsafe_legacy_pickle,
        )

    def _step_artifact_blank(self, df: pd.DataFrame, event_freq_to_width: dict | None = None, width_s: float = 0.003, **state):
        events = state.get("event_times")
        if events is None:
            try:
                events = self._event_times(state["session_id"], state["stim_pair"])
            except Exception:
                events = []
        if event_freq_to_width:
            try:
                row = self.dataloader.get_stim_row(state["session_id"], state["stim_pair"])
                freq = float(row["stim_freq"])
                width_s = float(event_freq_to_width.get(freq, width_s))
            except Exception:
                pass
        out = preproc.artifact_blank(df, events, fs=state.get("sfreq"), width_s=width_s)
        out.attrs["sfreq"] = state.get("sfreq") or df.attrs.get("sfreq")
        return out

    def _step_notch_filter(self, df: pd.DataFrame, notch_freq: float = 60.0, bw: float = 2.0, harm: bool = True, **state):
        out = preproc.notch_line(df, notch_freq=notch_freq, bw=bw, harm=harm, fs=state.get("sfreq"))
        out.attrs["sfreq"] = state.get("sfreq") or df.attrs.get("sfreq")
        return out

    def _step_bandpass_filter(self, df: pd.DataFrame, lowcut: float, highcut: float, order: int = 4, **state):
        out = preproc.bandpass(df, lowcut=lowcut, highcut=highcut, order=order, fs=state.get("sfreq"))
        out.attrs["sfreq"] = state.get("sfreq") or df.attrs.get("sfreq")
        return out

    def _step_decimate(self, df: pd.DataFrame, fs_new: float, **state):
        return preproc.decimate(df, fs_new=fs_new, fs=state.get("sfreq"))

    def _step_commonavg_ref(
        self,
        df: pd.DataFrame,
        skipped_chs: Iterable[str] | None = None,
        method: str = "mean",
        zscore_threshold: float = 8.0,
        min_reference_channels: int = 4,
        **state,
    ):
        skipped = _expand_skips(skipped_chs, state)
        skipped.update(self.dataloader.bad_channels(state.get("session_id")))
        numeric = df.select_dtypes(include=[np.number])
        ref_cols = [c for c in numeric.columns if c not in skipped]
        return preproc.common_reference(
            df,
            reference_channels=ref_cols,
            method=method,
            zscore_threshold=zscore_threshold,
            min_reference_channels=min_reference_channels,
        )

    def _step_bipolar_ref(
        self,
        df: pd.DataFrame,
        skipped_chs: Iterable[str] | None = None,
        keep_unpaired: bool = False,
        **state,
    ):
        """Re-reference each contact to its next contact on the same lead.

        Adjacent contacts ``X1`` and ``X2`` become a single bipolar channel
        ``X1-X2`` holding ``X1 - X2``. Contacts that cannot be paired (lead ends,
        skipped/stim contacts, or non-adjacent gaps) are dropped unless
        ``keep_unpaired`` is set. Non-numeric columns are preserved.
        """

        skipped = _expand_skips(skipped_chs, state)
        numeric = df.select_dtypes(include=[np.number])
        groups: dict[str, list[tuple[int, str]]] = {}
        for col in numeric.columns:
            parsed = _split_channel(col)
            if parsed:
                groups.setdefault(parsed[0], []).append((parsed[1], col))

        bipolar: dict[str, pd.Series] = {}
        used: set[str] = set()
        for _, contacts in groups.items():
            contacts = sorted(contacts)
            for (n1, c1), (n2, c2) in zip(contacts[:-1], contacts[1:]):
                if n2 - n1 != 1:
                    continue
                if c1 in skipped or c2 in skipped:
                    continue
                bipolar[f"{c1}-{c2}"] = numeric[c1] - numeric[c2]
                used.update({c1, c2})

        if not bipolar:
            out = df.copy()
            out.attrs.update(df.attrs)
            return out

        out = pd.DataFrame(bipolar, index=df.index)
        if keep_unpaired:
            for col in numeric.columns:
                if col not in used:
                    out[col] = numeric[col]
        for col in df.columns:
            if col not in numeric.columns:
                out[col] = df[col]
        out.attrs.update(df.attrs)
        out.attrs["sfreq"] = state.get("sfreq") or df.attrs.get("sfreq")
        return out

    def _step_laplacian_ref(
        self,
        df: pd.DataFrame,
        skipped_chs: Iterable[str] | None = None,
        ref_grounds: Iterable[str] | None = None,
        decay: float = 1.0,
        **state,
    ):
        skipped = _expand_skips(skipped_chs, state) | set(ref_grounds or [])
        numeric = df.select_dtypes(include=[np.number])
        out = df.copy()
        groups: dict[str, list[tuple[int, str]]] = {}
        for col in numeric.columns:
            parsed = _split_channel(col)
            if parsed:
                groups.setdefault(parsed[0], []).append((parsed[1], col))
        for _, contacts in groups.items():
            contacts = sorted(contacts)
            by_num = {n: col for n, col in contacts}
            for n, col in contacts:
                if col in skipped:
                    continue
                neigh = [by_num[k] for k in (n - 1, n + 1) if k in by_num and by_num[k] not in skipped]
                if not neigh:
                    continue
                weights = np.array([np.exp(-abs(_split_channel(ch)[1] - n) / max(decay, 1e-9)) for ch in neigh])
                ref = numeric[neigh].mul(weights / weights.sum(), axis=1).sum(axis=1)
                out[col] = numeric[col] - ref
        out.attrs.update(df.attrs)
        return out

    def _step_reject_bad_channels(
        self,
        df: pd.DataFrame,
        bad_channels: Iterable[str] | str | None = None,
        method: str = "drop",
        detection_params: dict | None = None,
        **state,
    ):
        if bad_channels is None:
            bad_channels = self.dataloader.bad_channels(state.get("session_id"))
        out = reject_bad_channels(df, bad_channels=bad_channels, method=method, detection_params=detection_params)
        out.attrs["sfreq"] = state.get("sfreq") or df.attrs.get("sfreq")
        return out

    @classmethod
    def install_standard_scripts(cls) -> None:
        """Reset preprocessing to ERPy's built-in execution path."""

        return None

    @classmethod
    def use_standard_scripts(cls, bash_type: str = "simple") -> None:
        """Select ERPy's built-in local preprocessing hooks."""

        cls.PYTHON_SCRIPT = None
        cls.BASH_SCRIPT = None

    def queue_job(self, pipeline_steps=None, session_ids: Iterable[str] | None = None, backend: str | None = None, **kwargs: Any):
        """Process selected acquisitions and return the job-runner summary."""

        runner = JobRunner(
            backend=backend,
            queue=kwargs.get("queue"),
            numcpu=kwargs.get("numcpu"),
            memory=kwargs.get("memory"),
            job_name=kwargs.get("job_name", "erpy_pipeline"),
        )
        steps = pipeline_steps or self.get_standard_pipeline(kwargs.get("pipeline_name", "blank_filt"))
        sessions = list(session_ids or self.dataloader.session_ids)
        lines = []
        for sid in sessions:
            pairs = kwargs.get("stim_pairs") or self.dataloader.stim_pairs(sid)
            if isinstance(pairs, str):
                pairs = [p.strip() for p in pairs.split(",") if p.strip()]
            for sp in pairs:
                self.process_stim_session(sid, sp, steps, save_processed=True, overwrite=bool(kwargs.get("overwrite_existing", False)))
                lines.append(f"{sid}/{sp}: processed")
        return runner, "\n".join(lines), ""

    @property
    def epochs(self) -> EpochsRunner:
        """Return the epoch-job accessor bound to this pipeline."""

        return EpochsRunner(self)


def _event_processing_window(
    event_times: list[Any],
    *,
    tmin: float,
    tmax: float,
    margin_s: float | None,
) -> tuple[Any, Any] | None:
    if margin_s is None or not event_times:
        return None
    margin_s = max(float(margin_s), 0.0)
    values = pd.Series(event_times)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return (
            float(numeric.min()) + float(tmin) - margin_s,
            float(numeric.max()) + float(tmax) + margin_s,
        )
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().mean() > 0.8:
        start = parsed.min() + pd.to_timedelta(float(tmin) - margin_s, unit="s")
        stop = parsed.max() + pd.to_timedelta(float(tmax) + margin_s, unit="s")
        return start, stop
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return (
        float(numeric.min()) + float(tmin) - margin_s,
        float(numeric.max()) + float(tmax) + margin_s,
    )


def _crop_continuous_window(df: pd.DataFrame, start: Any, stop: Any) -> pd.DataFrame:
    attrs = dict(df.attrs)
    if isinstance(df.index, pd.DatetimeIndex):
        selected = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(stop))].copy()
    else:
        values = pd.to_numeric(pd.Series(df.index), errors="coerce").to_numpy(dtype=float)
        selected = df.loc[(values >= float(start)) & (values <= float(stop))].copy()
    if selected.empty:
        raise ValueError(f"ERP processing window {start!r} to {stop!r} contains no samples")
    selected.attrs.update(attrs)
    selected.attrs["processing_window_start"] = str(start)
    selected.attrs["processing_window_stop"] = str(stop)
    return selected


def _stim_channels(stim_pair: str) -> list[str]:
    parts = str(stim_pair).split("_")
    if len(parts) >= 3:
        return [f"{parts[0]}{parts[1]}", f"{parts[0]}{parts[2]}"]
    return []


def _split_channel(label: str) -> tuple[str, int] | None:
    match = re_match_channel(str(label))
    if match is None:
        return None
    return match


def re_match_channel(label: str) -> tuple[str, int] | None:
    import re

    text = label.replace("_", "")
    match = re.match(r"([A-Za-z]+)(\d+)$", text)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _expand_skips(skipped_chs: Iterable[str] | None, state: dict[str, Any]) -> set[str]:
    if skipped_chs is None:
        return set(state.get("stim_chs") or [])
    values = set(skipped_chs)
    if "stim_chs" in values:
        values.remove("stim_chs")
        values.update(state.get("stim_chs") or [])
    return values
