"""Preprocessing pipelines and event-locked epoch extraction."""

from __future__ import annotations

import copy
import os
import re
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from ._legacy import LegacyScriptHooks
from ._metadata import UnsafeLegacyMetadataError, _read_metadata_node, _write_metadata_node
from .artifact_anchor import POST_ARTIFACT_ZERO, detect_post_artifact_anchor
from .bad_channels import reject_bad_channels
from .defaults import DEFAULT_BASELINE, DEFAULT_PIPELINE, DEFAULT_TMAX, DEFAULT_TMIN
from .epochs import Epochs, EpochsRunner
from .utils import preproc
from .utils.jobrunner import JobRunner
from .utils.utils import ensure_dir, infer_sfreq, parse_event_times, stable_hash

#: An ordered preprocessing recipe: ``[(step_name, parameters), ...]``.
PipelineSteps = list[tuple[str, dict[str, Any]]]
#: Either the name of a standard pipeline or an explicit list of steps.
PipelineSpec = Union[str, PipelineSteps]

# Building blocks of the standard pipelines. Their parameters are part of the
# cache file names (through ``Pipeline.pipeline_id``), so changing a value here
# deliberately invalidates previously processed data.
_BLANK = ("artifact_blank", {"width_s": 0.003})
_NOTCH = ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True})
_BANDPASS = ("bandpass_filter", {"lowcut": 0.1, "highcut": 50.0, "order": 4})
_BANDPASS_WIDE = ("bandpass_filter", {"lowcut": 0.1, "highcut": 180.0, "order": 4})
_DECIMATE = ("decimate", {"fs_new": 500.0})
_COMMON_AVERAGE = ("commonavg_ref", {"method": "robust_mean", "zscore_threshold": 8.0})
_REJECT_BAD_CHANNELS = ("reject_bad_channels", {"bad_channels": "auto", "method": "drop"})

#: Named preprocessing recipes accepted wherever a pipeline is requested.
#:
#: The standard pipelines apply a robust common-average reference after
#: blanking and filtering, excluding the stimulation contacts. ``*_noref``,
#: ``raw`` and ``none`` skip re-referencing; ``*_bipolar`` and ``*_laplacian``
#: use the alternative montages.
STANDARD_PIPELINES: dict[str, PipelineSteps] = {
    "raw": [],
    "none": [],
    "blank_filt": [_BLANK, _NOTCH, _BANDPASS, _COMMON_AVERAGE],
    "blank_filt_noref": [_BLANK, _NOTCH, _BANDPASS],
    "blank_filt_bipolar": [_BLANK, _NOTCH, _BANDPASS, ("bipolar_ref", {})],
    "blank_filt_laplacian": [_BLANK, _NOTCH, _BANDPASS, ("laplacian_ref", {})],
    "blank_dec_filt": [_BLANK, _DECIMATE, _NOTCH, _BANDPASS, _COMMON_AVERAGE],
    "blank_dec_wide": [_BLANK, _DECIMATE, _NOTCH, _BANDPASS_WIDE, _COMMON_AVERAGE],
    "blank_filt_reject": [_REJECT_BAD_CHANNELS, _BLANK, _NOTCH, _BANDPASS, _COMMON_AVERAGE],
}

#: Width of the post-anchor window subtracted from each trial. A short window
#: is far more robust than a single sample on the still-decaying artifact tail.
_POST_ANCHOR_BASELINE_S = 0.005

_CONTACT_LABEL = re.compile(r"([A-Za-z]+)(\d+)$")


class UnsafeLegacyProcessedMetadataError(UnsafeLegacyMetadataError):
    """Raised when pickle-backed processed-data metadata are not trusted."""


class Pipeline(LegacyScriptHooks):
    """Registry-based preprocessing and epoch extraction.

    Each step in :attr:`registry` maps a name to a function that transforms a
    continuous recording. A pipeline is an ordered list of
    ``(step_name, parameters)`` tuples, either one of
    :data:`STANDARD_PIPELINES` or a custom list.

    Examples
    --------
    >>> pipeline = Pipeline(dataloader)
    >>> steps = pipeline.get_standard_pipeline("blank_filt")
    >>> epochs, processed = pipeline.process_and_epoch("SESSION", "STIM_A_1_2", steps)
    """

    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.registry = {
            "artifact_blank": self._step_artifact_blank,
            "notch_filter": self._step_notch_filter,
            "bandpass_filter": self._step_bandpass_filter,
            "decimate": self._step_decimate,
            "commonavg_ref": self._step_commonavg_ref,
            "car_ref": self._step_commonavg_ref,  # alias
            "bipolar_ref": self._step_bipolar_ref,
            "laplacian_ref": self._step_laplacian_ref,
            "reject_bad_channels": self._step_reject_bad_channels,
        }

    # ------------------------------------------------------------------
    # Pipeline definitions
    # ------------------------------------------------------------------
    def get_standard_pipeline(self, name: str = DEFAULT_PIPELINE) -> PipelineSteps:
        """Return an independent copy of a named preprocessing step sequence.

        Available presets (see :data:`STANDARD_PIPELINES`) combine artifact
        blanking, line-noise filtering, band-pass filtering, optional
        decimation, and a selected reference.
        """

        if name not in STANDARD_PIPELINES:
            raise KeyError(f"Unknown standard pipeline {name!r}. Available: {sorted(STANDARD_PIPELINES)}")
        return copy.deepcopy(STANDARD_PIPELINES[name])

    def build_custom_pipeline(self, steps: PipelineSteps) -> PipelineSteps:
        """Validate and normalize a user-supplied preprocessing sequence."""

        validated = []
        for step, params in steps:
            if step not in self.registry:
                raise KeyError(f"Unknown pipeline step {step!r}. Available: {sorted(self.registry)}")
            validated.append((step, dict(params or {})))
        return validated

    def resolve_pipeline(self, pipeline: PipelineSpec) -> PipelineSteps:
        """Return validated steps for a standard-pipeline name or a step list."""

        if isinstance(pipeline, str):
            return self.get_standard_pipeline(pipeline)
        return self.build_custom_pipeline(pipeline)

    def pipeline_id(self, pipeline_steps: PipelineSteps | None) -> str:
        """Return a stable identifier for a preprocessing step sequence."""

        return stable_hash(pipeline_steps or [], length=10)

    # ------------------------------------------------------------------
    # Cache locations and event files
    # ------------------------------------------------------------------
    def _processed_path(self, session_id: str, stim_pair: str, pipeline_steps) -> Path:
        name = f"{stim_pair}_{session_id}_{self.pipeline_id(pipeline_steps)}_processed.h5"
        return self.dataloader.get_session_path(session_id, "processed") / name

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
        epoch_tag = stable_hash(
            {
                "tmin": tmin,
                "tmax": tmax,
                "baseline": baseline,
                "zero_time": zero_time,
                "artifact_anchor_params": artifact_anchor_params or {},
            },
            length=8,
        )
        name = f"{stim_pair}_{session_id}_{self.pipeline_id(pipeline_steps)}_{epoch_tag}_epochs.h5"
        return self.dataloader.get_session_path(session_id, "epochs") / name

    def _load_event_info(self, session_id: str, stim_pair: str) -> pd.DataFrame:
        path = self.dataloader.get_event_path(session_id, stim_pair)
        if not path.exists():
            raise FileNotFoundError(f"Missing event file: {path}")
        events = pd.read_csv(path)
        if "times" not in events.columns:
            raise ValueError(f"Event file must contain a 'times' column: {path}")
        return events

    def _event_times(self, session_id: str, stim_pair: str) -> np.ndarray:
        times = parse_event_times(self._load_event_info(session_id, stim_pair)["times"])
        return times if isinstance(times, np.ndarray) else times.to_numpy()

    # ------------------------------------------------------------------
    # Continuous preprocessing
    # ------------------------------------------------------------------
    def process(
        self,
        pipeline_steps: PipelineSteps | None = None,
        session_id: str | None = None,
        stim_pair: str | None = None,
        **kwargs: Any,
    ):
        """Run preprocessing for one acquisition or all matching acquisitions.

        With both ``session_id`` and ``stim_pair`` the processed recording is
        returned directly; otherwise a ``{(session_id, stim_pair): frame}``
        dictionary covers every matching acquisition.
        """

        if session_id is not None and stim_pair is not None:
            return self.process_stim_session(session_id, stim_pair, pipeline_steps, **kwargs)
        results = {}
        for session in [session_id] if session_id else self.dataloader.session_ids:
            for pair in [stim_pair] if stim_pair else self.dataloader.stim_pairs(session):
                results[(session, pair)] = self.process_stim_session(session, pair, pipeline_steps, **kwargs)
        return results

    def process_stim_session(
        self,
        session_id: str,
        stim_pair: str,
        pipeline_steps: PipelineSteps | None = None,
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
        A cached result is reused only when ``input_source="processed"``;
        ``load_from_raw`` is accepted for backward compatibility and has no
        further effect.
        """

        steps = self.build_custom_pipeline(pipeline_steps or [])
        path = self._processed_path(session_id, stim_pair, steps)
        if input_source == "processed" and path.exists() and not overwrite:
            return self.load_processed_data(path)

        df = self.dataloader.load_stim_data(
            session_id,
            stim_pair,
            source=input_source,
            raw_file=raw_file,
            stim_start=stim_start,
        )
        if time_window is not None:
            df = _crop_continuous_window(df, *time_window)
        sfreq = float(
            df.attrs.get("sfreq") or self.dataloader.get_sampling_freq(session_id, stim_pair) or infer_sfreq(df)
        )
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

    def process_session(
        self, session_id: str, pipeline_steps=None, stim_pairs: Iterable[str] | None = None, **kwargs: Any
    ):
        """Preprocess selected stimulation pairs within one session."""

        pairs = list(stim_pairs) if stim_pairs is not None else self.dataloader.stim_pairs(session_id)
        return {pair: self.process_stim_session(session_id, pair, pipeline_steps, **kwargs) for pair in pairs}

    def process_all_sessions(self, pipeline_steps=None, session_ids: Iterable[str] | None = None, **kwargs: Any):
        """Preprocess selected or registered sessions and stimulation pairs."""

        sessions = list(session_ids) if session_ids is not None else self.dataloader.session_ids
        return {session: self.process_session(session, pipeline_steps, **kwargs) for session in sessions}

    # ------------------------------------------------------------------
    # Epoching
    # ------------------------------------------------------------------
    def epoch_data(
        self,
        df: pd.DataFrame,
        event_times: Iterable[Any],
        tmin: float = DEFAULT_TMIN,
        tmax: float = DEFAULT_TMAX,
        baseline: tuple[float, float] | None = None,
        zero_time: float | str | None = POST_ARTIFACT_ZERO,
        stim_ch: str | list[str] | None = None,
        artifact_anchor_params: dict[str, Any] | None = None,
    ) -> Epochs:
        """Extract event-locked trials from a continuous processed recording.

        Every trial is first corrected by its ``baseline`` window. When
        ``zero_time`` is set, each trial is then recentred on a short window
        that starts at one anchor time shared by all trials: the fixed
        ``zero_time`` (seconds), or ERPy's post-artifact anchor, which stores
        per-trial anchor diagnostics in the audit metadata.
        """

        events = list(event_times)
        if not events:
            raise ValueError("process_and_epoch requires at least one event")
        sfreq = float(df.attrs.get("sfreq") or infer_sfreq(df))
        n_pre = int(round(abs(tmin) * sfreq))
        n_post = int(round(tmax * sfreq))
        times = np.round((np.arange(n_pre + n_post + 1) - n_pre) / sfreq, decimals=9)

        anchor_params = dict(artifact_anchor_params or {})
        post_baseline_s = float(anchor_params.pop("post_baseline_s", _POST_ANCHOR_BASELINE_S))
        n_window = max(int(round(post_baseline_s * sfreq)), 1)

        values = df.select_dtypes(include=[np.number])
        trials = _extract_baselined_trials(
            values, _event_sample_positions(df.index, events), n_pre, n_post, times, baseline
        )
        if not trials:
            raise ValueError("No full epochs could be extracted; check event times and tmin/tmax")

        anchor_reports: list[dict[str, Any]] = []
        common_anchor_time = float("nan")
        if zero_time is not None:
            if zero_time == POST_ARTIFACT_ZERO:
                anchor_reports, anchor_index = _post_artifact_anchor(
                    trials, times, stim_ch=stim_ch, baseline=baseline, anchor_params=anchor_params
                )
            else:
                anchor_reports, anchor_index = _fixed_anchor(trials, times, float(zero_time))
            common_anchor_time = float(times[anchor_index])
            for report in anchor_reports:
                report["common_anchor_time_s"] = common_anchor_time
                report["common_anchor_index"] = int(anchor_index)
                report["post_baseline_window_s"] = float(n_window / sfreq)
            _subtract_window_mean(trials, anchor_index, min(anchor_index + n_window, len(times)))

        index = pd.MultiIndex.from_tuples(
            [(trial_id, float(t)) for trial_id, _ in trials for t in times],
            names=["epoch", "time"],
        )
        epochs_df = pd.DataFrame(
            np.vstack([trial.to_numpy(dtype=float) for _, trial in trials]),
            index=index,
            columns=list(values.columns),
        )
        return Epochs(
            epochs_df=epochs_df,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            stim_ch=stim_ch,
            metadata={
                "n_events_input": len(events),
                "n_epochs": len(trials),
                "signal_units": df.attrs.get("signal_units"),
                "value_scale_to_uv_applied": df.attrs.get("value_scale_to_uv_applied", 1.0),
                "reference_method": df.attrs.get("reference_method"),
                "reference_channels": df.attrs.get("reference_channels"),
                "reference_zscore_threshold": df.attrs.get("reference_zscore_threshold"),
                "zero_time_strategy": zero_time,
                "zero_anchor_reports": anchor_reports,
                "artifact_anchor_params": anchor_params,
                "common_anchor_time_s": common_anchor_time,
                "post_baseline_window_s": float(n_window / sfreq),
                "post_baseline_samples": int(n_window),
            },
            zero_time=zero_time,
        )

    def process_and_epoch(
        self,
        session_id: str,
        stim_pair: str,
        pipeline_steps: PipelineSteps | None = None,
        event_times: Iterable[Any] | None = None,
        tmin: float = DEFAULT_TMIN,
        tmax: float = DEFAULT_TMAX,
        baseline: tuple[float, float] | None = DEFAULT_BASELINE,
        zero_time: float | str | None = POST_ARTIFACT_ZERO,
        artifact_anchor_params: dict[str, Any] | None = None,
        processing_margin_s: float | None = None,
        save_processed: bool = False,
        save_epochs: bool = False,
        **kwargs: Any,
    ) -> tuple[Epochs, pd.DataFrame]:
        """Preprocess one acquisition and extract its event-locked epochs.

        Returns ``(epochs, processed)``. Event times default to the saved
        event file. ``processing_margin_s`` limits preprocessing to the span of
        the events plus that margin instead of the whole recording.
        """

        steps = self.build_custom_pipeline(pipeline_steps or [])
        if event_times is None:
            event_times = self._event_times(session_id, stim_pair)
        event_times = list(event_times)
        processed = self.process_stim_session(
            session_id,
            stim_pair,
            steps,
            save_processed=save_processed,
            event_times=event_times,
            time_window=_event_processing_window(event_times, tmin=tmin, tmax=tmax, margin_s=processing_margin_s),
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_processed_data(self, df: pd.DataFrame, path: str | Path, **metadata: Any) -> Path:
        """Atomically save processed data with deterministic JSON provenance."""

        path = Path(path)
        ensure_dir(path.parent)
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix or '.h5'}")
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
        metadata = _read_metadata_node(
            path,
            "data",
            allow_unsafe_legacy_pickle=allow_unsafe_legacy_pickle,
            legacy_error_type=UnsafeLegacyProcessedMetadataError,
            metadata_description="ERPy processed-data",
            unsafe_opt_in_example="Pipeline.load_processed_data(..., allow_unsafe_legacy_pickle=True)",
        )
        with pd.HDFStore(path, mode="r") as store:
            df = store["data"]
            if metadata is None and allow_unsafe_legacy_pickle:
                metadata = getattr(store.get_storer("data").attrs, "metadata", {}) or {}
        df.attrs.update(metadata or {})
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

    # ------------------------------------------------------------------
    # Registered preprocessing steps
    #
    # Each step receives the recording, its own parameters, and the run
    # ``state`` (session, stimulation pair, sampling frequency, stimulation
    # contacts, plus any caller context such as ``event_times``).
    # ------------------------------------------------------------------
    def _step_artifact_blank(
        self, df: pd.DataFrame, event_freq_to_width: dict | None = None, width_s: float = 0.003, **state
    ):
        events = state.get("event_times")
        if events is None:
            try:
                events = self._event_times(state["session_id"], state["stim_pair"])
            except Exception:
                events = []
        if event_freq_to_width:
            try:
                row = self.dataloader.get_stim_row(state["session_id"], state["stim_pair"])
                width_s = float(event_freq_to_width.get(float(row["stim_freq"]), width_s))
            except Exception:
                pass
        out = preproc.artifact_blank(df, events, fs=state.get("sfreq"), width_s=width_s)
        return _carry_sfreq(out, df, state)

    def _step_notch_filter(
        self, df: pd.DataFrame, notch_freq: float = 60.0, bw: float = 2.0, harm: bool = True, **state
    ):
        out = preproc.notch_line(df, notch_freq=notch_freq, bw=bw, harm=harm, fs=state.get("sfreq"))
        return _carry_sfreq(out, df, state)

    def _step_bandpass_filter(self, df: pd.DataFrame, lowcut: float, highcut: float, order: int = 4, **state):
        out = preproc.bandpass(df, lowcut=lowcut, highcut=highcut, order=order, fs=state.get("sfreq"))
        return _carry_sfreq(out, df, state)

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
        """Subtract a common average that excludes stimulation and bad contacts."""

        skipped = _expand_skips(skipped_chs, state)
        skipped.update(self.dataloader.bad_channels(state.get("session_id")))
        numeric = df.select_dtypes(include=[np.number])
        return preproc.common_reference(
            df,
            reference_channels=[column for column in numeric.columns if column not in skipped],
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

        bipolar: dict[str, pd.Series] = {}
        used: set[str] = set()
        for contacts in _contacts_by_lead(numeric.columns).values():
            for (first_number, first), (second_number, second) in zip(contacts[:-1], contacts[1:]):
                if second_number - first_number != 1:
                    continue
                if first in skipped or second in skipped:
                    continue
                bipolar[f"{first}-{second}"] = numeric[first] - numeric[second]
                used.update({first, second})

        if not bipolar:
            out = df.copy()
            out.attrs.update(df.attrs)
            return out

        out = pd.DataFrame(bipolar, index=df.index)
        if keep_unpaired:
            for column in numeric.columns:
                if column not in used:
                    out[column] = numeric[column]
        for column in df.columns:
            if column not in numeric.columns:
                out[column] = df[column]
        out.attrs.update(df.attrs)
        return _carry_sfreq(out, df, state)

    def _step_laplacian_ref(
        self,
        df: pd.DataFrame,
        skipped_chs: Iterable[str] | None = None,
        ref_grounds: Iterable[str] | None = None,
        decay: float = 1.0,
        **state,
    ):
        """Subtract the distance-weighted mean of each contact's lead neighbours."""

        skipped = _expand_skips(skipped_chs, state) | set(ref_grounds or [])
        numeric = df.select_dtypes(include=[np.number])
        out = df.copy()
        for contacts in _contacts_by_lead(numeric.columns).values():
            label_by_number = dict(contacts)
            for number, column in contacts:
                if column in skipped:
                    continue
                neighbours = [
                    (neighbour, label_by_number[neighbour])
                    for neighbour in (number - 1, number + 1)
                    if neighbour in label_by_number and label_by_number[neighbour] not in skipped
                ]
                if not neighbours:
                    continue
                weights = np.array([np.exp(-abs(neighbour - number) / max(decay, 1e-9)) for neighbour, _ in neighbours])
                labels = [label for _, label in neighbours]
                reference = numeric[labels].mul(weights / weights.sum(), axis=1).sum(axis=1)
                out[column] = numeric[column] - reference
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
        return _carry_sfreq(out, df, state)

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------
    def queue_job(
        self, pipeline_steps=None, session_ids: Iterable[str] | None = None, backend: str | None = None, **kwargs: Any
    ):
        """Process selected acquisitions and return the job-runner summary."""

        runner = JobRunner(
            backend=backend,
            queue=kwargs.get("queue"),
            numcpu=kwargs.get("numcpu"),
            memory=kwargs.get("memory"),
            job_name=kwargs.get("job_name", "erpy_pipeline"),
        )
        steps = pipeline_steps or self.get_standard_pipeline(kwargs.get("pipeline_name", DEFAULT_PIPELINE))
        overwrite = bool(kwargs.get("overwrite_existing", False))
        lines = []
        for session_id in list(session_ids or self.dataloader.session_ids):
            pairs = kwargs.get("stim_pairs") or self.dataloader.stim_pairs(session_id)
            if isinstance(pairs, str):
                pairs = [pair.strip() for pair in pairs.split(",") if pair.strip()]
            for stim_pair in pairs:
                self.process_stim_session(session_id, stim_pair, steps, save_processed=True, overwrite=overwrite)
                lines.append(f"{session_id}/{stim_pair}: processed")
        return runner, "\n".join(lines), ""

    @property
    def epochs(self) -> EpochsRunner:
        """Return the epoch-job accessor bound to this pipeline."""

        return EpochsRunner(self)


# ----------------------------------------------------------------------
# Epoching helpers
# ----------------------------------------------------------------------
def _event_sample_positions(index: pd.Index, events: list[Any]) -> np.ndarray:
    """Return the row position of each event in a continuous recording index."""

    if isinstance(index, pd.DatetimeIndex):
        return index.searchsorted(pd.to_datetime(pd.Series(events), errors="coerce"))
    index_seconds = pd.to_numeric(pd.Series(index), errors="coerce").to_numpy(dtype=float)
    event_seconds = pd.to_numeric(pd.Series(events), errors="coerce").to_numpy(dtype=float)
    return np.searchsorted(index_seconds, event_seconds)


def _extract_baselined_trials(
    values: pd.DataFrame,
    positions: np.ndarray,
    n_pre: int,
    n_post: int,
    times: np.ndarray,
    baseline: tuple[float, float] | None,
) -> list[tuple[int, pd.DataFrame]]:
    """Cut one baseline-corrected, time-indexed trial per event.

    Events too close to either end of the recording for a full epoch are
    skipped; the returned identifiers are positions in the original event list.
    """

    trials: list[tuple[int, pd.DataFrame]] = []
    for trial_id, position in enumerate(positions):
        start = int(position) - n_pre
        stop = int(position) + n_post + 1
        if start < 0 or stop > len(values):
            continue
        trial = values.iloc[start:stop].copy()
        if len(trial) != len(times):
            continue
        trial.index = times
        if baseline is not None:
            in_baseline = (times >= baseline[0]) & (times <= baseline[1])
            if in_baseline.any():
                trial = trial - trial.loc[in_baseline].mean(axis=0)
        trials.append((int(trial_id), trial))
    return trials


def _post_artifact_anchor(
    trials: list[tuple[int, pd.DataFrame]],
    times: np.ndarray,
    *,
    stim_ch: str | list[str] | None,
    baseline: tuple[float, float] | None,
    anchor_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Locate the post-artifact anchor shared by all trials.

    The anchor is detected on the trial average, which is robust to per-trial
    noise and makes the averaged waveform genuinely zero at the anchor. (A
    per-trial, single-sample anchor leaves the average non-zero because trials
    settle at slightly different times.) Per-trial anchors are still computed
    and returned as audit reports; their median is the fallback when the
    averaged waveform yields no usable anchor.
    """

    n_times = len(times)
    reports = []
    for trial_id, trial in trials:
        report = detect_post_artifact_anchor(trial, stim_ch=stim_ch, baseline_window=baseline, **anchor_params)
        reports.append({**report, "epoch": int(trial_id), "zero_time_strategy": POST_ARTIFACT_ZERO})

    columns = trials[0][1].columns
    mean_trial = pd.DataFrame(
        np.nanmean(np.stack([trial.to_numpy(dtype=float) for _, trial in trials]), axis=0),
        index=times,
        columns=columns,
    )
    mean_report = detect_post_artifact_anchor(mean_trial, stim_ch=stim_ch, baseline_window=baseline, **anchor_params)
    anchor_index = int(mean_report.get("anchor_index", -1))
    if not 0 <= anchor_index < n_times:
        per_trial = [int(report.get("anchor_index", -1)) for report in reports]
        per_trial = [index for index in per_trial if 0 <= index < n_times]
        anchor_index = int(np.median(per_trial)) if per_trial else int(np.argmin(np.abs(times)))
    return reports, anchor_index


def _fixed_anchor(
    trials: list[tuple[int, pd.DataFrame]], times: np.ndarray, zero_time: float
) -> tuple[list[dict[str, Any]], int]:
    """Use the sample nearest a caller-chosen zero time as the common anchor."""

    anchor_index = int(np.argmin(np.abs(times - zero_time)))
    reports = [
        {
            "epoch": int(trial_id),
            "anchor_time_s": float(times[anchor_index]),
            "anchor_index": int(anchor_index),
            "artifact_peak_time_s": np.nan,
            "artifact_peak_score_z": np.nan,
            "onset_score_z": np.nan,
            "onset_is_artifact": False,
            "settled": True,
            "method": "fixed_zero_time",
            "reference_channels": "",
            "zero_time_strategy": "fixed",
        }
        for trial_id, _ in trials
    ]
    return reports, anchor_index


def _subtract_window_mean(trials: list[tuple[int, pd.DataFrame]], start: int, stop: int) -> None:
    """Recentre each trial in place on its mean over samples ``start:stop``."""

    for _, trial in trials:
        window = trial.iloc[start:stop].replace([np.inf, -np.inf], np.nan)
        trial -= window.mean(axis=0).fillna(0.0)


# ----------------------------------------------------------------------
# Continuous-recording helpers
# ----------------------------------------------------------------------
def _event_processing_window(
    event_times: list[Any],
    *,
    tmin: float,
    tmax: float,
    margin_s: float | None,
) -> tuple[Any, Any] | None:
    """Return the recording span needed to epoch ``event_times``, or ``None``."""

    if margin_s is None or not event_times:
        return None
    margin_s = max(float(margin_s), 0.0)

    def numeric_window(seconds: pd.Series) -> tuple[float, float]:
        return (
            float(seconds.min()) + float(tmin) - margin_s,
            float(seconds.max()) + float(tmax) + margin_s,
        )

    values = pd.Series(event_times)
    if pd.api.types.is_numeric_dtype(values):
        return numeric_window(pd.to_numeric(values, errors="coerce").dropna())
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().mean() > 0.8:
        start = parsed.min() + pd.to_timedelta(float(tmin) - margin_s, unit="s")
        stop = parsed.max() + pd.to_timedelta(float(tmax) + margin_s, unit="s")
        return start, stop
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else numeric_window(numeric)


def _crop_continuous_window(df: pd.DataFrame, start: Any, stop: Any) -> pd.DataFrame:
    attrs = dict(df.attrs)
    if isinstance(df.index, pd.DatetimeIndex):
        selected = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(stop))].copy()
    else:
        seconds = pd.to_numeric(pd.Series(df.index), errors="coerce").to_numpy(dtype=float)
        selected = df.loc[(seconds >= float(start)) & (seconds <= float(stop))].copy()
    if selected.empty:
        raise ValueError(f"ERP processing window {start!r} to {stop!r} contains no samples")
    selected.attrs.update(attrs)
    selected.attrs["processing_window_start"] = str(start)
    selected.attrs["processing_window_stop"] = str(stop)
    return selected


def _carry_sfreq(out: pd.DataFrame, source: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    """Record the sampling frequency on a step's output and return it."""

    out.attrs["sfreq"] = state.get("sfreq") or source.attrs.get("sfreq")
    return out


def _stim_channels(stim_pair: str) -> list[str]:
    """Return the two contact labels named by a pair such as ``LACC_1_2``."""

    parts = str(stim_pair).split("_")
    if len(parts) >= 3:
        return [f"{parts[0]}{parts[1]}", f"{parts[0]}{parts[2]}"]
    return []


def _split_channel(label: str) -> tuple[str, int] | None:
    """Split a contact label such as ``LACC3`` into ``("LACC", 3)``."""

    match = _CONTACT_LABEL.match(str(label).replace("_", ""))
    if not match:
        return None
    return match.group(1), int(match.group(2))


#: Retained name of :func:`_split_channel`.
re_match_channel = _split_channel


def _contacts_by_lead(columns: Iterable[str]) -> dict[str, list[tuple[int, str]]]:
    """Group contact labels by lead as sorted ``(contact_number, label)`` lists."""

    leads: dict[str, list[tuple[int, str]]] = {}
    for column in columns:
        parsed = _split_channel(column)
        if parsed:
            leads.setdefault(parsed[0], []).append((parsed[1], column))
    return {lead: sorted(contacts) for lead, contacts in leads.items()}


def _expand_skips(skipped_chs: Iterable[str] | None, state: dict[str, Any]) -> set[str]:
    """Resolve channels excluded from a reference.

    ``None`` means the stimulation contacts; the literal entry ``"stim_chs"``
    in a list is replaced by them.
    """

    if skipped_chs is None:
        return set(state.get("stim_chs") or [])
    values = set(skipped_chs)
    if "stim_chs" in values:
        values.remove("stim_chs")
        values.update(state.get("stim_chs") or [])
    return values
