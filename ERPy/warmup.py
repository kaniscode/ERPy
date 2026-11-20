"""Precompute event, processed-signal, and epoch caches before cohort-scale analysis."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from .dataloader import DataLoader
from .defaults import DEFAULT_BASELINE, DEFAULT_PIPELINE, DEFAULT_TMAX, DEFAULT_TMIN
from .events import detect_events_from_artifacts
from .pipeline import Pipeline
from .utils.utils import parse_event_times


@dataclass
class WarmConfig:
    """Settings applied to every acquisition during a warm-up run."""

    pipeline: str = DEFAULT_PIPELINE
    tmin: float = DEFAULT_TMIN
    tmax: float = DEFAULT_TMAX
    baseline: tuple[float, float] | None = DEFAULT_BASELINE
    event_method: str = "auto"
    overwrite_events: bool = False
    overwrite_epochs: bool = False


_WARM_CONFIG_FIELDS = frozenset(field.name for field in dataclasses.fields(WarmConfig))


def warm_session(
    dataloader: DataLoader,
    session_id: str,
    stim_pairs: Iterable[str] | None = None,
    config: WarmConfig | None = None,
    **event_kwargs,
) -> pd.DataFrame:
    """Create events and epoch caches for one session.

    Returns one status row per stimulation pair. A failure is recorded in the
    ``stage`` and ``error`` columns instead of interrupting the run.
    ``event_kwargs`` are forwarded to event detection.
    """

    config = config or WarmConfig()
    pipeline = Pipeline(dataloader)
    steps = pipeline.get_standard_pipeline(config.pipeline)
    rows = []
    for stim_pair in list(stim_pairs) if stim_pairs is not None else dataloader.stim_pairs(session_id):
        status = {"session_id": session_id, "stim_pair": stim_pair}
        event_status = "cached"
        if config.overwrite_events or not dataloader.get_event_path(session_id, stim_pair).exists():
            try:
                detect_events_from_artifacts(
                    dataloader,
                    session_id,
                    stim_pair,
                    method=config.event_method,
                    save_events=True,
                    **event_kwargs,
                )
                event_status = "created"
            except Exception as exc:
                rows.append({**status, "ok": False, "stage": "events", "error": str(exc)})
                continue
        try:
            event_info = pipeline._load_event_info(session_id, stim_pair)
            epochs, _ = pipeline.process_and_epoch(
                session_id,
                stim_pair,
                steps,
                event_times=parse_event_times(event_info["times"]),
                tmin=config.tmin,
                tmax=config.tmax,
                baseline=config.baseline,
                save_processed=True,
                save_epochs=True,
            )
        except Exception as exc:
            rows.append({**status, "ok": False, "stage": "epoch", "error": str(exc)})
            continue
        rows.append(
            {
                **status,
                "ok": True,
                "stage": "complete",
                "event_status": event_status,
                "n_epochs": epochs.n_trials(),
                "error": "",
            }
        )
    return pd.DataFrame(rows)


def warm_patient(
    patient_id: str,
    config_path: str | None = None,
    session_ids: Iterable[str] | None = None,
    stim_pairs: Iterable[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Warm the caches of selected sessions (default: all) for one patient.

    Keyword arguments naming a :class:`WarmConfig` field configure the run;
    all others are forwarded to event detection.
    """

    dataloader = DataLoader(patient_id, config_path=config_path)
    config = WarmConfig(**{key: value for key, value in kwargs.items() if key in _WARM_CONFIG_FIELDS})
    event_kwargs = {key: value for key, value in kwargs.items() if key not in _WARM_CONFIG_FIELDS}
    sessions = list(session_ids) if session_ids is not None else dataloader.session_ids
    tables = [
        warm_session(dataloader, session_id, stim_pairs=stim_pairs, config=config, **event_kwargs)
        for session_id in sessions
    ]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def warm_cohort(patient_ids: Iterable[str], config_path: str | None = None, **kwargs) -> pd.DataFrame:
    """Warm the caches of several patients and concatenate their status tables."""

    tables = []
    for patient_id in patient_ids:
        table = warm_patient(patient_id, config_path=config_path, **kwargs)
        table.insert(0, "patient_id", patient_id)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def submit_warmup_jobs(
    patient_id: str,
    config_path: str | None = None,
    session_ids: Iterable[str] | None = None,
    backend: str | None = None,
    one_job_per: str = "session",
    **kwargs,
) -> pd.DataFrame:
    """Run a patient warm-up and return it in the shape of a submission manifest.

    Scheduler dispatch is kept transparent: the work runs immediately through
    the local code path, and the returned table mirrors what a job submission
    would report.
    """

    result = warm_patient(patient_id, config_path=config_path, session_ids=session_ids, **kwargs)
    if result.empty:
        return result
    result.insert(0, "patient_id", patient_id)
    result["backend"] = backend or "local"
    result["submitted"] = result["ok"]
    result["one_job_per"] = one_job_per
    return result
