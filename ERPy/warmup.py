from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .dataloader import DataLoader
from .events import detect_events_from_artifacts
from .pipeline import Pipeline


@dataclass
class WarmConfig:
    pipeline: str = "blank_filt"
    tmin: float = -0.5
    tmax: float = 1.0
    baseline: tuple[float, float] | None = (-0.5, -0.03)
    event_method: str = "auto"
    overwrite_events: bool = False
    overwrite_epochs: bool = False


def warm_session(
    dataloader: DataLoader,
    session_id: str,
    stim_pairs: Iterable[str] | None = None,
    config: WarmConfig | None = None,
    **event_kwargs,
) -> pd.DataFrame:
    config = config or WarmConfig()
    pipeline = Pipeline(dataloader)
    steps = pipeline.get_standard_pipeline(config.pipeline)
    rows = []
    for stim_pair in (list(stim_pairs) if stim_pairs is not None else dataloader.stim_pairs(session_id)):
        event_path = dataloader.get_event_path(session_id, stim_pair)
        event_status = "cached"
        if config.overwrite_events or not event_path.exists():
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
                rows.append({"session_id": session_id, "stim_pair": stim_pair, "ok": False, "stage": "events", "error": str(exc)})
                continue
        try:
            event_info = pipeline._load_event_info(session_id, stim_pair)
            event_times = _parse_event_times(event_info["times"])
            epochs, _ = pipeline.process_and_epoch(
                session_id,
                stim_pair,
                steps,
                event_times=event_times,
                tmin=config.tmin,
                tmax=config.tmax,
                baseline=config.baseline,
                save_processed=True,
                save_epochs=True,
            )
            rows.append(
                {
                    "session_id": session_id,
                    "stim_pair": stim_pair,
                    "ok": True,
                    "stage": "complete",
                    "event_status": event_status,
                    "n_epochs": epochs.n_trials(),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append({"session_id": session_id, "stim_pair": stim_pair, "ok": False, "stage": "epoch", "error": str(exc)})
    return pd.DataFrame(rows)


def warm_patient(
    patient_id: str,
    config_path: str | None = None,
    session_ids: Iterable[str] | None = None,
    stim_pairs: Iterable[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    dataloader = DataLoader(patient_id, config_path=config_path)
    config = WarmConfig(**{k: v for k, v in kwargs.items() if k in WarmConfig.__annotations__})
    event_kwargs = {k: v for k, v in kwargs.items() if k not in WarmConfig.__annotations__}
    rows = []
    for session_id in (list(session_ids) if session_ids is not None else dataloader.session_ids):
        rows.append(warm_session(dataloader, session_id, stim_pairs=stim_pairs, config=config, **event_kwargs))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def warm_cohort(patient_ids: Iterable[str], config_path: str | None = None, **kwargs) -> pd.DataFrame:
    rows = []
    for patient_id in patient_ids:
        df = warm_patient(patient_id, config_path=config_path, **kwargs)
        df.insert(0, "patient_id", patient_id)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def submit_warmup_jobs(
    patient_id: str,
    config_path: str | None = None,
    session_ids: Iterable[str] | None = None,
    backend: str | None = None,
    one_job_per: str = "session",
    **kwargs,
) -> pd.DataFrame:
    # The recovered package keeps scheduler dispatch transparent: local backend
    # runs immediately, while the returned table mirrors a submission manifest.
    result = warm_patient(patient_id, config_path=config_path, session_ids=session_ids, **kwargs)
    if result.empty:
        return result
    result.insert(0, "patient_id", patient_id)
    result["backend"] = backend or "local"
    result["submitted"] = result["ok"]
    result["one_job_per"] = one_job_per
    return result


def _parse_event_times(values):
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy()
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed if parsed.notna().mean() > 0.8 else values
