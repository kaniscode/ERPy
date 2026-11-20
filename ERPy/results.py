"""The result object returned by the one-call analysis and its QC annotation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .bad_channels import ArtifactResponseReport
from .epochs import Epochs

#: Detector method whose rows form the primary result.
_PRIMARY_METHOD = "crp_energy"

#: Columns that together identify one recorded response.
_RESPONSE_IDENTITY_COLUMNS = ("patient_id", "session_id", "stim_pair", "acquisition_id", "raw_file", "channel")

#: Channel-level artifact summary columns copied onto the detection table.
_ARTIFACT_SUMMARY_COLUMNS = (
    "channel",
    "n_epochs",
    "n_bad_response",
    "n_clean_response",
    "bad_response_fraction",
    "n_hard_artifact_response",
    "hard_artifact_fraction",
    "artifact_reasons",
    "max_peak_abs",
    "max_ptp",
    "max_saturation_fraction",
    "max_plateau_fraction",
    "max_plateau_run_s",
    "max_rail_fraction",
)

_TRUE_STRINGS = {"true", "1", "yes", "y"}


@dataclass(repr=False)
class AnalysisResult:
    """One-call result for the common ERPy stimulation-response workflow.

    Attributes
    ----------
    epochs
        Epochs as extracted, before trial-by-channel artifact rejection.
    clean_epochs
        Epochs after artifact rejection; the input to the detectors.
    detections
        Long-form detector table (one row per channel and method).
    qc_detections
        ``detections`` annotated with artifact-QC eligibility columns.
    artifact_report, artifact_summary
        Trial-by-channel artifact scores and their per-channel summary.
    metadata
        Audit record of every effective setting; written by :meth:`save`.

    Examples
    --------
    >>> result = patient.analyze("EXAMPLE_SESSION", "STIM_A_1_2")
    >>> result.primary_significant[["channel", "q_joint"]]
    >>> result.save("analysis_tables")
    """

    epochs: Epochs
    detections: pd.DataFrame
    artifact_report: ArtifactResponseReport
    artifact_summary: pd.DataFrame
    qc_detections: pd.DataFrame
    clean_epochs: Epochs | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        label = "/".join(
            str(self.metadata[key]) for key in ("patient_id", "session_id", "stim_pair") if key in self.metadata
        )
        n_trials, _, n_channels = self.epochs.shape
        parts = [
            f"{n_trials} trials, {n_channels} channels",
            f"{len(self.primary_significant)} primary responses after QC",
        ]
        return f"<AnalysisResult {label} | {' | '.join(parts)}>" if label else f"<AnalysisResult | {' | '.join(parts)}>"

    @property
    def significant(self) -> pd.DataFrame:
        """Return significant response rows from the full detection table."""

        return _significant_rows(self.detections)

    @property
    def primary_significant(self) -> pd.DataFrame:
        """Return one CRP-energy row per primary response after artifact QC."""

        data = self.qc_detections.copy()
        if "primary_qc_pass" not in data.columns:
            return data.iloc[0:0].copy()
        data = data[data["primary_qc_pass"].fillna(False)].copy()
        if "method" in data.columns:
            primary_rows = data[data["method"].astype(str).eq(_PRIMARY_METHOD)]
            if not primary_rows.empty:
                return primary_rows.reset_index(drop=True)
        identity = [column for column in _RESPONSE_IDENTITY_COLUMNS if column in data.columns]
        if identity:
            data = data.drop_duplicates(identity)
        return data.reset_index(drop=True)

    @property
    def qc_pass_detections(self) -> pd.DataFrame:
        """Return detection rows that pass response-artifact QC."""

        if self.qc_detections.empty or "artifact_qc_pass" not in self.qc_detections.columns:
            return self.qc_detections.copy()
        return self.qc_detections[self.qc_detections["artifact_qc_pass"].fillna(False)].copy()

    def save(self, output_dir: str | Path, prefix: str | None = None) -> dict[str, Path]:
        """Save the standard tables and deterministic audit metadata.

        The metadata record includes resolved epoch and pipeline options,
        whether persistent-channel screening was configured, effective
        response-artifact settings, detector windows and overrides, and one
        method-native parameter record for every executed detector.

        This method writes only to ``output_dir``. It does not upload data,
        contact a registry, or create a release-record identifier.
        Projects that use an opaque release-record label should keep its local
        mapping under their own governance controls.

        Returns
        -------
        dict
            Path of every written file, keyed by table name.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix or _result_prefix(self.metadata)
        tables = {
            "detections": self.detections,
            "detections_qc": self.qc_detections,
            "detections_qc_pass": self.qc_pass_detections,
            "detections_primary": self.primary_significant,
            "artifact_response_table": self.artifact_report.table,
            "artifact_channel_summary": self.artifact_summary,
            "zero_anchor_report": self.epochs.post_artifact_anchor_report(),
        }
        paths = {name: output_dir / f"{prefix}_{name}.csv" for name in tables}
        for name, table in tables.items():
            table.to_csv(paths[name], index=False)
        paths["metadata"] = output_dir / f"{prefix}_metadata.json"
        paths["metadata"].write_text(
            json.dumps(_metadata_value(self.metadata), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return paths


def _metadata_value(value: Any) -> Any:
    """Convert analysis metadata to deterministic, JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_metadata_value(item) for item in sorted(value, key=str)]
    if isinstance(value, Epochs):
        return {
            "type": "Epochs",
            "n_trials": int(value.n_trials()),
            "n_channels": int(len(value.channels)),
            "sfreq": float(value.sfreq),
            "tmin": float(value.tmin),
            "tmax": float(value.tmax),
        }
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return f"{module}.{name}".lstrip(".")
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _metadata_value(item())
        except (TypeError, ValueError):
            pass
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _as_bool_series(values: pd.Series) -> pd.Series:
    """Interpret a Boolean or text column (for example read from CSV) as flags."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin(_TRUE_STRINGS)


def _annotate_detection_artifact_qc(
    detections: pd.DataFrame,
    artifact_summary: pd.DataFrame,
    *,
    max_bad_response_fraction: float,
    max_hard_artifact_fraction: float,
    min_clean_responses: int,
    exclude_boundary_peaks: bool = True,
) -> pd.DataFrame:
    """Add artifact-QC eligibility to a detection table.

    A channel passes (``artifact_qc_pass``) when its fraction of artifactual
    responses is below ``max_bad_response_fraction``, its fraction of hard
    artifacts is below ``max_hard_artifact_fraction``, at least
    ``min_clean_responses`` clean responses remain, and (optionally) its peak
    is not at the response-window boundary. ``primary_qc_pass`` and
    ``shape_magnitude_qc_pass`` intersect that eligibility with the
    corresponding detector decisions.
    """

    out = detections.copy()
    if out.empty or "channel" not in out.columns:
        out["artifact_qc_pass"] = True
        out["primary_qc_pass"] = False
        out["crp_energy_qc_pass"] = False
        out["shape_magnitude_qc_pass"] = False
        return out

    if artifact_summary.empty or "channel" not in artifact_summary.columns:
        out["bad_response_fraction"] = 0.0
        out["hard_artifact_fraction"] = 0.0
        out["n_clean_response"] = pd.to_numeric(
            out.get("n_trials", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0)
        out["artifact_reasons"] = ""
    else:
        keep = [column for column in _ARTIFACT_SUMMARY_COLUMNS if column in artifact_summary.columns]
        out = out.merge(artifact_summary[keep], on="channel", how="left", suffixes=("", "_artifact"))

    zeros = pd.Series(0.0, index=out.index)
    bad_fraction = out["bad_response_fraction"] if "bad_response_fraction" in out else zeros
    hard_fraction = out["hard_artifact_fraction"] if "hard_artifact_fraction" in out else zeros
    if "n_clean_response" in out:
        clean_count = out["n_clean_response"]
    elif {"n_epochs", "n_bad_response"}.issubset(out.columns):
        clean_count = pd.to_numeric(out["n_epochs"], errors="coerce") - pd.to_numeric(
            out["n_bad_response"], errors="coerce"
        )
    else:
        clean_count = pd.Series(0, index=out.index)
    out["bad_response_fraction"] = pd.to_numeric(bad_fraction, errors="coerce").fillna(0.0)
    out["hard_artifact_fraction"] = pd.to_numeric(hard_fraction, errors="coerce").fillna(0.0)
    out["n_clean_response"] = pd.to_numeric(clean_count, errors="coerce").fillna(0).astype(int)
    out["artifact_reasons"] = out.get("artifact_reasons", "").fillna("").astype(str)

    out["artifact_hard_fail"] = out["hard_artifact_fraction"] >= float(max_hard_artifact_fraction)
    out["artifact_insufficient_clean_responses"] = out["n_clean_response"] < int(min_clean_responses)
    if "peak_at_response_boundary" in out.columns:
        boundary_peak = _as_bool_series(out["peak_at_response_boundary"])
    else:
        boundary_peak = pd.Series(False, index=out.index)
    out["artifact_boundary_peak_fail"] = boundary_peak & bool(exclude_boundary_peaks)
    out["artifact_qc_pass"] = (
        (out["bad_response_fraction"] < float(max_bad_response_fraction))
        & ~out["artifact_hard_fail"]
        & ~out["artifact_insufficient_clean_responses"]
        & ~out["artifact_boundary_peak_fail"]
    )

    out["primary_qc_pass"] = _decision_after_qc(out, "primary_significant")
    out["crp_energy_qc_pass"] = out["primary_qc_pass"]
    out["shape_magnitude_qc_pass"] = _decision_after_qc(out, "shape_magnitude_significant")
    return out


def _decision_after_qc(table: pd.DataFrame, decision_column: str) -> pd.Series | bool:
    """Intersect a detector decision column with artifact-QC eligibility."""

    if decision_column not in table.columns:
        return False
    return _as_bool_series(table[decision_column]) & table["artifact_qc_pass"]


def _significant_rows(detections: pd.DataFrame) -> pd.DataFrame:
    """Return rows flagged by the consensus rule, else by ``significant``."""

    if detections.empty:
        return detections.copy()
    for column in ("consensus_ch", "significant"):
        if column in detections.columns:
            return detections[detections[column].fillna(False)].copy()
    return detections.copy()


def _result_prefix(metadata: dict[str, Any]) -> str:
    """Build the default file-name prefix ``<patient>_<session>_<stim_pair>``."""

    parts = [
        str(metadata.get("patient_id", "patient")),
        str(metadata.get("session_id", "session")),
        str(metadata.get("stim_pair", "stim")),
    ]
    return "_".join(part.replace("/", "-").replace(" ", "-") for part in parts if part)
