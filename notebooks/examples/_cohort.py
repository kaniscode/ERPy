"""Read the restricted, deidentified trial export for the lead cohort example.

Only the numeric trial export is read. The adjacent private locator metadata
is deliberately not loaded or returned. No recording data are distributed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ERPy import Epochs


INPUT_SHA256 = "918a15e56a857f173d41d7b29dbe6e92bd06c20915e5ae56805d64b3a35b55ad"
DISPLAY_CHANNEL = "Subgenual cingulate recording"


def load_cohort_example() -> tuple[Epochs, pd.DataFrame, dict]:
    """Return the verified selected recording with generic display labels."""
    directory = os.environ.get("ERPY_COHORT_EXAMPLE_DIR")
    if not directory:
        raise FileNotFoundError(
            "Set ERPY_COHORT_EXAMPLE_DIR to the authorized cohort method-example "
            "directory containing all_trials.csv.gz. The restricted recording "
            "is not distributed; stored notebook figures can be viewed on GitHub."
        )
    path = Path(directory).expanduser() / "all_trials.csv.gz"
    if not path.is_file():
        raise FileNotFoundError("The configured cohort directory lacks all_trials.csv.gz")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != INPUT_SHA256:
        raise ValueError("Cohort input checksum does not match the documented recording")
    table = pd.read_csv(path)
    required = {"trial", "time_ms", "amplitude_uv", "trial_status", "artifact_reason"}
    if set(table.columns) != required:
        raise ValueError("Unexpected cohort trial-export columns")
    if table.duplicated(["trial", "time_ms"]).any():
        raise ValueError("Duplicate cohort trial/time samples")
    if table.groupby("trial")["trial_status"].nunique().ne(1).any():
        raise ValueError("Inconsistent source QC status within a trial")
    status = table.groupby("trial", sort=True).trial_status.first()
    if not status.isin(["retained", "excluded"]).all():
        raise ValueError("Unknown source trial status")
    matrix = table.pivot(index="trial", columns="time_ms", values="amplitude_uv").sort_index(axis=1)
    if matrix.shape != (25, 2001) or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Incomplete or unexpected cohort recording dimensions")
    times = matrix.columns.to_numpy(dtype=float) / 1000.0
    if not np.allclose(np.diff(times), 0.0005) or not np.allclose(times[[0, -1]], [-0.4, 0.6]):
        raise ValueError("Unexpected cohort sampling interval or epoch window")
    # Replace original trial labels with consecutive example-only numbers.
    trial_qc = pd.DataFrame({"trial": np.arange(len(matrix)), "source_status": status.reindex(matrix.index).to_numpy()})
    index = pd.MultiIndex.from_product([np.arange(len(matrix)), times], names=["epoch", "time"])
    frame = pd.DataFrame({DISPLAY_CHANNEL: matrix.to_numpy().reshape(-1)}, index=index)
    epochs = Epochs(
        frame, sfreq=2000.0, tmin=-0.4, tmax=0.6, baseline=(-0.4, -0.03),
        stim_ch=[], zero_time=0.015,
        metadata={"source": "deidentified CNS/ACC–PAG cohort recording", "source_qc": "retained/excluded status from the generating analysis"},
    )
    provenance = {
        "data_source": "deidentified CNS/ACC–PAG cohort recording",
        "stimulation": "ACC stimulation",
        "recording": DISPLAY_CHANNEL,
        "input_sha256": INPUT_SHA256,
        "n_trials": len(matrix),
        "n_source_retained": int(status.eq("retained").sum()),
        "sampling_hz": 2000.0,
        "exported_window_ms": [-400.0, 600.0],
        "selection": "previously selected method illustration; outcome-enriched",
        "analysis_scope": "one exported recording contact; acquisition family unavailable in this export",
    }
    return epochs, trial_qc, provenance
