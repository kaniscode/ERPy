from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_public_spes",
    ROOT / "validation" / "validate_public_spes.py",
)
public_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = public_validation
SPEC.loader.exec_module(public_validation)


def test_public_lowpass_marks_gamma_unavailable_and_excludes_it_from_consensus() -> None:
    methods = [
        "kundu_rolston",
        "keller_zscore",
        "crp_significance",
        "crp_energy",
        "crowther_gamma",
        "peak_amplitude",
        "rms_response",
    ]
    calls = {
        "single_plus_invalid_gamma": {
            "crp_significance",
            "crp_energy",
            "crowther_gamma",
        },
        "two_valid_plus_invalid_gamma": {
            "kundu_rolston",
            "keller_zscore",
            "crp_energy",
            "crowther_gamma",
        },
    }
    rows = []
    for channel, significant_methods in calls.items():
        for method in methods:
            rows.append(
                {
                    "channel": channel,
                    "method": method,
                    "method_available": True,
                    "availability_reason": "",
                    "detector_input": "primary_epochs",
                    "significant": method in significant_methods,
                    "score": 4.2,
                    "p_value": 0.001,
                    "threshold": 0.05,
                    "notes": "",
                    "signi_snr": 4.2,
                    "signi_p_value_raw": 0.001,
                    "signi_p_value_bonferroni": 0.01,
                    "high_gamma_z": 3.0,
                }
            )

    corrected = public_validation._finalize_public_detection_availability(
        pd.DataFrame(rows),
        primary_passband_hz=(0.5, 80.0),
        min_consensus=2,
    )
    gamma = corrected[corrected["method"].eq("crowther_gamma")]
    assert len(gamma) == 2
    assert not gamma["method_available"].any()
    assert not gamma["significant"].any()
    assert gamma["availability_reason"].str.contains("0.5-80 Hz", regex=False).all()
    assert gamma["availability_reason"].str.contains("no independent wideband_epochs", regex=False).all()
    assert gamma[
        [
            "score",
            "p_value",
            "threshold",
            "signi_snr",
            "signi_p_value_raw",
            "signi_p_value_bonferroni",
            "high_gamma_z",
        ]
    ].isna().all().all()

    channel_status = corrected.drop_duplicates("channel").set_index("channel")
    assert channel_status.loc["single_plus_invalid_gamma", "n_methods_available"] == len(
        public_validation.PUBLIC_CONSENSUS_METHODS
    )
    assert channel_status.loc["single_plus_invalid_gamma", "n_methods_significant"] == 1
    assert not bool(channel_status.loc["single_plus_invalid_gamma", "consensus_ch"])
    assert channel_status.loc["two_valid_plus_invalid_gamma", "n_methods_significant"] == 2
    assert bool(channel_status.loc["two_valid_plus_invalid_gamma", "consensus_ch"])
    assert bool(channel_status.loc["two_valid_plus_invalid_gamma", "consensus"])
