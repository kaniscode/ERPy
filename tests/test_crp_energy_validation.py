from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import ERPy as ep


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_crp_energy_real_baseline",
    ROOT / "validation" / "validate_crp_energy_real_baseline.py",
)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


def test_real_baseline_injection_fixture_recovers_component_classes() -> None:
    times = np.arange(-0.5, 0.401, 0.002)
    rng = np.random.default_rng(10)
    values = rng.normal(0.0, 1.0, (24, len(times), 2))
    index = pd.MultiIndex.from_product(
        [range(len(values)), times],
        names=["epoch", "time"],
    )
    epochs = ep.Epochs(
        pd.DataFrame(
            values.reshape(-1, 2),
            index=index,
            columns=["C1", "C2"],
        ),
        sfreq=500.0,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.5, -0.03),
    )
    injected, provenance = validator.injection_epochs(
        epochs,
        baseline_window=(-0.5, -0.03),
        response_window=(0.015, 0.35),
        channel="C1",
    )
    results = ep.detect(
        injected,
        method="crp_energy",
        baseline_window=(-0.5, -0.03),
        response_window=(0.015, 0.35),
        canonical_energy_cv=False,
        return_arrays=False,
    ).set_index("channel")

    assert provenance["source_channel"] == "C1"
    assert results["classification"].to_dict() == validator.EXPECTED_CLASSES
    assert bool(results.loc["reproducible_energetic", "significant"])
    assert not results.drop(index="reproducible_energetic")["significant"].any()
