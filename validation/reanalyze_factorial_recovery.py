#!/usr/bin/env python3
"""Estimate recovery uncertainty while preserving the synthetic design weights.

Input is the frozen benchmark's scenario_results.csv.gz, one row per family.
This reanalyses saved decisions; it does not regenerate signals or fit a detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(path: Path, *, seed: int = 20260907, resamples: int = 20000) -> dict:
    rows = pd.read_csv(path)
    needed = {"signal_present", "n_trials_total", "snr_nominal", "artifact_fraction",
              "crp__energy__call", "crp__energy__evaluable"}
    if not needed.issubset(rows):
        raise ValueError(f"Missing columns: {sorted(needed - set(rows))}")
    for column in ("signal_present", "crp__energy__call", "crp__energy__evaluable"):
        if rows[column].isna().any() or not rows[column].isin([True, False]).all():
            raise ValueError(f"{column} must contain boolean values without missingness")
    target = rows.loc[rows.signal_present].copy()
    if target.empty:
        raise ValueError("No injected-target families")
    cells = list(target.groupby(["n_trials_total", "snr_nominal", "artifact_fraction"], sort=True))
    rng = np.random.default_rng(seed)
    boot_calls = np.zeros(resamples, dtype=np.int64)
    boot_evaluable = np.zeros(resamples, dtype=np.int64)
    cell_summary = []
    for key, cell in cells:
        calls = cell["crp__energy__call"].to_numpy(dtype=bool)
        evaluable = cell["crp__energy__evaluable"].to_numpy(dtype=bool)
        if np.any(calls & ~evaluable):
            raise ValueError("A detected target must be evaluable")
        index = rng.integers(0, len(cell), size=(resamples, len(cell)))
        boot_calls += calls[index].sum(axis=1)
        boot_evaluable += evaluable[index].sum(axis=1)
        cell_summary.append(dict(zip(("trials", "snr", "mask_fraction"), map(float, key))) |
                            {"families": len(cell), "evaluable": int(evaluable.sum()), "calls": int(calls.sum())})
    if np.any(boot_evaluable == 0):
        raise ValueError("Conditional recovery undefined in at least one bootstrap resample")
    total_calls = int(target["crp__energy__call"].sum())
    total_evaluable = int(target["crp__energy__evaluable"].sum())
    return {
        "input_name": path.name,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "method": "Percentile bootstrap of whole families independently within each fixed factorial cell",
        "seed": seed, "resamples": resamples, "design_cells": len(cells),
        "injected_families": len(target), "evaluable_targets": total_evaluable, "detected_targets": total_calls,
        "conditional_recovery": total_calls / total_evaluable,
        "conditional_95_percentile_interval": np.quantile(boot_calls / boot_evaluable, [.025, .975]).tolist(),
        "overall_recovery": total_calls / len(target),
        "overall_95_percentile_interval": np.quantile(boot_calls / len(target), [.025, .975]).tolist(),
        "interpretation": "Equal weighting of the declared design cells; not clinical sensitivity or prevalence-weighted performance. Evaluability is fixed by this design and receives no sampling interval.",
        "cells": cell_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("resamples must be positive")
    result = summarize(args.input, seed=args.seed, resamples=args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
