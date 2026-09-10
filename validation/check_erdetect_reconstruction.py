#!/usr/bin/env python3
"""Compare reconstructed public epoch means with archived ER-detect means.

This checks timing, channel/pair identity, units and baseline preparation;
it does not compare detector accuracy or use human response labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from predict_erdetect_epochs import load_epochs


def canonical(pair: str) -> tuple[str, ...]:
    return tuple(sorted(x.strip().upper() for x in pair.split("-")))


def compare(root: Path, subjects: list[str]) -> list[dict]:
    rows = []
    for subject in subjects:
        matches = list((root / "data/ds004774/derivatives/app_detect_output").glob(f"sub-{subject}_*/erdetect_data.mat"))
        if len(matches) != 1:
            raise ValueError(f"Expected one archived output for {subject}")
        reference = loadmat(matches[0], simplify_cells=True)
        pairs = {canonical(str(p)): i for i, p in enumerate(np.atleast_1d(reference["stimpair_labels"]))}
        channels = {str(c).strip().upper(): i for i, c in enumerate(np.atleast_1d(reference["channel_labels"]))}
        times_ref = np.asarray(reference["epoch_time_s"], dtype=float)
        for path in sorted((root / "epochs" / subject).glob("*.npz")):
            values, times, names, _, pair, _ = load_epochs(path)
            if canonical(pair) not in pairs:
                continue
            if not np.allclose(times, times_ref[:len(times)], atol=1e-10, rtol=0):
                raise ValueError(f"Time-axis mismatch: {subject} {pair}")
            average = np.mean(values, axis=0)
            for index, name in enumerate(names):
                if name.upper() not in channels:
                    continue
                archived = reference["ccep_average"][channels[name.upper()], pairs[canonical(pair)], :len(times)]
                delta = average[:, index] - archived
                valid = np.isfinite(delta)
                if not valid.any():
                    continue
                rows.append({"subject": subject, "stimpair": pair, "channel": name,
                             "n_samples": int(valid.sum()), "max_abs_difference_uv": float(np.max(np.abs(delta[valid]))),
                             "root_mean_square_difference_uv": float(np.sqrt(np.mean(delta[valid] ** 2))),
                             "mean_offset_uv": float(np.mean(delta[valid])),
                             "centered_root_mean_square_difference_uv": float(np.std(delta[valid]))})
        print(f"Checked archived means for {subject}", flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--subject", action="append", required=True)
    a = p.parse_args()
    rows = compare(a.root, a.subject)
    a.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(a.output / "reconstructed_mean_comparison.csv", index=False)
    summary = {"checked_channel_pair_means": len(frame), "subjects": a.subject,
               "max_abs_difference_uv": float(frame.max_abs_difference_uv.max()),
               "median_root_mean_square_difference_uv": float(frame.root_mean_square_difference_uv.median()),
               "max_centered_root_mean_square_difference_uv": float(frame.centered_root_mean_square_difference_uv.max()),
               "note": "Direct comparison before accuracy scoring; float32 epoch storage causes small rounding differences."}
    (a.output / "reconstructed_mean_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
