#!/usr/bin/env python3
"""Validate CRP-energy classes by injecting responses into saved real baseline noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ERPy as ep


EXPECTED_CLASSES = {
    "reproducible_energetic": "reproducible_energetic",
    "reproducible_low_energy": "reproducible_low_energy",
    "energetic_inconsistent": "energetic_inconsistent",
    "polarity_reversal": "energetic_inconsistent",
    "no_response": "no_response",
}


def matched_indices(
    times: np.ndarray,
    *,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    response = np.flatnonzero(
        (times >= max(float(response_window[0]), 0.015))
        & (times <= float(response_window[1]))
    )
    baseline_candidates = np.flatnonzero(
        (times >= float(baseline_window[0]))
        & (times <= float(baseline_window[1]))
    )
    if not len(response) or len(baseline_candidates) < len(response):
        raise ValueError("Epochs do not contain matched response and baseline windows")
    return baseline_candidates[-len(response) :], response


def choose_baseline_channel(
    values: np.ndarray,
    channels: list[str],
    baseline_indices: np.ndarray,
    response_indices: np.ndarray,
    *,
    minimum_trials: int,
) -> tuple[int, np.ndarray]:
    best: tuple[int, int, np.ndarray] | None = None
    combined = np.r_[baseline_indices, response_indices]
    for channel_index, _ in enumerate(channels):
        clean = np.all(np.isfinite(values[:, combined, channel_index]), axis=1)
        baseline = values[clean][:, baseline_indices, channel_index]
        scale = float(np.sqrt(np.mean(baseline * baseline))) if baseline.size else 0.0
        score = int(clean.sum()) if np.isfinite(scale) and scale > 0 else 0
        if best is None or score > best[0]:
            best = (score, channel_index, clean)
    if best is None or best[0] < int(minimum_trials):
        raise ValueError(
            "No channel has at least {} finite matched trials".format(minimum_trials)
        )
    return best[1], best[2]


def injection_epochs(
    epochs: ep.Epochs,
    *,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
    channel: str | None = None,
    minimum_trials: int = 8,
    maximum_trials: int = 24,
    random_state: int = 42,
) -> tuple[ep.Epochs, dict[str, object]]:
    values, times, channel_names = epochs.as_array()
    times = np.asarray(times, dtype=float)
    channels = [str(channel) for channel in channel_names]
    baseline_indices, response_indices = matched_indices(
        times,
        baseline_window=baseline_window,
        response_window=response_window,
    )
    if channel is None:
        channel_index, clean = choose_baseline_channel(
            values,
            channels,
            baseline_indices,
            response_indices,
            minimum_trials=minimum_trials,
        )
    else:
        if str(channel) not in channels:
            raise KeyError("Channel {!r} is absent from saved epochs".format(channel))
        channel_index = channels.index(str(channel))
        clean = np.all(
            np.isfinite(
                values[
                    :,
                    np.r_[baseline_indices, response_indices],
                    channel_index,
                ]
            ),
            axis=1,
        )
        if int(clean.sum()) < int(minimum_trials):
            raise ValueError(
                "Channel {!r} has only {} finite matched trials".format(
                    channel,
                    int(clean.sum()),
                )
            )
    selected = np.flatnonzero(clean)[: int(maximum_trials)]
    source = np.asarray(values[selected, :, channel_index], dtype=float)
    baseline_mean = np.mean(source[:, baseline_indices], axis=1)
    source = source - baseline_mean[:, None]

    baseline_segments = source[:, baseline_indices]
    baseline_rms = np.sqrt(np.mean(baseline_segments * baseline_segments, axis=1))
    finite_rms = baseline_rms[np.isfinite(baseline_rms) & (baseline_rms > 0)]
    if not len(finite_rms):
        raise ValueError("Selected real baseline has zero or nonfinite RMS")
    noise_scale = float(np.exp(np.mean(np.log(finite_rms))))

    rng = np.random.default_rng(random_state)
    response_noise = np.roll(baseline_segments, shift=1, axis=0).copy()
    response_times = times[response_indices]
    duration = max(float(response_times[-1] - response_times[0]), 1e-3)
    first = float(response_times[0] + 0.24 * duration)
    second = float(response_times[0] + 0.58 * duration)
    waveform = (
        np.exp(-0.5 * ((response_times - first) / max(0.06 * duration, 0.008)) ** 2)
        - 0.72
        * np.exp(-0.5 * ((response_times - second) / max(0.10 * duration, 0.012)) ** 2)
    )
    waveform /= max(float(np.sqrt(np.mean(waveform * waveform))), 1e-12)
    strong = 4.0 * noise_scale
    small_fraction = 0.45
    small = small_fraction * noise_scale
    small_noise_weight = float(np.sqrt(1.0 - small_fraction**2))
    random_energy = rng.normal(size=response_noise.shape)
    random_energy -= np.mean(random_energy, axis=0, keepdims=True)
    random_energy /= max(
        float(np.sqrt(np.mean(random_energy * random_energy))),
        1e-12,
    )
    polarity = np.where(np.arange(len(source)) % 2 == 0, 1.0, -1.0)

    scenario_names = list(EXPECTED_CLASSES)
    injected = np.repeat(source[:, :, None], len(scenario_names), axis=2)
    injected[:, response_indices, scenario_names.index("no_response")] = response_noise
    injected[:, response_indices, scenario_names.index("reproducible_energetic")] = (
        response_noise + strong * waveform[None, :]
    )
    injected[:, response_indices, scenario_names.index("reproducible_low_energy")] = (
        small_noise_weight * response_noise + small * waveform[None, :]
    )
    injected[:, response_indices, scenario_names.index("energetic_inconsistent")] = (
        response_noise + strong * random_energy
    )
    injected[:, response_indices, scenario_names.index("polarity_reversal")] = (
        response_noise + polarity[:, None] * strong * waveform[None, :]
    )

    index = pd.MultiIndex.from_product(
        [range(len(source)), times],
        names=["epoch", "time"],
    )
    validation_epochs = ep.Epochs(
        pd.DataFrame(
            injected.reshape(-1, len(scenario_names)),
            index=index,
            columns=scenario_names,
        ),
        sfreq=float(epochs.sfreq),
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=baseline_window,
        metadata={"validation_source_channel": channels[channel_index]},
    )
    provenance = {
        "source_channel": channels[channel_index],
        "n_trials": int(len(source)),
        "real_baseline_geometric_rms": noise_scale,
        "strong_injection_rms": strong,
        "small_injection_rms": small,
        "random_state": int(random_state),
    }
    return validation_epochs, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--baseline-start", type=float, default=-0.5)
    parser.add_argument("--baseline-stop", type=float, default=-0.03)
    parser.add_argument("--response-start", type=float, default=0.015)
    parser.add_argument("--response-stop", type=float, default=0.35)
    parser.add_argument("--minimum-trials", type=int, default=8)
    parser.add_argument("--maximum-trials", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_window = (args.baseline_start, args.baseline_stop)
    response_window = (args.response_start, args.response_stop)
    epochs = ep.Epochs.from_hdf(args.epochs)
    validation_epochs, provenance = injection_epochs(
        epochs,
        baseline_window=baseline_window,
        response_window=response_window,
        channel=args.channel,
        minimum_trials=args.minimum_trials,
        maximum_trials=args.maximum_trials,
        random_state=args.random_state,
    )
    results = ep.detect(
        validation_epochs,
        method="crp_energy",
        baseline_window=baseline_window,
        response_window=response_window,
        alpha=0.05,
        correction="fdr_bh",
        min_clean_trials=args.minimum_trials,
        n_permutations=5000,
        canonical_energy_cv=True,
        random_state=args.random_state,
        return_arrays=False,
    )
    results["expected_classification"] = results["channel"].map(EXPECTED_CLASSES)
    results["classification_matches_expected"] = (
        results["classification"] == results["expected_classification"]
    )
    results["p_joint_identity"] = np.isclose(
        pd.to_numeric(results["p_joint"], errors="coerce"),
        results[["p_crp", "p_energy"]].apply(
            pd.to_numeric, errors="coerce"
        ).max(axis=1),
        equal_nan=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary_path = args.output.with_suffix(".json")
    summary = {
        **provenance,
        "epochs_path": str(args.epochs),
        "results_path": str(args.output),
        "all_classes_match": bool(results["classification_matches_expected"].all()),
        "joint_identity_exact": bool(results["p_joint_identity"].all()),
        "results": results[
            [
                "channel",
                "classification",
                "expected_classification",
                "p_crp",
                "p_energy",
                "p_joint",
                "q_joint",
                "significant",
            ]
        ].to_dict("records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.strict and not (
        summary["all_classes_match"] and summary["joint_identity_exact"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
