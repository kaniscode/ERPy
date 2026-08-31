"""Deterministic synthetic teaching data for the ERPy example notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ERPy import Epochs


CHANNELS = [
    "STIM_A1",
    "STIM_A2",
    "CONTACT_B1",
    "CONTACT_B2",
    "CONTACT_C1",
    "CONTACT_C2",
]


def make_epochs(
    *,
    seed: int = 23,
    n_trials: int = 18,
    sfreq: float = 1_000.0,
    response_scale: float = 1.0,
) -> Epochs:
    """Return a compact synthetic stimulation-epoch object.

    The data contain an early biphasic response, a slower component, coherent
    oscillatory activity, one low-amplitude channel, and two deliberately
    contaminated trial/contact pairs for quality-control demonstrations.
    """

    rng = np.random.default_rng(seed)
    times = np.arange(-0.5, 0.501, 1.0 / sfreq)
    n_times = len(times)
    values = rng.normal(0.0, 2.2, (n_trials, n_times, len(CHANNELS)))

    early_negative = -42.0 * np.exp(-0.5 * ((times - 0.028) / 0.008) ** 2)
    early_positive = 24.0 * np.exp(-0.5 * ((times - 0.062) / 0.014) ** 2)
    late_negative = -16.0 * np.exp(-0.5 * ((times - 0.170) / 0.052) ** 2)
    canonical = (early_negative + early_positive + late_negative) * response_scale
    channel_gain = np.asarray([1.00, 0.82, 0.58, 0.34, 0.12, 0.04])

    for trial in range(n_trials):
        gain = rng.normal(1.0, 0.10)
        latency = rng.normal(0.0, 0.0015)
        shifted = np.interp(times - latency, times, canonical, left=0.0, right=0.0)
        values[trial] += gain * shifted[:, None] * channel_gain[None, :]
        phase = rng.uniform(0, 2 * np.pi)
        burst = np.exp(-0.5 * ((times - 0.09) / 0.08) ** 2)
        values[trial, :, 0] += 2.4 * burst * np.sin(2 * np.pi * 10 * times + phase)
        values[trial, :, 1] += 1.7 * burst * np.sin(2 * np.pi * 10 * times + phase + 0.25)

    onset = (times >= 0.0) & (times <= 0.004)
    values[:, onset, :] += 90.0 * np.exp(-times[onset, None] / 0.0012)
    if n_trials > 2:
        values[2, (times >= 0.11) & (times <= 0.13), 2] = 115.0
    if n_trials > 11:
        values[11, (times >= 0.18) & (times <= 0.19), 4] = np.nan

    index = pd.MultiIndex.from_product(
        [range(n_trials), times], names=["epoch", "time"]
    )
    frame = pd.DataFrame(
        values.reshape(n_trials * n_times, len(CHANNELS)),
        index=index,
        columns=CHANNELS,
    )
    return Epochs(
        frame,
        sfreq=sfreq,
        tmin=float(times[0]),
        tmax=float(times[-1]),
        baseline=(-0.5, -0.03),
        stim_ch=["STIM_A1", "STIM_A2"],
        zero_time=0.015,
        metadata={"source": "deterministic synthetic teaching data"},
    )


def make_detection_table(epochs: Epochs) -> pd.DataFrame:
    """Return a compact detector-shaped table for plotting examples."""

    gains = dict(zip(CHANNELS, [46.0, 37.0, 27.0, 16.0, 7.0, 2.0]))
    methods = [
        "kundu_rolston",
        "keller_zscore",
        "crp_significance",
        "crp_energy",
        "crowther_gamma",
        "peak_amplitude",
        "rms_response",
    ]
    rows = []
    for channel in epochs.channels:
        amplitude = gains[channel]
        for method_index, method in enumerate(methods):
            significant = amplitude >= (18.0 + 1.8 * method_index)
            rows.append(
                {
                    "channel": channel,
                    "method": method,
                    "significant": significant,
                    "primary_significant": channel
                    in {"STIM_A1", "STIM_A2", "CONTACT_B1"},
                    "consensus_ch": amplitude >= 20.0,
                    "peak_amplitude_uv": amplitude,
                    "rms_response_uv": amplitude / 3.4,
                    "score": amplitude / 6.0,
                    "p_value": 0.006 if significant else 0.35,
                    "threshold": 0.05,
                }
            )
    return pd.DataFrame(rows)


def make_electrode_metadata() -> pd.DataFrame:
    """Return fictitious MNI-like coordinates for visualization only."""

    return pd.DataFrame(
        {
            "elec_label": CHANNELS,
            "channel": CHANNELS,
            "anat_label": [
                "left frontal",
                "left frontal",
                "left temporal",
                "left temporal",
                "right cingulate",
                "right cingulate",
            ],
            "hemisphere": ["L", "L", "L", "L", "R", "R"],
            "group": [
                "STIM_A",
                "STIM_A",
                "CONTACT_B",
                "CONTACT_B",
                "CONTACT_C",
                "CONTACT_C",
            ],
            "mni_x": [-31.0, -27.0, -48.0, -42.0, 10.0, 15.0],
            "mni_y": [20.0, 17.0, -12.0, -5.0, 22.0, 16.0],
            "mni_z": [31.0, 28.0, 5.0, 9.0, 24.0, 20.0],
        }
    )


def make_edges() -> pd.DataFrame:
    """Return a small directed response-edge table."""

    return pd.DataFrame(
        {
            "stim_pair": ["STIM_A_1_2"] * 5 + ["CONTACT_B_1_2"] * 4,
            "stim_elec": ["STIM_A1"] * 5 + ["CONTACT_B1"] * 4,
            "record_elec": [
                "STIM_A2",
                "CONTACT_B1",
                "CONTACT_B2",
                "CONTACT_C1",
                "CONTACT_C2",
                "STIM_A1",
                "STIM_A2",
                "CONTACT_C1",
                "CONTACT_C2",
            ],
            "weight": [0.82, 0.72, 0.49, 0.36, 0.19, 0.66, 0.53, 0.42, 0.27],
            "stim_region": ["left frontal"] * 5 + ["left temporal"] * 4,
            "record_region": [
                "left frontal",
                "left temporal",
                "left temporal",
                "right cingulate",
                "right cingulate",
                "left frontal",
                "left frontal",
                "right cingulate",
                "right cingulate",
            ],
        }
    )


def make_metric_table() -> pd.DataFrame:
    """Return node-by-time graph metrics for plotting examples."""

    times = np.arange(0.0, 0.301, 0.025)
    rows = []
    for node_index, node in enumerate(CHANNELS):
        center = 0.06 + 0.025 * node_index
        profile = np.exp(-0.5 * ((times - center) / 0.045) ** 2)
        for time_index, (time, value) in enumerate(zip(times, profile)):
            for metric, metric_value in (
                ("hub", value * (1.0 - 0.06 * node_index)),
                ("authority", profile[::-1][time_index]),
            ):
                rows.append(
                    {
                        "time_s": float(time),
                        "time_ms": float(time * 1_000.0),
                        "node": node,
                        "metric": metric,
                        "value": float(metric_value),
                    }
                )
    return pd.DataFrame(rows)
