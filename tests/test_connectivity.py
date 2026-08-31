from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import signal

import ERPy as ep
from ERPy.connectivity import _pairwise_granger_score
from ERPy.epochs import Epochs


def _causal_epochs(n_trials: int = 20, n_times: int = 220, sfreq: float = 200.0) -> Epochs:
    rng = np.random.default_rng(11)
    times = np.arange(n_times) / sfreq
    rows, index = [], []
    for trial in range(n_trials):
        x = rng.standard_normal(n_times)
        y = rng.standard_normal(n_times) * 0.2
        z = rng.standard_normal(n_times)
        for t in range(2, n_times):
            x[t] += 0.45 * x[t - 1]
            y[t] += 0.65 * y[t - 1] + 0.85 * x[t - 1]
        rows.append(np.vstack([x, y, z]).T)
        index.extend((trial, float(t)) for t in times)
    df = pd.DataFrame(
        np.vstack(rows),
        index=pd.MultiIndex.from_tuples(index, names=["epoch", "time"]),
        columns=["X", "Y", "Z"],
    )
    return Epochs(epochs_df=df, sfreq=sfreq, tmin=0.0, tmax=float(times[-1]), baseline=None)


def test_epoch_mutual_information_and_granger():
    epochs = _causal_epochs()
    mi = epochs.spectral.mi_matrix(channels=["X", "Y", "Z"], time_window=(0.1, 0.9), n_bins=12)
    assert mi.loc["X", "Y"] > mi.loc["X", "Z"]

    granger, pvals = epochs.spectral.granger_matrix(
        channels=["X", "Y", "Z"],
        time_window=(0.1, 1.0),
        maxlag=2,
        return_pvalues=True,
    )
    assert granger.loc["X", "Y"] > granger.loc["Y", "X"]
    assert pvals.loc["X", "Y"] < 0.05


def test_post_stim_rest_window_and_dataframe_connectivity():
    sfreq = 200.0
    index = pd.date_range("2026-01-01 12:00:00", periods=1000, freq=pd.to_timedelta(1 / sfreq, unit="s"))
    t = np.arange(len(index)) / sfreq
    df = pd.DataFrame(
        {
            "A": np.sin(2 * np.pi * 10 * t),
            "B": np.sin(2 * np.pi * 10 * t + 0.4),
            "NOISE": np.random.default_rng(4).standard_normal(len(t)),
        },
        index=index,
    )
    rest = ep.post_stim_rest_window(df, pd.Timestamp("2026-01-01 12:00:01"), start_offset_s=0.5, duration_s=2.0)
    assert not rest.empty
    plv = ep.connectivity_matrix_from_dataframe(rest, sfreq, method="plv", band=(8, 12))
    assert plv.loc["A", "B"] > plv.loc["A", "NOISE"]


def test_vectorized_continuous_connectivity_matches_pairwise_reference():
    rng = np.random.default_rng(17)
    sfreq = 250.0
    channels = [f"C{index}" for index in range(6)]
    values = rng.standard_normal((4000, len(channels)))
    values[:, 1] += 0.8 * values[:, 0]
    frame = pd.DataFrame(values, columns=channels)

    coherence = ep.coherence_matrix_from_dataframe(frame, sfreq, band=(8.0, 13.0))
    for source in range(len(channels)):
        for target in range(source + 1, len(channels)):
            frequencies, pairwise = signal.coherence(
                values[:, source],
                values[:, target],
                fs=sfreq,
            )
            selected = (frequencies >= 8.0) & (frequencies <= 13.0)
            assert coherence.iloc[source, target] == pytest.approx(
                float(np.mean(pairwise[selected])),
                abs=1e-12,
            )

    granger = ep.granger_causality_matrix_from_dataframe(
        frame,
        maxlag=5,
    )
    for source in range(len(channels)):
        for target in range(len(channels)):
            if source == target:
                continue
            reference, _ = _pairwise_granger_score(
                values[:, source],
                values[:, target],
                lag=5,
                min_samples=30,
            )
            assert granger.iloc[source, target] == pytest.approx(reference, abs=1e-11)


def test_multitaper_coherence_returns_shared_and_lagged_alpha_sensitivity():
    sfreq = 500.0
    time = np.arange(0.0, 2.0, 1.0 / sfreq)
    rng = np.random.default_rng(29)
    reference = np.sin(2.0 * np.pi * 10.0 * time)
    frame = pd.DataFrame(
        {
            "reference": reference + 0.08 * rng.standard_normal(len(time)),
            "in_phase": reference + 0.08 * rng.standard_normal(len(time)),
            "quadrature": np.sin(2.0 * np.pi * 10.0 * time + np.pi / 2.0)
            + 0.08 * rng.standard_normal(len(time)),
            "noise": rng.standard_normal(len(time)),
        }
    )

    coherence, imaginary = ep.multitaper_coherence_matrices_from_dataframe(
        frame, sfreq, band=(8.0, 13.0)
    )

    assert coherence.loc["reference", "in_phase"] > coherence.loc[
        "reference", "noise"
    ]
    assert imaginary.loc["reference", "quadrature"] > imaginary.loc[
        "reference", "in_phase"
    ]
    assert np.allclose(np.diag(coherence), 1.0)
    assert np.allclose(np.diag(imaginary), 0.0)


def test_bidirectional_edge_summary_tracks_asymmetry():
    edges = pd.DataFrame(
        [
            {"patient_id": "P1", "source": "L ACC", "target": "R PAG", "weight": 8.0},
            {"patient_id": "P1", "source": "R PAG", "target": "L ACC", "weight": 2.0},
            {"patient_id": "P1", "source": "L ACC", "target": "R SFG", "weight": 7.0},
            {"patient_id": "P2", "source": "L ACC", "target": "R PAG", "weight": 3.0},
            {"patient_id": "P2", "source": "R PAG", "target": "L ACC", "weight": 6.0},
        ]
    )
    summary = ep.bidirectional_edge_summary(edges, group_cols=["patient_id"])
    p1 = summary[summary.patient_id == "P1"].iloc[0]
    assert p1.node_a == "L ACC"
    assert p1.node_b == "R PAG"
    assert p1.asymmetry_log2 > 0
    assert p1.dominant_direction == "L ACC->R PAG"


def test_bidirectional_edge_summary_handles_no_reciprocal_edges():
    edges = pd.DataFrame(
        [
            {"source": "ACC", "target": "Thalamus", "weight": 1.0},
            {"source": "ACC", "target": "Striatum", "weight": 2.0},
        ]
    )

    summary = ep.bidirectional_edge_summary(edges)

    assert summary.empty
    assert "asymmetry_log2" in summary.columns


def test_opportunity_conditioned_reciprocity_ignores_untested_reverse_edges():
    opportunities = pd.DataFrame(
        [
            {"patient_id": "P1", "source": "ACC", "target": "Thalamus", "responding": True},
            {"patient_id": "P1", "source": "Thalamus", "target": "ACC", "responding": True},
            {"patient_id": "P1", "source": "ACC", "target": "Striatum", "responding": True},
            {"patient_id": "P1", "source": "Striatum", "target": "ACC", "responding": False},
            {"patient_id": "P1", "source": "PAG", "target": "ACC", "responding": True},
            {"patient_id": "P2", "source": "ACC", "target": "Thalamus", "responding": False},
            {"patient_id": "P2", "source": "Thalamus", "target": "ACC", "responding": False},
        ]
    )

    summary = ep.opportunity_conditioned_reciprocity(
        opportunities,
        group_cols=["patient_id"],
    ).set_index("patient_id")

    assert summary.loc["P1", "bidirectionally_tested_pairs"] == 2
    assert summary.loc["P1", "pairs_with_any_response"] == 2
    assert summary.loc["P1", "bidirectional_pairs"] == 1
    assert summary.loc["P1", "unidirectional_pairs"] == 1
    assert summary.loc["P1", "reciprocity"] == pytest.approx(0.5)
    assert summary.loc["P2", "bidirectionally_tested_pairs"] == 1
    assert summary.loc["P2", "pairs_with_any_response"] == 0
    assert np.isnan(summary.loc["P2", "reciprocity"])


def test_weighted_directed_node_metrics_use_strength_for_local_reaching():
    edges = pd.DataFrame(
        [
            {"patient_id": "P1", "source": "ACC", "target": "Thalamus", "weight": 10.0},
            {"patient_id": "P1", "source": "Striatum", "target": "Thalamus", "weight": 1.0},
            {"patient_id": "P1", "source": "Thalamus", "target": "ACC", "weight": 4.0},
        ]
    )

    metrics = ep.weighted_directed_node_metrics(
        edges,
        group_cols=["patient_id"],
    ).set_index("node")

    assert metrics.loc["ACC", "out_strength"] == pytest.approx(10.0)
    assert metrics.loc["ACC", "sender_receiver_index"] == pytest.approx(6 / 14)
    assert (
        metrics.loc["ACC", "local_reaching_centrality"]
        > metrics.loc["Striatum", "local_reaching_centrality"]
    )
    assert metrics["out_strength_share"].sum() == pytest.approx(1.0)


def test_dynamic_target_distribution_metrics_quantify_broadcast_and_change():
    distributions = pd.DataFrame(
        [
            {"site": "A", "time_s": 0.0, "target": "ACC", "weight": 1.0},
            {"site": "A", "time_s": 0.0, "target": "Thalamus", "weight": 0.0},
            {"site": "A", "time_s": 1.0, "target": "ACC", "weight": 1.0},
            {"site": "A", "time_s": 1.0, "target": "Thalamus", "weight": 1.0},
        ]
    )

    timecourse, summary = ep.dynamic_target_distribution_metrics(
        distributions,
        group_cols=["site"],
        scaffold_nodes=["ACC", "Thalamus"],
    )

    assert timecourse["broadcast_participation"].tolist() == pytest.approx(
        [0.0, 1.0]
    )
    assert summary.loc[0, "broadcast_participation"] == pytest.approx(2 / 3)
    assert summary.loc[0, "broadcast_entropy"] == pytest.approx(2 / 3)
    assert summary.loc[0, "broadcast_effective_targets"] == pytest.approx(5 / 3)
    assert summary.loc[0, "scaffold_weight_fraction"] == pytest.approx(1.0)
    assert 0 < summary.loc[0, "reconfiguration_mean_jsd"] < 1


def test_dynamic_target_distribution_metrics_marks_empty_bins_undefined():
    distributions = pd.DataFrame(
        [
            {"site": "A", "time_s": 0.0, "target": "ACC", "weight": 0.0},
            {"site": "A", "time_s": 0.0, "target": "Thalamus", "weight": 0.0},
            {"site": "A", "time_s": 1.0, "target": "ACC", "weight": 1.0},
            {"site": "A", "time_s": 1.0, "target": "Thalamus", "weight": 1.0},
        ]
    )

    timecourse, summary = ep.dynamic_target_distribution_metrics(
        distributions,
        group_cols=["site"],
        scaffold_nodes=["ACC", "Thalamus"],
    )

    empty = timecourse.loc[timecourse["time_s"].eq(0.0)].iloc[0]
    assert np.isnan(empty["broadcast_participation"])
    assert np.isnan(empty["broadcast_entropy"])
    assert np.isnan(empty["broadcast_effective_targets"])
    assert np.isnan(empty["scaffold_weight_fraction"])
    assert summary.loc[0, "participation_peak"] == pytest.approx(1.0)
    assert summary.loc[0, "participation_peak_time"] == pytest.approx(1.0)


def test_latency_wavefront_gradient_recovers_known_direction():
    detections = pd.DataFrame(
        [
            {"patient_id": "P1", "session_id": "A", "stim_pair": "LACC_1_2", "channel": f"C{i}", "consensus_ch": True, "n1_latency_ms": 20.0 + 0.5 * x, "peak_amplitude_uv": 100 + i}
            for i, x in enumerate([0.0, 10.0, 20.0, 30.0, 40.0], start=1)
        ]
    )
    elec_meta = pd.DataFrame(
        [
            {"patient_id": "P1", "session_id": "A", "elec_label": f"C{i}", "anat_label": "L ACC" if i <= 2 else "L Caudate", "mni_x": x, "mni_y": 0.0, "mni_z": 0.0}
            for i, x in enumerate([0.0, 10.0, 20.0, 30.0, 40.0], start=1)
        ]
    )
    wavefront = ep.latency_wavefront_table(detections, elec_meta)
    assert len(wavefront) == 5
    assert wavefront["in_scaffold"].all()

    gradients = ep.estimate_latency_gradient(wavefront, group_cols=["patient_id", "session_id", "stim_pair"], min_points=4)
    row = gradients.iloc[0]
    assert row["status"] == "ok"
    assert row["gradient_x_ms_per_mm"] == pytest.approx(0.5, abs=1e-6)
    assert row["apparent_speed_m_per_s"] == pytest.approx(2.0, abs=1e-6)
    assert row["propagation_unit_x"] == pytest.approx(1.0, abs=1e-6)


def test_latency_wavefront_collapses_detector_methods_per_acquisition_contact():
    detections = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "session_id": "A",
                "stim_pair": "LACC_1_2",
                "acquisition_id": acquisition,
                "channel": channel,
                "method": method,
                "consensus_ch": True,
                "n1_latency_ms": latency + method_offset,
                "peak_amplitude_uv": 100.0 + method_offset,
            }
            for acquisition in ("run001", "run002")
            for channel, latency in (("C1", 20.0), ("C2", 25.0))
            for method, method_offset in (
                ("keller_zscore", 0.0),
                ("peak_amplitude", 1.0),
                ("rms_response", 2.0),
            )
        ]
    )
    elec_meta = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "session_id": "A",
                "elec_label": channel,
                "anat_label": "L ACC",
                "mni_x": x,
                "mni_y": 0.0,
                "mni_z": 0.0,
            }
            for channel, x in (("C1", 0.0), ("C2", 10.0))
        ]
    )

    wavefront = ep.latency_wavefront_table(detections, elec_meta)

    assert len(wavefront) == 4
    assert wavefront["n_detector_rows"].eq(3).all()
    assert wavefront["n1_latency_ms"].tolist() == [21.0, 26.0, 21.0, 26.0]
    assert wavefront["detector_methods"].str.count(";").eq(2).all()


def test_latency_wavefront_excludes_both_stimulation_contacts():
    detections = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "session_id": "A",
                "stim_pair": "LACC_1_2",
                "acquisition_id": "run001",
                "channel": channel,
                "consensus_ch": True,
                "n1_latency_ms": latency,
                "peak_amplitude_uv": 50.0,
            }
            for channel, latency in (
                ("LACC1-ref", 10.0),
                ("LACC2-ref", 12.0),
                ("C1", 25.0),
            )
        ]
    )
    elec_meta = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "session_id": "A",
                "elec_label": channel,
                "anat_label": "L ACC",
                "mni_x": x,
                "mni_y": 0.0,
                "mni_z": 0.0,
            }
            for channel, x in (("LACC01", 0.0), ("LACC02", 5.0), ("C1", 10.0))
        ]
    )

    wavefront = ep.latency_wavefront_table(detections, elec_meta)

    assert wavefront["record_elec"].tolist() == ["C1"]


def test_canonicalize_connectivity_edges_only_reorients_undirected_metrics():
    edges = pd.DataFrame(
        [
            {
                "method": "plv",
                "directed": False,
                "source": "Z2",
                "target": "A1",
                "source_region": "PAG",
                "target_region": "ACC",
                "source_mni_x": 8.0,
                "target_mni_x": -4.0,
            },
            {
                "method": "granger",
                "directed": True,
                "source": "Z2",
                "target": "A1",
                "source_region": "PAG",
                "target_region": "ACC",
                "source_mni_x": 8.0,
                "target_mni_x": -4.0,
            },
        ]
    )

    result = ep.canonicalize_connectivity_edges(edges)

    assert result.loc[0, ["source", "target"]].tolist() == ["A1", "Z2"]
    assert result.loc[0, ["source_region", "target_region"]].tolist() == [
        "ACC",
        "PAG",
    ]
    assert result.loc[0, ["source_mni_x", "target_mni_x"]].tolist() == [
        -4.0,
        8.0,
    ]
    assert result.loc[1, ["source", "target"]].tolist() == ["Z2", "A1"]

    region_result = ep.canonicalize_connectivity_edges(
        edges,
        source_col="source_region",
        target_col="target_region",
    )
    assert region_result.loc[
        0,
        ["source_region", "target_region"],
    ].tolist() == ["ACC", "PAG"]
