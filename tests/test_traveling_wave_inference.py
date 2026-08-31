import numpy as np
import pandas as pd

from ERPy.traveling_waves import (
    latency_distance_permutation_test,
    latency_gradient_permutation_test,
)


def test_latency_gradients_detects_ordered_spatial_latency():
    rng = np.random.default_rng(4)
    coords = rng.normal(size=(18, 3)) * 20.0
    latency = 40.0 + coords @ np.array([0.4, -0.2, 0.1])
    frame = pd.DataFrame(
        {
            "acquisition": "A",
            "latency_ms": latency,
            "mni_x": coords[:, 0],
            "mni_y": coords[:, 1],
            "mni_z": coords[:, 2],
            "metric_abs": np.linspace(1.0, 2.0, len(coords)),
        }
    )

    result = latency_gradient_permutation_test(
        frame,
        group_cols=["acquisition"],
        n_permutations=199,
        random_state=8,
    )

    assert len(result) == 1
    assert result.loc[0, "latency_r2"] > 0.99
    assert result.loc[0, "gradient_p_value"] <= 0.01
    assert bool(result.loc[0, "gradient_fdr_significant"])


def test_latency_gradients_preserves_too_few_contact_groups():
    frame = pd.DataFrame(
        {
            "acquisition": ["A"] * 3,
            "latency_ms": [10.0, 20.0, 30.0],
            "mni_x": [0.0, 1.0, 2.0],
            "mni_y": [0.0, 1.0, 2.0],
            "mni_z": [0.0, 1.0, 2.0],
        }
    )

    result = latency_gradient_permutation_test(
        frame,
        group_cols=["acquisition"],
        n_permutations=20,
    )

    assert result.loc[0, "status"] == "too_few_contacts"
    assert np.isnan(result.loc[0, "gradient_p_value"])


def test_unweighted_permutation_path_detects_spatial_order():
    rng = np.random.default_rng(14)
    coords = rng.normal(size=(20, 3)) * 15.0
    latency = 35.0 + coords @ np.array([0.3, 0.15, -0.05])
    frame = pd.DataFrame(
        {
            "latency_ms": latency,
            "mni_x": coords[:, 0],
            "mni_y": coords[:, 1],
            "mni_z": coords[:, 2],
        }
    )

    result = latency_gradient_permutation_test(
        frame,
        weight_col=None,
        n_permutations=199,
        random_state=22,
    )

    assert result.loc[0, "latency_r2"] > 0.99
    assert result.loc[0, "gradient_p_value"] <= 0.01


def test_latency_distance_detects_outward_radial_order():
    distance = np.linspace(5.0, 95.0, 24)
    latency = 18.0 + 0.5 * distance
    frame = pd.DataFrame(
        {
            "acquisition": "A",
            "latency_ms": latency,
            "distance_from_stim_mm": distance,
        }
    )

    result = latency_distance_permutation_test(
        frame,
        group_cols=["acquisition"],
        weight_col=None,
        n_permutations=199,
        random_state=17,
    )

    assert len(result) == 1
    assert np.isclose(result.loc[0, "distance_slope_ms_per_mm"], 0.5)
    assert np.isclose(result.loc[0, "apparent_radial_speed_m_per_s"], 2.0)
    assert bool(result.loc[0, "outward_latency_gradient"])
    assert result.loc[0, "distance_r2"] > 0.99
    assert result.loc[0, "distance_p_value"] <= 0.01
    assert bool(result.loc[0, "distance_fdr_significant"])


def test_latency_distance_retains_inward_and_too_few_statuses():
    inward = pd.DataFrame(
        {
            "latency_ms": [50.0, 40.0, 30.0, 20.0, 10.0],
            "distance_from_stim_mm": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    inward_result = latency_distance_permutation_test(
        inward,
        weight_col=None,
        n_permutations=49,
        random_state=2,
    )
    assert inward_result.loc[0, "distance_slope_ms_per_mm"] < 0
    assert not bool(inward_result.loc[0, "outward_latency_gradient"])

    too_few_result = latency_distance_permutation_test(
        inward.head(3),
        weight_col=None,
        n_permutations=20,
    )
    assert too_few_result.loc[0, "status"] == "too_few_contacts"
    assert np.isnan(too_few_result.loc[0, "distance_p_value"])
