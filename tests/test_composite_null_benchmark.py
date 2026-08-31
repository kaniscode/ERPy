from __future__ import annotations

import json

import pytest

from validation import benchmark_crp_energy_composite_nulls as benchmark


@pytest.mark.parametrize(
    ("arm", "expected_component"),
    [
        (benchmark.PROJECTION_ALT_ENERGY_NULL, "p_R_call"),
        (benchmark.ENERGY_ALT_PROJECTION_NULL, "p_E_call"),
    ],
)
def test_composite_null_family_exercises_declared_branch_exactly(
    arm: str,
    expected_component: str,
) -> None:
    config = benchmark.CompositeNullConfig(n_replicates=1)
    first = benchmark.run_family(config, arm, 0)
    second = benchmark.run_family(config, arm, 0)

    assert first == second
    assert first[expected_component]
    assert first["p_joint"] == pytest.approx(max(first["p_R"], first["p_E"]))
    assert first["testing_family_size"] == 4
    assert first["projection_test_exact"]
    assert first["projection_randomizations"] == 2 ** 11 - 1
    assert first["energy_test_exact"]
    assert first["energy_randomizations"] == 2 ** 12 - 1


def test_composite_null_quick_outputs_are_complete(tmp_path) -> None:
    config = benchmark.CompositeNullConfig(n_replicates=2)
    result = benchmark.execute(config, output_dir=tmp_path, jobs=1)

    assert result["n_families"] == 4
    assert {row["arm"] for row in result["summary"]} == set(benchmark.ARMS)
    required = {
        "README.md",
        "SHA256SUMS.txt",
        "composite_null_config.json",
        "composite_null_results.csv.gz",
        "composite_null_summary.csv",
        "composite_null_summary.json",
        "provenance.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == required
    config_record = json.loads((tmp_path / "composite_null_config.json").read_text())
    assert config_record["exact_projection_assignments"] == 2 ** 11
    assert config_record["exact_energy_assignments"] == 2 ** 12
