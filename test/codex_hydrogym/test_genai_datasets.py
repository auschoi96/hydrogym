"""Scenario-matrix data quality and leakage tests."""

from codex_hydrogym.genai.datasets import build_scenario_matrix, gepa_scenario_records


def test_scenario_matrix_is_labeled_balanced_and_group_split_by_seed():
    scenarios = build_scenario_matrix()

    assert len(scenarios) == 24
    assert len({scenario.scenario_id for scenario in scenarios}) == 24
    assert all(scenario.scenario_id.startswith("codex_hydrogym_") for scenario in scenarios)
    assert {scenario.reynolds_number for scenario in scenarios} == {100, 200, 400}

    splits_by_seed = {}
    for scenario in scenarios:
        splits_by_seed.setdefault(scenario.seed, set()).add(scenario.split)
    assert all(len(splits) == 1 for splits in splits_by_seed.values())
    assert splits_by_seed[41] == {"validation"}


def test_gepa_records_exclude_validation_and_have_expectations():
    records = gepa_scenario_records()

    assert len(records) == 18
    assert all(set(record) == {"inputs", "expectations"} for record in records)
    assert all(record["inputs"]["scenario"]["split"] == "train" for record in records)
    assert all("expected_response" in record["expectations"] for record in records)
