"""Power analysis uses the production decision and matching t critical."""

import pytest

from codex_hydrogym.memalign_h1.harness import DECISION_FAIL, DECISION_INCONCLUSIVE, DECISION_PASS
from codex_hydrogym.memalign_h1.power import (
    exact_normal_probabilities,
    groups_for_power,
    minimum_effect_for_power,
    simulate_probabilities,
    t_critical_95,
)


def test_four_group_conditional_power_thresholds_are_preregistered():
    assert minimum_effect_for_power(probability=0.5, group_sd=0.5) == pytest.approx(0.7170775009)
    assert minimum_effect_for_power(probability=0.8, group_sd=0.5) == pytest.approx(1.0639747052)


def test_required_group_count_recomputes_matching_df_critical():
    assert groups_for_power(probability=0.8, true_improvement=0.25, group_sd=0.5) == 34
    assert t_critical_95(34) == pytest.approx(2.0345152974493383)
    assert t_critical_95(34) != pytest.approx(t_critical_95(4))


def test_simulation_reports_pass_fail_and_inconclusive_with_unequal_groups():
    result = simulate_probabilities(
        true_improvement=0.25,
        group_sd=0.5,
        group_sizes=(2, 3, 5, 8),
        distribution="scaled_t5",
        replicates=10_000,
    )
    probabilities = result["probabilities"]
    assert result["group_sizes"] == [2, 3, 5, 8]
    assert set((DECISION_PASS, DECISION_FAIL, DECISION_INCONCLUSIVE)) <= set(probabilities)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_exact_null_has_symmetric_two_sided_error():
    probabilities = exact_normal_probabilities(true_improvement=0.0, group_sd=0.5, group_count=4)
    assert probabilities[DECISION_PASS] == pytest.approx(0.025)
    assert probabilities[DECISION_FAIL] == pytest.approx(0.025)
    assert probabilities[DECISION_INCONCLUSIVE] == pytest.approx(0.95)
