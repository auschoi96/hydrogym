"""Calibration and contract tests for the paired-delta instrument."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from statistics import fmean

import pytest

from codex_hydrogym.agent_eval.instrument_validation import (
    DECISION_EFFECT,
    DECISION_NO_EFFECT,
    GROUP_CLUSTERS,
    MIN_MEAN_NOISY_SPEARMAN,
    MONOTONICITY_TOLERANCE,
    NOISE_SIGMA,
    DeterministicTierJudge,
    RewardCandidateGenerator,
    SeededNoisyTierJudge,
    harness,
    run_calibration,
    run_ceiling_pinned_reference,
    run_validation,
)
from codex_hydrogym.gate0.ensemble_diagnostic import _mean_ci


@pytest.fixture(scope="module")
def calibration():
    return run_calibration()


def _null_arm_reports_no_effect():
    return run_validation(SeededNoisyTierJudge(seed=11, sigma=NOISE_SIGMA), seed=11)


def test_null_arm_interval_contains_zero_and_reports_no_effect(calibration):
    # The claim is repeated-sampling FPR, not one degenerate null interval.
    assert calibration.null_false_positive_rate <= 0.05
    assert calibration.null_false_positive_interval_95["lower"] <= 0.05
    assert calibration.null_false_positive_interval_95["lower"] < calibration.null_false_positive_interval_95["upper"]
    assert calibration.replicates >= 200


def test_null_rate_is_below_five_percent_over_2000_replicates():
    # A larger block prevents the gate from resting on one lucky 500-seed draw.
    calibration = run_calibration(replicates=2000)
    assert calibration.null_false_positive_rate <= 0.05


def test_null_arms_differ_only_by_random_seed_while_staying_in_tier():
    arm_a = RewardCandidateGenerator(tier=2, groups=GROUP_CLUSTERS, seed=20260826).generate()
    arm_b = RewardCandidateGenerator(tier=2, groups=GROUP_CLUSTERS, seed=20260827).generate()
    assert {candidate.group_id for candidate in arm_a} == {candidate.group_id for candidate in arm_b}
    assert {candidate.text for candidate in arm_a} != {candidate.text for candidate in arm_b}
    assert all(candidate.tier == 2 for candidate in (*arm_a, *arm_b))
    # The stored candidate seed genuinely determines rendering.
    for candidate in arm_a:
        assert candidate.text == RewardCandidateGenerator(tier=2, groups=2, seed=0)._render(candidate.seed)


def test_positive_arms_exclude_zero_and_recover_known_ordering(calibration):
    # Positive evidence is stochastic: nominal interval coverage and power are
    # measured over 500 replicates, never inferred from a constant delta.
    assert calibration.positive_coverage_interval_95["lower"] <= 0.95
    assert calibration.positive_coverage_interval_95["upper"] >= 0.95
    assert calibration.positive_detection_rate > 0.0
    assert calibration.mean_ordering_spearman >= MIN_MEAN_NOISY_SPEARMAN
    for small in calibration.tier_pairs:
        for large in calibration.tier_pairs:
            if large.true_delta > small.true_delta:
                assert large.detection_rate + MONOTONICITY_TOLERANCE >= small.detection_rate


def test_ceiling_pinned_judge_reports_no_effect_against_a_real_gap():
    reference = run_ceiling_pinned_reference()
    assert reference.control_tier == 0 and reference.candidate_tier == 4
    assert set(reference.paired_deltas.values()) == {0.0}
    assert reference.decision == DECISION_NO_EFFECT
    # This degenerate reference identifies a dead judge; it is not null-arm
    # calibration, which is measured under nonzero stochastic variance above.
    assert reference.clustered_interval_95["lower"] == reference.clustered_interval_95["upper"]


def test_paired_delta_statistic_reuses_the_existing_clustered_interval():
    result = run_validation(SeededNoisyTierJudge(seed=23, sigma=NOISE_SIGMA), seed=23)
    arm = result.positive_arms[0]
    expected = _mean_ci(tuple(arm.paired_deltas.values()), arm.t_critical_95)
    assert dict(arm.clustered_interval_95) == expected
    assert arm.paired_mean_delta == fmean(arm.paired_deltas.values())
    assert arm.t_degrees_of_freedom == 9
    assert math.isclose(arm.t_critical_95, 2.262157162798205)


def _paired_subtractions(module: ast.Module) -> list[ast.BinOp]:
    matches = [
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "deltas"
            for target in node.targets
        )
        and isinstance(node.value, ast.BinOp)
    ]
    assert matches
    return matches


def test_paired_delta_expression_is_pinned_to_notebook():
    harness_tree = ast.parse(inspect.getsource(harness._paired_arm))
    notebook_path = Path(harness.__file__).parents[2] / "notebooks" / "coding_agent_memalign_proof.py"
    notebook_tree = ast.parse(notebook_path.read_text())
    # Exact AST equality means treatment labels, operand order, and subtraction
    # operator drift in either copy fails CI without importing the notebook.
    harness_expressions = _paired_subtractions(harness_tree)
    notebook_expressions = _paired_subtractions(notebook_tree)
    assert len(harness_expressions) == 1
    assert all(ast.dump(expression) == ast.dump(harness_expressions[0]) for expression in notebook_expressions)


def test_validation_is_deterministic_for_a_fixed_seed(calibration):
    assert calibration == run_calibration()
    with pytest.raises(ValueError, match="exactly 10"):
        run_validation(groups=9)
    with pytest.raises(ValueError, match="exactly 10"):
        run_validation(groups=12)
    with pytest.raises(TypeError):
        run_validation(seed="not-an-int")


def test_every_stochastic_interval_has_positive_width():
    # run_calibration asserts this for every null and positive arm in every
    # replicate. Keep a direct check to make the invariant visible in tests.
    result = _null_arm_reports_no_effect()
    for arm in (result.null_arm, *result.positive_arms):
        assert arm.clustered_interval_95["lower"] < arm.clustered_interval_95["upper"]
        assert len(set(arm.paired_deltas.values())) > 1


def test_noisy_ordering_is_not_circular(calibration):
    # No assertion compares recovered values with the judge's score dictionary.
    # Only noisy rank recovery against the independently constructed tier order
    # is accepted, at the preregistered aggregate tolerance.
    assert calibration.mean_ordering_spearman >= MIN_MEAN_NOISY_SPEARMAN
