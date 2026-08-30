"""Tests for the instrument-validation harness.

These tests assert both directions the instrument must satisfy:

- null arm (two groups from the SAME constructed tier, differing only by
  random seed): the clustered 95% interval contains zero and the decision is
  ``no_effect``;
- positive arms (groups from deliberately different tiers): every clustered
  95% interval excludes zero, the decision is ``effect``, and the recovered
  tier ordering has perfect rank correlation with the known ladder.

Nothing here is a treatment result. The deterministic stub judge is monotone
in the known quality tier by construction; see ``judges.py`` for the stated
assumption.
"""

from __future__ import annotations

import math
from statistics import fmean

import pytest

from codex_hydrogym.agent_eval.instrument_validation import (
    DECISION_EFFECT,
    DECISION_NO_EFFECT,
    QUALITY_TIERS,
    DeterministicTierJudge,
    run_ceiling_pinned_reference,
    run_validation,
)
from codex_hydrogym.agent_eval.instrument_validation.generator import (
    TIER_BASE_SCORE,
    RewardCandidateGenerator,
)
from codex_hydrogym.agent_eval.instrument_validation.harness import T_CRITICAL_95_DF9
from codex_hydrogym.gate0.ensemble_diagnostic import _mean_ci


def _null_arm_reports_no_effect():
    result = run_validation()
    interval = result.null_arm.clustered_interval_95
    assert interval["lower"] <= 0.0 <= interval["upper"]
    assert result.null_arm.decision == DECISION_NO_EFFECT
    assert result.null_arm.paired_mean_delta == 0.0
    return result


def test_null_arm_interval_contains_zero_and_reports_no_effect():
    result = _null_arm_reports_no_effect()
    # Both null groups come from the same tier, so a correct instrument must
    # find no significant difference: every per-group paired delta is zero.
    assert set(result.null_arm.paired_deltas.values()) == {0.0}
    assert result.null_arm.control_tier == result.null_arm.candidate_tier


def test_null_arms_differ_only_by_random_seed_while_staying_in_tier():
    tier = 2
    arm_a = RewardCandidateGenerator(tier=tier, groups=10, seed=20260826).generate()
    arm_b = RewardCandidateGenerator(tier=tier, groups=10, seed=20260827).generate()
    assert {candidate.group_id for candidate in arm_a} == {candidate.group_id for candidate in arm_b}
    # The seeds differ, so the generated texts differ...
    assert {candidate.text for candidate in arm_a} != {candidate.text for candidate in arm_b}
    # ...but every candidate still resolves to the same constructed tier score.
    judge = DeterministicTierJudge()
    assert all(judge.score(candidate) == TIER_BASE_SCORE[tier] for candidate in (*arm_a, *arm_b))


def test_positive_arms_exclude_zero_and_recover_known_ordering():
    result = run_validation()
    for arm in result.positive_arms:
        assert arm.control_tier < arm.candidate_tier
        assert arm.clustered_interval_95["lower"] > 0.0, arm.arm_name
        assert arm.decision == DECISION_EFFECT, arm.arm_name
        assert arm.paired_mean_delta == arm.clustered_interval_95["mean"]
    # A deliberate tier gap must be detected wherever it sits on the ladder.
    assert {arm.candidate_tier for arm in result.positive_arms} == set(range(1, 5))
    assert {arm.control_tier for arm in result.positive_arms} == set(range(0, 4))
    # Ordering is recovered: perfect rank correlation with the known ladder.
    assert math.isclose(result.ordering.spearman, 1.0)
    scores = result.ordering.tier_scores
    assert all(scores[tier] == TIER_BASE_SCORE[tier] for tier in QUALITY_TIERS)
    for tier in QUALITY_TIERS[:-1]:
        assert scores[tier] < scores[tier + 1]


def test_ceiling_pinned_judge_reports_no_effect_against_a_real_gap():
    reference = run_ceiling_pinned_reference()
    assert reference.control_tier == 0 and reference.candidate_tier == 4
    assert set(reference.paired_deltas.values()) == {0.0}
    assert reference.clustered_interval_95["lower"] == 0.0
    assert reference.clustered_interval_95["upper"] == 0.0
    assert reference.decision == DECISION_NO_EFFECT


def test_paired_delta_statistic_reuses_the_existing_clustered_interval():
    result = run_validation()
    # The harness must use the existing per-cluster mean-interval statistic.
    arm = result.positive_arms[0]
    expected = _mean_ci(tuple(arm.paired_deltas.values()), T_CRITICAL_95_DF9)
    assert dict(arm.clustered_interval_95) == expected
    assert arm.paired_mean_delta == fmean(arm.paired_deltas.values())
    # The paired-delta expression is the frozen protocol form: candidate arm
    # minus control arm per group, never the reverse.
    judge = DeterministicTierJudge()
    control = RewardCandidateGenerator(tier=arm.control_tier, groups=arm.group_clusters, seed=20260836).generate()
    candidate = RewardCandidateGenerator(tier=arm.candidate_tier, groups=arm.group_clusters, seed=20260837).generate()
    control_scores = {item.group_id: judge.score(item) for item in control}
    candidate_scores = {item.group_id: judge.score(item) for item in candidate}
    expected_deltas = {group_id: candidate_scores[group_id] - control_scores[group_id] for group_id in control_scores}
    assert arm.paired_deltas == expected_deltas


def test_validation_is_deterministic_for_a_fixed_seed():
    first = run_validation()
    second = run_validation()
    assert first == second
    with pytest.raises(ValueError):
        run_validation(groups=1)
    with pytest.raises(TypeError):
        run_validation(seed="not-an-int")
