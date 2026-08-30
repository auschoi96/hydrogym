"""Behavioral tests for the frozen policy-quality metric."""

import pytest

from codex_hydrogym.training.frozen_control_metric import CandidateOutcomes, SeedOutcome, compare_candidates


def _candidate(name, candidate_tke, reward):
    return CandidateOutcomes(
        candidate=name,
        outcomes=tuple(
            SeedOutcome(
                seed=seed,
                uncontrolled_mean_tke=2.0 + 0.1 * seed,
                candidate_mean_tke=candidate_tke + 0.05 * seed,
                mean_control_l1=0.5,
                mean_candidate_reward=reward + seed,
            )
            for seed in range(4)
        ),
    )


def test_metric_and_detector_are_invariant_to_positive_affine_reward_rescaling():
    original_candidates = (
        _candidate("better_metric", 1.0, 10.0),
        _candidate("higher_reward_worse_metric", 1.5, 20.0),
    )
    transformed_candidates = tuple(
        CandidateOutcomes(
            candidate=candidate.candidate,
            outcomes=tuple(
                SeedOutcome(
                    seed=outcome.seed,
                    uncontrolled_mean_tke=outcome.uncontrolled_mean_tke,
                    candidate_mean_tke=outcome.candidate_mean_tke,
                    mean_control_l1=outcome.mean_control_l1,
                    mean_candidate_reward=7.0 * outcome.mean_candidate_reward + 31.0,
                )
                for outcome in candidate.outcomes
            ),
        )
        for candidate in original_candidates
    )

    original = compare_candidates(original_candidates, matched_mean_control_l1=0.5)
    transformed = compare_candidates(transformed_candidates, matched_mean_control_l1=0.5)

    assert [item["per_seed"] for item in original["candidates"]] != [
        item["per_seed"] for item in transformed["candidates"]
    ]
    assert [item["clustered_interval_95"] for item in original["candidates"]] == [
        item["clustered_interval_95"] for item in transformed["candidates"]
    ]
    assert transformed["reward_metric_correlation"] == pytest.approx(original["reward_metric_correlation"])
    assert transformed["reward_hacking_candidates"] == original["reward_hacking_candidates"]


@pytest.mark.parametrize("seed_count", [1, 2])
def test_candidate_refuses_fewer_than_three_seeds(seed_count):
    outcomes = tuple(
        SeedOutcome(
            seed=seed, uncontrolled_mean_tke=2.0, candidate_mean_tke=1.0, mean_control_l1=0.5, mean_candidate_reward=1.0
        )
        for seed in range(seed_count)
    )

    with pytest.raises(ValueError, match="refuses fewer than 3"):
        CandidateOutcomes(candidate="single_seed_is_not_evidence", outcomes=outcomes)


def test_reward_hacking_candidate_is_flagged_when_reward_rises_as_metric_falls():
    report = compare_candidates(
        (
            _candidate("good_control", 0.8, 10.0),
            _candidate("middling", 1.2, 20.0),
            _candidate("reward_hacker", 1.6, 30.0),
        ),
        matched_mean_control_l1=0.5,
    )

    assert report["reward_metric_correlation"] == pytest.approx(-1.0, abs=0.02)
    assert report["reward_metric_diverged"] is True
    assert "reward_hacker" in report["reward_hacking_candidates"]
    assert (
        report["candidates"][0]["clustered_interval_95"]["mean"]
        > report["candidates"][2]["clustered_interval_95"]["mean"]
    )
    assert len(report["candidates"][0]["per_seed"]) == 4


def test_unmatched_actuation_cost_is_rejected_instead_of_rewarded():
    expensive = _candidate("expensive", 0.5, 100.0)
    expensive = CandidateOutcomes(
        candidate=expensive.candidate,
        outcomes=tuple(
            SeedOutcome(
                seed=outcome.seed,
                uncontrolled_mean_tke=outcome.uncontrolled_mean_tke,
                candidate_mean_tke=outcome.candidate_mean_tke,
                mean_control_l1=0.8,
                mean_candidate_reward=outcome.mean_candidate_reward,
            )
            for outcome in expensive.outcomes
        ),
    )

    with pytest.raises(ValueError, match="unmatched mean_control_l1"):
        compare_candidates((expensive, _candidate("matched", 1.0, 1.0)), matched_mean_control_l1=0.5)
