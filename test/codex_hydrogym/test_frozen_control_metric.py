"""Behavioral tests for the frozen policy-quality metric."""

import json

import pytest

from codex_hydrogym.training.frozen_control_metric import (
    PREREGISTERED_ARTIFACT_PATH,
    PREREGISTRATION_STATUS,
    CandidateOutcomes,
    SeedOutcome,
    compare_candidates,
    load_preregistration,
    seed_outcome_from_intervals,
    write_preregistration,
)

WINDOW_COUNT = 2
WINDOW_INTERVALS = 3


def _outcome(seed, *, candidate_tke, control_l1=0.5, reward, uncontrolled_tke=None):
    return SeedOutcome(
        seed=seed,
        uncontrolled_mean_tke=2.0 + 0.1 * seed if uncontrolled_tke is None else uncontrolled_tke,
        candidate_mean_tke=candidate_tke + 0.05 * seed,
        mean_control_l1=control_l1,
        mean_candidate_reward=reward + seed,
        window_count=WINDOW_COUNT,
        window_intervals=WINDOW_INTERVALS,
    )


def _candidate(name, candidate_tke, reward, *, control_l1=0.5):
    return CandidateOutcomes(
        candidate=name,
        outcomes=tuple(
            _outcome(seed, candidate_tke=candidate_tke, control_l1=control_l1, reward=reward) for seed in range(4)
        ),
    )


@pytest.mark.parametrize(
    ("scale", "shift"),
    [
        (7.0, 31.0),
        (0.25, -100.0),
        (1000.0, 0.0),
        (1.0e-3, 250000.0),
    ],
)
def test_metric_and_detector_are_invariant_to_positive_affine_reward_rescaling(scale, shift):
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
                    mean_candidate_reward=scale * outcome.mean_candidate_reward + shift,
                    window_count=outcome.window_count,
                    window_intervals=outcome.window_intervals,
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
    assert transformed["cost_admitted_candidates"] == original["cost_admitted_candidates"]
    assert [item["q_s_spread"] for item in original["candidates"]] == [
        item["q_s_spread"] for item in transformed["candidates"]
    ]


@pytest.mark.parametrize("seed_count", [1, 2, 5])
def test_candidate_requires_exactly_four_seed_clusters(seed_count):
    outcomes = tuple(
        SeedOutcome(
            seed=seed,
            uncontrolled_mean_tke=2.0,
            candidate_mean_tke=1.0,
            mean_control_l1=0.5,
            mean_candidate_reward=1.0,
            window_count=1,
            window_intervals=4,
        )
        for seed in range(seed_count)
    )

    with pytest.raises(ValueError, match="exactly 4"):
        CandidateOutcomes(candidate="wrong_seed_count", outcomes=outcomes)


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
    assert report["cost_admitted_candidates"] == ["good_control", "middling", "reward_hacker"]
    assert report["candidates"][0]["q_s_spread"]["min"] > 0.0


def test_unmatched_actuation_cost_is_rejected_instead_of_rewarded():
    expensive = _candidate("expensive", 0.5, 100.0, control_l1=0.8)
    report = compare_candidates((expensive, _candidate("matched", 1.0, 1.0)), matched_mean_control_l1=0.5)

    by_name = {item["candidate"]: item for item in report["candidates"]}
    assert by_name["expensive"]["cost_admitted"] is False
    assert any("unmatched mean_control_l1" in reason for reason in by_name["expensive"]["cost_rejection_reasons"])
    assert by_name["expensive"]["max_relative_cost_deviation"] == pytest.approx(0.6)
    assert by_name["matched"]["cost_admitted"] is True
    assert report["cost_admitted_candidates"] == ["matched"]
    assert report["cost_rejected_candidates"]["expensive"] == by_name["expensive"]["cost_rejection_reasons"]
    # The rejected candidate is excluded from the reward-vs-metric diagnostic instead of
    # being rewarded, and a single out-of-band seed no longer aborts the whole report.
    assert report["reward_metric_correlation"] is None
    assert report["reward_hacking_candidates"] == []
    assert by_name["expensive"]["clustered_interval_95"]["mean"] > by_name["matched"]["clustered_interval_95"]["mean"]


def test_partially_out_of_band_seed_rejects_only_that_candidate():
    flaky = CandidateOutcomes(
        candidate="one_bad_seed",
        outcomes=tuple(
            _outcome(seed, candidate_tke=1.0, control_l1=0.8 if seed == 2 else 0.5, reward=50.0) for seed in range(4)
        ),
    )
    report = compare_candidates((flaky, _candidate("matched", 1.0, 1.0)), matched_mean_control_l1=0.5)

    by_name = {item["candidate"]: item for item in report["candidates"]}
    assert by_name["one_bad_seed"]["cost_admitted"] is False
    assert len(by_name["one_bad_seed"]["cost_rejection_reasons"]) == 1
    assert "seed 2" in by_name["one_bad_seed"]["cost_rejection_reasons"][0]
    assert report["cost_admitted_candidates"] == ["matched"]


def test_candidates_with_different_window_structures_are_refused():
    narrow = _candidate("narrow", 1.0, 1.0)
    wide = CandidateOutcomes(
        candidate="wide",
        outcomes=tuple(
            SeedOutcome(
                seed=outcome.seed,
                uncontrolled_mean_tke=outcome.uncontrolled_mean_tke,
                candidate_mean_tke=outcome.candidate_mean_tke,
                mean_control_l1=outcome.mean_control_l1,
                mean_candidate_reward=outcome.mean_candidate_reward,
                window_count=3,
                window_intervals=4,
            )
            for outcome in narrow.outcomes
        ),
    )
    with pytest.raises(ValueError, match="identical evaluation windows"):
        compare_candidates((narrow, wide), matched_mean_control_l1=0.5)


def test_seed_outcome_aggregates_identical_windows_for_both_arms():
    uncontrolled = [{"mean_tke": 2.0 + 0.01 * index, "control_l1": 0.0} for index in range(6)]
    candidate = [
        {
            "mean_tke": 1.0 + 0.01 * index,
            "control_l1": 0.4 + 0.1 * (index % 2),
            "reward_total": 5.0 - 0.1 * index,
        }
        for index in range(6)
    ]
    outcome = seed_outcome_from_intervals(
        seed=401,
        uncontrolled_intervals=uncontrolled,
        candidate_intervals=candidate,
        window_count=2,
        window_intervals=3,
    )

    assert outcome.uncontrolled_mean_tke == pytest.approx(sum(2.0 + 0.01 * i for i in range(6)) / 6)
    assert outcome.candidate_mean_tke == pytest.approx(sum(1.0 + 0.01 * i for i in range(6)) / 6)
    assert outcome.mean_control_l1 == pytest.approx(0.45)
    assert outcome.mean_candidate_reward == pytest.approx(sum(5.0 - 0.1 * i for i in range(6)) / 6)
    assert (outcome.window_count, outcome.window_intervals) == (2, 3)


def test_seed_outcome_producer_refuses_mismatched_window_counts():
    uncontrolled = [{"mean_tke": 2.0, "control_l1": 0.0}] * 6
    short = [{"mean_tke": 1.0, "control_l1": 0.4, "reward_total": 1.0}] * 5

    with pytest.raises(ValueError, match="identical evaluation windows"):
        seed_outcome_from_intervals(
            seed=401,
            uncontrolled_intervals=uncontrolled,
            candidate_intervals=short,
            window_count=2,
            window_intervals=3,
        )


@pytest.mark.parametrize(
    ("uncontrolled_record", "candidate_record", "match"),
    [
        ({"control_l1": 0.0}, {"mean_tke": 1.0, "control_l1": 0.4, "reward_total": 1.0}, "missing"),
        (
            {"mean_tke": 2.0, "control_l1": 0.0},
            {"mean_tke": float("nan"), "control_l1": 0.4, "reward_total": 1.0},
            "finite",
        ),
        (
            {"mean_tke": 2.0, "control_l1": 0.0},
            {"mean_tke": 1.0, "control_l1": -0.1, "reward_total": 1.0},
            "nonnegative",
        ),
    ],
)
def test_seed_outcome_producer_validates_interval_records(uncontrolled_record, candidate_record, match):
    with pytest.raises(ValueError, match=match):
        seed_outcome_from_intervals(
            seed=401,
            uncontrolled_intervals=[uncontrolled_record] * 6,
            candidate_intervals=[candidate_record] * 6,
            window_count=2,
            window_intervals=3,
        )


def test_diverged_seed_is_flagged_not_silently_dropped():
    diverged = CandidateOutcomes(
        candidate="diverged",
        outcomes=tuple(
            SeedOutcome(
                seed=seed,
                uncontrolled_mean_tke=2.0,
                candidate_mean_tke=2002.0 if seed == 3 else 1.0,
                mean_control_l1=0.5,
                mean_candidate_reward=1.0,
                window_count=WINDOW_COUNT,
                window_intervals=WINDOW_INTERVALS,
            )
            for seed in range(4)
        ),
    )
    report = compare_candidates((diverged, _candidate("healthy", 1.0, 1.0)), matched_mean_control_l1=0.5)

    entry = {item["candidate"]: item for item in report["candidates"]}["diverged"]
    assert entry["seeds_below_floor"] == [3]
    assert [item["below_floor"] for item in entry["per_seed"]] == [False, False, False, True]
    assert entry["q_s_spread"]["min"] == pytest.approx(-1000.0)
    assert entry["q_s_spread"]["max"] == pytest.approx(0.5)
    assert entry["q_s_spread"]["stdev"] > 0.0
    # The interval is still computed over all four clusters; the flag informs, it never drops.
    assert entry["clustered_interval_95"]["mean"] == pytest.approx((0.5 * 3 - 1000.0) / 4)


def test_no_target_cost_is_preregistered_yet():
    # Tripwire: recording a C* requires the real frozen-config rollout run and a
    # conscious update of this test together with a preregistration artifact.
    assert PREREGISTERED_ARTIFACT_PATH is None


def test_preregistration_round_trip_is_write_once(tmp_path):
    path = tmp_path / "preregistration.json"
    preregistration = write_preregistration(
        path,
        target_mean_control_l1=0.5,
        relative_cost_tolerance=0.05,
        window_count=2,
        window_intervals=3,
        seeds=(401, 503, 607, 709),
        derivation="placeholder: no real frozen-config rollout has been executed",
    )
    assert preregistration.status == PREREGISTRATION_STATUS

    loaded = load_preregistration(path)
    assert loaded == preregistration

    with pytest.raises(FileExistsError):
        write_preregistration(
            path,
            target_mean_control_l1=0.5,
            relative_cost_tolerance=0.05,
            window_count=2,
            window_intervals=3,
            seeds=(401, 503, 607, 709),
            derivation="second write",
        )


def test_preregistration_load_rejects_tampered_status(tmp_path):
    path = tmp_path / "preregistration.json"
    write_preregistration(
        path,
        target_mean_control_l1=0.5,
        relative_cost_tolerance=0.05,
        window_count=2,
        window_intervals=3,
        seeds=(401, 503, 607, 709),
        derivation="placeholder",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "decided_after_seeing_candidates"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="status mismatch"):
        load_preregistration(path)


def test_preregistration_binds_the_comparison_to_the_frozen_target(tmp_path):
    path = tmp_path / "preregistration.json"
    preregistration = write_preregistration(
        path,
        target_mean_control_l1=0.5,
        relative_cost_tolerance=0.05,
        window_count=WINDOW_COUNT,
        window_intervals=WINDOW_INTERVALS,
        seeds=(0, 1, 2, 3),
        derivation="placeholder",
    )
    candidates = (_candidate("alpha", 1.0, 10.0), _candidate("beta", 1.2, 20.0))

    report = compare_candidates(candidates, matched_mean_control_l1=0.5, preregistration=preregistration)
    assert report["preregistration"]["status"] == PREREGISTRATION_STATUS
    assert report["preregistration"]["target_mean_control_l1"] == 0.5

    with pytest.raises(ValueError, match="preregistered target_mean_control_l1"):
        compare_candidates(candidates, matched_mean_control_l1=0.6, preregistration=preregistration)


def test_historical_seed_reuse_is_warned_not_silent():
    outcomes = tuple(
        SeedOutcome(
            seed=seed,
            uncontrolled_mean_tke=2.0,
            candidate_mean_tke=1.0,
            mean_control_l1=0.5,
            mean_candidate_reward=1.0,
            window_count=1,
            window_intervals=4,
        )
        for seed in (7, 101, 211, 307)
    )
    candidates = (
        CandidateOutcomes(candidate="historical", outcomes=outcomes),
        CandidateOutcomes(candidate="also_historical", outcomes=outcomes),
    )

    with pytest.warns(UserWarning, match="historical Gate 0 v1/v2 seeds"):
        report = compare_candidates(candidates, matched_mean_control_l1=0.5)

    assert any("historical Gate 0 v1/v2 seeds" in message for message in report["seed_identity_warnings"])
    assert any("not the frozen ensemble-diagnostic seed set" in message for message in report["seed_identity_warnings"])
