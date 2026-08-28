"""Strict reward-candidate and evaluation-data contracts."""

from dataclasses import replace
import json

import pytest

from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.genai.contracts import (
    REWARD_CANDIDATE_SCHEMA_VERSION,
    RewardCandidate,
    RolloutEvidence,
    build_candidate_evaluation_record,
    build_gepa_record,
    parse_reward_candidate,
)


_CONTEXT_FINGERPRINT = "a" * 64
_FROZEN_TRAINING_FINGERPRINT = "b" * 64
_HELDOUT_EVIDENCE_DIGEST = "c" * 64


def _candidate_dict():
    return {
        "schema_version": REWARD_CANDIDATE_SCHEMA_VERSION,
        "candidate_id": "codex_hydrogym_candidate_001",
        "reward_alpha": 1.5,
        "learning_rate": 0.0002,
        "entropy_coefficient": 0.01,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "num_updates": 7,
        "hypothesis": "Increase the TKE penalty while retaining moderate exploration.",
        "rationale": "The baseline is stable but suppresses turbulence slowly.",
    }


def _evidence(run_id: str, *, tke: float, control: float, passed: bool = True):
    return RolloutEvidence(
        run_id=run_id,
        context_fingerprint=_CONTEXT_FINGERPRINT,
        frozen_training_fingerprint=_FROZEN_TRAINING_FINGERPRINT,
        heldout_evidence_digest=_HELDOUT_EVIDENCE_DIGEST,
        mean_tke=tke,
        control_l1=control,
        reward_total=-tke - control,
        physics_gates_passed=passed,
        artifact_uri=f"runs:/{run_id}/codex_hydrogym/evidence",
    )


def test_candidate_json_is_strict_bounded_and_executable():
    candidate = parse_reward_candidate(json.dumps(_candidate_dict()))
    baseline = replace(
        KolmogorovPPOConfig(),
        num_envs=2,
        num_steps=4,
        num_minibatches=2,
        total_timesteps=16,
    )
    configured = candidate.apply(baseline)

    assert configured.run_name == candidate.candidate_id
    assert configured.reward_alpha == 1.5
    assert configured.total_timesteps == 7 * baseline.total_batch_size
    assert baseline.total_timesteps == 16


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reward_alpha", 0.0),
        ("learning_rate", float("nan")),
        ("entropy_coefficient", 0.2),
        ("num_updates", True),
        ("candidate_id", "unlabeled-candidate"),
    ],
)
def test_candidate_rejects_unbounded_or_unlabeled_values(field, value):
    payload = _candidate_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        parse_reward_candidate(payload)


def test_candidate_rejects_executable_or_unknown_fields():
    payload = _candidate_dict()
    payload["reward_code"] = "import os"

    with pytest.raises(ValueError, match="unknown=.*reward_code"):
        parse_reward_candidate(payload)


def test_evaluation_record_requires_comparable_real_rollouts():
    candidate = RewardCandidate(**_candidate_dict())
    baseline = _evidence("baseline-run", tke=2.0, control=0.4)
    candidate_rollout = _evidence("candidate-run", tke=1.6, control=0.5)

    record = build_candidate_evaluation_record(
        candidate=candidate,
        baseline=baseline,
        candidate_rollout=candidate_rollout,
    )

    assert record["inputs"]["project"] == "codex_hydrogym"
    assert record["outputs"]["delta_mean_tke"] == pytest.approx(-0.4)
    assert "expected_response" in record["expectations"]

    mismatched = replace(candidate_rollout, context_fingerprint="d" * 64)
    with pytest.raises(ValueError, match="held-out context"):
        build_candidate_evaluation_record(candidate=candidate, baseline=baseline, candidate_rollout=mismatched)

    mismatched_training = replace(candidate_rollout, frozen_training_fingerprint="d" * 64)
    with pytest.raises(ValueError, match="frozen-training"):
        build_candidate_evaluation_record(
            candidate=candidate,
            baseline=baseline,
            candidate_rollout=mismatched_training,
        )


def test_rollout_evidence_requires_digest_bound_heldout_lineage():
    with pytest.raises(ValueError, match="context_fingerprint"):
        replace(_evidence("run", tke=1.0, control=0.1), context_fingerprint="not-a-digest")

    with pytest.raises(ValueError, match="heldout_evidence_digest"):
        replace(_evidence("run", tke=1.0, control=0.1), heldout_evidence_digest="not-a-digest")

    with pytest.raises(ValueError, match="frozen_training_fingerprint"):
        replace(_evidence("run", tke=1.0, control=0.1), frozen_training_fingerprint="not-a-digest")


def test_gepa_record_always_contains_inputs_and_expectations():
    record = build_gepa_record(
        scenario={"reynolds_number": 200, "seed": 9},
        expected_behavior="Return one bounded JSON reward candidate and preserve every physics constraint.",
    )

    assert set(record) == {"inputs", "expectations"}
    assert record["inputs"]["scenario"]["seed"] == 9
