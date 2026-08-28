"""Reward-only proposal, deterministic compilation, and execution tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from codex_hydrogym import REWARD_FORMULA_VERSION
from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.genai.contracts import (
    REWARD_SPEC_SCHEMA_VERSION,
    RewardSpec,
    parse_reward_spec,
)
from codex_hydrogym.training.rewards import (
    DeterministicRewardWrapper,
    compile_reward_spec,
    compiled_reward_from_config,
    frozen_training_fingerprint,
    parse_compiled_reward,
    reward_terms,
)


EVIDENCE_DIGEST = "a" * 64
APPROVAL_DIGEST = "b" * 64


def _spec(**overrides) -> RewardSpec:
    values = {
        "evidence_digest": EVIDENCE_DIGEST,
        "control_l1_weight": 0.25,
        "action_delta_l2_weight": 0.1,
    }
    values.update(overrides)
    return RewardSpec(**values)


def test_reward_spec_v2_contains_no_ppo_or_budget_fields_and_has_stable_digest():
    spec = _spec()
    payload = spec.as_dict()

    assert payload == {
        "evidence_digest": EVIDENCE_DIGEST,
        "control_l1_weight": 0.25,
        "action_delta_l2_weight": 0.1,
        "formula_version": REWARD_FORMULA_VERSION,
        "schema_version": REWARD_SPEC_SCHEMA_VERSION,
    }
    assert len(spec.canonical_digest()) == 64
    assert spec.canonical_digest() == _spec().canonical_digest()
    assert spec.canonical_digest() != _spec(control_l1_weight=0.3).canonical_digest()

    with pytest.raises(ValueError, match="reward-only v2"):
        parse_reward_spec({**payload, "learning_rate": 1.0e-4})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("control_l1_weight", 0.049),
        ("control_l1_weight", 1.001),
        ("action_delta_l2_weight", -0.001),
        ("action_delta_l2_weight", 0.251),
    ],
)
def test_reward_spec_v2_enforces_coefficient_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        _spec(**{field: value})


def test_compiler_binds_calibration_approval_and_preserves_frozen_training_config():
    baseline = KolmogorovPPOConfig(run_name="reward_baseline")
    compiled = compile_reward_spec(
        _spec(),
        reference_tke=3.0,
        calibration_evidence_digest=EVIDENCE_DIGEST,
        approval_digest=APPROVAL_DIGEST,
    )
    candidate = compiled.apply(baseline)

    assert frozen_training_fingerprint(candidate) == frozen_training_fingerprint(baseline)
    assert candidate.learning_rate == baseline.learning_rate
    assert candidate.gamma == baseline.gamma
    assert candidate.gae_lambda == baseline.gae_lambda
    assert candidate.total_timesteps == baseline.total_timesteps
    assert compiled_reward_from_config(candidate) == compiled
    assert parse_compiled_reward(compiled.as_dict()) == compiled

    tampered = dict(compiled.as_dict())
    tampered["reference_tke"] = 4.0
    with pytest.raises(ValueError, match="canonical payload"):
        parse_compiled_reward(tampered)

    with pytest.raises(ValueError, match="calibration evidence"):
        compile_reward_spec(
            _spec(),
            reference_tke=3.0,
            calibration_evidence_digest="c" * 64,
            approval_digest=APPROVAL_DIGEST,
        )


def test_config_rejects_tampered_compiled_digest_on_recovery():
    baseline = KolmogorovPPOConfig(run_name="reward_baseline")
    compiled = compile_reward_spec(
        _spec(),
        reference_tke=3.0,
        calibration_evidence_digest=EVIDENCE_DIGEST,
        approval_digest=APPROVAL_DIGEST,
    )
    candidate = compiled.apply(baseline)
    tampered = replace(candidate, reward_compiled_digest="c" * 64)

    with pytest.raises(ValueError, match="reward_compiled_digest"):
        compiled_reward_from_config(tampered)


def test_reward_terms_are_exact_in_eager_and_jit_execution():
    compiled = compile_reward_spec(
        _spec(),
        reference_tke=2.0,
        calibration_evidence_digest=EVIDENCE_DIGEST,
        approval_digest=APPROVAL_DIGEST,
    )

    eager = reward_terms(mean_tke=4.0, control_l1=2.0, action_delta_l2=4.0, compiled=compiled)
    jitted = jax.jit(
        lambda tke, effort, delta: reward_terms(
            mean_tke=tke,
            control_l1=effort,
            action_delta_l2=delta,
            compiled=compiled,
        )["reward_total"]
    )(4.0, 2.0, 4.0)

    assert float(eager["reward_tke"]) == pytest.approx(-2.0)
    assert float(eager["reward_action_l1"]) == pytest.approx(-0.25)
    assert float(eager["reward_action_delta_l2"]) == pytest.approx(-0.1)
    assert float(eager["reward_total"]) == pytest.approx(-2.35)
    assert float(jitted) == pytest.approx(-2.35)


class _FakeEnvironment:
    def step(self, key, state, action, params=None):
        del key, params
        info = {
            "mean_tke": state,
            "control_l1": jnp.sum(jnp.abs(action)),
            "action_delta_l2": jnp.sum(jnp.square(action - 0.5)),
            "reward_tke": -999.0,
            "reward_action_l1": -999.0,
            "reward_total": -999.0,
        }
        return state + 1.0, state + 1.0, -999.0, False, info


def test_wrapper_replaces_legacy_reward_and_decomposition():
    compiled = compile_reward_spec(
        _spec(),
        reference_tke=2.0,
        calibration_evidence_digest=EVIDENCE_DIGEST,
        approval_digest=APPROVAL_DIGEST,
    )
    wrapper = DeterministicRewardWrapper(_FakeEnvironment(), compiled)

    observation, state, reward, done, info = wrapper.step(
        jax.random.PRNGKey(0),
        jnp.asarray(4.0),
        jnp.asarray([0.5, -0.5, 0.0, 0.0]),
    )

    assert float(observation) == pytest.approx(5.0)
    assert float(state) == pytest.approx(5.0)
    assert done is False
    assert float(reward) == pytest.approx(-2.1625)
    assert float(info["reward_total"]) == pytest.approx(-2.1625)
    assert float(
        info["reward_tke"] + info["reward_action_l1"] + info["reward_action_delta_l2"]
    ) == pytest.approx(float(info["reward_total"]))
