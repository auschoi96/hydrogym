"""Runtime physics-gate contracts for codex_hydrogym training."""

import jax
import numpy as np
import pytest

from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.training.ppo import make_train
from codex_hydrogym.training.validation import PhysicsValidationError, validate_training_result


def _validation_config() -> KolmogorovPPOConfig:
    return KolmogorovPPOConfig(
        run_name="validation",
        seed=31,
        grid_size=(16, 16),
        obs_size=4,
        dt=0.05,
        action_time=0.1,
        save_time=0.05,
        initial_perturbation_amplitude=1.0e-3,
        max_episode_steps=2,
        num_envs=1,
        num_steps=1,
        total_timesteps=1,
        update_epochs=1,
        num_minibatches=1,
    )


def test_valid_training_result_passes_all_physics_gates():
    config = _validation_config()
    result = jax.jit(make_train(config))(jax.random.PRNGKey(config.seed))

    report = validate_training_result(config, result["runner_state"], result["metrics"])

    assert report.project_label == "codex_hydrogym"
    assert report.passed
    assert len(report.gates) >= 8
    assert all(gate.passed for gate in report.gates)


def test_reward_tampering_fails_the_named_gate():
    config = _validation_config()
    result = jax.jit(make_train(config))(jax.random.PRNGKey(config.seed))
    metrics = dict(result["metrics"])
    metrics["reward_total"] = np.asarray(metrics["reward_total"]) + 1.0

    report = validate_training_result(config, result["runner_state"], metrics)

    assert not report.passed
    assert not next(gate for gate in report.gates if gate.name == "reward_decomposition_identity").passed
    with pytest.raises(PhysicsValidationError, match="reward_decomposition_identity"):
        report.raise_if_failed()
