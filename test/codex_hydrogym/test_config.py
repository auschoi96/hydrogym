"""Configuration contracts for the codex_hydrogym experiment harness."""

import numpy as np
import pytest

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.config import FORCED_MODE_QUADRATURE_BASIS_VERSION, KolmogorovPPOConfig, config_fingerprint
from hydrogym.jax.kolmogorov_contract import (
    LEGACY_SPEED_GRID_OBSERVATION_MODE,
    LEGACY_SPEED_GRID_OBSERVATION_VERSION,
    SIGNED_FORCED_MODE_OBSERVATION_MODE,
    SIGNED_FORCED_MODE_OBSERVATION_VERSION,
)


def test_default_config_is_labeled_and_has_integral_work_units():
    config = KolmogorovPPOConfig()

    assert config.project_label == PROJECT_LABEL == "codex_hydrogym"
    assert config.total_batch_size == config.num_envs * config.num_steps
    assert config.num_updates * config.total_batch_size == config.total_timesteps
    assert config.save_steps == 1_000
    assert config.action_steps == 10_000
    assert config.action_steps % config.save_steps == 0
    assert config.run_name.startswith(PROJECT_LABEL)
    assert config.forcing_phase == 0.0
    assert config.actuation_basis_version == FORCED_MODE_QUADRATURE_BASIS_VERSION
    assert config.observation_mode == LEGACY_SPEED_GRID_OBSERVATION_MODE
    assert config.observation_contract_version == LEGACY_SPEED_GRID_OBSERVATION_VERSION


def test_config_rejects_accidental_zero_physics_reward():
    with pytest.raises(ValueError, match="reward_alpha=0"):
        KolmogorovPPOConfig(reward_alpha=0.0)

    explicit_stub = KolmogorovPPOConfig(reward_alpha=0.0, allow_zero_physics_reward=True)
    assert explicit_stub.reward_alpha == 0.0


def test_config_requires_explicitly_diverse_vectorized_resets():
    with pytest.raises(ValueError, match="initial_perturbation_amplitude"):
        KolmogorovPPOConfig(initial_perturbation_amplitude=0.0, num_envs=2)

    deterministic = KolmogorovPPOConfig(
        initial_perturbation_amplitude=0.0,
        num_envs=2,
        allow_deterministic_resets=True,
    )
    assert deterministic.allow_deterministic_resets is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grid_size": (15, 16)}, "even"),
        ({"grid_size": (16, 8)}, "actuator"),
        ({"obs_size": 7}, "divide"),
        ({"save_time": 0.3}, "integer multiple"),
        ({"num_minibatches": 3}, "divisible"),
        ({"total_timesteps": 4_100}, "multiple"),
        ({"forcing_phase": float("nan")}, "forcing_phase"),
        ({"actuation_basis_version": "legacy_sine_modes"}, "actuation_basis_version"),
        ({"observation_mode": "unsigned_private_sensor"}, "observation_mode"),
        (
            {
                "observation_mode": SIGNED_FORCED_MODE_OBSERVATION_MODE,
                "observation_contract_version": LEGACY_SPEED_GRID_OBSERVATION_VERSION,
            },
            "observation_contract_version",
        ),
    ],
)
def test_config_rejects_shape_and_batch_settings_that_break_jit(overrides, message):
    with pytest.raises(ValueError, match=message):
        KolmogorovPPOConfig(**overrides)


def test_mlflow_parameters_are_flat_and_include_reproducibility_fields():
    params = KolmogorovPPOConfig(seed=17, precision="float64").as_mlflow_params()

    assert params["project_label"] == "codex_hydrogym"
    assert params["seed"] == 17
    assert params["precision"] == "float64"
    assert params["grid_size"] == "64x64"
    assert params["num_updates"] > 0
    assert all(not isinstance(value, (dict, list, tuple)) for value in params.values())


def test_json_style_grid_list_is_canonicalized_to_a_tuple():
    config = KolmogorovPPOConfig(grid_size=[64, 64])

    assert config.grid_size == (64, 64)


def test_forcing_phase_and_action_basis_are_checkpoint_provenance():
    default = KolmogorovPPOConfig()
    phased = KolmogorovPPOConfig(forcing_phase=np.pi / 2.0)

    assert default.flow_config()["forcing_phase"] == 0.0
    assert phased.flow_config()["forcing_phase"] == np.pi / 2.0
    assert config_fingerprint(default) != config_fingerprint(phased)


def test_signed_observation_contract_is_resolved_and_fingerprinted():
    default = KolmogorovPPOConfig()
    signed = KolmogorovPPOConfig(observation_mode=SIGNED_FORCED_MODE_OBSERVATION_MODE)

    assert signed.observation_contract_version == SIGNED_FORCED_MODE_OBSERVATION_VERSION
    assert signed.environment_config()["observation_mode"] == SIGNED_FORCED_MODE_OBSERVATION_MODE
    assert config_fingerprint(default) != config_fingerprint(signed)
