"""End-to-end CPU smoke contracts for the codex_hydrogym PPO learner."""

import os
from dataclasses import replace

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import numpy as np

from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.training.ppo import build_environment, make_initialize, make_train, make_update


def _tiny_config() -> KolmogorovPPOConfig:
    return KolmogorovPPOConfig(
        run_name="cpu_smoke",
        seed=23,
        grid_size=(16, 16),
        obs_size=4,
        dt=0.05,
        action_time=1.0,
        save_time=0.5,
        initial_perturbation_amplitude=1.0e-3,
        max_episode_steps=4,
        num_envs=2,
        num_steps=2,
        total_timesteps=4,
        update_epochs=1,
        num_minibatches=1,
    )


def test_environment_builder_threads_physics_configuration():
    config = _tiny_config()
    env, params = build_environment(config, vectorized=False)

    assert env.name == "KolmogorovFlow"
    assert env.flow.grid_size == (16, 16)
    assert env.flow.obs_size == 4
    assert env.flow.Re == 200.0
    assert params.dt == 0.05
    assert params.action_time == 1.0
    assert params.save_time == 0.5
    assert params.initial_perturbation_amplitude == 1.0e-3
    assert env.low == params.min_action
    assert env.high == params.max_action


def test_tiny_jitted_ppo_training_is_finite_reproducible_and_decomposed():
    config = _tiny_config()
    train = jax.jit(make_train(config))
    key = jax.random.PRNGKey(config.seed)

    first = train(key)
    repeated = train(key)
    first_metrics = first["metrics"]
    repeated_metrics = repeated["metrics"]

    expected_metrics = {
        "mean_tke",
        "control_l1",
        "control_l2",
        "action_delta_l2",
        "reward_tke",
        "reward_action_l1",
        "reward_total",
        "loss_total",
        "loss_actor",
        "loss_value",
        "entropy",
    }
    assert expected_metrics <= set(first_metrics)

    for name in expected_metrics:
        actual = np.asarray(first_metrics[name])
        replay = np.asarray(repeated_metrics[name])
        assert np.isfinite(actual).all(), name
        np.testing.assert_allclose(actual, replay, rtol=1.0e-6, atol=1.0e-7)

    first_params = jax.tree_util.tree_leaves(first["runner_state"][0].params)
    repeated_params = jax.tree_util.tree_leaves(repeated["runner_state"][0].params)
    for actual, replay in zip(first_params, repeated_params, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(replay), rtol=1.0e-6, atol=1.0e-7)


def test_two_update_chunks_are_identical_to_uninterrupted_training():
    config = replace(_tiny_config(), run_name="chunk_equivalence", total_timesteps=8)
    key = jax.random.PRNGKey(config.seed)

    uninterrupted = jax.jit(make_train(config))(key)
    runner_state = jax.jit(make_initialize(config))(key)
    first_chunk = jax.jit(make_update(config, 1))(runner_state)
    second_chunk = jax.jit(make_update(config, 1))(first_chunk["runner_state"])

    assert int(second_chunk["runner_state"].completed_updates) == config.num_updates
    uninterrupted_leaves = jax.tree_util.tree_leaves(uninterrupted["runner_state"])
    chunked_leaves = jax.tree_util.tree_leaves(second_chunk["runner_state"])
    assert len(uninterrupted_leaves) == len(chunked_leaves)
    for actual, expected in zip(chunked_leaves, uninterrupted_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1.0e-6, atol=1.0e-7)

    for name, expected in uninterrupted["metrics"].items():
        actual = np.concatenate(
            [np.asarray(first_chunk["metrics"][name]), np.asarray(second_chunk["metrics"][name])],
            axis=0,
        )
        np.testing.assert_allclose(actual, np.asarray(expected), rtol=1.0e-6, atol=1.0e-7)


def test_update_chunk_count_must_be_positive_integer():
    config = _tiny_config()
    for invalid in (0, -1, 1.5, True):
        try:
            make_update(config, invalid)
        except ValueError as error:
            assert "positive integer" in str(error)
        else:
            raise AssertionError(f"accepted invalid update_count={invalid!r}")
