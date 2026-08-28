"""Restart and integrity contracts for codex_hydrogym Flax checkpoints."""

from dataclasses import replace
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.training.checkpoints import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    config_fingerprint,
    restore_checkpoint,
    save_checkpoint,
)
from codex_hydrogym.training.ppo import make_initialize, make_train, make_update


def _checkpoint_config() -> KolmogorovPPOConfig:
    return KolmogorovPPOConfig(
        run_name="checkpoint",
        seed=29,
        grid_size=(16, 16),
        obs_size=4,
        dt=0.05,
        action_time=1.0,
        save_time=0.5,
        initial_perturbation_amplitude=1.0e-3,
        max_episode_steps=4,
        num_envs=2,
        num_steps=2,
        total_timesteps=8,
        update_epochs=1,
        num_minibatches=1,
    )


def _assert_trees_equal(actual, expected) -> None:
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf), rtol=1.0e-6, atol=1.0e-7)


def test_checkpoint_restores_every_runner_leaf_and_continues_exactly(tmp_path: Path):
    config = _checkpoint_config()
    key = jax.random.PRNGKey(config.seed)
    uninterrupted = jax.jit(make_train(config))(key)["runner_state"]
    initial = jax.jit(make_initialize(config))(key)
    first_update = jax.jit(make_update(config, 1))(initial)["runner_state"]

    checkpoint = save_checkpoint(tmp_path / "codex_hydrogym_checkpoint", first_update, config)
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["project_label"] == "codex_hydrogym"
    assert manifest["config_fingerprint"] == config_fingerprint(config)
    assert manifest["completed_updates"] == 1
    assert manifest["state_file"].endswith(".msgpack")
    assert (checkpoint / manifest["state_file"]).read_bytes()[:1] != b"\x80"

    restored = restore_checkpoint(checkpoint, config, target_state=initial)
    _assert_trees_equal(restored, first_update)
    resumed = jax.jit(make_update(config, 1))(restored)["runner_state"]
    _assert_trees_equal(resumed, uninterrupted)


def test_checkpoint_rejects_wrong_config_corruption_and_implicit_overwrite(tmp_path: Path):
    config = _checkpoint_config()
    initial = make_initialize(config)(jax.random.PRNGKey(config.seed))
    checkpoint = save_checkpoint(tmp_path / "codex_hydrogym_checkpoint", initial, config)

    with pytest.raises(FileExistsError, match="already exists"):
        save_checkpoint(checkpoint, initial, config)
    with pytest.raises(CheckpointCompatibilityError, match="fingerprint"):
        restore_checkpoint(checkpoint, replace(config, seed=config.seed + 1), target_state=initial)

    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    state_path = checkpoint / manifest["state_file"]
    corrupted = bytearray(state_path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0x01
    state_path.write_bytes(corrupted)
    with pytest.raises(CheckpointIntegrityError, match="checksum"):
        restore_checkpoint(checkpoint, config, target_state=initial)
