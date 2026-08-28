"""Headless runner and artifact contracts for codex_hydrogym."""

import json
from pathlib import Path

import pytest

from codex_hydrogym.training.runner import load_config, run_training, smoke_config


def test_json_config_loading_canonicalizes_and_labels(tmp_path: Path):
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps({"run_name": "json", "grid_size": [64, 64]}), encoding="utf-8")

    config = load_config(config_path)

    assert config.run_name == "codex_hydrogym_json"
    assert config.grid_size == (64, 64)


def test_headless_runner_checkpoints_validates_and_renders(tmp_path: Path):
    output = tmp_path / "codex_hydrogym_test_run"

    result = run_training(smoke_config(), output, checkpoint_every=1)

    assert int(result.runner_state.completed_updates) == 1
    assert result.validation.passed
    assert result.checkpoints[-1].name.startswith("codex_hydrogym_checkpoint_update_")
    assert (result.checkpoints[-1] / "manifest.json").is_file()
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.artifacts.files())
    manifest = json.loads(result.artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["project_label"] == "codex_hydrogym"
    assert manifest["physics_validation_passed"] is True

    with pytest.raises(FileExistsError, match="not empty"):
        run_training(smoke_config(), output)


def test_runner_requires_labeled_output_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="codex_hydrogym"):
        run_training(smoke_config(), tmp_path / "unlabeled")
