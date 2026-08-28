"""Real local-MLflow integration contract for codex_hydrogym training."""

from pathlib import Path
import os
import subprocess
import sys

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from codex_hydrogym.modeling.schema import ACTION_COLUMNS, observation_columns
from codex_hydrogym.training.runner import run_training, smoke_config


def _artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> set[str]:
    paths: set[str] = set()
    for artifact in client.list_artifacts(run_id, path):
        paths.add(artifact.path)
        if artifact.is_dir:
            paths.update(_artifact_paths(client, run_id, artifact.path))
    return paths


def test_real_mlflow_run_contains_loadable_deterministic_policy(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    old_tracking_uri = mlflow.get_tracking_uri()
    old_registry_uri = mlflow.get_registry_uri()
    artifact_root = tmp_path / "codex_hydrogym_mlflow_artifacts"
    artifact_root.mkdir()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'codex_hydrogym_mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "codex_hydrogym_local_integration",
        artifact_location=artifact_root.as_uri(),
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    try:
        result = run_training(
            smoke_config(),
            tmp_path / "codex_hydrogym_mlflow_run",
            enable_mlflow=True,
            mlflow_module=mlflow,
        )
        client = MlflowClient()
        run = client.get_run(result.mlflow_run_id)

        assert run.info.status == "FINISHED"
        assert run.data.tags["codex_hydrogym.project"] == "codex_hydrogym"
        assert run.data.tags["codex_hydrogym.physics_validation_passed"] == "true"
        assert len(run.data.tags["codex_hydrogym.evaluation_context_fingerprint"]) == 64
        assert len(run.data.tags["codex_hydrogym.frozen_training_fingerprint"]) == 64
        assert run.data.params["codex_hydrogym.seed"] == str(smoke_config().seed)
        assert "train/mean_tke" in run.data.metrics
        assert run.data.metrics["physics/all_passed"] == 1.0

        artifacts = _artifact_paths(client, result.mlflow_run_id)
        assert "codex_hydrogym/evidence/codex_hydrogym_artifact_manifest.json" in artifacts
        assert "codex_hydrogym/checkpoint/manifest.json" in artifacts
        assert any(path.endswith(".msgpack") for path in artifacts)

        assert result.policy_model_uri
        assert result.registered_model_name is None
        assert run.data.tags["codex_hydrogym.policy_model_uri"] == result.policy_model_uri
        assert run.data.tags["codex_hydrogym.model_alias"] == "logged_only"
        model_info = mlflow.models.get_model_info(result.policy_model_uri)
        assert model_info.metadata["project_label"] == "codex_hydrogym"
        assert model_info.metadata["physics_validation_passed"] is True
        assert [item.name for item in model_info.signature.inputs.inputs] == list(
            observation_columns(smoke_config().obs_size**2)
        )
        assert [item.name for item in model_info.signature.outputs.inputs] == list(ACTION_COLUMNS)

        policy = mlflow.pyfunc.load_model(result.policy_model_uri)
        model_input = pd.DataFrame(
            np.zeros((2, smoke_config().obs_size**2), dtype=np.float32),
            columns=observation_columns(smoke_config().obs_size**2),
        )
        first = policy.predict(model_input)
        second = policy.predict(model_input)
        pd.testing.assert_frame_equal(first, second)
        assert tuple(first.columns) == ACTION_COLUMNS
        assert np.all(np.isfinite(first.to_numpy()))
        assert np.max(np.abs(first.to_numpy())) <= 0.5

        clean_environment = dict(os.environ)
        clean_environment.pop("PYTHONPATH", None)
        clean_load = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import mlflow, numpy as np, pandas as pd, sys; "
                    "mlflow.set_tracking_uri(sys.argv[2]); "
                    "model=mlflow.pyfunc.load_model(sys.argv[1]); "
                    "columns=[item.name for item in model.metadata.signature.inputs.inputs]; "
                    "output=model.predict(pd.DataFrame(np.zeros((1,len(columns)),dtype=np.float32),columns=columns)); "
                    "assert output.shape==(1,4); print('codex_hydrogym_clean_load_ok')"
                ),
                result.policy_model_uri,
                mlflow.get_tracking_uri(),
            ],
            cwd=tmp_path,
            env=clean_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert clean_load.returncode == 0, clean_load.stderr
        assert "codex_hydrogym_clean_load_ok" in clean_load.stdout
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(old_tracking_uri)
        mlflow.set_registry_uri(old_registry_uri)
