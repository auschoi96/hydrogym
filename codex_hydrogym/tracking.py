"""MLflow lifecycle helpers for work labeled codex_hydrogym."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from codex_hydrogym import PROJECT_LABEL, PROJECT_TAG


def evaluation_context_fingerprint(config) -> str:
    """Hash exogenous physics and budget fields, excluding candidate tunables."""
    context = {
        "seed": config.seed,
        "precision": config.precision,
        "reynolds_number": config.reynolds_number,
        "forcing_wavenumber": config.forcing_wavenumber,
        "forcing_phase": config.forcing_phase,
        "actuation_basis_version": config.actuation_basis_version,
        "observation_mode": config.observation_mode,
        "observation_contract_version": config.observation_contract_version,
        "grid_size": list(config.grid_size),
        "obs_size": config.obs_size,
        "dt": config.dt,
        "action_time": config.action_time,
        "save_time": config.save_time,
        "initial_perturbation_amplitude": config.initial_perturbation_amplitude,
        "max_episode_steps": config.max_episode_steps,
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "total_timesteps": config.total_timesteps,
        "update_epochs": config.update_epochs,
        "num_minibatches": config.num_minibatches,
    }
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _prefixed_run_name(run_name: str) -> str:
    return run_name if run_name.startswith(PROJECT_LABEL) else f"{PROJECT_LABEL}_{run_name}"


@contextmanager
def managed_mlflow_run(
    *,
    component: str,
    run_name: str = "run",
    extra_tags: Mapping[str, Any] | None = None,
    mlflow_module=None,
) -> Iterator[Any]:
    """Reuse an active run or own exactly one newly attached/created run.

    AI Runtime injects ``MLFLOW_RUN_ID``. When present and no run is active,
    this function attaches to that run before downstream tools such as GEPA
    initialize their own tracking.
    """
    mlflow = mlflow_module or importlib.import_module("mlflow")
    injected_run_id = os.environ.get("MLFLOW_RUN_ID")
    active_run = mlflow.active_run()
    owns_run = active_run is None

    if active_run is not None and injected_run_id and active_run.info.run_id != injected_run_id:
        raise RuntimeError(
            f"Active MLflow run {active_run.info.run_id} does not match injected MLFLOW_RUN_ID {injected_run_id}"
        )

    if owns_run:
        if injected_run_id:
            active_run = mlflow.start_run(run_id=injected_run_id)
            run_origin = "ai_runtime"
        else:
            active_run = mlflow.start_run(run_name=_prefixed_run_name(run_name))
            run_origin = "standalone"
    else:
        run_origin = "ai_runtime" if injected_run_id else "active"

    tags = {
        PROJECT_TAG: PROJECT_LABEL,
        "codex_hydrogym.component": component,
        "codex_hydrogym.run_origin": run_origin,
    }
    if extra_tags:
        tags.update({str(key): str(value) for key, value in extra_tags.items()})
    mlflow.set_tags(tags)

    try:
        yield active_run
    except BaseException:
        if owns_run:
            mlflow.end_run(status="FAILED")
        raise
    else:
        if owns_run:
            mlflow.end_run(status="FINISHED")


def log_training_evidence(
    *,
    config,
    metrics: Mapping[str, Any],
    validation,
    artifact_paths,
    final_checkpoint: str | Path,
    first_update: int,
    mlflow_module=None,
) -> None:
    """Log PPO curves, physics gates, artifacts, and restart state to the active run."""
    mlflow = mlflow_module or importlib.import_module("mlflow")
    if mlflow.active_run() is None:
        raise RuntimeError("log_training_evidence requires an active MLflow run")

    from codex_hydrogym.training.checkpoints import config_fingerprint
    from codex_hydrogym.training.rewards import frozen_training_fingerprint

    mlflow.log_params({f"{PROJECT_LABEL}.{key}": value for key, value in config.as_mlflow_params().items()})
    update_count = len(next(iter(metrics.values())))
    for update_offset in range(update_count):
        update_metrics = {}
        for name, raw_value in metrics.items():
            value = np.asarray(raw_value[update_offset], dtype=np.float64)
            mean = float(np.mean(value))
            if np.isfinite(mean):
                update_metrics[f"train/{name}"] = mean
        mlflow.log_metrics(update_metrics, step=first_update + update_offset + 1)

    completed_updates = first_update + update_count
    gate_metrics = {f"physics/{gate.name}": float(gate.value) for gate in validation.gates}
    gate_metrics["physics/all_passed"] = float(validation.passed)
    mlflow.log_metrics(gate_metrics, step=completed_updates)
    mlflow.set_tags(
        {
            f"{PROJECT_LABEL}.physics_validation_passed": str(bool(validation.passed)).lower(),
            f"{PROJECT_LABEL}.config_fingerprint": config_fingerprint(config),
            f"{PROJECT_LABEL}.evaluation_context_fingerprint": evaluation_context_fingerprint(config),
            f"{PROJECT_LABEL}.frozen_training_fingerprint": frozen_training_fingerprint(config),
            f"{PROJECT_LABEL}.completed_updates": str(completed_updates),
            f"{PROJECT_LABEL}.training_backend": "jax_ppo",
        }
    )

    evidence_path = f"{PROJECT_LABEL}/evidence"
    for path in artifact_paths.files():
        mlflow.log_artifact(str(path), artifact_path=evidence_path)
    mlflow.log_artifacts(str(final_checkpoint), artifact_path=f"{PROJECT_LABEL}/checkpoint")
