"""Headless CLI and orchestration for codex_hydrogym PPO training."""

import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.tracking import log_training_evidence, managed_mlflow_run
from codex_hydrogym.training.artifacts import ArtifactPaths, write_training_artifacts
from codex_hydrogym.training.checkpoints import restore_checkpoint, save_checkpoint
from codex_hydrogym.training.frozen_metric_pin import assert_frozen_metric_source
from codex_hydrogym.training.ppo import RunnerState, make_initialize, make_update
from codex_hydrogym.training.validation import PhysicsValidationReport, validate_training_result


@dataclass(frozen=True)
class TrainingRunResult:
    runner_state: RunnerState
    metrics: dict[str, Any]
    validation: PhysicsValidationReport
    artifacts: ArtifactPaths
    checkpoints: tuple[Path, ...]
    mlflow_run_id: str | None
    policy_model_uri: str | None
    registered_model_name: str | None
    registered_model_version: str | None
    model_alias: str | None


def smoke_config() -> KolmogorovPPOConfig:
    """A fast CPU-safe end-to-end configuration, not a quality benchmark."""
    return KolmogorovPPOConfig(
        run_name="local_smoke",
        seed=41,
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


def load_config(path: str | Path) -> KolmogorovPPOConfig:
    """Load a validated config from a JSON object."""
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read training config: {config_path}") from error
    if not isinstance(value, dict):
        raise ValueError("training config must be a JSON object")
    try:
        return KolmogorovPPOConfig(**value)
    except TypeError as error:
        raise ValueError(f"invalid training config fields: {error}") from error


def _combine_metric_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    names = set(chunks[0])
    if any(set(chunk) != names for chunk in chunks[1:]):
        raise RuntimeError("PPO metric keys changed between update chunks")
    return {name: jnp.concatenate([chunk[name] for chunk in chunks], axis=0) for name in sorted(names)}


def run_training(
    config: KolmogorovPPOConfig,
    output_directory: str | Path,
    *,
    checkpoint_every: int = 1,
    resume_from: str | Path | None = None,
    overwrite: bool = False,
    fail_on_physics: bool = True,
    enable_mlflow: bool = False,
    registered_model_name: str | None = None,
    run_tags: Mapping[str, Any] | None = None,
    mlflow_module=None,
    mlflow_client=None,
) -> TrainingRunResult:
    """Train in restartable JIT chunks and emit evidence after every successful run."""
    # Tamper evidence: abort before ANY training work unless the imported frozen-metric
    # source matches the reviewed pin (frozen_control_metric docstring, guard boundary).
    assert_frozen_metric_source()
    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer")
    if registered_model_name is not None and not enable_mlflow:
        raise ValueError("registered_model_name requires enable_mlflow=True")
    if registered_model_name is not None:
        from codex_hydrogym.modeling.registry import validate_registered_model_name

        registered_model_name = validate_registered_model_name(registered_model_name)
    output = Path(output_directory)
    if PROJECT_LABEL not in output.name:
        raise ValueError(f"output directory name must contain {PROJECT_LABEL!r}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mlflow_tags = {f"{PROJECT_LABEL}.training_backend": "jax_ppo"}
    if run_tags:
        mlflow_tags.update({str(key): str(value) for key, value in run_tags.items()})
    run_context = (
        managed_mlflow_run(
            component="ppo_training",
            run_name=config.run_name,
            extra_tags=mlflow_tags,
            mlflow_module=mlflow_module,
        )
        if enable_mlflow
        else nullcontext(None)
    )
    with run_context as active_run:
        key = jax.random.PRNGKey(config.seed)
        runner_state = jax.jit(make_initialize(config))(key)
        if resume_from is not None:
            runner_state = restore_checkpoint(resume_from, config, target_state=runner_state)

        first_update = int(np.asarray(runner_state.completed_updates))
        remaining = config.num_updates - first_update
        if remaining <= 0:
            raise ValueError("checkpoint has no configured PPO updates remaining")

        metric_chunks: list[dict[str, Any]] = []
        checkpoint_paths: list[Path] = []
        compiled_updates: dict[int, Any] = {}
        while remaining:
            update_count = min(checkpoint_every, remaining)
            if update_count not in compiled_updates:
                compiled_updates[update_count] = jax.jit(make_update(config, update_count))
            result = compiled_updates[update_count](runner_state)
            runner_state = result["runner_state"]
            metric_chunks.append(result["metrics"])
            completed = int(np.asarray(runner_state.completed_updates))
            checkpoint_path = output / f"{PROJECT_LABEL}_checkpoint_update_{completed:06d}"
            save_checkpoint(checkpoint_path, runner_state, config, overwrite=overwrite)
            checkpoint_paths.append(checkpoint_path)
            remaining -= update_count

        metrics = _combine_metric_chunks(metric_chunks)
        validation = validate_training_result(config, runner_state, metrics)
        artifacts = write_training_artifacts(
            output,
            config,
            runner_state,
            metrics,
            validation,
            overwrite=overwrite,
        )
        policy_model = None
        if enable_mlflow:
            log_training_evidence(
                config=config,
                metrics=metrics,
                validation=validation,
                artifact_paths=artifacts,
                final_checkpoint=checkpoint_paths[-1],
                first_update=first_update,
                mlflow_module=mlflow_module,
            )
            if validation.passed:
                from codex_hydrogym.modeling.registry import log_policy_model

                policy_model = log_policy_model(
                    config=config,
                    runner_state=runner_state,
                    validation=validation,
                    artifact_paths=artifacts,
                    final_checkpoint=checkpoint_paths[-1],
                    registered_model_name=registered_model_name,
                    mlflow_module=mlflow_module,
                    mlflow_client=mlflow_client,
                )
            else:
                if mlflow_module is None:
                    import mlflow as mlflow_module

                mlflow = mlflow_module
                mlflow.set_tag(
                    f"{PROJECT_LABEL}.model_registration_skipped",
                    "physics_validation_failed",
                )
        if fail_on_physics:
            validation.raise_if_failed()
        mlflow_run_id = active_run.info.run_id if active_run is not None else None

    return TrainingRunResult(
        runner_state,
        metrics,
        validation,
        artifacts,
        tuple(checkpoint_paths),
        mlflow_run_id,
        policy_model.model_uri if policy_model is not None else None,
        policy_model.registered_model_name if policy_model is not None else None,
        policy_model.registered_model_version if policy_model is not None else None,
        policy_model.alias if policy_model is not None else None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run labeled, restartable codex_hydrogym PPO training")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="JSON KolmogorovPPOConfig fields")
    source.add_argument("--smoke", action="store_true", help="run the fast CPU smoke workload")
    parser.add_argument("--output-dir", type=Path, default=Path("codex_hydrogym_outputs"))
    parser.add_argument("--checkpoint-every", type=int, default=1, metavar="UPDATES")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-physics-failures", action="store_true")
    parser.add_argument("--mlflow", action="store_true", help="log the run to the configured MLflow tracking URI")
    parser.add_argument(
        "--mlflow-experiment", help="optional experiment name; otherwise use the active/default experiment"
    )
    parser.add_argument(
        "--registered-model-name",
        help="optional catalog.schema.codex_hydrogym_* Unity Catalog model name",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = smoke_config() if args.smoke else load_config(args.config)
    mlflow_module = None
    if args.mlflow:
        import mlflow

        mlflow_module = mlflow
        if args.mlflow_experiment:
            mlflow.set_experiment(args.mlflow_experiment)
    result = run_training(
        config,
        args.output_dir,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        overwrite=args.overwrite,
        fail_on_physics=not args.allow_physics_failures,
        enable_mlflow=args.mlflow,
        registered_model_name=args.registered_model_name,
        mlflow_module=mlflow_module,
    )
    print(
        json.dumps(
            {
                "project_label": PROJECT_LABEL,
                "run_name": config.run_name,
                "completed_updates": int(np.asarray(result.runner_state.completed_updates)),
                "physics_validation_passed": result.validation.passed,
                "artifact_manifest": str(result.artifacts.manifest),
                "last_checkpoint": str(result.checkpoints[-1]),
                "mlflow_run_id": result.mlflow_run_id,
                "policy_model_uri": result.policy_model_uri,
                "registered_model_name": result.registered_model_name,
                "registered_model_version": result.registered_model_version,
                "model_alias": result.model_alias,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
