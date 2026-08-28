"""Physics-gated MLflow logging, UC registration, and alias promotion."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping, Sequence

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.config import KolmogorovPPOConfig, config_fingerprint
from codex_hydrogym.modeling.schema import (
    ACTION_COLUMNS,
    ACTION_DIM,
    ACTION_MAXIMUM,
    ACTION_MINIMUM,
    observation_columns,
)
from codex_hydrogym.tracking import evaluation_context_fingerprint
from codex_hydrogym.training.rewards import frozen_training_fingerprint


REGISTRY_URI = "databricks-uc"
CANDIDATE_ALIAS = "candidate"
PRODUCTION_ALIAS = "production"
POLICY_ASSET_FORMAT = "codex_hydrogym.flax_policy.v1"
MODEL_ARTIFACT_NAME = "codex_hydrogym_controller"

_INFERENCE_DISTRIBUTIONS = ("mlflow", "pandas", "numpy", "jax", "jaxlib", "flax", "distrax")
_TRAINING_DISTRIBUTIONS = (
    *_INFERENCE_DISTRIBUTIONS,
    "optax",
    "gymnax",
    "gymnasium",
    "chex",
    "navix",
    "tree-math",
    "scipy",
)


@dataclass(frozen=True)
class PolicyModelRecord:
    source_run_id: str
    model_uri: str
    registered_model_name: str | None
    registered_model_version: str | None
    alias: str | None


@dataclass(frozen=True)
class PolicyPromotionRecord:
    registered_model_name: str
    registered_model_version: str
    alias: str
    baseline_run_id: str
    candidate_run_id: str


def validate_registered_model_name(name: str) -> str:
    """Require an explicit, labeled three-level Unity Catalog model name."""
    if not isinstance(name, str):
        raise TypeError("registered model name must be a string")
    normalized = name.strip()
    parts = normalized.split(".")
    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise ValueError("registered model name must be catalog.schema.model")
    if any("CONFIGURE" in part.upper() or "REPLACE" in part.upper() for part in parts):
        raise ValueError("registered model name still contains a deployment placeholder")
    if not parts[-1].startswith(PROJECT_LABEL):
        raise ValueError(f"registered model leaf name must start with {PROJECT_LABEL}")
    return normalized


def _installed_versions(distributions: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required dependency is not installed: {distribution}") from error
    return versions


def _normalization_state(config: KolmogorovPPOConfig, runner_state) -> tuple[list[float], list[float]]:
    import jax
    import numpy as np

    dimension = config.obs_size**2
    if not config.normalize_environment:
        return np.zeros(dimension, dtype=np.float64).tolist(), np.ones(dimension, dtype=np.float64).tolist()

    state = runner_state.env_state
    while hasattr(state, "env_state"):
        mean = getattr(state, "mean", None)
        variance = getattr(state, "var", None)
        if mean is not None and variance is not None:
            mean_array = np.asarray(jax.device_get(mean), dtype=np.float64)
            variance_array = np.asarray(jax.device_get(variance), dtype=np.float64)
            if mean_array.ndim > 0 and mean_array.shape[-1] == dimension and variance_array.shape == mean_array.shape:
                means = mean_array.reshape((-1, dimension))
                variances = variance_array.reshape((-1, dimension))
                if not np.allclose(means, means[0], rtol=1.0e-6, atol=1.0e-8):
                    raise RuntimeError("vectorized observation normalizer means diverged by environment")
                if not np.allclose(variances, variances[0], rtol=1.0e-6, atol=1.0e-8):
                    raise RuntimeError("vectorized observation normalizer variances diverged by environment")
                if not np.all(np.isfinite(means[0])) or not np.all(np.isfinite(variances[0])):
                    raise RuntimeError("observation normalizer contains non-finite values")
                if np.any(variances[0] < 0.0):
                    raise RuntimeError("observation normalizer contains negative variance")
                return means[0].tolist(), variances[0].tolist()
        state = state.env_state
    raise RuntimeError("normalized training state does not contain observation statistics")


def _read_checkpoint_manifest(checkpoint_directory: Path, config: KolmogorovPPOConfig) -> dict[str, Any]:
    try:
        manifest_bytes = (checkpoint_directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot package the final checkpoint manifest") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("final checkpoint manifest must be a JSON object")
    if manifest.get("project_label") != PROJECT_LABEL:
        raise RuntimeError("final checkpoint has the wrong project label")
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise RuntimeError("final checkpoint has the wrong config fingerprint")
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest


def _write_model_assets(
    directory: Path,
    *,
    config: KolmogorovPPOConfig,
    runner_state,
    source_run_id: str,
    final_checkpoint: Path,
) -> tuple[Path, Path]:
    from flax import serialization
    import jax

    policy_directory = directory / "codex_hydrogym_policy"
    policy_directory.mkdir()
    params_payload = serialization.to_bytes(jax.device_get(runner_state.train_state.params))
    params_path = policy_directory / "params.msgpack"
    params_path.write_bytes(params_payload)

    checkpoint = _read_checkpoint_manifest(final_checkpoint, config)
    observation_mean, observation_variance = _normalization_state(config, runner_state)
    policy_manifest = {
        "format": POLICY_ASSET_FORMAT,
        "project_label": PROJECT_LABEL,
        "source_run_id": source_run_id,
        "config_fingerprint": config_fingerprint(config),
        "evaluation_context_fingerprint": evaluation_context_fingerprint(config),
        "frozen_training_fingerprint": frozen_training_fingerprint(config),
        "physics_validation_passed": True,
        "deterministic_inference": True,
        "policy_action": "clipped_multivariate_normal_mode",
        "params_file": params_path.name,
        "params_bytes": len(params_payload),
        "params_sha256": hashlib.sha256(params_payload).hexdigest(),
        "checkpoint_format": checkpoint.get("format"),
        "checkpoint_state_sha256": checkpoint.get("state_sha256"),
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "observation_dimension": config.obs_size**2,
        "observation_columns": list(observation_columns(config.obs_size**2)),
        "observation_semantics": "raw_kolmogorov_sensor_grid",
        "observation_normalized_from_checkpoint": bool(config.normalize_environment),
        "observation_mean": observation_mean,
        "observation_variance": observation_variance,
        "normalization_epsilon": 1.0e-8,
        "action_dimension": ACTION_DIM,
        "action_columns": list(ACTION_COLUMNS),
        "action_minimum": ACTION_MINIMUM,
        "action_maximum": ACTION_MAXIMUM,
    }
    (policy_directory / "manifest.json").write_text(
        json.dumps(policy_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    dependency_versions = _installed_versions(_TRAINING_DISTRIBUTIONS)
    environment_path = directory / "codex_hydrogym_environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "project_label": PROJECT_LABEL,
                "source_run_id": source_run_id,
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "executable": sys.executable,
                "dependencies": dependency_versions,
                "jax_platform": jax.default_backend(),
                "cuda_visible_devices_configured": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return policy_directory, environment_path


def log_policy_model(
    *,
    config: KolmogorovPPOConfig,
    runner_state,
    validation,
    artifact_paths,
    final_checkpoint: str | Path,
    registered_model_name: str | None = None,
    mlflow_module=None,
    mlflow_client=None,
) -> PolicyModelRecord:
    """Log one deterministic model and optionally register its exact artifact in UC."""
    if not validation.passed:
        from codex_hydrogym.training.validation import PhysicsValidationError

        raise PhysicsValidationError("codex_hydrogym model logging requires every physics gate to pass")

    mlflow = mlflow_module or importlib.import_module("mlflow")
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("codex_hydrogym model logging requires an active MLflow run")
    source_run_id = active_run.info.run_id
    final_checkpoint = Path(final_checkpoint)

    if registered_model_name is not None:
        registered_model_name = validate_registered_model_name(registered_model_name)
        mlflow.set_registry_uri(REGISTRY_URI)

    model_tags = {
        "project": PROJECT_LABEL,
        "source_run_id": source_run_id,
        "config_fingerprint": config_fingerprint(config),
        "evaluation_context_fingerprint": evaluation_context_fingerprint(config),
        "frozen_training_fingerprint": frozen_training_fingerprint(config),
        "physics_validation_passed": "true",
        "deterministic_inference": "true",
        "training_backend": "jax_ppo",
        "promotion_state": "awaiting_heldout_ppo",
    }
    model_metadata: Mapping[str, Any] = {
        "project_label": PROJECT_LABEL,
        "model_kind": "flax_jax_ppo_controller",
        "source_run_id": source_run_id,
        "config_fingerprint": config_fingerprint(config),
        "evaluation_context_fingerprint": evaluation_context_fingerprint(config),
        "frozen_training_fingerprint": frozen_training_fingerprint(config),
        "physics_validation_passed": True,
        "deterministic_inference": True,
        "observation_dimension": config.obs_size**2,
        "action_dimension": ACTION_DIM,
        "action_bounds": [ACTION_MINIMUM, ACTION_MAXIMUM],
    }

    with tempfile.TemporaryDirectory(prefix="codex_hydrogym_mlflow_model_") as temporary_name:
        temporary_directory = Path(temporary_name)
        policy_directory, environment_path = _write_model_assets(
            temporary_directory,
            config=config,
            runner_state=runner_state,
            source_run_id=source_run_id,
            final_checkpoint=final_checkpoint,
        )
        import numpy as np
        import pandas as pd

        numpy_dtype = np.float64 if config.precision == "float64" else np.float32
        input_example = pd.DataFrame(
            np.zeros((1, config.obs_size**2), dtype=numpy_dtype),
            columns=observation_columns(config.obs_size**2),
        )
        output_example = pd.DataFrame(
            np.zeros((1, ACTION_DIM), dtype=numpy_dtype),
            columns=ACTION_COLUMNS,
        )
        signature = mlflow.models.infer_signature(input_example, output_example)
        repository_root = Path(__file__).parents[2]
        code_paths = [repository_root / PROJECT_LABEL, repository_root / "hydrogym"]
        if any(not path.is_dir() for path in code_paths):
            raise RuntimeError("codex_hydrogym source directories are unavailable for MLflow packaging")
        pip_requirements = [
            f"{distribution}=={version}"
            for distribution, version in _installed_versions(_INFERENCE_DISTRIBUTIONS).items()
        ]
        log_kwargs: dict[str, Any] = {
            "name": MODEL_ARTIFACT_NAME,
            "python_model": str(Path(__file__).with_name("policy_model.py")),
            "artifacts": {
                "policy": str(policy_directory),
                "checkpoint": str(final_checkpoint),
                "config": str(artifact_paths.config),
                "physics_validation": str(artifact_paths.validation),
                "environment": str(environment_path),
            },
            "code_paths": [str(path) for path in code_paths],
            "signature": signature,
            "input_example": input_example,
            "pip_requirements": pip_requirements,
            "metadata": dict(model_metadata),
            "tags": model_tags,
            "model_type": "codex_hydrogym_flax_ppo_controller",
        }
        if registered_model_name is not None:
            log_kwargs.update(
                registered_model_name=registered_model_name,
                await_registration_for=600,
            )
        model_info = mlflow.pyfunc.log_model(**log_kwargs)

    model_uri = str(model_info.model_uri)
    registered_version = getattr(model_info, "registered_model_version", None)
    alias = None
    if registered_model_name is not None:
        if registered_version is None:
            raise RuntimeError("MLflow did not return the registered controller version")
        registered_version = str(registered_version)
        if mlflow_client is None:
            client_type = importlib.import_module("mlflow.tracking").MlflowClient
            mlflow_client = client_type(registry_uri=REGISTRY_URI)
        mlflow_client.set_registered_model_tag(registered_model_name, "project", PROJECT_LABEL)
        mlflow_client.set_registered_model_tag(
            registered_model_name,
            "artifact_kind",
            "flax_jax_ppo_controller",
        )
        for key, value in model_tags.items():
            mlflow_client.set_model_version_tag(registered_model_name, registered_version, key, value)
        mlflow_client.set_registered_model_alias(registered_model_name, CANDIDATE_ALIAS, registered_version)
        alias = CANDIDATE_ALIAS

    mlflow.set_tags(
        {
            f"{PROJECT_LABEL}.policy_model_uri": model_uri,
            f"{PROJECT_LABEL}.registered_model_name": registered_model_name or "",
            f"{PROJECT_LABEL}.registered_model_version": registered_version or "",
            f"{PROJECT_LABEL}.model_alias": alias or "logged_only",
        }
    )
    return PolicyModelRecord(source_run_id, model_uri, registered_model_name, registered_version, alias)


def _validate_promotion_evidence(
    baseline,
    candidate,
    *,
    minimum_tke_improvement: float,
    maximum_control_increase: float,
) -> None:
    if not 0.0 <= minimum_tke_improvement <= 1.0:
        raise ValueError("minimum_tke_improvement must be in [0, 1]")
    if not 0.0 <= maximum_control_increase <= 10.0:
        raise ValueError("maximum_control_increase must be in [0, 10]")
    if baseline.context_fingerprint != candidate.context_fingerprint:
        raise ValueError("production evidence must use the same held-out context")
    if baseline.frozen_training_fingerprint != candidate.frozen_training_fingerprint:
        raise ValueError("production evidence must use the same frozen-training fingerprint")
    if not baseline.physics_gates_passed or not candidate.physics_gates_passed:
        raise ValueError("all baseline and candidate physics gates must pass")
    if baseline.mean_tke <= 0.0:
        raise ValueError("baseline mean TKE must be positive")
    relative_improvement = (baseline.mean_tke - candidate.mean_tke) / baseline.mean_tke
    if relative_improvement < minimum_tke_improvement:
        raise ValueError("candidate did not meet the held-out mean-TKE improvement threshold")
    if baseline.control_l1 <= 0.0:
        if candidate.control_l1 > 1.0e-7:
            raise ValueError("candidate introduced control effort against a zero-control baseline")
    elif (candidate.control_l1 - baseline.control_l1) / baseline.control_l1 > maximum_control_increase:
        raise ValueError("candidate exceeded the held-out control-effort threshold")


def promote_policy_model(
    *,
    registered_model_name: str,
    registered_model_version: str | int,
    baseline_run_id: str,
    candidate_run_id: str,
    minimum_tke_improvement: float = 0.02,
    maximum_control_increase: float = 0.25,
    mlflow_module=None,
    mlflow_client=None,
) -> PolicyPromotionRecord:
    """Move production only when the candidate version matches passing PPO evidence."""
    registered_model_name = validate_registered_model_name(registered_model_name)
    registered_model_version = str(registered_model_version)
    mlflow = mlflow_module or importlib.import_module("mlflow")
    mlflow.set_registry_uri(REGISTRY_URI)
    if mlflow_client is None:
        client_type = importlib.import_module("mlflow.tracking").MlflowClient
        mlflow_client = client_type(registry_uri=REGISTRY_URI)

    candidate_alias = mlflow_client.get_model_version_by_alias(registered_model_name, CANDIDATE_ALIAS)
    if str(candidate_alias.version) != registered_model_version:
        raise ValueError("only the current candidate alias can be considered for production")
    model_version = mlflow_client.get_model_version(registered_model_name, registered_model_version)
    version_tags = dict(model_version.tags or {})
    if version_tags.get("project") != PROJECT_LABEL:
        raise ValueError("registered model version has the wrong project tag")
    if version_tags.get("physics_validation_passed") != "true":
        raise ValueError("registered model version lacks passing physics validation")
    if version_tags.get("source_run_id") != candidate_run_id:
        raise ValueError("candidate evidence run does not own the registered model version")

    optimization = importlib.import_module("codex_hydrogym.genai.optimization")
    baseline = optimization.rollout_evidence_from_run(run_id=baseline_run_id, mlflow_client=mlflow_client)
    candidate = optimization.rollout_evidence_from_run(run_id=candidate_run_id, mlflow_client=mlflow_client)
    if version_tags.get("evaluation_context_fingerprint") != candidate.context_fingerprint:
        raise ValueError("registered model fingerprint does not match candidate evidence")
    if version_tags.get("frozen_training_fingerprint") != candidate.frozen_training_fingerprint:
        raise ValueError("registered model frozen-training fingerprint does not match candidate evidence")
    _validate_promotion_evidence(
        baseline,
        candidate,
        minimum_tke_improvement=minimum_tke_improvement,
        maximum_control_increase=maximum_control_increase,
    )

    promotion_tags = {
        "promotion_state": "production",
        "promotion_baseline_run_id": baseline_run_id,
        "promotion_candidate_run_id": candidate_run_id,
        "promotion_baseline_evidence_digest": baseline.heldout_evidence_digest,
        "promotion_candidate_evidence_digest": candidate.heldout_evidence_digest,
        "promotion_evaluation_context_fingerprint": candidate.context_fingerprint,
        "promotion_frozen_training_fingerprint": candidate.frozen_training_fingerprint,
    }
    for key, value in promotion_tags.items():
        mlflow_client.set_model_version_tag(registered_model_name, registered_model_version, key, value)
    # The production alias is the serving-impacting mutation and must be last.
    # If any provenance write fails, the candidate remains unpromoted.
    mlflow_client.set_registered_model_alias(
        registered_model_name,
        PRODUCTION_ALIAS,
        registered_model_version,
    )
    return PolicyPromotionRecord(
        registered_model_name,
        registered_model_version,
        PRODUCTION_ALIAS,
        baseline_run_id,
        candidate_run_id,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="codex_hydrogym MLflow controller lifecycle")
    subcommands = parser.add_subparsers(dest="command", required=True)
    promote = subcommands.add_parser("promote", help="gate the candidate controller before production")
    promote.add_argument("--registered-model-name", required=True)
    promote.add_argument("--registered-model-version", required=True)
    promote.add_argument("--baseline-run-id", required=True)
    promote.add_argument("--candidate-run-id", required=True)
    promote.add_argument("--minimum-tke-improvement", type=float, default=0.02)
    promote.add_argument("--maximum-control-increase", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mlflow = importlib.import_module("mlflow")
    mlflow.set_tracking_uri("databricks")
    if args.command != "promote":
        raise AssertionError(f"unhandled command: {args.command}")
    record = promote_policy_model(
        registered_model_name=args.registered_model_name,
        registered_model_version=args.registered_model_version,
        baseline_run_id=args.baseline_run_id,
        candidate_run_id=args.candidate_run_id,
        minimum_tke_improvement=args.minimum_tke_improvement,
        maximum_control_increase=args.maximum_control_increase,
        mlflow_module=mlflow,
    )
    print(json.dumps({"project_label": PROJECT_LABEL, **asdict(record)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
