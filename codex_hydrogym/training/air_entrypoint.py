"""AI Runtime entrypoint for labeled codex_hydrogym PPO workloads."""

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.modeling.registry import validate_registered_model_name
from codex_hydrogym.training.rewards import compiled_reward_from_config, parse_compiled_reward
from codex_hydrogym.training.runner import run_training


def load_air_parameters(path: str | Path) -> dict[str, Any]:
    """Load the AIR HYPERPARAMETERS_PATH payload and reject unlabeled jobs."""
    parameter_path = Path(path)
    try:
        value = yaml.safe_load(parameter_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read AIR parameters: {parameter_path}") from error
    if not isinstance(value, dict):
        raise ValueError("AIR parameters must be a YAML mapping")
    if value.get("project_label") != PROJECT_LABEL:
        raise ValueError(f"AIR parameters must set project_label: {PROJECT_LABEL}")
    return value


def config_from_air_parameters(parameters: Mapping[str, Any]) -> KolmogorovPPOConfig:
    ppo = parameters.get("ppo")
    if not isinstance(ppo, dict):
        raise ValueError("AIR parameters must contain a ppo mapping")
    try:
        return KolmogorovPPOConfig(**ppo)
    except TypeError as error:
        raise ValueError(f"invalid AIR PPO parameters: {error}") from error


def registered_model_name_from_air_parameters(parameters: Mapping[str, Any]) -> str:
    """Require AIR runs to persist their validated controller in Unity Catalog."""
    name = parameters.get("registered_model_name")
    if name is None:
        raise ValueError("AIR parameters must contain registered_model_name")
    return validate_registered_model_name(name)


def _labeled_prompt_uri(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("prompts:/"):
        raise ValueError("AIR parameters must contain a labeled codex_hydrogym prompt_uri")
    prompt_identifier = value.removeprefix("prompts:/").split("@", 1)[0].split("/", 1)[0]
    if not prompt_identifier or not prompt_identifier.split(".")[-1].startswith(f"{PROJECT_LABEL}_"):
        raise ValueError("AIR parameters must contain a labeled codex_hydrogym prompt_uri")
    return value


def alignment_tags_from_air_parameters(
    parameters: Mapping[str, Any],
    *,
    config: KolmogorovPPOConfig | None = None,
) -> dict[str, str]:
    """Reject prompt-only alignment claims and bind an approved compiled reward."""
    stage = parameters.get("alignment_stage")
    if stage not in {"baseline", "aligned"}:
        raise ValueError("AIR parameters must set alignment_stage to baseline or aligned")
    prompt_uri = _labeled_prompt_uri(parameters.get("prompt_uri"))
    tags = {
        f"{PROJECT_LABEL}.alignment_stage": stage,
        f"{PROJECT_LABEL}.prompt_uri": prompt_uri,
    }
    manifest_value = parameters.get("reward_manifest")
    if stage == "baseline":
        if manifest_value is not None:
            raise ValueError("baseline AIR parameters cannot contain reward_manifest")
        if config is not None and compiled_reward_from_config(config) is not None:
            raise ValueError("baseline AIR config cannot contain a compiled candidate reward")
        return tags

    if config is None:
        raise ValueError("aligned AIR metadata validation requires the materialized PPO config")
    if not isinstance(manifest_value, Mapping):
        raise ValueError("aligned AIR parameters require an immutable reward_manifest")
    manifest = parse_compiled_reward(manifest_value)
    configured = compiled_reward_from_config(config)
    if configured is None or configured != manifest:
        raise ValueError("aligned PPO config does not match its compiled reward_manifest")
    tags.update(
        {
            f"{PROJECT_LABEL}.reward_formula_version": manifest.formula_version,
            f"{PROJECT_LABEL}.reward_spec_digest": manifest.reward_spec_digest,
            f"{PROJECT_LABEL}.reward_evidence_digest": manifest.evidence_digest,
            f"{PROJECT_LABEL}.reward_approval_digest": manifest.approval_digest,
            f"{PROJECT_LABEL}.reward_compiled_digest": manifest.canonical_digest(),
        }
    )
    return tags


def main() -> int:
    parameter_path = os.environ.get("HYPERPARAMETERS_PATH")
    if not parameter_path:
        raise RuntimeError("AI Runtime did not inject HYPERPARAMETERS_PATH")
    parameters = load_air_parameters(parameter_path)
    config = config_from_air_parameters(parameters)
    registered_model_name = registered_model_name_from_air_parameters(parameters)
    run_tags = alignment_tags_from_air_parameters(parameters, config=config)
    checkpoint_every = parameters.get("checkpoint_every", 1)
    output_directory = Path(os.environ.get("CODEX_HYDROGYM_OUTPUT_DIR", "/tmp/codex_hydrogym_air"))

    result = run_training(
        config,
        output_directory,
        checkpoint_every=checkpoint_every,
        enable_mlflow=True,
        registered_model_name=registered_model_name,
        run_tags=run_tags,
    )
    print(
        json.dumps(
            {
                "project_label": PROJECT_LABEL,
                "run_name": config.run_name,
                "completed_updates": int(result.runner_state.completed_updates),
                "physics_validation_passed": result.validation.passed,
                "mlflow_run_id": result.mlflow_run_id,
                "checkpoint": str(result.checkpoints[-1]),
                "policy_model_uri": result.policy_model_uri,
                "registered_model_name": result.registered_model_name,
                "registered_model_version": result.registered_model_version,
                "model_alias": result.model_alias,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
