"""Static and parameter contracts for codex_hydrogym AI Runtime assets."""

from pathlib import Path

import pytest
import yaml

from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.genai.contracts import RewardSpec
from codex_hydrogym.training.air_entrypoint import (
    alignment_tags_from_air_parameters,
    config_from_air_parameters,
    load_air_parameters,
    registered_model_name_from_air_parameters,
)
from codex_hydrogym.training.rewards import compile_reward_spec


ROOT = Path(__file__).parents[2]
AIR_ROOT = ROOT / "codex_hydrogym" / "deploy" / "air"


def test_h100_workload_is_labeled_single_device_and_builds_valid_config():
    workload = yaml.safe_load((AIR_ROOT / "workload.h100.yaml").read_text(encoding="utf-8"))

    assert workload["experiment_name"].startswith("codex_hydrogym")
    assert workload["mlflow_run_name"].startswith("codex_hydrogym")
    assert workload["compute"] == {"num_accelerators": 1, "accelerator_type": "GPU_1xH100"}
    assert workload["environment"]["version"] == "5"
    assert "jax[cuda12]==0.7.2" in workload["environment"]["dependencies"]
    assert "docker_image" not in workload["environment"]
    assert workload["code_source"]["snapshot"]["root_path"] == "../../.."
    assert workload["code_source"]["snapshot"]["include_paths"] == ["codex_hydrogym", "hydrogym"]
    assert workload["command"].endswith("python -m codex_hydrogym.training.air_entrypoint")
    assert workload["mlflow_experiment_directory"].startswith("/Workspace/")

    config = config_from_air_parameters(workload["parameters"])
    assert config.run_name.startswith("codex_hydrogym")
    assert config.grid_size == (64, 64)
    assert config.num_updates == 32
    assert registered_model_name_from_air_parameters(workload["parameters"]) == (
        "austin_choi_omni_agent_catalog.codex_hydrogym.codex_hydrogym_ppo_controller"
    )
    assert alignment_tags_from_air_parameters(workload["parameters"]) == {
        "codex_hydrogym.alignment_stage": "baseline",
        "codex_hydrogym.prompt_uri": "prompts:/codex_hydrogym_reward_student@baseline",
    }
    assert (
        registered_model_name_from_air_parameters(
            {"registered_model_name": "catalog.schema.codex_hydrogym_ppo_controller"}
        )
        == "catalog.schema.codex_hydrogym_ppo_controller"
    )


def test_air_parameter_file_requires_exact_project_label(tmp_path: Path):
    path = tmp_path / "parameters.yaml"
    path.write_text("project_label: someone_else\nppo: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="codex_hydrogym"):
        load_air_parameters(path)


@pytest.mark.parametrize(
    "parameters",
    [
        {"alignment_stage": "unknown", "prompt_uri": "prompts:/codex_hydrogym_reward_student@baseline"},
        {"alignment_stage": "baseline", "prompt_uri": "prompts:/someone_else@baseline"},
    ],
)
def test_air_alignment_metadata_is_labeled_and_bounded(parameters):
    with pytest.raises(ValueError, match="alignment_stage|prompt_uri"):
        alignment_tags_from_air_parameters(parameters)


def test_aligned_air_run_requires_matching_approved_compiled_reward():
    evidence_digest = "a" * 64
    compiled = compile_reward_spec(
        RewardSpec(
            evidence_digest=evidence_digest,
            control_l1_weight=0.25,
            action_delta_l2_weight=0.1,
        ),
        reference_tke=3.0,
        calibration_evidence_digest=evidence_digest,
        approval_digest="b" * 64,
    )
    config = compiled.apply(KolmogorovPPOConfig(run_name="aligned_trial"))
    parameters = {
        "alignment_stage": "aligned",
        "prompt_uri": (
            "prompts:/austin_choi_omni_agent_catalog.codex_hydrogym."
            "codex_hydrogym_reward_revision/7"
        ),
        "reward_manifest": compiled.as_dict(),
    }

    tags = alignment_tags_from_air_parameters(parameters, config=config)
    assert tags["codex_hydrogym.prompt_uri"] == parameters["prompt_uri"]
    assert tags["codex_hydrogym.reward_compiled_digest"] == compiled.canonical_digest()
    assert tags["codex_hydrogym.reward_approval_digest"] == "b" * 64

    with pytest.raises(ValueError, match="reward_manifest"):
        alignment_tags_from_air_parameters(
            {key: value for key, value in parameters.items() if key != "reward_manifest"},
            config=config,
        )
    with pytest.raises(ValueError, match="does not match"):
        alignment_tags_from_air_parameters(
            parameters,
            config=KolmogorovPPOConfig(run_name="uncompiled"),
        )


def test_air_rejects_fully_qualified_prompt_with_unlabeled_leaf():
    with pytest.raises(ValueError, match="prompt_uri"):
        alignment_tags_from_air_parameters(
            {
                "alignment_stage": "baseline",
                "prompt_uri": "prompts:/catalog.schema.someone_else/1",
            }
        )


def test_dockerfile_uses_air_base_absolute_entrypoint_and_gpu_jax():
    dockerfile = (AIR_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM databricksruntime/air:dcs-base-aws-runtime")
    assert '"jax[cuda12]==0.7.2"' in dockerfile
    assert 'com.databricks.demo.project="codex_hydrogym"' in dockerfile
    assert 'CMD ["python", "-m", "codex_hydrogym.training.air_entrypoint"]' in dockerfile
