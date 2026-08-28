"""Fresh Declarative Automation Bundle contracts for codex_hydrogym."""

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def _yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_bundle_targets_only_requested_workspace_and_requires_verified_models():
    bundle = _yaml(ROOT / "databricks.yml")

    assert bundle["bundle"]["name"] == "codex_hydrogym"
    assert set(bundle["targets"]) == {"demo"}
    assert bundle["targets"]["demo"]["workspace"]["host"] == (
        "https://fevm-austin-choi-omni-agent.cloud.databricks.com"
    )
    assert bundle["targets"]["demo"]["mode"] == "production"
    required_model_name = bundle["variables"]["registered_model_name"]
    assert "default" not in required_model_name
    model_services = {
        name: bundle["variables"][name]
        for name in (
            "student_model",
            "primary_judge_model",
            "audit_judge_models",
            "reflection_models",
            "small_task_model",
        )
    }
    assert all(variable["default"] == "__CONFIGURE_BEFORE_DEPLOY__" for variable in model_services.values())
    configured = bundle["targets"]["demo"]["variables"]
    assert configured == {
        "catalog": "austin_choi_omni_agent_catalog",
        "schema": "codex_hydrogym",
        "student_model": "system.ai.kimi-k3",
        "primary_judge_model": "system.ai.claude-opus-5",
        "audit_judge_models": "system.ai.gpt-5-6-sol,system.ai.deepseek-v4-pro-0813,system.ai.glm-5-2",
        "reflection_models": "system.ai.gpt-5-6-sol,system.ai.kimi-k3,system.ai.claude-opus-5",
        "small_task_model": "system.ai.deepseek-v4-flash-0731",
        "registered_model_name": (
            "austin_choi_omni_agent_catalog.codex_hydrogym.codex_hydrogym_ppo_controller"
        ),
    }


def test_all_alignment_workflows_and_app_share_one_managed_experiment():
    experiment_resources = _yaml(ROOT / "resources" / "codex_hydrogym_experiment.experiment.yml")["resources"][
        "experiments"
    ]
    assert experiment_resources == {
        "codex_hydrogym_experiment": {
            "name": "/Shared/codex_hydrogym",
            "tags": [
                {"key": "project", "value": "codex_hydrogym"},
                {"key": "purpose", "value": "fluid_rl_human_alignment"},
            ],
        }
    }

    experiment_reference = "${resources.experiments.codex_hydrogym_experiment.id}"
    jobs = _yaml(ROOT / "resources" / "codex_hydrogym.jobs.yml")["resources"]["jobs"]
    for job_name in ("codex_hydrogym_bootstrap_feedback", "codex_hydrogym_memalign", "codex_hydrogym_gepa"):
        parameters = jobs[job_name]["tasks"][0]["python_wheel_task"]["parameters"]
        experiment_flag = parameters.index("--experiment-id")
        assert parameters[experiment_flag + 1] == experiment_reference

    app = _yaml(ROOT / "resources" / "codex_hydrogym_app.app.yml")["resources"]["apps"]["codex_hydrogym_app"]
    experiment = next(resource for resource in app["resources"] if resource["name"] == "experiment")
    assert experiment["experiment"]["experiment_id"] == experiment_reference


def test_app_is_evidence_only_and_uses_managed_experiment_resource():
    resource = _yaml(ROOT / "resources" / "codex_hydrogym_app.app.yml")["resources"]["apps"][
        "codex_hydrogym_app"
    ]

    assert resource["name"] == "codex-hydrogym"
    assert resource["source_code_path"] == "../codex_hydrogym/appkit/codex-hydrogym"
    assert resource["config"]["command"] == ["npm", "run", "start"]
    env = {item["name"]: item for item in resource["config"]["env"]}
    assert set(env) == {"MLFLOW_EXPERIMENT_ID"}
    assert env["MLFLOW_EXPERIMENT_ID"]["value_from"] == "experiment"
    assert len(resource["resources"]) == 1
    experiment = resource["resources"][0]
    assert experiment["name"] == "experiment"
    assert experiment["experiment"]["permission"] == "CAN_EDIT"
    assert "job" not in experiment


def test_schema_and_registered_controller_are_managed_bundle_resources():
    schema = _yaml(ROOT / "resources" / "codex_hydrogym_schema.schema.yml")["resources"]["schemas"]
    assert schema == {
        "codex_hydrogym_schema": {
            "name": "${var.schema}",
            "catalog_name": "${var.catalog}",
            "comment": "Fresh Unity Catalog schema for codex_hydrogym MLflow models and fluid-RL demo assets.",
        }
    }
    resource = _yaml(ROOT / "resources" / "codex_hydrogym_controller.registered_model.yml")["resources"][
        "registered_models"
    ]

    assert set(resource) == {"codex_hydrogym_controller"}
    controller = resource["codex_hydrogym_controller"]
    assert controller["name"] == "codex_hydrogym_ppo_controller"
    assert controller["catalog_name"] == "${resources.schemas.codex_hydrogym_schema.catalog_name}"
    assert controller["schema_name"] == "${resources.schemas.codex_hydrogym_schema.name}"
    assert "codex_hydrogym" in controller["comment"]


def test_appkit_runtime_manifest_and_bundle_use_the_same_production_command():
    appkit_root = ROOT / "codex_hydrogym" / "appkit" / "codex-hydrogym"
    manifest = _yaml(appkit_root / "app.yaml")
    assert manifest["command"] == _yaml(ROOT / "resources" / "codex_hydrogym_app.app.yml")["resources"][
        "apps"
    ]["codex_hydrogym_app"]["config"]["command"]
    package = _yaml(appkit_root / "package.json")
    assert package["dependencies"]["@databricks/appkit"] == "0.57.0"
    assert package["dependencies"]["@databricks/appkit-ui"] == "0.57.0"
    assert "gradio" not in package["dependencies"]
    assert "streamlit" not in package["dependencies"]

    standalone = _yaml(appkit_root / "databricks.yml")
    app = standalone["resources"]["apps"]["app"]
    assert app["config"]["command"] == ["npm", "run", "start"]
    assert {item["name"] for item in app["config"]["env"]} == {"MLFLOW_EXPERIMENT_ID"}
    assert len(app["resources"]) == 1
    assert app["resources"][0]["experiment"]["experiment_id"] == "${var.experiment_id}"
    assert app["resources"][0]["experiment"]["permission"] == "CAN_EDIT"

    server_source = (appkit_root / "server" / "server.ts").read_text(encoding="utf-8")
    assert "jobs()" not in server_source


def test_serverless_jobs_use_client_four_wheel_and_separate_human_pause_points():
    jobs = _yaml(ROOT / "resources" / "codex_hydrogym.jobs.yml")["resources"]["jobs"]

    assert set(jobs) == {
        "codex_hydrogym_bootstrap_feedback",
        "codex_hydrogym_memalign",
        "codex_hydrogym_gepa",
        "codex_hydrogym_promote",
        "codex_hydrogym_promote_controller",
    }
    for key, job in jobs.items():
        assert key.startswith("codex_hydrogym_")
        assert job["name"].startswith("codex_hydrogym_")
        assert job["max_concurrent_runs"] == 1
        environment = job["environments"][0]
        assert environment["spec"]["client"] == "4"
        dependencies = environment["spec"]["dependencies"]
        assert dependencies[0] == "../dist/*.whl"
        assert any(dependency.startswith("mlflow[databricks]>=3.11.1") for dependency in dependencies)
        task = job["tasks"][0]
        assert task["task_key"].startswith("codex_hydrogym_")
        assert "libraries" not in task
        expected_entry_point = (
            "codex-hydrogym-model" if key == "codex_hydrogym_promote_controller" else "codex-hydrogym-genai"
        )
        assert task["python_wheel_task"]["entry_point"] == expected_entry_point
    assert "depends_on" not in jobs["codex_hydrogym_memalign"]["tasks"][0]
    assert "depends_on" not in jobs["codex_hydrogym_gepa"]["tasks"][0]
