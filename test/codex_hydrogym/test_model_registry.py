"""MLflow controller packaging and Unity Catalog promotion contracts."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.modeling.registry import (
    CANDIDATE_ALIAS,
    PRODUCTION_ALIAS,
    REGISTRY_URI,
    log_policy_model,
    promote_policy_model,
    validate_registered_model_name,
)
from codex_hydrogym.tracking import evaluation_context_fingerprint
from codex_hydrogym.training.rewards import frozen_training_fingerprint
from codex_hydrogym.training.runner import run_training, smoke_config
from codex_hydrogym.training.validation import PhysicsValidationError


class _FakePyfunc:
    def __init__(self):
        self.kwargs = None
        self.policy_manifest = None
        self.environment = None

    def log_model(self, **kwargs):
        self.kwargs = kwargs
        self.policy_manifest = json.loads((Path(kwargs["artifacts"]["policy"]) / "manifest.json").read_text())
        self.environment = json.loads(Path(kwargs["artifacts"]["environment"]).read_text())
        assert (Path(kwargs["artifacts"]["checkpoint"]) / "manifest.json").is_file()
        return SimpleNamespace(model_uri="models:/m-codex-hydrogym", registered_model_version="7")


class _FakeMlflow:
    def __init__(self, run_id="candidate-run"):
        self._active_run = SimpleNamespace(info=SimpleNamespace(run_id=run_id))
        self.models = SimpleNamespace(
            infer_signature=lambda inputs, outputs: (tuple(inputs.columns), tuple(outputs.columns))
        )
        self.pyfunc = _FakePyfunc()
        self.registry_uris = []
        self.tags = {}

    def active_run(self):
        return self._active_run

    def set_registry_uri(self, uri):
        self.registry_uris.append(uri)

    def set_tags(self, tags):
        self.tags.update(tags)


class _FakeRegistryClient:
    def __init__(self):
        self.registered_tags = []
        self.version_tags = []
        self.aliases = []
        self.operations = []
        self.runs = {}
        self.model_version = None

    def set_registered_model_tag(self, name, key, value):
        self.registered_tags.append((name, key, value))

    def set_model_version_tag(self, name, version, key, value):
        self.version_tags.append((name, str(version), key, value))
        self.operations.append(("version_tag", name, str(version), key, value))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases.append((name, alias, str(version)))
        self.operations.append(("alias", name, alias, str(version)))

    def get_model_version_by_alias(self, name, alias):
        assert alias == CANDIDATE_ALIAS
        return SimpleNamespace(version="7")

    def get_model_version(self, name, version):
        return self.model_version

    def get_run(self, run_id):
        return self.runs[run_id]


@pytest.fixture(scope="module")
def trained_result(tmp_path_factory):
    output = tmp_path_factory.mktemp("codex_hydrogym_registry") / "codex_hydrogym_training"
    return run_training(smoke_config(), output)


def _evidence_run(
    run_id: str,
    *,
    mean_tke: float,
    control_l1: float,
    context: str,
    frozen_training: str,
    evidence_digest: str,
):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            artifact_uri=f"file:///codex_hydrogym/{run_id}",
            status="FINISHED",
        ),
        data=SimpleNamespace(
            metrics={
                "heldout/mean_tke": mean_tke,
                "heldout/control_l1": control_l1,
                "heldout/reward_total": -mean_tke - control_l1,
                "heldout/physics_all_passed": 1.0,
            },
            tags={
                f"{PROJECT_LABEL}.evaluation_context_fingerprint": context,
                f"{PROJECT_LABEL}.frozen_training_fingerprint": frozen_training,
                f"{PROJECT_LABEL}.heldout_evidence_digest": evidence_digest,
            },
        ),
    )


def test_uc_name_requires_three_levels_no_placeholder_and_project_label():
    assert validate_registered_model_name("catalog.schema.codex_hydrogym_controller") == (
        "catalog.schema.codex_hydrogym_controller"
    )
    with pytest.raises(ValueError, match="catalog.schema.model"):
        validate_registered_model_name("schema.codex_hydrogym_controller")
    with pytest.raises(ValueError, match="placeholder"):
        validate_registered_model_name("REPLACE.catalog.codex_hydrogym_controller")
    with pytest.raises(ValueError, match="start with codex_hydrogym"):
        validate_registered_model_name("catalog.schema.controller")


def test_passing_controller_is_packaged_registered_and_given_candidate_alias(trained_result):
    mlflow = _FakeMlflow()
    client = _FakeRegistryClient()
    name = "catalog.schema.codex_hydrogym_controller"

    record = log_policy_model(
        config=smoke_config(),
        runner_state=trained_result.runner_state,
        validation=trained_result.validation,
        artifact_paths=trained_result.artifacts,
        final_checkpoint=trained_result.checkpoints[-1],
        registered_model_name=name,
        mlflow_module=mlflow,
        mlflow_client=client,
    )

    assert mlflow.registry_uris == [REGISTRY_URI]
    assert mlflow.pyfunc.kwargs["registered_model_name"] == name
    assert mlflow.pyfunc.kwargs["name"].startswith(PROJECT_LABEL)
    assert set(mlflow.pyfunc.kwargs["artifacts"]) == {
        "policy",
        "checkpoint",
        "config",
        "physics_validation",
        "environment",
    }
    assert mlflow.pyfunc.policy_manifest["physics_validation_passed"] is True
    assert mlflow.pyfunc.policy_manifest["deterministic_inference"] is True
    assert mlflow.pyfunc.policy_manifest["source_run_id"] == "candidate-run"
    expected_frozen = frozen_training_fingerprint(smoke_config())
    assert mlflow.pyfunc.policy_manifest["frozen_training_fingerprint"] == expected_frozen
    assert mlflow.pyfunc.kwargs["metadata"]["frozen_training_fingerprint"] == expected_frozen
    assert mlflow.pyfunc.kwargs["tags"]["frozen_training_fingerprint"] == expected_frozen
    assert mlflow.pyfunc.environment["source_run_id"] == "candidate-run"
    assert "jax" in mlflow.pyfunc.environment["dependencies"]
    assert client.aliases == [(name, CANDIDATE_ALIAS, "7")]
    assert record.registered_model_version == "7"
    assert record.alias == CANDIDATE_ALIAS
    assert mlflow.tags[f"{PROJECT_LABEL}.registered_model_name"] == name


def test_failed_physics_never_reaches_mlflow_model_logging(trained_result):
    failed = replace(trained_result.validation, passed=False)
    with pytest.raises(PhysicsValidationError, match="every physics gate"):
        log_policy_model(
            config=smoke_config(),
            runner_state=trained_result.runner_state,
            validation=failed,
            artifact_paths=trained_result.artifacts,
            final_checkpoint=trained_result.checkpoints[-1],
            mlflow_module=_FakeMlflow(),
        )


def test_production_alias_requires_candidate_ownership_and_heldout_improvement():
    mlflow = _FakeMlflow()
    client = _FakeRegistryClient()
    name = "catalog.schema.codex_hydrogym_controller"
    context = evaluation_context_fingerprint(smoke_config())
    frozen_training = frozen_training_fingerprint(smoke_config())
    client.model_version = SimpleNamespace(
        tags={
            "project": PROJECT_LABEL,
            "physics_validation_passed": "true",
            "source_run_id": "candidate-run",
            "evaluation_context_fingerprint": context,
            "frozen_training_fingerprint": frozen_training,
        }
    )
    client.runs = {
        "baseline-run": _evidence_run(
            "baseline-run",
            mean_tke=10.0,
            control_l1=2.0,
            context=context,
            frozen_training=frozen_training,
            evidence_digest="a" * 64,
        ),
        "candidate-run": _evidence_run(
            "candidate-run",
            mean_tke=9.0,
            control_l1=2.2,
            context=context,
            frozen_training=frozen_training,
            evidence_digest="b" * 64,
        ),
    }

    record = promote_policy_model(
        registered_model_name=name,
        registered_model_version="7",
        baseline_run_id="baseline-run",
        candidate_run_id="candidate-run",
        mlflow_module=mlflow,
        mlflow_client=client,
    )

    assert record.alias == PRODUCTION_ALIAS
    assert client.aliases == [(name, PRODUCTION_ALIAS, "7")]
    assert (name, "7", "promotion_state", "production") in client.version_tags
    assert (name, "7", "promotion_baseline_evidence_digest", "a" * 64) in client.version_tags
    assert (name, "7", "promotion_candidate_evidence_digest", "b" * 64) in client.version_tags
    assert client.operations[-1] == ("alias", name, PRODUCTION_ALIAS, "7")

    client.model_version.tags["source_run_id"] = "another-run"
    with pytest.raises(ValueError, match="does not own"):
        promote_policy_model(
            registered_model_name=name,
            registered_model_version="7",
            baseline_run_id="baseline-run",
            candidate_run_id="candidate-run",
            mlflow_module=mlflow,
            mlflow_client=client,
        )

    client.model_version.tags["source_run_id"] = "candidate-run"
    client.runs["baseline-run"].data.tags[f"{PROJECT_LABEL}.frozen_training_fingerprint"] = "f" * 64
    client.runs["candidate-run"].data.tags[f"{PROJECT_LABEL}.frozen_training_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="model frozen-training fingerprint"):
        promote_policy_model(
            registered_model_name=name,
            registered_model_version="7",
            baseline_run_id="baseline-run",
            candidate_run_id="candidate-run",
            mlflow_module=mlflow,
            mlflow_client=client,
        )
