# Databricks notebook source
# ruff: noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC # HydroGym AIR entry-point engineering validation
# MAGIC
# MAGIC This notebook performs a **zero-CFD** validation of the replication wrapper's
# MAGIC MLflow ownership/teardown behavior and self-describing summary schema. It uses
# MAGIC synthetic in-memory MLflow doubles and temporary files only. It does not import
# MAGIC or run the CFD solver, inspect reserved cases, alter primary evidence, execute
# MAGIC Gate 0, or train an RL policy.

# COMMAND ----------

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import uuid


WORKSPACE_ROOT = "/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication"


def _widget(name: str, default: str) -> str:
    try:
        return str(dbutils.widgets.get(name))
    except Exception:
        dbutils.widgets.text(name, default)
        return str(dbutils.widgets.get(name))


ENTRYPOINT_PATH = Path(
    _widget(
        "entrypoint_path",
        f"{WORKSPACE_ROOT}/engineering_validation/gate0_replication_entrypoint.py",
    )
)
EXPECTED_ENTRYPOINT_SHA256 = _widget("expected_entrypoint_sha256", "")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_entrypoint() -> ModuleType:
    module_name = f"codex_hydrogym_air_entrypoint_validation_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ENTRYPOINT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the uploaded AIR entry point")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMlflow(ModuleType):
    def __init__(self, *, preexisting_run: bool = False, fail_log_dict: bool = False):
        super().__init__("mlflow")
        self._active = object() if preexisting_run else None
        self.fail_log_dict = fail_log_dict
        self.start_calls: list[dict[str, object]] = []
        self.end_calls: list[str] = []
        self.tags: list[dict[str, object]] = []
        self.logged_dicts: list[tuple[object, str]] = []

    def active_run(self):
        return self._active

    def start_run(self, **kwargs):
        if self._active is not None:
            raise RuntimeError("synthetic MLflow run already active")
        self.start_calls.append(dict(kwargs))
        self._active = object()
        return self._active

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active is None:
            raise RuntimeError("synthetic MLflow run is not active")
        self.end_calls.append(status)
        self._active = None

    def set_tags(self, tags: dict[str, object]) -> None:
        self.tags.append(dict(tags))

    def log_dict(self, payload: object, artifact_file: str) -> None:
        if self.fail_log_dict:
            raise RuntimeError("synthetic log failure")
        self.logged_dicts.append((payload, artifact_file))


def _run_preflight_main(module: ModuleType, fake_mlflow: FakeMlflow) -> int:
    prior_mlflow = sys.modules.get("mlflow")
    prior_action = os.environ.get("CODEX_HYDROGYM_ACTION")
    prior_run_id = os.environ.get("MLFLOW_RUN_ID")
    sys.modules["mlflow"] = fake_mlflow
    os.environ["CODEX_HYDROGYM_ACTION"] = "preflight"
    os.environ["MLFLOW_RUN_ID"] = "synthetic-zero-cfd-validation"
    module._validate = lambda: (  # type: ignore[attr-defined]
        {"cfds_executed": 0, "validation_role": "engineering_only"},
        object(),
        Path("/unused/protocol.json"),
        {},
    )
    try:
        return int(module.main())  # type: ignore[attr-defined]
    finally:
        if prior_mlflow is None:
            sys.modules.pop("mlflow", None)
        else:
            sys.modules["mlflow"] = prior_mlflow
        if prior_action is None:
            os.environ.pop("CODEX_HYDROGYM_ACTION", None)
        else:
            os.environ["CODEX_HYDROGYM_ACTION"] = prior_action
        if prior_run_id is None:
            os.environ.pop("MLFLOW_RUN_ID", None)
        else:
            os.environ["MLFLOW_RUN_ID"] = prior_run_id


observed_entrypoint_sha256 = hashlib.sha256(ENTRYPOINT_PATH.read_bytes()).hexdigest()
if not EXPECTED_ENTRYPOINT_SHA256 or observed_entrypoint_sha256 != EXPECTED_ENTRYPOINT_SHA256:
    raise RuntimeError("uploaded AIR entry-point SHA-256 does not match the reviewed source")

entrypoint = _load_entrypoint()
checks: dict[str, bool] = {}

with tempfile.TemporaryDirectory(prefix="codex_hydrogym_air_validation_") as temporary_dir:
    artifact_path = Path(temporary_dir) / "artifact.json"
    body = {"schema_version": "synthetic.v1", "value": 7}
    artifact = entrypoint._write_immutable_json(artifact_path, body)
    checks["immutable_summary_digest_is_canonical"] = artifact["artifact_digest"] == _digest(body)
    checks["identical_immutable_write_is_idempotent"] = (
        entrypoint._write_immutable_json(artifact_path, body) == artifact
    )
    try:
        entrypoint._write_immutable_json(
            Path(temporary_dir) / "invalid.json",
            {"artifact_digest": "would_shadow_the_summary_digest"},
        )
    except ValueError:
        checks["artifact_digest_shadowing_is_rejected"] = True
    else:
        checks["artifact_digest_shadowing_is_rejected"] = False

synthetic_result_digest = "a" * 64
summary_body = entrypoint._air_run_summary_body(
    {
        "artifact_digest": synthetic_result_digest,
        "analysis": {"supports_designing_full_gate": True},
        "study_fingerprint": entrypoint.STUDY_FINGERPRINT,
    },
    output_dir=Path("/Workspace/synthetic-zero-cfd-output"),
    runner_exit_code=0,
)
checks["summary_body_reserves_artifact_digest"] = "artifact_digest" not in summary_body
checks["summary_names_result_digest_explicitly"] = (
    summary_body.get("result_artifact_digest") == synthetic_result_digest
)
checks["summary_schema_is_explicit_v2"] = (
    summary_body.get("schema_version")
    == "codex_hydrogym.ensemble_replication_air_summary.v2"
)
checks["summary_body_digest_is_distinct_from_result_digest"] = (
    _digest(summary_body) != synthetic_result_digest
)

success_mlflow = FakeMlflow()
checks["owned_preflight_returns_success"] = _run_preflight_main(entrypoint, success_mlflow) == 0
checks["owned_preflight_attaches_injected_run"] = success_mlflow.start_calls == [
    {"run_id": "synthetic-zero-cfd-validation"}
]
checks["owned_preflight_ends_finished"] = success_mlflow.end_calls == ["FINISHED"]

failure_mlflow = FakeMlflow(fail_log_dict=True)
try:
    _run_preflight_main(entrypoint, failure_mlflow)
except RuntimeError as error:
    checks["owned_failure_preserves_original_error"] = str(error) == "synthetic log failure"
else:
    checks["owned_failure_preserves_original_error"] = False
checks["owned_failure_ends_failed"] = failure_mlflow.end_calls == ["FAILED"]

outer_owned_mlflow = FakeMlflow(preexisting_run=True)
checks["outer_owned_preflight_returns_success"] = (
    _run_preflight_main(entrypoint, outer_owned_mlflow) == 0
)
checks["outer_owned_run_is_not_restarted"] = outer_owned_mlflow.start_calls == []
checks["outer_owned_run_is_not_ended"] = outer_owned_mlflow.end_calls == []

failed_checks = sorted(name for name, passed in checks.items() if passed is not True)
if failed_checks:
    raise RuntimeError("AIR entry-point engineering checks failed: " + ", ".join(failed_checks))

result = {
    "action": "air_entrypoint_engineering_validation",
    "cfds_executed": 0,
    "checks": checks,
    "checks_passed": len(checks),
    "entrypoint_sha256": observed_entrypoint_sha256,
    "primary_evidence_modified": False,
    "reserved_cases_opened": False,
    "rl_training_performed": False,
    "summary_schema_version": summary_body["schema_version"],
}
dbutils.notebook.exit(_canonical(result))
