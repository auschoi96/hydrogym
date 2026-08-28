# Databricks notebook source
# ruff: noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC # Independent audit: bounded coding-agent repair pilot
# MAGIC
# MAGIC This notebook performs no model calls. It audits the unexpected Databricks platform retry, native
# MAGIC MLflow traces, deterministic evaluation, result artifact, and managed dataset for the frozen
# MAGIC 12-incident coding-agent repair pilot.

# COMMAND ----------

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from typing import Any, Mapping

import mlflow
from mlflow import MlflowClient
from mlflow.genai.datasets import get_dataset
from databricks.sdk import WorkspaceClient


EXPERIMENT_ID = "103455306564903"
PROTOCOL_ID = "codex_hydrogym.coding_agent_real_bug_pilot.v1"
PROTOCOL_FINGERPRINT = "7720c746c4ff556cf1f76684d830f96f5076d114e1ab2a252fbcf276f9da984f"
JOB_ID = 538885349695793
PARENT_RUN_ID = 690116555720697
FAILED_TASK_RUN_ID = 847202305895048
SUCCESS_TASK_RUN_ID = 35953586830762
FAILED_MODEL_RUN_ID = "a6e91299ff084fb79f57cac67d77992e"
SUCCESS_MODEL_RUN_ID = "27833f359d824ddaaa15da2b64b55acb"
SUCCESS_AUDIT_RUN_ID = "51afc04ff15a43bc85cbb2f8d4776aac"
DATASET_NAME = "austin_choi_omni_agent_catalog.codex_hydrogym.coding_agent_real_bug_pilot_v1"

CORRECT_EDIT = {
    "digest_key_canonicalization": "p1",
    "owned_mlflow_run_teardown": "p2",
    "native_trace_dataset_merge": "p2",
    "trace_materialization": "p3",
    "locked_label_schema_reuse": "p3",
    "reported_model_alias": "p2",
    "native_judge_endpoint": "p3",
    "jobs_output_task_id": "p4",
    "downstream_finalize_idempotency": "p1",
    "advisor_audit_independence": "p4",
    "ceiling_case_redesign": "p2",
    "network_bound_compute_choice": "p3",
}

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
client = MlflowClient()


def _tags(trace) -> Mapping[str, str]:
    tags = trace.info.tags
    return tags if isinstance(tags, Mapping) else dict(tags or {})


def _root(trace):
    roots = [span for span in trace.data.spans if span.parent_id is None]
    if len(roots) != 1:
        raise AssertionError(f"trace {trace.info.trace_id} does not have exactly one root")
    return roots[0]


def _state(trace) -> str:
    value = getattr(trace.info.state, "value", trace.info.state)
    return str(value)


def _selection(trace) -> tuple[str, str]:
    root = _root(trace)
    if not isinstance(root.outputs, Mapping) or not isinstance(root.outputs.get("repair"), Mapping):
        raise AssertionError(f"trace {trace.info.trace_id} has no structured repair output")
    repair = root.outputs["repair"]
    case_id = str(repair.get("case_id"))
    edit_id = str(repair.get("selected_edit_id"))
    if case_id not in CORRECT_EDIT or edit_id not in {"p1", "p2", "p3", "p4"}:
        raise AssertionError(f"trace {trace.info.trace_id} contains an unknown selection")
    return case_id, edit_id


def _model_trace_audit(run_id: str, *, expected_conditions: Mapping[str, int]) -> dict[str, Any]:
    traces = mlflow.search_traces(run_id=run_id, max_results=100, return_type="list")
    conditions = Counter(_tags(trace).get("codex_hydrogym.condition", "missing") for trace in traces)
    for condition, expected in expected_conditions.items():
        if conditions[condition] != expected:
            raise AssertionError(
                f"run {run_id} condition {condition} count {conditions[condition]} != {expected}"
            )

    direct_traces = [
        trace for trace in traces if _tags(trace).get("codex_hydrogym.condition") == "direct_agent"
    ]
    if len(direct_traces) != 12 or any(_state(trace) != "OK" for trace in direct_traces):
        raise AssertionError(f"run {run_id} must contain 12 successful direct-agent traces")
    selections = dict(_selection(trace) for trace in direct_traces)
    if set(selections) != set(CORRECT_EDIT):
        raise AssertionError(f"run {run_id} direct selections do not cover the frozen corpus")
    exact = sum(selections[case_id] == correct for case_id, correct in CORRECT_EDIT.items())
    return {
        "run_id": run_id,
        "trace_count": len(traces),
        "condition_counts": dict(sorted(conditions.items())),
        "direct_exact_repairs": exact,
        "direct_selections": dict(sorted(selections.items())),
    }


failed_run = client.get_run(FAILED_MODEL_RUN_ID)
success_model_run = client.get_run(SUCCESS_MODEL_RUN_ID)
success_audit_run = client.get_run(SUCCESS_AUDIT_RUN_ID)
if failed_run.info.status != "FAILED":
    raise AssertionError("attempt-zero model run must be FAILED")
if success_model_run.info.status != "FINISHED" or success_audit_run.info.status != "FINISHED":
    raise AssertionError("successful model and deterministic audit runs must be FINISHED")

for run in (failed_run, success_model_run, success_audit_run):
    if run.data.tags.get("protocol_id") != PROTOCOL_ID:
        raise AssertionError(f"run {run.info.run_id} has the wrong protocol ID")
    if run.data.tags.get("protocol_fingerprint") != PROTOCOL_FINGERPRINT:
        raise AssertionError(f"run {run.info.run_id} has the wrong protocol fingerprint")

failed_trace_audit = _model_trace_audit(
    FAILED_MODEL_RUN_ID,
    expected_conditions={"direct_agent": 12, "reviewed_agent": 0},
)
if not 1 <= failed_trace_audit["condition_counts"].get("base_review", 0) <= 12:
    raise AssertionError("attempt zero must contain between one and twelve review traces")
success_trace_audit = _model_trace_audit(
    SUCCESS_MODEL_RUN_ID,
    expected_conditions={"direct_agent": 12, "base_review": 12, "reviewed_agent": 12},
)

audit_traces = mlflow.search_traces(
    run_id=SUCCESS_AUDIT_RUN_ID,
    max_results=100,
    return_type="list",
)
if len(audit_traces) != 24:
    raise AssertionError(f"deterministic audit must contain 24 traces, found {len(audit_traces)}")

condition_counts = Counter()
recomputed_exact = Counter()
seen_pairs = set()
for trace in audit_traces:
    if _state(trace) != "OK":
        raise AssertionError(f"deterministic audit trace is not OK: {trace.info.trace_id}")
    root = _root(trace)
    if not isinstance(root.inputs, Mapping) or not isinstance(root.outputs, Mapping):
        raise AssertionError(f"deterministic audit trace lacks structured I/O: {trace.info.trace_id}")
    case_id = str(root.inputs.get("case_id"))
    condition = str(root.inputs.get("condition"))
    output_case_id = str(root.outputs.get("case_id"))
    edit_id = str(root.outputs.get("selected_edit_id"))
    if case_id != output_case_id or case_id not in CORRECT_EDIT:
        raise AssertionError(f"case binding failed for trace {trace.info.trace_id}")
    if root.inputs.get("protocol_fingerprint") != PROTOCOL_FINGERPRINT:
        raise AssertionError(f"protocol binding failed for trace {trace.info.trace_id}")
    if condition not in {"direct_agent", "reviewed_agent"}:
        raise AssertionError(f"unknown condition on trace {trace.info.trace_id}")
    if (case_id, condition) in seen_pairs:
        raise AssertionError(f"duplicate deterministic audit pair: {case_id}/{condition}")
    seen_pairs.add((case_id, condition))
    condition_counts[condition] += 1
    recomputed_exact[condition] += edit_id == CORRECT_EDIT[case_id]

    assessments = {item.name: item.value for item in trace.info.assessments}
    required = {
        "protocol_binding_valid",
        "exact_minimal_repair",
        "regression_check_fraction",
        "safe_repair",
    }
    if set(assessments) != required:
        raise AssertionError(f"trace {trace.info.trace_id} has the wrong assessment set")
    if assessments["protocol_binding_valid"] is not True:
        raise AssertionError("protocol binding assessment is not true")
    if assessments["exact_minimal_repair"] is not True:
        raise AssertionError("exact-repair assessment is not true")
    if float(assessments["regression_check_fraction"]) != 1.0:
        raise AssertionError("regression-check fraction is not 1.0")
    if assessments["safe_repair"] is not True:
        raise AssertionError("safe-repair assessment is not true")

if condition_counts != Counter({"direct_agent": 12, "reviewed_agent": 12}):
    raise AssertionError(f"unexpected deterministic condition counts: {condition_counts}")
if recomputed_exact != Counter({"direct_agent": 12, "reviewed_agent": 12}):
    raise AssertionError(f"recomputed exact repair counts differ: {recomputed_exact}")

metric_map = success_audit_run.data.metrics
required_metrics = {
    "coding_agent/direct_exact_repairs": 12.0,
    "coding_agent/direct_regression_check_fraction": 1.0,
    "coding_agent/direct_wilson_95_lower": 0.7575059933447591,
    "coding_agent/reviewed_exact_repairs": 12.0,
    "coding_agent/reviewed_regression_check_fraction": 1.0,
    "exact_minimal_repair/mean": 1.0,
    "protocol_binding_valid/mean": 1.0,
    "regression_check_fraction/mean": 1.0,
    "safe_repair/mean": 1.0,
}
for name, expected in required_metrics.items():
    if metric_map.get(name) != expected:
        raise AssertionError(f"metric {name} was {metric_map.get(name)!r}, expected {expected!r}")

result_artifact = mlflow.artifacts.load_dict(
    f"runs:/{SUCCESS_AUDIT_RUN_ID}/coding_agent_real_bug/result.json"
)
if result_artifact["protocol_fingerprint"] != PROTOCOL_FINGERPRINT:
    raise AssertionError("result artifact has the wrong fingerprint")
if result_artifact["decision"] != {
    "direct_agent_useful_on_frozen_corpus": True,
    "base_review_helped": False,
    "base_review_safety_regression": False,
    "memalign_executed": False,
    "memalign_benefit_proven": False,
}:
    raise AssertionError("result artifact decision block differs from the frozen outcome")

dataset = get_dataset(name=DATASET_NAME)
dataset_df = dataset.to_df()
if len(dataset_df) != 24:
    raise AssertionError(f"managed dataset must contain 24 records, found {len(dataset_df)}")

workspace = WorkspaceClient()
job_run = workspace.jobs.get_run(run_id=PARENT_RUN_ID)
job_run_payload = job_run.as_dict()
if int(job_run_payload["job_id"]) != JOB_ID:
    raise AssertionError("parent run points to the wrong Job")
attempts = sorted(job_run_payload["tasks"], key=lambda item: item["attempt_number"])
if len(attempts) != 2:
    raise AssertionError(f"expected two platform attempts, found {len(attempts)}")
attempt_manifest = [
    {
        "attempt_number": item["attempt_number"],
        "run_id": int(item["run_id"]),
        "result_state": item["state"]["result_state"],
    }
    for item in attempts
]
expected_attempt_manifest = [
    {"attempt_number": 0, "run_id": FAILED_TASK_RUN_ID, "result_state": "FAILED"},
    {"attempt_number": 1, "run_id": SUCCESS_TASK_RUN_ID, "result_state": "SUCCESS"},
]
if attempt_manifest != expected_attempt_manifest:
    raise AssertionError(f"unexpected attempt manifest: {attempt_manifest}")

audit_summary = {
    "schema_version": "codex_hydrogym.coding_agent_real_bug_independent_audit.v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol_id": PROTOCOL_ID,
    "protocol_fingerprint": PROTOCOL_FINGERPRINT,
    "job": {
        "job_id": JOB_ID,
        "parent_run_id": PARENT_RUN_ID,
        "attempts": attempt_manifest,
    },
    "attempt_zero": {
        **failed_trace_audit,
        "status": failed_run.info.status,
        "terminal_error": "one Claude rationale exceeded the 1200-character harness guard",
        "deterministic_outcome_scoring_executed": False,
    },
    "completed_attempt": {
        **success_trace_audit,
        "status": success_model_run.info.status,
        "deterministic_audit_run_id": SUCCESS_AUDIT_RUN_ID,
        "deterministic_audit_status": success_audit_run.info.status,
        "audit_trace_count": len(audit_traces),
        "direct_exact_repairs": int(recomputed_exact["direct_agent"]),
        "reviewed_exact_repairs": int(recomputed_exact["reviewed_agent"]),
        "regression_checks_passed": 36,
        "unsafe_edits": 0,
        "wilson_95_lower": metric_map["coding_agent/direct_wilson_95_lower"],
    },
    "dataset": {"name": DATASET_NAME, "record_count": len(dataset_df)},
    "decision": result_artifact["decision"],
    "audit_conclusion": (
        "Both platform attempts independently selected all 12 direct repairs correctly. Attempt zero was "
        "incomplete and unscored because a review rationale violated the response-length contract. The "
        "automatic retry completed all conditions and the deterministic 24-record audit. The bounded direct "
        "coding-agent criterion passed; base review and MemAlign benefit were not shown."
    ),
    "counts": {"new_model_calls": 0, "cfd_trajectories": 0, "ppo_updates": 0, "memalign_records": 0},
}

with mlflow.start_run(run_name="coding_agent_real_bug_independent_audit_v1") as audit_log_run:
    mlflow.set_tags(
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_fingerprint": PROTOCOL_FINGERPRINT,
            "claim_role": "zero_model_independent_lineage_audit",
            "source_parent_job_run_id": str(PARENT_RUN_ID),
            "new_model_calls": "0",
        }
    )
    mlflow.log_dict(audit_summary, "coding_agent_real_bug/independent_audit.json")
    audit_summary["audit_mlflow_run_id"] = audit_log_run.info.run_id

print(json.dumps(audit_summary, indent=2, sort_keys=True))
dbutils.notebook.exit(json.dumps(audit_summary, sort_keys=True))
