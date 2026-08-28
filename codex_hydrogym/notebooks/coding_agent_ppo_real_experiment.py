# Databricks notebook source
# ruff: noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC # Real coding-agent PPO experiment
# MAGIC
# MAGIC This is the review and launch surface for `codex_hydrogym.coding_agent_ppo.v1`.
# MAGIC The persistent Job runs a genuine TRL PPO weight update on one H100 in Databricks AI Runtime.
# MAGIC The policy edits one return expression in isolated project-derived source snapshots; generated code
# MAGIC is statically constrained and executed against hidden cases in a time-limited child process.
# MAGIC
# MAGIC Conditions are the identical `Qwen/Qwen2.5-Coder-0.5B-Instruct` policy immediately before and after
# MAGIC 24 PPO updates. The held-out split contains 12 distinct task and group IDs. No CFD, fluid-controller PPO,
# MAGIC production promotion, working-repository mutation, or App deployment occurs.
# MAGIC
# MAGIC MLflow MemAlign is intentionally not impersonated: it requires attributable HUMAN assessments. The completed
# MAGIC run preserves measured patch records and trace IDs, but AIR's native span-body uploads failed at its object-
# MAGIC storage boundary. These artifacts cannot be presented as a trace-native or HUMAN-labeled MemAlign source.
# MAGIC
# MAGIC ## Completed result — August 27, 2026
# MAGIC
# MAGIC Job run `622161538716123` completed `SUCCESS` on one H100 with 24 PPO updates and 192 rollouts.
# MAGIC Held-out exact repairs changed from `1/12` before PPO to `0/12` after PPO. Hidden-case coverage changed
# MAGIC from `15/36` to `7/36`; both conditions had zero unsafe outputs. The frozen directional endpoint therefore
# MAGIC failed (`exploratory_positive=false`). This proves that the weight-update/evaluation path works, but this
# MAGIC particular PPO intervention made the small held-out result worse and does not establish a coding-agent or
# MAGIC MemAlign benefit.

# COMMAND ----------

from __future__ import annotations

import json

from databricks.sdk import WorkspaceClient


PROTOCOL_ID = "codex_hydrogym.coding_agent_ppo.v1"
EXPECTED_JOB_NAME = "codex_hydrogym real coding-agent PPO v1"
DEFAULT_JOB_ID = "683906346871429"
COMPLETED_RUN_ID = 622161538716123
COMPLETED_MLFLOW_RUN_ID = "5517069a75334139ba8a7b8e27417828"

dbutils.widgets.dropdown("action", "review", ["review", "launch", "latest"], "Action")
dbutils.widgets.text("job_id", DEFAULT_JOB_ID, "Persistent AI Runtime Job ID")

ACTION = dbutils.widgets.get("action").strip()
JOB_ID_TEXT = dbutils.widgets.get("job_id").strip()
if not JOB_ID_TEXT.isdigit():
    raise ValueError("job_id has not been bound to the persistent AI Runtime Job")
JOB_ID = int(JOB_ID_TEXT)


def _json_safe(value):
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


workspace = WorkspaceClient()
job = workspace.jobs.get(job_id=JOB_ID)
settings = job.settings
if settings is None or settings.name != EXPECTED_JOB_NAME:
    raise ValueError(f"job {JOB_ID} is not the frozen coding-agent PPO Job")
tasks = list(settings.tasks or [])
if len(tasks) != 1 or tasks[0].ai_runtime_task is None:
    raise ValueError("frozen Job must contain exactly one AI Runtime task")
deployments = list(tasks[0].ai_runtime_task.deployments or [])
if len(deployments) != 1:
    raise ValueError("frozen Job must contain exactly one AI Runtime deployment")
compute = deployments[0].compute
if compute is None or str(compute.accelerator_type) != "GPU_1xH100" or compute.accelerator_count != 1:
    raise ValueError("frozen Job must use exactly one H100")

workspace_host = "https://fevm-austin-choi-omni-agent.cloud.databricks.com"
workspace_id = "7474647489683936"
job_url = f"{workspace_host}/jobs/{JOB_ID}?o={workspace_id}"
completed_run_url = f"{workspace_host}/jobs/{JOB_ID}/runs/{COMPLETED_RUN_ID}?o={workspace_id}"
completed_mlflow_url = (
    f"{workspace_host}/ml/experiments/103455306564903/runs/{COMPLETED_MLFLOW_RUN_ID}?o={workspace_id}"
)

if ACTION == "review":
    result = {
        "project": "codex_hydrogym",
        "protocol_id": PROTOCOL_ID,
        "job_id": JOB_ID,
        "job_url": job_url,
        "job_name": settings.name,
        "task_key": tasks[0].task_key,
        "accelerator_type": str(compute.accelerator_type),
        "accelerator_count": compute.accelerator_count,
        "environment": _json_safe(settings.environments),
        "completed_run_id": COMPLETED_RUN_ID,
        "completed_run_url": completed_run_url,
        "completed_mlflow_run_id": COMPLETED_MLFLOW_RUN_ID,
        "completed_mlflow_url": completed_mlflow_url,
        "completed_result": {
            "updates": 24,
            "rollouts": 192,
            "base_heldout_full_repairs": "1/12",
            "ppo_heldout_full_repairs": "0/12",
            "base_heldout_hidden_cases": "15/36",
            "ppo_heldout_hidden_cases": "7/36",
            "base_heldout_unsafe_outputs": 0,
            "ppo_heldout_unsafe_outputs": 0,
            "exploratory_positive": False,
            "native_trace_status": "trace IDs retained; span bodies missing after AIR object-storage upload failure",
        },
        "ppo_weight_update": True,
        "fluid_ppo": False,
        "cfd": False,
        "memalign": False,
        "memalign_reason": "awaiting attributable HUMAN coding-patch-quality labels",
    }
elif ACTION == "launch":
    run = workspace.jobs.run_now(job_id=JOB_ID)
    run_id = int(run.run_id)
    result = {
        "project": "codex_hydrogym",
        "protocol_id": PROTOCOL_ID,
        "job_id": JOB_ID,
        "run_id": run_id,
        "run_url": f"{workspace_host}/jobs/{JOB_ID}/runs/{run_id}?o={workspace_id}",
        "status": "submitted",
    }
else:
    runs = list(workspace.jobs.list_runs(job_id=JOB_ID, limit=1, active_only=False, completed_only=False))
    if not runs:
        result = {"project": "codex_hydrogym", "protocol_id": PROTOCOL_ID, "job_id": JOB_ID, "latest": None}
    else:
        latest = runs[0]
        result = {
            "project": "codex_hydrogym",
            "protocol_id": PROTOCOL_ID,
            "job_id": JOB_ID,
            "run_id": int(latest.run_id),
            "run_url": latest.run_page_url,
            "state": _json_safe(latest.state),
            "start_time": latest.start_time,
            "end_time": latest.end_time,
        }

display(result)
dbutils.notebook.exit(json.dumps(result, sort_keys=True))
