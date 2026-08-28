"""Databricks App backend for codex_hydrogym review and Gateway calls.

This directory is intentionally self-contained so it can be deployed as the
Databricks App source root without shipping the full CFD training repository.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import os
from typing import Any
from urllib.parse import urlsplit


PROJECT_LABEL = "codex_hydrogym"
FEEDBACK_NAME = "fluid_reward_plausibility"
PLACEHOLDER = "__CONFIGURE_BEFORE_DEPLOY__"


def normalize_workspace_host(value: str) -> str:
    host = value.strip().rstrip("/")
    if "://" not in host:
        host = f"https://{host}"
    parsed = urlsplit(host)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"}:
        raise ValueError("DATABRICKS_HOST must be an HTTPS workspace origin without a path")
    return f"https://{parsed.netloc}"


def reviewer_identity(headers: Mapping[str, str] | None, environ: Mapping[str, str] | None = None) -> str | None:
    normalized_headers = {str(key).lower(): str(value).strip() for key, value in (headers or {}).items()}
    for name in ("x-forwarded-email", "x-forwarded-preferred-username", "x-forwarded-user"):
        if normalized_headers.get(name):
            return normalized_headers[name]
    env = os.environ if environ is None else environ
    local_reviewer = env.get("CODEX_HYDROGYM_REVIEWER", "").strip()
    return local_reviewer or None


def resolve_app_token(*, config_factory=None) -> str:
    """Use Databricks App service-principal OAuth; never hardcode credentials."""
    if config_factory is None:
        config_factory = importlib.import_module("databricks.sdk.core").Config
    authorization = config_factory().authenticate().get("Authorization", "")
    if not authorization.startswith("Bearer ") or len(authorization) <= len("Bearer "):
        raise RuntimeError("Databricks App authentication did not produce a bearer token")
    return authorization[len("Bearer ") :]


@dataclass(frozen=True)
class AppModelPortfolio:
    student: str
    primary_judge: str
    audit_judges: tuple[str, ...]
    reflection_models: tuple[str, ...]
    utility: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AppModelPortfolio":
        env = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = env.get(name, "").strip()
            if not value or value == PLACEHOLDER:
                raise ValueError(f"{name} is not configured with a verified Unity model-service ID")
            return value

        def model_list(name: str, minimum: int) -> tuple[str, ...]:
            values = tuple(item.strip() for item in required(name).split(",") if item.strip())
            if len(values) < minimum:
                raise ValueError(f"{name} requires at least {minimum} model-service IDs")
            return values

        return cls(
            student=required("CODEX_HYDROGYM_STUDENT_MODEL"),
            primary_judge=required("CODEX_HYDROGYM_PRIMARY_JUDGE_MODEL"),
            audit_judges=model_list("CODEX_HYDROGYM_AUDIT_JUDGE_MODELS", 3),
            reflection_models=model_list("CODEX_HYDROGYM_REFLECTION_MODELS", 2),
            utility=required("CODEX_HYDROGYM_SMALL_TASK_MODEL"),
        )

    def role_models(self) -> dict[str, str]:
        roles = {
            "Student · reward proposal": self.student,
            "Primary judge · MemAlign target": self.primary_judge,
            "Utility · small tasks": self.utility,
        }
        roles.update({f"Audit judge {index}": model for index, model in enumerate(self.audit_judges, 1)})
        roles.update({f"Reflection model {index}": model for index, model in enumerate(self.reflection_models, 1)})
        return roles


@dataclass(frozen=True)
class ReviewTrace:
    trace_id: str
    request: Any
    response: Any
    timestamp_ms: int | None
    review_state: str
    assessments: tuple[dict[str, Any], ...]


def _trace_tags(trace: Any) -> Mapping[str, Any]:
    return getattr(getattr(trace, "info", None), "tags", {}) or {}


def _assessment_dict(assessment: Any) -> dict[str, Any]:
    source = getattr(assessment, "source", None)
    return {
        "name": getattr(assessment, "name", None),
        "value": getattr(assessment, "value", None),
        "rationale": getattr(assessment, "rationale", None),
        "source_type": str(getattr(source, "source_type", "")),
        "source_id": getattr(source, "source_id", None),
    }


def to_review_trace(trace: Any) -> ReviewTrace:
    info = getattr(trace, "info", None)
    data = getattr(trace, "data", None)
    trace_id = getattr(info, "trace_id", None) or getattr(info, "request_id", None)
    if not trace_id:
        raise ValueError("MLflow trace has no trace ID")
    tags = _trace_tags(trace)
    assessments = tuple(_assessment_dict(item) for item in (getattr(info, "assessments", ()) or ()))
    return ReviewTrace(
        trace_id=str(trace_id),
        request=getattr(data, "request", None),
        response=getattr(data, "response", None),
        timestamp_ms=getattr(info, "timestamp_ms", None),
        review_state=str(tags.get(f"{PROJECT_LABEL}.review_state", "unknown")),
        assessments=assessments,
    )


def search_review_traces(*, experiment_id: str, max_results: int = 50, mlflow_module=None) -> list[ReviewTrace]:
    """Fetch pending and already-labeled traces without unsupported OR filters."""
    if not experiment_id.strip():
        raise ValueError("MLFLOW_EXPERIMENT_ID is required")
    mlflow = mlflow_module or importlib.import_module("mlflow")
    by_id = {}
    for state in ("pending_human", "human_labeled"):
        traces = mlflow.search_traces(
            locations=[experiment_id],
            filter_string=f"tags.`{PROJECT_LABEL}.review_state` = '{state}'",
            order_by=["attributes.timestamp_ms DESC"],
            max_results=max_results,
            return_type="list",
            include_spans=False,
        )
        for trace in traces:
            review_trace = to_review_trace(trace)
            by_id[review_trace.trace_id] = review_trace
    return sorted(by_id.values(), key=lambda item: item.timestamp_ms or 0, reverse=True)[:max_results]


def _already_reviewed(trace: Any, reviewer: str) -> bool:
    for assessment in getattr(getattr(trace, "info", None), "assessments", ()) or ():
        source = getattr(assessment, "source", None)
        if (
            getattr(assessment, "name", None) == FEEDBACK_NAME
            and str(getattr(source, "source_type", None)) == "HUMAN"
            and getattr(source, "source_id", None) == reviewer
        ):
            return True
    return False


def submit_human_feedback(
    *,
    trace_id: str,
    score: float,
    rationale: str,
    reviewer: str,
    mlflow_module=None,
):
    mlflow = mlflow_module or importlib.import_module("mlflow")
    if not trace_id.strip() or not reviewer.strip():
        raise ValueError("trace ID and attributable reviewer identity are required")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1.0 <= float(score) <= 5.0:
        raise ValueError("score must be in [1, 5]")
    rationale = " ".join(rationale.split())
    if not rationale or len(rationale) > 4_000:
        raise ValueError("rationale must contain 1 to 4000 characters")
    trace = mlflow.get_trace(trace_id)
    if trace is None:
        raise ValueError(f"trace does not exist: {trace_id}")
    if _already_reviewed(trace, reviewer):
        raise ValueError("you already submitted fluid_reward_plausibility for this trace")
    entities = importlib.import_module("mlflow.entities")
    assessment = mlflow.log_feedback(
        trace_id=trace_id,
        name=FEEDBACK_NAME,
        value=float(score),
        rationale=rationale,
        source=entities.AssessmentSource(
            source_type=entities.AssessmentSourceType.HUMAN,
            source_id=reviewer,
        ),
        metadata={"project": PROJECT_LABEL, "reviewer": reviewer},
    )
    mlflow.set_trace_tag(trace_id=trace_id, key=f"{PROJECT_LABEL}.review_state", value="human_labeled")
    return assessment


def recent_training_runs(*, experiment_id: str, max_results: int = 20, mlflow_module=None) -> list[dict[str, Any]]:
    mlflow = mlflow_module or importlib.import_module("mlflow")
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`{PROJECT_LABEL}.training_backend` = 'jax_ppo'",
        order_by=["start_time DESC"],
        max_results=max_results,
        output_format="list",
    )
    output = []
    for run in runs:
        output.append(
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "mean_tke": run.data.metrics.get("train/mean_tke"),
                "control_l1": run.data.metrics.get("train/control_l1"),
                "physics_passed": run.data.metrics.get("physics/all_passed") == 1.0,
                "completed_updates": run.data.tags.get(f"{PROJECT_LABEL}.completed_updates"),
                "registered_model": run.data.tags.get(f"{PROJECT_LABEL}.registered_model_name"),
                "model_version": run.data.tags.get(f"{PROJECT_LABEL}.registered_model_version"),
                "model_alias": run.data.tags.get(f"{PROJECT_LABEL}.model_alias"),
                "artifact_uri": run.info.artifact_uri,
            }
        )
    return output


class UnityGateway:
    def __init__(self, *, workspace_host: str, token: str, openai_client=None):
        self.base_url = f"{normalize_workspace_host(workspace_host)}/ai-gateway/mlflow/v1"
        if openai_client is None:
            openai_client = importlib.import_module("openai").OpenAI(
                api_key=token,
                base_url=self.base_url,
                timeout=90.0,
            )
        self.client = openai_client

    def chat(self, *, model: str, prompt: str) -> dict[str, Any]:
        if not prompt.strip() or len(prompt) > 12_000:
            raise ValueError("prompt must contain 1 to 12000 characters")
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1_500,
            extra_headers={
                "Databricks-Ai-Gateway-Request-Tags": json.dumps(
                    {"project": PROJECT_LABEL, "component": "app_model_lab"},
                    sort_keys=True,
                )
            },
        )
        usage = response.usage.model_dump() if response.usage is not None else {}
        return {
            "text": response.choices[0].message.content or "",
            "model": getattr(response, "model", model),
            "request_id": getattr(response, "id", None),
            "usage": usage,
        }
