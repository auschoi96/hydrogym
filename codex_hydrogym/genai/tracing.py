"""Shared MLflow tracing for both coding-agent harness arms."""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.genai.contracts import AGENT_FEEDBACK_SCHEMA_VERSION, AgentFeedback, RunBundle, parse_agent_feedback
from codex_hydrogym.genai.harnesses import (
    FeedbackHarness,
    HarnessExecutionError,
    prompt_digest,
    render_feedback_prompt,
    validate_feedback_identity,
)

UC_TRACING_FLAG = "CODEX_HYDROGYM_ENABLE_UC_TRACING"
UC_CATALOG_ENV = "MLFLOW_TRACING_UC_CATALOG_NAME"
UC_SCHEMA_ENV = "MLFLOW_TRACING_UC_SCHEMA_NAME"
UC_WAREHOUSE_ENV = "MLFLOW_TRACING_SQL_WAREHOUSE_ID"
DEFAULT_UC_CATALOG = "austin_choi_omni_agent_catalog"
DEFAULT_UC_SCHEMA = "codex_hydrogym"
DEFAULT_DATABRICKS_HOST = "https://fevm-austin-choi-omni-agent.cloud.databricks.com"


@dataclass(frozen=True)
class UCTracePreflight:
    """Offline diagnostic for the inputs needed by a UC trace destination."""

    enabled: bool
    catalog_name: str | None
    schema_name: str | None
    warehouse_id: str | None
    missing: tuple[str, ...]
    workspace_host: str | None

    @property
    def ready(self) -> bool:
        return self.enabled and not self.missing

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "warehouse_id": self.warehouse_id,
            "missing": list(self.missing),
            "workspace_host": self.workspace_host,
            "network_check": "not performed (diagnostic only)",
        }

    def report(self) -> str:
        message = json.dumps(self.as_dict(), sort_keys=True)
        print(message)
        return message


def _enabled(environ: Mapping[str, str]) -> bool:
    return environ.get(UC_TRACING_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def preflight_uc_trace_destination(*, environ: Mapping[str, str] | None = None) -> UCTracePreflight:
    """Report UC destination readiness without contacting Databricks or creating tables."""
    env = os.environ if environ is None else environ
    enabled = _enabled(env)
    catalog_name = env.get(UC_CATALOG_ENV, DEFAULT_UC_CATALOG).strip() or None
    schema_name = env.get(UC_SCHEMA_ENV, DEFAULT_UC_SCHEMA).strip() or None
    warehouse_id = env.get(UC_WAREHOUSE_ENV, "").strip() or None
    missing = []
    if enabled and not catalog_name:
        missing.append(UC_CATALOG_ENV)
    if enabled and not schema_name:
        missing.append(UC_SCHEMA_ENV)
    if enabled and not warehouse_id:
        missing.append(UC_WAREHOUSE_ENV)
    return UCTracePreflight(
        enabled=enabled,
        catalog_name=catalog_name,
        schema_name=schema_name,
        warehouse_id=warehouse_id,
        missing=tuple(missing),
        workspace_host=env.get("DATABRICKS_HOST", DEFAULT_DATABRICKS_HOST),
    )


def configure_uc_trace_destination(*, environ: Mapping[str, str] | None = None, mlflow_module=None):
    """Configure MLflow's OTel exporter destination when explicitly enabled.

    This is intentionally local/offline at configuration time: MLflow performs any
    workspace interaction when spans are exported, not while this function runs.
    """
    preflight = preflight_uc_trace_destination(environ=environ)
    if not preflight.enabled:
        return None
    if preflight.missing:
        missing = ", ".join(preflight.missing)
        raise RuntimeError(f"UC OTel tracing is enabled but required configuration is missing: {missing}")

    if mlflow_module is None:
        destination = importlib.import_module("mlflow.tracing.destination")
    else:
        destination = mlflow_module.tracing.destination
    location = destination.UCSchemaLocation(preflight.catalog_name, preflight.schema_name)
    destination.set_destination(location)
    return location




@dataclass(frozen=True)
class HarnessAnalysis:
    """A validated result plus operational measurements, never fluid evidence."""

    arm: str
    adapter_id: str
    model: str
    feedback: AgentFeedback
    prompt_sha256: str
    latency_ms: float
    runtime_metadata: Mapping[str, Any]
    trace_id: str | None

    def dataset_output(self) -> dict[str, Any]:
        """Return only judge-visible content; arm and runtime stay in record tags."""
        return {"analysis": self.feedback.as_dict()}


async def analyze_run_bundle(
    *,
    bundle: RunBundle,
    harness: FeedbackHarness,
    mlflow_module=None,
) -> HarnessAnalysis:
    """Run one arm under an AGENT root span and validate its exact JSON."""
    mlflow = mlflow_module or importlib.import_module("mlflow")
    prompt = render_feedback_prompt(bundle)
    digest = prompt_digest(prompt)
    evidence_kind = (
        "measured"
        if all(arm.evidence_kind == "measured" for arm in (bundle.candidate, *bundle.comparators))
        else "nonmeasured"
    )
    attributes = {
        "project": PROJECT_LABEL,
        "bundle_id": bundle.bundle_id,
        "group_id": bundle.group_id,
        "harness_arm": harness.arm,
        "harness_adapter_id": harness.adapter_id,
        "model": harness.model,
        "task_contract_version": bundle.task_contract_version,
        "evidence_digest": bundle.evidence_digest,
        "prompt_sha256": digest,
    }
    trace_tags = {
        f"{PROJECT_LABEL}.bundle_id": bundle.bundle_id,
        f"{PROJECT_LABEL}.group_id": bundle.group_id,
        f"{PROJECT_LABEL}.harness_arm": harness.arm,
        f"{PROJECT_LABEL}.harness_adapter_id": harness.adapter_id,
        f"{PROJECT_LABEL}.model": harness.model,
        f"{PROJECT_LABEL}.evidence_digest": bundle.evidence_digest,
        f"{PROJECT_LABEL}.evidence_kind": evidence_kind,
    }

    with mlflow.start_span(
        name="hydrogym_feedback_agent",
        span_type="AGENT",
        attributes=attributes,
    ) as root_span:
        mlflow.update_current_trace(
            tags=trace_tags,
            metadata={
                f"{PROJECT_LABEL}.task_contract_version": bundle.task_contract_version,
                f"{PROJECT_LABEL}.prompt_sha256": digest,
            },
            request_preview=f"Critique RunBundle {bundle.bundle_id}",
        )
        root_span.set_inputs(
            {
                "run_bundle": bundle.as_dict(),
                "deterministic_comparison_issues": list(bundle.comparison_issues()),
            }
        )

        started = time.perf_counter()
        with mlflow.start_span(name="harness_call", span_type="TASK") as harness_span:
            harness_span.set_inputs({"prompt": prompt})
            raw = await harness.generate(prompt)
            if raw.arm != harness.arm:
                raise HarnessExecutionError(
                    f"harness returned arm {raw.arm!r}, expected {harness.arm!r}"
                )
            if raw.model != harness.model:
                raise HarnessExecutionError(
                    f"harness returned model {raw.model!r}, expected {harness.model!r}"
                )
            if raw.adapter_id != harness.adapter_id:
                raise HarnessExecutionError(
                    f"harness returned adapter_id {raw.adapter_id!r}, expected {harness.adapter_id!r}"
                )
            harness_span.set_outputs(
                {
                    "response": raw.text,
                    "runtime_metadata": dict(raw.metadata),
                }
            )
        latency_ms = (time.perf_counter() - started) * 1_000.0

        with mlflow.start_span(name="contract_validation", span_type="GUARDRAIL") as validation_span:
            validation_span.set_inputs(
                {
                    "schema_version": AGENT_FEEDBACK_SCHEMA_VERSION,
                    "response": raw.text,
                }
            )
            feedback = parse_agent_feedback(raw.text)
            validate_feedback_identity(bundle, feedback)
            validation_span.set_outputs({"analysis": feedback.as_dict()})

        root_span.set_outputs({"analysis": feedback.as_dict()})
        trace_id = getattr(root_span, "trace_id", None)

    return HarnessAnalysis(
        arm=raw.arm,
        adapter_id=raw.adapter_id,
        model=raw.model,
        feedback=feedback,
        prompt_sha256=digest,
        latency_ms=latency_ms,
        runtime_metadata=raw.metadata,
        trace_id=trace_id,
    )


# Configure only when the operator opts in; importing this module remains a genuine no-op.
configure_uc_trace_destination()
