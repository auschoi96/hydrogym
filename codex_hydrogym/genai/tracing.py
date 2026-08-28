"""Shared MLflow tracing for both coding-agent harness arms."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import time
from typing import Any, Mapping

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.genai.contracts import (
    AGENT_FEEDBACK_SCHEMA_VERSION,
    AgentFeedback,
    RunBundle,
    parse_agent_feedback,
)
from codex_hydrogym.genai.harnesses import (
    FeedbackHarness,
    HarnessExecutionError,
    prompt_digest,
    render_feedback_prompt,
    validate_feedback_identity,
)


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
