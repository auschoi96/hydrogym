"""Judge-name preflight: predict MemAlign's assessment filter before it runs.

MemAlign's ``align(judge, traces)`` calls ``trace_to_dspy_example`` per trace
(mlflow/genai/judges/optimizers/dspy_utils.py:338), which keeps only assessments whose
sanitized name (lowercase, stripped) equals the sanitized judge name and whose source
type is HUMAN (dspy_utils.py:391-396).  Everything else is SILENTLY DROPPED, and align
then raises "No valid feedback records found in traces." when nothing survives
(optimizer.py:656-663).  This module mirrors that filter exactly and turns the late
opaque failure into an actionable precondition report: how many assessments would
survive for a given judge, and why each rejection happened (name mismatch vs non-HUMAN
source).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME
from codex_hydrogym.memalign_h1 import AUDIT_JUDGE_NAME, MEMALIGN_JUDGE_NAMES

PREFLIGHT_SCHEMA_VERSION = "codex_hydrogym.memalign_h1.preflight.v1"

OUTCOME_SURVIVES = "survives"
OUTCOME_NAME_MISMATCH = "rejected_name_mismatch"
OUTCOME_NON_HUMAN_SOURCE = "rejected_non_human_source"

_HUMAN_SOURCE_TOKEN = "HUMAN"


def _sanitize_assessment_name(name: Any) -> str:
    """Mirror mlflow dspy_utils._sanitize_assessment_name: lowercase, strip whitespace."""
    return str(name).lower().strip()


def classify_assessment(assessment: Any, *, judge_name: str) -> str:
    """Return the trace_to_dspy_example outcome for one assessment.

    Outcomes: ``survives``, ``rejected_name_mismatch``, or ``rejected_non_human_source``.
    Sanitized (lowercase, stripped) names are compared exactly as dspy_utils does, so a
    machine-written assessment whose name matches the judge is still rejected when its
    source is not HUMAN.
    """
    name = getattr(assessment, "name", None)
    source = getattr(assessment, "source", None)
    source_type = getattr(source, "source_type", None)
    if _sanitize_assessment_name(name) != _sanitize_assessment_name(judge_name):
        return OUTCOME_NAME_MISMATCH
    if str(source_type) != _HUMAN_SOURCE_TOKEN:
        return OUTCOME_NON_HUMAN_SOURCE
    return OUTCOME_SURVIVES


def _assessment_name(assessment: Any) -> str:
    name = getattr(assessment, "name", None)
    return "unknown" if name is None else str(name)


def _assessment_source_type(assessment: Any) -> str:
    source = getattr(assessment, "source", None)
    source_type = getattr(source, "source_type", None)
    return "<missing>" if source_type is None else str(source_type)


def _span_payload_hint(trace: Any) -> Mapping[str, bool]:
    """Informational payload presence mirroring dspy_utils extract_* helpers."""
    info = getattr(trace, "info", None)
    request = getattr(info, "request", None) if info is not None else None
    response = getattr(info, "response", None) if info is not None else None
    expectations = getattr(info, "expectations", None) if info is not None else None
    return {
        "has_request": request not in (None, ""),
        "has_response": response not in (None, ""),
        "has_expectations": expectations not in (None, (), []),
    }


def preflight_judge_labels(*, judge_name: str, traces: Sequence[Any]) -> dict[str, Any]:
    """Report exactly how many assessments would survive MemAlign's filter.

    Every assessment in every candidate trace is classified per assessment, and each
    trace is reported with its survivors and rejections plus the reason for each
    rejection.  ``memalign_would_fail`` is True exactly when zero assessments survive
    (the precondition behind optimizer.py's "No valid feedback records found in
    traces.").  The three repository judge names are the supported surface; labels
    written under any other name are reported as name mismatches so the collision
    between ``fluid_reward_plausibility`` (the labeling App), ``critic_quality`` (the
    adjudication pipeline and H1 target), and ``codex_hydrogym_revision_audit_v1`` (the
    model-scored outcome auditor) is visible before any alignment run.
    """
    if not isinstance(judge_name, str) or not judge_name.strip():
        raise ValueError("judge_name is required")
    judge_name = judge_name.strip()
    if judge_name not in MEMALIGN_JUDGE_NAMES:
        raise ValueError(
            f"judge_name must be one of {MEMALIGN_JUDGE_NAMES}; "
            "the name must match the label schema that humans actually fill in"
        )
    if not traces:
        raise ValueError("at least one candidate trace is required")

    sanitized_judge = _sanitize_assessment_name(judge_name)
    trace_rows: list[dict[str, Any]] = []
    surviving_assessments = 0
    traces_with_survivors = 0
    rejections: dict[str, int] = {OUTCOME_NAME_MISMATCH: 0, OUTCOME_NON_HUMAN_SOURCE: 0}
    human_assessments_total = 0
    for position, trace in enumerate(traces, start=1):
        info = getattr(trace, "info", None)
        trace_id = getattr(info, "trace_id", None)
        if trace_id is None:
            trace_id = f"trace_{position}"
        assessments = list(getattr(info, "assessments", ()) or ())
        classified: list[dict[str, Any]] = []
        survivors = 0
        for assessment in assessments:
            name = _assessment_name(assessment)
            source_type = _assessment_source_type(assessment)
            outcome = classify_assessment(assessment, judge_name=judge_name)
            if outcome == OUTCOME_SURVIVES:
                survivors += 1
            else:
                rejections[outcome] += 1
            if source_type == _HUMAN_SOURCE_TOKEN:
                human_assessments_total += 1
            classified.append(
                {
                    "name": name,
                    "sanitized_name": _sanitize_assessment_name(name),
                    "source_type": source_type,
                    "outcome": outcome,
                    "reason": (
                        "name does not match the sanitized judge name"
                        if outcome == OUTCOME_NAME_MISMATCH
                        else "source is not HUMAN"
                        if outcome == OUTCOME_NON_HUMAN_SOURCE
                        else "passes the name and HUMAN-source filter"
                    ),
                }
            )
        surviving_assessments += survivors
        if survivors:
            traces_with_survivors += 1
        trace_rows.append(
            {
                "trace_id": str(trace_id),
                "assessments": classified,
                "surviving_assessments": survivors,
                "span_payload": _span_payload_hint(trace),
                "would_produce_example": survivors > 0 and any(hint for hint in _span_payload_hint(trace).values()),
            }
        )

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "judge_name": judge_name,
        "sanitized_judge_name": sanitized_judge,
        "supported_judge_names": list(MEMALIGN_JUDGE_NAMES),
        "totals": {
            "traces": len(traces),
            "assessments": sum(len(row["assessments"]) for row in trace_rows),
            "surviving_assessments": surviving_assessments,
            "traces_with_survivors": traces_with_survivors,
            "human_assessments_total": human_assessments_total,
            "rejected_name_mismatch": rejections[OUTCOME_NAME_MISMATCH],
            "rejected_non_human_source": rejections[OUTCOME_NON_HUMAN_SOURCE],
        },
        "memalign_would_fail": surviving_assessments == 0,
        "trace_rows": trace_rows,
    }


def preflight_all_known_judges(*, traces: Sequence[Any]) -> dict[str, Any]:
    """Run the preflight for every repository judge name and expose the collision."""
    judges: dict[str, dict[str, Any]] = {}
    for name in MEMALIGN_JUDGE_NAMES:
        report = preflight_judge_labels(judge_name=name, traces=traces)
        judges[name] = report["totals"] | {"memalign_would_fail": report["memalign_would_fail"]}
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "judges": judges,
        "collision_summary": (
            "labels survive only under the judge name whose exact schema the labeling "
            "session used; the App writes "
            f"{FEEDBACK_ASSESSMENT_NAME!r}, the adjudication pipeline targets "
            f"{CRITIC_QUALITY_ASSESSMENT_NAME!r}, and the outcome auditor "
            f"{AUDIT_JUDGE_NAME!r} is model-scored (LLM_JUDGE) until a human labels it."
        ),
    }
