"""Human-feedback capture and MLflow Review App setup."""

from __future__ import annotations

import importlib
import re
from typing import Any, Iterable

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME, PROJECT_LABEL


FEEDBACK_INSTRUCTION = """
Rate this bounded fluid-RL reward proposal from 1 to 5. Focus on physical plausibility, whether the hypothesis is
falsifiable, the expected mean-TKE versus control-effort trade-off, and whether the proposed validation preserves
all deterministic physics gates. A language-model score is advisory; only held-out PPO evidence can support
promotion. Add a comment explaining the most important reason for your score.
""".strip()

CRITIC_QUALITY_FEEDBACK_INSTRUCTION = """
Adjudicate the coding agent's experiment critique from 1 to 5. Score physics diagnosis, statistical validity,
reproducibility and provenance, cost awareness, and claim discipline together. This label measures critique
quality only; it is never evidence that a controller improved the fluid. Submit exactly one consensus label for
each trace and explain the decisive reason.
""".strip()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def create_feedback_label_schema(*, label_schemas_module=None):
    """Create the schema whose name must exactly match the alignable judge."""
    schemas = label_schemas_module or importlib.import_module("mlflow.genai.label_schemas")
    return schemas.create_label_schema(
        name=FEEDBACK_ASSESSMENT_NAME,
        type="feedback",
        title="Fluid reward plausibility (1-5)",
        input=schemas.InputNumeric(min_value=1.0, max_value=5.0),
        instruction=FEEDBACK_INSTRUCTION,
        enable_comment=True,
        overwrite=True,
    )


def create_critic_quality_label_schema(*, label_schemas_module=None):
    """Create the one adjudicated label schema consumed by MemAlign."""
    schemas = label_schemas_module or importlib.import_module("mlflow.genai.label_schemas")
    return schemas.create_label_schema(
        name=CRITIC_QUALITY_ASSESSMENT_NAME,
        type="feedback",
        title="Critic quality consensus (1-5)",
        input=schemas.InputNumeric(min_value=1.0, max_value=5.0),
        instruction=CRITIC_QUALITY_FEEDBACK_INSTRUCTION,
        enable_comment=True,
        overwrite=True,
    )


def create_human_labeling_session(
    *,
    session_name: str,
    dataset_name: str,
    assigned_users: Iterable[str],
    mlflow_genai_module=None,
):
    """Create a native MLflow labeling session and attach the evaluation dataset."""
    genai = mlflow_genai_module or importlib.import_module("mlflow.genai")
    users = [user.strip() for user in assigned_users if user.strip()]
    if not users:
        raise ValueError("at least one assigned human reviewer is required")
    if not session_name.startswith(PROJECT_LABEL):
        raise ValueError(f"session_name must start with {PROJECT_LABEL}")
    create_feedback_label_schema()
    session = genai.create_labeling_session(
        name=session_name,
        assigned_users=users,
        label_schemas=[FEEDBACK_ASSESSMENT_NAME],
    )
    return session.add_dataset(dataset_name=dataset_name)


def create_critic_quality_labeling_session(
    *,
    session_name: str,
    dataset_name: str,
    assigned_users: Iterable[str],
    mlflow_genai_module=None,
    label_schemas_module=None,
):
    """Create the labeling session whose exact target is ``critic_quality``."""
    genai = mlflow_genai_module or importlib.import_module("mlflow.genai")
    users = [user.strip() for user in assigned_users if user.strip()]
    if not users:
        raise ValueError("at least one assigned human reviewer is required")
    if not session_name.startswith(PROJECT_LABEL):
        raise ValueError(f"session_name must start with {PROJECT_LABEL}")
    create_critic_quality_label_schema(label_schemas_module=label_schemas_module)
    session = genai.create_labeling_session(
        name=session_name,
        assigned_users=users,
        label_schemas=[CRITIC_QUALITY_ASSESSMENT_NAME],
    )
    return session.add_dataset(dataset_name=dataset_name)


def _assessment_source(assessment: Any) -> tuple[str | None, str | None]:
    source = getattr(assessment, "source", None)
    return getattr(source, "source_type", None), getattr(source, "source_id", None)


def existing_reviewer_feedback(trace: Any, reviewer: str) -> Any | None:
    assessments = getattr(getattr(trace, "info", None), "assessments", ()) or ()
    for assessment in assessments:
        source_type, source_id = _assessment_source(assessment)
        if (
            getattr(assessment, "name", None) == FEEDBACK_ASSESSMENT_NAME
            and str(source_type) == "HUMAN"
            and source_id == reviewer
        ):
            return assessment
    return None


def matching_human_feedback(trace: Any, *, assessment_name: str) -> tuple[Any, ...]:
    """Return HUMAN feedback assessments with the exact unsanitized judge name."""
    assessments = getattr(getattr(trace, "info", None), "assessments", ()) or ()
    matches = []
    for assessment in assessments:
        source_type, _source_id = _assessment_source(assessment)
        if getattr(assessment, "name", None) == assessment_name and str(source_type) == "HUMAN":
            matches.append(assessment)
    return tuple(matches)


def enroll_critic_quality_trace(
    *,
    trace_id: str,
    fold: str,
    mlflow_module=None,
) -> None:
    """Expose one validated measured trace to the App's adjudication queue."""
    if not isinstance(trace_id, str) or not trace_id.strip():
        raise ValueError("trace_id is required")
    if fold not in {"train", "test"}:
        raise ValueError("critic_quality fold must be train or test")
    mlflow = mlflow_module or importlib.import_module("mlflow")
    trace = mlflow.get_trace(trace_id.strip())
    if trace is None:
        raise ValueError(f"trace does not exist: {trace_id}")
    info = getattr(trace, "info", None)
    tags = getattr(info, "tags", {}) or {}
    required_text = {
        f"{PROJECT_LABEL}.bundle_id": tags.get(f"{PROJECT_LABEL}.bundle_id"),
        f"{PROJECT_LABEL}.group_id": tags.get(f"{PROJECT_LABEL}.group_id"),
    }
    if any(not isinstance(value, str) or not value.strip() for value in required_text.values()):
        raise ValueError("critic_quality trace is missing bundle or group provenance")
    if tags.get(f"{PROJECT_LABEL}.harness_arm") not in {"codex", "claude"}:
        raise ValueError("critic_quality trace has no recognized harness arm")
    if tags.get(f"{PROJECT_LABEL}.evidence_kind") != "measured":
        raise ValueError("only measured evidence may enter critic_quality review")
    evidence_digest = tags.get(f"{PROJECT_LABEL}.evidence_digest")
    if not isinstance(evidence_digest, str) or _SHA256.fullmatch(evidence_digest) is None:
        raise ValueError("critic_quality trace has no valid evidence digest")
    if matching_human_feedback(trace, assessment_name=CRITIC_QUALITY_ASSESSMENT_NAME):
        raise ValueError("trace already has its one adjudicated critic_quality HUMAN label")

    mlflow.set_trace_tag(
        trace_id=trace_id.strip(),
        key=f"{PROJECT_LABEL}.critic_fold",
        value=fold,
    )
    mlflow.set_trace_tag(
        trace_id=trace_id.strip(),
        key=f"{PROJECT_LABEL}.critic_review_state",
        value="pending_adjudication",
    )


def submit_human_feedback(
    *,
    trace_id: str,
    score: float,
    rationale: str,
    reviewer: str,
    mlflow_module=None,
):
    """Persist one attributable human assessment for MemAlign."""
    mlflow = mlflow_module or importlib.import_module("mlflow")
    if not trace_id.strip():
        raise ValueError("trace_id is required")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1.0 <= float(score) <= 5.0:
        raise ValueError("score must be numeric and in [1, 5]")
    reviewer = reviewer.strip()
    rationale = " ".join(rationale.split())
    if not reviewer:
        raise ValueError("an attributable reviewer identity is required")
    if not rationale or len(rationale) > 4_000:
        raise ValueError("rationale must contain 1 to 4000 characters")

    trace = mlflow.get_trace(trace_id)
    if trace is None:
        raise ValueError(f"trace does not exist: {trace_id}")
    if existing_reviewer_feedback(trace, reviewer) is not None:
        raise ValueError("this reviewer already submitted fluid_reward_plausibility for the trace")

    entities = importlib.import_module("mlflow.entities")
    assessment = mlflow.log_feedback(
        trace_id=trace_id,
        name=FEEDBACK_ASSESSMENT_NAME,
        value=float(score),
        rationale=rationale,
        source=entities.AssessmentSource(
            source_type=entities.AssessmentSourceType.HUMAN,
            source_id=reviewer,
        ),
        metadata={
            "project": PROJECT_LABEL,
            "reviewer": reviewer,
            "feedback_contract": FEEDBACK_ASSESSMENT_NAME,
        },
    )
    mlflow.set_trace_tag(trace_id=trace_id, key=f"{PROJECT_LABEL}.review_state", value="human_labeled")
    return assessment


def submit_adjudicated_critic_quality(
    *,
    trace_id: str,
    score: int,
    rationale: str,
    adjudicator: str,
    mlflow_module=None,
):
    """Persist the sole HUMAN ``critic_quality`` label for one trace."""
    mlflow = mlflow_module or importlib.import_module("mlflow")
    if not isinstance(trace_id, str) or not trace_id.strip():
        raise ValueError("trace_id is required")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("score must be an integer in [1, 5]")
    if not isinstance(adjudicator, str) or not adjudicator.strip():
        raise ValueError("an attributable adjudicator identity is required")
    if not isinstance(rationale, str):
        raise TypeError("rationale must be text")
    normalized_rationale = " ".join(rationale.split())
    if not normalized_rationale or len(normalized_rationale) > 4_000:
        raise ValueError("rationale must contain 1 to 4000 characters")

    trace = mlflow.get_trace(trace_id)
    if trace is None:
        raise ValueError(f"trace does not exist: {trace_id}")
    existing = matching_human_feedback(trace, assessment_name=CRITIC_QUALITY_ASSESSMENT_NAME)
    if existing:
        raise ValueError("trace already has its one adjudicated critic_quality HUMAN label")

    entities = importlib.import_module("mlflow.entities")
    assessment = mlflow.log_feedback(
        trace_id=trace_id,
        name=CRITIC_QUALITY_ASSESSMENT_NAME,
        value=score,
        rationale=normalized_rationale,
        source=entities.AssessmentSource(
            source_type=entities.AssessmentSourceType.HUMAN,
            source_id=adjudicator.strip(),
        ),
        metadata={
            "project": PROJECT_LABEL,
            "adjudicator": adjudicator.strip(),
            "label_role": "consensus",
        },
    )
    mlflow.set_trace_tag(
        trace_id=trace_id,
        key=f"{PROJECT_LABEL}.critic_review_state",
        value="adjudicated",
    )
    return assessment
