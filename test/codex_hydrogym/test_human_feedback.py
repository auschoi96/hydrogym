"""Human assessment capture for MemAlign."""

from types import SimpleNamespace

import pytest

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME
from codex_hydrogym.genai.feedback import (
    create_critic_quality_label_schema,
    create_critic_quality_labeling_session,
    create_feedback_label_schema,
    enroll_critic_quality_trace,
    submit_adjudicated_critic_quality,
    submit_human_feedback,
)


class _Schemas:
    class InputNumeric:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def __init__(self):
        self.kwargs = None

    def create_label_schema(self, **kwargs):
        self.kwargs = kwargs
        return kwargs


class _FakeMlflow:
    def __init__(self, trace):
        self.trace = trace
        self.feedback_kwargs = None
        self.tag_kwargs = None
        self.tag_calls = []

    def get_trace(self, trace_id):
        return self.trace if trace_id == "trace-1" else None

    def log_feedback(self, **kwargs):
        self.feedback_kwargs = kwargs
        return SimpleNamespace(assessment_id="assessment-1")

    def set_trace_tag(self, **kwargs):
        self.tag_kwargs = kwargs
        self.tag_calls.append(kwargs)


class _LabelingSession:
    def __init__(self):
        self.dataset_name = None

    def add_dataset(self, *, dataset_name):
        self.dataset_name = dataset_name
        return self


class _GenAI:
    def __init__(self):
        self.kwargs = None
        self.session = _LabelingSession()

    def create_labeling_session(self, **kwargs):
        self.kwargs = kwargs
        return self.session


def test_label_schema_matches_judge_name_and_accepts_comments():
    schemas = _Schemas()
    create_feedback_label_schema(label_schemas_module=schemas)

    assert schemas.kwargs["name"] == FEEDBACK_ASSESSMENT_NAME
    assert schemas.kwargs["enable_comment"] is True
    assert schemas.kwargs["input"].kwargs == {"min_value": 1.0, "max_value": 5.0}


def test_human_feedback_is_attributable_labeled_and_tagged():
    mlflow = _FakeMlflow(SimpleNamespace(info=SimpleNamespace(assessments=[])))

    assessment = submit_human_feedback(
        trace_id="trace-1",
        score=4,
        rationale="Plausible, but the control budget needs a stronger justification.",
        reviewer="expert@example.com",
        mlflow_module=mlflow,
    )

    assert assessment.assessment_id == "assessment-1"
    assert mlflow.feedback_kwargs["name"] == FEEDBACK_ASSESSMENT_NAME
    assert mlflow.feedback_kwargs["source"].source_type == "HUMAN"
    assert mlflow.feedback_kwargs["source"].source_id == "expert@example.com"
    assert mlflow.feedback_kwargs["metadata"]["project"] == "codex_hydrogym"
    assert mlflow.tag_kwargs["value"] == "human_labeled"


def test_duplicate_reviewer_feedback_is_rejected():
    existing = SimpleNamespace(
        name=FEEDBACK_ASSESSMENT_NAME,
        source=SimpleNamespace(source_type="HUMAN", source_id="expert@example.com"),
    )
    mlflow = _FakeMlflow(SimpleNamespace(info=SimpleNamespace(assessments=[existing])))

    with pytest.raises(ValueError, match="already submitted"):
        submit_human_feedback(
            trace_id="trace-1",
            score=3,
            rationale="Second attempt",
            reviewer="expert@example.com",
            mlflow_module=mlflow,
        )


def test_critic_quality_schema_and_single_consensus_label():
    schemas = _Schemas()
    create_critic_quality_label_schema(label_schemas_module=schemas)
    assert schemas.kwargs["name"] == CRITIC_QUALITY_ASSESSMENT_NAME

    mlflow = _FakeMlflow(SimpleNamespace(info=SimpleNamespace(assessments=[])))
    assessment = submit_adjudicated_critic_quality(
        trace_id="trace-1",
        score=4,
        rationale="The critique catches the confound and proposes the cheapest falsification.",
        adjudicator="panel@example.com",
        mlflow_module=mlflow,
    )
    assert assessment.assessment_id == "assessment-1"
    assert mlflow.feedback_kwargs["name"] == CRITIC_QUALITY_ASSESSMENT_NAME
    assert mlflow.feedback_kwargs["value"] == 4
    assert mlflow.feedback_kwargs["metadata"]["label_role"] == "consensus"

    existing = SimpleNamespace(
        name=CRITIC_QUALITY_ASSESSMENT_NAME,
        source=SimpleNamespace(source_type="HUMAN", source_id="first-panel@example.com"),
    )
    duplicate = _FakeMlflow(SimpleNamespace(info=SimpleNamespace(assessments=[existing])))
    with pytest.raises(ValueError, match="one adjudicated"):
        submit_adjudicated_critic_quality(
            trace_id="trace-1",
            score=5,
            rationale="A second panel cannot overwrite the consensus label.",
            adjudicator="second-panel@example.com",
            mlflow_module=duplicate,
        )


def test_critic_quality_labeling_session_uses_exact_memalign_target():
    schemas = _Schemas()
    genai = _GenAI()

    session = create_critic_quality_labeling_session(
        session_name="codex_hydrogym_critic_quality_pilot",
        dataset_name="catalog.schema.critic_quality_pilot",
        assigned_users=["expert@example.com"],
        mlflow_genai_module=genai,
        label_schemas_module=schemas,
    )

    assert schemas.kwargs["name"] == CRITIC_QUALITY_ASSESSMENT_NAME
    assert genai.kwargs["label_schemas"] == [CRITIC_QUALITY_ASSESSMENT_NAME]
    assert genai.kwargs["assigned_users"] == ["expert@example.com"]
    assert session.dataset_name == "catalog.schema.critic_quality_pilot"


def test_only_measured_trace_with_locked_provenance_enters_critic_review_queue():
    tags = {
        "codex_hydrogym.bundle_id": "bundle-1",
        "codex_hydrogym.group_id": "group-1",
        "codex_hydrogym.harness_arm": "codex",
        "codex_hydrogym.evidence_kind": "measured",
        "codex_hydrogym.evidence_digest": "a" * 64,
    }
    mlflow = _FakeMlflow(SimpleNamespace(info=SimpleNamespace(assessments=[], tags=tags)))

    enroll_critic_quality_trace(trace_id="trace-1", fold="train", mlflow_module=mlflow)

    assert mlflow.tag_calls == [
        {
            "trace_id": "trace-1",
            "key": "codex_hydrogym.critic_fold",
            "value": "train",
        },
        {
            "trace_id": "trace-1",
            "key": "codex_hydrogym.critic_review_state",
            "value": "pending_adjudication",
        },
    ]

    tags["codex_hydrogym.evidence_kind"] = "synthetic_contract"
    synthetic = _FakeMlflow(SimpleNamespace(info=SimpleNamespace(assessments=[], tags=tags)))
    with pytest.raises(ValueError, match="only measured"):
        enroll_critic_quality_trace(trace_id="trace-1", fold="train", mlflow_module=synthetic)
