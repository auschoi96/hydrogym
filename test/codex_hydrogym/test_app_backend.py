"""Databricks App backend contracts without workspace access."""

from types import SimpleNamespace

import pytest

from codex_hydrogym.app.backend import (
    AppModelPortfolio,
    FEEDBACK_NAME,
    UnityGateway,
    reviewer_identity,
    search_review_traces,
    submit_human_feedback,
)


def _trace(trace_id="trace-1", state="pending_human", timestamp=1, assessments=None):
    return SimpleNamespace(
        info=SimpleNamespace(
            trace_id=trace_id,
            request_id=trace_id,
            timestamp_ms=timestamp,
            tags={"codex_hydrogym.review_state": state},
            assessments=assessments or [],
        ),
        data=SimpleNamespace(request={"scenario": trace_id}, response={"candidate": trace_id}),
    )


def test_reviewer_identity_prefers_forwarded_user_and_requires_explicit_local_fallback():
    assert reviewer_identity({"X-Forwarded-Email": "expert@example.com"}, {}) == "expert@example.com"
    assert reviewer_identity({}, {"CODEX_HYDROGYM_REVIEWER": "local@example.com"}) == "local@example.com"
    assert reviewer_identity({}, {}) is None


def test_model_portfolio_refuses_unverified_placeholders():
    with pytest.raises(ValueError, match="not configured"):
        AppModelPortfolio.from_env({"CODEX_HYDROGYM_STUDENT_MODEL": "__CONFIGURE_BEFORE_DEPLOY__"})


def test_review_trace_search_uses_two_supported_filters_and_deduplicates():
    calls = []

    def search_traces(**kwargs):
        calls.append(kwargs)
        if "pending_human" in kwargs["filter_string"]:
            return [_trace(timestamp=10)]
        return [_trace(state="human_labeled", timestamp=20), _trace("trace-2", "human_labeled", 15)]

    mlflow = SimpleNamespace(search_traces=search_traces)
    traces = search_review_traces(experiment_id="123", mlflow_module=mlflow)

    assert len(calls) == 2
    assert [trace.trace_id for trace in traces] == ["trace-1", "trace-2"]
    assert traces[0].review_state == "human_labeled"


class _FakeMlflow:
    def __init__(self, trace):
        self.trace = trace
        self.feedback = None
        self.tag = None

    def get_trace(self, trace_id):
        return self.trace

    def log_feedback(self, **kwargs):
        self.feedback = kwargs
        return SimpleNamespace(assessment_id="assessment-1")

    def set_trace_tag(self, **kwargs):
        self.tag = kwargs


def test_app_feedback_is_an_attributable_mlflow_assessment():
    mlflow = _FakeMlflow(_trace())
    submit_human_feedback(
        trace_id="trace-1",
        score=5,
        rationale="Bounded, falsifiable, and appropriately conservative.",
        reviewer="expert@example.com",
        mlflow_module=mlflow,
    )

    assert mlflow.feedback["name"] == FEEDBACK_NAME
    assert mlflow.feedback["source"].source_type == "HUMAN"
    assert mlflow.feedback["source"].source_id == "expert@example.com"
    assert mlflow.tag["value"] == "human_labeled"


class _Completions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="gateway-request",
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="critique"))],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 12}),
        )


def test_app_model_lab_calls_unity_mlflow_gateway_surface():
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    gateway = UnityGateway(
        workspace_host="https://fevm.example.cloud.databricks.com",
        token="oauth",
        openai_client=client,
    )
    response = gateway.chat(model="system.ai.example", prompt="Review this candidate")

    assert gateway.base_url.endswith("/ai-gateway/mlflow/v1")
    assert response["text"] == "critique"
    assert completions.kwargs["model"] == "system.ai.example"
    assert "temperature" not in completions.kwargs
