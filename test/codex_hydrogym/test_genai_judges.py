"""Judge portfolio contracts for direct Unity AI Gateway routing."""

from typing import Literal

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME
from codex_hydrogym.genai.judges import (
    CRITIC_QUALITY_JUDGE_INSTRUCTIONS,
    FLUID_REWARD_JUDGE_INSTRUCTIONS,
    make_critic_quality_judge,
    make_fluid_reward_judges,
)
from codex_hydrogym.genai.portfolio import ModelPortfolio


def _portfolio():
    return ModelPortfolio(
        student_model="system.ai.deepseek-flash",
        primary_judge_model="system.ai.claude-opus-5",
        audit_judge_models=(
            "system.ai.gpt-5-6-sol",
            "system.ai.kimi-k3",
            "system.ai.deepseek-pro",
        ),
        reflection_models=("system.ai.glm-5-2", "system.ai.gpt-5-6-sol"),
        small_task_model="system.ai.deepseek-flash",
    )


def test_primary_and_audit_judges_use_gateway_and_exact_alignment_name():
    calls = []

    def fake_make_judge(**kwargs):
        calls.append(kwargs)
        return kwargs

    judges = make_fluid_reward_judges(
        portfolio=_portfolio(),
        workspace_host="https://fevm.example.cloud.databricks.com",
        token="app-oauth",
        make_judge_fn=fake_make_judge,
    )

    assert len(judges) == 4
    assert calls[0]["name"] == FEEDBACK_ASSESSMENT_NAME
    assert calls[0]["model"] == "openai:/system.ai.claude-opus-5"
    assert all(call["base_url"].endswith("/ai-gateway/mlflow/v1") for call in calls)
    assert all(call["extra_headers"]["Authorization"] == "Bearer app-oauth" for call in calls)
    assert all(call["feedback_value_type"] is float for call in calls)
    assert all("inference_params" not in call for call in calls)
    assert "failed or missing physics gate caps the score at 2" in FLUID_REWARD_JUDGE_INSTRUCTIONS


def test_persistable_judge_never_serializes_bearer_token():
    calls = []
    make_fluid_reward_judges(
        portfolio=_portfolio(),
        workspace_host="https://fevm.example.cloud.databricks.com",
        token="short-lived-secret",
        make_judge_fn=lambda **kwargs: calls.append(kwargs) or kwargs,
        include_authorization_header=False,
    )

    assert all("Authorization" not in call["extra_headers"] for call in calls)


def test_composite_critic_judge_is_one_integer_scale_and_never_fluid_evidence():
    calls = []
    judge = make_critic_quality_judge(
        model="databricks:/critic-endpoint",
        make_judge_fn=lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    assert judge is calls[0]
    assert calls[0]["name"] == CRITIC_QUALITY_ASSESSMENT_NAME
    assert calls[0]["feedback_value_type"] == Literal[1, 2, 3, 4, 5]
    assert "physics diagnosis" in CRITIC_QUALITY_JUDGE_INSTRUCTIONS
    assert "as proof of fluid improvement" in CRITIC_QUALITY_JUDGE_INSTRUCTIONS
    assert "{{ inputs }}" in calls[0]["instructions"]
    assert "{{ outputs }}" in calls[0]["instructions"]
