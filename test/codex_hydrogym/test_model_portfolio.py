"""Model portfolio and direct Unity AI Gateway contracts."""

import json
from types import SimpleNamespace

import pytest

from codex_hydrogym.genai.gateway import UnityAIGatewayClient, normalize_workspace_host, resolve_databricks_token
from codex_hydrogym.genai.portfolio import ModelPortfolio


def _portfolio_env() -> dict[str, str]:
    return {
        "CODEX_HYDROGYM_STUDENT_MODEL": "system.ai.deepseek-flash",
        "CODEX_HYDROGYM_PRIMARY_JUDGE_MODEL": "system.ai.claude-opus-5",
        "CODEX_HYDROGYM_AUDIT_JUDGE_MODELS": ("system.ai.gpt-5-6-sol,system.ai.kimi-k3,system.ai.deepseek-pro"),
        "CODEX_HYDROGYM_REFLECTION_MODELS": "system.ai.glm-5-2,system.ai.gpt-5-6-sol",
        "CODEX_HYDROGYM_SMALL_TASK_MODEL": "system.ai.deepseek-flash",
    }


def test_portfolio_requires_distinct_explicit_model_services_by_role():
    portfolio = ModelPortfolio.from_env(_portfolio_env())

    assert portfolio.student_model == "system.ai.deepseek-flash"
    assert len(portfolio.audit_judge_models) == 3
    assert len(portfolio.reflection_models) == 2
    assert portfolio.embedding_model == "databricks:/databricks-gte-large-en"
    assert len(portfolio.gateway_models()) == 6
    assert portfolio.mlflow_gateway_uri(portfolio.primary_judge_model) == "openai:/system.ai.claude-opus-5"
    assert all(key.startswith("codex_hydrogym.") for key in portfolio.as_tags())


def test_portfolio_does_not_guess_missing_workspace_model_ids():
    env = _portfolio_env()
    del env["CODEX_HYDROGYM_PRIMARY_JUDGE_MODEL"]

    with pytest.raises(ValueError, match="PRIMARY_JUDGE"):
        ModelPortfolio.from_env(env)


def test_workspace_host_and_app_oauth_token_are_normalized():
    assert normalize_workspace_host("fevm.example.cloud.databricks.com/") == (
        "https://fevm.example.cloud.databricks.com"
    )
    token = resolve_databricks_token(
        {},
        config_factory=lambda: SimpleNamespace(authenticate=lambda: {"Authorization": "Bearer app-oauth-token"}),
    )
    assert token == "app-oauth-token"

    with pytest.raises(ValueError, match="without a path"):
        normalize_workspace_host("https://example.cloud.databricks.com/workspace")


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="request-1",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="bounded candidate"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 2}),
        )


def test_gateway_calls_mlflow_surface_and_tags_every_request():
    completions = _FakeCompletions()
    fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    gateway = UnityAIGatewayClient(
        workspace_host="https://fevm.example.cloud.databricks.com",
        token="secret",
        openai_client=fake_openai,
    )

    response = gateway.chat(
        model="system.ai.claude-opus-5",
        messages=[{"role": "user", "content": "Review this reward."}],
        request_tags={"role": "primary_judge"},
    )

    assert gateway.base_url == "https://fevm.example.cloud.databricks.com/ai-gateway/mlflow/v1"
    assert response.text == "bounded candidate"
    assert response.finish_reason == "stop"
    tags = json.loads(completions.kwargs["extra_headers"]["Databricks-Ai-Gateway-Request-Tags"])
    assert tags == {
        "component": "fluid_rl_outer_loop",
        "project": "codex_hydrogym",
        "role": "primary_judge",
    }
    assert "temperature" not in completions.kwargs
    assert "response_format" not in completions.kwargs

    gateway.chat(
        model="system.ai.kimi-k3",
        messages=[{"role": "user", "content": "Propose a bounded candidate."}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    assert completions.kwargs["temperature"] == 0.2
    assert completions.kwargs["response_format"] == {"type": "json_object"}

    with pytest.raises(ValueError, match="non-empty mapping"):
        gateway.chat(
            model="system.ai.kimi-k3",
            messages=[{"role": "user", "content": "Propose a bounded candidate."}],
            response_format={},
        )
