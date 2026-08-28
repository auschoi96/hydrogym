"""Direct OpenAI-compatible Unity AI Gateway client for codex_hydrogym."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import os
from typing import Any
from urllib.parse import urlsplit

from codex_hydrogym import PROJECT_LABEL


def normalize_workspace_host(value: str) -> str:
    host = value.strip().rstrip("/")
    if "://" not in host:
        host = f"https://{host}"
    parsed = urlsplit(host)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"}:
        raise ValueError("DATABRICKS_HOST must be an HTTPS workspace origin without a path")
    return f"https://{parsed.netloc}"


def resolve_databricks_token(
    environ: Mapping[str, str] | None = None,
    *,
    config_factory: Callable[[], Any] | None = None,
) -> str:
    """Resolve a PAT locally or an OAuth token from Databricks App SP auth."""
    env = os.environ if environ is None else environ
    if token := env.get("DATABRICKS_TOKEN"):
        return token

    if config_factory is None:
        config_module = importlib.import_module("databricks.sdk.core")
        config_factory = config_module.Config
    authorization = config_factory().authenticate().get("Authorization", "")
    prefix = "Bearer "
    if not authorization.startswith(prefix) or len(authorization) == len(prefix):
        raise RuntimeError("Databricks authentication did not produce a bearer token")
    return authorization[len(prefix) :]


@dataclass(frozen=True)
class GatewayResponse:
    text: str
    model: str
    request_id: str | None
    finish_reason: str | None
    usage: dict[str, Any]


class UnityAIGatewayClient:
    """Thin direct client for the provider-agnostic MLflow Gateway surface."""

    def __init__(
        self,
        *,
        workspace_host: str,
        token: str,
        openai_client=None,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.workspace_host = normalize_workspace_host(workspace_host)
        self.base_url = f"{self.workspace_host}/ai-gateway/mlflow/v1"
        if not token:
            raise ValueError("a Databricks bearer token is required")
        if not 0.0 < timeout_seconds < 120.0:
            raise ValueError("timeout_seconds must be positive and below the Databricks Apps proxy timeout")
        if openai_client is None:
            openai_module = importlib.import_module("openai")
            openai_client = openai_module.OpenAI(
                api_key=token,
                base_url=self.base_url,
                timeout=timeout_seconds,
            )
        self._client = openai_client

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        config_factory: Callable[[], Any] | None = None,
        openai_client=None,
    ) -> "UnityAIGatewayClient":
        env = os.environ if environ is None else environ
        host = env.get("DATABRICKS_HOST")
        if not host:
            raise ValueError("DATABRICKS_HOST is required")
        return cls(
            workspace_host=host,
            token=resolve_databricks_token(env, config_factory=config_factory),
            openai_client=openai_client,
        )

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int = 1024,
        temperature: float | None = None,
        request_tags: Mapping[str, str] | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> GatewayResponse:
        tags = {
            "project": PROJECT_LABEL,
            "component": "fluid_rl_outer_loop",
            **({str(key): str(value) for key, value in request_tags.items()} if request_tags else {}),
        }
        request: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "extra_headers": {"Databricks-Ai-Gateway-Request-Tags": json.dumps(tags, sort_keys=True)},
        }
        if temperature is not None:
            request["temperature"] = temperature
        if response_format is not None:
            if not isinstance(response_format, Mapping) or not response_format:
                raise ValueError("response_format must be a non-empty mapping")
            request["response_format"] = dict(response_format)
        response = self._client.chat.completions.create(
            **request,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage.model_dump() if response.usage is not None else {}
        return GatewayResponse(
            text=text,
            model=getattr(response, "model", model),
            request_id=getattr(response, "id", None),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
        )
