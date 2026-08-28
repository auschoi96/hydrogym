"""Validated model-role portfolio for the codex_hydrogym outer loop."""

from dataclasses import dataclass
import os
import re
from typing import Mapping


_MODEL_SERVICE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+$")


def _model_service(value: str, field: str) -> str:
    value = value.strip()
    if not value or _MODEL_SERVICE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a fully qualified Unity AI Gateway model service")
    return value


def _model_list(value: str, field: str, minimum: int) -> tuple[str, ...]:
    models = tuple(_model_service(item, field) for item in value.split(",") if item.strip())
    if len(models) < minimum:
        raise ValueError(f"{field} must contain at least {minimum} comma-separated model services")
    if len(set(models)) != len(models):
        raise ValueError(f"{field} must not contain duplicate model services")
    return models


@dataclass(frozen=True)
class ModelPortfolio:
    """Separate student, judge, reflection, and utility roles."""

    student_model: str
    primary_judge_model: str
    audit_judge_models: tuple[str, ...]
    reflection_models: tuple[str, ...]
    small_task_model: str
    embedding_model: str = "databricks:/databricks-gte-large-en"

    def __post_init__(self) -> None:
        object.__setattr__(self, "student_model", _model_service(self.student_model, "student_model"))
        object.__setattr__(
            self,
            "primary_judge_model",
            _model_service(self.primary_judge_model, "primary_judge_model"),
        )
        object.__setattr__(
            self,
            "audit_judge_models",
            _model_list(",".join(self.audit_judge_models), "audit_judge_models", 3),
        )
        object.__setattr__(
            self,
            "reflection_models",
            _model_list(",".join(self.reflection_models), "reflection_models", 2),
        )
        object.__setattr__(self, "small_task_model", _model_service(self.small_task_model, "small_task_model"))
        if not self.embedding_model.startswith("databricks:/"):
            raise ValueError("embedding_model must use the MLflow databricks:/ model URI format")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ModelPortfolio":
        env = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = env.get(name)
            if not value:
                raise ValueError(f"required model portfolio variable is missing: {name}")
            return value

        return cls(
            student_model=required("CODEX_HYDROGYM_STUDENT_MODEL"),
            primary_judge_model=required("CODEX_HYDROGYM_PRIMARY_JUDGE_MODEL"),
            audit_judge_models=_model_list(
                required("CODEX_HYDROGYM_AUDIT_JUDGE_MODELS"),
                "audit_judge_models",
                3,
            ),
            reflection_models=_model_list(
                required("CODEX_HYDROGYM_REFLECTION_MODELS"),
                "reflection_models",
                2,
            ),
            small_task_model=required("CODEX_HYDROGYM_SMALL_TASK_MODEL"),
            embedding_model=env.get(
                "CODEX_HYDROGYM_EMBEDDING_MODEL",
                "databricks:/databricks-gte-large-en",
            ),
        )

    def gateway_models(self) -> tuple[str, ...]:
        """Return every model service that must be permission-checked preflight."""
        return tuple(
            dict.fromkeys(
                (
                    self.student_model,
                    self.primary_judge_model,
                    *self.audit_judge_models,
                    *self.reflection_models,
                    self.small_task_model,
                )
            )
        )

    @staticmethod
    def mlflow_gateway_uri(model_service: str) -> str:
        """Address a Unity model service through MLflow's OpenAI adapter.

        The explicit OpenAI provider is intentional: callers supply the Unity AI
        Gateway ``base_url`` and bearer header, so judge/reflection traffic uses
        the same OpenAI-compatible surface as direct student calls.
        """
        return f"openai:/{_model_service(model_service, 'model_service')}"

    def as_tags(self) -> dict[str, str]:
        return {
            "codex_hydrogym.student_model": self.student_model,
            "codex_hydrogym.primary_judge_model": self.primary_judge_model,
            "codex_hydrogym.audit_judge_models": ",".join(self.audit_judge_models),
            "codex_hydrogym.reflection_models": ",".join(self.reflection_models),
            "codex_hydrogym.small_task_model": self.small_task_model,
            "codex_hydrogym.embedding_model": self.embedding_model,
        }
