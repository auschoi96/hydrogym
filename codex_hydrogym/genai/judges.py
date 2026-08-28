"""MLflow judges for bounded fluid-reward proposals and rollout evidence."""

from __future__ import annotations

import importlib
import json
from typing import Any, Iterable, Literal, Mapping

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME, PROJECT_LABEL
from codex_hydrogym.genai.gateway import normalize_workspace_host
from codex_hydrogym.genai.portfolio import ModelPortfolio


FLUID_REWARD_JUDGE_INSTRUCTIONS = """
You are assessing {{ outputs }} for the exact criterion fluid_reward_plausibility in the context of {{ inputs }}.
Use a numeric 1-to-5 score and explain the decisive evidence.

1 = unsafe, untestable, physically incoherent, outside the published bounds, or presented as proven without a
held-out rollout.
2 = bounded but weakly motivated, internally inconsistent, or likely to trade lower TKE for excessive control.
3 = physically plausible and testable, but missing an important trade-off, uncertainty, or comparison.
4 = strong bounded hypothesis with a clear expected TKE/control trade-off and an appropriate validation plan.
5 = exceptionally clear, conservative, falsifiable, and supported by comparable real rollout evidence whose
deterministic physics gates all passed.

Hard rules:
- Never infer that an LLM score proves CFD performance.
- A failed or missing physics gate caps the score at 2.
- Arbitrary source code, solver mutation, or an out-of-contract parameter caps the score at 1.
- Prefer held-out mean-TKE improvement only when control effort remains proportionate.
""".strip()


CRITIC_QUALITY_JUDGE_INSTRUCTIONS = """
Evaluate the quality of the experiment critique in {{ outputs }} using only the RunBundle in {{ inputs }}.
Do not judge whether the fluid controller itself is good. Do not treat an agent recommendation, this score,
MemAlign agreement, or any other language-model output as proof of fluid improvement. Ignore the opaque case_id.

Score exactly one integer from 1 through 5:
1 = The critique invents evidence, misses a decisive confound, recommends unsafe/unbounded work, or claims fluid
    improvement without comparable held-out metrics.
2 = It notices some concerns but has a major physics, statistics, provenance, or claim-discipline failure.
3 = It is directionally useful and bounded but misses an important comparator, falsification, or cost detail.
4 = It accurately diagnoses the physics and statistical evidence, cites reproducible bundle fields/artifacts,
    proposes the cheapest decisive next check, and maintains the claim boundary.
5 = It does all of level 4 exceptionally well, explicitly distinguishes feedback quality from fluid performance,
    and identifies the most decision-relevant falsification or stopping condition.

Apply all five dimensions together: physics diagnosis, statistical validity, reproducibility/provenance, cost
awareness, and claim discipline. A failed deterministic gate or unresolved comparison issue that the critique
ignores caps the score at 2. Missing or fabricated evidence caps the score at 1. Return only the integer score
and a concise rationale grounded in {{ inputs }} and {{ outputs }}.
""".strip()


def make_critic_quality_judge(
    *,
    model: str,
    make_judge_fn=None,
    base_url: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
):
    """Create the single 1-5 judge that MemAlign may later align."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be an explicit non-empty MLflow model URI")
    if make_judge_fn is None:
        make_judge_fn = importlib.import_module("mlflow.genai").make_judge
    kwargs: dict[str, Any] = {
        "name": CRITIC_QUALITY_ASSESSMENT_NAME,
        "instructions": CRITIC_QUALITY_JUDGE_INSTRUCTIONS,
        "feedback_value_type": Literal[1, 2, 3, 4, 5],
        "model": model.strip(),
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    if extra_headers is not None:
        kwargs["extra_headers"] = dict(extra_headers)
    return make_judge_fn(**kwargs)


def _gateway_base_url(workspace_host: str) -> str:
    return f"{normalize_workspace_host(workspace_host)}/ai-gateway/mlflow/v1"


def _gateway_headers(token: str, *, role: str) -> dict[str, str]:
    if not token:
        raise ValueError("a Databricks bearer token is required")
    return {
        "Authorization": f"Bearer {token}",
        "Databricks-Ai-Gateway-Request-Tags": json.dumps(
            {"project": PROJECT_LABEL, "component": "fluid_rl_outer_loop", "role": role},
            sort_keys=True,
        ),
    }


def make_fluid_reward_judges(
    *,
    portfolio: ModelPortfolio,
    workspace_host: str,
    token: str,
    make_judge_fn=None,
    include_authorization_header: bool = True,
) -> tuple[Any, ...]:
    """Create one alignable primary judge plus independent audit judges."""
    if make_judge_fn is None:
        make_judge_fn = importlib.import_module("mlflow.genai.judges").make_judge
    base_url = _gateway_base_url(workspace_host)

    def make(name: str, model: str, role: str):
        headers = _gateway_headers(token, role=role)
        if not include_authorization_header:
            headers.pop("Authorization")
        return make_judge_fn(
            name=name,
            instructions=FLUID_REWARD_JUDGE_INSTRUCTIONS,
            feedback_value_type=float,
            model=portfolio.mlflow_gateway_uri(model),
            base_url=base_url,
            extra_headers=headers,
        )

    primary = make(FEEDBACK_ASSESSMENT_NAME, portfolio.primary_judge_model, "primary_judge")
    audits = tuple(
        make(f"{PROJECT_LABEL}_audit_judge_{index}", model, f"audit_judge_{index}")
        for index, model in enumerate(portfolio.audit_judge_models, start=1)
    )
    return (primary, *audits)


def evaluate_reward_proposals(
    *,
    records: Iterable[Mapping[str, Any]],
    portfolio: ModelPortfolio,
    workspace_host: str,
    token: str,
    mlflow_module=None,
):
    """Score proposal/rollout records and tag successful traces for humans."""
    mlflow = mlflow_module or importlib.import_module("mlflow")
    judges = make_fluid_reward_judges(
        portfolio=portfolio,
        workspace_host=workspace_host,
        token=token,
    )
    result = mlflow.genai.evaluate(data=list(records), scorers=list(judges))
    result_df = result.result_df
    if "state" in result_df and "trace_id" in result_df:
        ok_trace_ids = result_df.loc[result_df["state"] == "OK", "trace_id"]
        for trace_id in ok_trace_ids:
            mlflow.set_trace_tag(trace_id=trace_id, key=f"{PROJECT_LABEL}.review_state", value="pending_human")
    return result
