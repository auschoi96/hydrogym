"""MemAlign and GEPA orchestration with hard physics promotion gates."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from typing import Any, MutableMapping

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME, PROJECT_LABEL
from codex_hydrogym.genai.contracts import (
    REWARD_CANDIDATE_JSON_SCHEMA,
    RolloutEvidence,
    parse_reward_candidate,
)
from codex_hydrogym.genai.datasets import gepa_scenario_records
from codex_hydrogym.genai.feedback import matching_human_feedback
from codex_hydrogym.genai.gateway import UnityAIGatewayClient, normalize_workspace_host
from codex_hydrogym.genai.judges import make_fluid_reward_judges
from codex_hydrogym.genai.portfolio import ModelPortfolio


STUDENT_PROMPT_NAME = "codex_hydrogym_reward_student"
STUDENT_PROMPT_TEMPLATE = """
You propose one bounded PPO reward/training candidate for a fluid-dynamics experiment.

Scenario:
{{scenario}}

Return only a JSON object conforming to codex_hydrogym.reward_candidate.v1. Do not return Markdown, Python,
formulas, shell commands, solver changes, or claims that the candidate is already validated. Use a candidate_id
beginning with codex_hydrogym_. The hypothesis must be physically falsifiable. The rationale must discuss both
held-out mean TKE and control effort. All proposals remain advisory until a real H100 PPO rollout passes every
deterministic physics gate.
""".strip()


@contextmanager
def unity_gateway_environment(
    *,
    workspace_host: str,
    token: str,
    environ: MutableMapping[str, str] | None = None,
) -> Iterator[str]:
    """Temporarily point LiteLLM/DSPy-based MLflow optimizers at Unity Gateway."""
    env = os.environ if environ is None else environ
    base_url = f"{normalize_workspace_host(workspace_host)}/ai-gateway/mlflow/v1"
    if not token:
        raise ValueError("a Databricks bearer token is required")
    updates = {
        "OPENAI_API_KEY": token,
        "OPENAI_API_BASE": base_url,
        "OPENAI_BASE_URL": base_url,
        "DATABRICKS_HOST": normalize_workspace_host(workspace_host),
        "DATABRICKS_TOKEN": token,
    }
    previous = {name: env.get(name) for name in updates}
    env.update(updates)
    try:
        yield base_url
    finally:
        for name, value in previous.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value


def register_base_judge(
    *,
    experiment_id: str,
    portfolio: ModelPortfolio,
    workspace_host: str,
    token: str,
):
    """Register the exact judge that human labels will later align."""
    primary = make_fluid_reward_judges(
        portfolio=portfolio,
        workspace_host=workspace_host,
        token=token,
        include_authorization_header=False,
    )[0]
    with unity_gateway_environment(workspace_host=workspace_host, token=token):
        return primary.register(experiment_id=experiment_id)


def _has_matching_human_feedback(trace: Any) -> bool:
    assessments = getattr(getattr(trace, "info", None), "assessments", ()) or ()
    for assessment in assessments:
        source = getattr(assessment, "source", None)
        if (
            getattr(assessment, "name", None) == FEEDBACK_ASSESSMENT_NAME
            and str(getattr(source, "source_type", None)) == "HUMAN"
        ):
            return True
    return False


def align_fluid_reward_judge(
    *,
    experiment_id: str,
    portfolio: ModelPortfolio,
    workspace_host: str,
    token: str,
    retrieval_k: int = 5,
    mlflow_module=None,
    get_scorer_fn=None,
    optimizer_factory=None,
):
    """Legacy alignment path that returns an inline judge without overwriting the base."""
    if not 1 <= retrieval_k <= 20:
        raise ValueError("retrieval_k must be in [1, 20]")
    mlflow = mlflow_module or importlib.import_module("mlflow")
    if get_scorer_fn is None:
        get_scorer_fn = importlib.import_module("mlflow.genai.scorers").get_scorer
    if optimizer_factory is None:
        optimizer_factory = importlib.import_module("mlflow.genai.judges.optimizers").MemAlignOptimizer

    traces = mlflow.search_traces(
        locations=[experiment_id],
        filter_string=f"tags.`{PROJECT_LABEL}.review_state` = 'human_labeled'",
        return_type="list",
        include_spans=True,
    )
    traces = [trace for trace in traces if _has_matching_human_feedback(trace)]
    if not traces:
        raise ValueError(f"no traces contain human {FEEDBACK_ASSESSMENT_NAME} assessments")

    base_judge = get_scorer_fn(name=FEEDBACK_ASSESSMENT_NAME, experiment_id=experiment_id)
    reflection_model = portfolio.mlflow_gateway_uri(portfolio.reflection_models[0])
    optimizer = optimizer_factory(
        reflection_lm=reflection_model,
        retrieval_k=retrieval_k,
        embedding_model=portfolio.embedding_model,
    )
    with unity_gateway_environment(workspace_host=workspace_host, token=token):
        return base_judge.align(traces=traces, optimizer=optimizer)


def align_critic_quality_judge(
    *,
    train_traces: Sequence[Any],
    train_bundle_ids: Sequence[str],
    heldout_bundle_ids: Sequence[str],
    base_judge: Any,
    reflection_lm: str,
    embedding_model: str,
    retrieval_k: int = 5,
    required_arms: Sequence[str] = ("codex", "claude"),
    optimizer_factory=None,
):
    """Align on a locked training fold for one fixed agent or a paired-agent study."""
    traces = list(train_traces)
    train_ids = set(train_bundle_ids)
    heldout_ids = set(heldout_bundle_ids)
    if not traces:
        raise ValueError("train_traces must not be empty")
    if not train_ids or not heldout_ids:
        raise ValueError("both train and held-out bundle manifests are required")
    if train_ids & heldout_ids:
        raise ValueError("train and held-out bundle manifests must be disjoint")
    if not 1 <= retrieval_k <= 20:
        raise ValueError("retrieval_k must be in [1, 20]")
    if getattr(base_judge, "name", CRITIC_QUALITY_ASSESSMENT_NAME) != CRITIC_QUALITY_ASSESSMENT_NAME:
        raise ValueError("base_judge must be named critic_quality")
    expected_arms = set(required_arms)
    if not expected_arms or not expected_arms <= {"codex", "claude"}:
        raise ValueError("required_arms must contain codex, claude, or both")

    seen_arms: dict[str, set[str]] = {}
    trace_counts: dict[str, int] = {}
    for trace in traces:
        info = getattr(trace, "info", None)
        tags = getattr(info, "tags", {}) or {}
        bundle_id = tags.get(f"{PROJECT_LABEL}.bundle_id")
        arm = tags.get(f"{PROJECT_LABEL}.harness_arm")
        if bundle_id not in train_ids:
            raise ValueError("alignment trace is outside the locked training bundle manifest")
        if bundle_id in heldout_ids:
            raise ValueError("held-out bundle leaked into alignment traces")
        if arm not in expected_arms:
            raise ValueError("alignment trace is missing a valid harness-arm tag")
        seen_arms.setdefault(bundle_id, set()).add(arm)
        trace_counts[bundle_id] = trace_counts.get(bundle_id, 0) + 1
        labels = matching_human_feedback(trace, assessment_name=CRITIC_QUALITY_ASSESSMENT_NAME)
        if len(labels) != 1:
            raise ValueError("each alignment trace must have exactly one adjudicated critic_quality label")
    if set(seen_arms) != train_ids:
        raise ValueError("alignment traces do not cover the full locked training bundle manifest")
    if any(arms != expected_arms for arms in seen_arms.values()):
        raise ValueError("each training bundle must contribute every required harness arm")
    if any(count != len(expected_arms) for count in trace_counts.values()):
        raise ValueError("each training bundle must contribute exactly one trace per required arm")

    if optimizer_factory is None:
        optimizer_factory = importlib.import_module("mlflow.genai.judges.optimizers").MemAlignOptimizer
    optimizer = optimizer_factory(
        reflection_lm=reflection_lm,
        retrieval_k=retrieval_k,
        embedding_model=embedding_model,
    )
    return base_judge.align(traces=traces, optimizer=optimizer)


def register_aligned_critic_quality_judge(*, aligned_judge: Any, experiment_id: str):
    """Persist MemAlign output as a new scorer version while preserving the base."""
    if getattr(aligned_judge, "name", None) != CRITIC_QUALITY_ASSESSMENT_NAME:
        raise ValueError("aligned_judge must be named critic_quality")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    register = getattr(aligned_judge, "register", None)
    if not callable(register):
        raise TypeError("aligned_judge does not support scorer registration")
    return register(experiment_id=experiment_id.strip())


def register_student_prompt(*, mlflow_module=None):
    """Register the bounded student prompt and its JSON response contract."""
    mlflow = mlflow_module or importlib.import_module("mlflow")
    return mlflow.genai.register_prompt(
        name=STUDENT_PROMPT_NAME,
        template=STUDENT_PROMPT_TEMPLATE,
        commit_message="codex_hydrogym bounded fluid reward student baseline",
        tags={"project": PROJECT_LABEL, "purpose": "fluid_reward_candidate"},
        response_format=REWARD_CANDIDATE_JSON_SCHEMA,
    )


def generate_reward_candidate(
    *,
    scenario: Mapping[str, Any],
    prompt_uri: str,
    portfolio: ModelPortfolio,
    gateway: UnityAIGatewayClient,
    mlflow_module=None,
):
    """Load the current prompt version, call the student, and validate its JSON."""
    mlflow = mlflow_module or importlib.import_module("mlflow")
    prompt = mlflow.genai.load_prompt(prompt_uri)
    content = prompt.format(scenario=json.dumps(dict(scenario), sort_keys=True))
    response = gateway.chat(
        model=portfolio.student_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1_200,
        temperature=0.2,
        request_tags={"role": "student", "prompt_uri": prompt_uri},
    )
    return parse_reward_candidate(response.text)


def normalized_judge_aggregation(scores: Mapping[str, Any]) -> float:
    """Map the aligned 1-5 judge feedback onto GEPA's 0-1 objective."""
    feedback = scores.get(FEEDBACK_ASSESSMENT_NAME)
    value = feedback
    if hasattr(value, "feedback"):
        value = value.feedback
    if hasattr(value, "value"):
        value = value.value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, (numeric - 1.0) / 4.0))


@dataclass(frozen=True)
class GepaCandidateResult:
    prompt_name: str
    prompt_version: int
    initial_score: float | None
    final_score: float | None
    promoted_alias: str = "candidate"


def run_gepa_student_optimization(
    *,
    experiment_id: str,
    prompt_uri: str,
    portfolio: ModelPortfolio,
    workspace_host: str,
    token: str,
    max_metric_calls: int = 75,
    train_data: Sequence[Mapping[str, Any]] | None = None,
    mlflow_module=None,
    optimizer_factory=None,
    get_scorer_fn=None,
) -> GepaCandidateResult:
    """Optimize the student prompt; never grant production without PPO gates."""
    if not 10 <= max_metric_calls <= 500:
        raise ValueError("max_metric_calls must be in [10, 500]")
    mlflow = mlflow_module or importlib.import_module("mlflow")
    if optimizer_factory is None:
        optimizer_factory = importlib.import_module("mlflow.genai.optimize.optimizers").GepaPromptOptimizer
    if get_scorer_fn is None:
        get_scorer_fn = importlib.import_module("mlflow.genai.scorers").get_scorer
    records = list(gepa_scenario_records() if train_data is None else train_data)
    if not records or any(set(record) != {"inputs", "expectations"} for record in records):
        raise ValueError("every GEPA record must contain exactly inputs and expectations")

    gateway = UnityAIGatewayClient(workspace_host=workspace_host, token=token)

    def predict_fn(scenario):
        return generate_reward_candidate(
            scenario=scenario,
            prompt_uri=prompt_uri,
            portfolio=portfolio,
            gateway=gateway,
            mlflow_module=mlflow,
        ).as_dict()

    aligned_judge = get_scorer_fn(name=FEEDBACK_ASSESSMENT_NAME, experiment_id=experiment_id)
    optimizer = optimizer_factory(
        reflection_model=portfolio.mlflow_gateway_uri(portfolio.reflection_models[1]),
        max_metric_calls=max_metric_calls,
        display_progress_bar=True,
    )
    with unity_gateway_environment(workspace_host=workspace_host, token=token):
        result = mlflow.genai.optimize_prompts(
            predict_fn=predict_fn,
            train_data=records,
            prompt_uris=[prompt_uri],
            optimizer=optimizer,
            scorers=[aligned_judge],
            aggregation=normalized_judge_aggregation,
        )

    optimized = result.optimized_prompts[0]
    registered = mlflow.genai.register_prompt(
        name=STUDENT_PROMPT_NAME,
        template=optimized.template,
        commit_message=f"codex_hydrogym GEPA candidate using {FEEDBACK_ASSESSMENT_NAME}",
        tags={
            "project": PROJECT_LABEL,
            "optimization": "GEPA",
            "judge": FEEDBACK_ASSESSMENT_NAME,
            "promotion_state": "awaiting_heldout_ppo",
        },
        response_format=REWARD_CANDIDATE_JSON_SCHEMA,
    )
    mlflow.genai.set_prompt_alias(name=STUDENT_PROMPT_NAME, alias="candidate", version=registered.version)
    return GepaCandidateResult(
        prompt_name=STUDENT_PROMPT_NAME,
        prompt_version=registered.version,
        initial_score=result.initial_eval_score,
        final_score=result.final_eval_score,
    )


def promote_prompt_after_rollout(
    *,
    prompt_name: str,
    prompt_version: int,
    baseline: RolloutEvidence,
    candidate: RolloutEvidence,
    minimum_tke_improvement: float = 0.02,
    maximum_control_increase: float = 0.25,
    mlflow_module=None,
) -> None:
    """Promote only after comparable held-out evidence passes every hard gate."""
    if not 0.0 <= minimum_tke_improvement <= 1.0:
        raise ValueError("minimum_tke_improvement must be in [0, 1]")
    if not 0.0 <= maximum_control_increase <= 10.0:
        raise ValueError("maximum_control_increase must be in [0, 10]")
    if baseline.context_fingerprint != candidate.context_fingerprint:
        raise ValueError("promotion evidence must use the same held-out context")
    if baseline.frozen_training_fingerprint != candidate.frozen_training_fingerprint:
        raise ValueError("promotion evidence must use the same frozen-training fingerprint")
    if not baseline.physics_gates_passed or not candidate.physics_gates_passed:
        raise ValueError("all baseline and candidate physics gates must pass")
    if baseline.mean_tke <= 0.0:
        raise ValueError("baseline mean TKE must be positive")
    relative_improvement = (baseline.mean_tke - candidate.mean_tke) / baseline.mean_tke
    if relative_improvement < minimum_tke_improvement:
        raise ValueError("candidate did not meet the held-out mean-TKE improvement threshold")
    if baseline.control_l1 <= 0.0:
        if candidate.control_l1 > 1.0e-7:
            raise ValueError("candidate introduced control effort against a zero-control baseline")
    elif (candidate.control_l1 - baseline.control_l1) / baseline.control_l1 > maximum_control_increase:
        raise ValueError("candidate exceeded the held-out control-effort threshold")

    mlflow = mlflow_module or importlib.import_module("mlflow")
    mlflow.genai.set_prompt_alias(name=prompt_name, alias="production", version=prompt_version)


def rollout_evidence_from_run(*, run_id: str, mlflow_client=None) -> RolloutEvidence:
    """Read dedicated held-out metrics; training curves are never promotion evidence."""
    if mlflow_client is None:
        mlflow_client = importlib.import_module("mlflow").MlflowClient()
    run = mlflow_client.get_run(run_id)
    if getattr(run.info, "status", None) != "FINISHED":
        raise ValueError("held-out promotion evidence must come from a FINISHED MLflow run")
    metrics = run.data.metrics
    tags = run.data.tags
    required_metrics = (
        "heldout/mean_tke",
        "heldout/control_l1",
        "heldout/reward_total",
        "heldout/physics_all_passed",
    )
    missing_metrics = [name for name in required_metrics if name not in metrics]
    required_tags = {
        "context_fingerprint": f"{PROJECT_LABEL}.evaluation_context_fingerprint",
        "frozen_training_fingerprint": f"{PROJECT_LABEL}.frozen_training_fingerprint",
        "heldout_evidence_digest": f"{PROJECT_LABEL}.heldout_evidence_digest",
    }
    missing_tags = [tag_key for tag_key in required_tags.values() if not tags.get(tag_key)]
    if missing_metrics or missing_tags:
        raise ValueError(
            f"run lacks codex_hydrogym promotion evidence; missing_metrics={missing_metrics}, "
            f"missing_tags={missing_tags}"
        )
    return RolloutEvidence(
        run_id=run_id,
        context_fingerprint=tags[required_tags["context_fingerprint"]],
        frozen_training_fingerprint=tags[required_tags["frozen_training_fingerprint"]],
        heldout_evidence_digest=tags[required_tags["heldout_evidence_digest"]],
        mean_tke=metrics["heldout/mean_tke"],
        control_l1=metrics["heldout/control_l1"],
        reward_total=metrics["heldout/reward_total"],
        physics_gates_passed=metrics["heldout/physics_all_passed"] == 1.0,
        artifact_uri=run.info.artifact_uri,
    )
