"""Runnable MLflow/MemAlign/GEPA workflows for codex_hydrogym jobs."""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.genai.contracts import build_gepa_record
from codex_hydrogym.genai.datasets import build_scenario_matrix
from codex_hydrogym.genai.feedback import create_human_labeling_session
from codex_hydrogym.genai.gateway import UnityAIGatewayClient, resolve_databricks_token
from codex_hydrogym.genai.judges import evaluate_reward_proposals
from codex_hydrogym.genai.optimization import (
    STUDENT_PROMPT_NAME,
    align_fluid_reward_judge,
    generate_reward_candidate,
    promote_prompt_after_rollout,
    register_base_judge,
    register_student_prompt,
    rollout_evidence_from_run,
    run_gepa_student_optimization,
)
from codex_hydrogym.genai.portfolio import ModelPortfolio


def _runtime(args):
    import mlflow

    host = os.environ.get("DATABRICKS_HOST")
    experiment_id = args.experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")
    if not host:
        raise RuntimeError("DATABRICKS_HOST is required")
    if not experiment_id:
        raise RuntimeError("MLFLOW_EXPERIMENT_ID is required")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=experiment_id)
    token = resolve_databricks_token()
    portfolio_env = {
        "CODEX_HYDROGYM_STUDENT_MODEL": args.student_model or os.environ.get("CODEX_HYDROGYM_STUDENT_MODEL", ""),
        "CODEX_HYDROGYM_PRIMARY_JUDGE_MODEL": args.primary_judge_model
        or os.environ.get("CODEX_HYDROGYM_PRIMARY_JUDGE_MODEL", ""),
        "CODEX_HYDROGYM_AUDIT_JUDGE_MODELS": args.audit_judge_models
        or os.environ.get("CODEX_HYDROGYM_AUDIT_JUDGE_MODELS", ""),
        "CODEX_HYDROGYM_REFLECTION_MODELS": args.reflection_models
        or os.environ.get("CODEX_HYDROGYM_REFLECTION_MODELS", ""),
        "CODEX_HYDROGYM_SMALL_TASK_MODEL": args.small_task_model
        or os.environ.get("CODEX_HYDROGYM_SMALL_TASK_MODEL", ""),
        "CODEX_HYDROGYM_EMBEDDING_MODEL": os.environ.get(
            "CODEX_HYDROGYM_EMBEDDING_MODEL",
            "databricks:/databricks-gte-large-en",
        ),
    }
    return mlflow, host, token, experiment_id, ModelPortfolio.from_env(portfolio_env)


def _bootstrap(args) -> dict:
    mlflow, host, token, experiment_id, portfolio = _runtime(args)
    prompt = register_student_prompt(mlflow_module=mlflow)
    mlflow.genai.set_prompt_alias(name=STUDENT_PROMPT_NAME, alias="baseline", version=prompt.version)
    registered_judge = register_base_judge(
        experiment_id=experiment_id,
        portfolio=portfolio,
        workspace_host=host,
        token=token,
    )

    gateway = UnityAIGatewayClient(workspace_host=host, token=token)
    scenarios = build_scenario_matrix()[: args.proposal_count]
    records = []
    for scenario in scenarios:
        candidate = generate_reward_candidate(
            scenario=scenario.as_dict(),
            prompt_uri=prompt.uri,
            portfolio=portfolio,
            gateway=gateway,
            mlflow_module=mlflow,
        )
        record = build_gepa_record(
            scenario=scenario.as_dict(),
            expected_behavior=(
                "Return one bounded candidate that is physically plausible, falsifiable, and explicitly subject "
                "to held-out PPO and deterministic physics validation."
            ),
        )
        record["outputs"] = candidate.as_dict()
        records.append(record)

    result = evaluate_reward_proposals(
        records=records,
        portfolio=portfolio,
        workspace_host=host,
        token=token,
        mlflow_module=mlflow,
    )
    datasets = mlflow.genai.datasets
    try:
        dataset = datasets.get_dataset(name=args.dataset_name)
    except Exception:
        dataset = datasets.create_dataset(
            name=args.dataset_name,
            experiment_id=experiment_id,
            tags={"project": PROJECT_LABEL, "purpose": "human_reward_feedback"},
        )
    traces = mlflow.search_traces(run_id=result.run_id, return_type="pandas")
    if "inputs" not in traces.columns and "request" in traces.columns:
        traces = traces.rename(columns={"request": "inputs"})
    if "outputs" not in traces.columns and "response" in traces.columns:
        traces = traces.rename(columns={"response": "outputs"})
    dataset = dataset.merge_records(traces)

    session = create_human_labeling_session(
        session_name=args.session_name,
        dataset_name=dataset.name,
        assigned_users=args.reviewer,
        mlflow_genai_module=mlflow.genai,
    )
    return {
        "project": PROJECT_LABEL,
        "experiment_id": experiment_id,
        "evaluation_run_id": result.run_id,
        "dataset_name": dataset.name,
        "labeling_session_url": session.url,
        "prompt_uri": prompt.uri,
        "judge_name": registered_judge.name,
        "proposal_count": len(records),
    }


def _align(args) -> dict:
    _mlflow, host, token, experiment_id, portfolio = _runtime(args)
    judge = align_fluid_reward_judge(
        experiment_id=experiment_id,
        portfolio=portfolio,
        workspace_host=host,
        token=token,
        retrieval_k=args.retrieval_k,
    )
    return {
        "project": PROJECT_LABEL,
        "experiment_id": experiment_id,
        "judge_name": judge.name,
        "alignment": "MemAlign",
        "embedding_model": portfolio.embedding_model,
    }


def _optimize(args) -> dict:
    _mlflow, host, token, experiment_id, portfolio = _runtime(args)
    outcome = run_gepa_student_optimization(
        experiment_id=experiment_id,
        prompt_uri=args.prompt_uri,
        portfolio=portfolio,
        workspace_host=host,
        token=token,
        max_metric_calls=args.max_metric_calls,
    )
    return {
        "project": PROJECT_LABEL,
        "prompt_name": outcome.prompt_name,
        "prompt_version": outcome.prompt_version,
        "initial_score": outcome.initial_score,
        "final_score": outcome.final_score,
        "alias": outcome.promoted_alias,
        "promotion_state": "awaiting_heldout_ppo",
    }


def _promote(args) -> dict:
    mlflow, _host, _token, _experiment_id, _portfolio = _runtime(args)
    baseline = rollout_evidence_from_run(run_id=args.baseline_run_id)
    candidate = rollout_evidence_from_run(run_id=args.candidate_run_id)
    promote_prompt_after_rollout(
        prompt_name=STUDENT_PROMPT_NAME,
        prompt_version=args.prompt_version,
        baseline=baseline,
        candidate=candidate,
        minimum_tke_improvement=args.minimum_tke_improvement,
        maximum_control_increase=args.maximum_control_increase,
        mlflow_module=mlflow,
    )
    return {
        "project": PROJECT_LABEL,
        "prompt_name": STUDENT_PROMPT_NAME,
        "prompt_version": args.prompt_version,
        "alias": "production",
        "baseline_run_id": args.baseline_run_id,
        "candidate_run_id": args.candidate_run_id,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="codex_hydrogym MLflow human-feedback outer loop")
    parser.add_argument("--experiment-id")
    parser.add_argument("--student-model")
    parser.add_argument("--primary-judge-model")
    parser.add_argument("--audit-judge-models")
    parser.add_argument("--reflection-models")
    parser.add_argument("--small-task-model")
    subcommands = parser.add_subparsers(dest="command", required=True)

    bootstrap = subcommands.add_parser("bootstrap", help="generate proposals and open human labeling")
    bootstrap.add_argument("--dataset-name", default="codex_hydrogym_reward_proposals")
    bootstrap.add_argument("--session-name", default="codex_hydrogym_fluid_reward_review")
    bootstrap.add_argument("--reviewer", action="append", required=True)
    bootstrap.add_argument("--proposal-count", type=int, choices=range(6, 25), default=12, metavar="6..24")

    align = subcommands.add_parser("align", help="align the primary judge with MemAlign")
    align.add_argument("--retrieval-k", type=int, default=5)

    optimize = subcommands.add_parser("optimize", help="optimize the student prompt with GEPA")
    optimize.add_argument(
        "--prompt-uri",
        default="prompts:/codex_hydrogym_reward_student@baseline",
    )
    optimize.add_argument("--max-metric-calls", type=int, default=75)

    promote = subcommands.add_parser("promote", help="promote only after comparable PPO physics gates")
    promote.add_argument("--prompt-version", type=int, required=True)
    promote.add_argument("--baseline-run-id", required=True)
    promote.add_argument("--candidate-run-id", required=True)
    promote.add_argument("--minimum-tke-improvement", type=float, default=0.02)
    promote.add_argument("--maximum-control-increase", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"bootstrap": _bootstrap, "align": _align, "optimize": _optimize, "promote": _promote}
    result = handlers[args.command](args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
