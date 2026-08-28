"""MemAlign, GEPA, and hard-promotion orchestration tests."""

import json
from types import SimpleNamespace

import pytest

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME
from codex_hydrogym.genai.contracts import REWARD_CANDIDATE_SCHEMA_VERSION, RolloutEvidence
from codex_hydrogym.genai.gateway import GatewayResponse
from codex_hydrogym.genai.optimization import (
    align_critic_quality_judge,
    align_fluid_reward_judge,
    generate_reward_candidate,
    normalized_judge_aggregation,
    promote_prompt_after_rollout,
    register_aligned_critic_quality_judge,
    rollout_evidence_from_run,
    unity_gateway_environment,
)
from codex_hydrogym.genai.portfolio import ModelPortfolio


_CONTEXT_FINGERPRINT = "a" * 64
_FROZEN_TRAINING_FINGERPRINT = "b" * 64
_HELDOUT_EVIDENCE_DIGEST = "c" * 64


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


def _candidate_json():
    return json.dumps(
        {
            "schema_version": REWARD_CANDIDATE_SCHEMA_VERSION,
            "candidate_id": "codex_hydrogym_gateway_candidate",
            "reward_alpha": 1.2,
            "learning_rate": 0.0001,
            "entropy_coefficient": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.985,
            "num_updates": 10,
            "hypothesis": "A modest TKE penalty increase should improve suppression.",
            "rationale": "Validate against mean TKE and control effort on a held-out seed.",
        }
    )


class _Gateway:
    def __init__(self):
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return GatewayResponse(
            text=_candidate_json(),
            model=kwargs["model"],
            request_id="r1",
            finish_reason="stop",
            usage={},
        )


class _Prompt:
    def format(self, **kwargs):
        return f"scenario={kwargs['scenario']}"


def test_student_generation_uses_direct_gateway_and_strict_parser():
    gateway = _Gateway()
    mlflow = SimpleNamespace(genai=SimpleNamespace(load_prompt=lambda uri: _Prompt()))

    candidate = generate_reward_candidate(
        scenario={"reynolds_number": 200, "seed": 41},
        prompt_uri="prompts:/codex_hydrogym_reward_student@candidate",
        portfolio=_portfolio(),
        gateway=gateway,
        mlflow_module=mlflow,
    )

    assert candidate.candidate_id == "codex_hydrogym_gateway_candidate"
    assert gateway.kwargs["model"] == "system.ai.deepseek-flash"
    assert gateway.kwargs["request_tags"]["role"] == "student"


def test_unity_gateway_environment_is_scoped_and_restored():
    environ = {"OPENAI_API_KEY": "old-key", "UNRELATED": "keep"}

    with unity_gateway_environment(
        workspace_host="fevm.example.cloud.databricks.com",
        token="oauth-token",
        environ=environ,
    ) as base_url:
        assert base_url.endswith("/ai-gateway/mlflow/v1")
        assert environ["OPENAI_API_KEY"] == "oauth-token"
        assert environ["OPENAI_API_BASE"] == base_url
        assert environ["DATABRICKS_TOKEN"] == "oauth-token"

    assert environ == {"OPENAI_API_KEY": "old-key", "UNRELATED": "keep"}


class _AlignedJudge:
    def __init__(self):
        self.name = CRITIC_QUALITY_ASSESSMENT_NAME
        self.align_kwargs = None
        self.update_kwargs = None
        self.register_kwargs = None

    def align(self, **kwargs):
        self.align_kwargs = kwargs
        return self

    def update(self, **kwargs):
        self.update_kwargs = kwargs
        return self

    def register(self, **kwargs):
        self.register_kwargs = kwargs
        return SimpleNamespace(name=self.name, version=2)


class _Optimizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_memalign_filters_human_feedback_and_pins_embedding_model():
    assessment = SimpleNamespace(
        name=FEEDBACK_ASSESSMENT_NAME,
        source=SimpleNamespace(source_type="HUMAN", source_id="expert@example.com"),
    )
    trace = SimpleNamespace(info=SimpleNamespace(assessments=[assessment]))
    mlflow = SimpleNamespace(search_traces=lambda **kwargs: [trace])
    judge = _AlignedJudge()
    created = []

    def optimizer_factory(**kwargs):
        optimizer = _Optimizer(**kwargs)
        created.append(optimizer)
        return optimizer

    result = align_fluid_reward_judge(
        experiment_id="123",
        portfolio=_portfolio(),
        workspace_host="https://fevm.example.cloud.databricks.com",
        token="oauth",
        mlflow_module=mlflow,
        get_scorer_fn=lambda **kwargs: judge,
        optimizer_factory=optimizer_factory,
    )

    assert result is judge
    assert created[0].kwargs["embedding_model"] == "databricks:/databricks-gte-large-en"
    assert created[0].kwargs["reflection_lm"] == "openai:/system.ai.glm-5-2"
    assert judge.align_kwargs["traces"] == [trace]
    assert judge.update_kwargs is None


def _critic_trace(bundle_id, arm, *, labels=1):
    assessments = [
        SimpleNamespace(
            name=CRITIC_QUALITY_ASSESSMENT_NAME,
            source=SimpleNamespace(source_type="HUMAN", source_id=f"panel-{index}"),
        )
        for index in range(labels)
    ]
    return SimpleNamespace(
        info=SimpleNamespace(
            tags={
                "codex_hydrogym.bundle_id": bundle_id,
                "codex_hydrogym.harness_arm": arm,
            },
            assessments=assessments,
        )
    )


def test_critic_memalign_uses_only_locked_train_traces_and_stays_inline():
    traces = [
        _critic_trace(bundle_id, arm)
        for bundle_id in ("bundle-1", "bundle-2")
        for arm in ("codex", "claude")
    ]
    judge = _AlignedJudge()
    optimizers = []

    aligned = align_critic_quality_judge(
        train_traces=traces,
        train_bundle_ids=["bundle-1", "bundle-2"],
        heldout_bundle_ids=["bundle-3"],
        base_judge=judge,
        reflection_lm="databricks:/reflection",
        embedding_model="databricks:/embedding",
        optimizer_factory=lambda **kwargs: optimizers.append(_Optimizer(**kwargs)) or optimizers[-1],
    )

    assert aligned is judge
    assert judge.align_kwargs["traces"] == traces
    assert judge.update_kwargs is None
    assert optimizers[0].kwargs["embedding_model"] == "databricks:/embedding"

    codex_only = [_critic_trace(bundle_id, "codex") for bundle_id in ("bundle-1", "bundle-2")]
    codex_judge = _AlignedJudge()
    assert (
        align_critic_quality_judge(
            train_traces=codex_only,
            train_bundle_ids=["bundle-1", "bundle-2"],
            heldout_bundle_ids=["bundle-3"],
            base_judge=codex_judge,
            reflection_lm="databricks:/reflection",
            embedding_model="databricks:/embedding",
            required_arms=("codex",),
            optimizer_factory=_Optimizer,
        )
        is codex_judge
    )

    registered = register_aligned_critic_quality_judge(aligned_judge=codex_judge, experiment_id="123")
    assert registered.version == 2
    assert codex_judge.register_kwargs == {"experiment_id": "123"}

    with pytest.raises(ValueError, match="held-out bundle leaked|outside the locked"):
        align_critic_quality_judge(
            train_traces=[*traces, _critic_trace("bundle-3", "codex")],
            train_bundle_ids=["bundle-1", "bundle-2"],
            heldout_bundle_ids=["bundle-3"],
            base_judge=_AlignedJudge(),
            reflection_lm="databricks:/reflection",
            embedding_model="databricks:/embedding",
            optimizer_factory=_Optimizer,
        )

    with pytest.raises(ValueError, match="exactly one adjudicated"):
        align_critic_quality_judge(
            train_traces=[
                _critic_trace("bundle-1", "codex", labels=2),
                _critic_trace("bundle-1", "claude"),
            ],
            train_bundle_ids=["bundle-1"],
            heldout_bundle_ids=["bundle-3"],
            base_judge=_AlignedJudge(),
            reflection_lm="databricks:/reflection",
            embedding_model="databricks:/embedding",
            optimizer_factory=_Optimizer,
        )


def test_gepa_aggregation_normalizes_exact_judge_feedback():
    feedback = SimpleNamespace(feedback=SimpleNamespace(value=5.0))
    assert normalized_judge_aggregation({FEEDBACK_ASSESSMENT_NAME: feedback}) == 1.0
    assert normalized_judge_aggregation({FEEDBACK_ASSESSMENT_NAME: 3.0}) == 0.5
    assert normalized_judge_aggregation({}) == 0.0


def _evidence(
    run_id,
    *,
    tke,
    control,
    passed=True,
    context=_CONTEXT_FINGERPRINT,
    frozen_training_fingerprint=_FROZEN_TRAINING_FINGERPRINT,
    heldout_evidence_digest=_HELDOUT_EVIDENCE_DIGEST,
):
    return RolloutEvidence(
        run_id=run_id,
        context_fingerprint=context,
        frozen_training_fingerprint=frozen_training_fingerprint,
        heldout_evidence_digest=heldout_evidence_digest,
        mean_tke=tke,
        control_l1=control,
        reward_total=-tke - control,
        physics_gates_passed=passed,
        artifact_uri=f"runs:/{run_id}/evidence",
    )


def test_prompt_promotion_requires_real_physics_improvement_and_control_budget():
    aliases = []
    mlflow = SimpleNamespace(genai=SimpleNamespace(set_prompt_alias=lambda **kwargs: aliases.append(kwargs)))
    baseline = _evidence("baseline", tke=2.0, control=1.0)
    candidate = _evidence("candidate", tke=1.8, control=1.1)

    promote_prompt_after_rollout(
        prompt_name="codex_hydrogym_reward_student",
        prompt_version=3,
        baseline=baseline,
        candidate=candidate,
        mlflow_module=mlflow,
    )
    assert aliases == [{"name": "codex_hydrogym_reward_student", "alias": "production", "version": 3}]

    with pytest.raises(ValueError, match="physics gates"):
        promote_prompt_after_rollout(
            prompt_name="codex_hydrogym_reward_student",
            prompt_version=4,
            baseline=baseline,
            candidate=_evidence("bad", tke=1.0, control=1.0, passed=False),
            mlflow_module=mlflow,
        )
    with pytest.raises(ValueError, match="control-effort"):
        promote_prompt_after_rollout(
            prompt_name="codex_hydrogym_reward_student",
            prompt_version=4,
            baseline=baseline,
            candidate=_evidence("expensive", tke=1.0, control=2.0),
            mlflow_module=mlflow,
        )
    with pytest.raises(ValueError, match="frozen-training"):
        promote_prompt_after_rollout(
            prompt_name="codex_hydrogym_reward_student",
            prompt_version=4,
            baseline=baseline,
            candidate=_evidence(
                "different-training",
                tke=1.0,
                control=1.0,
                frozen_training_fingerprint="d" * 64,
            ),
            mlflow_module=mlflow,
        )


def test_rollout_evidence_is_read_from_labeled_mlflow_run():
    run = SimpleNamespace(
        info=SimpleNamespace(artifact_uri="dbfs:/codex_hydrogym/run-1", status="FINISHED"),
        data=SimpleNamespace(
            metrics={
                "heldout/mean_tke": 1.2,
                "heldout/control_l1": 0.4,
                "heldout/reward_total": -1.6,
                "heldout/physics_all_passed": 1.0,
            },
            tags={
                "codex_hydrogym.evaluation_context_fingerprint": _CONTEXT_FINGERPRINT,
                "codex_hydrogym.frozen_training_fingerprint": _FROZEN_TRAINING_FINGERPRINT,
                "codex_hydrogym.heldout_evidence_digest": _HELDOUT_EVIDENCE_DIGEST,
            },
        ),
    )
    evidence = rollout_evidence_from_run(
        run_id="run-1",
        mlflow_client=SimpleNamespace(get_run=lambda run_id: run),
    )

    assert evidence.run_id == "run-1"
    assert evidence.physics_gates_passed is True
    assert evidence.context_fingerprint == _CONTEXT_FINGERPRINT
    assert evidence.frozen_training_fingerprint == _FROZEN_TRAINING_FINGERPRINT
    assert evidence.heldout_evidence_digest == _HELDOUT_EVIDENCE_DIGEST


def test_rollout_evidence_rejects_training_metrics_as_heldout_evidence():
    run = SimpleNamespace(
        info=SimpleNamespace(artifact_uri="dbfs:/codex_hydrogym/train-only", status="FINISHED"),
        data=SimpleNamespace(
            metrics={
                "train/mean_tke": 1.2,
                "train/control_l1": 0.4,
                "train/reward_total": -1.6,
                "physics/all_passed": 1.0,
            },
            tags={
                "codex_hydrogym.evaluation_context_fingerprint": _CONTEXT_FINGERPRINT,
                "codex_hydrogym.frozen_training_fingerprint": _FROZEN_TRAINING_FINGERPRINT,
                "codex_hydrogym.heldout_evidence_digest": _HELDOUT_EVIDENCE_DIGEST,
            },
        ),
    )

    with pytest.raises(ValueError, match="heldout/mean_tke"):
        rollout_evidence_from_run(
            run_id="train-only",
            mlflow_client=SimpleNamespace(get_run=lambda run_id: run),
        )


@pytest.mark.parametrize(
    ("tag_name", "tag_value"),
    [
        ("codex_hydrogym.frozen_training_fingerprint", None),
        ("codex_hydrogym.heldout_evidence_digest", None),
        ("codex_hydrogym.heldout_evidence_digest", "not-a-digest"),
    ],
)
def test_rollout_evidence_rejects_missing_or_malformed_lineage(tag_name, tag_value):
    tags = {
        "codex_hydrogym.evaluation_context_fingerprint": _CONTEXT_FINGERPRINT,
        "codex_hydrogym.frozen_training_fingerprint": _FROZEN_TRAINING_FINGERPRINT,
        "codex_hydrogym.heldout_evidence_digest": _HELDOUT_EVIDENCE_DIGEST,
    }
    if tag_value is None:
        tags.pop(tag_name)
    else:
        tags[tag_name] = tag_value
    run = SimpleNamespace(
        info=SimpleNamespace(artifact_uri="dbfs:/codex_hydrogym/run-1", status="FINISHED"),
        data=SimpleNamespace(
            metrics={
                "heldout/mean_tke": 1.2,
                "heldout/control_l1": 0.4,
                "heldout/reward_total": -1.6,
                "heldout/physics_all_passed": 1.0,
            },
            tags=tags,
        ),
    )

    with pytest.raises(ValueError, match="missing_tags|lowercase SHA-256"):
        rollout_evidence_from_run(
            run_id="run-1",
            mlflow_client=SimpleNamespace(get_run=lambda run_id: run),
        )


def test_rollout_evidence_rejects_incomplete_runs():
    run = SimpleNamespace(
        info=SimpleNamespace(artifact_uri="dbfs:/codex_hydrogym/run-1", status="RUNNING"),
        data=SimpleNamespace(metrics={}, tags={}),
    )

    with pytest.raises(ValueError, match="FINISHED"):
        rollout_evidence_from_run(
            run_id="run-1",
            mlflow_client=SimpleNamespace(get_run=lambda run_id: run),
        )
