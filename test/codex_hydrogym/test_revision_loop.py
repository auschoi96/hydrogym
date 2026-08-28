"""Paired base-versus-aligned reviewer revision lineage tests."""

import asyncio
import json
from types import SimpleNamespace

import mlflow
import pytest

from codex_hydrogym.genai.contracts import AGENT_FEEDBACK_SCHEMA_VERSION, parse_agent_feedback
from codex_hydrogym.genai.datasets import build_harness_sanity_bundles
from codex_hydrogym.genai.harnesses import RawHarnessResponse, feedback_id_for_bundle
from codex_hydrogym.genai.revision import (
    REVISION_PROMPT_NAME,
    REVISION_PROMPT_TEMPLATE,
    ReviewerAdvice,
    agent_feedback_digest,
    analyze_paired_revisions,
    resolve_revision_prompt,
)
from codex_hydrogym.genai.tracing import HarnessAnalysis


def _feedback(bundle, *, diagnosis="The evidence does not authorize a trial."):
    return parse_agent_feedback(
        {
            "schema_version": AGENT_FEEDBACK_SCHEMA_VERSION,
            "feedback_id": feedback_id_for_bundle(bundle),
            "decision": "collect_evidence",
            "diagnosis": diagnosis,
            "evidence": ["The bundle is a synthetic Gate-0 sanity record."],
            "falsification_test": "Run the preregistered deterministic comparison before PPO.",
            "claim_boundary": "This is feedback-process plumbing, not fluid-improvement evidence.",
            "estimated_cost": "cpu_gate",
            "reward_spec": None,
        }
    )


def _initial(bundle):
    feedback = _feedback(bundle)
    return HarnessAnalysis(
        arm="codex",
        adapter_id="codex_sdk",
        model="gpt-test",
        feedback=feedback,
        prompt_sha256="c" * 64,
        latency_ms=1.0,
        runtime_metadata={"transport": "test"},
        trace_id="tr-initial",
    )


def _advice(bundle, initial, *, treatment, rationale, version):
    return ReviewerAdvice(
        reviewer_name="critic_quality",
        reviewer_version=version,
        treatment=treatment,
        score=3.0 if treatment == "base" else 2.0,
        rationale=rationale,
        bundle_evidence_digest=bundle.evidence_digest,
        initial_draft_digest=agent_feedback_digest(initial.feedback),
        source_trace_id=f"tr-review-{treatment}",
    )


class _Prompt:
    uri = f"prompts:/{REVISION_PROMPT_NAME}/7"
    version = 7
    template = REVISION_PROMPT_TEMPLATE

    def format(self, **values):
        rendered = self.template
        for name, value in values.items():
            rendered = rendered.replace("{{" + name + "}}", str(value))
        assert "{{" not in rendered
        return rendered


class _PromptRegistry:
    def __init__(self):
        self.load_calls = []

    def load_prompt(self, uri):
        self.load_calls.append(uri)
        return _Prompt()


class _SequenceHarness:
    arm = "codex"
    adapter_id = "codex_sdk"
    model = "gpt-test"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        feedback = self.responses.pop(0)
        return RawHarnessResponse(
            arm=self.arm,
            adapter_id=self.adapter_id,
            model=self.model,
            text=json.dumps(feedback.as_dict()),
            metadata={"transport": "test", "forbidden_activity_count": 0},
        )


def test_paired_revisions_reuse_one_draft_and_registered_prompt_with_native_lineage(tmp_path, monkeypatch):
    bundle = build_harness_sanity_bundles()[0]
    initial = _initial(bundle)
    base_advice = _advice(bundle, initial, treatment="base", rationale="Add a clearer stop condition.", version="1")
    aligned_advice = _advice(
        bundle,
        initial,
        treatment="aligned",
        rationale="Do not imply that judge agreement measures fluid performance.",
        version="2",
    )
    harness = _SequenceHarness(
        [
            _feedback(bundle, diagnosis="Base-review revision still requests deterministic evidence."),
            _feedback(bundle, diagnosis="Aligned-review revision tightens the claim boundary."),
        ]
    )
    registry = _PromptRegistry()
    monkeypatch.setattr(mlflow.genai, "load_prompt", registry.load_prompt)
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.set_experiment("paired-revision-lineage-test")
    try:
        base, aligned = asyncio.run(
            analyze_paired_revisions(
                bundle=bundle,
                initial=initial,
                base_advice=base_advice,
                aligned_advice=aligned_advice,
                harness=harness,
                prompt_uri=f"prompts:/{REVISION_PROMPT_NAME}@candidate",
                mlflow_module=mlflow,
            )
        )
        traces = mlflow.search_traces(locations=[experiment.experiment_id], return_type="list")
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)

    assert registry.load_calls == [
        f"prompts:/{REVISION_PROMPT_NAME}@candidate",
        f"prompts:/{REVISION_PROMPT_NAME}@candidate",
    ]
    assert len(harness.prompts) == 2
    assert agent_feedback_digest(initial.feedback) in harness.prompts[0]
    assert agent_feedback_digest(initial.feedback) in harness.prompts[1]
    assert "Add a clearer stop condition." in harness.prompts[0]
    assert "Do not imply that judge agreement" in harness.prompts[1]
    assert base.initial_draft_digest == aligned.initial_draft_digest == agent_feedback_digest(initial.feedback)
    assert base.prompt_uri == aligned.prompt_uri == _Prompt.uri
    assert base.prompt_version == aligned.prompt_version == "7"
    assert base.prompt_template_sha256 == aligned.prompt_template_sha256
    assert base.prompt_sha256 != aligned.prompt_sha256
    assert {trace.info.tags["codex_hydrogym.treatment"] for trace in traces} == {"base", "aligned"}
    assert all(trace.info.tags["codex_hydrogym.initial_draft_digest"] == base.initial_draft_digest for trace in traces)
    assert all(
        {(span.name, span.span_type) for span in trace.data.spans}
        == {
            ("hydrogym_reward_revision_agent", "AGENT"),
            ("registered_prompt_harness_call", "TASK"),
            ("revision_contract_validation", "GUARDRAIL"),
        }
        for trace in traces
    )


def test_reviewer_advice_rejects_cross_draft_and_cross_bundle(monkeypatch):
    bundle, other_bundle = build_harness_sanity_bundles()[:2]
    initial = _initial(bundle)
    wrong_draft = _advice(bundle, initial, treatment="base", rationale="Review.", version="1")
    wrong_draft = ReviewerAdvice(
        **{**wrong_draft.as_dict(), "initial_draft_digest": "d" * 64}
    )
    monkeypatch.setattr(mlflow.genai, "load_prompt", lambda _uri: _Prompt())

    with pytest.raises(ValueError, match="different initial draft"):
        resolve_revision_prompt(
            prompt_uri=f"prompts:/{REVISION_PROMPT_NAME}/7",
            bundle=bundle,
            initial_draft=initial.feedback,
            reviewer_advice=wrong_draft,
            mlflow_module=mlflow,
        )

    cross_bundle = _advice(bundle, initial, treatment="base", rationale="Review.", version="1")
    with pytest.raises(ValueError, match="different RunBundle"):
        cross_bundle.validate_binding(bundle=other_bundle, draft=initial.feedback)


def test_revision_requires_same_coding_agent_provenance(tmp_path, monkeypatch):
    bundle = build_harness_sanity_bundles()[0]
    initial = _initial(bundle)
    advice = _advice(bundle, initial, treatment="base", rationale="Review.", version="1")
    registry = _PromptRegistry()
    monkeypatch.setattr(mlflow.genai, "load_prompt", registry.load_prompt)
    mismatched = _SequenceHarness([_feedback(bundle)])
    mismatched.model = "different-model"
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("revision-agent-mismatch-test")
    try:
        with pytest.raises(ValueError, match="same coding-agent"):
            asyncio.run(
                analyze_paired_revisions(
                    bundle=bundle,
                    initial=initial,
                    base_advice=advice,
                    aligned_advice=ReviewerAdvice(
                        **{**advice.as_dict(), "treatment": "aligned", "reviewer_version": "2"}
                    ),
                    harness=mismatched,
                    prompt_uri=f"prompts:/{REVISION_PROMPT_NAME}/7",
                    mlflow_module=mlflow,
                )
            )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)
