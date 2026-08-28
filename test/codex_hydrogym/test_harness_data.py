"""Evidence contracts, sanity corpus, grouped splits, and MLflow records."""

import asyncio
import copy
from dataclasses import replace
import hashlib
import json

import mlflow
import pytest

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.genai.contracts import (
    AGENT_FEEDBACK_SCHEMA_VERSION,
    REQUIRED_PHYSICS_GATES,
    AgentFeedback,
    EvidenceArm,
    RunBundle,
    parse_agent_feedback,
    parse_run_bundle,
)
from codex_hydrogym.genai.datasets import (
    build_harness_sanity_bundles,
    grouped_bundle_split,
    harness_screen_records,
    paired_harness_records,
    publish_harness_records,
)
from codex_hydrogym.genai.harnesses import (
    HARNESS_ADAPTER_ARMS,
    HARNESS_ADAPTER_IDS,
    RawHarnessResponse,
    feedback_id_for_bundle,
    prompt_digest,
    render_feedback_prompt,
)
from codex_hydrogym.genai.tracing import HarnessAnalysis, analyze_run_bundle


def _measured_arm(case, arm_id, *, observations, tke, effort, context=None):
    context = context or f"context-{case}"
    return EvidenceArm(
        arm_id=arm_id,
        run_id=f"run_{case}_{arm_id}",
        evidence_kind="measured",
        artifact_ref=f"runs:/run_{case}_{arm_id}/evidence.json",
        artifact_sha256=("%064x" % (abs(hash((case, arm_id))) % (2**256))),
        context_fingerprint=("%064x" % (abs(hash(context)) % (2**256))),
        controller_kind="feedback" if observations else "constant_open_loop",
        uses_observations=observations,
        mean_tke=tke,
        control_effort=effort,
        physics_gates=dict.fromkeys(REQUIRED_PHYSICS_GATES, True),
        metrics={},
    )


def _measured_bundle(case, *, group=None):
    return RunBundle(
        bundle_id=f"bundle_{case}",
        group_id=group or f"group_{case}",
        task_contract_version="gate0.v1",
        task={"objective": "feedback necessity"},
        training={"budget_locked": True},
        candidate=_measured_arm(case, "candidate", observations=True, tke=0.7, effort=0.4),
        comparators=(_measured_arm(case, "constant", observations=False, tke=1.0, effort=0.4),),
        diagnostics=(),
        artifact_refs=(f"runs:/run_{case}_candidate/evidence.json",),
    )


def _feedback(bundle):
    return AgentFeedback(
        schema_version=AGENT_FEEDBACK_SCHEMA_VERSION,
        feedback_id=feedback_id_for_bundle(bundle),
        decision="collect_evidence",
        diagnosis="Run the observation-shuffle ablation before any RL claim.",
        evidence=("candidate and constant are comparable",),
        falsification_test="Shuffle observations while preserving actions.",
        claim_boundary="Critic quality is not fluid improvement.",
        estimated_cost="cpu_gate",
        reward_spec=None,
    )


def _analysis(bundle, arm, *, adapter_id=None):
    adapter_id = adapter_id or {"codex": "codex_sdk", "claude": "claude_agent_sdk"}[arm]
    return HarnessAnalysis(
        arm=arm,
        adapter_id=adapter_id,
        model=f"{arm}-model",
        feedback=_feedback(bundle),
        prompt_sha256=prompt_digest(render_feedback_prompt(bundle)),
        latency_ms=12.0,
        runtime_metadata={"total_cost_usd": 0.01},
        trace_id=f"tr-{hashlib.sha256(adapter_id.encode()).hexdigest()[:32]}",
    )


def _screen_analyses(bundle):
    return [
        _analysis(bundle, arm, adapter_id=adapter_id)
        for adapter_id, arm in HARNESS_ADAPTER_ARMS.items()
    ]


class _StaticHarness:
    def __init__(self, bundle, *, arm, adapter_id, response=None):
        self.arm = arm
        self.adapter_id = adapter_id
        self.model = f"{arm}-model"
        feedback = _feedback(bundle).as_dict() if response is None else response
        self.response = json.dumps(feedback)

    async def generate(self, _prompt):
        return RawHarnessResponse(
            arm=self.arm,
            adapter_id=self.adapter_id,
            model=self.model,
            text=self.response,
            metadata={"total_cost_usd": 0.01},
        )


def _native_analysis(bundle, *, arm="codex", adapter_id="codex_sdk", response=None):
    return asyncio.run(
        analyze_run_bundle(
            bundle=bundle,
            harness=_StaticHarness(
                bundle,
                arm=arm,
                adapter_id=adapter_id,
                response=response,
            ),
            mlflow_module=mlflow,
        )
    )


def _single_record(bundle, analysis):
    other_arm = "claude" if analysis.arm == "codex" else "codex"
    records = paired_harness_records(
        bundle=bundle,
        analyses=[analysis, _analysis(bundle, other_arm)],
        fold="test",
        prompt_version="critic.v1",
        blinding_salt="single-native-record-salt",
    )
    return next(record for record in records if record["tags"]["arm"] == analysis.arm)


def test_sanity_bundles_are_nonclaiming_and_expose_each_gate0_failure():
    bundles = build_harness_sanity_bundles()

    assert len(bundles) == 5
    assert len({bundle.bundle_id for bundle in bundles}) == 5
    assert all(bundle.task["claim_allowed"] is False for bundle in bundles)
    assert all(bundle.comparison_issues() for bundle in bundles)
    assert all(
        arm.evidence_kind != "measured"
        for bundle in bundles
        for arm in (bundle.candidate, *bundle.comparators)
    )

    open_loop = bundles[0]
    assert open_loop.candidate.mean_tke == pytest.approx(1.2599737644)
    assert open_loop.comparators[0].mean_tke == pytest.approx(3.1696414948)
    assert "does not use observations" in " ".join(open_loop.comparison_issues())
    assert open_loop.candidate.artifact_sha256 == (
        "2c43041be0c1ad93f1fa1b0cb7523e86dfbb09b92af995be65397dd8aa0c69ce"
    )

    assert "different context fingerprint" in " ".join(bundles[1].comparison_issues())
    assert bundles[2].candidate.physics_gates["spectral_tail_controlled"] is False
    assert bundles[3].candidate.mean_tke - bundles[3].comparators[0].mean_tke == pytest.approx(-0.001)
    assert bundles[4].candidate.mean_tke < next(
        arm.mean_tke for arm in bundles[4].comparators if arm.arm_id == "constant_frontier"
    )


def test_parsers_reject_string_sequences_and_bundle_data_is_immutable():
    bundle = _measured_bundle("parser")
    payload = bundle.as_dict()
    payload["diagnostics"] = "not-a-list"
    with pytest.raises(TypeError, match="diagnostics"):
        parse_run_bundle(payload)

    payload = bundle.as_dict()
    payload["artifact_refs"] = "not-a-list"
    with pytest.raises(TypeError, match="artifact_refs"):
        parse_run_bundle(payload)

    feedback = _feedback(bundle).as_dict()
    feedback["evidence"] = "not-a-list"
    with pytest.raises(TypeError, match="evidence"):
        parse_agent_feedback(feedback)

    source = {"objective": {"nested": [1, 2]}}
    immutable = replace(bundle, task=source)
    digest = immutable.evidence_digest
    source["objective"]["nested"].append(3)
    assert immutable.evidence_digest == digest
    with pytest.raises(TypeError):
        immutable.task["new"] = "mutation"


def test_contract_rejects_overflow_and_flags_open_loop_pareto_domination():
    with pytest.raises(ValueError, match="finite"):
        replace(_measured_bundle("overflow").candidate, mean_tke=10**10_000)

    candidate = _measured_arm("dominated", "candidate", observations=False, tke=2.0, effort=0.7)
    constant = _measured_arm("dominated", "constant", observations=False, tke=1.0, effort=0.5)
    bundle = replace(_measured_bundle("dominated"), candidate=candidate, comparators=(constant,))
    issues = " ".join(bundle.comparison_issues())
    assert "does not use observations" in issues
    assert "Pareto-dominates" in issues


def test_grouped_split_is_deterministic_and_rejects_provenance_relabeling():
    bundles = [_measured_bundle(f"split{index}") for index in range(4)]
    first = grouped_bundle_split(bundles, test_group_count=1)
    second = grouped_bundle_split(list(reversed(bundles)), test_group_count=1)
    assert first == second
    assert list(first.values()).count("test") == 1

    leaked = replace(
        bundles[1],
        candidate=replace(
            bundles[1].candidate,
            run_id=bundles[0].candidate.run_id,
            artifact_sha256=bundles[0].candidate.artifact_sha256,
        ),
    )
    with pytest.raises(ValueError, match="multiple group_id"):
        grouped_bundle_split([bundles[0], leaked, *bundles[2:]], test_group_count=1)


def test_paired_records_are_blinded_and_reserve_gold_name():
    bundle = _measured_bundle("records")
    analyses = [_analysis(bundle, "codex"), _analysis(bundle, "claude")]
    records = paired_harness_records(
        bundle=bundle,
        analyses=analyses,
        fold="test",
        prompt_version="critic.v1",
        blinding_salt="secret-test-salt",
        expectations_by_arm={
            "codex": {"critic_quality_gold": 4},
            "claude": {"critic_quality_gold": 3},
        },
    )

    assert len(records) == 2
    assert records[0]["inputs"] != records[1]["inputs"]
    assert set(records[0]["inputs"]) == {"case_id", "run_bundle", "task_contract_version"}
    assert "harness_arm" not in records[0]["inputs"]
    assert {record["tags"]["arm"] for record in records} == {"codex", "claude"}
    assert {record["tags"]["adapter_id"] for record in records} == {
        "codex_sdk",
        "claude_agent_sdk",
    }
    assert {record["source"]["source_type"] for record in records} == {"TRACE"}
    assert {
        record["source"]["source_data"]["trace_id"] for record in records
    } == {analysis.trace_id for analysis in analyses}
    assert all("critic_quality" not in record["expectations"] for record in records)

    with pytest.raises(ValueError, match="reserved"):
        paired_harness_records(
            bundle=bundle,
            analyses=analyses,
            fold="test",
            prompt_version="critic.v1",
            blinding_salt="secret-test-salt",
            expectations_by_arm={
                "codex": {"critic_quality": 4, "critic_quality_gold": 4},
                "claude": {"critic_quality_gold": 3},
            },
        )

    with pytest.raises(ValueError, match="sanity fold"):
        paired_harness_records(
            bundle=build_harness_sanity_bundles()[0],
            analyses=[
                _analysis(build_harness_sanity_bundles()[0], "codex"),
                _analysis(build_harness_sanity_bundles()[0], "claude"),
            ],
            fold="train",
            prompt_version="critic.v1",
            blinding_salt="secret-test-salt",
        )


@pytest.mark.parametrize("trace_id", [None, "", "   "])
def test_harness_records_require_trace_lineage(trace_id):
    bundle = _measured_bundle("missing-trace")
    analyses = [
        replace(_analysis(bundle, "codex"), trace_id=trace_id),
        _analysis(bundle, "claude"),
    ]

    with pytest.raises(ValueError, match="trace_id must be non-empty"):
        paired_harness_records(
            bundle=bundle,
            analyses=analyses,
            fold="test",
            prompt_version="critic.v1",
            blinding_salt="missing-trace-salt",
        )


def test_paired_records_require_distinct_source_traces():
    bundle = _measured_bundle("duplicate-paired-trace")
    codex = _analysis(bundle, "codex")
    claude = replace(_analysis(bundle, "claude"), trace_id=codex.trace_id)

    with pytest.raises(ValueError, match="distinct trace_id"):
        paired_harness_records(
            bundle=bundle,
            analyses=[codex, claude],
            fold="test",
            prompt_version="critic.v1",
            blinding_salt="duplicate-paired-trace-salt",
        )


def test_screen_records_preserve_four_adapter_provenance_without_unblinding_inputs():
    bundle = _measured_bundle("screen")
    analyses = _screen_analyses(bundle)
    records = harness_screen_records(
        bundle=bundle,
        analyses=analyses,
        fold="test",
        prompt_version="critic.v1",
        blinding_salt="screen-test-salt",
    )

    assert len(records) == 4
    assert [record["tags"]["adapter_id"] for record in records] == list(HARNESS_ADAPTER_IDS)
    assert len({record["tags"]["prompt_sha256"] for record in records}) == 1
    assert len({record["inputs"]["case_id"] for record in records}) == 4
    assert len({record["source"]["source_data"]["trace_id"] for record in records}) == 4
    assert all(record["source"]["source_type"] == "TRACE" for record in records)
    assert all(
        set(record["inputs"]) == {"case_id", "run_bundle", "task_contract_version"}
        for record in records
    )
    assert all(
        record["tags"]["adapter_id"] not in repr(record["inputs"]) for record in records
    )


def test_screen_records_require_exact_uncrossed_adapter_set_and_distinct_traces():
    bundle = _measured_bundle("screen-validation")
    analyses = _screen_analyses(bundle)
    kwargs = {
        "bundle": bundle,
        "fold": "test",
        "prompt_version": "critic.v1",
        "blinding_salt": "screen-validation-salt",
    }

    with pytest.raises(ValueError, match="exactly one result for each harness adapter"):
        harness_screen_records(analyses=analyses[:-1], **kwargs)
    with pytest.raises(ValueError, match="exactly one result for each harness adapter"):
        harness_screen_records(analyses=[*analyses[:-1], analyses[0]], **kwargs)

    cross_wired = [replace(analyses[0], arm="claude"), *analyses[1:]]
    with pytest.raises(ValueError, match="does not belong"):
        harness_screen_records(analyses=cross_wired, **kwargs)

    repeated_trace = [*analyses[:-1], replace(analyses[-1], trace_id=analyses[0].trace_id)]
    with pytest.raises(ValueError, match="distinct trace_id"):
        harness_screen_records(analyses=repeated_trace, **kwargs)


def test_native_mlflow_dataset_keeps_four_adapters_trace_sourced_and_idempotent(tmp_path):
    bundle = _measured_bundle("native")
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment("harness-dataset-test")
    try:
        mlflow.set_experiment(experiment_id=experiment_id)
        analyses = [
            _native_analysis(bundle, arm=arm, adapter_id=adapter_id)
            for adapter_id, arm in HARNESS_ADAPTER_ARMS.items()
        ]
        records = harness_screen_records(
            bundle=bundle,
            analyses=analyses,
            fold="test",
            prompt_version="critic.v1",
            blinding_salt="native-test-salt",
        )
        dataset = publish_harness_records(
            dataset_name="harness-paired-test",
            experiment_id=experiment_id,
            records=records,
            mlflow_module=mlflow,
        )
        assert len(dataset.to_df()) == 4
        dataset = publish_harness_records(
            dataset_name="harness-paired-test",
            experiment_id=experiment_id,
            records=records,
            mlflow_module=mlflow,
        )
        frame = dataset.to_df()
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)

    assert len(frame) == 4
    assert len({row["case_id"] for row in frame["inputs"]}) == 4
    assert set(frame["source_type"]) == {"TRACE"}
    assert set(frame["source_id"]) == {analysis.trace_id for analysis in analyses}


def test_publish_rejects_trace_ids_missing_from_active_tracking_store(tmp_path):
    bundle = _measured_bundle("missing-native-trace")
    records = harness_screen_records(
        bundle=bundle,
        analyses=_screen_analyses(bundle),
        fold="test",
        prompt_version="critic.v1",
        blinding_salt="missing-native-trace-salt",
    )
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment("missing-harness-trace-test")
    try:
        with pytest.raises(ValueError, match="does not exist in the active tracking store"):
            publish_harness_records(
                dataset_name="missing-harness-trace-dataset",
                experiment_id=experiment_id,
                records=records,
                mlflow_module=mlflow,
            )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_publish_rejects_trace_from_another_experiment(tmp_path):
    bundle = _measured_bundle("wrong-experiment")
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    source_experiment_id = mlflow.create_experiment("harness-source-experiment")
    target_experiment_id = mlflow.create_experiment("harness-target-experiment")
    try:
        mlflow.set_experiment(experiment_id=source_experiment_id)
        record = _single_record(bundle, _native_analysis(bundle))
        with pytest.raises(ValueError, match="does not belong to the target experiment"):
            publish_harness_records(
                dataset_name="wrong-experiment-dataset",
                experiment_id=target_experiment_id,
                records=[record],
                mlflow_module=mlflow,
            )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_publish_rejects_error_trace(tmp_path):
    bundle = _measured_bundle("error-trace")
    malformed = _feedback(bundle).as_dict()
    malformed["feedback_id"] = "analysis_wrong_bundle"
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.set_experiment("harness-error-publication-test")
    try:
        with pytest.raises(ValueError, match="feedback_id must equal"):
            _native_analysis(bundle, response=malformed)
        traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            return_type="list",
        )
        assert len(traces) == 1
        record = _single_record(
            bundle,
            replace(_analysis(bundle, "codex"), trace_id=traces[0].info.trace_id),
        )
        with pytest.raises(ValueError, match="must have state OK"):
            publish_harness_records(
                dataset_name="error-trace-dataset",
                experiment_id=experiment.experiment_id,
                records=[record],
                mlflow_module=mlflow,
            )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_publish_rejects_tampered_trace_provenance_bundle_and_outputs(tmp_path):
    bundle = _measured_bundle("tampered-publication")
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.set_experiment("harness-tampered-publication-test")
    try:
        record = _single_record(bundle, _native_analysis(bundle))
        for tag_name in (
            "bundle_id",
            "group_id",
            "arm",
            "adapter_id",
            "model",
            "evidence_digest",
            "prompt_sha256",
            "task_contract_version",
        ):
            tampered = copy.deepcopy(record)
            tampered["tags"][tag_name] = f"tampered-{tag_name}"
            with pytest.raises(ValueError, match=f"record tag {tag_name}"):
                publish_harness_records(
                    dataset_name=f"tampered-{tag_name}-dataset",
                    experiment_id=experiment.experiment_id,
                    records=[tampered],
                    mlflow_module=mlflow,
                )

        tampered_bundle = copy.deepcopy(record)
        tampered_bundle["inputs"]["run_bundle"]["task"]["objective"] = "tampered objective"
        with pytest.raises(ValueError, match="RunBundle does not match"):
            publish_harness_records(
                dataset_name="tampered-bundle-dataset",
                experiment_id=experiment.experiment_id,
                records=[tampered_bundle],
                mlflow_module=mlflow,
            )

        tampered_outputs = copy.deepcopy(record)
        tampered_outputs["outputs"]["analysis"]["decision"] = "stop"
        with pytest.raises(ValueError, match="outputs do not match"):
            publish_harness_records(
                dataset_name="tampered-outputs-dataset",
                experiment_id=experiment.experiment_id,
                records=[tampered_outputs],
                mlflow_module=mlflow,
            )

        tampered_contract = copy.deepcopy(record)
        tampered_contract["inputs"]["task_contract_version"] = "tampered-contract"
        with pytest.raises(ValueError, match="input must match"):
            publish_harness_records(
                dataset_name="tampered-contract-input-dataset",
                experiment_id=experiment.experiment_id,
                records=[tampered_contract],
                mlflow_module=mlflow,
            )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_publish_rejects_unexpected_root_name_or_type(tmp_path):
    bundle = _measured_bundle("invalid-root")
    analysis = _analysis(bundle, "codex")
    trace_tags = {
        f"{PROJECT_LABEL}.bundle_id": bundle.bundle_id,
        f"{PROJECT_LABEL}.group_id": bundle.group_id,
        f"{PROJECT_LABEL}.harness_arm": analysis.arm,
        f"{PROJECT_LABEL}.harness_adapter_id": analysis.adapter_id,
        f"{PROJECT_LABEL}.model": analysis.model,
        f"{PROJECT_LABEL}.evidence_digest": bundle.evidence_digest,
    }
    trace_metadata = {
        f"{PROJECT_LABEL}.task_contract_version": bundle.task_contract_version,
        f"{PROJECT_LABEL}.prompt_sha256": analysis.prompt_sha256,
    }
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.set_experiment("harness-invalid-root-test")
    try:
        for index, (root_name, span_type, error_match) in enumerate(
            (
                ("unexpected_feedback_agent", "AGENT", "unexpected root span"),
                ("hydrogym_feedback_agent", "TASK", "span type AGENT"),
            )
        ):
            with mlflow.start_span(name=root_name, span_type=span_type) as span:
                mlflow.update_current_trace(tags=trace_tags, metadata=trace_metadata)
                span.set_inputs({"run_bundle": bundle.as_dict()})
                span.set_outputs(analysis.dataset_output())
                trace_id = span.trace_id
            record = _single_record(bundle, replace(analysis, trace_id=trace_id))
            with pytest.raises(ValueError, match=error_match):
                publish_harness_records(
                    dataset_name=f"invalid-root-dataset-{index}",
                    experiment_id=experiment.experiment_id,
                    records=[record],
                    mlflow_module=mlflow,
                )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_publish_recomputes_provenance_from_the_canonical_bundle(tmp_path):
    bundle = _measured_bundle("canonical-provenance")
    analysis = _analysis(bundle, "codex")
    base_trace_tags = {
        f"{PROJECT_LABEL}.bundle_id": bundle.bundle_id,
        f"{PROJECT_LABEL}.group_id": bundle.group_id,
        f"{PROJECT_LABEL}.harness_arm": analysis.arm,
        f"{PROJECT_LABEL}.harness_adapter_id": analysis.adapter_id,
        f"{PROJECT_LABEL}.model": analysis.model,
        f"{PROJECT_LABEL}.evidence_digest": bundle.evidence_digest,
    }
    base_trace_metadata = {
        f"{PROJECT_LABEL}.task_contract_version": bundle.task_contract_version,
        f"{PROJECT_LABEL}.prompt_sha256": analysis.prompt_sha256,
    }
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.set_experiment("harness-canonical-provenance-test")
    try:
        for index, record_tag in enumerate(
            (
                "bundle_id",
                "group_id",
                "evidence_digest",
                "task_contract_version",
                "prompt_sha256",
            )
        ):
            forged_value = f"forged-{record_tag}"
            trace_tags = dict(base_trace_tags)
            trace_metadata = dict(base_trace_metadata)
            trace_key = {
                "bundle_id": f"{PROJECT_LABEL}.bundle_id",
                "group_id": f"{PROJECT_LABEL}.group_id",
                "evidence_digest": f"{PROJECT_LABEL}.evidence_digest",
                "task_contract_version": f"{PROJECT_LABEL}.task_contract_version",
                "prompt_sha256": f"{PROJECT_LABEL}.prompt_sha256",
            }[record_tag]
            target = trace_metadata if record_tag in {"task_contract_version", "prompt_sha256"} else trace_tags
            target[trace_key] = forged_value
            with mlflow.start_span(name="hydrogym_feedback_agent", span_type="AGENT") as span:
                mlflow.update_current_trace(tags=trace_tags, metadata=trace_metadata)
                span.set_inputs({"run_bundle": bundle.as_dict()})
                span.set_outputs(analysis.dataset_output())
                trace_id = span.trace_id
            record = _single_record(bundle, replace(analysis, trace_id=trace_id))
            record["tags"][record_tag] = forged_value
            if record_tag == "task_contract_version":
                record["inputs"]["task_contract_version"] = forged_value
            with pytest.raises(ValueError, match=f"record tag {record_tag}"):
                publish_harness_records(
                    dataset_name=f"forged-provenance-dataset-{index}",
                    experiment_id=experiment.experiment_id,
                    records=[record],
                    mlflow_module=mlflow,
                )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_publish_rejects_same_named_dataset_from_another_experiment(tmp_path):
    bundle = _measured_bundle("dataset-experiment")
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    other_experiment_id = mlflow.create_experiment("other-dataset-experiment")
    target_experiment_id = mlflow.create_experiment("target-dataset-experiment")
    dataset_name = "cross-experiment-harness-dataset"
    try:
        mlflow.genai.datasets.create_dataset(
            name=dataset_name,
            experiment_id=other_experiment_id,
        )
        mlflow.set_experiment(experiment_id=target_experiment_id)
        record = _single_record(bundle, _native_analysis(bundle))
        with pytest.raises(ValueError, match="not associated with the target experiment"):
            publish_harness_records(
                dataset_name=dataset_name,
                experiment_id=target_experiment_id,
                records=[record],
                mlflow_module=mlflow,
            )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)
