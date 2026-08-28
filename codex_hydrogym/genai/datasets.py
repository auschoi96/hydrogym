"""Leakage-resistant scenario data for the codex_hydrogym outer loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
from typing import Any, Iterable, Mapping, Sequence

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, PROJECT_LABEL
from codex_hydrogym.genai.contracts import (
    REQUIRED_PHYSICS_GATES,
    EvidenceArm,
    RunBundle,
    build_gepa_record,
    parse_run_bundle,
)
from codex_hydrogym.genai.harnesses import (
    HARNESS_ADAPTER_ARMS,
    HARNESS_ADAPTER_IDS,
    HARNESS_ARMS,
    prompt_digest,
    render_feedback_prompt,
)
from codex_hydrogym.genai.tracing import HarnessAnalysis


@dataclass(frozen=True)
class FluidScenario:
    scenario_id: str
    reynolds_number: int
    seed: int
    control_budget: str
    split: str
    objective: str = "suppress held-out mean TKE without disproportionate control effort"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture_arm(
    *,
    case: str,
    arm_id: str,
    controller_kind: str,
    uses_observations: bool,
    mean_tke: float,
    control_effort: float,
    context: str,
    evidence_kind: str = "synthetic_contract",
    failed_gate: str | None = None,
    artifact_sha256: str | None = None,
    recorded_gates: Mapping[str, bool] | None = None,
) -> EvidenceArm:
    gates = dict.fromkeys(REQUIRED_PHYSICS_GATES, True) if recorded_gates is None else dict(recorded_gates)
    if failed_gate is not None:
        gates[failed_gate] = False
    scheme = "diagnostic" if evidence_kind == "unverified_diagnostic" else "synthetic"
    artifact_ref = f"{scheme}://{case}/{arm_id}"
    return EvidenceArm(
        arm_id=arm_id,
        run_id=f"{scheme}_{case}_{arm_id}",
        evidence_kind=evidence_kind,
        artifact_ref=artifact_ref,
        artifact_sha256=artifact_sha256 or _digest(artifact_ref),
        context_fingerprint=_digest(context),
        controller_kind=controller_kind,
        uses_observations=uses_observations,
        mean_tke=mean_tke,
        control_effort=control_effort,
        physics_gates=gates,
        metrics={},
    )


def build_harness_sanity_bundles() -> tuple[RunBundle, ...]:
    """Return five non-claiming Gate-0 cases; none may enter MemAlign training."""
    full_task = {
        "objective": "test whether state-conditioned feedback beats observation-free control",
        "evidence_status": "contract_only",
        "claim_allowed": False,
    }
    training = {
        "stage": "gate0_cpu_before_ppo",
        "budget_locked": True,
        "note": "sanity data cannot authorize compute or support a fluid-improvement claim",
    }

    diagnostic_context = "sibling-f5eadec-dt0.002-seed0-unverified"
    diagnostic_sha = "2c43041be0c1ad93f1fa1b0cb7523e86dfbb09b92af995be65397dd8aa0c69ce"
    constant = _fixture_arm(
        case="open_loop_alias",
        arm_id="constant_oppose",
        controller_kind="constant_open_loop",
        uses_observations=False,
        mean_tke=1.2599737644,
        control_effort=0.5,
        context=diagnostic_context,
        evidence_kind="unverified_diagnostic",
        artifact_sha256=diagnostic_sha,
        recorded_gates={"finite_state_and_metrics": True},
    )
    zero = _fixture_arm(
        case="open_loop_alias",
        arm_id="zero_control",
        controller_kind="zero_open_loop",
        uses_observations=False,
        mean_tke=3.1696414948,
        control_effort=0.0,
        context=diagnostic_context,
        evidence_kind="unverified_diagnostic",
        artifact_sha256=diagnostic_sha,
        recorded_gates={"finite_state_and_metrics": True},
    )

    bundles = [
        RunBundle(
            bundle_id="sanity_open_loop_alias",
            group_id="sanity_group_open_loop_alias",
            task_contract_version="gate0.v1",
            task={**full_task, "failure_mode": "forcing and constant actuator share the same mode"},
            training={**training, "source_commit": "f5eadecb2de4dc2812c7988adca5ce8bfd6b51e6"},
            candidate=constant,
            comparators=(zero,),
            diagnostics=("No raw execution log or MLflow run survives; this is diagnostic context only.",),
            artifact_refs=(constant.artifact_ref,),
        )
    ]

    cross_candidate = _fixture_arm(
        case="cross_context",
        arm_id="feedback_candidate",
        controller_kind="signed_modal_feedback",
        uses_observations=True,
        mean_tke=0.70,
        control_effort=0.4,
        context="cross-context-candidate",
    )
    cross_constant = _fixture_arm(
        case="cross_context",
        arm_id="constant_frontier",
        controller_kind="constant_open_loop",
        uses_observations=False,
        mean_tke=1.00,
        control_effort=0.4,
        context="cross-context-comparator",
    )
    bundles.append(
        RunBundle(
            bundle_id="sanity_cross_context_false_win",
            group_id="sanity_group_cross_context",
            task_contract_version="gate0.v1",
            task={**full_task, "failure_mode": "candidate and comparator contexts differ"},
            training=training,
            candidate=cross_candidate,
            comparators=(cross_constant,),
            diagnostics=("The lower candidate TKE is not a comparable result.",),
            artifact_refs=(cross_candidate.artifact_ref, cross_constant.artifact_ref),
        )
    )

    failed_candidate = _fixture_arm(
        case="failed_physics",
        arm_id="failed_candidate",
        controller_kind="signed_modal_feedback",
        uses_observations=True,
        mean_tke=0.60,
        control_effort=0.4,
        context="failed-physics-shared",
        failed_gate="spectral_tail_controlled",
    )
    failed_constant = _fixture_arm(
        case="failed_physics",
        arm_id="valid_constant",
        controller_kind="constant_open_loop",
        uses_observations=False,
        mean_tke=1.00,
        control_effort=0.4,
        context="failed-physics-shared",
    )
    bundles.append(
        RunBundle(
            bundle_id="sanity_failed_physics_false_win",
            group_id="sanity_group_failed_physics",
            task_contract_version="gate0.v1",
            task={**full_task, "failure_mode": "lower TKE with a failed physics gate"},
            training=training,
            candidate=failed_candidate,
            comparators=(failed_constant,),
            diagnostics=("spectral_tail_controlled is false for the candidate.",),
            artifact_refs=(failed_candidate.artifact_ref, failed_constant.artifact_ref),
        )
    )

    shuffle_candidate = _fixture_arm(
        case="shuffle_null",
        arm_id="observed_candidate",
        controller_kind="signed_modal_feedback",
        uses_observations=True,
        mean_tke=0.700,
        control_effort=0.4,
        context="shuffle-null-shared",
    )
    shuffled = _fixture_arm(
        case="shuffle_null",
        arm_id="shuffled_observations",
        controller_kind="shuffled_observation_ablation",
        uses_observations=True,
        mean_tke=0.701,
        control_effort=0.4,
        context="shuffle-null-shared",
    )
    shuffle_constant = _fixture_arm(
        case="shuffle_null",
        arm_id="constant_frontier",
        controller_kind="constant_open_loop",
        uses_observations=False,
        mean_tke=1.000,
        control_effort=0.4,
        context="shuffle-null-shared",
    )
    bundles.append(
        RunBundle(
            bundle_id="sanity_shuffled_observation_null",
            group_id="sanity_group_shuffle_null",
            task_contract_version="gate0.v1",
            task={**full_task, "failure_mode": "observation shuffling preserves the apparent gain"},
            training=training,
            candidate=shuffle_candidate,
            comparators=(shuffled, shuffle_constant),
            diagnostics=("Candidate and shuffled-observation TKE differ by only 0.001.",),
            artifact_refs=(
                shuffle_candidate.artifact_ref,
                shuffled.artifact_ref,
                shuffle_constant.artifact_ref,
            ),
        )
    )

    positive_candidate = _fixture_arm(
        case="positive_pattern",
        arm_id="observed_candidate",
        controller_kind="signed_modal_feedback",
        uses_observations=True,
        mean_tke=0.70,
        control_effort=0.4,
        context="positive-pattern-shared",
    )
    positive_shuffled = _fixture_arm(
        case="positive_pattern",
        arm_id="shuffled_observations",
        controller_kind="shuffled_observation_ablation",
        uses_observations=True,
        mean_tke=0.95,
        control_effort=0.4,
        context="positive-pattern-shared",
    )
    positive_constant = _fixture_arm(
        case="positive_pattern",
        arm_id="constant_frontier",
        controller_kind="constant_open_loop",
        uses_observations=False,
        mean_tke=1.00,
        control_effort=0.4,
        context="positive-pattern-shared",
    )
    oracle = _fixture_arm(
        case="positive_pattern",
        arm_id="privileged_phase_oracle",
        controller_kind="privileged_oracle",
        uses_observations=True,
        mean_tke=0.60,
        control_effort=0.4,
        context="positive-pattern-shared",
    )
    bundles.append(
        RunBundle(
            bundle_id="sanity_positive_causal_pattern",
            group_id="sanity_group_positive_pattern",
            task_contract_version="gate0.v1",
            task={**full_task, "failure_mode": "positive control remains synthetic and non-actionable"},
            training=training,
            candidate=positive_candidate,
            comparators=(positive_shuffled, positive_constant, oracle),
            diagnostics=("The relationships mimic a credible causal pattern, but every arm is synthetic.",),
            artifact_refs=(
                positive_candidate.artifact_ref,
                positive_shuffled.artifact_ref,
                positive_constant.artifact_ref,
                oracle.artifact_ref,
            ),
        )
    )
    return tuple(bundles)


def build_scenario_matrix() -> tuple[FluidScenario, ...]:
    """Return 24 deterministic scenarios; entire seeds belong to one split.

    Grouping by seed prevents the same stochastic trajectory family from appearing
    in both GEPA training and held-out validation.
    """
    scenarios = []
    for reynolds_number in (100, 200, 400):
        for seed in (11, 23, 37, 41):
            split = "validation" if seed == 41 else "train"
            for control_budget in ("conservative", "balanced"):
                scenarios.append(
                    FluidScenario(
                        scenario_id=(f"{PROJECT_LABEL}_re{reynolds_number}_seed{seed}_{control_budget}"),
                        reynolds_number=reynolds_number,
                        seed=seed,
                        control_budget=control_budget,
                        split=split,
                    )
                )
    return tuple(scenarios)


def gepa_scenario_records(scenarios: Iterable[FluidScenario] | None = None) -> list[dict[str, Any]]:
    """Create prompt-optimization records with both required top-level fields."""
    selected = build_scenario_matrix() if scenarios is None else tuple(scenarios)
    expected_behavior = (
        "Return exactly one codex_hydrogym.reward_candidate.v1 JSON object. Keep every scalar inside the "
        "published bounds, propose no executable code or solver mutation, state a falsifiable fluid-dynamics "
        "hypothesis, and treat all language-model judgments as advisory until a held-out PPO rollout passes "
        "every deterministic physics gate."
    )
    return [
        build_gepa_record(scenario=scenario.as_dict(), expected_behavior=expected_behavior)
        for scenario in selected
        if scenario.split == "train"
    ]


def grouped_bundle_split(
    bundles: Sequence[RunBundle],
    *,
    test_group_count: int,
    split_salt: str = "codex_hydrogym_harness_v1",
) -> dict[str, str]:
    """Assign whole ``group_id`` values to a deterministic train/test split."""
    if not bundles:
        raise ValueError("at least one run bundle is required")
    bundle_ids = [bundle.bundle_id for bundle in bundles]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("bundle_id values must be unique")
    groups = sorted({bundle.group_id for bundle in bundles})
    if any(
        arm.evidence_kind != "measured"
        for bundle in bundles
        for arm in (bundle.candidate, *bundle.comparators)
    ):
        raise ValueError("train/test splits may contain only measured evidence")
    provenance_groups: dict[tuple[str, str], set[str]] = {}
    for bundle in bundles:
        for arm in (bundle.candidate, *bundle.comparators):
            provenance_groups.setdefault((arm.run_id, arm.artifact_sha256), set()).add(bundle.group_id)
    if any(len(group_ids) > 1 for group_ids in provenance_groups.values()):
        raise ValueError("the same run artifact cannot appear under multiple group_id values")
    if isinstance(test_group_count, bool) or not isinstance(test_group_count, int):
        raise TypeError("test_group_count must be an integer")
    if not 1 <= test_group_count < len(groups):
        raise ValueError("test_group_count must leave at least one train and one test group")
    if not isinstance(split_salt, str) or not split_salt.strip():
        raise ValueError("split_salt must be non-empty")
    ranked_groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{split_salt}:{group}".encode("utf-8")).hexdigest(),
    )
    test_groups = set(ranked_groups[:test_group_count])
    return {
        bundle.bundle_id: "test" if bundle.group_id in test_groups else "train"
        for bundle in bundles
    }


def blinded_case_id(*, bundle: RunBundle, arm: str, blinding_salt: str) -> str:
    """Create an opaque arm-specific input key without putting the arm in judge-visible text."""
    if arm not in HARNESS_ARMS:
        raise ValueError(f"arm must be one of {HARNESS_ARMS}")
    if not isinstance(blinding_salt, str) or not blinding_salt.strip():
        raise ValueError("blinding_salt must be non-empty")
    payload = f"{blinding_salt}:{bundle.evidence_digest}:{arm}"
    return f"case_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def blinded_adapter_case_id(*, bundle: RunBundle, adapter_id: str, blinding_salt: str) -> str:
    """Create an opaque adapter-specific key without exposing adapter identity to a judge."""
    if adapter_id not in HARNESS_ADAPTER_ARMS:
        raise ValueError(f"adapter_id must be one of {HARNESS_ADAPTER_IDS}")
    if not isinstance(blinding_salt, str) or not blinding_salt.strip():
        raise ValueError("blinding_salt must be non-empty")
    payload = f"{blinding_salt}:{bundle.evidence_digest}:{adapter_id}"
    return f"case_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _validate_analysis_provenance(analysis: HarnessAnalysis) -> None:
    if analysis.arm not in HARNESS_ARMS:
        raise ValueError(f"analysis arm must be one of {HARNESS_ARMS}")
    if analysis.adapter_id not in HARNESS_ADAPTER_ARMS:
        raise ValueError(f"analysis adapter_id must be one of {HARNESS_ADAPTER_IDS}")
    if HARNESS_ADAPTER_ARMS[analysis.adapter_id] != analysis.arm:
        raise ValueError("analysis adapter_id does not belong to its declared harness arm")


def _harness_record(
    *,
    bundle: RunBundle,
    analysis: HarnessAnalysis,
    case_id: str,
    fold: str,
    prompt_version: str,
    expectations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one record while preserving its originating MLflow trace."""
    _validate_analysis_provenance(analysis)
    if not isinstance(analysis.trace_id, str) or not analysis.trace_id.strip():
        raise ValueError("analysis trace_id must be non-empty for dataset lineage")
    trace_id = analysis.trace_id.strip()
    tags = {
        "project": PROJECT_LABEL,
        "bundle_id": bundle.bundle_id,
        "group_id": bundle.group_id,
        "arm": analysis.arm,
        "adapter_id": analysis.adapter_id,
        "fold": fold,
        "model": analysis.model,
        "prompt_version": prompt_version,
        "prompt_sha256": analysis.prompt_sha256,
        "evidence_digest": bundle.evidence_digest,
        "task_contract_version": bundle.task_contract_version,
        "latency_ms": f"{analysis.latency_ms:.3f}",
        "source_trace_id": trace_id,
    }
    total_cost = analysis.runtime_metadata.get("total_cost_usd")
    if isinstance(total_cost, (int, float)) and not isinstance(total_cost, bool):
        tags["total_cost_usd"] = str(float(total_cost))
    record: dict[str, Any] = {
        "inputs": {
            "case_id": case_id,
            "run_bundle": bundle.as_dict(),
            "task_contract_version": bundle.task_contract_version,
        },
        "outputs": analysis.dataset_output(),
        "tags": tags,
        "source": {
            "source_type": "TRACE",
            "source_data": {"trace_id": trace_id},
        },
    }
    if expectations is not None:
        materialized_expectations = dict(expectations)
        if not materialized_expectations:
            raise ValueError("arm expectations must not be empty")
        if CRITIC_QUALITY_ASSESSMENT_NAME in materialized_expectations:
            raise ValueError(
                "gold expectations must use critic_quality_gold; critic_quality is reserved for Feedback"
            )
        if "critic_quality_gold" not in materialized_expectations:
            raise ValueError("adjudicated expectations must contain critic_quality_gold")
        gold = materialized_expectations["critic_quality_gold"]
        if isinstance(gold, bool) or not isinstance(gold, int) or not 1 <= gold <= 5:
            raise ValueError("critic_quality_gold must be an integer in [1, 5]")
        record["expectations"] = materialized_expectations
    return record


def paired_harness_records(
    *,
    bundle: RunBundle,
    analyses: Sequence[HarnessAnalysis],
    fold: str,
    prompt_version: str,
    blinding_salt: str,
    expectations_by_arm: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build two blinded native-MLflow records that cannot deduplicate each other."""
    for analysis in analyses:
        _validate_analysis_provenance(analysis)
    by_arm = {analysis.arm: analysis for analysis in analyses}
    if set(by_arm) != set(HARNESS_ARMS) or len(analyses) != len(HARNESS_ARMS):
        raise ValueError("analyses must contain exactly one Codex arm and one Claude arm")
    prompt_digests = {analysis.prompt_sha256 for analysis in analyses}
    if len(prompt_digests) != 1:
        raise ValueError("paired arms must use the byte-identical prompt")
    if fold not in {"sanity", "train", "test"}:
        raise ValueError("fold must be sanity, train, or test")
    if fold != "sanity" and any(
        arm.evidence_kind != "measured" for arm in (bundle.candidate, *bundle.comparators)
    ):
        raise ValueError("synthetic and unverified evidence is restricted to the sanity fold")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("prompt_version must be non-empty")
    if expectations_by_arm is not None and set(expectations_by_arm) != set(HARNESS_ARMS):
        raise ValueError("expectations_by_arm must contain exactly the Codex and Claude arms")

    records = []
    for arm in HARNESS_ARMS:
        analysis = by_arm[arm]
        case_id = blinded_case_id(bundle=bundle, arm=arm, blinding_salt=blinding_salt)
        records.append(
            _harness_record(
                bundle=bundle,
                analysis=analysis,
                case_id=case_id,
                fold=fold,
                prompt_version=prompt_version,
                expectations=expectations_by_arm[arm] if expectations_by_arm is not None else None,
            )
        )

    if records[0]["inputs"] == records[1]["inputs"]:
        raise AssertionError("paired MLflow records must have distinct inputs")
    if len({record["source"]["source_data"]["trace_id"] for record in records}) != len(records):
        raise ValueError("each paired harness analysis must have a distinct trace_id")
    return records


def harness_screen_records(
    *,
    bundle: RunBundle,
    analyses: Sequence[HarnessAnalysis],
    fold: str,
    prompt_version: str,
    blinding_salt: str,
) -> list[dict[str, Any]]:
    """Build one blinded TRACE-sourced record for each of the four harness adapters."""
    for analysis in analyses:
        _validate_analysis_provenance(analysis)
    by_adapter = {analysis.adapter_id: analysis for analysis in analyses}
    if set(by_adapter) != set(HARNESS_ADAPTER_IDS) or len(analyses) != len(HARNESS_ADAPTER_IDS):
        raise ValueError("analyses must contain exactly one result for each harness adapter")
    prompt_digests = {analysis.prompt_sha256 for analysis in analyses}
    if len(prompt_digests) != 1:
        raise ValueError("screen adapters must use the byte-identical prompt")
    if fold not in {"sanity", "train", "test"}:
        raise ValueError("fold must be sanity, train, or test")
    if fold != "sanity" and any(
        arm.evidence_kind != "measured" for arm in (bundle.candidate, *bundle.comparators)
    ):
        raise ValueError("synthetic and unverified evidence is restricted to the sanity fold")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("prompt_version must be non-empty")

    records = [
        _harness_record(
            bundle=bundle,
            analysis=by_adapter[adapter_id],
            case_id=blinded_adapter_case_id(
                bundle=bundle,
                adapter_id=adapter_id,
                blinding_salt=blinding_salt,
            ),
            fold=fold,
            prompt_version=prompt_version,
        )
        for adapter_id in HARNESS_ADAPTER_IDS
    ]
    if len({record["inputs"]["case_id"] for record in records}) != len(records):
        raise AssertionError("screen MLflow records must have distinct inputs")
    if len({record["source"]["source_data"]["trace_id"] for record in records}) != len(records):
        raise ValueError("each harness adapter analysis must have a distinct trace_id")
    return records


def publish_harness_records(
    *,
    dataset_name: str,
    experiment_id: str,
    records: Sequence[Mapping[str, Any]],
    mlflow_module=None,
):
    """Merge harness records through MLflow's native managed-dataset API."""
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be non-empty")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    experiment_id = experiment_id.strip()
    materialized = [dict(record) for record in records]
    if not materialized:
        raise ValueError("at least one record is required")
    mlflow = mlflow_module or importlib.import_module("mlflow")
    trace_ids: list[str] = []
    for record in materialized:
        source = record.get("source")
        if not isinstance(source, Mapping) or source.get("source_type") != "TRACE":
            raise ValueError("every harness record must have a top-level TRACE source")
        source_data = source.get("source_data")
        trace_id = source_data.get("trace_id") if isinstance(source_data, Mapping) else None
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("every harness TRACE source must contain a non-empty trace_id")
        trace_id = trace_id.strip()
        tags = record.get("tags")
        if not isinstance(tags, Mapping) or tags.get("source_trace_id") != trace_id:
            raise ValueError("harness record source_trace_id tag must match its TRACE source")
        trace_ids.append(trace_id)
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("every published harness record must reference a distinct trace_id")
    trace_tag_fields = {
        f"{PROJECT_LABEL}.bundle_id": "bundle_id",
        f"{PROJECT_LABEL}.group_id": "group_id",
        f"{PROJECT_LABEL}.harness_arm": "arm",
        f"{PROJECT_LABEL}.harness_adapter_id": "adapter_id",
        f"{PROJECT_LABEL}.model": "model",
        f"{PROJECT_LABEL}.evidence_digest": "evidence_digest",
    }
    trace_metadata_fields = {
        f"{PROJECT_LABEL}.task_contract_version": "task_contract_version",
        f"{PROJECT_LABEL}.prompt_sha256": "prompt_sha256",
    }
    for record, trace_id in zip(materialized, trace_ids, strict=True):
        try:
            trace = mlflow.get_trace(trace_id)
        except Exception as error:
            raise ValueError(
                f"harness source trace is not readable from the active tracking store: {trace_id}"
            ) from error
        if trace is None:
            raise ValueError(f"harness source trace does not exist in the active tracking store: {trace_id}")
        trace_info = getattr(trace, "info", None)
        trace_location = getattr(trace_info, "trace_location", None)
        mlflow_experiment = getattr(trace_location, "mlflow_experiment", None)
        source_experiment_id = getattr(mlflow_experiment, "experiment_id", None)
        if source_experiment_id != experiment_id:
            raise ValueError(
                "harness source trace does not belong to the target experiment: "
                f"{trace_id} belongs to {source_experiment_id!r}, expected {experiment_id!r}"
            )
        trace_state = getattr(trace_info, "state", None)
        if getattr(trace_state, "value", trace_state) != "OK":
            raise ValueError(f"harness source trace must have state OK: {trace_id}")

        record_tags = record.get("tags")
        if not isinstance(record_tags, Mapping):
            raise ValueError("every harness record must contain provenance tags")
        trace_tags = getattr(trace_info, "tags", None)
        trace_metadata = getattr(trace_info, "trace_metadata", None)
        if not isinstance(trace_tags, Mapping) or not isinstance(trace_metadata, Mapping):
            raise ValueError(f"harness source trace is missing provenance: {trace_id}")
        for trace_key, record_key in trace_tag_fields.items():
            expected = record_tags.get(record_key)
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError(f"harness record tag {record_key} must be non-empty")
            if trace_tags.get(trace_key) != expected:
                raise ValueError(
                    f"harness source trace provenance does not match record tag {record_key}: {trace_id}"
                )
        for trace_key, record_key in trace_metadata_fields.items():
            expected = record_tags.get(record_key)
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError(f"harness record tag {record_key} must be non-empty")
            if trace_metadata.get(trace_key) != expected:
                raise ValueError(
                    f"harness source trace provenance does not match record tag {record_key}: {trace_id}"
                )

        spans = getattr(getattr(trace, "data", None), "spans", None)
        if not isinstance(spans, Sequence):
            raise ValueError(f"harness source trace has no auditable spans: {trace_id}")
        roots = [span for span in spans if getattr(span, "parent_id", None) is None]
        if len(roots) != 1:
            raise ValueError(f"harness source trace must contain exactly one root span: {trace_id}")
        root = roots[0]
        if getattr(root, "name", None) != "hydrogym_feedback_agent":
            raise ValueError(f"harness source trace has an unexpected root span: {trace_id}")
        if getattr(root, "span_type", None) != "AGENT":
            raise ValueError(f"harness source trace root must have span type AGENT: {trace_id}")

        record_inputs = record.get("inputs")
        if not isinstance(record_inputs, Mapping):
            raise ValueError("every harness record must contain inputs")
        record_bundle = record_inputs.get("run_bundle")
        root_inputs = getattr(root, "inputs", None)
        if (
            not isinstance(record_bundle, Mapping)
            or not isinstance(root_inputs, Mapping)
            or root_inputs.get("run_bundle") != record_bundle
        ):
            raise ValueError(f"harness source trace RunBundle does not match the record: {trace_id}")
        if record_inputs.get("task_contract_version") != record_tags["task_contract_version"]:
            raise ValueError("harness record task_contract_version input must match its provenance tag")
        try:
            canonical_bundle = parse_run_bundle(record_bundle)
        except (TypeError, ValueError) as error:
            raise ValueError("harness record must contain a valid canonical RunBundle") from error
        derived_provenance = {
            "bundle_id": canonical_bundle.bundle_id,
            "group_id": canonical_bundle.group_id,
            "evidence_digest": canonical_bundle.evidence_digest,
            "task_contract_version": canonical_bundle.task_contract_version,
            "prompt_sha256": prompt_digest(render_feedback_prompt(canonical_bundle)),
        }
        for record_key, derived_value in derived_provenance.items():
            if record_tags[record_key] != derived_value:
                raise ValueError(
                    f"harness record tag {record_key} does not match its canonical RunBundle"
                )
        if getattr(root, "outputs", None) != record.get("outputs"):
            raise ValueError(f"harness source trace outputs do not match the record: {trace_id}")
    datasets = mlflow.genai.datasets
    try:
        dataset = datasets.get_dataset(name=dataset_name)
    except Exception as error:
        if getattr(error, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
            raise
        dataset = datasets.create_dataset(name=dataset_name, experiment_id=experiment_id)
    if getattr(dataset, "name", None) != dataset_name:
        raise ValueError("MLflow returned a harness dataset with an unexpected name")
    try:
        dataset_experiment_ids = dataset.experiment_ids
    except NotImplementedError:
        # Databricks managed datasets use their fully qualified UC table name as
        # identity; MLflow's public wrapper does not expose experiment links.
        dataset_experiment_ids = None
    if dataset_experiment_ids is not None:
        if (
            not isinstance(dataset_experiment_ids, Sequence)
            or isinstance(dataset_experiment_ids, (str, bytes))
            or experiment_id not in {str(value) for value in dataset_experiment_ids}
        ):
            raise ValueError("existing harness dataset is not associated with the target experiment")
    return dataset.merge_records(materialized)
