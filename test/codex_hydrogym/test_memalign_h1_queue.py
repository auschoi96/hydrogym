"""Labeling queue: locked group-disjoint folds, frozen SHA-256 manifest, loud leaks."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, PROJECT_LABEL
from codex_hydrogym.genai.contracts import EvidenceArm, RunBundle
from codex_hydrogym.genai.datasets import REQUIRED_PHYSICS_GATES
from codex_hydrogym.memalign_h1 import H1_SPLIT_SALT, PROTOCOL_ID
from codex_hydrogym.memalign_h1.queue import (
    QUEUE_MANIFEST_SCHEMA_VERSION,
    build_locked_fold_manifest,
    enroll_locked_folds,
    freeze_manifest,
    manifest_digest,
    select_coding_agent_traces,
)


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _measured_bundle(bundle_id, group_id, arm, *, shared_comparator=None):
    arm_id = f"{bundle_id}_{arm}"
    candidate = EvidenceArm(
        arm_id=arm_id,
        run_id=f"measured_run_{arm_id}",
        evidence_kind="measured",
        artifact_ref=f"mlflow://runs/{arm_id}/artifact",
        artifact_sha256=_sha256(f"artifact:{arm_id}"),
        context_fingerprint=_sha256(f"context:{group_id}"),
        controller_kind="signed_modal_feedback",
        uses_observations=True,
        mean_tke=0.7,
        control_effort=0.4,
        physics_gates={gate: True for gate in REQUIRED_PHYSICS_GATES},
        metrics={},
    )
    comparator = shared_comparator or EvidenceArm(
        arm_id=f"{arm_id}_zero",
        run_id=f"measured_run_{arm_id}_zero",
        evidence_kind="measured",
        artifact_ref=f"mlflow://runs/{arm_id}_zero/artifact",
        artifact_sha256=_sha256(f"artifact:{arm_id}_zero"),
        context_fingerprint=_sha256(f"context:{group_id}"),
        controller_kind="zero_open_loop",
        uses_observations=False,
        mean_tke=1.0,
        control_effort=0.0,
        physics_gates={gate: True for gate in REQUIRED_PHYSICS_GATES},
        metrics={},
    )
    return RunBundle(
        bundle_id=bundle_id,
        group_id=group_id,
        task_contract_version="gate0.v1",
        task={"objective": "test H1 queue folding", "claim_allowed": False},
        training={"stage": "locked_fold"},
        candidate=candidate,
        comparators=(comparator,),
        diagnostics=("measured fixture for offline queue tests.",),
        artifact_refs=(candidate.artifact_ref, comparator.artifact_ref),
    )


def _record(bundle, arm, trace_id):
    return {
        "trace_id": trace_id,
        "bundle": bundle,
        "bundle_id": bundle.bundle_id,
        "group_id": bundle.group_id,
        "arm": arm,
        "evidence_digest": bundle.evidence_digest,
    }


def _fixture_bundles(group_count=8):
    bundles = []
    for index in range(1, group_count + 1):
        group = f"h1_group_{index:02d}"
        bundles.append(_measured_bundle(f"bundle_{group}_codex", group, "codex"))
        bundles.append(_measured_bundle(f"bundle_{group}_claude", group, "claude"))
    return bundles


def _candidate_records(bundles):
    return [
        _record(bundle, "claude" if bundle.bundle_id.endswith("_claude") else "codex", f"trace_{bundle.bundle_id}")
        for bundle in bundles
    ]


def _tagged_trace(bundle, arm, trace_id, *, labeled=False):
    assessments = []
    if labeled:
        assessments.append(
            SimpleNamespace(
                name=CRITIC_QUALITY_ASSESSMENT_NAME,
                source=SimpleNamespace(source_type="HUMAN", source_id="panel@example.com"),
            )
        )
    return SimpleNamespace(
        info=SimpleNamespace(
            trace_id=trace_id,
            tags={
                f"{PROJECT_LABEL}.bundle_id": bundle.bundle_id,
                f"{PROJECT_LABEL}.group_id": bundle.group_id,
                f"{PROJECT_LABEL}.harness_arm": arm,
                f"{PROJECT_LABEL}.evidence_kind": "measured",
                f"{PROJECT_LABEL}.evidence_digest": bundle.evidence_digest,
            },
            assessments=assessments,
        ),
        data=SimpleNamespace(
            spans=[
                SimpleNamespace(
                    parent_id=None,
                    name="hydrogym_feedback_agent",
                    span_type="AGENT",
                    inputs={"run_bundle": bundle.as_dict()},
                )
            ]
        ),
    )


class _FakeMlflow:
    def __init__(self, traces):
        self.by_id = {trace.info.trace_id: trace for trace in traces}
        self.tag_calls = []
        self.search_kwargs = None

    def search_traces(self, **kwargs):
        self.search_kwargs = kwargs
        return list(self.by_id.values())

    def get_trace(self, trace_id):
        return self.by_id.get(trace_id)

    def set_trace_tag(self, **kwargs):
        self.tag_calls.append(kwargs)


def test_selection_skips_non_runnable_and_already_labeled_traces_with_reasons():
    bundles = _fixture_bundles(2)
    codex_bundle, claude_bundle = bundles[0], bundles[1]
    selectable = _tagged_trace(codex_bundle, "codex", "trace-sel-1")
    already_labeled = _tagged_trace(claude_bundle, "claude", "trace-sel-2", labeled=True)
    no_bundle_input = SimpleNamespace(
        info=SimpleNamespace(
            trace_id="trace-ppo-patch",
            tags={
                f"{PROJECT_LABEL}.bundle_id": codex_bundle.bundle_id,
                f"{PROJECT_LABEL}.group_id": codex_bundle.group_id,
                f"{PROJECT_LABEL}.harness_arm": "codex",
                f"{PROJECT_LABEL}.evidence_kind": "measured",
                f"{PROJECT_LABEL}.evidence_digest": codex_bundle.evidence_digest,
            },
            assessments=[],
        ),
        data=SimpleNamespace(
            spans=[
                SimpleNamespace(
                    parent_id=None,
                    name="coding_agent_patch",
                    span_type="AGENT",
                    inputs={"task": {"task_id": "train_something"}},
                )
            ]
        ),
    )
    wrong_arm = SimpleNamespace(
        info=SimpleNamespace(
            trace_id="trace-other-arm",
            tags={f"{PROJECT_LABEL}.harness_arm": "solver"},
            assessments=[],
        ),
        data=SimpleNamespace(spans=[]),
    )
    mlflow = _FakeMlflow([selectable, already_labeled, no_bundle_input, wrong_arm])

    selection = select_coding_agent_traces(experiment_id="123", mlflow_module=mlflow)

    assert selection["counts"]["searched"] == 4
    assert selection["counts"]["selected"] == 1
    assert selection["candidates"][0]["trace_id"] == "trace-sel-1"
    reasons = [row["reason"] for row in selection["skipped"]]
    assert any("already has an adjudicated" in reason for reason in reasons)
    assert any("no parseable canonical RunBundle" in reason for reason in reasons)
    assert any("harness-arm" in reason for reason in reasons)


def test_selection_output_feeds_manifest_without_hand_editing():
    bundles = _fixture_bundles(8)
    traces = [
        _tagged_trace(
            bundle,
            "claude" if bundle.bundle_id.endswith("_claude") else "codex",
            f"trace_{bundle.bundle_id}",
        )
        for bundle in bundles
    ]
    selection = select_coding_agent_traces(experiment_id="123", mlflow_module=_FakeMlflow(traces))

    manifest = build_locked_fold_manifest(records=selection["candidates"])

    assert manifest["counts"]["traces"] == 16
    assert manifest["counts"]["heldout_groups"] == 4


def test_locked_manifest_is_group_disjoint_ordered_and_digest_frozen():
    bundles = _fixture_bundles(8)
    records = _candidate_records(bundles)

    manifest = build_locked_fold_manifest(records=records, test_group_count=4, split_salt=H1_SPLIT_SALT)

    assert manifest["schema_version"] == QUEUE_MANIFEST_SCHEMA_VERSION
    assert manifest["protocol_id"] == PROTOCOL_ID
    assert manifest["counts"]["groups"] == 8
    assert manifest["counts"]["train_groups"] + manifest["counts"]["heldout_groups"] == 8
    assert manifest["counts"]["traces"] == 16
    assert manifest["counts"]["train_traces"] + manifest["counts"]["heldout_traces"] == 16
    assert manifest["counts"]["heldout_groups"] == 4
    assert manifest["digest"] == manifest_digest(manifest)

    fold_by_group = manifest["fold_by_group"]
    assert set(fold_by_group.values()) <= {"train", "heldout"}
    group_to_rows = {}
    for row in manifest["rows"]:
        assert row["fold"] == fold_by_group[row["group_id"]]
        group_to_rows.setdefault(row["group_id"], []).append(row)
    assert all(len({row["arm"] for row in rows}) == 2 for rows in group_to_rows.values())

    row_keys = [(row["fold"], row["group_id"], row["bundle_id"], row["arm"]) for row in manifest["rows"]]
    assert row_keys == sorted(row_keys)

    re_manifest = build_locked_fold_manifest(records=records, test_group_count=4, split_salt=H1_SPLIT_SALT)
    assert re_manifest["digest"] == manifest["digest"]

    mutated = json.loads(json.dumps(manifest))
    mutated["rows"][0]["fold"] = "heldout" if mutated["rows"][0]["fold"] == "train" else "train"
    assert manifest_digest(mutated) != manifest["digest"]


def test_manifest_rejects_artifact_reuse_across_groups_loudly():
    shared_comparator = EvidenceArm(
        arm_id="shared_zero",
        run_id="measured_run_shared_zero",
        evidence_kind="measured",
        artifact_ref="mlflow://runs/shared_zero/artifact",
        artifact_sha256=_sha256("artifact:shared_zero"),
        context_fingerprint=_sha256("context:shared"),
        controller_kind="zero_open_loop",
        uses_observations=False,
        mean_tke=1.0,
        control_effort=0.0,
        physics_gates={gate: True for gate in REQUIRED_PHYSICS_GATES},
        metrics={},
    )
    bundles = []
    for index in range(1, 6):
        group = f"h1_group_{index:02d}"
        bundles.append(
            _measured_bundle(f"bundle_{group}_codex", group, "codex", shared_comparator=shared_comparator)
        )
        bundles.append(_measured_bundle(f"bundle_{group}_claude", group, "claude"))

    with pytest.raises(ValueError, match="same run artifact"):
        build_locked_fold_manifest(records=_candidate_records(bundles), test_group_count=4)


def test_manifest_rejects_declared_identity_mismatch():
    bundles = _fixture_bundles(5)
    records = _candidate_records(bundles)
    records[0]["group_id"] = "h1_group_other"

    with pytest.raises(ValueError, match="must match its RunBundle"):
        build_locked_fold_manifest(records=records, test_group_count=4)


def test_manifest_enforces_frozen_group_count_and_complete_arms():
    with pytest.raises(ValueError, match="frozen held-out group count"):
        build_locked_fold_manifest(records=_candidate_records(_fixture_bundles(8)), test_group_count=3)
    incomplete = _candidate_records(_fixture_bundles(5))[:-1]
    with pytest.raises(ValueError, match="exactly one trace for each harness arm"):
        build_locked_fold_manifest(records=incomplete)


def test_enroll_tags_locked_folds_and_rechecks_the_frozen_digest(tmp_path):
    bundles = _fixture_bundles(8)
    records = _candidate_records(bundles)
    manifest = build_locked_fold_manifest(records=records, test_group_count=4, split_salt=H1_SPLIT_SALT)

    traces = [_tagged_trace(record["bundle"], record["arm"], record["trace_id"]) for record in records]
    mlflow = _FakeMlflow(traces)
    frozen_record = freeze_manifest(manifest=manifest, path=tmp_path / "manifest.freeze.json")
    outcome = enroll_locked_folds(manifest=manifest, frozen_record=frozen_record, mlflow_module=mlflow)

    assert outcome["count"] == 16
    assert outcome["review_state"] == "pending_adjudication"
    assert len(mlflow.tag_calls) == 48
    fold_by_trace = {row["trace_id"]: row["fold"] for row in manifest["rows"]}
    for call in mlflow.tag_calls:
        expected_tag = "train" if fold_by_trace[call["trace_id"]] == "train" else "test"
        if call["key"] == f"{PROJECT_LABEL}.critic_fold":
            assert call["value"] == expected_tag

    mutated = json.loads(json.dumps(manifest))
    target_group = next(iter(mutated["fold_by_group"]))
    mutated["fold_by_group"][target_group] = (
        "heldout" if mutated["fold_by_group"][target_group] == "train" else "train"
    )
    mutated["digest"] = manifest_digest(mutated)
    with pytest.raises(ValueError, match="externally frozen digest"):
        enroll_locked_folds(
            manifest=mutated,
            frozen_record=frozen_record,
            mlflow_module=_FakeMlflow(traces),
        )

    with pytest.raises(FileExistsError):
        freeze_manifest(manifest=manifest, path=tmp_path / "manifest.freeze.json")
