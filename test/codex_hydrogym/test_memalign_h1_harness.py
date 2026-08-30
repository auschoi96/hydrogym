"""H1 harness: held-out judge agreement, frozen group-clustered statistics, loud leaks."""

import copy
from types import SimpleNamespace

import pytest

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME
from codex_hydrogym.gate0.ensemble_diagnostic import _mean_ci
from codex_hydrogym.genai.optimization import align_critic_quality_judge
from codex_hydrogym.memalign_h1 import (
    FROZEN_BOOTSTRAP_REPLICATES,
    FROZEN_BOOTSTRAP_SEED,
    FROZEN_HELDOUT_GROUP_COUNT,
    FROZEN_T_CRITICAL_95,
)
from codex_hydrogym.memalign_h1.harness import (
    DECISION_DEGENERATE,
    DECISION_FAIL,
    DECISION_INCONCLUSIVE,
    DECISION_PASS,
    decide_h1,
    evaluate_h1,
    group_clustered_bootstrap_delta_mae_ci,
    group_clustered_delta_mae_ci,
    heldout_agreement_metrics,
    mae,
)
from codex_hydrogym.memalign_h1.queue import manifest_digest


def _row(group_id, base_score, aligned_score, human_score):
    return {
        "group_id": group_id,
        "base_score": base_score,
        "aligned_score": aligned_score,
        "human_score": human_score,
    }


_HUMAN_SCORES = (3.0, 2.0, 2.5, 1.5)


def _four_group_rows(*, base, aligned):
    """One held-out trace per group; ``None`` means perfect agreement for that judge."""

    def score(value, human):
        return human if value is None else value

    return [
        _row(f"h1_group_{index:02d}", score(base, human), score(aligned, human), human)
        for index, human in enumerate(_HUMAN_SCORES, start=1)
    ]


def test_mae_and_group_clustered_interval_reuse_frozen_statistics():
    assert mae([1.0, 2.0, 3.0], [2.0, 2.0, 4.0]) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="equally sized"):
        mae([1.0], [])

    per_group = {"a": -0.5, "b": -0.25, "c": -0.1, "d": -0.4}
    interval = group_clustered_delta_mae_ci(per_group_delta_mae=per_group)
    expected = _mean_ci(tuple(per_group.values()), FROZEN_T_CRITICAL_95)
    for key, value in expected.items():
        assert interval[key] == pytest.approx(value)
    assert interval["upper"] < 0.0
    assert interval["standard_error"] > 0.0
    assert interval["width"] > 0.0

    with pytest.raises(ValueError, match="exactly 4 group clusters"):
        group_clustered_delta_mae_ci(per_group_delta_mae={"a": 0.0, "b": 0.0, "c": 0.0})


def test_group_clustered_bootstrap_is_fixed_seed_reproducible():
    per_group = {"a": -0.5, "b": -0.25, "c": -0.1, "d": -0.4}
    first = group_clustered_bootstrap_delta_mae_ci(per_group_delta_mae=per_group, seed=7021)
    second = group_clustered_bootstrap_delta_mae_ci(per_group_delta_mae=per_group, seed=7021)

    assert first == second
    assert first["seed"] == FROZEN_BOOTSTRAP_SEED
    assert first["replicates"] == FROZEN_BOOTSTRAP_REPLICATES


def test_decision_rule_pass_fail_inconclusive():
    assert decide_h1(delta_mae_interval={"lower": -0.9, "upper": -0.1}) == DECISION_PASS
    assert decide_h1(delta_mae_interval={"lower": 0.1, "upper": 0.9}) == DECISION_FAIL
    assert decide_h1(delta_mae_interval={"lower": -0.1, "upper": 0.1}) == DECISION_INCONCLUSIVE
    assert (
        decide_h1(
            delta_mae_interval={
                "lower": 0.5,
                "upper": 0.5,
                "variance_state": DECISION_DEGENERATE,
            }
        )
        == DECISION_DEGENERATE
    )


def test_heldout_agreement_reports_per_dimension_mae_including_regressions():
    improving = _four_group_rows(base=5.0, aligned=None)  # aligned == human labels
    metrics = heldout_agreement_metrics(rows=improving)

    dimension = metrics["per_dimension"]["value"]
    assert dimension["heldout_traces"] == 4
    assert dimension["heldout_groups"] == FROZEN_HELDOUT_GROUP_COUNT
    assert dimension["delta_mae"] == pytest.approx(dimension["aligned_mae"] - dimension["base_mae"])
    assert dimension["delta_mae"] < 0.0
    assert metrics["group_clustered_ci_95"]["value"]["upper"] < 0.0
    assert metrics["group_clustered_ci_95"]["value"]["standard_error"] > 0.0
    assert metrics["group_clustered_ci_95"]["value"]["width"] > 0.0
    assert metrics["decision"] == DECISION_PASS

    regressing = _four_group_rows(base=None, aligned=5.0)  # base == human labels
    regression = heldout_agreement_metrics(rows=regressing)
    assert regression["per_dimension"]["value"]["delta_mae"] > 0.0
    assert regression["group_clustered_ci_95"]["value"]["lower"] > 0.0
    assert regression["group_clustered_ci_95"]["value"]["standard_error"] > 0.0
    assert regression["group_clustered_ci_95"]["value"]["width"] > 0.0
    assert regression["decision"] == DECISION_FAIL

    straddling = {"lower": -0.1, "upper": 0.1}
    assert decide_h1(delta_mae_interval=straddling) == DECISION_INCONCLUSIVE


def _critic_trace(bundle_id, arm, *, source_type="HUMAN", score=None):
    assessments = [
        SimpleNamespace(
            name=CRITIC_QUALITY_ASSESSMENT_NAME,
            source=SimpleNamespace(source_type=source_type, source_id="panel@example.com"),
            value=score,
        )
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


class _RecordingOptimizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self


def test_machine_produced_critic_quality_assessment_is_rejected_by_alignment():
    machine = _critic_trace("bundle-1", "codex", source_type="LLM_JUDGE")
    with pytest.raises(ValueError, match="exactly one adjudicated"):
        align_critic_quality_judge(
            train_traces=[machine, _critic_trace("bundle-1", "claude")],
            train_bundle_ids=["bundle-1"],
            heldout_bundle_ids=["bundle-2"],
            base_judge=SimpleNamespace(name=CRITIC_QUALITY_ASSESSMENT_NAME),
            reflection_lm="databricks:/reflection",
            embedding_model="databricks:/embedding",
            optimizer_factory=_RecordingOptimizer,
        )


def test_heldout_bundle_leakage_into_train_fails_loudly():
    leaked = [
        _critic_trace("bundle-1", "codex"),
        _critic_trace("bundle-1", "claude"),
        _critic_trace("bundle-2", "codex"),  # a held-out bundle's trace in the train fold
    ]
    with pytest.raises(ValueError, match="outside the locked training bundle manifest"):
        align_critic_quality_judge(
            train_traces=leaked,
            train_bundle_ids=["bundle-1"],
            heldout_bundle_ids=["bundle-2"],
            base_judge=SimpleNamespace(name=CRITIC_QUALITY_ASSESSMENT_NAME),
            reflection_lm="databricks:/reflection",
            embedding_model="databricks:/embedding",
            optimizer_factory=_RecordingOptimizer,
        )


class _RecordingAligner:
    """Stub aligner that records its inputs and refuses any held-out content."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        heldout_ids = set(kwargs["heldout_bundle_ids"])
        if set(kwargs["train_bundle_ids"]) & heldout_ids:
            raise ValueError("held-out bundle leaked into alignment")
        return SimpleNamespace(name=CRITIC_QUALITY_ASSESSMENT_NAME)


def _heldout_provenance(rows):
    digest = "d" * 64
    manifest_rows = []
    traces = {}
    enriched = []
    for index, supplied in enumerate(rows, start=1):
        row = dict(supplied)
        trace_id = f"heldout-trace-{index}"
        assessment_id = f"assessment-{index}"
        bundle_id = f"heldout-bundle-{index}"
        arm = "codex" if index % 2 else "claude"
        evidence_digest = f"{index:064x}"
        manifest_rows.append(
            {
                "trace_id": trace_id,
                "bundle_id": bundle_id,
                "group_id": row["group_id"],
                "arm": arm,
                "fold": "heldout",
                "evidence_digest": evidence_digest,
            }
        )
        enriched.append(
            {
                **row,
                "trace_id": trace_id,
                "assessment_id": assessment_id,
                "assessment_name": CRITIC_QUALITY_ASSESSMENT_NAME,
                "source_type": "HUMAN",
                "source_id": "panel@example.com",
                "manifest_digest": digest,
            }
        )
        traces[trace_id] = SimpleNamespace(
            info=SimpleNamespace(
                tags={
                    "codex_hydrogym.group_id": row["group_id"],
                    "codex_hydrogym.bundle_id": bundle_id,
                    "codex_hydrogym.harness_arm": arm,
                    "codex_hydrogym.evidence_digest": evidence_digest,
                    "codex_hydrogym.critic_fold": "test",
                    "codex_hydrogym.memalign_manifest_digest": digest,
                },
                assessments=[
                    SimpleNamespace(
                        assessment_id=assessment_id,
                        name=CRITIC_QUALITY_ASSESSMENT_NAME,
                        value=row["human_score"],
                        source=SimpleNamespace(source_type="HUMAN", source_id="panel@example.com"),
                    )
                ],
            )
        )
    mlflow = SimpleNamespace(get_trace=lambda trace_id: traces.get(trace_id))
    manifest = {
        "digest": digest,
        "rows": manifest_rows,
        "fold_by_group": {row["group_id"]: "heldout" for row in manifest_rows},
        "counts": {"heldout_groups": 4, "heldout_traces": 4},
    }
    digest = manifest_digest(manifest)
    manifest["digest"] = digest
    for row in enriched:
        row["manifest_digest"] = digest
        traces[row["trace_id"]].info.tags["codex_hydrogym.memalign_manifest_digest"] = digest
    frozen_record = {
        "schema_version": "codex_hydrogym.memalign_h1.freeze_record.v1",
        "manifest_digest": digest,
        "commitment": {
            "rows": manifest["rows"],
            "fold_by_group": manifest["fold_by_group"],
            "counts": manifest["counts"],
        },
    }
    return manifest, frozen_record, enriched, mlflow


def test_evaluate_h1_keeps_heldout_labels_out_of_alignment():
    train_traces = [
        _critic_trace("train-1", "codex"),
        _critic_trace("train-1", "claude"),
        _critic_trace("train-2", "codex"),
        _critic_trace("train-2", "claude"),
    ]
    raw_heldout_rows = _four_group_rows(base=5.0, aligned=None)
    manifest, frozen_record, heldout_rows, mlflow = _heldout_provenance(raw_heldout_rows)
    aligner = _RecordingAligner()

    report = evaluate_h1(
        train_traces=train_traces,
        train_bundle_ids=["train-1", "train-2"],
        heldout_bundle_ids=[f"bundle-{index}" for index in range(1, 5)],
        base_judge=SimpleNamespace(name=CRITIC_QUALITY_ASSESSMENT_NAME),
        reflection_lm="databricks:/reflection",
        embedding_model="databricks:/embedding",
        heldout_rows=heldout_rows,
        manifest=manifest,
        frozen_record=frozen_record,
        mlflow_module=mlflow,
        optimizer_factory=_RecordingOptimizer,
        align_fn=aligner,
    )

    assert len(aligner.calls) == 1
    call = aligner.calls[0]
    assert set(call["train_bundle_ids"]) == {"train-1", "train-2"}
    assert not (set(call["train_bundle_ids"]) & set(call["heldout_bundle_ids"]))
    assert all(getattr(trace.info, "assessments", []) for trace in call["train_traces"])
    # The held-out human scores exist only in the rows that never reach the aligner;
    # none of the objects handed to the aligner carry a human score value.
    for trace in call["train_traces"]:
        assert all(getattr(assessment, "value", None) is None for assessment in trace.info.assessments)
    assert report["decision"] in {DECISION_PASS, DECISION_FAIL, DECISION_INCONCLUSIVE}
    assert report["heldout_label_count"] == 4


def test_evaluate_h1_rejects_machine_or_uncommitted_heldout_labels():
    manifest, frozen_record, heldout_rows, mlflow = _heldout_provenance(
        _four_group_rows(base=5.0, aligned=None)
    )
    heldout_rows[0]["source_type"] = "LLM_JUDGE"
    with pytest.raises(ValueError, match="exact attributable HUMAN"):
        evaluate_h1(
            train_traces=[],
            train_bundle_ids=["train-1"],
            heldout_bundle_ids=["heldout-1"],
            base_judge=SimpleNamespace(name=CRITIC_QUALITY_ASSESSMENT_NAME),
            reflection_lm="databricks:/reflection",
            embedding_model="databricks:/embedding",
            heldout_rows=heldout_rows,
            manifest=manifest,
            frozen_record=frozen_record,
            mlflow_module=mlflow,
            align_fn=_RecordingAligner(),
        )


def test_h1_harness_statistics_are_group_clustered_not_per_trace():
    """Two traces in one group never inflate the effective cluster count."""
    rows = []
    expected_deltas = (0.2, 0.4, 0.6, 0.8)
    for group_index, delta in enumerate(expected_deltas, start=1):
        group = f"h1_group_{group_index:02d}"
        # Independently varying group deltas; base is exact and aligned is worse.
        rows.append(_row(group, 2.0, 2.0 + delta, 2.0))
        rows.append(_row(group, 3.0, 3.0 + delta, 3.0))
    metrics = heldout_agreement_metrics(rows=rows)
    assert metrics["per_dimension"]["value"]["heldout_traces"] == 8
    assert metrics["per_dimension"]["value"]["heldout_groups"] == FROZEN_HELDOUT_GROUP_COUNT
    ci = metrics["group_clustered_ci_95"]["value"]
    assert ci["group_clusters"] == FROZEN_HELDOUT_GROUP_COUNT
    assert ci["t_critical"] == FROZEN_T_CRITICAL_95
    deltas = metrics["per_dimension"]["value"]["per_group_delta_mae"]
    assert len(deltas) == FROZEN_HELDOUT_GROUP_COUNT
    assert sorted(deltas.values()) == pytest.approx(list(expected_deltas))
    assert ci["standard_error"] > 0.0
    assert ci["width"] > 0.0
    assert metrics["decision"] == DECISION_FAIL


def test_zero_group_variance_is_explicitly_degenerate():
    interval = group_clustered_delta_mae_ci(
        per_group_delta_mae={"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
    )
    assert interval["standard_error"] == 0.0
    assert interval["width"] == 0.0
    assert interval["variance_state"] == DECISION_DEGENERATE
    assert decide_h1(delta_mae_interval=interval) == DECISION_DEGENERATE
