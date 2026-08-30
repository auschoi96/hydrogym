"""Judge-name preflight: make MemAlign's silent label filter an actionable precondition."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME
from codex_hydrogym.memalign_h1 import AUDIT_JUDGE_NAME, MEMALIGN_JUDGE_NAMES
from codex_hydrogym.memalign_h1.preflight import (
    OUTCOME_NAME_MISMATCH,
    OUTCOME_NON_HUMAN_SOURCE,
    OUTCOME_SURVIVES,
    classify_assessment,
    preflight_judge_labels,
)


def _assessment(name, source_type):
    return SimpleNamespace(
        name=name,
        source=SimpleNamespace(source_type=source_type, source_id="reviewer@example.com"),
    )


def _trace(trace_id, *assessments):
    return SimpleNamespace(
        info=SimpleNamespace(
            trace_id=trace_id,
            assessments=list(assessments),
            request={"case_id": trace_id},
            response={"analysis": {"critique": "text"}},
            expectations={"required_findings": ["gate"]},
        )
    )


def _traces_with_collision():
    """One trace per label world: the App label, the adjudication label, model scores."""
    return [
        _trace(
            "app-labeled",
            _assessment(FEEDBACK_ASSESSMENT_NAME, "HUMAN"),
        ),
        _trace(
            "panel-labeled",
            _assessment(CRITIC_QUALITY_ASSESSMENT_NAME, "HUMAN"),
        ),
        _trace(
            "model-scored",
            _assessment(CRITIC_QUALITY_ASSESSMENT_NAME, "LLM_JUDGE"),
            _assessment(AUDIT_JUDGE_NAME, "LLM_JUDGE"),
        ),
        _trace(
            "other-name",
            _assessment("some_other_schema", "HUMAN"),
        ),
    ]


def test_preflight_counts_survivors_and_rejection_reasons_for_every_judge_name():
    traces = _traces_with_collision()

    for judge_name in MEMALIGN_JUDGE_NAMES:
        report = preflight_judge_labels(judge_name=judge_name, traces=traces)
        totals = report["totals"]
        assert totals["traces"] == 4
        assert report["sanitized_judge_name"] == judge_name.lower().strip()

    app_report = preflight_judge_labels(judge_name=FEEDBACK_ASSESSMENT_NAME, traces=traces)
    assert app_report["totals"]["surviving_assessments"] == 1
    assert app_report["totals"]["rejected_name_mismatch"] == 4
    assert app_report["totals"]["rejected_non_human_source"] == 0
    assert app_report["memalign_would_fail"] is False

    critic_report = preflight_judge_labels(judge_name=CRITIC_QUALITY_ASSESSMENT_NAME, traces=traces)
    assert critic_report["totals"]["surviving_assessments"] == 1
    assert critic_report["totals"]["rejected_name_mismatch"] == 3
    assert critic_report["totals"]["rejected_non_human_source"] == 1
    assert critic_report["totals"]["human_assessments_total"] == 3

    audit_report = preflight_judge_labels(judge_name=AUDIT_JUDGE_NAME, traces=traces)
    assert audit_report["totals"]["surviving_assessments"] == 0
    assert audit_report["totals"]["rejected_name_mismatch"] == 4
    assert audit_report["totals"]["rejected_non_human_source"] == 1
    assert audit_report["memalign_would_fail"] is True


def test_machine_assessment_matching_the_judge_name_is_rejected_non_human():
    machine = _trace("machine-1", _assessment(CRITIC_QUALITY_ASSESSMENT_NAME, "LLM_JUDGE"))
    report = preflight_judge_labels(judge_name=CRITIC_QUALITY_ASSESSMENT_NAME, traces=[machine])

    assert report["totals"]["surviving_assessments"] == 0
    assert report["totals"]["rejected_non_human_source"] == 1
    assert report["totals"]["rejected_name_mismatch"] == 0
    assert report["memalign_would_fail"] is True
    row = report["trace_rows"][0]
    assert row["assessments"][0]["outcome"] == OUTCOME_NON_HUMAN_SOURCE


def test_collision_labels_under_the_app_name_are_invisible_to_the_critic_quality_judge():
    app_labels = [
        _trace("p1", _assessment(FEEDBACK_ASSESSMENT_NAME, "HUMAN")),
        _trace("p2", _assessment(FEEDBACK_ASSESSMENT_NAME, "HUMAN")),
    ]

    visible = preflight_judge_labels(judge_name=FEEDBACK_ASSESSMENT_NAME, traces=app_labels)
    assert visible["totals"]["surviving_assessments"] == 2
    assert visible["memalign_would_fail"] is False

    invisible = preflight_judge_labels(judge_name=CRITIC_QUALITY_ASSESSMENT_NAME, traces=app_labels)
    assert invisible["totals"]["surviving_assessments"] == 0
    assert invisible["totals"]["rejected_name_mismatch"] == 2
    assert invisible["memalign_would_fail"] is True


def test_sanitized_name_matching_mirrors_mlflow_lowercase_strip():
    assessment = _assessment("  Critic_Quality ", "HUMAN")

    assert classify_assessment(assessment, judge_name="critic_quality") == OUTCOME_SURVIVES
    assert classify_assessment(assessment, judge_name=CRITIC_QUALITY_ASSESSMENT_NAME) == OUTCOME_SURVIVES
    assert classify_assessment(assessment, judge_name=FEEDBACK_ASSESSMENT_NAME) == OUTCOME_NAME_MISMATCH


def test_preflight_rejects_unknown_judge_name_and_empty_trace_list():
    with pytest.raises(ValueError, match="judge_name must be one of"):
        preflight_judge_labels(judge_name="not_a_judge", traces=[_trace("t1")])
    with pytest.raises(ValueError, match="at least one candidate trace"):
        preflight_judge_labels(judge_name=CRITIC_QUALITY_ASSESSMENT_NAME, traces=[])


def test_audit_judge_name_matches_the_frozen_notebook_constant():
    notebook_path = Path(__file__).resolve().parents[2] / "codex_hydrogym/notebooks/coding_agent_memalign_proof.py"
    source = notebook_path.read_text(encoding="utf-8")
    assert f'AUDIT_JUDGE_NAME = "{AUDIT_JUDGE_NAME}"' in source
