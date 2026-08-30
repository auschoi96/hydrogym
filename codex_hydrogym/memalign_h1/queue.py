"""Labeling queue: select coding-agent traces, lock train/held-out folds BY GROUP.

The queue selects measured coding-agent traces that still need an adjudicated
``critic_quality`` HUMAN label, assigns whole ``group_id`` values to locked
train/held-out folds, and emits an ordered manifest whose SHA-256 freezes the split
BEFORE any label is collected.  The fold ranking and the fold-lock invariant are the
repository's own: the SHA-256 group ranking comes from
``codex_hydrogym.genai.datasets.grouped_bundle_split`` and the per-trace fold tag comes
from ``codex_hydrogym.genai.feedback.enroll_critic_quality_trace``.  None of that logic
is reimplemented here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, PROJECT_LABEL
from codex_hydrogym.genai.contracts import RunBundle, parse_run_bundle
from codex_hydrogym.genai.datasets import grouped_bundle_split
from codex_hydrogym.genai.feedback import enroll_critic_quality_trace, matching_human_feedback
from codex_hydrogym.memalign_h1 import FROZEN_HELDOUT_GROUP_COUNT, H1_SPLIT_SALT, PROTOCOL_ID

QUEUE_MANIFEST_SCHEMA_VERSION = "codex_hydrogym.memalign_h1.queue_manifest.v1"

HARNESS_ARMS = ("codex", "claude")
_EVIDENCE_KIND_TAG = f"{PROJECT_LABEL}.evidence_kind"


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def manifest_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical-ordered manifest payload (digest excluded)."""
    body = {key: value for key, value in payload.items() if key != "digest"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateTrace:
    """One measured coding-agent trace eligible for critic_quality adjudication."""

    trace_id: str
    bundle: RunBundle
    arm: str
    evidence_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "bundle_id": self.bundle.bundle_id,
            "group_id": self.bundle.group_id,
            "arm": self.arm,
            "evidence_digest": self.evidence_digest,
        }


def _root_span_bundle(trace: Any) -> RunBundle | None:
    """Recover the canonical RunBundle from the root AGENT span inputs."""
    spans = getattr(getattr(trace, "data", None), "spans", None)
    if not isinstance(spans, Sequence):
        return None
    roots = [span for span in spans if getattr(span, "parent_id", None) is None]
    if len(roots) != 1:
        return None
    inputs = getattr(roots[0], "inputs", None)
    if not isinstance(inputs, Mapping):
        return None
    run_bundle = inputs.get("run_bundle")
    if run_bundle is None:
        return None
    try:
        return parse_run_bundle(run_bundle)
    except (TypeError, ValueError):
        return None


def select_coding_agent_traces(
    *,
    experiment_id: str,
    mlflow_module=None,
    max_results: int = 2000,
    required_arms: Sequence[str] = HARNESS_ARMS,
) -> dict[str, Any]:
    """Select measured harness-arm traces that need the adjudicated HUMAN label.

    Rejects, with explicit reasons: traces without one of the required harness-arm
    tags, traces whose root span does not carry a parseable canonical RunBundle (for
    example the PPO patch traces from coding_rl/experiment.py, which carry a
    RepairTask payload instead of a RunBundle and cannot enter critic_quality folds),
    and traces that already carry an adjudicated ``critic_quality`` HUMAN label.
    """
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id is required")
    expected_arms = set(required_arms)
    if not expected_arms or not expected_arms <= set(HARNESS_ARMS):
        raise ValueError(f"required_arms must be a non-empty subset of {HARNESS_ARMS}")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 100_000:
        raise ValueError("max_results must be an integer in [1, 100000]")
    mlflow = mlflow_module or importlib.import_module("mlflow")

    candidates = mlflow.search_traces(
        experiment_ids=[experiment_id.strip()],
        filter_string=f"tags.`{_EVIDENCE_KIND_TAG}` = 'measured'",
        max_results=max_results,
        return_type="list",
        include_spans=True,
    )

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for position, trace in enumerate(candidates, start=1):
        info = getattr(trace, "info", None)
        tags = getattr(info, "tags", {}) or {}
        trace_id = getattr(info, "trace_id", None)
        if trace_id is None:
            trace_id = f"trace_{position}"
        trace_id = str(trace_id)
        arm = tags.get(f"{PROJECT_LABEL}.harness_arm")
        if arm not in expected_arms:
            skipped.append({"trace_id": trace_id, "reason": "missing or unrecognized harness-arm tag"})
            continue
        bundle = _root_span_bundle(trace)
        if bundle is None:
            skipped.append(
                {
                    "trace_id": trace_id,
                    "reason": "root span carries no parseable canonical RunBundle",
                }
            )
            continue
        if matching_human_feedback(trace, assessment_name=CRITIC_QUALITY_ASSESSMENT_NAME):
            skipped.append({"trace_id": trace_id, "reason": "already has an adjudicated critic_quality HUMAN label"})
            continue
        records.append(
            CandidateTrace(
                trace_id=trace_id,
                bundle=bundle,
                arm=str(arm),
                evidence_digest=bundle.evidence_digest,
            ).as_dict()
        )

    return {
        "schema_version": QUEUE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": experiment_id.strip(),
        "candidates": records,
        "skipped": skipped,
        "counts": {
            "searched": len(candidates),
            "selected": len(records),
            "skipped": len(skipped),
        },
    }


def build_locked_fold_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    test_group_count: int = FROZEN_HELDOUT_GROUP_COUNT,
    split_salt: str = H1_SPLIT_SALT,
) -> dict[str, Any]:
    """Assign whole groups to locked train/held-out folds and freeze the manifest.

    The group ranking and the no-group-straddles-folds invariant come from the
    repository's ``grouped_bundle_split``; every candidate record is converted back to
    its canonical RunBundle so the split is literally the same function the harness
    corpus uses.  The returned manifest carries an ordered row list and a SHA-256
    digest over the canonical payload; it must be persisted and frozen BEFORE any
    label is collected, and the same digest must gate enrollment.
    """
    materialized = [dict(record) for record in records]
    if not materialized:
        raise ValueError("at least one candidate record is required")
    bundles: list[RunBundle] = []
    for record in materialized:
        bundle = record.get("bundle")
        if not isinstance(bundle, RunBundle):
            raise ValueError("every candidate record must carry its canonical RunBundle")
        bundles.append(bundle)
    trace_ids = [str(record["trace_id"]) for record in materialized]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("candidate trace_id values must be unique")
    for record, bundle in zip(materialized, bundles, strict=True):
        record["bundle"] = bundle
        if record.get("bundle_id") != bundle.bundle_id or record.get("group_id") != bundle.group_id:
            raise ValueError("candidate record bundle/group identity must match its RunBundle")

    bundle_folds = grouped_bundle_split(
        bundles,
        test_group_count=test_group_count,
        split_salt=split_salt,
    )
    group_map: dict[str, str] = {}
    for bundle in bundles:
        raw_fold = bundle_folds[bundle.bundle_id]
        fold = "heldout" if raw_fold == "test" else "train"
        prior = group_map.get(bundle.group_id)
        if prior is not None and prior != fold:
            raise ValueError(
                f"group {bundle.group_id!r} straddles folds ({prior!r} vs {fold!r}); the split is not group-disjoint"
            )
        group_map[bundle.group_id] = fold

    rows = []
    for record in materialized:
        fold = group_map[str(record["group_id"])]
        rows.append(
            {
                "trace_id": str(record["trace_id"]),
                "bundle_id": str(record["bundle_id"]),
                "group_id": str(record["group_id"]),
                "arm": str(record["arm"]),
                "fold": fold,
                "evidence_digest": str(record["evidence_digest"]),
            }
        )
    rows.sort(key=lambda row: (row["fold"], row["group_id"], row["bundle_id"], row["arm"], row["trace_id"]))

    payload = {
        "schema_version": QUEUE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "split_salt": split_salt,
        "test_group_count": test_group_count,
        "fold_by_group": group_map,
        "rows": rows,
        "counts": {
            "groups": len(group_map),
            "train_groups": sum(fold == "train" for fold in group_map.values()),
            "heldout_groups": sum(fold == "heldout" for fold in group_map.values()),
            "traces": len(rows),
            "train_traces": sum(row["fold"] == "train" for row in rows),
            "heldout_traces": sum(row["fold"] == "heldout" for row in rows),
        },
    }
    payload["digest"] = manifest_digest(payload)
    return payload


def enroll_locked_folds(
    *,
    manifest: Mapping[str, Any],
    mlflow_module=None,
    require_digest: str | None = None,
) -> dict[str, Any]:
    """Tag every manifest trace with its locked fold, reusing enroll_critic_quality_trace.

    ``require_digest`` re-verifies the manifest SHA-256 before any tag is written so a
    manifest mutated after freezing cannot drive enrollment.  Held-out folds map to the
    repository's ``test`` fold tag -- the same tag the pair of harness arms already
    carries -- and enrollment refuses any trace that already has its one adjudicated
    label.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    expected_digest = manifest.get("digest")
    if not isinstance(expected_digest, str) or manifest_digest(manifest) != expected_digest:
        raise ValueError("manifest digest does not match its canonical payload")
    if require_digest is not None and expected_digest != require_digest:
        raise ValueError(f"manifest digest {expected_digest} does not match the frozen digest {require_digest}")
    mlflow = mlflow_module or importlib.import_module("mlflow")

    rows = manifest.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("manifest must contain a non-empty ordered row list")
    enrolled: list[dict[str, Any]] = []
    for row in rows:
        fold = row.get("fold")
        if fold not in {"train", "heldout"}:
            raise ValueError("manifest row fold must be train or heldout")
        enroll_critic_quality_trace(
            trace_id=str(row["trace_id"]),
            fold="train" if fold == "train" else "test",
            mlflow_module=mlflow,
        )
        enrolled.append(
            {
                "trace_id": str(row["trace_id"]),
                "fold": fold,
                "critic_fold_tag": "train" if fold == "train" else "test",
            }
        )
    return {
        "schema_version": QUEUE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "digest": expected_digest,
        "enrolled": enrolled,
        "count": len(enrolled),
        "review_state": "pending_adjudication",
    }
