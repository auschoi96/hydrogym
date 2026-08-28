"""Offline preparation proof for the paired coding-agent sanity corpus.

This module deliberately stops before either SDK boundary.  It validates the
five canonical, non-claiming sanity bundles and renders the exact prompt that a
future Codex SDK or Claude Agent SDK run would receive.  It does not import an
SDK, call a model, start an MLflow trace, publish a dataset, or contact
Databricks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.genai.contracts import RunBundle, parse_run_bundle
from codex_hydrogym.genai.datasets import build_harness_sanity_bundles
from codex_hydrogym.genai.harnesses import prompt_digest, render_feedback_prompt


DRY_RUN_MANIFEST_SCHEMA_VERSION = "codex_hydrogym.harness_dry_run.v1"
DRY_RUN_RECORD_SCHEMA_VERSION = "codex_hydrogym.intended_harness_arm.v1"
DRY_RUN_ADAPTER_IDS = ("codex_sdk", "claude_agent_sdk")

_CASE_ID_SALT = "codex_hydrogym.offline_sanity.v1"
_CANONICAL_BUNDLE_GROUPS = {
    "sanity_open_loop_alias": "sanity_group_open_loop_alias",
    "sanity_cross_context_false_win": "sanity_group_cross_context",
    "sanity_failed_physics_false_win": "sanity_group_failed_physics",
    "sanity_shuffled_observation_null": "sanity_group_shuffle_null",
    "sanity_positive_causal_pattern": "sanity_group_positive_pattern",
}
_RECORD_FIELDS = {
    "schema_version",
    "row_id",
    "case_id",
    "bundle_id",
    "group_id",
    "adapter_id",
    "prompt",
    "prompt_sha256",
    "evidence_digest",
    "inputs",
}


def _row_id(*, bundle: RunBundle, adapter_id: str) -> str:
    payload = f"{bundle.evidence_digest}:{adapter_id}"
    return f"row_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _case_id(*, bundle: RunBundle, adapter_id: str) -> str:
    payload = f"{_CASE_ID_SALT}:{bundle.evidence_digest}:{adapter_id}"
    return f"case_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def validate_sanity_bundles(bundles: Sequence[RunBundle]) -> tuple[RunBundle, ...]:
    """Return canonical-order bundles after strict offline validation."""
    materialized = tuple(bundles)
    if len(materialized) != len(_CANONICAL_BUNDLE_GROUPS):
        raise ValueError("the offline proof requires exactly five canonical sanity bundles")
    if any(not isinstance(bundle, RunBundle) for bundle in materialized):
        raise TypeError("every sanity bundle must be a RunBundle")

    bundle_ids = [bundle.bundle_id for bundle in materialized]
    group_ids = [bundle.group_id for bundle in materialized]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("sanity bundle_id values must be unique")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("sanity group_id values must be unique")
    if set(bundle_ids) != set(_CANONICAL_BUNDLE_GROUPS):
        raise ValueError("sanity bundle_id values do not match the canonical corpus")

    by_id = {bundle.bundle_id: bundle for bundle in materialized}
    ordered = tuple(by_id[bundle_id] for bundle_id in _CANONICAL_BUNDLE_GROUPS)
    for bundle in ordered:
        if bundle.group_id != _CANONICAL_BUNDLE_GROUPS[bundle.bundle_id]:
            raise ValueError(f"{bundle.bundle_id} has a non-canonical group_id")
        if bundle.task.get("claim_allowed") is not False:
            raise ValueError(f"{bundle.bundle_id} must explicitly forbid fluid-improvement claims")
        if not bundle.comparison_issues():
            raise ValueError(f"{bundle.bundle_id} must retain at least one deterministic comparison issue")
        if any(arm.evidence_kind == "measured" for arm in (bundle.candidate, *bundle.comparators)):
            raise ValueError(f"{bundle.bundle_id} cannot contain measured evidence in the sanity corpus")

        reparsed = parse_run_bundle(bundle.canonical_json())
        if reparsed.as_dict() != bundle.as_dict() or reparsed.evidence_digest != bundle.evidence_digest:
            raise ValueError(f"{bundle.bundle_id} failed its canonical JSON round trip")
    return ordered


def build_sanity_bundles() -> tuple[RunBundle, ...]:
    """Build and validate the five existing Gate-0 sanity fixtures."""
    return validate_sanity_bundles(build_harness_sanity_bundles())


def validate_dry_run_records(
    *,
    bundles: Sequence[RunBundle],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Validate prompt parity, record identity, and absence of fabricated output."""
    canonical_bundles = validate_sanity_bundles(bundles)
    materialized = tuple(records)
    expected_count = len(canonical_bundles) * len(DRY_RUN_ADAPTER_IDS)
    if len(materialized) != expected_count:
        raise ValueError(f"the offline proof requires exactly {expected_count} intended arm records")
    if any(not isinstance(record, Mapping) for record in materialized):
        raise TypeError("every intended arm record must be a mapping")

    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for bundle in canonical_bundles:
        prompt = render_feedback_prompt(bundle)
        digest = prompt_digest(prompt)
        for adapter_id in DRY_RUN_ADAPTER_IDS:
            row_id = _row_id(bundle=bundle, adapter_id=adapter_id)
            case_id = _case_id(bundle=bundle, adapter_id=adapter_id)
            expected[(bundle.bundle_id, adapter_id)] = {
                "schema_version": DRY_RUN_RECORD_SCHEMA_VERSION,
                "row_id": row_id,
                "case_id": case_id,
                "bundle_id": bundle.bundle_id,
                "group_id": bundle.group_id,
                "adapter_id": adapter_id,
                "prompt": prompt,
                "prompt_sha256": digest,
                "evidence_digest": bundle.evidence_digest,
                "inputs": {
                    "case_id": case_id,
                    "run_bundle": bundle.as_dict(),
                    "task_contract_version": bundle.task_contract_version,
                },
            }

    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in materialized:
        if set(record) != _RECORD_FIELDS:
            raise ValueError("intended arm record fields do not match the dry-run contract")
        key = (record["bundle_id"], record["adapter_id"])
        if key in observed:
            raise ValueError(f"duplicate intended arm record for bundle={key[0]}, adapter_id={key[1]}")
        observed[key] = record
    if set(observed) != set(expected):
        raise ValueError("intended arm records must contain both SDK arms for every sanity bundle")
    for key, expected_record in expected.items():
        if dict(observed[key]) != expected_record:
            raise ValueError(f"intended arm record does not match its canonical input: {key}")

    case_ids = [record["case_id"] for record in materialized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("every intended arm record must have a unique opaque case_id")
    row_ids = [record["row_id"] for record in materialized]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("every intended arm record must have a unique stable row_id")


def build_dry_run_records(
    bundles: Sequence[RunBundle] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Render ten planned records without invoking either coding-agent SDK."""
    selected = build_sanity_bundles() if bundles is None else validate_sanity_bundles(bundles)
    records: list[dict[str, Any]] = []
    for bundle in selected:
        prompt = render_feedback_prompt(bundle)
        digest = prompt_digest(prompt)
        for adapter_id in DRY_RUN_ADAPTER_IDS:
            row_id = _row_id(bundle=bundle, adapter_id=adapter_id)
            case_id = _case_id(bundle=bundle, adapter_id=adapter_id)
            records.append(
                {
                    "schema_version": DRY_RUN_RECORD_SCHEMA_VERSION,
                    "row_id": row_id,
                    "case_id": case_id,
                    "bundle_id": bundle.bundle_id,
                    "group_id": bundle.group_id,
                    "adapter_id": adapter_id,
                    "prompt": prompt,
                    "prompt_sha256": digest,
                    "evidence_digest": bundle.evidence_digest,
                    "inputs": {
                        "case_id": case_id,
                        "run_bundle": bundle.as_dict(),
                        "task_contract_version": bundle.task_contract_version,
                    },
                }
            )
    validate_dry_run_records(bundles=selected, records=records)
    return tuple(records)


def build_dry_run_manifest() -> dict[str, Any]:
    """Return a deterministic, JSON-serializable preparation manifest."""
    bundles = build_sanity_bundles()
    records = build_dry_run_records(bundles)
    manifest = {
        "schema_version": DRY_RUN_MANIFEST_SCHEMA_VERSION,
        "project": PROJECT_LABEL,
        "mode": "offline_preparation_only",
        "bundle_count": len(bundles),
        "record_count": len(records),
        "adapter_ids": list(DRY_RUN_ADAPTER_IDS),
        "records": list(records),
        "execution": {
            "status": "not_executed",
            "sdk_imports_performed": False,
            "agent_calls_performed": False,
            "model_calls_performed": False,
            "mlflow_calls_performed": False,
            "databricks_calls_performed": False,
            "publishing_performed": False,
        },
        "claim_boundary": (
            "This manifest proves deterministic bundle validation, prompt rendering, and paired record "
            "preparation only; it is not agent-quality, MemAlign, RL, or fluid-improvement evidence."
        ),
    }
    # Refuse values that are not strict JSON before exposing or writing them.
    json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return manifest


def render_manifest_json(manifest: Mapping[str, Any] | None = None) -> str:
    """Serialize a manifest reproducibly for stdout or an explicit file path."""
    selected = build_dry_run_manifest() if manifest is None else dict(manifest)
    return json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="prepare the paired coding-agent sanity manifest without external calls"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this explicit path instead of stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = render_manifest_json()
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
