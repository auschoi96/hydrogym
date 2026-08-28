"""Fully offline proof for preparing the two coding-agent harness arms."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from codex_hydrogym.genai.contracts import parse_run_bundle
from codex_hydrogym.genai.dry_run import (
    DRY_RUN_ADAPTER_IDS,
    build_dry_run_manifest,
    build_dry_run_records,
    build_sanity_bundles,
    main,
    render_manifest_json,
    validate_dry_run_records,
    validate_sanity_bundles,
)


def test_builds_exactly_five_bundles_and_ten_planned_arm_records():
    bundles = build_sanity_bundles()
    records = build_dry_run_records(bundles)

    assert len(bundles) == 5
    assert len({bundle.bundle_id for bundle in bundles}) == 5
    assert len({bundle.group_id for bundle in bundles}) == 5
    assert len(records) == 10
    assert {record["adapter_id"] for record in records} == set(DRY_RUN_ADAPTER_IDS)
    assert len({record["case_id"] for record in records}) == 10
    assert len({record["row_id"] for record in records}) == 10
    assert all("outputs" not in record for record in records)
    runtime_only = {
        "latency_ms",
        "cost",
        "session_id",
        "thread_id",
        "trace_id",
        "timestamp",
        "runtime_metadata",
    }
    assert all(runtime_only.isdisjoint(record) for record in records)
    assert all(bundle.task["claim_allowed"] is False for bundle in bundles)


def test_paired_arms_share_exact_prompt_and_evidence_but_not_case_id():
    records = build_dry_run_records()

    for bundle_id in {record["bundle_id"] for record in records}:
        pair = [record for record in records if record["bundle_id"] == bundle_id]
        assert len(pair) == 2
        assert pair[0]["prompt"] == pair[1]["prompt"]
        assert pair[0]["prompt_sha256"] == pair[1]["prompt_sha256"]
        assert pair[0]["evidence_digest"] == pair[1]["evidence_digest"]
        assert pair[0]["case_id"] != pair[1]["case_id"]
        assert pair[0]["row_id"] != pair[1]["row_id"]
        for record in pair:
            identity = f"{record['evidence_digest']}:{record['adapter_id']}"
            expected_row_id = f"row_{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
            assert record["row_id"] == expected_row_id
        assert all(adapter_id not in record["case_id"] for record in pair for adapter_id in DRY_RUN_ADAPTER_IDS)


def test_manifest_is_deterministic_strict_json_and_explicitly_not_executed():
    first = render_manifest_json()
    second = render_manifest_json()
    manifest = json.loads(first)

    assert first == second
    assert first == json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    assert manifest == build_dry_run_manifest()
    assert manifest["bundle_count"] == 5
    assert manifest["record_count"] == 10
    assert manifest["execution"] == {
        "status": "not_executed",
        "sdk_imports_performed": False,
        "agent_calls_performed": False,
        "model_calls_performed": False,
        "mlflow_calls_performed": False,
        "databricks_calls_performed": False,
        "publishing_performed": False,
    }
    assert "not agent-quality, MemAlign, RL, or fluid-improvement evidence" in manifest["claim_boundary"]


def test_every_embedded_run_bundle_round_trips_through_the_strict_parser():
    for record in build_dry_run_records():
        raw = record["inputs"]["run_bundle"]
        reparsed = parse_run_bundle(json.dumps(raw, sort_keys=True))

        assert reparsed.as_dict() == raw
        assert reparsed.evidence_digest == record["evidence_digest"]


def test_validation_rejects_duplicate_noncanonical_and_tampered_inputs():
    bundles = build_sanity_bundles()
    duplicate = (bundles[0], bundles[0], *bundles[2:])
    with pytest.raises(ValueError, match="bundle_id values must be unique"):
        validate_sanity_bundles(duplicate)

    wrong_group = (replace(bundles[0], group_id="sanity_group_wrong"), *bundles[1:])
    with pytest.raises(ValueError, match="non-canonical group_id"):
        validate_sanity_bundles(wrong_group)

    records = list(build_dry_run_records(bundles))
    records[0] = {**records[0], "prompt": records[0]["prompt"] + "\ntampered"}
    with pytest.raises(ValueError, match="does not match its canonical input"):
        validate_dry_run_records(bundles=bundles, records=records)


def test_default_cli_prints_json_and_explicit_output_path_is_the_only_write(tmp_path, capsys):
    assert main([]) == 0
    stdout_manifest = json.loads(capsys.readouterr().out)
    assert stdout_manifest["record_count"] == 10
    assert list(tmp_path.iterdir()) == []

    output = tmp_path / "dry-run.json"
    assert main(["--output", str(output)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text(encoding="utf-8")) == stdout_manifest


def test_offline_build_does_not_import_external_runtimes_or_sdks():
    project_root = Path(__file__).resolve().parents[2]
    script = """
import json
import sys

before = set(sys.modules)
from codex_hydrogym.genai.dry_run import build_dry_run_manifest

manifest = build_dry_run_manifest()
loaded = set(sys.modules) - before
forbidden = ("openai_codex", "claude_agent_sdk", "mlflow", "databricks")
unexpected = sorted(
    name for name in loaded
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)
assert unexpected == [], unexpected
assert manifest["execution"]["model_calls_performed"] is False
print(json.dumps({"record_count": manifest["record_count"], "unexpected": unexpected}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"record_count": 10, "unexpected": []}
    assert completed.stderr == ""
