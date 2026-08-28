# Databricks notebook source
# ruff: noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC # HydroGym Re=100 Gate 0 v3 protocol review
# MAGIC
# MAGIC This notebook installs the reviewed wheel, freezes and round-trips the exact v3
# MAGIC protocol, independently checks its held-out seed derivation and source manifest,
# MAGIC exercises the analysis on synthetic pass/fail payloads, and proves the run stage
# MAGIC fails closed without human review. It executes **zero CFD**, opens no reserved case,
# MAGIC and performs no reinforcement learning.

# COMMAND ----------

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


WORKSPACE_ROOT = "/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_v3"
DEFAULT_WHEEL = f"{WORKSPACE_ROOT}/hydrogym-1.0.0-py3-none-any.whl"
EXPECTED_WHEEL_SHA256 = "e381b42d415b0644fd773be67ef9aab94133289e6559933bf72e079da50e2e51"
SEED_NAMESPACE = "codex_hydrogym:re100_gate0_v3:heldout_seed:v1"
RESERVED_SEEDS = (907, 1009)
PRIOR_SEEDS = {
    7,
    101,
    211,
    307,
    401,
    503,
    607,
    709,
    1100085772,
    619716833,
    1680869979,
    270788329,
    1326527252,
    625393611,
    901546380,
    1422036434,
    373522063,
    1374108181,
}


def _widget(name: str, default: str) -> str:
    try:
        return str(dbutils.widgets.get(name))
    except Exception:
        dbutils.widgets.text(name, default)
        return str(dbutils.widgets.get(name))


WHEEL_PATH = Path(_widget("wheel_path", DEFAULT_WHEEL))
EXPECTED_SHA256 = _widget("expected_wheel_sha256", EXPECTED_WHEEL_SHA256)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _derive_additional_seeds() -> tuple[int, ...]:
    excluded = set((*PRIOR_SEEDS, *RESERVED_SEEDS))
    selected: list[int] = []
    counter = 0
    while len(selected) < 10:
        material = f"{SEED_NAMESPACE}:{counter}".encode("ascii")
        candidate = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)
        counter += 1
        if candidate and candidate not in excluded and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


wheel_sha256 = hashlib.sha256(WHEEL_PATH.read_bytes()).hexdigest()
if wheel_sha256 != EXPECTED_SHA256:
    raise RuntimeError("v3 wheel SHA-256 does not match the reviewed artifact")
with zipfile.ZipFile(WHEEL_PATH) as archive:
    names = set(archive.namelist())
    if "codex_hydrogym/gate0/re100_v3.py" not in names:
        raise RuntimeError("reviewed wheel does not contain the v3 runner")

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        str(WHEEL_PATH),
    ]
)

from codex_hydrogym.gate0 import re100_v3 as v3


spec = v3.Re100Gate0V3Spec()
config = v3.materialized_gate_config(spec)
lock = v3.frozen_controller_lock(config)
implementation_files, implementation_digest = v3._implementation_manifest_v3()
output_dir = Path(WORKSPACE_ROOT) / "evidence" / (
    f"{spec.fingerprint[:12]}-{implementation_digest[:12]}"
)

checks: dict[str, bool] = {}
checks["independent_seed_derivation_matches"] = spec.additional_heldout_seeds == _derive_additional_seeds()
checks["reserved_seeds_are_first_and_unmodified"] = spec.heldout_seeds[:2] == RESERVED_SEEDS
checks["heldout_seed_count_is_12"] = len(spec.heldout_seeds) == 12
checks["heldout_phases_are_reserved_pair"] = spec.heldout_phase_turns == (0.1875, 0.6875)
checks["full_arm_set_is_restored"] = spec.arms == (
    "zero",
    "fixed",
    "oracle",
    "signed_feedback",
    "observation_deranged",
)
checks["expected_trajectory_count_is_360"] = spec.expected_trajectory_count == 360
checks["expected_window_count_is_720"] = spec.expected_window_count == 720
checks["controller_lock_is_gain_2"] = lock.feedback_gain == 2.0
checks["controller_lock_copies_v2_fixed_action"] = lock.fixed_action == (
    0.1767766952966369,
    0.17677669529663687,
    0.0,
    0.0,
)
checks["jax_float64_is_frozen"] = spec.precision == "float64"
checks["one_h100_air_backend_is_frozen"] = (
    spec.execution_service,
    spec.accelerator_type,
    spec.accelerator_count,
) == ("Databricks AI Runtime", "GPU_1xH100", 1)

for relative_path, expected_digest in implementation_files.items():
    observed = hashlib.sha256((Path(v3.__file__).resolve().parents[2] / relative_path).read_bytes()).hexdigest()
    if observed != expected_digest:
        raise RuntimeError(f"installed v3 implementation hash mismatch: {relative_path}")
checks["all_implementation_hashes_match"] = True

freeze_exit = v3.main(["--stage", "freeze", "--output-dir", str(output_dir)])
review_exit = v3.main(["--stage", "review", "--output-dir", str(output_dir)])
checks["freeze_and_review_exit_successfully"] = freeze_exit == 0 and review_exit == 0
protocol_path = output_dir / "protocol.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
protocol_body = {key: value for key, value in protocol.items() if key != "artifact_digest"}
checks["protocol_digest_is_canonical"] = protocol["artifact_digest"] == _digest(protocol_body)
checks["protocol_binds_study_fingerprint"] = protocol["study_fingerprint"] == spec.fingerprint
checks["protocol_binds_implementation"] = protocol["implementation_digest"] == implementation_digest
checks["protocol_keeps_execution_unauthorized"] = protocol["execution_authorized"] is False
checks["protocol_records_reserved_cases_unopened"] = protocol["reserved_cases_opened"] is False
checks["protocol_records_no_rl"] = protocol["rl_training_performed"] is False
checks["protocol_excludes_prior_observations"] = (
    protocol["execution_plan"]["prior_or_local_observations_in_analysis"] == 0
)

try:
    v3.main(["--stage", "run", "--output-dir", str(output_dir)])
except RuntimeError as error:
    checks["run_fails_closed_without_human_review"] = "human review attestation" in str(error)
else:
    checks["run_fails_closed_without_human_review"] = False


def _synthetic_payload(condition_label: str, *, temporal_failure: bool = False) -> dict[str, object]:
    base_values = {
        "zero": 1.90,
        "fixed": 1.95,
        "oracle": 0.70,
        "signed_feedback": 1.10,
        "observation_deranged": 2.00,
    }
    if temporal_failure and condition_label == "temporal":
        base_values["zero"] = 2.50
    traces: list[dict[str, object]] = []
    for seed in spec.heldout_seeds:
        for phase in spec.heldout_phase_turns:
            for arm in spec.arms:
                value = base_values[arm]
                traces.append(
                    {
                        "arm": arm,
                        "case": {"seed": seed, "phase_turns": phase},
                        "numerical_gates": {
                            name: True for name in v3.REQUIRED_NUMERICAL_GATES
                        },
                        "windows": [
                            {
                                "window_index": index,
                                "mean_tke": value,
                                "rms_l2_effort": 0.0 if arm == "zero" else 0.25,
                            }
                            for index in range(spec.scoring_windows)
                        ],
                    }
                )
    return {
        "primary_gates": {name: True for name in v3.REQUIRED_PRIMARY_GATES},
        "traces": traces,
    }


synthetic_pass = {
    label: _synthetic_payload(label) for label in ("base", "temporal", "spatial")
}
pass_analysis = v3.analyze_conditions(spec, synthetic_pass)
checks["synthetic_conjunctive_pass_is_accepted"] = pass_analysis["passed"] is True
synthetic_failure = {
    label: _synthetic_payload(label, temporal_failure=True)
    for label in ("base", "temporal", "spatial")
}
failure_analysis = v3.analyze_conditions(spec, synthetic_failure)
checks["synthetic_temporal_failure_is_rejected"] = failure_analysis["passed"] is False

failed = sorted(name for name, passed in checks.items() if passed is not True)
if failed:
    raise RuntimeError("v3 zero-CFD protocol review failed: " + ", ".join(failed))

result = {
    "action": "re100_gate0_v3_protocol_review",
    "cfds_executed": 0,
    "checks": checks,
    "checks_passed": len(checks),
    "expected_trajectory_count": spec.expected_trajectory_count,
    "implementation_digest": implementation_digest,
    "output_dir": str(output_dir),
    "protocol_artifact_digest": protocol["artifact_digest"],
    "reserved_cases_opened": False,
    "rl_training_performed": False,
    "study_fingerprint": spec.fingerprint,
    "wheel_sha256": wheel_sha256,
}
dbutils.notebook.exit(_canonical(result))
