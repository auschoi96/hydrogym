# Databricks notebook source
# ruff: noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC # HydroGym ensemble-replication independent audit
# MAGIC
# MAGIC This notebook performs a **zero-CFD**, standard-library-only audit of the completed
# MAGIC Databricks execution. It does not import the production analyzer. It independently checks
# MAGIC frozen identities, raw and canonical hashes, exact cases and arms, numerical gates, paired
# MAGIC state identities, scoring-window arithmetic, confidence intervals, convergence predicates,
# MAGIC and the final decision. It cannot authorize Gate 0, PPO, or a fluid-improvement claim.

# COMMAND ----------

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
import zipfile


STUDY_FINGERPRINT = "269507101a5206fccab3c90504f7a46009f28381070a0d97875a06429fb19b62"
IMPLEMENTATION_DIGEST = "a5ab894e5ff4d3b669da274771f247e58f06aab992873fc9fe76dfdcf8622d8c"
PROTOCOL_ARTIFACT_DIGEST = "3914aedc99979693bf693772a56eef83c3c242c6cd72dc7fda8c07583d781c87"
WHEEL_SHA256 = "91ae939efbacfbd8e3e3aedcf07d1c1e02f9dac642e7d8d381c107ba6505ddc1"
TRANSITION_SHA256 = "69f000608ae8b0aa9b0d8f3433ce1108617936e20f9763d77fe398ad5c96a428"
TRANSITION_ARTIFACT_DIGEST = "deb256b550dd7d3d0fc88746db4ca7b0cbcdeb60f8b921d4fe19bb0466ad2e8a"
BACKEND_AMENDMENT_SHA256 = "ec39d51fdbe288080a52730f5d665acea4ee8bcb1ee00c26f7900a2107156bdf"
BACKEND_AMENDMENT_ARTIFACT_DIGEST = (
    "28747c56c53d2dd251dee8f17f49ef1c67c5d5b01d5fc9b3ea9d1b8f8c84e181"
)
PACKAGED_PROTOCOL = "codex_hydrogym/evidence/ensemble_replication/269507101a52-a5ab894e5ff4/protocol.json"
PRIMARY_OUTPUT_DIR = (
    "/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication/"
    "evidence/269507101a52-a5ab894e5ff4/databricks-primary-20260825"
)
WORKSPACE_ROOT = "/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication"
SEEDS = (
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
)
PHASES = (0.0625, 0.5625)
CONDITIONS = {
    "base": {
        "label": "base",
        "grid_size": [64, 64],
        "dt": 0.002,
        "arm_relative_difference_limit": None,
        "effect_difference_limit": None,
    },
    "temporal": {
        "label": "temporal",
        "grid_size": [64, 64],
        "dt": 0.001,
        "arm_relative_difference_limit": 0.02,
        "effect_difference_limit": 0.02,
    },
    "spatial": {
        "label": "spatial",
        "grid_size": [96, 96],
        "dt": 0.002,
        "arm_relative_difference_limit": 0.05,
        "effect_difference_limit": 0.03,
    },
}
REQUIRED_NUMERICAL_GATES = {
    "cfl_controlled",
    "finite_state_and_metrics",
    "incompressible_velocity",
    "nonnegative_tke",
    "reward_tke_identity",
    "spectral_tail_controlled",
    "zero_mean_vorticity",
}
T_CRITICAL_95 = 2.262157162798205
T_CRITICAL_90 = 1.8331129326562368
MINIMUM_EFFECT = 0.05
SCORING_WINDOWS = 2
INTERVALS_PER_WINDOW = 100


def _widget(name: str, default: str) -> str:
    try:
        return str(dbutils.widgets.get(name))
    except Exception:
        dbutils.widgets.text(name, default)
        return str(dbutils.widgets.get(name))


OUTPUT_DIR = Path(_widget("output_dir", PRIMARY_OUTPUT_DIR))
WHEEL_PATH = Path(_widget("wheel_path", f"{WORKSPACE_ROOT}/hydrogym-1.0.0-py3-none-any.whl"))
TRANSITION_PATH = Path(_widget("transition_path", f"{WORKSPACE_ROOT}/platform_transition.json"))
BACKEND_AMENDMENT_PATH = Path(
    _widget("backend_amendment_path", f"{WORKSPACE_ROOT}/execution_backend_amendment.json")
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def _load_artifact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_digest = payload.get("artifact_digest")
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if artifact_digest != _digest(body):
        raise RuntimeError(f"canonical artifact digest mismatch: {path}")
    return payload


def _mean_ci(values: list[float], t_critical: float) -> dict[str, float]:
    if len(values) != len(SEEDS):
        raise RuntimeError("confidence interval does not contain the exact ten seed clusters")
    mean = fmean(values)
    standard_error = stdev(values) / math.sqrt(len(values))
    half_width = t_critical * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _expected_cases() -> dict[str, dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    for seed_index, seed in enumerate(SEEDS):
        for phase_index, phase in enumerate(PHASES):
            case_id = f"development_p{phase_index:02d}_s{seed_index:02d}_{seed}"
            cases[case_id] = {
                "split": "development",
                "phase_index": phase_index,
                "phase_turns": phase,
                "seed_index": seed_index,
                "seed": seed,
            }
    return cases


EXPECTED_CASES = _expected_cases()


def _validate_trace_set(payload: dict[str, object], label: str) -> None:
    if payload.get("status") != "completed":
        raise RuntimeError(f"{label}: condition is not completed")
    if payload.get("study_fingerprint") != STUDY_FINGERPRINT:
        raise RuntimeError(f"{label}: study fingerprint mismatch")
    if payload.get("implementation_digest") != IMPLEMENTATION_DIGEST:
        raise RuntimeError(f"{label}: implementation digest mismatch")
    if payload.get("condition") != CONDITIONS[label]:
        raise RuntimeError(f"{label}: execution condition mismatch")
    if set(payload.get("case_ids", [])) != set(EXPECTED_CASES):
        raise RuntimeError(f"{label}: case-id set mismatch")
    traces = payload.get("traces")
    if not isinstance(traces, list) or len(traces) != 40:
        raise RuntimeError(f"{label}: expected exactly 40 traces")

    seen: dict[tuple[str, str], dict[str, object]] = {}
    digest_names = (
        "initial_state_digest",
        "control_start_digest",
        "scored_start_digest",
        "controller_input_history_digest",
        "action_history_digest",
        "state_history_digest",
    )
    for trace in traces:
        if not isinstance(trace, dict) or trace.get("condition") != CONDITIONS[label]:
            raise RuntimeError(f"{label}: trace condition mismatch")
        case = trace.get("case")
        if not isinstance(case, dict):
            raise RuntimeError(f"{label}: trace case is missing")
        case_id = next(
            (candidate for candidate, expected in EXPECTED_CASES.items() if case == expected),
            None,
        )
        arm = trace.get("arm")
        if case_id is None or arm not in {"zero", "signed_feedback"}:
            raise RuntimeError(f"{label}: unexpected case or arm")
        key = (case_id, str(arm))
        if key in seen:
            raise RuntimeError(f"{label}: duplicate case/arm trace")
        seen[key] = trace
        if any(not _is_sha256(trace.get(name)) for name in digest_names):
            raise RuntimeError(f"{label}: invalid trace history digest")

        gates = trace.get("numerical_gates")
        if not isinstance(gates, dict) or set(gates) != REQUIRED_NUMERICAL_GATES:
            raise RuntimeError(f"{label}: numerical-gate schema mismatch")
        if any(value is not True for value in gates.values()):
            raise RuntimeError(f"{label}: numerical gate failed")

        windows = trace.get("windows")
        if not isinstance(windows, list) or len(windows) != SCORING_WINDOWS:
            raise RuntimeError(f"{label}: scoring-window count mismatch")
        expected_start = trace["scored_start_digest"]
        all_tke: list[float] = []
        all_effort: list[float] = []
        for index, window in enumerate(windows):
            if not isinstance(window, dict):
                raise RuntimeError(f"{label}: invalid scoring window")
            if window.get("window_index") != index or window.get("interval_count") != INTERVALS_PER_WINDOW:
                raise RuntimeError(f"{label}: scoring-window identity mismatch")
            if window.get("start_state_digest") != expected_start or not _is_sha256(
                window.get("end_state_digest")
            ):
                raise RuntimeError(f"{label}: scoring windows are not state-contiguous")
            expected_start = window["end_state_digest"]
            tke = [float(value) for value in window.get("interval_mean_tke", [])]
            effort = [float(value) for value in window.get("interval_action_l2", [])]
            if len(tke) != INTERVALS_PER_WINDOW or len(effort) != INTERVALS_PER_WINDOW:
                raise RuntimeError(f"{label}: interval history length mismatch")
            if any(not math.isfinite(value) or value < 0.0 for value in (*tke, *effort)):
                raise RuntimeError(f"{label}: invalid interval metric")
            if not _close(float(window["mean_tke"]), fmean(tke)):
                raise RuntimeError(f"{label}: window TKE mean does not reproduce")
            if not _close(float(window["rms_l2_effort"]), math.sqrt(fmean(value**2 for value in effort))):
                raise RuntimeError(f"{label}: window effort RMS does not reproduce")
            all_tke.extend(tke)
            all_effort.extend(effort)
        if not _close(float(trace["mean_tke"]), fmean(all_tke)):
            raise RuntimeError(f"{label}: trace TKE mean does not reproduce")
        if not _close(float(trace["rms_l2_effort"]), math.sqrt(fmean(value**2 for value in all_effort))):
            raise RuntimeError(f"{label}: trace effort RMS does not reproduce")

    expected_pairs = {
        (case_id, arm) for case_id in EXPECTED_CASES for arm in ("zero", "signed_feedback")
    }
    if set(seen) != expected_pairs:
        raise RuntimeError(f"{label}: exact paired trace set mismatch")
    for case_id in EXPECTED_CASES:
        zero = seen[(case_id, "zero")]
        feedback = seen[(case_id, "signed_feedback")]
        if zero["initial_state_digest"] != feedback["initial_state_digest"]:
            raise RuntimeError(f"{label}: paired initial-state mismatch")
        if zero["control_start_digest"] != feedback["control_start_digest"]:
            raise RuntimeError(f"{label}: paired control-start mismatch")


def _blocks(payload: dict[str, object]) -> dict[tuple[int, float, int], dict[str, float]]:
    blocks: dict[tuple[int, float, int], dict[str, float]] = {}
    for trace in payload["traces"]:
        case = trace["case"]
        for window in trace["windows"]:
            key = (int(case["seed"]), float(case["phase_turns"]), int(window["window_index"]))
            blocks.setdefault(key, {})[str(trace["arm"])] = float(window["mean_tke"])
    if len(blocks) != 40 or any(set(arms) != {"zero", "signed_feedback"} for arms in blocks.values()):
        raise RuntimeError("paired block identity mismatch")
    return blocks


def _recompute_analysis(payloads: dict[str, dict[str, object]], claim_boundary: str) -> dict[str, object]:
    blocks_by_condition = {label: _blocks(payload) for label, payload in payloads.items()}
    expected_keys = set(blocks_by_condition["base"])
    if any(set(blocks) != expected_keys for blocks in blocks_by_condition.values()):
        raise RuntimeError("conditions do not contain identical paired blocks")

    condition_metrics: dict[str, dict[str, object]] = {}
    seed_effects_by_condition: dict[str, dict[int, float]] = {}
    for label in ("base", "temporal", "spatial"):
        blocks = blocks_by_condition[label]
        zero_values = [arms["zero"] for arms in blocks.values()]
        feedback_values = [arms["signed_feedback"] for arms in blocks.values()]
        relative_effects = {
            key: (arms["zero"] - arms["signed_feedback"]) / arms["zero"]
            for key, arms in blocks.items()
        }
        seed_effects = {
            seed: fmean(
                effect
                for (block_seed, _phase, _window), effect in relative_effects.items()
                if block_seed == seed
            )
            for seed in SEEDS
        }
        seed_effects_by_condition[label] = seed_effects
        zero_mean = fmean(zero_values)
        feedback_mean = fmean(feedback_values)
        condition_metrics[label] = {
            "zero_mean_tke": zero_mean,
            "feedback_mean_tke": feedback_mean,
            "aggregate_relative_effect": (zero_mean - feedback_mean) / zero_mean,
            "seed_cluster_relative_effects": seed_effects,
            "seed_cluster_effect_ci_95": _mean_ci(list(seed_effects.values()), T_CRITICAL_95),
            "window_relative_effect_means": {
                str(index): fmean(
                    effect
                    for (_seed, _phase, window), effect in relative_effects.items()
                    if window == index
                )
                for index in range(SCORING_WINDOWS)
            },
            "feedback_window_win_fraction": sum(effect > 0.0 for effect in relative_effects.values())
            / len(relative_effects),
            "minimum_window_relative_effect": min(relative_effects.values()),
        }

    base = condition_metrics["base"]
    refinement_metrics: dict[str, dict[str, object]] = {}
    for label in ("temporal", "spatial"):
        condition = CONDITIONS[label]
        metrics = condition_metrics[label]
        paired_differences = [
            seed_effects_by_condition[label][seed] - seed_effects_by_condition["base"][seed]
            for seed in SEEDS
        ]
        effect_ci = _mean_ci(paired_differences, T_CRITICAL_90)
        effect_limit = float(condition["effect_difference_limit"])
        arm_limit = float(condition["arm_relative_difference_limit"])
        zero_difference = abs(float(metrics["zero_mean_tke"]) - float(base["zero_mean_tke"])) / float(
            base["zero_mean_tke"]
        )
        feedback_difference = abs(
            float(metrics["feedback_mean_tke"]) - float(base["feedback_mean_tke"])
        ) / float(base["feedback_mean_tke"])
        effect_difference = abs(
            float(metrics["aggregate_relative_effect"]) - float(base["aggregate_relative_effect"])
        )
        refinement_metrics[label] = {
            "zero_arm_relative_difference": zero_difference,
            "feedback_arm_relative_difference": feedback_difference,
            "maximum_arm_relative_difference": max(zero_difference, feedback_difference),
            "arm_relative_difference_limit": arm_limit,
            "aggregate_effect_difference": effect_difference,
            "effect_difference_limit": effect_limit,
            "paired_seed_effect_difference_ci_90": effect_ci,
            "arm_point_convergence": max(zero_difference, feedback_difference) <= arm_limit,
            "effect_point_convergence": effect_difference <= effect_limit,
            "effect_equivalence_ci_supported": (
                effect_ci["lower"] >= -effect_limit and effect_ci["upper"] <= effect_limit
            ),
        }

    screening = {
        "numerical_validity": all(
            value is True
            for payload in payloads.values()
            for trace in payload["traces"]
            for value in trace["numerical_gates"].values()
        ),
        "feedback_wins_every_window": all(
            metrics["feedback_window_win_fraction"] == 1.0 for metrics in condition_metrics.values()
        ),
        "positive_seed_cluster_effect_ci": all(
            metrics["seed_cluster_effect_ci_95"]["lower"] >= MINIMUM_EFFECT
            for metrics in condition_metrics.values()
        ),
        "both_window_means_material": all(
            value >= MINIMUM_EFFECT
            for metrics in condition_metrics.values()
            for value in metrics["window_relative_effect_means"].values()
        ),
        "temporal_arm_point_convergence": refinement_metrics["temporal"]["arm_point_convergence"],
        "temporal_effect_point_convergence": refinement_metrics["temporal"]["effect_point_convergence"],
        "temporal_effect_equivalence_ci_supported": refinement_metrics["temporal"][
            "effect_equivalence_ci_supported"
        ],
        "spatial_arm_point_convergence": refinement_metrics["spatial"]["arm_point_convergence"],
        "spatial_effect_point_convergence": refinement_metrics["spatial"]["effect_point_convergence"],
        "spatial_effect_equivalence_ci_supported": refinement_metrics["spatial"][
            "effect_equivalence_ci_supported"
        ],
    }
    return json.loads(
        _canonical(
            {
                "condition_metrics": condition_metrics,
                "refinement_metrics": refinement_metrics,
                "screening": screening,
                "supports_designing_full_gate": all(value is True for value in screening.values()),
                "claim_boundary": claim_boundary,
            }
        )
    )


# COMMAND ----------

if str(OUTPUT_DIR) != PRIMARY_OUTPUT_DIR:
    raise RuntimeError("audit output directory differs from the frozen Databricks namespace")

protocol = _load_artifact(OUTPUT_DIR / "protocol.json")
if protocol.get("artifact_digest") != PROTOCOL_ARTIFACT_DIGEST:
    raise RuntimeError("protocol artifact digest differs from the frozen digest")
if protocol.get("study_fingerprint") != STUDY_FINGERPRINT:
    raise RuntimeError("protocol study fingerprint mismatch")
if protocol.get("implementation_digest") != IMPLEMENTATION_DIGEST:
    raise RuntimeError("protocol implementation digest mismatch")
if tuple(protocol["spec"]["seeds"]) != SEEDS or tuple(protocol["spec"]["phase_turns"]) != PHASES:
    raise RuntimeError("protocol seed or phase set mismatch")
if protocol["spec"]["conditions"] != list(CONDITIONS.values()):
    raise RuntimeError("protocol condition set mismatch")
if protocol["execution_plan"]["expected_trajectory_count"] != 120:
    raise RuntimeError("protocol trajectory count mismatch")
if protocol["predecessor_evidence"]["prior_observations_in_replication_analysis"] != 0:
    raise RuntimeError("prior observations entered the replication analysis")

if _sha256(WHEEL_PATH) != WHEEL_SHA256:
    raise RuntimeError("reviewed wheel raw SHA-256 mismatch")
with zipfile.ZipFile(WHEEL_PATH) as archive:
    packaged_protocol = json.loads(archive.read(PACKAGED_PROTOCOL))
    if _canonical(packaged_protocol) != _canonical(protocol):
        raise RuntimeError("remote protocol differs from the wheel-packaged protocol")
    for relative_path, expected_digest in protocol["implementation_files"].items():
        if hashlib.sha256(archive.read(relative_path)).hexdigest() != expected_digest:
            raise RuntimeError(f"wheel implementation source mismatch: {relative_path}")

if _sha256(TRANSITION_PATH) != TRANSITION_SHA256:
    raise RuntimeError("platform-transition raw SHA-256 mismatch")
transition = _load_artifact(TRANSITION_PATH)
if transition.get("artifact_digest") != TRANSITION_ARTIFACT_DIGEST:
    raise RuntimeError("platform-transition artifact digest mismatch")
if transition["local_execution"]["partial_artifact"]["content_or_metrics_inspected"] is not False:
    raise RuntimeError("local partial artifact was not recorded as blind")
if transition["databricks_execution"]["output_namespace"] != str(OUTPUT_DIR):
    raise RuntimeError("platform-transition output namespace mismatch")
if transition["databricks_execution"]["prior_or_local_results_in_analysis"] != 0:
    raise RuntimeError("platform transition did not exclude prior/local results")
if transition["databricks_execution"]["sole_analysis_set"] is not True:
    raise RuntimeError("Databricks output is not the sole analysis set")

if _sha256(BACKEND_AMENDMENT_PATH) != BACKEND_AMENDMENT_SHA256:
    raise RuntimeError("backend-amendment raw SHA-256 mismatch")
amendment = _load_artifact(BACKEND_AMENDMENT_PATH)
if amendment.get("artifact_digest") != BACKEND_AMENDMENT_ARTIFACT_DIGEST:
    raise RuntimeError("backend-amendment artifact digest mismatch")
if amendment.get("previous_transition_artifact_digest") != TRANSITION_ARTIFACT_DIGEST:
    raise RuntimeError("backend amendment does not bind the transition")
if amendment.get("authorized_change") != {
    "accelerator_count": 1,
    "accelerator_type": "GPU_1xH100",
    "execution_service": "Databricks AI Runtime",
    "jax_backend": "gpu",
    "precision": "float64",
}:
    raise RuntimeError("backend amendment differs from the reviewed H100 contract")

execution_context = _load_artifact(OUTPUT_DIR / "databricks_execution_context.json")
expected_context = {
    "accelerator_count": 1,
    "accelerator_type": "GPU_1xH100",
    "claim_role": "primary_decision_bearing_execution",
    "compute_backend": "gpu",
    "decision_bearing_execution": True,
    "execution_backend_amendment_artifact_digest": BACKEND_AMENDMENT_ARTIFACT_DIGEST,
    "implementation_digest": IMPLEMENTATION_DIGEST,
    "jax_enable_x64": True,
    "local_partial_artifact_in_analysis": False,
    "platform_transition_artifact_digest": TRANSITION_ARTIFACT_DIGEST,
    "prior_or_local_results_in_analysis": 0,
    "python_major_minor": "3.12",
    "runner_schema": "codex_hydrogym.ensemble_replication_air.v1",
    "sole_analysis_set": True,
    "study_fingerprint": STUDY_FINGERPRINT,
    "wheel_sha256": WHEEL_SHA256,
}
for key, expected in expected_context.items():
    if execution_context.get(key) != expected:
        raise RuntimeError(f"execution context mismatch: {key}")

condition_payloads: dict[str, dict[str, object]] = {}
for label in ("base", "temporal", "spatial"):
    payload = _load_artifact(OUTPUT_DIR / f"condition_{label}.json")
    _validate_trace_set(payload, label)
    condition_payloads[label] = payload

recomputed_analysis = _recompute_analysis(condition_payloads, str(protocol["claim_boundary"]))
result = _load_artifact(OUTPUT_DIR / "result.json")
if result.get("status") != "completed":
    raise RuntimeError("result status is not completed")
if result.get("study_fingerprint") != STUDY_FINGERPRINT:
    raise RuntimeError("result study fingerprint mismatch")
if result.get("implementation_digest") != IMPLEMENTATION_DIGEST:
    raise RuntimeError("result implementation digest mismatch")
if result.get("fixed_seed_cluster_count") != 10 or result.get("trajectory_count") != 120:
    raise RuntimeError("result sample or trajectory count mismatch")
if result.get("prior_observations_in_analysis") != 0:
    raise RuntimeError("result pooled prior observations")
expected_condition_digests = {
    label: payload["artifact_digest"] for label, payload in condition_payloads.items()
}
if result.get("condition_artifact_digests") != expected_condition_digests:
    raise RuntimeError("result does not bind the exact condition artifacts")
if _canonical(result.get("analysis")) != _canonical(recomputed_analysis):
    raise RuntimeError("independently recomputed analysis differs from result.json")

air_summary_path = OUTPUT_DIR / "air_run_summary.json"
air_summary = json.loads(air_summary_path.read_text(encoding="utf-8"))
# The entry point passed a body whose `artifact_digest` was the result digest to
# `_write_immutable_json`; that helper then replaced the field with the digest of
# the complete preimage. Reconstruct that preimage explicitly. This file is not
# a conventional self-validating artifact because it lacks a separately named
# `result_artifact_digest` field.
air_summary_preimage = {**air_summary, "artifact_digest": result["artifact_digest"]}
if air_summary.get("artifact_digest") != _digest(air_summary_preimage):
    raise RuntimeError("AIR summary nested artifact digest does not reproduce")
if air_summary.get("runner_exit_code") != 0:
    raise RuntimeError("scientific runner did not exit successfully")
if air_summary.get("supports_designing_full_gate") is not True:
    raise RuntimeError("AIR summary decision differs from the result")

summary = {
    "action": "independent_audit",
    "artifact_and_source_hashes_valid": True,
    "air_summary_nested_digest_reproduced": True,
    "air_summary_schema_is_self_describing": False,
    "cfds_executed": 0,
    "condition_artifact_digests": expected_condition_digests,
    "implementation_digest": IMPLEMENTATION_DIGEST,
    "independent_analysis_reproduced": True,
    "job_wrapper_result": "failed_after_result_due_to_60_minute_low_gpu_watchdog",
    "numerical_gate_evaluations": 120 * len(REQUIRED_NUMERICAL_GATES),
    "prior_or_local_observations_in_analysis": 0,
    "result_artifact_digest": result["artifact_digest"],
    "screening": result["analysis"]["screening"],
    "study_fingerprint": STUDY_FINGERPRINT,
    "supports_designing_full_gate": result["analysis"]["supports_designing_full_gate"],
    "trace_count": 120,
    "window_count": 240,
}
encoded_summary = _canonical(summary)
print("CODEX_HYDROGYM_AUDIT_JSON=" + encoded_summary)
try:
    dbutils.notebook.exit(encoded_summary)
except NameError:
    print(json.dumps(summary, indent=2, sort_keys=True))
