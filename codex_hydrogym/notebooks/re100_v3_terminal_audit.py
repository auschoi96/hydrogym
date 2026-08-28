# Databricks notebook source
# ruff: noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC # HydroGym Re=100 Gate 0 v3 — terminal independent audit
# MAGIC
# MAGIC This is a **zero-CFD**, **zero-RL**, standard-library-only audit of the one completed
# MAGIC held-out Gate 0 v3 execution. It does not import HydroGym, JAX, NumPy, MLflow, or the
# MAGIC production analyzer. It independently validates artifact identities, the exact 360
# MAGIC trajectories and 720 windows, all 2,520 numerical-gate values, controller and derangement
# MAGIC contracts, primary gates, confidence intervals, refinement predicates, and the terminal
# MAGIC decision.
# MAGIC
# MAGIC The notebook is read-only. It cannot retry the primary, change the frozen margins, authorize
# MAGIC PPO, or support a coding-agent/MemAlign benefit claim.

# COMMAND ----------

from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence


STUDY_FINGERPRINT = "885ff77559dadd18cc54d91a30ecb6a48477a4c2baed46fc728635ea3eae8b38"
IMPLEMENTATION_DIGEST = "17fd18a51e8bfb2e8b6d018e7fe824a9b68921fe38d62d231e6634d6203b9dfe"
PROTOCOL_ARTIFACT_DIGEST = "024039795a851caa0a1ea77580983aa2c869d40d05564f0765fdc56f1920db3f"
PROTOCOL_RAW_SHA256 = "2dc2ee5ac7287d74e99d45bcd69ce7c3af9ef5403d7d3c32ea81e0ec816a74d6"
REVIEW_ARTIFACT_DIGEST = "39b4ab964755ff1ea1d7747939ac77517219faa2648702adbc4d022040902667"
REVIEW_RAW_SHA256 = "4dba8cab5f561ce3800915d3aaa0d5827f33afba526282d1f0b48f758c7a605d"
EXECUTION_CONTEXT_ARTIFACT_DIGEST = (
    "935f7d11dea82e6742e7893a10092f3d2ce9a027427fbfe6478dca8996c99495"
)
EXECUTION_CONTEXT_RAW_SHA256 = "6b24f51a158b46dab355e949f0dd3359d49f6e9b5a2818f6451c955c371811ff"
RESULT_ARTIFACT_DIGEST = "97f7002bf33e87beb02eef1a8f27f0ce73139641a6867f02cb61205ce67b9636"
RESULT_RAW_SHA256 = "04c4d04782a507e7a04b878b8576b6186b7d1fff13c667db74d42a5e71091934"
CONDITION_ARTIFACT_DIGESTS = {
    "base": "2d5f90d618de898b8457a0eeab695352a73521d2b6ea87153c8a823b1ae5a3e8",
    "temporal": "8697ecbd559d147b88f61e2a8cb089561d0a65672979c96dc8314075b0156fb5",
    "spatial": "ac031fd7efd626737e4b2f9fcbcee0e90299b361b0d53b0cf9414ca2f806c3a4",
}
FROZEN_CONTROLLER_LOCK_DIGEST = (
    "5d8390b4f0952bca54d10cb14f44af56acfefb9d85c6909a157a1f32cffe2755"
)
MATERIALIZED_CONFIG_FINGERPRINT = (
    "603d0f8780fe279a1ffa5d97f389ba5b481944aa8ad953a9612056ff27f84cf5"
)
EXECUTION_TOKEN_SHA256 = "f6cf4be64101d4642489de6bd9c9558c67dfb152c3795c56471a3d0a237b51c5"
WHEEL_SHA256 = "e381b42d415b0644fd773be67ef9aab94133289e6559933bf72e079da50e2e51"

PRIMARY_OUTPUT_DIR = (
    "/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_v3/"
    "evidence/885ff77559da-17fd18a51e8b"
)
EXPECTED_NAMESPACE_FILES = {
    "protocol.json",
    "review_attestation.json",
    "databricks_execution_context.json",
    "condition_base.json",
    "condition_temporal.json",
    "condition_spatial.json",
    "result.json",
}

CLAIM_BOUNDARY = (
    "Independently held-out controllability, causal-ablation, numerical-validity, and "
    "refinement-equivalence gate for fixed hand-designed controllers. No RL, learned policy, "
    "reward proposal, coding-agent comparison, MemAlign, GEPA, deployment, or fluid-improvement "
    "claim is performed. A pass only authorizes repairing and separately evaluating the PPO task "
    "contract."
)
HELDOUT_SEED_NAMESPACE = "codex_hydrogym:re100_gate0_v3:heldout_seed:v1"
RESERVED_SEEDS = (907, 1009)
PRIOR_SEEDS = (
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
)
ADDITIONAL_HELDOUT_SEEDS = (
    23642790,
    909551482,
    680171583,
    492410189,
    1839799188,
    1654216587,
    553944181,
    2146419921,
    164508189,
    743151186,
)
HELDOUT_SEEDS = (*RESERVED_SEEDS, *ADDITIONAL_HELDOUT_SEEDS)
PHASES = (0.1875, 0.6875)
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
ARMS = ("zero", "fixed", "oracle", "signed_feedback", "observation_deranged")
EFFECT_PAIRS = (
    ("oracle_vs_zero", "zero", "oracle"),
    ("oracle_vs_fixed", "fixed", "oracle"),
    ("feedback_vs_zero", "zero", "signed_feedback"),
    ("feedback_vs_fixed", "fixed", "signed_feedback"),
    ("feedback_vs_deranged", "observation_deranged", "signed_feedback"),
)
ROBUST_FEEDBACK_PAIRS = (
    "feedback_vs_zero",
    "feedback_vs_fixed",
    "feedback_vs_deranged",
)
REQUIRED_NUMERICAL_GATES = {
    "cfl_controlled",
    "finite_state_and_metrics",
    "incompressible_velocity",
    "nonnegative_tke",
    "reward_tke_identity",
    "spectral_tail_controlled",
    "zero_mean_vorticity",
}
REQUIRED_PRIMARY_GATES = {
    "action_marginal_exact",
    "deranged_independent_seed_block_wins",
    "feedback_beats_fixed_in_opposite_phase_pairs",
    "feedback_materially_beats_deranged",
    "feedback_materially_beats_fixed",
    "feedback_materially_beats_zero",
    "feedback_opposite_phase_pairs_material",
    "feedback_within_oracle_effort_budget",
    "fixed_independent_seed_block_wins",
    "matched_initial_states",
    "numerical_validity",
    "observation_marginal_exact",
    "oracle_beats_fixed_in_opposite_phase_pairs",
    "oracle_materially_beats_fixed",
    "oracle_materially_beats_zero",
    "oracle_opposite_phase_pairs_material",
    "phase_and_seed_derangement",
    "rotation_invariant_effort_exact",
    "source_observation_trajectories_differ",
    "uncontrolled_horizon_exact",
}
EXPECTED_PACKAGES = {
    "chex": "0.1.92",
    "flax": "0.12.0",
    "gymnasium": "1.3.0",
    "gymnax": "0.0.9",
    "jax": "0.7.2",
    "jaxlib": "0.7.2",
    "matplotlib": "3.11.1",
    "navix": "0.11.0",
    "numpy": "2.5.2",
    "omegaconf": "2.3.1",
    "scipy": "1.18.0",
    "toml": "0.10.2",
    "tree-math": "0.2.1",
}

FIXED_ACTION = (0.1767766952966369, 0.17677669529663687, 0.0, 0.0)
FEEDBACK_GAIN = 2.0
RADIAL_ACTION_BOUND = 0.5
UNCONTROLLED_INTERVALS = 100
CONTROLLER_WARMUP_INTERVALS = 50
SCORING_WINDOWS = 2
INTERVALS_PER_WINDOW = 100
SCORED_INTERVALS = SCORING_WINDOWS * INTERVALS_PER_WINDOW
HISTORY_LENGTH = CONTROLLER_WARMUP_INTERVALS + SCORED_INTERVALS
MINIMUM_ABSOLUTE_EFFECT = 0.005
MINIMUM_RELATIVE_EFFECT = 0.05
MINIMUM_SEED_WIN_FRACTION = 2.0 / 3.0
EFFORT_MATCH_ATOL = 1.0e-12
T_CRITICAL_95 = 2.200985160082949
T_CRITICAL_90 = 1.7958848187036691


def _widget(name: str, default: str) -> str:
    try:
        return str(dbutils.widgets.get(name))
    except Exception:
        dbutils.widgets.text(name, default)
        return str(dbutils.widgets.get(name))


OUTPUT_DIR = Path(_widget("output_dir", PRIMARY_OUTPUT_DIR))


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


def _finite(value: object, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"{label} must be finite")
    return converted


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def _load_artifact(
    path: Path,
    *,
    expected_artifact_digest: str,
    expected_raw_sha256: str | None = None,
) -> dict[str, Any]:
    raw_sha256 = _sha256(path)
    if expected_raw_sha256 is not None and raw_sha256 != expected_raw_sha256:
        raise RuntimeError(f"raw SHA-256 mismatch: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"artifact is not an object: {path.name}")
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if payload.get("artifact_digest") != _digest(body):
        raise RuntimeError(f"canonical artifact digest does not reproduce: {path.name}")
    if payload.get("artifact_digest") != expected_artifact_digest:
        raise RuntimeError(f"artifact identity mismatch: {path.name}")
    return payload


def _load_result_with_frozen_integer_key_preimage(
    path: Path,
) -> tuple[dict[str, Any], str]:
    """Validate the result's raw bytes and reconstruct its pre-JSON digest preimage.

    The producer computed ``artifact_digest`` while seed-cluster maps still had integer keys.
    JSON then converted those keys to strings. Sorting integers before serialization and sorting
    their decimal strings after reload are different orders, so the ordinary round-trip digest
    is intentionally expected not to match. This reconstructs exactly the typed preimage without
    accepting that serialization bug for any other artifact.
    """

    if _sha256(path) != RESULT_RAW_SHA256:
        raise RuntimeError("result raw SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("artifact_digest") != RESULT_ARTIFACT_DIGEST:
        raise RuntimeError("result declared artifact identity mismatch")
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    post_json_canonical_digest = _digest(body)
    if post_json_canonical_digest == RESULT_ARTIFACT_DIGEST:
        raise RuntimeError("expected result integer-key round-trip defect is absent")

    typed_preimage = copy.deepcopy(body)
    for label in ("base", "temporal", "spatial"):
        for name, _baseline, _candidate in EFFECT_PAIRS:
            seed_map = typed_preimage["analysis"]["condition_metrics"][label]["effect_metrics"][name][
                "seed_cluster_relative_effects"
            ]
            if set(seed_map) != {str(seed) for seed in HELDOUT_SEEDS}:
                raise RuntimeError("result seed-cluster key set mismatch")
            typed_preimage["analysis"]["condition_metrics"][label]["effect_metrics"][name][
                "seed_cluster_relative_effects"
            ] = {int(seed): value for seed, value in seed_map.items()}
    if _digest(typed_preimage) != RESULT_ARTIFACT_DIGEST:
        raise RuntimeError("result pre-serialization integer-key digest does not reproduce")
    return payload, post_json_canonical_digest


def _derive_additional_heldout_seeds() -> tuple[int, ...]:
    excluded = set((*PRIOR_SEEDS, *RESERVED_SEEDS))
    selected: list[int] = []
    counter = 0
    while len(selected) < 10:
        material = f"{HELDOUT_SEED_NAMESPACE}:{counter}".encode("ascii")
        candidate = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)
        counter += 1
        if candidate and candidate not in excluded and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


def _expected_cases() -> dict[str, dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    for seed_index, seed in enumerate(HELDOUT_SEEDS):
        for phase_index, phase in enumerate(PHASES):
            case_id = f"heldout_p{phase_index:02d}_s{seed_index:02d}_{seed}"
            cases[case_id] = {
                "split": "heldout",
                "phase_index": phase_index,
                "phase_turns": phase,
                "seed_index": seed_index,
                "seed": seed,
            }
    return cases


EXPECTED_CASES = _expected_cases()
CASE_ID_BY_CANONICAL_BODY = {_canonical(case): case_id for case_id, case in EXPECTED_CASES.items()}


def _expected_derangement() -> dict[str, str]:
    by_cell = {
        (int(case["phase_index"]), int(case["seed_index"])): case_id
        for case_id, case in EXPECTED_CASES.items()
    }
    return {
        case_id: by_cell[
            (
                (int(case["phase_index"]) + 1) % len(PHASES),
                (int(case["seed_index"]) + 1) % len(HELDOUT_SEEDS),
            )
        ]
        for case_id, case in EXPECTED_CASES.items()
    }


EXPECTED_DERANGEMENT = _expected_derangement()


def _radial_clip(first: float, second: float) -> tuple[float, float, float, float]:
    radius = math.hypot(first, second)
    if radius > RADIAL_ACTION_BOUND:
        scale = RADIAL_ACTION_BOUND / radius
        first, second = first * scale, second * scale
    return (float(first), float(second), 0.0, 0.0)


def _expected_action(
    arm: str,
    case: Mapping[str, object],
    controller_input: Sequence[float],
) -> tuple[float, float, float, float]:
    if arm == "zero":
        return (0.0, 0.0, 0.0, 0.0)
    if arm == "fixed":
        return FIXED_ACTION
    if arm == "oracle":
        phase_radians = 2.0 * math.pi * float(case["phase_turns"])
        return _radial_clip(
            -RADIAL_ACTION_BOUND * math.cos(phase_radians),
            -RADIAL_ACTION_BOUND * math.sin(phase_radians),
        )
    if arm in {"signed_feedback", "observation_deranged"}:
        return _radial_clip(
            -FEEDBACK_GAIN * float(controller_input[0]),
            -FEEDBACK_GAIN * float(controller_input[1]),
        )
    raise RuntimeError(f"unsupported arm: {arm}")


def _mean_ci(values: Sequence[float], t_critical: float) -> dict[str, float]:
    if len(values) != len(HELDOUT_SEEDS):
        raise RuntimeError("confidence interval does not contain exactly 12 seed clusters")
    mean = fmean(values)
    standard_error = stdev(values) / math.sqrt(len(values))
    half_width = t_critical * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _material(baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> bool:
    baseline_mean = fmean(float(trace["mean_tke"]) for trace in baseline)
    candidate_mean = fmean(float(trace["mean_tke"]) for trace in candidate)
    absolute = baseline_mean - candidate_mean
    relative = absolute / baseline_mean if baseline_mean > 0.0 else -math.inf
    return absolute >= MINIMUM_ABSOLUTE_EFFECT and relative >= MINIMUM_RELATIVE_EFFECT


def _opposite_pairs_material(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> bool:
    baseline_by_cell = {
        (int(trace["case"]["seed"]), float(trace["case"]["phase_turns"])): trace
        for trace in baseline
    }
    candidate_by_cell = {
        (int(trace["case"]["seed"]), float(trace["case"]["phase_turns"])): trace
        for trace in candidate
    }
    checked: set[tuple[float, float]] = set()
    for phase in PHASES:
        opposite = next(
            other for other in PHASES if math.isclose((phase + 0.5) % 1.0, other, abs_tol=1.0e-12)
        )
        pair = tuple(sorted((phase, opposite)))
        if pair in checked:
            continue
        checked.add(pair)
        pair_baseline = [baseline_by_cell[(seed, member)] for seed in HELDOUT_SEEDS for member in pair]
        pair_candidate = [candidate_by_cell[(seed, member)] for seed in HELDOUT_SEEDS for member in pair]
        if not _material(pair_baseline, pair_candidate):
            return False
    return True


def _validate_trace(
    trace: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str, int]:
    if trace.get("condition") != CONDITIONS[label]:
        raise RuntimeError(f"{label}: trace condition mismatch")
    arm = trace.get("arm")
    if arm not in ARMS:
        raise RuntimeError(f"{label}: unexpected arm")
    case = trace.get("case")
    if not isinstance(case, dict):
        raise RuntimeError(f"{label}: trace case is missing")
    case_id = CASE_ID_BY_CANONICAL_BODY.get(_canonical(case))
    if case_id is None:
        raise RuntimeError(f"{label}: unexpected case")
    for name in ("initial_state_digest", "control_start_digest", "scored_start_digest"):
        if not _is_sha256(trace.get(name)):
            raise RuntimeError(f"{label}: invalid {name}")
    prelude = trace.get("uncontrolled_reset_prelude_intervals")
    explicit = trace.get("explicit_uncontrolled_intervals")
    if (
        type(prelude) is not int
        or type(explicit) is not int
        or prelude < 0
        or explicit < 0
        or prelude + explicit != UNCONTROLLED_INTERVALS
    ):
        raise RuntimeError(f"{label}: uncontrolled horizon mismatch")
    if trace.get("uses_live_observation") is not (arm == "signed_feedback"):
        raise RuntimeError(f"{label}: live-observation flag mismatch")

    gates = trace.get("numerical_gates")
    if not isinstance(gates, dict) or set(gates) != REQUIRED_NUMERICAL_GATES:
        raise RuntimeError(f"{label}: numerical-gate schema mismatch")
    if any(type(value) is not bool for value in gates.values()):
        raise RuntimeError(f"{label}: non-boolean numerical gate")

    raw_inputs = trace.get("controller_input_history")
    raw_actions = trace.get("action_history")
    if (
        not isinstance(raw_inputs, list)
        or not isinstance(raw_actions, list)
        or len(raw_inputs) != HISTORY_LENGTH
        or len(raw_actions) != HISTORY_LENGTH
    ):
        raise RuntimeError(f"{label}: controller/action history length mismatch")
    inputs = tuple(tuple(_finite(value, "controller input") for value in item) for item in raw_inputs)
    actions = tuple(tuple(_finite(value, "action") for value in item) for item in raw_actions)
    if any(len(value) != 2 for value in inputs) or any(len(value) != 4 for value in actions):
        raise RuntimeError(f"{label}: controller/action dimensionality mismatch")
    if trace.get("controller_input_history_digest") != _digest(inputs):
        raise RuntimeError(f"{label}: controller-input history digest mismatch")
    if trace.get("action_history_digest") != _digest(actions):
        raise RuntimeError(f"{label}: action history digest mismatch")
    for controller_input, action in zip(inputs, actions, strict=True):
        if not _close(math.hypot(*action), math.hypot(action[0], action[1])):
            raise RuntimeError(f"{label}: action left the frozen two-dimensional subspace")
        if math.hypot(action[0], action[1]) > RADIAL_ACTION_BOUND + 1.0e-12:
            raise RuntimeError(f"{label}: action exceeded radial bound")
        expected_action = _expected_action(str(arm), case, controller_input)
        if any(not _close(observed, expected) for observed, expected in zip(action, expected_action, strict=True)):
            raise RuntimeError(f"{label}: {arm} action does not reproduce from the frozen controller")

    windows = trace.get("windows")
    if not isinstance(windows, list) or len(windows) != SCORING_WINDOWS:
        raise RuntimeError(f"{label}: scoring-window count mismatch")
    expected_start = trace["scored_start_digest"]
    all_tke: list[float] = []
    all_effort: list[float] = []
    all_states: list[str] = []
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise RuntimeError(f"{label}: malformed window")
        if window.get("window_index") != index or window.get("interval_count") != INTERVALS_PER_WINDOW:
            raise RuntimeError(f"{label}: window identity mismatch")
        if window.get("start_state_digest") != expected_start:
            raise RuntimeError(f"{label}: windows are not state-contiguous")
        tke = window.get("interval_mean_tke")
        effort = window.get("interval_action_l2")
        states = window.get("interval_state_digests")
        if not isinstance(tke, list) or not isinstance(effort, list) or not isinstance(states, list):
            raise RuntimeError(f"{label}: missing window histories")
        if not (len(tke) == len(effort) == len(states) == INTERVALS_PER_WINDOW):
            raise RuntimeError(f"{label}: window history length mismatch")
        tke_values = [_finite(value, "window TKE") for value in tke]
        effort_values = [_finite(value, "window effort") for value in effort]
        if any(value < 0.0 for value in (*tke_values, *effort_values)):
            raise RuntimeError(f"{label}: negative window metric")
        if any(not _is_sha256(value) for value in states):
            raise RuntimeError(f"{label}: invalid window state digest")
        if window.get("end_state_digest") != states[-1]:
            raise RuntimeError(f"{label}: window end-state mismatch")
        if not _close(float(window["mean_tke"]), fmean(tke_values)):
            raise RuntimeError(f"{label}: window TKE mean does not reproduce")
        if not _close(
            float(window["rms_l2_effort"]),
            math.sqrt(fmean(value**2 for value in effort_values)),
        ):
            raise RuntimeError(f"{label}: window RMS effort does not reproduce")
        scored_action_offset = CONTROLLER_WARMUP_INTERVALS + index * INTERVALS_PER_WINDOW
        scored_actions = actions[scored_action_offset : scored_action_offset + INTERVALS_PER_WINDOW]
        for stored_effort, action in zip(effort_values, scored_actions, strict=True):
            if not _close(stored_effort, math.hypot(action[0], action[1])):
                raise RuntimeError(f"{label}: interval effort does not match action history")
        expected_start = str(states[-1])
        all_tke.extend(tke_values)
        all_effort.extend(effort_values)
        all_states.extend(str(value) for value in states)
    if trace.get("state_history_digest") != _digest(tuple(all_states)):
        raise RuntimeError(f"{label}: scored state-history digest mismatch")
    if not _close(float(trace["mean_tke"]), fmean(all_tke)):
        raise RuntimeError(f"{label}: trace TKE mean does not reproduce")
    if not _close(float(trace["rms_l2_effort"]), math.sqrt(fmean(value**2 for value in all_effort))):
        raise RuntimeError(f"{label}: trace RMS effort does not reproduce")
    return case_id, str(arm), len(gates)


def _validate_condition(payload: dict[str, Any], label: str) -> dict[str, object]:
    if payload.get("status") != "completed":
        raise RuntimeError(f"{label}: condition is not completed")
    if payload.get("study_fingerprint") != STUDY_FINGERPRINT:
        raise RuntimeError(f"{label}: study mismatch")
    if payload.get("implementation_digest") != IMPLEMENTATION_DIGEST:
        raise RuntimeError(f"{label}: implementation mismatch")
    if payload.get("frozen_controller_lock_digest") != FROZEN_CONTROLLER_LOCK_DIGEST:
        raise RuntimeError(f"{label}: controller-lock mismatch")
    if payload.get("condition") != CONDITIONS[label]:
        raise RuntimeError(f"{label}: condition identity mismatch")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY or payload.get("rl_training_performed") is not False:
        raise RuntimeError(f"{label}: claim boundary or RL flag mismatch")
    primary_gates = payload.get("primary_gates")
    if not isinstance(primary_gates, dict) or set(primary_gates) != REQUIRED_PRIMARY_GATES:
        raise RuntimeError(f"{label}: primary-gate schema mismatch")
    if any(type(value) is not bool for value in primary_gates.values()):
        raise RuntimeError(f"{label}: non-boolean primary gate")
    if payload.get("case_ids") != list(EXPECTED_CASES):
        raise RuntimeError(f"{label}: exact ordered case set mismatch")
    traces = payload.get("traces")
    if not isinstance(traces, list) or len(traces) != len(EXPECTED_CASES) * len(ARMS):
        raise RuntimeError(f"{label}: expected exactly 120 traces")

    seen: dict[tuple[str, str], Mapping[str, Any]] = {}
    numerical_gate_evaluations = 0
    for trace in traces:
        if not isinstance(trace, dict):
            raise RuntimeError(f"{label}: trace is not an object")
        case_id, arm, gate_count = _validate_trace(trace, label=label)
        key = (case_id, arm)
        if key in seen:
            raise RuntimeError(f"{label}: duplicate case/arm trace")
        seen[key] = trace
        numerical_gate_evaluations += gate_count
    expected_pairs = {(case_id, arm) for case_id in EXPECTED_CASES for arm in ARMS}
    if set(seen) != expected_pairs:
        raise RuntimeError(f"{label}: exact case/arm trace set mismatch")

    for case_id in EXPECTED_CASES:
        initial_states = {seen[(case_id, arm)]["initial_state_digest"] for arm in ARMS}
        control_starts = {seen[(case_id, arm)]["control_start_digest"] for arm in ARMS}
        if len(initial_states) != 1 or len(control_starts) != 1:
            raise RuntimeError(f"{label}: paired arms do not share developed states")
        for arm in ARMS:
            source = seen[(case_id, arm)].get("source_case_id")
            if arm == "observation_deranged":
                if source != EXPECTED_DERANGEMENT[case_id]:
                    raise RuntimeError(f"{label}: derangement source mismatch")
                expected_inputs = seen[(str(source), "signed_feedback")]["controller_input_history"]
                if seen[(case_id, arm)]["controller_input_history"] != expected_inputs:
                    raise RuntimeError(f"{label}: deranged history is not the exact source history")
            elif source is not None:
                raise RuntimeError(f"{label}: non-deranged trace has a source case")

    grouped = {arm: [seen[(case_id, arm)] for case_id in EXPECTED_CASES] for arm in ARMS}
    aligned_inputs = [tuple(value) for trace in grouped["signed_feedback"] for value in trace["controller_input_history"]]
    deranged_inputs = [
        tuple(value) for trace in grouped["observation_deranged"] for value in trace["controller_input_history"]
    ]
    aligned_actions = [tuple(value) for trace in grouped["signed_feedback"] for value in trace["action_history"]]
    deranged_actions = [
        tuple(value) for trace in grouped["observation_deranged"] for value in trace["action_history"]
    ]
    observation_marginal_exact = Counter(aligned_inputs) == Counter(deranged_inputs)
    action_marginal_exact = Counter(aligned_actions) == Counter(deranged_actions)
    rotation_invariant_effort_exact = math.isclose(
        sum(action[0] ** 2 + action[1] ** 2 for action in aligned_actions),
        sum(action[0] ** 2 + action[1] ** 2 for action in deranged_actions),
        rel_tol=0.0,
        abs_tol=EFFORT_MATCH_ATOL,
    )
    source_observations_differ = all(
        _digest(trace["controller_input_history"][CONTROLLER_WARMUP_INTERVALS:])
        != _digest(seen[(case_id, "signed_feedback")]["controller_input_history"][CONTROLLER_WARMUP_INTERVALS:])
        for case_id, trace in ((case_id, seen[(case_id, "observation_deranged")]) for case_id in EXPECTED_CASES)
    )

    paired_seed_deltas: dict[str, dict[int, float]] = {}
    for baseline in ("zero", "fixed", "observation_deranged"):
        paired_seed_deltas[baseline] = {
            seed: fmean(
                float(trace["mean_tke"]) for trace in grouped[baseline] if int(trace["case"]["seed"]) == seed
            )
            - fmean(
                float(trace["mean_tke"])
                for trace in grouped["signed_feedback"]
                if int(trace["case"]["seed"]) == seed
            )
            for seed in HELDOUT_SEEDS
        }
    stored_deltas = payload.get("paired_seed_deltas")
    if not isinstance(stored_deltas, dict) or set(stored_deltas) != set(paired_seed_deltas):
        raise RuntimeError(f"{label}: paired-seed delta schema mismatch")
    for baseline, values in paired_seed_deltas.items():
        if set(stored_deltas[baseline]) != {str(seed) for seed in HELDOUT_SEEDS}:
            raise RuntimeError(f"{label}: paired-seed delta seed set mismatch")
        if any(not _close(float(stored_deltas[baseline][str(seed)]), value) for seed, value in values.items()):
            raise RuntimeError(f"{label}: paired-seed deltas do not reproduce")

    recomputed_primary_gates = {
        "matched_initial_states": all(
            len({seen[(case_id, arm)]["control_start_digest"] for arm in ARMS}) == 1
            for case_id in EXPECTED_CASES
        ),
        "uncontrolled_horizon_exact": all(
            int(trace["uncontrolled_reset_prelude_intervals"])
            + int(trace["explicit_uncontrolled_intervals"])
            == UNCONTROLLED_INTERVALS
            for trace in traces
        ),
        "numerical_validity": all(all(trace["numerical_gates"].values()) for trace in traces),
        "phase_and_seed_derangement": all(
            seen[(case_id, "observation_deranged")]["source_case_id"] == EXPECTED_DERANGEMENT[case_id]
            for case_id in EXPECTED_CASES
        ),
        "source_observation_trajectories_differ": source_observations_differ,
        "observation_marginal_exact": observation_marginal_exact,
        "action_marginal_exact": action_marginal_exact,
        "rotation_invariant_effort_exact": rotation_invariant_effort_exact,
        "oracle_materially_beats_zero": _material(grouped["zero"], grouped["oracle"]),
        "oracle_materially_beats_fixed": _material(grouped["fixed"], grouped["oracle"]),
        "oracle_opposite_phase_pairs_material": _opposite_pairs_material(grouped["zero"], grouped["oracle"]),
        "oracle_beats_fixed_in_opposite_phase_pairs": _opposite_pairs_material(
            grouped["fixed"], grouped["oracle"]
        ),
        "feedback_materially_beats_zero": _material(grouped["zero"], grouped["signed_feedback"]),
        "feedback_materially_beats_fixed": _material(grouped["fixed"], grouped["signed_feedback"]),
        "feedback_materially_beats_deranged": _material(
            grouped["observation_deranged"], grouped["signed_feedback"]
        ),
        "feedback_opposite_phase_pairs_material": _opposite_pairs_material(
            grouped["observation_deranged"], grouped["signed_feedback"]
        ),
        "feedback_beats_fixed_in_opposite_phase_pairs": _opposite_pairs_material(
            grouped["fixed"], grouped["signed_feedback"]
        ),
        "feedback_within_oracle_effort_budget": fmean(
            float(trace["rms_l2_effort"]) for trace in grouped["signed_feedback"]
        )
        <= fmean(float(trace["rms_l2_effort"]) for trace in grouped["oracle"]) + EFFORT_MATCH_ATOL,
        "fixed_independent_seed_block_wins": (
            sum(value > 0.0 for value in paired_seed_deltas["fixed"].values()) / len(HELDOUT_SEEDS)
            >= MINIMUM_SEED_WIN_FRACTION
        ),
        "deranged_independent_seed_block_wins": (
            sum(value > 0.0 for value in paired_seed_deltas["observation_deranged"].values())
            / len(HELDOUT_SEEDS)
            >= MINIMUM_SEED_WIN_FRACTION
        ),
    }
    if recomputed_primary_gates != primary_gates:
        differing = sorted(
            name for name in REQUIRED_PRIMARY_GATES if recomputed_primary_gates[name] is not primary_gates[name]
        )
        raise RuntimeError(f"{label}: stored primary gates do not reproduce: {differing}")

    return {
        "trace_count": len(traces),
        "window_count": len(traces) * SCORING_WINDOWS,
        "numerical_gate_evaluations": numerical_gate_evaluations,
        "all_numerical_gates_true": all(all(trace["numerical_gates"].values()) for trace in traces),
        "all_primary_gates_recomputed": all(recomputed_primary_gates.values()),
        "primary_gate_count": len(recomputed_primary_gates),
    }


def _blocks(payload: Mapping[str, Any]) -> dict[tuple[int, float, int], dict[str, dict[str, float]]]:
    blocks: dict[tuple[int, float, int], dict[str, dict[str, float]]] = {}
    for trace in payload["traces"]:
        case = trace["case"]
        for window in trace["windows"]:
            key = (int(case["seed"]), float(case["phase_turns"]), int(window["window_index"]))
            arm = str(trace["arm"])
            if arm in blocks.setdefault(key, {}):
                raise RuntimeError("duplicate arm inside analysis block")
            blocks[key][arm] = {
                "mean_tke": float(window["mean_tke"]),
                "rms_l2_effort": float(window["rms_l2_effort"]),
            }
    if len(blocks) != 48 or any(set(values) != set(ARMS) for values in blocks.values()):
        raise RuntimeError("analysis does not contain the exact 48 paired blocks per condition")
    return blocks


def _relative_effect(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1.0e-12)


def _condition_metrics(payload: Mapping[str, Any]) -> tuple[dict[str, object], dict[str, dict[int, float]]]:
    blocks = _blocks(payload)
    arm_mean_tke = {
        arm: fmean(values[arm]["mean_tke"] for values in blocks.values()) for arm in ARMS
    }
    arm_rms_l2_effort = {
        arm: math.sqrt(fmean(values[arm]["rms_l2_effort"] ** 2 for values in blocks.values()))
        for arm in ARMS
    }
    effect_metrics: dict[str, dict[str, object]] = {}
    seed_effects_by_pair: dict[str, dict[int, float]] = {}
    for name, baseline, candidate in EFFECT_PAIRS:
        block_effects = {
            key: _relative_effect(values[baseline]["mean_tke"], values[candidate]["mean_tke"])
            for key, values in blocks.items()
        }
        seed_effects = {
            seed: fmean(
                effect
                for (block_seed, _phase, _window), effect in block_effects.items()
                if block_seed == seed
            )
            for seed in HELDOUT_SEEDS
        }
        seed_effects_by_pair[name] = seed_effects
        effect_metrics[name] = {
            "baseline_arm": baseline,
            "candidate_arm": candidate,
            "aggregate_relative_effect": _relative_effect(arm_mean_tke[baseline], arm_mean_tke[candidate]),
            "seed_cluster_relative_effects": {str(seed): value for seed, value in seed_effects.items()},
            "seed_cluster_effect_ci_95": _mean_ci(tuple(seed_effects.values()), T_CRITICAL_95),
            "window_relative_effect_means": {
                str(index): fmean(
                    effect
                    for (_seed, _phase, window), effect in block_effects.items()
                    if window == index
                )
                for index in range(SCORING_WINDOWS)
            },
            "block_win_fraction": sum(value > 0.0 for value in block_effects.values()) / len(block_effects),
            "minimum_block_relative_effect": min(block_effects.values()),
        }
    primary_gates = {str(name): bool(value) for name, value in payload["primary_gates"].items()}
    robust_effects = {
        name: (
            float(effect_metrics[name]["seed_cluster_effect_ci_95"]["lower"]) >= MINIMUM_RELATIVE_EFFECT
            and all(
                float(value) >= MINIMUM_RELATIVE_EFFECT
                for value in effect_metrics[name]["window_relative_effect_means"].values()
            )
        )
        for name in ROBUST_FEEDBACK_PAIRS
    }
    screening = {
        "all_primary_gates": all(value is True for value in primary_gates.values()),
        "all_numerical_gates": all(
            all(value is True for value in trace["numerical_gates"].values()) for trace in payload["traces"]
        ),
        "robust_feedback_vs_zero": robust_effects["feedback_vs_zero"],
        "robust_feedback_vs_fixed": robust_effects["feedback_vs_fixed"],
        "robust_feedback_vs_deranged": robust_effects["feedback_vs_deranged"],
    }
    return (
        {
            "arm_mean_tke": arm_mean_tke,
            "arm_rms_l2_effort": arm_rms_l2_effort,
            "arm_order": sorted(ARMS, key=lambda arm: (arm_mean_tke[arm], arm)),
            "effect_metrics": effect_metrics,
            "primary_gates": primary_gates,
            "screening": screening,
        },
        seed_effects_by_pair,
    )


def _recompute_analysis(condition_payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    block_keys = {label: set(_blocks(payload)) for label, payload in condition_payloads.items()}
    if any(keys != block_keys["base"] for keys in block_keys.values()):
        raise RuntimeError("conditions do not contain identical paired block keys")

    metrics: dict[str, dict[str, object]] = {}
    seed_effects: dict[str, dict[str, dict[int, float]]] = {}
    for label in ("base", "temporal", "spatial"):
        condition_metric, condition_seed_effects = _condition_metrics(condition_payloads[label])
        metrics[label] = condition_metric
        seed_effects[label] = condition_seed_effects

    base = metrics["base"]
    base_screening = {
        **base["screening"],
        "feedback_beats_zero_in_every_block": (
            base["effect_metrics"]["feedback_vs_zero"]["block_win_fraction"] == 1.0
        ),
    }
    refinement_metrics: dict[str, dict[str, object]] = {}
    refinement_screening: dict[str, dict[str, bool]] = {}
    for label in ("temporal", "spatial"):
        current = metrics[label]
        condition = CONDITIONS[label]
        arm_limit = float(condition["arm_relative_difference_limit"])
        effect_limit = float(condition["effect_difference_limit"])
        arm_differences = {
            arm: abs(float(current["arm_mean_tke"][arm]) - float(base["arm_mean_tke"][arm]))
            / max(abs(float(base["arm_mean_tke"][arm])), 1.0e-12)
            for arm in ARMS
        }
        effect_differences: dict[str, float] = {}
        effect_equivalence: dict[str, dict[str, object]] = {}
        for name, _baseline, _candidate in EFFECT_PAIRS:
            base_effect = float(base["effect_metrics"][name]["aggregate_relative_effect"])
            current_effect = float(current["effect_metrics"][name]["aggregate_relative_effect"])
            effect_differences[name] = abs(current_effect - base_effect)
            paired_seed_differences = tuple(
                seed_effects[label][name][seed] - seed_effects["base"][name][seed]
                for seed in HELDOUT_SEEDS
            )
            interval = _mean_ci(paired_seed_differences, T_CRITICAL_90)
            effect_equivalence[name] = {
                "paired_seed_effect_difference_ci_90": interval,
                "equivalence_margin": effect_limit,
                "supported": interval["lower"] >= -effect_limit and interval["upper"] <= effect_limit,
            }
        refinement_screening[label] = {
            "condition_causal_and_numerical_screening": all(
                value is True for value in current["screening"].values()
            ),
            "all_arm_point_differences_inside_margin": max(arm_differences.values()) <= arm_limit,
            "all_effect_point_differences_inside_margin": max(effect_differences.values()) <= effect_limit,
            "all_effect_equivalence_intervals_inside_margin": all(
                value["supported"] is True for value in effect_equivalence.values()
            ),
            "arm_ordering_preserved": current["arm_order"] == base["arm_order"],
            "primary_decision_vector_preserved": current["primary_gates"] == base["primary_gates"],
        }
        refinement_metrics[label] = {
            "arm_relative_differences": arm_differences,
            "maximum_arm_relative_difference": max(arm_differences.values()),
            "arm_relative_difference_limit": arm_limit,
            "aggregate_effect_differences": effect_differences,
            "maximum_aggregate_effect_difference": max(effect_differences.values()),
            "effect_difference_limit": effect_limit,
            "effect_equivalence": effect_equivalence,
            "base_arm_order": base["arm_order"],
            "refined_arm_order": current["arm_order"],
        }
    screening: dict[str, bool] = {f"base_{name}": bool(value) for name, value in base_screening.items()}
    for label, values in refinement_screening.items():
        screening.update({f"{label}_{name}": bool(value) for name, value in values.items()})
    return {
        "condition_metrics": metrics,
        "base_screening": base_screening,
        "refinement_metrics": refinement_metrics,
        "refinement_screening": refinement_screening,
        "screening": screening,
        "passed": all(value is True for value in screening.values()),
        "prior_or_local_observations_in_analysis": 0,
        "heldout_gate_performed": True,
        "rl_training_performed": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


# COMMAND ----------

if str(OUTPUT_DIR) != PRIMARY_OUTPUT_DIR:
    raise RuntimeError("audit output_dir differs from the sole frozen decision-bearing namespace")
if not OUTPUT_DIR.is_dir():
    raise RuntimeError("frozen evidence namespace does not exist")
observed_names = {path.name for path in OUTPUT_DIR.iterdir()}
if observed_names != EXPECTED_NAMESPACE_FILES:
    raise RuntimeError(f"terminal namespace differs from the exact seven-file set: {sorted(observed_names)}")
if (OUTPUT_DIR / "air_run_summary.json").exists():
    raise RuntimeError("unexpected AIR run summary appeared after the recorded wrapper failure")

protocol = _load_artifact(
    OUTPUT_DIR / "protocol.json",
    expected_artifact_digest=PROTOCOL_ARTIFACT_DIGEST,
    expected_raw_sha256=PROTOCOL_RAW_SHA256,
)
if _derive_additional_heldout_seeds() != ADDITIONAL_HELDOUT_SEEDS:
    raise RuntimeError("prospective held-out seed derivation does not reproduce")
if protocol.get("study_fingerprint") != STUDY_FINGERPRINT:
    raise RuntimeError("protocol study fingerprint mismatch")
if protocol.get("implementation_digest") != IMPLEMENTATION_DIGEST:
    raise RuntimeError("protocol implementation digest mismatch")
if protocol.get("materialized_gate_config_fingerprint") != MATERIALIZED_CONFIG_FINGERPRINT:
    raise RuntimeError("materialized Gate 0 config fingerprint mismatch")
if protocol.get("frozen_controller_lock_digest") != FROZEN_CONTROLLER_LOCK_DIGEST:
    raise RuntimeError("protocol controller lock mismatch")
spec = protocol.get("spec")
if not isinstance(spec, dict):
    raise RuntimeError("protocol spec is missing")
if tuple(spec.get("heldout_seeds", ())) != HELDOUT_SEEDS or tuple(spec.get("heldout_phase_turns", ())) != PHASES:
    raise RuntimeError("protocol held-out sample mismatch")
if spec.get("conditions") != list(CONDITIONS.values()) or tuple(spec.get("arms", ())) != ARMS:
    raise RuntimeError("protocol condition or arm set mismatch")
if tuple(tuple(value) for value in spec.get("effect_pairs", ())) != EFFECT_PAIRS:
    raise RuntimeError("protocol effect-pair set mismatch")
if spec.get("claim_boundary") != CLAIM_BOUNDARY or protocol.get("claim_boundary") != CLAIM_BOUNDARY:
    raise RuntimeError("protocol claim boundary mismatch")
if (
    protocol.get("execution_authorized") is not False
    or protocol.get("reserved_cases_opened") is not False
    or protocol.get("rl_training_performed") is not False
):
    raise RuntimeError("protocol pre-execution flags were mutated")
plan = protocol.get("execution_plan", {})
if (
    plan.get("expected_trajectory_count") != 360
    or plan.get("expected_window_count") != 720
    or plan.get("prior_or_local_observations_in_analysis") != 0
    or plan.get("accelerator_type") != "GPU_1xH100"
    or plan.get("accelerator_count") != 1
):
    raise RuntimeError("protocol execution plan mismatch")
analysis_contract = protocol.get("analysis_contract", {})
if set(analysis_contract.get("required_numerical_gates", ())) != REQUIRED_NUMERICAL_GATES:
    raise RuntimeError("protocol numerical-gate contract mismatch")
if set(analysis_contract.get("required_primary_gates", ())) != REQUIRED_PRIMARY_GATES:
    raise RuntimeError("protocol primary-gate contract mismatch")
if (
    analysis_contract.get("failure_rule") != "any false predicate is a terminal v3 failure"
    or analysis_contract.get("interim_decision_looks_allowed") is not False
    or analysis_contract.get("seed_replacement_allowed") is not False
    or analysis_contract.get("post_result_sample_extension_allowed") is not False
):
    raise RuntimeError("protocol failure/extension contract mismatch")

review = _load_artifact(
    OUTPUT_DIR / "review_attestation.json",
    expected_artifact_digest=REVIEW_ARTIFACT_DIGEST,
    expected_raw_sha256=REVIEW_RAW_SHA256,
)
if (
    review.get("study_fingerprint") != STUDY_FINGERPRINT
    or review.get("implementation_digest") != IMPLEMENTATION_DIGEST
    or review.get("protocol_artifact_digest") != PROTOCOL_ARTIFACT_DIGEST
    or review.get("execution_token_sha256") != EXECUTION_TOKEN_SHA256
    or review.get("decision") != "approved_for_one_full_execution"
    or review.get("reserved_cases_opened_before_approval") is not False
):
    raise RuntimeError("review attestation does not bind the exact one-execution authorization")
approval = review.get("approval_basis", {})
if (
    approval.get("approved_expected_trajectory_count") != 360
    or approval.get("approved_expected_window_count") != 720
    or approval.get("approved_accelerator_type") != "GPU_1xH100"
    or approval.get("approved_accelerator_count") != 1
    or approval.get("approved_precision") != "float64"
):
    raise RuntimeError("review approval basis mismatch")

execution_context = _load_artifact(
    OUTPUT_DIR / "databricks_execution_context.json",
    expected_artifact_digest=EXECUTION_CONTEXT_ARTIFACT_DIGEST,
    expected_raw_sha256=EXECUTION_CONTEXT_RAW_SHA256,
)
expected_execution_context = {
    "accelerator_count": 1,
    "accelerator_type": "GPU_1xH100",
    "claim_role": "gate0_v3_primary_decision_bearing_execution",
    "compute_backend": "gpu",
    "decision_bearing_execution": True,
    "execution_token_sha256": EXECUTION_TOKEN_SHA256,
    "implementation_digest": IMPLEMENTATION_DIGEST,
    "jax_enable_x64": True,
    "package_versions": EXPECTED_PACKAGES,
    "precision": "float64",
    "prior_or_local_observations_in_analysis": 0,
    "protocol_artifact_digest": PROTOCOL_ARTIFACT_DIGEST,
    "review_attestation_artifact_digest": REVIEW_ARTIFACT_DIGEST,
    "runner_schema_version": "codex_hydrogym.gate0.re100_v3.air.v1",
    "sole_analysis_set": True,
    "study_fingerprint": STUDY_FINGERPRINT,
    "wheel_sha256": WHEEL_SHA256,
}
for key, expected in expected_execution_context.items():
    if execution_context.get(key) != expected:
        raise RuntimeError(f"execution-context mismatch: {key}")
try:
    started = datetime.fromisoformat(str(execution_context["execution_started_at_utc"]))
except (KeyError, ValueError) as error:
    raise RuntimeError("execution context has an invalid UTC timestamp") from error
if started.utcoffset() is None or started.utcoffset().total_seconds() != 0:
    raise RuntimeError("execution start timestamp is not UTC")

condition_payloads: dict[str, dict[str, Any]] = {}
condition_audits: dict[str, dict[str, object]] = {}
condition_raw_sha256: dict[str, str] = {}
for label in ("base", "temporal", "spatial"):
    path = OUTPUT_DIR / f"condition_{label}.json"
    payload = _load_artifact(path, expected_artifact_digest=CONDITION_ARTIFACT_DIGESTS[label])
    condition_payloads[label] = payload
    condition_audits[label] = _validate_condition(payload, label)
    condition_raw_sha256[label] = _sha256(path)

recomputed_analysis = _recompute_analysis(condition_payloads)
result, result_post_json_canonical_digest = _load_result_with_frozen_integer_key_preimage(
    OUTPUT_DIR / "result.json"
)
if (
    result.get("status") != "completed"
    or result.get("study_fingerprint") != STUDY_FINGERPRINT
    or result.get("implementation_digest") != IMPLEMENTATION_DIGEST
    or result.get("protocol_artifact_digest") != PROTOCOL_ARTIFACT_DIGEST
    or result.get("review_attestation_artifact_digest") != REVIEW_ARTIFACT_DIGEST
    or result.get("frozen_controller_lock_digest") != FROZEN_CONTROLLER_LOCK_DIGEST
    or result.get("condition_artifact_digests") != CONDITION_ARTIFACT_DIGESTS
    or result.get("fixed_seed_cluster_count") != 12
    or result.get("trajectory_count") != 360
    or result.get("window_count") != 720
    or result.get("prior_or_local_observations_in_analysis") != 0
):
    raise RuntimeError("terminal result identity/count/provenance mismatch")
if _canonical(result.get("analysis")) != _canonical(recomputed_analysis):
    raise RuntimeError("independent analysis does not exactly reproduce result.json")

false_predicates = sorted(name for name, value in recomputed_analysis["screening"].items() if value is False)
expected_false_predicates = ["temporal_all_effect_equivalence_intervals_inside_margin"]
if false_predicates != expected_false_predicates or recomputed_analysis["passed"] is not False:
    raise RuntimeError("terminal failure predicate set differs from the frozen result")
failed_equivalence = recomputed_analysis["refinement_metrics"]["temporal"]["effect_equivalence"][
    "feedback_vs_zero"
]
failed_interval = failed_equivalence["paired_seed_effect_difference_ci_90"]
margin = float(failed_equivalence["equivalence_margin"])
margin_miss = max(0.0, -margin - float(failed_interval["lower"]), float(failed_interval["upper"]) - margin)
if not _close(margin_miss, 0.000255923821008402):
    raise RuntimeError("failed temporal margin miss does not reproduce")

trace_count = sum(int(value["trace_count"]) for value in condition_audits.values())
window_count = sum(int(value["window_count"]) for value in condition_audits.values())
numerical_gate_evaluations = sum(
    int(value["numerical_gate_evaluations"]) for value in condition_audits.values()
)
if (trace_count, window_count, numerical_gate_evaluations) != (360, 720, 2520):
    raise RuntimeError("terminal audit count mismatch")

summary = {
    "action": "re100_v3_terminal_independent_audit",
    "audit_schema_version": "codex_hydrogym.gate0.re100_v3.terminal_audit.v1",
    "canonical_roundtrip_artifacts_valid_except_result": True,
    "condition_artifact_digests": CONDITION_ARTIFACT_DIGESTS,
    "condition_raw_sha256": condition_raw_sha256,
    "controller_and_derangement_contracts_reproduced": True,
    "cfds_executed": 0,
    "false_predicates": false_predicates,
    "failed_effect_pair": "feedback_vs_zero",
    "failed_temporal_ci_90": failed_interval,
    "frozen_equivalence_margin": [-margin, margin],
    "margin_miss": margin_miss,
    "margin_miss_percentage_points": 100.0 * margin_miss,
    "independent_analysis_exactly_reproduced": True,
    "implementation_digest": IMPLEMENTATION_DIGEST,
    "memalign_benefit_proven": False,
    "coding_agent_benefit_proven": False,
    "namespace_file_count": len(observed_names),
    "numerical_gate_evaluations": numerical_gate_evaluations,
    "ppo_authorized": False,
    "primary_gate_evaluations_recomputed": 3 * len(REQUIRED_PRIMARY_GATES),
    "production_analyzer_imported": False,
    "result_artifact_digest": RESULT_ARTIFACT_DIGEST,
    "result_digest_pre_serialization_integer_key_preimage_valid": True,
    "result_post_json_canonical_digest": result_post_json_canonical_digest,
    "result_roundtrip_canonical_digest_valid": False,
    "result_raw_sha256": RESULT_RAW_SHA256,
    "rl_training_performed": False,
    "scientific_decision": "terminal_gate0_v3_failure",
    "study_fingerprint": STUDY_FINGERPRINT,
    "trace_count": trace_count,
    "window_count": window_count,
    "wrapper_classification": (
        "producer hashed nested integer-key seed maps before JSON converted keys to strings; "
        "immediate result round-trip validation failed before AIR summary and final MLflow upload"
    ),
    "wrapper_root_cause_reproduced": True,
    "wrapper_failure_changes_scientific_decision": False,
}
encoded_summary = _canonical(summary)
print("CODEX_HYDROGYM_V3_TERMINAL_AUDIT_JSON=" + encoded_summary)
try:
    dbutils.notebook.exit(encoded_summary)
except NameError:
    print(json.dumps(summary, indent=2, sort_keys=True))
