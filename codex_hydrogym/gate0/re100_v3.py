"""Held-out Re=100 Gate 0 v3 with full causal arms and ensemble convergence.

The module can freeze and review the protocol without executing CFD.  The run
stage additionally requires a digest-bound human review attestation and a
separate execution token.  It performs no reinforcement learning.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

from codex_hydrogym.gate0.cli import _implementation_manifest
from codex_hydrogym.gate0.protocol import (
    REQUIRED_NUMERICAL_GATES,
    REQUIRED_PRIMARY_GATES,
    ArmTrace,
    Gate0Config,
    Gate0DevelopmentLock,
    HydroGymEpisodeFactory,
    run_gate0,
)


STUDY_ID = "offset_phase_fp64_re100_gate0_v3"
STUDY_SCHEMA_VERSION = "codex_hydrogym.gate0.re100_v3.v1"
REVIEW_SCHEMA_VERSION = "codex_hydrogym.gate0.re100_v3.review.v1"
CLAIM_BOUNDARY = (
    "Independently held-out controllability, causal-ablation, numerical-validity, and "
    "refinement-equivalence gate for fixed hand-designed controllers. No RL, learned policy, "
    "reward proposal, coding-agent comparison, MemAlign, GEPA, deployment, or fluid-improvement "
    "claim is performed. A pass only authorizes repairing and separately evaluating the PPO task "
    "contract."
)
SOURCE_V2_PROTOCOL_FINGERPRINT = (
    "2729e365a5712b824d4d1a2257ade19519aa454509a221790ddcabf2021b32a9"
)
SOURCE_V2_FINAL_REPORT_DIGEST = (
    "e8a19fa835a991f233ed7087c31a7c947c6a61a9a3c04ceb935db2870f8dd463"
)
SOURCE_V2_LOCK_ARTIFACT_DIGEST = (
    "4a71309d5362914856aaceb6009117cba1b01059f9266d95f83d44dfc1041eaf"
)
SOURCE_V2_LOCK_DIGEST = (
    "90c324567e20ab28cec778ad5d5dd53734619f2e04496046d9208e9c229e24df"
)
SOURCE_DIAGNOSTIC_RESULT_DIGEST = (
    "e45eb7f6b19b52d6580462ef22e0efcc87a6cd435916f88694ef25e375777d9b"
)
SOURCE_REPLICATION_STUDY_FINGERPRINT = (
    "269507101a5206fccab3c90504f7a46009f28381070a0d97875a06429fb19b62"
)
SOURCE_REPLICATION_RESULT_DIGEST = (
    "c783ea92679ad9c3d51fc44a612d15f9c2fa4b548c0dd4d1d99133ce3222e35a"
)
HELDOUT_SEED_NAMESPACE = "codex_hydrogym:re100_gate0_v3:heldout_seed:v1"
HELDOUT_SEED_METHOD = (
    "For counter values starting at zero, compute SHA-256(namespace + ':' + decimal counter), "
    "interpret the first eight digest bytes as an unsigned big-endian integer, reduce modulo "
    "2^31-1, and accept a nonzero candidate only when distinct and absent from every prior and "
    "reserved seed; stop after ten acceptances."
)
RESERVED_SEEDS = (907, 1009)
HELDOUT_PHASE_TURNS = (0.1875, 0.6875)
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
FIXED_ACTION = (0.1767766952966369, 0.17677669529663687, 0.0, 0.0)
FEEDBACK_GAIN = 2.0
SOURCE_DEVELOPMENT_FIXED_RMS_L2 = 0.25
SOURCE_DEVELOPMENT_FEEDBACK_RMS_L2 = 0.3313719131690823
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
_SEED_MODULUS = 2**31 - 1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_SOURCE = "codex_hydrogym/gate0/re100_v3.py"


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _encoded(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _artifact(body: Mapping[str, object]) -> dict[str, object]:
    if "artifact_digest" in body:
        raise ValueError("artifact bodies must not define artifact_digest")
    return {**body, "artifact_digest": _digest(body)}


def _validate_artifact(payload: Mapping[str, Any]) -> str:
    artifact_digest = payload.get("artifact_digest")
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if not _is_sha256(artifact_digest) or artifact_digest != _digest(body):
        raise RuntimeError("v3 artifact digest validation failed")
    return str(artifact_digest)


def _write_immutable(path: Path, payload: object) -> None:
    encoded = _encoded(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite non-identical v3 artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def derive_additional_heldout_seeds() -> tuple[int, ...]:
    excluded = set((*PRIOR_SEEDS, *RESERVED_SEEDS))
    selected: list[int] = []
    counter = 0
    while len(selected) < 10:
        material = f"{HELDOUT_SEED_NAMESPACE}:{counter}".encode("ascii")
        candidate = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % _SEED_MODULUS
        counter += 1
        if candidate and candidate not in excluded and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


ADDITIONAL_HELDOUT_SEEDS = derive_additional_heldout_seeds()
HELDOUT_SEEDS = (*RESERVED_SEEDS, *ADDITIONAL_HELDOUT_SEEDS)


@dataclass(frozen=True)
class ExecutionCondition:
    label: str
    grid_size: tuple[int, int]
    dt: float
    arm_relative_difference_limit: float | None
    effect_difference_limit: float | None

    def __post_init__(self) -> None:
        if self.label not in {"base", "temporal", "spatial"}:
            raise ValueError("unsupported v3 condition label")
        if len(self.grid_size) != 2 or self.grid_size[0] != self.grid_size[1]:
            raise ValueError("v3 conditions require square grids")
        if any(type(value) is not int or value <= 0 for value in self.grid_size):
            raise ValueError("v3 grid values must be positive integers")
        if _finite(self.dt, "condition dt") <= 0.0:
            raise ValueError("condition dt must be positive")
        limits = (self.arm_relative_difference_limit, self.effect_difference_limit)
        if self.label == "base" and limits != (None, None):
            raise ValueError("base condition must not define convergence limits")
        if self.label != "base" and any(
            value is None or not math.isfinite(value) or value <= 0.0 for value in limits
        ):
            raise ValueError("refinement conditions require finite positive limits")


CONDITIONS = (
    ExecutionCondition("base", (64, 64), 0.002, None, None),
    ExecutionCondition("temporal", (64, 64), 0.001, 0.02, 0.02),
    ExecutionCondition("spatial", (96, 96), 0.002, 0.05, 0.03),
)


@dataclass(frozen=True)
class Re100Gate0V3Spec:
    study_id: str = STUDY_ID
    schema_version: str = STUDY_SCHEMA_VERSION
    reynolds_number: float = 100.0
    precision: str = "float64"
    heldout_phase_turns: tuple[float, ...] = HELDOUT_PHASE_TURNS
    reserved_seeds: tuple[int, ...] = RESERVED_SEEDS
    additional_heldout_seeds: tuple[int, ...] = ADDITIONAL_HELDOUT_SEEDS
    heldout_seeds: tuple[int, ...] = HELDOUT_SEEDS
    heldout_seed_namespace: str = HELDOUT_SEED_NAMESPACE
    heldout_seed_method: str = HELDOUT_SEED_METHOD
    uncontrolled_burn_in_intervals: int = 100
    controller_warmup_intervals: int = 50
    scoring_windows: int = 2
    intervals_per_window: int = 100
    feedback_gain: float = FEEDBACK_GAIN
    fixed_action: tuple[float, float, float, float] = FIXED_ACTION
    radial_action_bound: float = 0.5
    minimum_relative_effect: float = 0.05
    minimum_absolute_effect: float = 0.005
    seed_cluster_t_critical_95: float = 2.200985160082949
    seed_cluster_t_critical_90: float = 1.7958848187036691
    conditions: tuple[ExecutionCondition, ...] = CONDITIONS
    arms: tuple[str, ...] = ARMS
    effect_pairs: tuple[tuple[str, str, str], ...] = EFFECT_PAIRS
    source_v2_protocol_fingerprint: str = SOURCE_V2_PROTOCOL_FINGERPRINT
    source_v2_final_report_digest: str = SOURCE_V2_FINAL_REPORT_DIGEST
    source_v2_lock_artifact_digest: str = SOURCE_V2_LOCK_ARTIFACT_DIGEST
    source_v2_lock_digest: str = SOURCE_V2_LOCK_DIGEST
    source_diagnostic_result_digest: str = SOURCE_DIAGNOSTIC_RESULT_DIGEST
    source_replication_study_fingerprint: str = SOURCE_REPLICATION_STUDY_FINGERPRINT
    source_replication_result_digest: str = SOURCE_REPLICATION_RESULT_DIGEST
    execution_service: str = "Databricks AI Runtime"
    accelerator_type: str = "GPU_1xH100"
    accelerator_count: int = 1
    prior_or_local_observations_in_analysis: int = 0
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.study_id != STUDY_ID or self.schema_version != STUDY_SCHEMA_VERSION:
            raise ValueError("v3 study identity/schema mismatch")
        if self.reynolds_number != 100.0 or self.precision != "float64":
            raise ValueError("v3 is fixed to Re=100 float64")
        if self.heldout_seed_namespace != HELDOUT_SEED_NAMESPACE:
            raise ValueError("v3 seed namespace mismatch")
        if self.heldout_seed_method != HELDOUT_SEED_METHOD:
            raise ValueError("v3 seed method mismatch")
        if self.reserved_seeds != RESERVED_SEEDS:
            raise ValueError("v3 reserved seeds must remain sealed")
        if self.additional_heldout_seeds != derive_additional_heldout_seeds():
            raise ValueError("v3 additional held-out seeds do not reproduce")
        if self.heldout_seeds != (*self.reserved_seeds, *self.additional_heldout_seeds):
            raise ValueError("v3 held-out seed ordering mismatch")
        if len(self.heldout_seeds) != 12 or len(set(self.heldout_seeds)) != 12:
            raise ValueError("v3 requires exactly 12 distinct held-out seed clusters")
        if set(self.heldout_seeds) & set(PRIOR_SEEDS):
            raise ValueError("v3 held-out seeds overlap prior development seeds")
        if self.heldout_phase_turns != HELDOUT_PHASE_TURNS or not math.isclose(
            (self.heldout_phase_turns[0] + 0.5) % 1.0,
            self.heldout_phase_turns[1],
            abs_tol=1.0e-12,
        ):
            raise ValueError("v3 phases must remain the reserved opposite-phase pair")
        if (
            self.uncontrolled_burn_in_intervals,
            self.controller_warmup_intervals,
            self.scoring_windows,
            self.intervals_per_window,
        ) != (100, 50, 2, 100):
            raise ValueError("v3 horizons must remain frozen")
        if self.feedback_gain != FEEDBACK_GAIN or self.fixed_action != FIXED_ACTION:
            raise ValueError("v3 controllers must remain frozen")
        if self.radial_action_bound != 0.5:
            raise ValueError("v3 radial bound must remain 0.5")
        if self.minimum_relative_effect != 0.05 or self.minimum_absolute_effect != 0.005:
            raise ValueError("v3 materiality floors must remain unchanged")
        if self.seed_cluster_t_critical_95 != 2.200985160082949:
            raise ValueError("v3 95% t critical must match 12 clusters (df=11)")
        if self.seed_cluster_t_critical_90 != 1.7958848187036691:
            raise ValueError("v3 90% t critical must match 12 clusters (df=11)")
        if self.conditions != CONDITIONS or self.arms != ARMS or self.effect_pairs != EFFECT_PAIRS:
            raise ValueError("v3 conditions, arms, and effects must remain frozen")
        if (self.execution_service, self.accelerator_type, self.accelerator_count) != (
            "Databricks AI Runtime",
            "GPU_1xH100",
            1,
        ):
            raise ValueError("v3 execution backend must remain one AIR H100")
        if self.prior_or_local_observations_in_analysis != 0:
            raise ValueError("v3 cannot pool prior or local observations")
        source_digests = (
            self.source_v2_protocol_fingerprint,
            self.source_v2_final_report_digest,
            self.source_v2_lock_artifact_digest,
            self.source_v2_lock_digest,
            self.source_diagnostic_result_digest,
            self.source_replication_study_fingerprint,
            self.source_replication_result_digest,
        )
        if any(not _is_sha256(value) for value in source_digests):
            raise ValueError("v3 source evidence identifiers must be SHA-256 digests")
        if self.claim_boundary != CLAIM_BOUNDARY:
            raise ValueError("v3 claim boundary must remain frozen")

    @property
    def scored_intervals(self) -> int:
        return self.scoring_windows * self.intervals_per_window

    @property
    def expected_case_count_per_condition(self) -> int:
        return len(self.heldout_seeds) * len(self.heldout_phase_turns)

    @property
    def expected_trajectory_count(self) -> int:
        return len(self.conditions) * self.expected_case_count_per_condition * len(self.arms)

    @property
    def expected_window_count(self) -> int:
        return self.expected_trajectory_count * self.scoring_windows

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _digest(self.as_dict())


def materialized_gate_config(spec: Re100Gate0V3Spec) -> Gate0Config:
    return replace(
        Gate0Config(),
        protocol_id=spec.study_id,
        heldout_phase_turns=spec.heldout_phase_turns,
        heldout_seeds=spec.heldout_seeds,
        convergence_seed=spec.heldout_seeds[0],
        reynolds_number=spec.reynolds_number,
        grid_size=spec.conditions[0].grid_size,
        precision=spec.precision,
        dt=spec.conditions[0].dt,
        uncontrolled_burn_in_intervals=spec.uncontrolled_burn_in_intervals,
        controller_warmup_intervals=spec.controller_warmup_intervals,
        scored_intervals=spec.scored_intervals,
        radial_action_bound=spec.radial_action_bound,
        minimum_relative_tke_reduction=spec.minimum_relative_effect,
        minimum_absolute_tke_reduction=spec.minimum_absolute_effect,
        spatial_refinement_grid_size=spec.conditions[2].grid_size,
        temporal_refinement_dt=spec.conditions[1].dt,
        maximum_temporal_arm_tke_relative_difference=float(
            spec.conditions[1].arm_relative_difference_limit
        ),
        maximum_temporal_effect_difference=float(spec.conditions[1].effect_difference_limit),
        maximum_spatial_arm_tke_relative_difference=float(
            spec.conditions[2].arm_relative_difference_limit
        ),
        maximum_spatial_effect_difference=float(spec.conditions[2].effect_difference_limit),
    )


def frozen_controller_lock(config: Gate0Config) -> Gate0DevelopmentLock:
    return Gate0DevelopmentLock(
        protocol_fingerprint=config.fingerprint,
        fixed_action=FIXED_ACTION,
        feedback_gain=FEEDBACK_GAIN,
        development_fixed_rms_l2=SOURCE_DEVELOPMENT_FIXED_RMS_L2,
        development_feedback_rms_l2=SOURCE_DEVELOPMENT_FEEDBACK_RMS_L2,
        development_case_ids=("source_v2_lock_digest:" + SOURCE_V2_LOCK_DIGEST,),
        search_scores=(),
        selection_rule=(
            "No v3 tuning. Copy the fixed action and gain from the digest-bound v2 development "
            "lock; gain 2 was separately retained unchanged in the four- and ten-seed development "
            "diagnostics."
        ),
    )


def _implementation_manifest_v3() -> tuple[dict[str, str], str]:
    implementation_files, _ = _implementation_manifest()
    implementation_files[_CONFIG_SOURCE] = hashlib.sha256(
        (_REPOSITORY_ROOT / _CONFIG_SOURCE).read_bytes()
    ).hexdigest()
    return implementation_files, _digest(implementation_files)


def _analysis_contract(spec: Re100Gate0V3Spec) -> dict[str, object]:
    return {
        "independent_unit": "seed_cluster",
        "within_seed_aggregation": "mean across the two phases and two scoring windows",
        "required_primary_gates": sorted(REQUIRED_PRIMARY_GATES),
        "required_numerical_gates": sorted(REQUIRED_NUMERICAL_GATES),
        "base_feedback_zero_every_block": True,
        "robust_feedback_effect_pairs": list(ROBUST_FEEDBACK_PAIRS),
        "robust_effect_floor": spec.minimum_relative_effect,
        "robust_effect_requirements": (
            "each condition and effect pair requires both window means and the seed-clustered "
            "two-sided 95% t-interval lower bound to meet the effect floor"
        ),
        "refinement_effect_pairs": [name for name, _baseline, _candidate in spec.effect_pairs],
        "temporal_arm_relative_margin": spec.conditions[1].arm_relative_difference_limit,
        "temporal_effect_equivalence_margin": spec.conditions[1].effect_difference_limit,
        "spatial_arm_relative_margin": spec.conditions[2].arm_relative_difference_limit,
        "spatial_effect_equivalence_margin": spec.conditions[2].effect_difference_limit,
        "refinement_requirements": [
            "all five arm point differences inside the arm margin",
            "all five aggregate effect differences inside the effect margin",
            "all five paired-seed 90% intervals wholly inside the effect margin",
            "exact five-arm ordering preservation",
            "exact primary decision-vector preservation",
            "all condition-level causal, robust-effect, and numerical predicates pass",
        ],
        "failure_rule": "any false predicate is a terminal v3 failure",
        "interim_decision_looks_allowed": False,
        "seed_replacement_allowed": False,
        "post_result_sample_extension_allowed": False,
        "multiplicity_rule": "conjunctive gates; no failed predicate may be dropped",
    }


def _protocol_payload(
    spec: Re100Gate0V3Spec,
    config: Gate0Config,
    lock: Gate0DevelopmentLock,
    implementation_files: Mapping[str, str],
    implementation_digest: str,
) -> dict[str, object]:
    return _artifact(
        {
            "status": "frozen_before_execution_review_required",
            "study_fingerprint": spec.fingerprint,
            "implementation_digest": implementation_digest,
            "implementation_files": dict(implementation_files),
            "spec": spec.as_dict(),
            "materialized_gate_config": config.as_dict(),
            "materialized_gate_config_fingerprint": config.fingerprint,
            "frozen_controller_lock": asdict(lock),
            "frozen_controller_lock_digest": lock.digest,
            "analysis_contract": _analysis_contract(spec),
            "execution_plan": {
                "condition_count": len(spec.conditions),
                "heldout_seed_cluster_count": len(spec.heldout_seeds),
                "phase_count": len(spec.heldout_phase_turns),
                "arm_count": len(spec.arms),
                "expected_case_count_per_condition": spec.expected_case_count_per_condition,
                "expected_trajectory_count": spec.expected_trajectory_count,
                "expected_window_count": spec.expected_window_count,
                "execution_service": spec.execution_service,
                "accelerator_type": spec.accelerator_type,
                "accelerator_count": spec.accelerator_count,
                "precision": spec.precision,
                "prior_or_local_observations_in_analysis": 0,
                "review_attestation_required": True,
                "separate_execution_token_required": True,
            },
            "predecessor_evidence": {
                "v2_protocol_fingerprint": spec.source_v2_protocol_fingerprint,
                "v2_final_report_digest": spec.source_v2_final_report_digest,
                "v2_lock_artifact_digest": spec.source_v2_lock_artifact_digest,
                "v2_lock_digest": spec.source_v2_lock_digest,
                "four_seed_diagnostic_result_digest": spec.source_diagnostic_result_digest,
                "ten_seed_replication_study_fingerprint": (
                    spec.source_replication_study_fingerprint
                ),
                "ten_seed_replication_result_digest": spec.source_replication_result_digest,
                "prior_observations_in_v3_analysis": 0,
            },
            "execution_authorized": False,
            "reserved_cases_opened": False,
            "rl_training_performed": False,
            "claim_boundary": spec.claim_boundary,
        }
    )


def _window_payload(
    records: Sequence[Any],
    index: int,
    start_state_digest: str,
) -> dict[str, object]:
    interval_tke = tuple(_finite(record.mean_tke, "interval TKE") for record in records)
    interval_effort = tuple(_finite(record.action_l2, "interval effort") for record in records)
    state_digests = tuple(str(record.state_digest) for record in records)
    if any(not _is_sha256(value) for value in state_digests):
        raise RuntimeError("v3 trace contains an invalid state digest")
    return {
        "window_index": index,
        "interval_count": len(records),
        "mean_tke": fmean(interval_tke),
        "rms_l2_effort": math.sqrt(fmean(value**2 for value in interval_effort)),
        "interval_mean_tke": interval_tke,
        "interval_action_l2": interval_effort,
        "interval_state_digests": state_digests,
        "start_state_digest": start_state_digest,
        "end_state_digest": state_digests[-1],
    }


def _trace_payload(
    trace: ArmTrace,
    condition: ExecutionCondition,
    spec: Re100Gate0V3Spec,
) -> dict[str, object]:
    if len(trace.records) != spec.scored_intervals:
        raise RuntimeError("v3 trace has an unexpected scored horizon")
    windows = tuple(
        _window_payload(
            trace.records[
                index * spec.intervals_per_window : (index + 1) * spec.intervals_per_window
            ],
            index,
            (
                trace.scored_start_digest
                if index == 0
                else trace.records[index * spec.intervals_per_window - 1].state_digest
            ),
        )
        for index in range(spec.scoring_windows)
    )
    controller_inputs = tuple(tuple(value) for value in trace.controller_input_history)
    actions = tuple(tuple(value) for value in trace.action_history)
    state_history = tuple(record.state_digest for record in trace.records)
    return {
        "condition": asdict(condition),
        "arm": trace.arm,
        "case": asdict(trace.case),
        "uses_live_observation": trace.uses_live_observation,
        "source_case_id": trace.source_case_id,
        "initial_state_digest": trace.initial_state_digest,
        "uncontrolled_reset_prelude_intervals": trace.uncontrolled_reset_prelude_intervals,
        "explicit_uncontrolled_intervals": trace.explicit_uncontrolled_intervals,
        "control_start_digest": trace.control_start_digest,
        "scored_start_digest": trace.scored_start_digest,
        "mean_tke": trace.mean_tke,
        "rms_l2_effort": trace.rms_l2_effort,
        "numerical_gates": dict(trace.numerical_gates),
        "controller_input_history": controller_inputs,
        "action_history": actions,
        "controller_input_history_digest": _digest(controller_inputs),
        "action_history_digest": _digest(actions),
        "state_history_digest": _digest(state_history),
        "windows": windows,
    }


def _run_condition(
    spec: Re100Gate0V3Spec,
    config: Gate0Config,
    lock: Gate0DevelopmentLock,
    condition: ExecutionCondition,
    implementation_digest: str,
) -> dict[str, object]:
    factory = HydroGymEpisodeFactory(
        config,
        execution_grid_size=condition.grid_size,
        execution_dt=condition.dt,
    )
    report = run_gate0(config, lock, factory)
    body: dict[str, object] = {
        "status": "completed",
        "study_fingerprint": spec.fingerprint,
        "implementation_digest": implementation_digest,
        "frozen_controller_lock_digest": lock.digest,
        "condition": asdict(condition),
        "case_ids": tuple(case.case_id for case in config.cases("heldout")),
        "primary_gates": dict(report.gates),
        "paired_seed_deltas": {
            baseline: {str(seed): value for seed, value in deltas.items()}
            for baseline, deltas in report.paired_seed_deltas.items()
        },
        "traces": tuple(_trace_payload(trace, condition, spec) for trace in report.traces),
        "rl_training_performed": False,
        "claim_boundary": spec.claim_boundary,
    }
    return _artifact(body)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def _expected_cases(config: Gate0Config) -> dict[str, dict[str, object]]:
    return {case.case_id: asdict(case) for case in config.cases("heldout")}


def _expected_derangement(config: Gate0Config) -> dict[str, str]:
    cases = config.cases("heldout")
    by_cell = {(case.phase_index, case.seed_index): case for case in cases}
    phase_count = len(config.heldout_phase_turns)
    seed_count = len(config.heldout_seeds)
    return {
        case.case_id: by_cell[
            ((case.phase_index + 1) % phase_count, (case.seed_index + 1) % seed_count)
        ].case_id
        for case in cases
    }


def _validate_trace(
    trace: Mapping[str, Any],
    *,
    condition: ExecutionCondition,
    spec: Re100Gate0V3Spec,
    expected_cases: Mapping[str, Mapping[str, object]],
) -> tuple[str, str]:
    if trace.get("condition") != json.loads(_canonical(asdict(condition))):
        raise RuntimeError("v3 trace condition mismatch")
    arm = trace.get("arm")
    if arm not in ARMS:
        raise RuntimeError("v3 trace arm mismatch")
    case = trace.get("case")
    if not isinstance(case, dict):
        raise RuntimeError("v3 trace case is missing")
    case_id = next(
        (candidate for candidate, expected in expected_cases.items() if case == expected),
        None,
    )
    if case_id is None:
        raise RuntimeError("v3 trace case identity mismatch")
    for name in ("initial_state_digest", "control_start_digest", "scored_start_digest"):
        if not _is_sha256(trace.get(name)):
            raise RuntimeError(f"v3 trace {name} is invalid")
    prelude = trace.get("uncontrolled_reset_prelude_intervals")
    explicit = trace.get("explicit_uncontrolled_intervals")
    if (
        type(prelude) is not int
        or type(explicit) is not int
        or prelude < 0
        or explicit < 0
        or prelude + explicit != spec.uncontrolled_burn_in_intervals
    ):
        raise RuntimeError("v3 trace uncontrolled horizon mismatch")
    expected_live = arm == "signed_feedback"
    if trace.get("uses_live_observation") is not expected_live:
        raise RuntimeError("v3 trace live-observation flag mismatch")
    gates = trace.get("numerical_gates")
    if not isinstance(gates, dict) or set(gates) != REQUIRED_NUMERICAL_GATES:
        raise RuntimeError("v3 numerical-gate schema mismatch")
    if any(type(value) is not bool for value in gates.values()):
        raise RuntimeError("v3 numerical-gate values must be booleans")

    expected_history_length = spec.controller_warmup_intervals + spec.scored_intervals
    controller_inputs = trace.get("controller_input_history")
    actions = trace.get("action_history")
    if (
        not isinstance(controller_inputs, list)
        or not isinstance(actions, list)
        or len(controller_inputs) != expected_history_length
        or len(actions) != expected_history_length
    ):
        raise RuntimeError("v3 controller/action history length mismatch")
    normalized_inputs = tuple(tuple(_finite(value, "controller input") for value in item) for item in controller_inputs)
    normalized_actions = tuple(tuple(_finite(value, "action") for value in item) for item in actions)
    if any(len(value) != 2 for value in normalized_inputs):
        raise RuntimeError("v3 controller inputs must be two-dimensional")
    if any(len(value) != 4 for value in normalized_actions):
        raise RuntimeError("v3 actions must be four-dimensional")
    if trace.get("controller_input_history_digest") != _digest(normalized_inputs):
        raise RuntimeError("v3 controller-input history digest mismatch")
    if trace.get("action_history_digest") != _digest(normalized_actions):
        raise RuntimeError("v3 action history digest mismatch")

    windows = trace.get("windows")
    if not isinstance(windows, list) or len(windows) != spec.scoring_windows:
        raise RuntimeError("v3 scoring-window count mismatch")
    expected_start = trace["scored_start_digest"]
    all_tke: list[float] = []
    all_effort: list[float] = []
    all_states: list[str] = []
    for index, window in enumerate(windows):
        if window.get("window_index") != index:
            raise RuntimeError("v3 scoring-window index mismatch")
        if window.get("interval_count") != spec.intervals_per_window:
            raise RuntimeError("v3 scoring-window length mismatch")
        if window.get("start_state_digest") != expected_start:
            raise RuntimeError("v3 scoring windows are not state-contiguous")
        tke = window.get("interval_mean_tke")
        effort = window.get("interval_action_l2")
        states = window.get("interval_state_digests")
        if not isinstance(tke, list) or not isinstance(effort, list) or not isinstance(states, list):
            raise RuntimeError("v3 window histories are missing")
        if not (len(tke) == len(effort) == len(states) == spec.intervals_per_window):
            raise RuntimeError("v3 window history length mismatch")
        tke_values = [_finite(value, "window TKE") for value in tke]
        effort_values = [_finite(value, "window effort") for value in effort]
        if any(value < 0.0 for value in (*tke_values, *effort_values)):
            raise RuntimeError("v3 window contains a negative metric")
        if any(not _is_sha256(value) for value in states):
            raise RuntimeError("v3 window state digest is invalid")
        if window.get("end_state_digest") != states[-1]:
            raise RuntimeError("v3 window end-state digest mismatch")
        if not _close(float(window["mean_tke"]), fmean(tke_values)):
            raise RuntimeError("v3 window TKE mean does not reproduce")
        if not _close(
            float(window["rms_l2_effort"]),
            math.sqrt(fmean(value**2 for value in effort_values)),
        ):
            raise RuntimeError("v3 window effort RMS does not reproduce")
        expected_start = states[-1]
        all_tke.extend(tke_values)
        all_effort.extend(effort_values)
        all_states.extend(str(value) for value in states)
    if trace.get("state_history_digest") != _digest(tuple(all_states)):
        raise RuntimeError("v3 state-history digest mismatch")
    if not _close(float(trace["mean_tke"]), fmean(all_tke)):
        raise RuntimeError("v3 trace TKE mean does not reproduce")
    if not _close(
        float(trace["rms_l2_effort"]),
        math.sqrt(fmean(value**2 for value in all_effort)),
    ):
        raise RuntimeError("v3 trace effort RMS does not reproduce")
    return case_id, str(arm)


def _load_condition(
    path: Path,
    spec: Re100Gate0V3Spec,
    config: Gate0Config,
    lock: Gate0DevelopmentLock,
    condition: ExecutionCondition,
    implementation_digest: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_artifact(payload)
    if payload.get("status") != "completed":
        raise RuntimeError("v3 condition is not completed")
    if payload.get("study_fingerprint") != spec.fingerprint:
        raise RuntimeError("v3 condition study fingerprint mismatch")
    if payload.get("implementation_digest") != implementation_digest:
        raise RuntimeError("v3 condition implementation digest mismatch")
    if payload.get("frozen_controller_lock_digest") != lock.digest:
        raise RuntimeError("v3 condition controller lock mismatch")
    if payload.get("condition") != json.loads(_canonical(asdict(condition))):
        raise RuntimeError("v3 condition execution identity mismatch")
    if payload.get("claim_boundary") != spec.claim_boundary:
        raise RuntimeError("v3 condition claim boundary mismatch")
    if payload.get("rl_training_performed") is not False:
        raise RuntimeError("v3 condition cannot contain RL training")
    gates = payload.get("primary_gates")
    if not isinstance(gates, dict) or set(gates) != REQUIRED_PRIMARY_GATES:
        raise RuntimeError("v3 primary-gate schema mismatch")
    if any(type(value) is not bool for value in gates.values()):
        raise RuntimeError("v3 primary-gate values must be booleans")

    expected_cases = _expected_cases(config)
    if set(payload.get("case_ids", ())) != set(expected_cases):
        raise RuntimeError("v3 condition case set mismatch")
    traces = payload.get("traces")
    if not isinstance(traces, list) or len(traces) != len(expected_cases) * len(ARMS):
        raise RuntimeError("v3 condition trace count mismatch")
    seen: dict[tuple[str, str], Mapping[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            raise RuntimeError("v3 trace payload must be an object")
        case_id, arm = _validate_trace(
            trace,
            condition=condition,
            spec=spec,
            expected_cases=expected_cases,
        )
        key = (case_id, arm)
        if key in seen:
            raise RuntimeError("v3 condition contains a duplicate trace")
        seen[key] = trace
    expected_pairs = {(case_id, arm) for case_id in expected_cases for arm in ARMS}
    if set(seen) != expected_pairs:
        raise RuntimeError("v3 condition exact trace set mismatch")

    derangement = _expected_derangement(config)
    for case_id in expected_cases:
        initial_states = {seen[(case_id, arm)]["initial_state_digest"] for arm in ARMS}
        control_starts = {seen[(case_id, arm)]["control_start_digest"] for arm in ARMS}
        if len(initial_states) != 1 or len(control_starts) != 1:
            raise RuntimeError("v3 paired arms do not share developed states")
        for arm in ARMS:
            source = seen[(case_id, arm)].get("source_case_id")
            if arm == "observation_deranged":
                if source != derangement[case_id]:
                    raise RuntimeError("v3 observation derangement mismatch")
            elif source is not None:
                raise RuntimeError("v3 non-deranged trace has a source case")

    aligned_inputs = [
        tuple(value)
        for case_id in expected_cases
        for value in seen[(case_id, "signed_feedback")]["controller_input_history"]
    ]
    deranged_inputs = [
        tuple(value)
        for case_id in expected_cases
        for value in seen[(case_id, "observation_deranged")]["controller_input_history"]
    ]
    aligned_actions = [
        tuple(value)
        for case_id in expected_cases
        for value in seen[(case_id, "signed_feedback")]["action_history"]
    ]
    deranged_actions = [
        tuple(value)
        for case_id in expected_cases
        for value in seen[(case_id, "observation_deranged")]["action_history"]
    ]
    recomputed_integrity_gates = {
        "matched_initial_states": True,
        "uncontrolled_horizon_exact": True,
        "numerical_validity": all(
            all(trace["numerical_gates"].values()) for trace in traces
        ),
        "phase_and_seed_derangement": True,
        "observation_marginal_exact": Counter(aligned_inputs) == Counter(deranged_inputs),
        "action_marginal_exact": Counter(aligned_actions) == Counter(deranged_actions),
        "rotation_invariant_effort_exact": math.isclose(
            sum(action[0] ** 2 + action[1] ** 2 for action in aligned_actions),
            sum(action[0] ** 2 + action[1] ** 2 for action in deranged_actions),
            rel_tol=0.0,
            abs_tol=config.effort_match_atol,
        ),
    }
    if any(gates[name] is not value for name, value in recomputed_integrity_gates.items()):
        raise RuntimeError("v3 stored integrity gate does not reproduce")
    return payload


def _mean_ci(values: Sequence[float], t_critical: float) -> dict[str, float]:
    if len(values) != 12:
        raise ValueError("v3 confidence intervals require exactly 12 seed clusters")
    mean = fmean(values)
    standard_error = stdev(values) / math.sqrt(len(values))
    half_width = t_critical * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _blocks(
    payload: Mapping[str, Any],
) -> dict[tuple[int, float, int], dict[str, dict[str, float]]]:
    blocks: dict[tuple[int, float, int], dict[str, dict[str, float]]] = {}
    for trace in payload["traces"]:
        case = trace["case"]
        for window in trace["windows"]:
            key = (
                int(case["seed"]),
                float(case["phase_turns"]),
                int(window["window_index"]),
            )
            blocks.setdefault(key, {})[str(trace["arm"])] = {
                "mean_tke": float(window["mean_tke"]),
                "rms_l2_effort": float(window["rms_l2_effort"]),
            }
    expected_count = 12 * 2 * 2
    if len(blocks) != expected_count or any(set(values) != set(ARMS) for values in blocks.values()):
        raise RuntimeError("v3 analysis block identity mismatch")
    return blocks


def _relative_effect(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1.0e-12)


def _condition_metrics(
    spec: Re100Gate0V3Spec,
    payload: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, dict[int, float]]]:
    blocks = _blocks(payload)
    arm_mean_tke = {
        arm: fmean(values[arm]["mean_tke"] for values in blocks.values()) for arm in ARMS
    }
    arm_rms_l2_effort = {
        arm: math.sqrt(
            fmean(values[arm]["rms_l2_effort"] ** 2 for values in blocks.values())
        )
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
            for seed in spec.heldout_seeds
        }
        seed_effects_by_pair[name] = seed_effects
        aggregate_effect = _relative_effect(arm_mean_tke[baseline], arm_mean_tke[candidate])
        effect_metrics[name] = {
            "baseline_arm": baseline,
            "candidate_arm": candidate,
            "aggregate_relative_effect": aggregate_effect,
            "seed_cluster_relative_effects": seed_effects,
            "seed_cluster_effect_ci_95": _mean_ci(
                tuple(seed_effects.values()),
                spec.seed_cluster_t_critical_95,
            ),
            "window_relative_effect_means": {
                str(index): fmean(
                    effect
                    for (_seed, _phase, window), effect in block_effects.items()
                    if window == index
                )
                for index in range(spec.scoring_windows)
            },
            "block_win_fraction": sum(value > 0.0 for value in block_effects.values())
            / len(block_effects),
            "minimum_block_relative_effect": min(block_effects.values()),
        }
    primary_gates = {str(name): bool(value) for name, value in payload["primary_gates"].items()}
    robust_effects = {
        name: (
            float(effect_metrics[name]["seed_cluster_effect_ci_95"]["lower"])
            >= spec.minimum_relative_effect
            and all(
                float(value) >= spec.minimum_relative_effect
                for value in effect_metrics[name]["window_relative_effect_means"].values()
            )
        )
        for name in ROBUST_FEEDBACK_PAIRS
    }
    screening = {
        "all_primary_gates": all(value is True for value in primary_gates.values()),
        "all_numerical_gates": all(
            all(value is True for value in trace["numerical_gates"].values())
            for trace in payload["traces"]
        ),
        "robust_feedback_vs_zero": robust_effects["feedback_vs_zero"],
        "robust_feedback_vs_fixed": robust_effects["feedback_vs_fixed"],
        "robust_feedback_vs_deranged": robust_effects["feedback_vs_deranged"],
    }
    return (
        {
            "arm_mean_tke": arm_mean_tke,
            "arm_rms_l2_effort": arm_rms_l2_effort,
            "arm_order": tuple(sorted(ARMS, key=lambda arm: (arm_mean_tke[arm], arm))),
            "effect_metrics": effect_metrics,
            "primary_gates": primary_gates,
            "screening": screening,
        },
        seed_effects_by_pair,
    )


def analyze_conditions(
    spec: Re100Gate0V3Spec,
    condition_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    expected_labels = {condition.label for condition in spec.conditions}
    if set(condition_payloads) != expected_labels:
        raise ValueError("v3 analysis requires the exact frozen condition set")
    block_keys = {label: set(_blocks(payload)) for label, payload in condition_payloads.items()}
    if any(keys != block_keys["base"] for keys in block_keys.values()):
        raise RuntimeError("v3 conditions do not contain identical paired blocks")

    metrics: dict[str, dict[str, object]] = {}
    seed_effects: dict[str, dict[str, dict[int, float]]] = {}
    for condition in spec.conditions:
        condition_metric, condition_seed_effects = _condition_metrics(
            spec,
            condition_payloads[condition.label],
        )
        metrics[condition.label] = condition_metric
        seed_effects[condition.label] = condition_seed_effects

    base = metrics["base"]
    base_feedback_zero = base["effect_metrics"]["feedback_vs_zero"]
    base_screening = {
        **base["screening"],
        "feedback_beats_zero_in_every_block": base_feedback_zero["block_win_fraction"] == 1.0,
    }

    refinement_metrics: dict[str, dict[str, object]] = {}
    refinement_screening: dict[str, dict[str, bool]] = {}
    for condition in spec.conditions[1:]:
        label = condition.label
        current = metrics[label]
        arm_limit = float(condition.arm_relative_difference_limit)
        effect_limit = float(condition.effect_difference_limit)
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
                for seed in spec.heldout_seeds
            )
            interval = _mean_ci(paired_seed_differences, spec.seed_cluster_t_critical_90)
            effect_equivalence[name] = {
                "paired_seed_effect_difference_ci_90": interval,
                "equivalence_margin": effect_limit,
                "supported": interval["lower"] >= -effect_limit
                and interval["upper"] <= effect_limit,
            }
        primary_gates_match = current["primary_gates"] == base["primary_gates"]
        condition_screening_passed = all(
            value is True for value in current["screening"].values()
        )
        refinement_screening[label] = {
            "condition_causal_and_numerical_screening": condition_screening_passed,
            "all_arm_point_differences_inside_margin": max(arm_differences.values())
            <= arm_limit,
            "all_effect_point_differences_inside_margin": max(effect_differences.values())
            <= effect_limit,
            "all_effect_equivalence_intervals_inside_margin": all(
                value["supported"] is True for value in effect_equivalence.values()
            ),
            "arm_ordering_preserved": current["arm_order"] == base["arm_order"],
            "primary_decision_vector_preserved": primary_gates_match,
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

    screening: dict[str, bool] = {
        f"base_{name}": bool(value) for name, value in base_screening.items()
    }
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
        "claim_boundary": spec.claim_boundary,
    }


def _validate_review_attestation(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    spec: Re100Gate0V3Spec,
    implementation_digest: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_artifact(payload)
    if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise RuntimeError("v3 review attestation schema mismatch")
    if payload.get("study_fingerprint") != spec.fingerprint:
        raise RuntimeError("v3 review attestation study mismatch")
    if payload.get("protocol_artifact_digest") != protocol["artifact_digest"]:
        raise RuntimeError("v3 review attestation protocol mismatch")
    if payload.get("implementation_digest") != implementation_digest:
        raise RuntimeError("v3 review attestation implementation mismatch")
    if payload.get("decision") != "approved_for_one_full_execution":
        raise RuntimeError("v3 review attestation does not approve one execution")
    if payload.get("reserved_cases_opened_before_approval") is not False:
        raise RuntimeError("v3 review attestation does not preserve reserved cases")
    if not isinstance(payload.get("reviewed_by"), str) or not payload["reviewed_by"].strip():
        raise RuntimeError("v3 review attestation reviewer is missing")
    if not _is_sha256(payload.get("execution_token_sha256")):
        raise RuntimeError("v3 review attestation execution-token hash is invalid")
    return payload


def _default_output(spec: Re100Gate0V3Spec, implementation_digest: str) -> Path:
    return Path("codex_hydrogym/evidence/gate0_v3") / (
        f"{spec.fingerprint[:12]}-{implementation_digest[:12]}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "review", "run"), default="review")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--review-attestation", type=Path)
    args = parser.parse_args(argv)

    spec = Re100Gate0V3Spec()
    config = materialized_gate_config(spec)
    lock = frozen_controller_lock(config)
    implementation_files, implementation_digest = _implementation_manifest_v3()
    output = args.output_dir or _default_output(spec, implementation_digest)
    expected_protocol = _protocol_payload(
        spec,
        config,
        lock,
        implementation_files,
        implementation_digest,
    )
    protocol_path = output / "protocol.json"
    if args.stage == "freeze":
        _write_immutable(protocol_path, expected_protocol)
    elif not protocol_path.exists():
        raise RuntimeError("freeze and review the exact v3 protocol before any execution")
    stored_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_artifact(stored_protocol)
    if _encoded(stored_protocol) != _encoded(expected_protocol):
        raise RuntimeError("stored v3 protocol differs from the frozen implementation")

    passed: bool | None = None
    if args.stage == "run":
        if args.review_attestation is None:
            raise RuntimeError("v3 run requires a digest-bound human review attestation")
        review = _validate_review_attestation(
            args.review_attestation,
            protocol=stored_protocol,
            spec=spec,
            implementation_digest=implementation_digest,
        )
        execution_token = os.environ.get("CODEX_HYDROGYM_GATE0_V3_EXECUTION_TOKEN")
        if not execution_token or hashlib.sha256(execution_token.encode("utf-8")).hexdigest() != review[
            "execution_token_sha256"
        ]:
            raise RuntimeError("v3 execution token does not match the approved review attestation")
        payloads: dict[str, dict[str, Any]] = {}
        condition_digests: dict[str, str] = {}
        for condition in spec.conditions:
            condition_path = output / f"condition_{condition.label}.json"
            if not condition_path.exists():
                _write_immutable(
                    condition_path,
                    _run_condition(
                        spec,
                        config,
                        lock,
                        condition,
                        implementation_digest,
                    ),
                )
            payload = _load_condition(
                condition_path,
                spec,
                config,
                lock,
                condition,
                implementation_digest,
            )
            payloads[condition.label] = payload
            condition_digests[condition.label] = str(payload["artifact_digest"])
        analysis = analyze_conditions(spec, payloads)
        result = _artifact(
            {
                "status": "completed",
                "study_fingerprint": spec.fingerprint,
                "implementation_digest": implementation_digest,
                "protocol_artifact_digest": stored_protocol["artifact_digest"],
                "review_attestation_artifact_digest": review["artifact_digest"],
                "frozen_controller_lock_digest": lock.digest,
                "condition_artifact_digests": condition_digests,
                "fixed_seed_cluster_count": len(spec.heldout_seeds),
                "trajectory_count": spec.expected_trajectory_count,
                "window_count": spec.expected_window_count,
                "prior_or_local_observations_in_analysis": 0,
                "analysis": analysis,
            }
        )
        result_path = output / "result.json"
        _write_immutable(result_path, result)
        stored_result = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_artifact(stored_result)
        if _encoded(stored_result) != _encoded(result):
            raise RuntimeError("stored v3 result does not round-trip")
        passed = bool(analysis["passed"])

    print(
        json.dumps(
            {
                "output_dir": str(output),
                "study_fingerprint": spec.fingerprint,
                "implementation_digest": implementation_digest,
                "protocol_artifact_digest": expected_protocol["artifact_digest"],
                "stage": args.stage,
                "expected_trajectory_count": spec.expected_trajectory_count,
                "reserved_cases_opened": args.stage == "run",
                "gate_passed": passed,
                "rl_training_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
