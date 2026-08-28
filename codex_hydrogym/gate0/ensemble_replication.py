"""Prospectively fixed replication-sizing study for the Re=100 ensemble screen.

This is a new development-only study. It does not append observations to the
completed four-seed diagnostic, open the reserved cases, perform Gate 0 v3, or
authorize reinforcement learning. The exact ten-seed sample and unchanged
screening rule are frozen before any CFD trajectory is executed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from codex_hydrogym.gate0.ensemble_diagnostic import (
    ExecutionCondition,
    _artifact,
    _canonical,
    _digest,
    _encoded,
    _load_condition,
    _run_condition,
    _study_implementation_manifest as _diagnostic_implementation_manifest,
    _validate_artifact,
    _write_immutable,
    analyze_conditions,
    study_gate_config as _diagnostic_gate_config,
)
from codex_hydrogym.gate0.protocol import REQUIRED_NUMERICAL_GATES, Gate0Config


STUDY_SCHEMA_VERSION = "codex_hydrogym.gate0.ensemble_replication.v1"
STUDY_ID = "re100_fresh_seed_windowed_convergence_replication_v1"
CLAIM_BOUNDARY = (
    "Exploratory development-only replication-sizing evidence. The fixed ten-seed sample is analyzed "
    "separately from the completed four-seed diagnostic; results must never be pooled or appended. No "
    "held-out gate, RL, reward proposal, coding-agent comparison, MemAlign, GEPA, deployment, or "
    "fluid-improvement claim is performed. Even a positive screen only supports designing a separately "
    "preregistered full gate."
)
SAMPLING_PLAN = (
    "Execute all ten prospectively selected seed clusters as one fixed sample. Do not inspect partial "
    "condition results for a decision, replace seeds, stop early, or extend the sample after results are seen."
)
SEED_SELECTION_NAMESPACE = "codex_hydrogym.re100_windowed_convergence_replication_v1.seed_selection"
SEED_SELECTION_METHOD = (
    "For counter values starting at zero, compute SHA-256(namespace + ':' + decimal counter), interpret "
    "the first eight digest bytes as an unsigned big-endian integer, reduce modulo 2^31-1, and accept the "
    "nonzero candidate if it is distinct and is not a prior or reserved seed; stop after ten acceptances."
)
SOURCE_DIAGNOSTIC_STUDY_FINGERPRINT = "19927dd9f42cdcda5a7faf938a1a9da7814e4bb3031a1f15b08670ece6dd6caf"
SOURCE_DIAGNOSTIC_RESULT_DIGEST = "e45eb7f6b19b52d6580462ef22e0efcc87a6cd435916f88694ef25e375777d9b"
SOURCE_V2_PROTOCOL_FINGERPRINT = "2729e365a5712b824d4d1a2257ade19519aa454509a221790ddcabf2021b32a9"
SOURCE_V2_FINAL_REPORT_DIGEST = "e8a19fa835a991f233ed7087c31a7c947c6a61a9a3c04ceb935db2870f8dd463"
PRIOR_GATE_SEEDS = (7, 101, 211, 307)
PRIOR_DIAGNOSTIC_SEEDS = (401, 503, 607, 709)
RESERVED_SEEDS = (907, 1009)
DEVELOPMENT_PHASE_TURNS = (0.0625, 0.5625)
RESERVED_PHASE_TURNS = (0.1875, 0.6875)
CONDITIONS = (
    ExecutionCondition("base", (64, 64), 0.002, None, None),
    ExecutionCondition("temporal", (64, 64), 0.001, 0.02, 0.02),
    ExecutionCondition("spatial", (96, 96), 0.002, 0.05, 0.03),
)
_SEED_MODULUS = 2**31 - 1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_SOURCE = "codex_hydrogym/gate0/ensemble_replication.py"


def derive_replication_seeds() -> tuple[int, ...]:
    """Derive the exact prospectively fixed seed set without investigator choice."""
    excluded = set((*PRIOR_GATE_SEEDS, *PRIOR_DIAGNOSTIC_SEEDS, *RESERVED_SEEDS))
    selected: list[int] = []
    counter = 0
    while len(selected) < 10:
        material = f"{SEED_SELECTION_NAMESPACE}:{counter}".encode("ascii")
        candidate = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % _SEED_MODULUS
        counter += 1
        if candidate != 0 and candidate not in excluded and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


REPLICATION_SEEDS = derive_replication_seeds()


def _lower_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class EnsembleReplicationSpec:
    study_id: str = STUDY_ID
    schema_version: str = STUDY_SCHEMA_VERSION
    reynolds_number: float = 100.0
    precision: str = "float64"
    phase_turns: tuple[float, ...] = DEVELOPMENT_PHASE_TURNS
    seeds: tuple[int, ...] = REPLICATION_SEEDS
    reserved_phase_turns: tuple[float, ...] = RESERVED_PHASE_TURNS
    reserved_seeds: tuple[int, ...] = RESERVED_SEEDS
    uncontrolled_burn_in_intervals: int = 100
    controller_warmup_intervals: int = 50
    scoring_windows: int = 2
    intervals_per_window: int = 100
    feedback_gain: float = 2.0
    radial_action_bound: float = 0.5
    minimum_relative_feedback_effect: float = 0.05
    seed_cluster_t_critical_95: float = 2.262157162798205
    seed_cluster_t_critical_90: float = 1.8331129326562368
    seed_selection_namespace: str = SEED_SELECTION_NAMESPACE
    seed_selection_method: str = SEED_SELECTION_METHOD
    sampling_plan: str = SAMPLING_PLAN
    source_diagnostic_study_fingerprint: str = SOURCE_DIAGNOSTIC_STUDY_FINGERPRINT
    source_diagnostic_result_digest: str = SOURCE_DIAGNOSTIC_RESULT_DIGEST
    source_v2_protocol_fingerprint: str = SOURCE_V2_PROTOCOL_FINGERPRINT
    source_v2_final_report_digest: str = SOURCE_V2_FINAL_REPORT_DIGEST
    conditions: tuple[ExecutionCondition, ...] = CONDITIONS
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.study_id != STUDY_ID or self.schema_version != STUDY_SCHEMA_VERSION:
            raise ValueError("replication study identity/schema mismatch")
        if self.reynolds_number != 100.0 or self.precision != "float64":
            raise ValueError("the replication is frozen to Re=100 float64")
        if self.seed_selection_namespace != SEED_SELECTION_NAMESPACE:
            raise ValueError("seed-selection namespace must remain frozen")
        if self.seed_selection_method != SEED_SELECTION_METHOD:
            raise ValueError("seed-selection method must remain frozen")
        if self.seeds != derive_replication_seeds() or len(self.seeds) != 10:
            raise ValueError("the replication requires the exact ten hash-derived seeds")
        excluded = set((*PRIOR_GATE_SEEDS, *PRIOR_DIAGNOSTIC_SEEDS, *RESERVED_SEEDS))
        if set(self.seeds) & excluded or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("replication seeds must be distinct and fresh")
        if self.phase_turns != DEVELOPMENT_PHASE_TURNS:
            raise ValueError("development phases must preserve the prior opposite-phase pair")
        if self.reserved_phase_turns != RESERVED_PHASE_TURNS or self.reserved_seeds != RESERVED_SEEDS:
            raise ValueError("the unopened reserved cases must remain sealed")
        if (
            self.uncontrolled_burn_in_intervals,
            self.controller_warmup_intervals,
            self.scoring_windows,
            self.intervals_per_window,
        ) != (100, 50, 2, 100):
            raise ValueError("replication horizons must preserve the prior diagnostic")
        if self.feedback_gain != 2.0 or self.radial_action_bound != 0.5:
            raise ValueError("replication controller contract mismatch")
        if self.minimum_relative_feedback_effect != 0.05:
            raise ValueError("replication materiality floor must remain five percent")
        if self.seed_cluster_t_critical_95 != 2.262157162798205:
            raise ValueError("95% t critical must match ten seed clusters (df=9)")
        if self.seed_cluster_t_critical_90 != 1.8331129326562368:
            raise ValueError("90% t critical must match ten seed clusters (df=9)")
        if self.conditions != CONDITIONS:
            raise ValueError("execution conditions and old convergence margins must remain frozen")
        if self.sampling_plan != SAMPLING_PLAN or self.claim_boundary != CLAIM_BOUNDARY:
            raise ValueError("sampling plan and claim boundary must remain frozen")
        exact_sources = (
            (self.source_diagnostic_study_fingerprint, SOURCE_DIAGNOSTIC_STUDY_FINGERPRINT),
            (self.source_diagnostic_result_digest, SOURCE_DIAGNOSTIC_RESULT_DIGEST),
            (self.source_v2_protocol_fingerprint, SOURCE_V2_PROTOCOL_FINGERPRINT),
            (self.source_v2_final_report_digest, SOURCE_V2_FINAL_REPORT_DIGEST),
        )
        if any(actual != expected or not _lower_sha256(actual) for actual, expected in exact_sources):
            raise ValueError("replication source evidence identity mismatch")

    @property
    def scored_intervals(self) -> int:
        return self.scoring_windows * self.intervals_per_window

    @property
    def expected_trajectory_count(self) -> int:
        return len(self.conditions) * len(self.seeds) * len(self.phase_turns) * 2

    @property
    def expected_paired_window_block_count(self) -> int:
        return len(self.conditions) * len(self.seeds) * len(self.phase_turns) * self.scoring_windows

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _digest(self.as_dict())


def study_gate_config(spec: EnsembleReplicationSpec) -> Gate0Config:
    """Materialize the unchanged physics contract while keeping held-out cases sealed."""
    config = _diagnostic_gate_config(spec)  # type: ignore[arg-type]
    return replace(
        config,
        minimum_relative_tke_reduction=spec.minimum_relative_feedback_effect,
        maximum_temporal_arm_tke_relative_difference=float(spec.conditions[1].arm_relative_difference_limit),
        maximum_temporal_effect_difference=float(spec.conditions[1].effect_difference_limit),
        maximum_spatial_arm_tke_relative_difference=float(spec.conditions[2].arm_relative_difference_limit),
        maximum_spatial_effect_difference=float(spec.conditions[2].effect_difference_limit),
    )


def analyze_replication_conditions(
    spec: EnsembleReplicationSpec,
    condition_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the unchanged screen and normalize integer seed keys before hashing."""
    analysis = analyze_conditions(spec, condition_payloads)  # type: ignore[arg-type]
    return json.loads(_canonical(analysis))


def _study_implementation_manifest() -> tuple[dict[str, str], str]:
    implementation_files, _ = _diagnostic_implementation_manifest()
    implementation_files[_CONFIG_SOURCE] = hashlib.sha256((_REPOSITORY_ROOT / _CONFIG_SOURCE).read_bytes()).hexdigest()
    return implementation_files, _digest(implementation_files)


def _protocol_payload(
    spec: EnsembleReplicationSpec,
    config: Gate0Config,
    implementation_files: Mapping[str, str],
    implementation_digest: str,
) -> dict[str, object]:
    return _artifact(
        {
            "status": "frozen_before_execution",
            "study_fingerprint": spec.fingerprint,
            "implementation_digest": implementation_digest,
            "implementation_files": dict(implementation_files),
            "spec": spec.as_dict(),
            "materialized_gate_config": config.as_dict(),
            "materialized_gate_config_fingerprint": config.fingerprint,
            "execution_plan": {
                "fixed_seed_cluster_count": len(spec.seeds),
                "development_case_count_per_condition": len(spec.seeds) * len(spec.phase_turns),
                "arm_count": 2,
                "condition_count": len(spec.conditions),
                "expected_trajectory_count": spec.expected_trajectory_count,
                "expected_paired_window_block_count": spec.expected_paired_window_block_count,
                "interim_decision_looks_allowed": False,
                "seed_replacement_allowed": False,
                "post_result_sample_extension_allowed": False,
            },
            "predecessor_evidence": {
                "diagnostic_study_fingerprint": spec.source_diagnostic_study_fingerprint,
                "diagnostic_result_digest": spec.source_diagnostic_result_digest,
                "v2_protocol_fingerprint": spec.source_v2_protocol_fingerprint,
                "v2_final_report_digest": spec.source_v2_final_report_digest,
                "prior_observations_in_replication_analysis": 0,
            },
            "claim_boundary": spec.claim_boundary,
        }
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def _validated_replication_condition(
    path: Path,
    spec: EnsembleReplicationSpec,
    config: Gate0Config,
    condition: ExecutionCondition,
    implementation_digest: str,
) -> dict[str, Any]:
    payload = _load_condition(
        path,
        spec,  # type: ignore[arg-type]
        condition,
        implementation_digest,
    )
    if payload.get("status") != "completed":
        raise RuntimeError("replication condition is not completed")
    expected_condition = json.loads(_canonical(asdict(condition)))
    expected_cases = {_canonical(asdict(case)): case.case_id for case in config.cases("development")}
    seen: dict[tuple[str, str], Mapping[str, Any]] = {}
    for trace in payload["traces"]:
        if trace.get("condition") != expected_condition:
            raise RuntimeError("replication trace condition identity mismatch")
        arm = trace.get("arm")
        if arm not in {"zero", "signed_feedback"}:
            raise RuntimeError("replication trace contains an unexpected arm")
        case_id = expected_cases.get(_canonical(trace.get("case")))
        if case_id is None or (case_id, arm) in seen:
            raise RuntimeError("replication trace case identity is missing or duplicated")
        seen[(case_id, arm)] = trace
        for name in (
            "initial_state_digest",
            "control_start_digest",
            "scored_start_digest",
            "controller_input_history_digest",
            "action_history_digest",
            "state_history_digest",
        ):
            if not _lower_sha256(trace.get(name)):
                raise RuntimeError(f"replication trace {name} is not a SHA-256 digest")
        gates = trace.get("numerical_gates")
        if not isinstance(gates, dict) or set(gates) != REQUIRED_NUMERICAL_GATES:
            raise RuntimeError("replication trace numerical-gate schema mismatch")
        if any(type(value) is not bool for value in gates.values()):
            raise RuntimeError("replication numerical-gate values must be booleans")
        windows = trace.get("windows")
        if not isinstance(windows, list) or len(windows) != spec.scoring_windows:
            raise RuntimeError("replication trace scoring-window count mismatch")
        all_tke: list[float] = []
        all_effort: list[float] = []
        expected_start_digest = trace["scored_start_digest"]
        for index, window in enumerate(windows):
            if window.get("window_index") != index:
                raise RuntimeError("replication scoring-window index mismatch")
            if window.get("interval_count") != spec.intervals_per_window:
                raise RuntimeError("replication scoring-window interval count mismatch")
            if window.get("start_state_digest") != expected_start_digest:
                raise RuntimeError("replication scoring windows are not state-contiguous")
            if not _lower_sha256(window.get("end_state_digest")):
                raise RuntimeError("replication window end-state digest is invalid")
            expected_start_digest = window["end_state_digest"]
            interval_tke = window.get("interval_mean_tke")
            interval_effort = window.get("interval_action_l2")
            if not isinstance(interval_tke, list) or not isinstance(interval_effort, list):
                raise RuntimeError("replication window interval histories are missing")
            if len(interval_tke) != spec.intervals_per_window or len(interval_effort) != len(interval_tke):
                raise RuntimeError("replication window interval-history length mismatch")
            tke_values = [float(value) for value in interval_tke]
            effort_values = [float(value) for value in interval_effort]
            if any(not math.isfinite(value) or value < 0.0 for value in (*tke_values, *effort_values)):
                raise RuntimeError("replication window contains invalid interval metrics")
            if not _close(float(window["mean_tke"]), fmean(tke_values)):
                raise RuntimeError("replication window TKE mean does not reproduce")
            expected_rms = math.sqrt(fmean(value**2 for value in effort_values))
            if not _close(float(window["rms_l2_effort"]), expected_rms):
                raise RuntimeError("replication window effort RMS does not reproduce")
            all_tke.extend(tke_values)
            all_effort.extend(effort_values)
        if not _close(float(trace["mean_tke"]), fmean(all_tke)):
            raise RuntimeError("replication trace TKE mean does not reproduce")
        if not _close(
            float(trace["rms_l2_effort"]),
            math.sqrt(fmean(value**2 for value in all_effort)),
        ):
            raise RuntimeError("replication trace effort RMS does not reproduce")
    expected_pairs = {
        (case.case_id, arm) for case in config.cases("development") for arm in ("zero", "signed_feedback")
    }
    if set(seen) != expected_pairs:
        raise RuntimeError("replication condition does not contain the exact trace set")
    for case in config.cases("development"):
        zero = seen[(case.case_id, "zero")]
        feedback = seen[(case.case_id, "signed_feedback")]
        if zero["initial_state_digest"] != feedback["initial_state_digest"]:
            raise RuntimeError("replication paired arms do not share the initial state")
        if zero["control_start_digest"] != feedback["control_start_digest"]:
            raise RuntimeError("replication paired arms do not share the control-start state")
    return payload


def _default_output(spec: EnsembleReplicationSpec, implementation_digest: str) -> Path:
    return Path("codex_hydrogym/evidence/ensemble_replication") / (
        f"{spec.fingerprint[:12]}-{implementation_digest[:12]}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "run"), default="freeze")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    spec = EnsembleReplicationSpec()
    config = study_gate_config(spec)
    implementation_files, implementation_digest = _study_implementation_manifest()
    output = args.output_dir or _default_output(spec, implementation_digest)
    expected_protocol = _protocol_payload(
        spec,
        config,
        implementation_files,
        implementation_digest,
    )
    protocol_path = output / "protocol.json"
    if args.stage == "freeze":
        _write_immutable(protocol_path, expected_protocol)
    elif not protocol_path.exists():
        raise RuntimeError("freeze and review the replication protocol before execution")
    stored_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_artifact(stored_protocol)
    if _encoded(stored_protocol) != _encoded(expected_protocol):
        raise RuntimeError("stored replication protocol does not match the frozen protocol")

    supports_designing_full_gate: bool | None = None
    if args.stage == "run":
        payloads: dict[str, dict[str, Any]] = {}
        condition_digests: dict[str, str] = {}
        for condition in spec.conditions:
            condition_path = output / f"condition_{condition.label}.json"
            if not condition_path.exists():
                _write_immutable(
                    condition_path,
                    _run_condition(
                        spec,  # type: ignore[arg-type]
                        config,
                        condition,
                        implementation_digest,
                    ),
                )
            payload = _validated_replication_condition(
                condition_path,
                spec,
                config,
                condition,
                implementation_digest,
            )
            payloads[condition.label] = payload
            condition_digests[condition.label] = str(payload["artifact_digest"])
        analysis = analyze_replication_conditions(spec, payloads)
        result_body: dict[str, object] = {
            "status": "completed",
            "study_fingerprint": spec.fingerprint,
            "implementation_digest": implementation_digest,
            "condition_artifact_digests": condition_digests,
            "fixed_seed_cluster_count": len(spec.seeds),
            "trajectory_count": spec.expected_trajectory_count,
            "prior_observations_in_analysis": 0,
            "analysis": analysis,
        }
        result_path = output / "result.json"
        expected_result = _artifact(result_body)
        _write_immutable(result_path, expected_result)
        stored_result = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_artifact(stored_result)
        if _encoded(stored_result) != _encoded(expected_result):
            raise RuntimeError("stored replication result does not round-trip")
        supports_designing_full_gate = bool(analysis["supports_designing_full_gate"])

    print(
        json.dumps(
            {
                "output_dir": str(output),
                "study_fingerprint": spec.fingerprint,
                "implementation_digest": implementation_digest,
                "stage": args.stage,
                "seeds": spec.seeds,
                "expected_trajectory_count": spec.expected_trajectory_count,
                "supports_designing_full_gate": supports_designing_full_gate,
                "rl_training_performed": False,
                "heldout_gate_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
