"""Local, immutable artifact runner for the preregistered CPU Gate 0."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from codex_hydrogym.gate0.protocol import (
    CLAIM_BOUNDARY,
    ArmTrace,
    DevelopmentCandidateScore,
    Gate0ConvergenceAttestation,
    Gate0ConvergenceRun,
    Gate0Config,
    Gate0Case,
    Gate0DevelopmentLock,
    Gate0DevelopmentSearchError,
    Gate0RefinementEvidence,
    Gate0RefinementProvenance,
    Gate0RefinementTraceSummary,
    Gate0Report,
    HydroGymEpisodeFactory,
    StepRecord,
    lock_development_controls,
    run_gate0,
    run_gate0_convergence,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_FILES = (
    "codex_hydrogym/gate0/cli.py",
    "codex_hydrogym/gate0/protocol.py",
    "hydrogym/jax/envs/kolmogorov.py",
    "hydrogym/jax/kolmogorov_contract.py",
    "hydrogym/jax/solvers/base.py",
    "hydrogym/jax/utils/utils.py",
)


def _encoded(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _implementation_manifest() -> tuple[dict[str, str], str]:
    manifest = {
        relative: hashlib.sha256((_REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
        for relative in _IMPLEMENTATION_FILES
    }
    return manifest, _digest(manifest)


def _write_immutable(path: Path, payload: object) -> None:
    encoded = _encoded(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite non-identical Gate 0 artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _validated_artifact_digest(payload: Mapping[str, Any]) -> str:
    artifact_digest = payload.get("artifact_digest")
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if not isinstance(artifact_digest, str) or artifact_digest != _digest(body):
        raise RuntimeError("Gate 0 artifact digest validation failed")
    return artifact_digest


def _load_lock(path: Path, config: Gate0Config) -> Gate0DevelopmentLock:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["lock"]
    scores = tuple(
        DevelopmentCandidateScore(
            controller_kind=str(item["controller_kind"]),
            candidate_index=int(item["candidate_index"]),
            action=None if item["action"] is None else tuple(float(value) for value in item["action"]),
            feedback_gain=None if item["feedback_gain"] is None else float(item["feedback_gain"]),
            mean_tke=float(item["mean_tke"]),
            rms_l2_effort=float(item["rms_l2_effort"]),
            case_mean_tke=tuple((str(case_id), float(value)) for case_id, value in item["case_mean_tke"]),
            numerical_valid=bool(item["numerical_valid"]),
        )
        for item in raw["search_scores"]
    )
    lock = Gate0DevelopmentLock(
        protocol_fingerprint=str(raw["protocol_fingerprint"]),
        fixed_action=tuple(float(value) for value in raw["fixed_action"]),
        feedback_gain=float(raw["feedback_gain"]),
        development_fixed_rms_l2=float(raw["development_fixed_rms_l2"]),
        development_feedback_rms_l2=float(raw["development_feedback_rms_l2"]),
        development_case_ids=tuple(str(value) for value in raw["development_case_ids"]),
        search_scores=scores,
        selection_rule=str(raw["selection_rule"]),
    )
    if lock.protocol_fingerprint != config.fingerprint or payload.get("lock_digest") != lock.digest:
        raise RuntimeError("development lock does not match the frozen Gate 0 protocol")
    return lock


def _mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def _report_payload(report: Gate0Report) -> dict[str, object]:
    traces = []
    for trace in report.traces:
        traces.append(
            {
                "arm": trace.arm,
                "case": asdict(trace.case),
                "uses_live_observation": trace.uses_live_observation,
                "initial_state_digest": trace.initial_state_digest,
                "uncontrolled_reset_prelude_intervals": trace.uncontrolled_reset_prelude_intervals,
                "explicit_uncontrolled_intervals": trace.explicit_uncontrolled_intervals,
                "control_start_digest": trace.control_start_digest,
                "scored_start_digest": trace.scored_start_digest,
                "source_case_id": trace.source_case_id,
                "mean_tke": trace.mean_tke,
                "rms_l2_effort": trace.rms_l2_effort,
                "integrated_l2_energy": trace.integrated_l2_energy,
                "controller_input_history": trace.controller_input_history,
                "action_history": trace.action_history,
                "records": [asdict(record) for record in trace.records],
                "numerical_gates": dict(trace.numerical_gates),
            }
        )
    body: dict[str, object] = {
        "protocol_fingerprint": report.protocol_fingerprint,
        "development_lock_digest": report.development_lock_digest,
        "primary_passed": report.primary_passed,
        "final_gate0_passed": report.passed,
        "gates": dict(report.gates),
        "paired_seed_deltas": {
            baseline: _mapping(values) for baseline, values in report.paired_seed_deltas.items()
        },
        "convergence_attestation_digest": report.convergence_attestation_digest,
        "convergence_gates": None if report.convergence_gates is None else dict(report.convergence_gates),
        "claim_boundary": report.claim_boundary,
        "rl_training_performed": report.rl_training_performed,
        "traces": traces,
    }
    return {**body, "artifact_digest": _digest(body)}


def _boolean_mapping(value: object, label: str) -> Mapping[str, bool]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or type(item) is not bool for key, item in value.items()
    ):
        raise RuntimeError(f"{label} must be a string-to-boolean mapping")
    return MappingProxyType(dict(value))


def _load_primary_report(
    path: Path,
    config: Gate0Config,
    lock: Gate0DevelopmentLock,
) -> tuple[Gate0Report, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact_digest = _validated_artifact_digest(payload)
        traces = []
        for item in payload["traces"]:
            if type(item["uses_live_observation"]) is not bool:
                raise RuntimeError("trace uses_live_observation must be boolean")
            raw_case = item["case"]
            case = Gate0Case(
                split=str(raw_case["split"]),
                phase_index=int(raw_case["phase_index"]),
                phase_turns=float(raw_case["phase_turns"]),
                seed_index=int(raw_case["seed_index"]),
                seed=int(raw_case["seed"]),
            )
            records = tuple(
                StepRecord(
                    index=int(record["index"]),
                    live_observation=tuple(float(value) for value in record["live_observation"]),
                    controller_observation=tuple(
                        float(value) for value in record["controller_observation"]
                    ),
                    action=tuple(float(value) for value in record["action"]),
                    mean_tke=float(record["mean_tke"]),
                    action_l1=float(record["action_l1"]),
                    action_l2=float(record["action_l2"]),
                    state_digest=str(record["state_digest"]),
                )
                for record in item["records"]
            )
            traces.append(
                ArmTrace(
                    arm=str(item["arm"]),
                    case=case,
                    uses_live_observation=item["uses_live_observation"],
                    initial_state_digest=str(item["initial_state_digest"]),
                    uncontrolled_reset_prelude_intervals=int(
                        item["uncontrolled_reset_prelude_intervals"]
                    ),
                    explicit_uncontrolled_intervals=int(item["explicit_uncontrolled_intervals"]),
                    control_start_digest=str(item["control_start_digest"]),
                    scored_start_digest=str(item["scored_start_digest"]),
                    source_case_id=(
                        None if item["source_case_id"] is None else str(item["source_case_id"])
                    ),
                    controller_input_history=tuple(
                        tuple(float(value) for value in observation)
                        for observation in item["controller_input_history"]
                    ),
                    action_history=tuple(
                        tuple(float(value) for value in action) for action in item["action_history"]
                    ),
                    records=records,
                    numerical_gates=_boolean_mapping(
                        item["numerical_gates"],
                        "trace numerical_gates",
                    ),
                )
            )
        paired_seed_deltas = MappingProxyType(
            {
                str(baseline): MappingProxyType(
                    {int(seed): float(value) for seed, value in values.items()}
                )
                for baseline, values in payload["paired_seed_deltas"].items()
            }
        )
        if type(payload["rl_training_performed"]) is not bool:
            raise RuntimeError("primary report rl_training_performed must be boolean")
        report = Gate0Report(
            protocol_fingerprint=str(payload["protocol_fingerprint"]),
            development_lock_digest=str(payload["development_lock_digest"]),
            traces=tuple(traces),
            gates=_boolean_mapping(payload["gates"], "primary gates"),
            paired_seed_deltas=paired_seed_deltas,
            convergence_attestation_digest=payload["convergence_attestation_digest"],
            convergence_gates=payload["convergence_gates"],
            claim_boundary=str(payload["claim_boundary"]),
            rl_training_performed=payload["rl_training_performed"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("malformed primary-report artifact") from error
    if report.protocol_fingerprint != config.fingerprint:
        raise RuntimeError("primary report does not match the frozen Gate 0 protocol")
    if report.development_lock_digest != lock.digest:
        raise RuntimeError("primary report does not match the development lock")
    if report.convergence_attestation_digest is not None or report.convergence_gates is not None:
        raise RuntimeError("primary report must not contain convergence evidence")
    if _encoded(_report_payload(report)) != _encoded(payload):
        raise RuntimeError("primary-report artifact does not round-trip through the strict codec")
    return report, artifact_digest


def _protocol_payload(
    config: Gate0Config,
    implementation_files: Mapping[str, str],
    implementation_digest: str,
) -> dict[str, object]:
    body = {
        "status": "frozen_before_execution",
        "protocol_fingerprint": config.fingerprint,
        "implementation_digest": implementation_digest,
        "implementation_files": dict(implementation_files),
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config.as_dict(),
    }
    return {**body, "artifact_digest": _digest(body)}


def _convergence_payload(convergence_run: Gate0ConvergenceRun) -> dict[str, object]:
    attestation = convergence_run.attestation
    body = {
        "status": "completed",
        "attestation": attestation.as_dict(),
        "attestation_digest": attestation.digest,
        "refinement_evidence": convergence_run.evidence_as_dict(),
    }
    return {**body, "artifact_digest": _digest(body)}


def _refinement_provenance(value: Mapping[str, Any]) -> Gate0RefinementProvenance:
    return Gate0RefinementProvenance(
        label=str(value["label"]),
        grid_size=tuple(int(item) for item in value["grid_size"]),
        dt=float(value["dt"]),
        case_ids=tuple(str(item) for item in value["case_ids"]),
    )


def _load_convergence_run(
    path: Path,
    config: Gate0Config,
    lock: Gate0DevelopmentLock,
    primary_report_digest: str,
) -> Gate0ConvergenceRun:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validated_artifact_digest(payload)
        raw_attestation = payload["attestation"]
        temporal_provenance = _refinement_provenance(raw_attestation["temporal_provenance"])
        spatial_provenance = _refinement_provenance(raw_attestation["spatial_provenance"])
        evidence_by_label = {}
        for label, provenance in (
            ("temporal", temporal_provenance),
            ("spatial", spatial_provenance),
        ):
            raw_with_digest = payload["refinement_evidence"][label]
            raw_evidence = {
                key: value for key, value in raw_with_digest.items() if key != "evidence_digest"
            }
            summaries = tuple(
                Gate0RefinementTraceSummary(
                    arm=str(item["arm"]),
                    case_id=str(item["case_id"]),
                    source_case_id=(
                        None if item["source_case_id"] is None else str(item["source_case_id"])
                    ),
                    mean_tke=float(item["mean_tke"]),
                    rms_l2_effort=float(item["rms_l2_effort"]),
                    integrated_l2_energy=float(item["integrated_l2_energy"]),
                    initial_state_digest=str(item["initial_state_digest"]),
                    control_start_digest=str(item["control_start_digest"]),
                    scored_start_digest=str(item["scored_start_digest"]),
                    uncontrolled_reset_prelude_intervals=int(
                        item["uncontrolled_reset_prelude_intervals"]
                    ),
                    explicit_uncontrolled_intervals=int(item["explicit_uncontrolled_intervals"]),
                    controller_input_digest=str(item["controller_input_digest"]),
                    action_history_digest=str(item["action_history_digest"]),
                    numerical_gates=_boolean_mapping(
                        item["numerical_gates"],
                        "refinement numerical_gates",
                    ),
                )
                for item in raw_evidence["traces"]
            )
            evidence = Gate0RefinementEvidence(
                provenance=provenance,
                target_case_ids=tuple(str(item) for item in raw_evidence["target_case_ids"]),
                traces=summaries,
                primary_order=tuple(str(item) for item in raw_evidence["primary_order"]),
                refined_order=tuple(str(item) for item in raw_evidence["refined_order"]),
                primary_decisions=_boolean_mapping(
                    raw_evidence["primary_decisions"],
                    "primary refinement decisions",
                ),
                refined_decisions=_boolean_mapping(
                    raw_evidence["refined_decisions"],
                    "refined refinement decisions",
                ),
            )
            if (
                raw_with_digest.get("evidence_digest") != evidence.digest
                or _encoded(evidence.as_dict()) != _encoded(raw_evidence)
            ):
                raise RuntimeError(f"{label} refinement evidence digest validation failed")
            evidence_by_label[label] = evidence
        attestation = Gate0ConvergenceAttestation(
            protocol_fingerprint=str(raw_attestation["protocol_fingerprint"]),
            development_lock_digest=str(raw_attestation["development_lock_digest"]),
            primary_report_digest=str(raw_attestation["primary_report_digest"]),
            convergence_seed=int(raw_attestation["convergence_seed"]),
            temporal_provenance=temporal_provenance,
            spatial_provenance=spatial_provenance,
            refinement_evidence_digests={
                str(label): str(value)
                for label, value in raw_attestation["refinement_evidence_digests"].items()
            },
            gates=_boolean_mapping(raw_attestation["gates"], "convergence gates"),
            metrics={str(name): float(value) for name, value in raw_attestation["metrics"].items()},
            schema_version=str(raw_attestation["schema_version"]),
        )
        convergence_run = Gate0ConvergenceRun(
            attestation=attestation,
            temporal_evidence=evidence_by_label["temporal"],
            spatial_evidence=evidence_by_label["spatial"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("malformed convergence artifact") from error
    if attestation.protocol_fingerprint != config.fingerprint:
        raise RuntimeError("convergence artifact does not match the frozen protocol")
    if attestation.development_lock_digest != lock.digest:
        raise RuntimeError("convergence artifact does not match the development lock")
    if attestation.primary_report_digest != primary_report_digest:
        raise RuntimeError("convergence artifact does not match the primary-report digest")
    if payload.get("attestation_digest") != attestation.digest:
        raise RuntimeError("convergence attestation digest validation failed")
    if _encoded(_convergence_payload(convergence_run)) != _encoded(payload):
        raise RuntimeError("convergence artifact does not round-trip through the strict codec")
    return convergence_run


def _development_failure_payload(error: Gate0DevelopmentSearchError) -> dict[str, object]:
    body = {**error.as_dict(), "failure_digest": error.digest}
    return {**body, "artifact_digest": _digest(body)}


def _load_development_failure(path: Path, config: Gate0Config) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validated_artifact_digest(payload)
    diagnostic = {
        key: value
        for key, value in payload.items()
        if key not in {"failure_digest", "artifact_digest"}
    }
    if payload.get("failure_digest") != _digest(diagnostic):
        raise RuntimeError("development-search failure digest validation failed")
    if payload.get("protocol_fingerprint") != config.fingerprint:
        raise RuntimeError("development-search failure does not match the frozen protocol")
    if payload.get("status") != "failed" or not isinstance(payload.get("failures"), list):
        raise RuntimeError("malformed development-search failure artifact")
    return payload


def _default_output(config: Gate0Config, implementation_digest: str) -> Path:
    run_id = f"{config.fingerprint[:12]}-{implementation_digest[:12]}"
    return Path("codex_hydrogym/evidence/gate0") / run_id


def _run_stage(
    stage: str,
    output: Path,
    config: Gate0Config,
    implementation_files: Mapping[str, str],
    implementation_digest: str,
) -> bool:
    effective_stage = "convergence" if stage == "all" else stage
    protocol_path = output / "protocol.json"
    _write_immutable(
        protocol_path,
        _protocol_payload(config, implementation_files, implementation_digest),
    )
    if effective_stage == "freeze":
        return True

    lock_path = output / "development_lock.json"
    failure_path = output / "development_search_failure.json"
    primary_path = output / "primary_report.json"
    if stage == "convergence" and not primary_path.exists():
        raise RuntimeError("convergence stage requires an existing primary-report artifact")
    if stage == "convergence" and not lock_path.exists():
        raise RuntimeError("convergence stage requires an existing development lock")
    if lock_path.exists() and failure_path.exists():
        raise RuntimeError("contradictory development lock and failure artifacts")
    if failure_path.exists():
        _load_development_failure(failure_path, config)
        return False
    if lock_path.exists():
        lock = _load_lock(lock_path, config)
    else:
        try:
            lock = lock_development_controls(config, HydroGymEpisodeFactory(config))
        except Gate0DevelopmentSearchError as error:
            _write_immutable(
                failure_path,
                _development_failure_payload(error),
            )
            return False
        body = {"protocol_fingerprint": config.fingerprint, "lock": asdict(lock), "lock_digest": lock.digest}
        _write_immutable(lock_path, {**body, "artifact_digest": _digest(body)})
    if effective_stage == "lock":
        return True

    if primary_path.exists():
        report, primary_report_digest = _load_primary_report(primary_path, config, lock)
    else:
        report = run_gate0(config, lock, HydroGymEpisodeFactory(config))
        primary_payload = _report_payload(report)
        _write_immutable(primary_path, primary_payload)
        report, primary_report_digest = _load_primary_report(primary_path, config, lock)
    if effective_stage == "primary" or not report.primary_passed:
        return report.primary_passed

    convergence_path = output / "convergence_attestation.json"
    final_path = output / "final_report.json"
    if final_path.exists() and not convergence_path.exists():
        raise RuntimeError("final report exists without its convergence artifact")
    if convergence_path.exists():
        convergence_run = _load_convergence_run(
            convergence_path,
            config,
            lock,
            primary_report_digest,
        )
    else:
        convergence_run = run_gate0_convergence(
            config,
            lock,
            report,
            primary_report_digest,
            lambda grid_size, dt: HydroGymEpisodeFactory(
                config,
                execution_grid_size=grid_size,
                execution_dt=dt,
            ),
        )
        _write_immutable(convergence_path, _convergence_payload(convergence_run))
    attestation = convergence_run.attestation
    completed = report.with_convergence(
        attestation,
        primary_report_digest=primary_report_digest,
    )
    final_payload = _report_payload(completed)
    _write_immutable(final_path, final_payload)
    stored_final = json.loads(final_path.read_text(encoding="utf-8"))
    _validated_artifact_digest(stored_final)
    if _encoded(stored_final) != _encoded(final_payload):
        raise RuntimeError("final report does not match primary and convergence evidence")
    return completed.passed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("freeze", "lock", "primary", "convergence", "all"),
        default="freeze",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    config = Gate0Config()
    implementation_files, implementation_digest = _implementation_manifest()
    output = args.output_dir or _default_output(config, implementation_digest)
    succeeded = _run_stage(
        args.stage,
        output,
        config,
        implementation_files,
        implementation_digest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "protocol_fingerprint": config.fingerprint,
                "implementation_digest": implementation_digest,
                "requested_stage": args.stage,
                "stage_succeeded": succeeded,
                "final_claim_requires_convergence": True,
            },
            sort_keys=True,
        )
    )
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
