"""Development-only ensemble diagnostic for the failed Re=100 Gate 0 v2.

This is an exploratory numerical study, not Gate 0 v3.  It uses fresh seeds,
two consecutive scoring windows, and paired zero/signed-feedback trajectories
to determine whether seed-clustered averaging makes the old convergence
margins plausible.  A positive result may support designing a new frozen gate;
it never authorizes PPO, MemAlign, GEPA, deployment, or a fluid-improvement
claim.
"""

from __future__ import annotations

import argparse
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
    ZERO_ACTION,
    FrozenSignedController,
    Gate0Config,
    HydroGymEpisodeFactory,
    _run_episode,
)


STUDY_SCHEMA_VERSION = "codex_hydrogym.gate0.ensemble_diagnostic.v1"
STUDY_ID = "re100_fresh_seed_windowed_convergence_diagnostic_v1"
CLAIM_BOUNDARY = (
    "Exploratory development-only numerical evidence. No held-out gate, RL, reward proposal, "
    "coding-agent comparison, MemAlign, GEPA, or fluid-improvement claim is performed. Even a "
    "positive screen only supports designing a separately preregistered full gate."
)
SOURCE_V2_PROTOCOL_FINGERPRINT = (
    "2729e365a5712b824d4d1a2257ade19519aa454509a221790ddcabf2021b32a9"
)
SOURCE_V2_FINAL_REPORT_DIGEST = (
    "e8a19fa835a991f233ed7087c31a7c947c6a61a9a3c04ceb935db2870f8dd463"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_SOURCE = "codex_hydrogym/gate0/ensemble_diagnostic.py"


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _encoded(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _finite(value: float, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


@dataclass(frozen=True)
class ExecutionCondition:
    label: str
    grid_size: tuple[int, int]
    dt: float
    arm_relative_difference_limit: float | None
    effect_difference_limit: float | None

    def __post_init__(self) -> None:
        if self.label not in {"base", "temporal", "spatial"}:
            raise ValueError("unsupported execution-condition label")
        if len(self.grid_size) != 2 or self.grid_size[0] != self.grid_size[1]:
            raise ValueError("execution conditions require square grids")
        if any(type(value) is not int or value <= 0 for value in self.grid_size):
            raise ValueError("execution-condition grid values must be positive integers")
        if _finite(self.dt, "execution-condition dt") <= 0.0:
            raise ValueError("execution-condition dt must be positive")
        limits = (self.arm_relative_difference_limit, self.effect_difference_limit)
        if self.label == "base" and limits != (None, None):
            raise ValueError("base condition must not define refinement limits")
        if self.label != "base" and any(
            value is None or not math.isfinite(value) or value <= 0.0 for value in limits
        ):
            raise ValueError("refinement conditions require finite positive limits")


@dataclass(frozen=True)
class EnsembleDiagnosticSpec:
    study_id: str = STUDY_ID
    schema_version: str = STUDY_SCHEMA_VERSION
    reynolds_number: float = 100.0
    precision: str = "float64"
    phase_turns: tuple[float, ...] = (0.0625, 0.5625)
    seeds: tuple[int, ...] = (401, 503, 607, 709)
    reserved_phase_turns: tuple[float, ...] = (0.1875, 0.6875)
    reserved_seeds: tuple[int, ...] = (907, 1009)
    uncontrolled_burn_in_intervals: int = 100
    controller_warmup_intervals: int = 50
    scoring_windows: int = 2
    intervals_per_window: int = 100
    feedback_gain: float = 2.0
    radial_action_bound: float = 0.5
    minimum_relative_feedback_effect: float = 0.05
    seed_cluster_t_critical_95: float = 3.182446305284263
    seed_cluster_t_critical_90: float = 2.353363434801823
    source_v2_protocol_fingerprint: str = SOURCE_V2_PROTOCOL_FINGERPRINT
    source_v2_final_report_digest: str = SOURCE_V2_FINAL_REPORT_DIGEST
    conditions: tuple[ExecutionCondition, ...] = (
        ExecutionCondition("base", (64, 64), 0.002, None, None),
        ExecutionCondition("temporal", (64, 64), 0.001, 0.02, 0.02),
        ExecutionCondition("spatial", (96, 96), 0.002, 0.05, 0.03),
    )
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.study_id != STUDY_ID or self.schema_version != STUDY_SCHEMA_VERSION:
            raise ValueError("study identity/schema mismatch")
        if self.precision != "float64" or self.reynolds_number != 100.0:
            raise ValueError("this diagnostic is frozen to Re=100 float64")
        if len(self.seeds) != 4 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("the diagnostic requires four distinct seed clusters")
        if set(self.seeds) & set(self.reserved_seeds):
            raise ValueError("development and reserved seeds must be disjoint")
        historical_seeds = {7, 101, 211, 307}
        if historical_seeds & (set(self.seeds) | set(self.reserved_seeds)):
            raise ValueError("diagnostic seeds must be fresh relative to Gate 0 v1/v2")
        if len(self.phase_turns) != 2 or not math.isclose(
            (self.phase_turns[0] + 0.5) % 1.0,
            self.phase_turns[1],
            abs_tol=1.0e-12,
        ):
            raise ValueError("development phases must be one opposite-phase pair")
        if len(self.reserved_phase_turns) != 2 or not math.isclose(
            (self.reserved_phase_turns[0] + 0.5) % 1.0,
            self.reserved_phase_turns[1],
            abs_tol=1.0e-12,
        ):
            raise ValueError("reserved phases must be one opposite-phase pair")
        if set(self.phase_turns) & set(self.reserved_phase_turns):
            raise ValueError("development and reserved phases must be disjoint")
        if self.scoring_windows != 2 or self.intervals_per_window != 100:
            raise ValueError("the diagnostic requires two 100-interval scoring windows")
        if self.uncontrolled_burn_in_intervals <= 0 or self.controller_warmup_intervals <= 0:
            raise ValueError("development horizons must be positive")
        if self.feedback_gain <= 0.0 or self.radial_action_bound != 0.5:
            raise ValueError("feedback controller contract mismatch")
        if not 0.0 < self.minimum_relative_feedback_effect < 1.0:
            raise ValueError("minimum feedback effect must lie in (0, 1)")
        if tuple(condition.label for condition in self.conditions) != (
            "base",
            "temporal",
            "spatial",
        ):
            raise ValueError("execution conditions must be base, temporal, spatial")
        if self.seed_cluster_t_critical_95 != 3.182446305284263:
            raise ValueError("95% t critical must match four seed clusters (df=3)")
        if self.seed_cluster_t_critical_90 != 2.353363434801823:
            raise ValueError("90% t critical must match four seed clusters (df=3)")
        for digest, label in (
            (self.source_v2_protocol_fingerprint, "source protocol fingerprint"),
            (self.source_v2_final_report_digest, "source final-report digest"),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{label} must be lowercase SHA-256")
        if self.claim_boundary != CLAIM_BOUNDARY:
            raise ValueError("claim boundary must remain frozen")

    @property
    def scored_intervals(self) -> int:
        return self.scoring_windows * self.intervals_per_window

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _digest(self.as_dict())


def study_gate_config(spec: EnsembleDiagnosticSpec) -> Gate0Config:
    """Materialize the existing physics contract for development cases only."""
    return replace(
        Gate0Config(),
        protocol_id=spec.study_id,
        development_phase_turns=spec.phase_turns,
        development_seeds=spec.seeds,
        heldout_phase_turns=spec.reserved_phase_turns,
        heldout_seeds=spec.reserved_seeds,
        convergence_seed=spec.reserved_seeds[0],
        reynolds_number=spec.reynolds_number,
        grid_size=spec.conditions[0].grid_size,
        precision=spec.precision,
        dt=spec.conditions[0].dt,
        uncontrolled_burn_in_intervals=spec.uncontrolled_burn_in_intervals,
        controller_warmup_intervals=spec.controller_warmup_intervals,
        scored_intervals=spec.scored_intervals,
        radial_action_bound=spec.radial_action_bound,
        spatial_refinement_grid_size=spec.conditions[2].grid_size,
        temporal_refinement_dt=spec.conditions[1].dt,
    )


def _study_implementation_manifest() -> tuple[dict[str, str], str]:
    implementation_files, _ = _implementation_manifest()
    implementation_files[_CONFIG_SOURCE] = hashlib.sha256(
        (_REPOSITORY_ROOT / _CONFIG_SOURCE).read_bytes()
    ).hexdigest()
    return implementation_files, _digest(implementation_files)


def _write_immutable(path: Path, payload: object) -> None:
    encoded = _encoded(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite non-identical diagnostic artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _artifact(body: Mapping[str, object]) -> dict[str, object]:
    return {**body, "artifact_digest": _digest(body)}


def _validate_artifact(payload: Mapping[str, Any]) -> str:
    artifact_digest = payload.get("artifact_digest")
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if not isinstance(artifact_digest, str) or artifact_digest != _digest(body):
        raise RuntimeError("diagnostic artifact digest validation failed")
    return artifact_digest


def _protocol_payload(
    spec: EnsembleDiagnosticSpec,
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
    return {
        "window_index": index,
        "interval_count": len(records),
        "mean_tke": fmean(interval_tke),
        "rms_l2_effort": math.sqrt(fmean(value**2 for value in interval_effort)),
        "interval_mean_tke": interval_tke,
        "interval_action_l2": interval_effort,
        "start_state_digest": start_state_digest,
        "end_state_digest": records[-1].state_digest,
    }


def _trace_payload(trace: Any, condition: ExecutionCondition, spec: EnsembleDiagnosticSpec) -> dict[str, object]:
    if len(trace.records) != spec.scored_intervals:
        raise RuntimeError("diagnostic trace has an unexpected scored horizon")
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
    return {
        "condition": asdict(condition),
        "arm": trace.arm,
        "case": asdict(trace.case),
        "initial_state_digest": trace.initial_state_digest,
        "control_start_digest": trace.control_start_digest,
        "scored_start_digest": trace.scored_start_digest,
        "mean_tke": trace.mean_tke,
        "rms_l2_effort": trace.rms_l2_effort,
        "numerical_gates": dict(trace.numerical_gates),
        "controller_input_history_digest": _digest(trace.controller_input_history),
        "action_history_digest": _digest(trace.action_history),
        "state_history_digest": _digest(tuple(record.state_digest for record in trace.records)),
        "windows": windows,
    }


def _run_condition(
    spec: EnsembleDiagnosticSpec,
    config: Gate0Config,
    condition: ExecutionCondition,
    implementation_digest: str,
) -> dict[str, object]:
    factory = HydroGymEpisodeFactory(
        config,
        execution_grid_size=condition.grid_size,
        execution_dt=condition.dt,
    )
    feedback = FrozenSignedController(spec.feedback_gain, spec.radial_action_bound)
    traces = []
    for case in config.cases("development"):
        traces.append(
            _trace_payload(
                _run_episode(config, factory, case, "zero", lambda _observation: ZERO_ACTION),
                condition,
                spec,
            )
        )
        traces.append(
            _trace_payload(
                _run_episode(config, factory, case, "signed_feedback", feedback),
                condition,
                spec,
            )
        )
    body: dict[str, object] = {
        "status": "completed",
        "study_fingerprint": spec.fingerprint,
        "implementation_digest": implementation_digest,
        "condition": asdict(condition),
        "case_ids": tuple(case.case_id for case in config.cases("development")),
        "traces": traces,
    }
    return _artifact(body)


def _load_condition(
    path: Path,
    spec: EnsembleDiagnosticSpec,
    condition: ExecutionCondition,
    implementation_digest: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_artifact(payload)
    if payload.get("study_fingerprint") != spec.fingerprint:
        raise RuntimeError("condition artifact does not match the study")
    if payload.get("implementation_digest") != implementation_digest:
        raise RuntimeError("condition artifact does not match the frozen implementation")
    if payload.get("condition") != json.loads(_canonical(asdict(condition))):
        raise RuntimeError("condition artifact execution identity mismatch")
    expected_cases = {
        f"development_p{phase_index:02d}_s{seed_index:02d}_{seed}"
        for seed_index, seed in enumerate(spec.seeds)
        for phase_index, _phase in enumerate(spec.phase_turns)
    }
    if set(payload.get("case_ids", ())) != expected_cases:
        raise RuntimeError("condition artifact case set mismatch")
    if len(payload.get("traces", ())) != 2 * len(expected_cases):
        raise RuntimeError("condition artifact trace count mismatch")
    return payload


def _mean_ci(values: Sequence[float], t_critical: float) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("confidence interval requires multiple independent seed clusters")
    mean = fmean(values)
    standard_error = stdev(values) / math.sqrt(len(values))
    half_width = t_critical * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _blocks(payload: Mapping[str, Any]) -> dict[tuple[int, float, int], dict[str, float]]:
    blocks: dict[tuple[int, float, int], dict[str, float]] = {}
    for trace in payload["traces"]:
        case = trace["case"]
        for window in trace["windows"]:
            key = (int(case["seed"]), float(case["phase_turns"]), int(window["window_index"]))
            blocks.setdefault(key, {})[str(trace["arm"])] = float(window["mean_tke"])
    if any(set(arms) != {"zero", "signed_feedback"} for arms in blocks.values()):
        raise RuntimeError("diagnostic block does not contain the exact paired arms")
    return blocks


def analyze_conditions(
    spec: EnsembleDiagnosticSpec,
    condition_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    if set(condition_payloads) != {condition.label for condition in spec.conditions}:
        raise ValueError("analysis requires the exact frozen condition set")
    blocks_by_condition = {
        label: _blocks(payload) for label, payload in condition_payloads.items()
    }
    expected_keys = set(blocks_by_condition["base"])
    if any(set(blocks) != expected_keys for blocks in blocks_by_condition.values()):
        raise RuntimeError("execution conditions do not contain identical paired blocks")

    condition_metrics: dict[str, dict[str, object]] = {}
    seed_effects_by_condition: dict[str, dict[int, float]] = {}
    for condition in spec.conditions:
        blocks = blocks_by_condition[condition.label]
        zero_values = [arms["zero"] for arms in blocks.values()]
        feedback_values = [arms["signed_feedback"] for arms in blocks.values()]
        relative_effects = {
            key: (arms["zero"] - arms["signed_feedback"]) / arms["zero"]
            for key, arms in blocks.items()
        }
        seed_effects = {
            seed: fmean(
                effect for (block_seed, _phase, _window), effect in relative_effects.items() if block_seed == seed
            )
            for seed in spec.seeds
        }
        seed_effects_by_condition[condition.label] = seed_effects
        zero_mean = fmean(zero_values)
        feedback_mean = fmean(feedback_values)
        aggregate_relative_effect = (zero_mean - feedback_mean) / zero_mean
        by_window = {
            str(index): fmean(
                effect
                for (_seed, _phase, window), effect in relative_effects.items()
                if window == index
            )
            for index in range(spec.scoring_windows)
        }
        condition_metrics[condition.label] = {
            "zero_mean_tke": zero_mean,
            "feedback_mean_tke": feedback_mean,
            "aggregate_relative_effect": aggregate_relative_effect,
            "seed_cluster_relative_effects": seed_effects,
            "seed_cluster_effect_ci_95": _mean_ci(
                tuple(seed_effects.values()),
                spec.seed_cluster_t_critical_95,
            ),
            "window_relative_effect_means": by_window,
            "feedback_window_win_fraction": sum(effect > 0.0 for effect in relative_effects.values())
            / len(relative_effects),
            "minimum_window_relative_effect": min(relative_effects.values()),
        }

    base_metrics = condition_metrics["base"]
    refinement_metrics: dict[str, dict[str, object]] = {}
    for condition in spec.conditions[1:]:
        metrics = condition_metrics[condition.label]
        paired_seed_differences = tuple(
            seed_effects_by_condition[condition.label][seed]
            - seed_effects_by_condition["base"][seed]
            for seed in spec.seeds
        )
        effect_ci_90 = _mean_ci(paired_seed_differences, spec.seed_cluster_t_critical_90)
        effect_limit = float(condition.effect_difference_limit)
        arm_limit = float(condition.arm_relative_difference_limit)
        zero_relative_difference = abs(
            float(metrics["zero_mean_tke"]) - float(base_metrics["zero_mean_tke"])
        ) / float(base_metrics["zero_mean_tke"])
        feedback_relative_difference = abs(
            float(metrics["feedback_mean_tke"])
            - float(base_metrics["feedback_mean_tke"])
        ) / float(base_metrics["feedback_mean_tke"])
        effect_difference = abs(
            float(metrics["aggregate_relative_effect"])
            - float(base_metrics["aggregate_relative_effect"])
        )
        refinement_metrics[condition.label] = {
            "zero_arm_relative_difference": zero_relative_difference,
            "feedback_arm_relative_difference": feedback_relative_difference,
            "maximum_arm_relative_difference": max(
                zero_relative_difference,
                feedback_relative_difference,
            ),
            "arm_relative_difference_limit": arm_limit,
            "aggregate_effect_difference": effect_difference,
            "effect_difference_limit": effect_limit,
            "paired_seed_effect_difference_ci_90": effect_ci_90,
            "arm_point_convergence": max(
                zero_relative_difference,
                feedback_relative_difference,
            )
            <= arm_limit,
            "effect_point_convergence": effect_difference <= effect_limit,
            "effect_equivalence_ci_supported": (
                effect_ci_90["lower"] >= -effect_limit and effect_ci_90["upper"] <= effect_limit
            ),
        }

    numerical_validity = all(
        all(value is True for value in trace["numerical_gates"].values())
        for payload in condition_payloads.values()
        for trace in payload["traces"]
    )
    feedback_wins_every_window = all(
        metrics["feedback_window_win_fraction"] == 1.0
        for metrics in condition_metrics.values()
    )
    positive_effect_ci = all(
        metrics["seed_cluster_effect_ci_95"]["lower"]
        >= spec.minimum_relative_feedback_effect
        for metrics in condition_metrics.values()
    )
    window_effects_material = all(
        all(
            value >= spec.minimum_relative_feedback_effect
            for value in metrics["window_relative_effect_means"].values()
        )
        for metrics in condition_metrics.values()
    )
    screening = {
        "numerical_validity": numerical_validity,
        "feedback_wins_every_window": feedback_wins_every_window,
        "positive_seed_cluster_effect_ci": positive_effect_ci,
        "both_window_means_material": window_effects_material,
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
    return {
        "condition_metrics": condition_metrics,
        "refinement_metrics": refinement_metrics,
        "screening": screening,
        "supports_designing_full_gate": all(value is True for value in screening.values()),
        "claim_boundary": spec.claim_boundary,
    }


def _default_output(spec: EnsembleDiagnosticSpec, implementation_digest: str) -> Path:
    return Path("codex_hydrogym/evidence/ensemble_diagnostic") / (
        f"{spec.fingerprint[:12]}-{implementation_digest[:12]}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "run"), default="freeze")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    spec = EnsembleDiagnosticSpec()
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
    _write_immutable(protocol_path, expected_protocol)
    if _encoded(json.loads(protocol_path.read_text(encoding="utf-8"))) != _encoded(
        expected_protocol
    ):
        raise RuntimeError("stored diagnostic protocol does not match the frozen protocol")

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
                        spec,
                        config,
                        condition,
                        implementation_digest,
                    ),
                )
            payload = _load_condition(
                condition_path,
                spec,
                condition,
                implementation_digest,
            )
            payloads[condition.label] = payload
            condition_digests[condition.label] = str(payload["artifact_digest"])
        analysis = analyze_conditions(spec, payloads)
        result_body: dict[str, object] = {
            "status": "completed",
            "study_fingerprint": spec.fingerprint,
            "implementation_digest": implementation_digest,
            "condition_artifact_digests": condition_digests,
            "analysis": analysis,
        }
        result_path = output / "result.json"
        expected_result = _artifact(result_body)
        _write_immutable(result_path, expected_result)
        stored_result = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_artifact(stored_result)
        if _encoded(stored_result) != _encoded(expected_result):
            raise RuntimeError("stored diagnostic result does not round-trip")
        supports_designing_full_gate = bool(analysis["supports_designing_full_gate"])

    print(
        json.dumps(
            {
                "output_dir": str(output),
                "study_fingerprint": spec.fingerprint,
                "implementation_digest": implementation_digest,
                "stage": args.stage,
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
