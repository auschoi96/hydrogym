"""Preregistered, deterministic CPU Gate 0 for observation-conditioned control.

This is a controllability/evaluator gate, not reinforcement learning.  It keeps
TKE and actuation effort separate and never ranks arms with the current scalar
PPO reward.  The shuffled arm reuses whole controller-input trajectories from a
one-to-one phase-and-seed-mismatched episode permutation.  Thus the global input
and action marginals, temporal structure, and effort are exact while the
same-episode state/action pairing is broken.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from codex_hydrogym import PROJECT_LABEL
from hydrogym.jax.kolmogorov_contract import SIGNED_FORCED_MODE_OBSERVATION_VERSION


GATE0_SCHEMA_VERSION = "codex_hydrogym.gate0.v1"
GATE0_CONVERGENCE_SCHEMA_VERSION = "codex_hydrogym.gate0.convergence.v1"
FORCED_ACTION_DIMENSION = 4
ZERO_ACTION = (0.0, 0.0, 0.0, 0.0)
REQUIRED_NUMERICAL_GATES = frozenset(
    {
        "finite_state_and_metrics",
        "nonnegative_tke",
        "reward_tke_identity",
        "zero_mean_vorticity",
        "incompressible_velocity",
        "spectral_tail_controlled",
        "cfl_controlled",
    }
)
REQUIRED_PRIMARY_GATES = frozenset(
    {
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
)
REQUIRED_CONVERGENCE_GATES = frozenset(
    {
        "spatial_arm_tke_convergence",
        "spatial_effect_convergence",
        "spatial_numerical_validity",
        "spatial_ordering_preservation",
        "spatial_pass_decision_preservation",
        "temporal_arm_tke_convergence",
        "temporal_effect_convergence",
        "temporal_numerical_validity",
        "temporal_ordering_preservation",
        "temporal_pass_decision_preservation",
    }
)
CLAIM_BOUNDARY = (
    "Gate 0 only tests adaptive forced-mode cancellation on static-phase episodes. "
    "It performs no RL, proves no learned improvement, and does not show that PPO can infer a per-reset phase; "
    "that requires phase randomization inside vectorized resets before training."
)

Action = tuple[float, float, float, float]
Observation = tuple[float, float]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _is_sha256_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_identifier(value: object, name: str) -> str:
    if not _is_sha256_identifier(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _integer_ratio(duration: float, interval: float, label: str) -> int:
    ratio = duration / interval
    rounded = int(round(ratio))
    if not math.isclose(ratio, rounded, rel_tol=1.0e-10, abs_tol=1.0e-9):
        raise ValueError(f"{label} must be an integer ratio")
    return rounded


def _action(value: Sequence[float], *, radial_bound: float) -> Action:
    if len(value) != FORCED_ACTION_DIMENSION:
        raise ValueError("Gate 0 actions must have four coefficients")
    result = tuple(_finite(item, "action coefficient") for item in value)
    if result[2:] != (0.0, 0.0):
        raise ValueError("Gate 0 is restricted to the two-dimensional forced-mode action subspace")
    if math.hypot(result[0], result[1]) > radial_bound + 1.0e-12:
        raise ValueError("forced-mode action exceeds the preregistered radial bound")
    return result  # type: ignore[return-value]


def _observation(value: Sequence[float]) -> Observation:
    if len(value) != 2:
        raise ValueError("Gate 0 requires two signed forced-mode observations")
    result = tuple(_finite(item, "observation coefficient") for item in value)
    return result  # type: ignore[return-value]


def _radial_clip(first: float, second: float, bound: float) -> Action:
    radius = math.hypot(first, second)
    if radius > bound:
        scale = bound / radius
        first, second = first * scale, second * scale
    return (float(first), float(second), 0.0, 0.0)


@dataclass(frozen=True, order=True)
class Gate0Case:
    split: str
    phase_index: int
    phase_turns: float
    seed_index: int
    seed: int

    @property
    def phase_radians(self) -> float:
        return 2.0 * math.pi * self.phase_turns

    @property
    def case_id(self) -> str:
        return f"{self.split}_p{self.phase_index:02d}_s{self.seed_index:02d}_{self.seed}"


def _default_constant_candidates() -> tuple[Action, ...]:
    candidates: list[Action] = [ZERO_ACTION]
    for radius in (0.25, 0.5):
        for index in range(8):
            angle = 2.0 * math.pi * index / 8.0
            candidates.append((radius * math.cos(angle), radius * math.sin(angle), 0.0, 0.0))
    return tuple(candidates)


@dataclass(frozen=True)
class Gate0Config:
    """Frozen before held-out execution; defaults deliberately exclude the observed 16² pilot."""

    protocol_id: str = "offset_phase_fp64_gate0_v1"
    development_phase_turns: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    development_seeds: tuple[int, ...] = (7,)
    heldout_phase_turns: tuple[float, ...] = (0.125, 0.375, 0.625, 0.875)
    heldout_seeds: tuple[int, ...] = (101, 211, 307)
    grid_size: tuple[int, int] = (48, 48)
    precision: str = "float64"
    reynolds_number: float = 200.0
    forcing_wavenumber: int = 4
    initial_perturbation_amplitude: float = 1.0e-3
    dt: float = 0.002
    action_time: float = 1.0
    save_time: float = 0.2
    uncontrolled_burn_in_intervals: int = 100
    controller_warmup_intervals: int = 50
    scored_intervals: int = 100
    radial_action_bound: float = 0.5
    constant_candidates: tuple[Action, ...] = _default_constant_candidates()
    feedback_gain_candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    minimum_relative_tke_reduction: float = 0.05
    minimum_absolute_tke_reduction: float = 0.005
    minimum_seed_win_fraction: float = 2.0 / 3.0
    effort_match_atol: float = 1.0e-12
    maximum_cfl: float = 0.5
    maximum_zero_mode_ratio: float = 1.0e-10
    maximum_divergence_ratio: float = 1.0e-10
    maximum_spectral_tail_fraction: float = 0.05
    maximum_reward_identity_relative_error: float = 1.0e-10
    convergence_seed: int = 101
    temporal_refinement_dt: float = 0.001
    spatial_refinement_grid_size: tuple[int, int] = (64, 64)
    maximum_temporal_arm_tke_relative_difference: float = 0.02
    maximum_spatial_arm_tke_relative_difference: float = 0.05
    maximum_temporal_effect_difference: float = 0.02
    maximum_spatial_effect_difference: float = 0.03
    observation_contract_version: str = SIGNED_FORCED_MODE_OBSERVATION_VERSION
    schema_version: str = GATE0_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GATE0_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {GATE0_SCHEMA_VERSION}")
        if self.observation_contract_version != SIGNED_FORCED_MODE_OBSERVATION_VERSION:
            raise ValueError("Gate 0 requires the signed forced-mode get_obs contract")
        if set(self.development_phase_turns) & set(self.heldout_phase_turns):
            raise ValueError("development and held-out phases must be disjoint")
        if set(self.development_seeds) & set(self.heldout_seeds):
            raise ValueError("development and held-out seeds must be disjoint")
        if len(self.heldout_phase_turns) < 2 or len(self.heldout_seeds) < 2:
            raise ValueError("held-out inference requires multiple independent phases and seeds")
        if any(not 0.0 <= phase < 1.0 for phase in (*self.development_phase_turns, *self.heldout_phase_turns)):
            raise ValueError("phase turns must lie in [0, 1)")
        if len(set(self.development_phase_turns)) != len(self.development_phase_turns) or len(
            set(self.heldout_phase_turns)
        ) != len(self.heldout_phase_turns):
            raise ValueError("phase sets must not contain duplicates")
        if any(
            not any(math.isclose((phase + 0.5) % 1.0, other, abs_tol=1.0e-12) for other in self.heldout_phase_turns)
            for phase in self.heldout_phase_turns
        ):
            raise ValueError("every held-out phase must have a preregistered opposite-phase partner")
        if len(set((*self.development_seeds, *self.heldout_seeds))) != len(
            (*self.development_seeds, *self.heldout_seeds)
        ):
            raise ValueError("seed sets must not contain duplicates")
        if any(
            type(seed) is not int or seed < 0
            for seed in (*self.development_seeds, *self.heldout_seeds)
        ):
            raise ValueError("seeds must be nonnegative integers")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        if any(value <= 0 for value in (*self.grid_size, self.dt, self.action_time, self.save_time)):
            raise ValueError("grid and integration values must be positive")
        if self.grid_size[0] != self.grid_size[1]:
            raise ValueError("Gate 0 requires a square grid for the current TKE normalization")
        if self.forcing_wavenumber <= 0 or self.forcing_wavenumber > self.grid_size[1] // 3:
            raise ValueError("forcing_wavenumber must lie inside the retained two-thirds band")
        _integer_ratio(self.save_time, self.dt, "save_time/dt")
        _integer_ratio(self.action_time, self.save_time, "action_time/save_time")
        for name in ("uncontrolled_burn_in_intervals", "controller_warmup_intervals", "scored_intervals"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.radial_action_bound <= 0.5:
            raise ValueError("radial_action_bound must be in (0, 0.5]")
        if not self.constant_candidates or not self.feedback_gain_candidates:
            raise ValueError("development search grids must not be empty")
        for candidate in self.constant_candidates:
            _action(candidate, radial_bound=self.radial_action_bound)
        if any(not math.isfinite(gain) or gain <= 0.0 for gain in self.feedback_gain_candidates):
            raise ValueError("feedback gain candidates must be finite and positive")
        if self.convergence_seed not in self.heldout_seeds:
            raise ValueError("convergence_seed must be one of the preregistered held-out seeds")
        if self.spatial_refinement_grid_size[0] != self.spatial_refinement_grid_size[1]:
            raise ValueError("spatial refinement requires a square grid")
        if self.spatial_refinement_grid_size[0] <= self.grid_size[0]:
            raise ValueError("spatial refinement grid must be finer than the primary grid")
        if not 0.0 < self.temporal_refinement_dt < self.dt:
            raise ValueError("temporal_refinement_dt must be smaller than primary dt")
        _integer_ratio(self.save_time, self.temporal_refinement_dt, "save_time/temporal_refinement_dt")
        for name in (
            "maximum_reward_identity_relative_error",
            "maximum_temporal_arm_tke_relative_difference",
            "maximum_spatial_arm_tke_relative_difference",
            "maximum_temporal_effect_difference",
            "maximum_spatial_effect_difference",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def cases(self, split: str) -> tuple[Gate0Case, ...]:
        phases = self.development_phase_turns if split == "development" else self.heldout_phase_turns
        seeds = self.development_seeds if split == "development" else self.heldout_seeds
        return tuple(
            Gate0Case(split, phase_index, phase, seed_index, seed)
            for seed_index, seed in enumerate(seeds)
            for phase_index, phase in enumerate(phases)
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _digest(self.as_dict())


@dataclass(frozen=True)
class EpisodeStep:
    observation: Observation
    mean_tke: float
    state_digest: str


class Gate0Episode(Protocol):
    observation: Observation
    state_digest: str
    uncontrolled_reset_prelude_intervals: int

    def advance(self, action: Action) -> EpisodeStep: ...

    def numerical_gates(self) -> Mapping[str, bool]: ...


EpisodeFactory = Callable[[Gate0Case], Gate0Episode]
RefinementFactoryBuilder = Callable[[tuple[int, int], float], EpisodeFactory]


@dataclass(frozen=True)
class FrozenSignedController:
    """Pure memoryless controller; the call receives observations and nothing else."""

    gain: float
    radial_bound: float

    def __call__(self, observation: Observation) -> Action:
        if len(observation) != 2 or any(not math.isfinite(float(value)) for value in observation):
            raise ValueError("signed controller requires two finite modal coefficients")
        return _radial_clip(-self.gain * observation[0], -self.gain * observation[1], self.radial_bound)


@dataclass(frozen=True)
class StepRecord:
    index: int
    live_observation: Observation
    controller_observation: Observation
    action: Action
    mean_tke: float
    action_l1: float
    action_l2: float
    state_digest: str


@dataclass(frozen=True)
class ArmTrace:
    arm: str
    case: Gate0Case
    uses_live_observation: bool
    initial_state_digest: str
    uncontrolled_reset_prelude_intervals: int
    explicit_uncontrolled_intervals: int
    control_start_digest: str
    scored_start_digest: str
    source_case_id: str | None
    controller_input_history: tuple[Observation, ...]
    action_history: tuple[Action, ...]
    records: tuple[StepRecord, ...]
    numerical_gates: Mapping[str, bool]

    @property
    def mean_tke(self) -> float:
        return sum(record.mean_tke for record in self.records) / len(self.records)

    @property
    def rms_l2_effort(self) -> float:
        return math.sqrt(sum(record.action_l2**2 for record in self.records) / len(self.records))

    @property
    def integrated_l2_energy(self) -> float:
        return sum(record.action_l2**2 for record in self.records)


def oracle_action(case: Gate0Case, radial_bound: float) -> Action:
    """Privileged cancellation using both quadrature coefficients and no observation."""
    return _radial_clip(
        -radial_bound * math.cos(case.phase_radians),
        -radial_bound * math.sin(case.phase_radians),
        radial_bound,
    )


def _run_episode(
    config: Gate0Config,
    factory: EpisodeFactory,
    case: Gate0Case,
    arm: str,
    controller: Callable[[Observation], Action],
    *,
    controller_inputs: Sequence[Observation] | None = None,
    source_case_id: str | None = None,
) -> ArmTrace:
    developed_episode = getattr(factory, "developed_episode", None)
    episode = (
        developed_episode(case, config.uncontrolled_burn_in_intervals)
        if callable(developed_episode)
        else factory(case)
    )
    initial_digest = episode.state_digest
    prelude = episode.uncontrolled_reset_prelude_intervals
    valid_prelude = (
        not isinstance(prelude, bool)
        and isinstance(prelude, int)
        and 0 <= prelude <= config.uncontrolled_burn_in_intervals
    )
    if not valid_prelude:
        raise ValueError("episode reset prelude must fit inside the preregistered uncontrolled horizon")
    explicit_uncontrolled = config.uncontrolled_burn_in_intervals - prelude
    for _ in range(explicit_uncontrolled):
        episode.advance(ZERO_ACTION)
    control_start_digest = episode.state_digest
    controller_input_history: list[Observation] = []
    action_history: list[Action] = []
    for index in range(config.controller_warmup_intervals):
        observation = _observation(
            episode.observation if controller_inputs is None else controller_inputs[index]
        )
        action = _action(controller(observation), radial_bound=config.radial_action_bound)
        controller_input_history.append(observation)
        action_history.append(action)
        episode.advance(action)
    scored_start_digest = episode.state_digest
    records = []
    offset = config.controller_warmup_intervals
    for index in range(config.scored_intervals):
        live_observation = _observation(episode.observation)
        controller_observation = (
            live_observation
            if controller_inputs is None
            else _observation(controller_inputs[offset + index])
        )
        action = _action(controller(controller_observation), radial_bound=config.radial_action_bound)
        controller_input_history.append(controller_observation)
        action_history.append(action)
        step = episode.advance(action)
        records.append(
            StepRecord(
                index=index,
                live_observation=live_observation,
                controller_observation=controller_observation,
                action=action,
                mean_tke=_finite(step.mean_tke, "mean_tke"),
                action_l1=sum(abs(value) for value in action),
                action_l2=math.hypot(action[0], action[1]),
                state_digest=step.state_digest,
            )
        )
    gates = dict(episode.numerical_gates())
    missing = REQUIRED_NUMERICAL_GATES - gates.keys()
    if missing:
        raise ValueError(f"episode omitted required numerical gates: {sorted(missing)}")
    return ArmTrace(
        arm,
        case,
        controller_inputs is None and arm == "signed_feedback",
        initial_digest,
        prelude,
        explicit_uncontrolled,
        control_start_digest,
        scored_start_digest,
        source_case_id,
        tuple(controller_input_history),
        tuple(action_history),
        tuple(records),
        MappingProxyType(gates),
    )


@dataclass(frozen=True)
class DevelopmentCandidateScore:
    controller_kind: str
    candidate_index: int
    action: Action | None
    feedback_gain: float | None
    mean_tke: float
    rms_l2_effort: float
    case_mean_tke: tuple[tuple[str, float], ...]
    numerical_valid: bool


@dataclass(frozen=True)
class DevelopmentSearchFailure:
    controller_kind: str
    candidate_index: int
    case_id: str
    failed_numerical_gates: tuple[str, ...] = ()
    completed_case_ids: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None


class Gate0DevelopmentSearchError(RuntimeError):
    """Development search failed without producing a controller lock."""

    def __init__(
        self,
        protocol_fingerprint: str,
        search_scores: Sequence[DevelopmentCandidateScore],
        failures: Sequence[DevelopmentSearchFailure],
    ) -> None:
        super().__init__("development controller search did not produce a valid lock")
        self.protocol_fingerprint = _sha256_identifier(
            protocol_fingerprint,
            "protocol_fingerprint",
        )
        self.search_scores = tuple(search_scores)
        self.failures = tuple(failures)

    def as_dict(self) -> dict[str, object]:
        reason = (
            "development_candidate_execution_error"
            if any(failure.error_type is not None for failure in self.failures)
            else "development_candidate_numerical_gate_failure"
        )
        return {
            "status": "failed",
            "reason": reason,
            "protocol_fingerprint": self.protocol_fingerprint,
            "search_scores": [asdict(score) for score in self.search_scores],
            "failures": [asdict(failure) for failure in self.failures],
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())


@dataclass(frozen=True)
class Gate0DevelopmentLock:
    protocol_fingerprint: str
    fixed_action: Action
    feedback_gain: float
    development_fixed_rms_l2: float
    development_feedback_rms_l2: float
    development_case_ids: tuple[str, ...]
    search_scores: tuple[DevelopmentCandidateScore, ...]
    selection_rule: str = (
        "feedback gain minimizes development TKE; fixed constant first minimizes absolute RMS-L2 effort mismatch "
        "to that feedback arm, then development TKE, then stable grid order"
    )

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


def lock_development_controls(config: Gate0Config, factory: EpisodeFactory) -> Gate0DevelopmentLock:
    """Tune only finite preregistered grids on development cases, never held-out cases."""
    cases = config.cases("development")
    completed_scores: list[DevelopmentCandidateScore] = []
    failures: list[DevelopmentSearchFailure] = []

    def score(
        controller: Callable[[Observation], Action],
        arm: str,
        candidate_index: int,
        *,
        action: Action | None = None,
        feedback_gain: float | None = None,
    ) -> DevelopmentCandidateScore:
        traces = []
        for case in cases:
            try:
                trace = _run_episode(config, factory, case, arm, controller)
            except Exception as error:
                failures.append(
                    DevelopmentSearchFailure(
                        controller_kind=arm,
                        candidate_index=candidate_index,
                        case_id=case.case_id,
                        completed_case_ids=tuple(trace.case.case_id for trace in traces),
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                )
                raise Gate0DevelopmentSearchError(
                    config.fingerprint,
                    completed_scores,
                    failures,
                ) from error
            traces.append(trace)
            failed_gates = tuple(
                sorted(
                    name
                    for name in REQUIRED_NUMERICAL_GATES
                    if trace.numerical_gates[name] is not True
                )
            )
            if failed_gates:
                failures.append(
                    DevelopmentSearchFailure(
                        controller_kind=arm,
                        candidate_index=candidate_index,
                        case_id=case.case_id,
                        failed_numerical_gates=failed_gates,
                    )
                )
        candidate_score = DevelopmentCandidateScore(
            controller_kind=arm,
            candidate_index=candidate_index,
            action=action,
            feedback_gain=feedback_gain,
            mean_tke=sum(trace.mean_tke for trace in traces) / len(traces),
            rms_l2_effort=sum(trace.rms_l2_effort for trace in traces) / len(traces),
            case_mean_tke=tuple((trace.case.case_id, trace.mean_tke) for trace in traces),
            numerical_valid=not any(
                failure.controller_kind == arm
                and failure.candidate_index == candidate_index
                for failure in failures
            ),
        )
        completed_scores.append(candidate_score)
        return candidate_score

    constant_scores = []
    for index, action in enumerate(config.constant_candidates):
        constant_scores.append(
            score(
                lambda _observation, action=action: action,
                "development_fixed",
                index,
                action=action,
            )
        )
    gain_scores = []
    for index, gain in enumerate(config.feedback_gain_candidates):
        gain_scores.append(
            score(
                FrozenSignedController(gain, config.radial_action_bound),
                "development_feedback",
                index,
                feedback_gain=gain,
            )
        )
    if not all(item.numerical_valid for item in (*constant_scores, *gain_scores)):
        raise Gate0DevelopmentSearchError(
            config.fingerprint,
            (*constant_scores, *gain_scores),
            failures,
        )
    gain_index = min(range(len(gain_scores)), key=lambda index: (gain_scores[index].mean_tke, index))
    feedback_effort = gain_scores[gain_index].rms_l2_effort
    constant_index = min(
        range(len(constant_scores)),
        key=lambda index: (
            abs(constant_scores[index].rms_l2_effort - feedback_effort),
            constant_scores[index].mean_tke,
            index,
        ),
    )
    return Gate0DevelopmentLock(
        protocol_fingerprint=config.fingerprint,
        fixed_action=config.constant_candidates[constant_index],
        feedback_gain=config.feedback_gain_candidates[gain_index],
        development_fixed_rms_l2=constant_scores[constant_index].rms_l2_effort,
        development_feedback_rms_l2=feedback_effort,
        development_case_ids=tuple(case.case_id for case in cases),
        search_scores=tuple((*constant_scores, *gain_scores)),
    )


def _derangement(cases: Sequence[Gate0Case]) -> dict[str, Gate0Case]:
    """Bijectively shift both phase and seed, preserving whole-episode time structure."""
    by_cell = {(case.phase_index, case.seed_index): case for case in cases}
    phase_count = len({case.phase_index for case in cases})
    seed_count = len({case.seed_index for case in cases})
    mapping = {
        case.case_id: by_cell[((case.phase_index + 1) % phase_count, (case.seed_index + 1) % seed_count)]
        for case in cases
    }
    target_source_pairs = ((case, mapping[case.case_id]) for case in cases)
    if any(
        source.phase_index == target.phase_index or source.seed_index == target.seed_index
        for target, source in target_source_pairs
    ):
        raise RuntimeError("episode derangement must mismatch both phase and seed")
    if len({source.case_id for source in mapping.values()}) != len(cases):
        raise RuntimeError("episode derangement must be one-to-one")
    return mapping


@dataclass(frozen=True)
class Gate0RefinementProvenance:
    """Immutable identity of one temporal or spatial refinement run set."""

    label: str
    grid_size: tuple[int, int]
    dt: float
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("refinement label must be a nonempty string")
        grid_size = tuple(self.grid_size)
        if len(grid_size) != 2 or any(type(value) is not int or value <= 0 for value in grid_size):
            raise ValueError("refinement grid_size must contain two positive integers")
        dt = _finite(self.dt, "refinement dt")
        if dt <= 0.0:
            raise ValueError("refinement dt must be positive")
        case_ids = tuple(self.case_ids)
        if not case_ids or any(not isinstance(case_id, str) or not case_id.strip() for case_id in case_ids):
            raise ValueError("refinement case_ids must contain nonempty strings")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("refinement case_ids must not contain duplicates")
        object.__setattr__(self, "grid_size", grid_size)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "case_ids", case_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "grid_size": list(self.grid_size),
            "dt": self.dt,
            "case_ids": list(self.case_ids),
        }


@dataclass(frozen=True)
class Gate0RefinementTraceSummary:
    arm: str
    case_id: str
    source_case_id: str | None
    mean_tke: float
    rms_l2_effort: float
    integrated_l2_energy: float
    initial_state_digest: str
    control_start_digest: str
    scored_start_digest: str
    uncontrolled_reset_prelude_intervals: int
    explicit_uncontrolled_intervals: int
    controller_input_digest: str
    action_history_digest: str
    numerical_gates: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not self.arm or not self.case_id:
            raise ValueError("refinement trace arm and case_id must be nonempty")
        for name in (
            "initial_state_digest",
            "control_start_digest",
            "scored_start_digest",
            "controller_input_digest",
            "action_history_digest",
        ):
            _sha256_identifier(getattr(self, name), name)
        for name in ("mean_tke", "rms_l2_effort", "integrated_l2_energy"):
            value = _finite(getattr(self, name), name)
            if name != "mean_tke" and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if any(
            type(value) is not int or value < 0
            for value in (
                self.uncontrolled_reset_prelude_intervals,
                self.explicit_uncontrolled_intervals,
            )
        ):
            raise ValueError("refinement trace horizon counts must be nonnegative integers")
        gates = dict(self.numerical_gates)
        if set(gates) != REQUIRED_NUMERICAL_GATES or any(
            type(value) is not bool for value in gates.values()
        ):
            raise ValueError("refinement trace numerical gates must match the required schema")
        object.__setattr__(self, "numerical_gates", MappingProxyType(gates))

    @classmethod
    def from_trace(cls, trace: ArmTrace) -> Gate0RefinementTraceSummary:
        return cls(
            arm=trace.arm,
            case_id=trace.case.case_id,
            source_case_id=trace.source_case_id,
            mean_tke=trace.mean_tke,
            rms_l2_effort=trace.rms_l2_effort,
            integrated_l2_energy=trace.integrated_l2_energy,
            initial_state_digest=trace.initial_state_digest,
            control_start_digest=trace.control_start_digest,
            scored_start_digest=trace.scored_start_digest,
            uncontrolled_reset_prelude_intervals=trace.uncontrolled_reset_prelude_intervals,
            explicit_uncontrolled_intervals=trace.explicit_uncontrolled_intervals,
            controller_input_digest=_digest(trace.controller_input_history),
            action_history_digest=_digest(trace.action_history),
            numerical_gates=trace.numerical_gates,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "case_id": self.case_id,
            "source_case_id": self.source_case_id,
            "mean_tke": self.mean_tke,
            "rms_l2_effort": self.rms_l2_effort,
            "integrated_l2_energy": self.integrated_l2_energy,
            "initial_state_digest": self.initial_state_digest,
            "control_start_digest": self.control_start_digest,
            "scored_start_digest": self.scored_start_digest,
            "uncontrolled_reset_prelude_intervals": self.uncontrolled_reset_prelude_intervals,
            "explicit_uncontrolled_intervals": self.explicit_uncontrolled_intervals,
            "controller_input_digest": self.controller_input_digest,
            "action_history_digest": self.action_history_digest,
            "numerical_gates": {
                name: self.numerical_gates[name] for name in sorted(self.numerical_gates)
            },
        }


@dataclass(frozen=True)
class Gate0RefinementEvidence:
    provenance: Gate0RefinementProvenance
    target_case_ids: tuple[str, ...]
    traces: tuple[Gate0RefinementTraceSummary, ...]
    primary_order: tuple[str, ...]
    refined_order: tuple[str, ...]
    primary_decisions: Mapping[str, bool]
    refined_decisions: Mapping[str, bool]

    def __post_init__(self) -> None:
        target_case_ids = tuple(self.target_case_ids)
        if not target_case_ids or len(set(target_case_ids)) != len(target_case_ids):
            raise ValueError("refinement target_case_ids must be nonempty and unique")
        if not set(target_case_ids) <= set(self.provenance.case_ids):
            raise ValueError("refinement target cases must be present in provenance")
        traces = tuple(self.traces)
        if not traces:
            raise ValueError("refinement evidence must contain trace summaries")
        feedback_by_case = {
            trace.case_id: trace for trace in traces if trace.arm == "signed_feedback"
        }
        for case_id in target_case_ids:
            case_traces = tuple(trace for trace in traces if trace.case_id == case_id)
            if {trace.arm for trace in case_traces} != set(_CONVERGENCE_ARM_ORDER):
                raise ValueError("each refinement target case must contain every convergence arm")
            if len({trace.control_start_digest for trace in case_traces}) != 1:
                raise ValueError("refinement target arms must share one control-start state")
            if len(
                {
                    (
                        trace.uncontrolled_reset_prelude_intervals,
                        trace.explicit_uncontrolled_intervals,
                    )
                    for trace in case_traces
                }
            ) != 1:
                raise ValueError("refinement target arms must share one uncontrolled horizon")
            deranged = next(
                trace for trace in case_traces if trace.arm == "observation_deranged"
            )
            source = feedback_by_case.get(deranged.source_case_id or "")
            if source is None or source.controller_input_digest != deranged.controller_input_digest:
                raise ValueError("deranged refinement inputs must bind to their feedback source")
        if set(self.primary_order) != set(_CONVERGENCE_ARM_ORDER) or set(
            self.refined_order
        ) != set(_CONVERGENCE_ARM_ORDER):
            raise ValueError("refinement orderings must contain every convergence arm")
        primary_decisions = dict(self.primary_decisions)
        refined_decisions = dict(self.refined_decisions)
        if (
            not primary_decisions
            or set(primary_decisions) != set(refined_decisions)
            or any(
                type(value) is not bool
                for value in (*primary_decisions.values(), *refined_decisions.values())
            )
        ):
            raise ValueError("refinement decision vectors must have identical boolean keys")
        object.__setattr__(self, "target_case_ids", target_case_ids)
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "primary_order", tuple(self.primary_order))
        object.__setattr__(self, "refined_order", tuple(self.refined_order))
        object.__setattr__(self, "primary_decisions", MappingProxyType(primary_decisions))
        object.__setattr__(self, "refined_decisions", MappingProxyType(refined_decisions))

    def as_dict(self) -> dict[str, object]:
        return {
            "provenance": self.provenance.as_dict(),
            "target_case_ids": list(self.target_case_ids),
            "traces": [trace.as_dict() for trace in self.traces],
            "primary_order": list(self.primary_order),
            "refined_order": list(self.refined_order),
            "primary_decisions": {
                name: self.primary_decisions[name] for name in sorted(self.primary_decisions)
            },
            "refined_decisions": {
                name: self.refined_decisions[name] for name in sorted(self.refined_decisions)
            },
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())


@dataclass(frozen=True)
class Gate0ConvergenceAttestation:
    """Content-addressed evidence linked to the frozen primary-report artifact."""

    protocol_fingerprint: str
    development_lock_digest: str
    primary_report_digest: str
    convergence_seed: int
    temporal_provenance: Gate0RefinementProvenance
    spatial_provenance: Gate0RefinementProvenance
    refinement_evidence_digests: Mapping[str, str]
    gates: Mapping[str, bool]
    metrics: Mapping[str, float]
    schema_version: str = GATE0_CONVERGENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GATE0_CONVERGENCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {GATE0_CONVERGENCE_SCHEMA_VERSION}")
        _sha256_identifier(self.protocol_fingerprint, "protocol_fingerprint")
        _sha256_identifier(self.development_lock_digest, "development_lock_digest")
        _sha256_identifier(self.primary_report_digest, "primary_report_digest")
        if type(self.convergence_seed) is not int or self.convergence_seed < 0:
            raise ValueError("convergence_seed must be a nonnegative integer")
        if not isinstance(self.temporal_provenance, Gate0RefinementProvenance):
            raise TypeError("temporal_provenance must be Gate0RefinementProvenance")
        if not isinstance(self.spatial_provenance, Gate0RefinementProvenance):
            raise TypeError("spatial_provenance must be Gate0RefinementProvenance")
        evidence_digests = dict(self.refinement_evidence_digests)
        if set(evidence_digests) != {"temporal", "spatial"}:
            raise ValueError("refinement evidence digests must contain temporal and spatial evidence")
        for label, value in evidence_digests.items():
            _sha256_identifier(value, f"{label}_refinement_evidence_digest")

        gates = dict(self.gates)
        if set(gates) != REQUIRED_CONVERGENCE_GATES:
            missing = sorted(REQUIRED_CONVERGENCE_GATES - gates.keys())
            extra = sorted(gates.keys() - REQUIRED_CONVERGENCE_GATES)
            raise ValueError(f"convergence gates must match the required schema; missing={missing}, extra={extra}")
        if any(type(value) is not bool for value in gates.values()):
            raise ValueError("convergence gate values must be booleans")

        if not self.metrics:
            raise ValueError("convergence metrics must not be empty")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("convergence metric names must be nonempty strings")
            if isinstance(value, bool):
                raise ValueError("convergence metric values must be numeric")
            metrics[name] = _finite(value, f"convergence metric {name!r}")

        object.__setattr__(self, "gates", MappingProxyType(gates))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(
            self,
            "refinement_evidence_digests",
            MappingProxyType(evidence_digests),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_fingerprint": self.protocol_fingerprint,
            "development_lock_digest": self.development_lock_digest,
            "primary_report_digest": self.primary_report_digest,
            "convergence_seed": self.convergence_seed,
            "temporal_provenance": self.temporal_provenance.as_dict(),
            "spatial_provenance": self.spatial_provenance.as_dict(),
            "refinement_evidence_digests": {
                label: self.refinement_evidence_digests[label]
                for label in sorted(self.refinement_evidence_digests)
            },
            "gates": {name: self.gates[name] for name in sorted(self.gates)},
            "metrics": {name: self.metrics[name] for name in sorted(self.metrics)},
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())


@dataclass(frozen=True)
class Gate0ConvergenceRun:
    attestation: Gate0ConvergenceAttestation
    temporal_evidence: Gate0RefinementEvidence
    spatial_evidence: Gate0RefinementEvidence

    def __post_init__(self) -> None:
        expected = {
            "temporal": self.temporal_evidence.digest,
            "spatial": self.spatial_evidence.digest,
        }
        if dict(self.attestation.refinement_evidence_digests) != expected:
            raise ValueError("attestation does not bind the supplied refinement evidence")

    def evidence_as_dict(self) -> dict[str, object]:
        return {
            "temporal": {
                **self.temporal_evidence.as_dict(),
                "evidence_digest": self.temporal_evidence.digest,
            },
            "spatial": {
                **self.spatial_evidence.as_dict(),
                "evidence_digest": self.spatial_evidence.digest,
            },
        }


@dataclass(frozen=True)
class Gate0Report:
    protocol_fingerprint: str
    development_lock_digest: str
    traces: tuple[ArmTrace, ...]
    gates: Mapping[str, bool]
    paired_seed_deltas: Mapping[str, Mapping[int, float]]
    convergence_attestation_digest: str | None = None
    convergence_gates: Mapping[str, bool] | None = None
    claim_boundary: str = CLAIM_BOUNDARY
    rl_training_performed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", tuple(self.traces))
        if isinstance(self.gates, Mapping):
            object.__setattr__(self, "gates", MappingProxyType(dict(self.gates)))
        if isinstance(self.paired_seed_deltas, Mapping):
            object.__setattr__(
                self,
                "paired_seed_deltas",
                MappingProxyType(
                    {
                        baseline: MappingProxyType(dict(values))
                        for baseline, values in self.paired_seed_deltas.items()
                    }
                ),
            )
        if isinstance(self.convergence_gates, Mapping):
            object.__setattr__(
                self,
                "convergence_gates",
                MappingProxyType(dict(self.convergence_gates)),
            )

    @property
    def primary_passed(self) -> bool:
        return (
            isinstance(self.gates, Mapping)
            and set(self.gates) == REQUIRED_PRIMARY_GATES
            and all(value is True for value in self.gates.values())
        )

    def with_convergence(
        self,
        attestation: Gate0ConvergenceAttestation,
        *,
        primary_report_digest: str,
    ) -> Gate0Report:
        """Attach matching content-addressed convergence evidence without mutating the primary report."""
        if not isinstance(attestation, Gate0ConvergenceAttestation):
            raise TypeError("attestation must be Gate0ConvergenceAttestation")
        _sha256_identifier(primary_report_digest, "primary_report_digest")
        if attestation.protocol_fingerprint != self.protocol_fingerprint:
            raise ValueError("convergence attestation protocol fingerprint does not match the primary report")
        if attestation.development_lock_digest != self.development_lock_digest:
            raise ValueError("convergence attestation development lock does not match the primary report")
        if attestation.primary_report_digest != primary_report_digest:
            raise ValueError("convergence attestation does not match the primary-report artifact digest")
        return replace(
            self,
            convergence_attestation_digest=attestation.digest,
            convergence_gates=attestation.gates,
        )

    @property
    def passed(self) -> bool:
        """Final evidence fails closed until preregistered temporal/spatial checks attest it."""
        return (
            self.primary_passed
            and _is_sha256_identifier(self.convergence_attestation_digest)
            and isinstance(self.convergence_gates, Mapping)
            and set(self.convergence_gates) == REQUIRED_CONVERGENCE_GATES
            and all(value is True for value in self.convergence_gates.values())
        )


def _mean_tke(traces: Sequence[ArmTrace]) -> float:
    return sum(trace.mean_tke for trace in traces) / len(traces)


def _material(config: Gate0Config, baseline: Sequence[ArmTrace], candidate: Sequence[ArmTrace]) -> bool:
    baseline_mean, candidate_mean = _mean_tke(baseline), _mean_tke(candidate)
    absolute = baseline_mean - candidate_mean
    relative = absolute / baseline_mean if baseline_mean > 0.0 else -math.inf
    return absolute >= config.minimum_absolute_tke_reduction and relative >= config.minimum_relative_tke_reduction


def _mean_effort(traces: Sequence[ArmTrace]) -> float:
    return sum(trace.rms_l2_effort for trace in traces) / len(traces)


def _opposite_pairs_material(
    config: Gate0Config,
    baseline: Sequence[ArmTrace],
    candidate: Sequence[ArmTrace],
) -> bool:
    return _opposite_pairs_material_for_seeds(
        config,
        baseline,
        candidate,
        config.heldout_seeds,
    )


def _opposite_pairs_material_for_seeds(
    config: Gate0Config,
    baseline: Sequence[ArmTrace],
    candidate: Sequence[ArmTrace],
    seeds: Sequence[int],
) -> bool:
    baseline_by_cell = {(trace.case.seed, trace.case.phase_turns): trace for trace in baseline}
    candidate_by_cell = {(trace.case.seed, trace.case.phase_turns): trace for trace in candidate}
    checked: set[tuple[float, float]] = set()
    for phase in config.heldout_phase_turns:
        opposite = next(
            other
            for other in config.heldout_phase_turns
            if math.isclose((phase + 0.5) % 1.0, other, abs_tol=1.0e-12)
        )
        pair = tuple(sorted((phase, opposite)))
        if pair in checked:
            continue
        checked.add(pair)
        pair_baseline = [
            baseline_by_cell[(seed, member)] for seed in seeds for member in pair
        ]
        pair_candidate = [
            candidate_by_cell[(seed, member)] for seed in seeds for member in pair
        ]
        if not _material(config, pair_baseline, pair_candidate):
            return False
    return True


def run_gate0(config: Gate0Config, lock: Gate0DevelopmentLock, factory: EpisodeFactory) -> Gate0Report:
    """Execute held-out arms exactly once under a development-only controller lock."""
    if lock.protocol_fingerprint != config.fingerprint:
        raise ValueError("development lock does not match the preregistered held-out protocol")
    cases = config.cases("heldout")
    feedback_controller = FrozenSignedController(lock.feedback_gain, config.radial_action_bound)
    traces_by_arm: dict[str, list[ArmTrace]] = {name: [] for name in ("zero", "fixed", "oracle", "signed_feedback")}
    for case in cases:
        traces_by_arm["zero"].append(_run_episode(config, factory, case, "zero", lambda _obs: ZERO_ACTION))
        traces_by_arm["fixed"].append(
            _run_episode(config, factory, case, "fixed", lambda _obs, action=lock.fixed_action: action)
        )
        traces_by_arm["oracle"].append(
            _run_episode(
                config,
                factory,
                case,
                "oracle",
                lambda _obs, case=case: oracle_action(case, config.radial_action_bound),
            )
        )
        traces_by_arm["signed_feedback"].append(
            _run_episode(config, factory, case, "signed_feedback", feedback_controller)
        )

    feedback_by_case = {trace.case.case_id: trace for trace in traces_by_arm["signed_feedback"]}
    derangement = _derangement(cases)
    shuffled = []
    for case in cases:
        source = feedback_by_case[derangement[case.case_id].case_id]
        inputs = source.controller_input_history
        shuffled.append(
            _run_episode(
                config,
                factory,
                case,
                "observation_deranged",
                feedback_controller,
                controller_inputs=inputs,
                source_case_id=source.case.case_id,
            )
        )
    traces_by_arm["observation_deranged"] = shuffled

    aligned_inputs = [value for trace in traces_by_arm["signed_feedback"] for value in trace.controller_input_history]
    shuffled_inputs = [value for trace in shuffled for value in trace.controller_input_history]
    aligned_actions = [value for trace in traces_by_arm["signed_feedback"] for value in trace.action_history]
    shuffled_actions = [value for trace in shuffled for value in trace.action_history]
    observation_marginal_exact = Counter(aligned_inputs) == Counter(shuffled_inputs)
    action_marginal_exact = Counter(aligned_actions) == Counter(shuffled_actions)
    effort_exact = math.isclose(
        sum(action[0] ** 2 + action[1] ** 2 for action in aligned_actions),
        sum(action[0] ** 2 + action[1] ** 2 for action in shuffled_actions),
        rel_tol=0.0,
        abs_tol=config.effort_match_atol,
    )
    starts_match = all(
        len({traces_by_arm[arm][index].control_start_digest for arm in traces_by_arm}) == 1
        for index in range(len(cases))
    )
    numerical_valid = all(
        all(trace.numerical_gates[name] for name in REQUIRED_NUMERICAL_GATES)
        for traces in traces_by_arm.values()
        for trace in traces
    )
    source_observations_differ = all(
        _digest([record.controller_observation for record in trace.records])
        != _digest([record.controller_observation for record in feedback_by_case[trace.case.case_id].records])
        for trace in shuffled
    )
    uncontrolled_horizon_exact = all(
        trace.uncontrolled_reset_prelude_intervals + trace.explicit_uncontrolled_intervals
        == config.uncontrolled_burn_in_intervals
        for traces in traces_by_arm.values()
        for trace in traces
    )

    paired_seed_deltas: dict[str, Mapping[int, float]] = {}
    for baseline_name in ("zero", "fixed", "observation_deranged"):
        deltas = {}
        for seed in config.heldout_seeds:
            baseline = [trace.mean_tke for trace in traces_by_arm[baseline_name] if trace.case.seed == seed]
            candidate = [trace.mean_tke for trace in traces_by_arm["signed_feedback"] if trace.case.seed == seed]
            deltas[seed] = sum(baseline) / len(baseline) - sum(candidate) / len(candidate)
        paired_seed_deltas[baseline_name] = MappingProxyType(deltas)

    seed_wins = sum(
        delta > 0.0 for delta in paired_seed_deltas["observation_deranged"].values()
    ) / len(config.heldout_seeds)
    gates = MappingProxyType(
        {
            "matched_initial_states": starts_match,
            "uncontrolled_horizon_exact": uncontrolled_horizon_exact,
            "numerical_validity": numerical_valid,
            "phase_and_seed_derangement": all(
                trace.source_case_id is not None
                and derangement[trace.case.case_id].case_id == trace.source_case_id
                for trace in shuffled
            ),
            "source_observation_trajectories_differ": source_observations_differ,
            "observation_marginal_exact": observation_marginal_exact,
            "action_marginal_exact": action_marginal_exact,
            "rotation_invariant_effort_exact": effort_exact,
            "oracle_materially_beats_zero": _material(config, traces_by_arm["zero"], traces_by_arm["oracle"]),
            "oracle_materially_beats_fixed": _material(
                config, traces_by_arm["fixed"], traces_by_arm["oracle"]
            ),
            "oracle_opposite_phase_pairs_material": _opposite_pairs_material(
                config, traces_by_arm["zero"], traces_by_arm["oracle"]
            ),
            "oracle_beats_fixed_in_opposite_phase_pairs": _opposite_pairs_material(
                config, traces_by_arm["fixed"], traces_by_arm["oracle"]
            ),
            "feedback_materially_beats_zero": _material(
                config, traces_by_arm["zero"], traces_by_arm["signed_feedback"]
            ),
            "feedback_materially_beats_fixed": _material(
                config, traces_by_arm["fixed"], traces_by_arm["signed_feedback"]
            ),
            "feedback_materially_beats_deranged": _material(
                config, traces_by_arm["observation_deranged"], traces_by_arm["signed_feedback"]
            ),
            "feedback_opposite_phase_pairs_material": _opposite_pairs_material(
                config, traces_by_arm["observation_deranged"], traces_by_arm["signed_feedback"]
            ),
            "feedback_beats_fixed_in_opposite_phase_pairs": _opposite_pairs_material(
                config, traces_by_arm["fixed"], traces_by_arm["signed_feedback"]
            ),
            "feedback_within_oracle_effort_budget": _mean_effort(traces_by_arm["signed_feedback"])
            <= _mean_effort(traces_by_arm["oracle"]) + config.effort_match_atol,
            "fixed_independent_seed_block_wins": (
                sum(delta > 0.0 for delta in paired_seed_deltas["fixed"].values())
                / len(config.heldout_seeds)
                >= config.minimum_seed_win_fraction
            ),
            "deranged_independent_seed_block_wins": seed_wins >= config.minimum_seed_win_fraction,
        }
    )
    arm_order = ("zero", "fixed", "oracle", "signed_feedback", "observation_deranged")
    all_traces = tuple(trace for arm in arm_order for trace in traces_by_arm[arm])
    return Gate0Report(
        protocol_fingerprint=config.fingerprint,
        development_lock_digest=lock.digest,
        traces=all_traces,
        gates=gates,
        paired_seed_deltas=MappingProxyType(paired_seed_deltas),
    )


_CONVERGENCE_ARM_ORDER = ("zero", "fixed", "oracle", "signed_feedback", "observation_deranged")
_CONVERGENCE_EFFECT_PAIRS = (
    ("oracle_vs_zero", "zero", "oracle"),
    ("oracle_vs_fixed", "fixed", "oracle"),
    ("feedback_vs_zero", "zero", "signed_feedback"),
    ("feedback_vs_fixed", "fixed", "signed_feedback"),
    ("feedback_vs_deranged", "observation_deranged", "signed_feedback"),
)


def _group_convergence_traces(
    traces: Sequence[ArmTrace],
    expected_case_ids: set[str],
) -> dict[str, tuple[ArmTrace, ...]]:
    grouped = {
        arm: tuple(trace for trace in traces if trace.arm == arm and trace.case.case_id in expected_case_ids)
        for arm in _CONVERGENCE_ARM_ORDER
    }
    for arm, arm_traces in grouped.items():
        case_ids = [trace.case.case_id for trace in arm_traces]
        if len(case_ids) != len(expected_case_ids) or set(case_ids) != expected_case_ids:
            raise ValueError(f"primary report does not contain exactly one {arm} trace per convergence case")
    return grouped


def _run_refinement_traces(
    config: Gate0Config,
    lock: Gate0DevelopmentLock,
    factory: EpisodeFactory,
    *,
    label: str,
    grid_size: tuple[int, int],
    dt: float,
) -> tuple[tuple[ArmTrace, ...], tuple[ArmTrace, ...], Gate0RefinementProvenance]:
    all_cases = config.cases("heldout")
    target_cases = tuple(case for case in all_cases if case.seed == config.convergence_seed)
    if not target_cases:
        raise ValueError("convergence_seed has no held-out cases")
    derangement = _derangement(all_cases)
    by_case_id = {case.case_id: case for case in all_cases}
    source_case_ids = {derangement[case.case_id].case_id for case in target_cases}
    source_cases = tuple(case for case in all_cases if case.case_id in source_case_ids)
    feedback_controller = FrozenSignedController(lock.feedback_gain, config.radial_action_bound)
    target_by_arm: dict[str, list[ArmTrace]] = {
        arm: [] for arm in _CONVERGENCE_ARM_ORDER
    }
    executed: list[ArmTrace] = []
    feedback_by_case: dict[str, ArmTrace] = {}

    for case in target_cases:
        traces = (
            _run_episode(config, factory, case, "zero", lambda _observation: ZERO_ACTION),
            _run_episode(
                config,
                factory,
                case,
                "fixed",
                lambda _observation, action=lock.fixed_action: action,
            ),
            _run_episode(
                config,
                factory,
                case,
                "oracle",
                lambda _observation, current_case=case: oracle_action(
                    current_case,
                    config.radial_action_bound,
                ),
            ),
            _run_episode(config, factory, case, "signed_feedback", feedback_controller),
        )
        for trace in traces:
            target_by_arm[trace.arm].append(trace)
            executed.append(trace)
        feedback_by_case[case.case_id] = traces[-1]

    for case in source_cases:
        if case.case_id in feedback_by_case:
            continue
        trace = _run_episode(config, factory, case, "signed_feedback", feedback_controller)
        feedback_by_case[case.case_id] = trace
        executed.append(trace)

    for case in target_cases:
        source_id = derangement[case.case_id].case_id
        source = feedback_by_case[source_id]
        trace = _run_episode(
            config,
            factory,
            case,
            "observation_deranged",
            feedback_controller,
            controller_inputs=source.controller_input_history,
            source_case_id=by_case_id[source_id].case_id,
        )
        target_by_arm[trace.arm].append(trace)
        executed.append(trace)

    target_traces = tuple(
        trace for arm in _CONVERGENCE_ARM_ORDER for trace in target_by_arm[arm]
    )
    executed_case_ids = tuple(dict.fromkeys(trace.case.case_id for trace in executed))
    provenance = Gate0RefinementProvenance(label, grid_size, dt, executed_case_ids)
    return target_traces, tuple(executed), provenance


def _relative_tke_effect(baseline: float, candidate: float) -> float:
    """Relative TKE reduction; the floor only defines the otherwise singular zero-baseline case."""
    return (baseline - candidate) / max(abs(baseline), 1.0e-12)


def _convergence_decisions(
    config: Gate0Config,
    traces_by_arm: Mapping[str, Sequence[ArmTrace]],
) -> dict[str, bool]:
    decisions = {
        name: _material(config, traces_by_arm[baseline], traces_by_arm[candidate])
        for name, baseline, candidate in _CONVERGENCE_EFFECT_PAIRS
    }
    decisions.update(
        {
            "oracle_zero_opposite_pairs": _opposite_pairs_material_for_seeds(
                config,
                traces_by_arm["zero"],
                traces_by_arm["oracle"],
                (config.convergence_seed,),
            ),
            "oracle_fixed_opposite_pairs": _opposite_pairs_material_for_seeds(
                config,
                traces_by_arm["fixed"],
                traces_by_arm["oracle"],
                (config.convergence_seed,),
            ),
            "feedback_deranged_opposite_pairs": _opposite_pairs_material_for_seeds(
                config,
                traces_by_arm["observation_deranged"],
                traces_by_arm["signed_feedback"],
                (config.convergence_seed,),
            ),
            "feedback_fixed_opposite_pairs": _opposite_pairs_material_for_seeds(
                config,
                traces_by_arm["fixed"],
                traces_by_arm["signed_feedback"],
                (config.convergence_seed,),
            ),
            "feedback_within_oracle_effort_budget": _mean_effort(
                traces_by_arm["signed_feedback"]
            )
            <= _mean_effort(traces_by_arm["oracle"]) + config.effort_match_atol,
        }
    )
    return decisions


def _evaluate_refinement(
    config: Gate0Config,
    primary_by_arm: Mapping[str, Sequence[ArmTrace]],
    refined_traces: Sequence[ArmTrace],
    executed_traces: Sequence[ArmTrace],
    provenance: Gate0RefinementProvenance,
    *,
    label: str,
    maximum_arm_tke_relative_difference: float,
    maximum_effect_difference: float,
) -> tuple[dict[str, bool], dict[str, float], Gate0RefinementEvidence]:
    expected_case_ids = {
        trace.case.case_id for trace in primary_by_arm[_CONVERGENCE_ARM_ORDER[0]]
    }
    refined_by_arm = _group_convergence_traces(refined_traces, expected_case_ids)
    primary_means = {arm: _mean_tke(primary_by_arm[arm]) for arm in _CONVERGENCE_ARM_ORDER}
    refined_means = {arm: _mean_tke(refined_by_arm[arm]) for arm in _CONVERGENCE_ARM_ORDER}
    arm_differences = {
        arm: abs(refined_means[arm] - primary_means[arm])
        / max(abs(primary_means[arm]), 1.0e-12)
        for arm in _CONVERGENCE_ARM_ORDER
    }
    primary_effects = {
        name: _relative_tke_effect(primary_means[baseline], primary_means[candidate])
        for name, baseline, candidate in _CONVERGENCE_EFFECT_PAIRS
    }
    refined_effects = {
        name: _relative_tke_effect(refined_means[baseline], refined_means[candidate])
        for name, baseline, candidate in _CONVERGENCE_EFFECT_PAIRS
    }
    effect_differences = {
        name: abs(refined_effects[name] - primary_effects[name]) for name in primary_effects
    }
    primary_order = tuple(sorted(_CONVERGENCE_ARM_ORDER, key=lambda arm: (primary_means[arm], arm)))
    refined_order = tuple(sorted(_CONVERGENCE_ARM_ORDER, key=lambda arm: (refined_means[arm], arm)))
    primary_decisions = _convergence_decisions(config, primary_by_arm)
    refined_decisions = _convergence_decisions(config, refined_by_arm)
    decision_mismatches = sum(
        primary_decisions[name] != refined_decisions[name] for name in primary_decisions
    )
    numerical_valid = all(
        set(trace.numerical_gates) >= REQUIRED_NUMERICAL_GATES
        and all(trace.numerical_gates[name] is True for name in REQUIRED_NUMERICAL_GATES)
        for trace in executed_traces
    )
    maximum_arm_difference = max(arm_differences.values())
    maximum_observed_effect_difference = max(effect_differences.values())
    gates = {
        f"{label}_numerical_validity": numerical_valid,
        f"{label}_arm_tke_convergence": (
            maximum_arm_difference <= maximum_arm_tke_relative_difference
        ),
        f"{label}_effect_convergence": (
            maximum_observed_effect_difference <= maximum_effect_difference
        ),
        f"{label}_ordering_preservation": primary_order == refined_order,
        f"{label}_pass_decision_preservation": decision_mismatches == 0,
    }
    metrics: dict[str, float] = {
        f"{label}_maximum_arm_tke_relative_difference": maximum_arm_difference,
        f"{label}_maximum_arm_tke_relative_difference_limit": (
            maximum_arm_tke_relative_difference
        ),
        f"{label}_maximum_effect_difference": maximum_observed_effect_difference,
        f"{label}_maximum_effect_difference_limit": maximum_effect_difference,
        f"{label}_pass_decision_mismatch_count": float(decision_mismatches),
    }
    for arm in _CONVERGENCE_ARM_ORDER:
        metrics[f"primary_{arm}_mean_tke"] = primary_means[arm]
        metrics[f"{label}_{arm}_mean_tke"] = refined_means[arm]
        metrics[f"{label}_{arm}_tke_relative_difference"] = arm_differences[arm]
    for name, _, _ in _CONVERGENCE_EFFECT_PAIRS:
        metrics[f"primary_{name}_relative_effect"] = primary_effects[name]
        metrics[f"{label}_{name}_relative_effect"] = refined_effects[name]
        metrics[f"{label}_{name}_absolute_effect_difference"] = effect_differences[name]
    evidence = Gate0RefinementEvidence(
        provenance=provenance,
        target_case_ids=tuple(
            trace.case.case_id for trace in primary_by_arm[_CONVERGENCE_ARM_ORDER[0]]
        ),
        traces=tuple(Gate0RefinementTraceSummary.from_trace(trace) for trace in executed_traces),
        primary_order=primary_order,
        refined_order=refined_order,
        primary_decisions=primary_decisions,
        refined_decisions=refined_decisions,
    )
    return gates, metrics, evidence


def run_gate0_convergence(
    config: Gate0Config,
    lock: Gate0DevelopmentLock,
    primary_report: Gate0Report,
    primary_report_digest: str,
    factory_builder: RefinementFactoryBuilder,
) -> Gate0ConvergenceRun:
    """Execute the preregistered one-seed temporal and spatial refinement checks.

    Arm convergence is relative to each primary arm mean. Effect convergence is
    the absolute change in relative TKE reduction ``(baseline-candidate)/baseline``.
    Ordering and the explicit materiality/opposite-pair/effort decision vector
    must also be identical at each refinement.
    """
    _sha256_identifier(primary_report_digest, "primary_report_digest")
    if lock.protocol_fingerprint != config.fingerprint:
        raise ValueError("development lock does not match the preregistered convergence protocol")
    if primary_report.protocol_fingerprint != config.fingerprint:
        raise ValueError("primary report does not match the preregistered convergence protocol")
    if primary_report.development_lock_digest != lock.digest:
        raise ValueError("primary report does not match the development lock")
    if not primary_report.primary_passed:
        raise ValueError("primary report must pass before convergence execution")
    target_case_ids = {
        case.case_id for case in config.cases("heldout") if case.seed == config.convergence_seed
    }
    primary_by_arm = _group_convergence_traces(primary_report.traces, target_case_ids)

    specifications = (
        (
            "temporal",
            config.grid_size,
            config.temporal_refinement_dt,
            config.maximum_temporal_arm_tke_relative_difference,
            config.maximum_temporal_effect_difference,
        ),
        (
            "spatial",
            config.spatial_refinement_grid_size,
            config.dt,
            config.maximum_spatial_arm_tke_relative_difference,
            config.maximum_spatial_effect_difference,
        ),
    )
    all_gates: dict[str, bool] = {}
    all_metrics: dict[str, float] = {}
    provenance: dict[str, Gate0RefinementProvenance] = {}
    evidence_by_label: dict[str, Gate0RefinementEvidence] = {}
    for label, grid_size, dt, arm_limit, effect_limit in specifications:
        factory = factory_builder(grid_size, dt)
        effective_grid_size = getattr(factory, "execution_grid_size", None)
        effective_dt = getattr(factory, "execution_dt", None)
        if effective_grid_size is None or tuple(effective_grid_size) != grid_size:
            raise ValueError(f"{label} refinement factory did not declare the requested grid_size")
        if effective_dt is None or not math.isclose(float(effective_dt), dt, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{label} refinement factory did not declare the requested dt")
        refined, executed, refinement_provenance = _run_refinement_traces(
            config,
            lock,
            factory,
            label=label,
            grid_size=grid_size,
            dt=dt,
        )
        gates, metrics, evidence = _evaluate_refinement(
            config,
            primary_by_arm,
            refined,
            executed,
            refinement_provenance,
            label=label,
            maximum_arm_tke_relative_difference=arm_limit,
            maximum_effect_difference=effect_limit,
        )
        all_gates.update(gates)
        all_metrics.update(metrics)
        provenance[label] = refinement_provenance
        evidence_by_label[label] = evidence

    attestation = Gate0ConvergenceAttestation(
        protocol_fingerprint=config.fingerprint,
        development_lock_digest=lock.digest,
        primary_report_digest=primary_report_digest,
        convergence_seed=config.convergence_seed,
        temporal_provenance=provenance["temporal"],
        spatial_provenance=provenance["spatial"],
        refinement_evidence_digests={
            label: evidence.digest for label, evidence in evidence_by_label.items()
        },
        gates=all_gates,
        metrics=all_metrics,
    )
    return Gate0ConvergenceRun(
        attestation=attestation,
        temporal_evidence=evidence_by_label["temporal"],
        spatial_evidence=evidence_by_label["spatial"],
    )


class _HydroGymRuntime:
    """One compiled environment per phase, shared by fresh deterministic episodes."""

    def __init__(
        self,
        config: Gate0Config,
        phase_radians: float,
        *,
        grid_size: tuple[int, int],
        dt: float,
    ):
        import jax
        import jax.numpy as jnp

        from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

        self.jax = jax
        self.jnp = jnp
        self.env = KolmogorovFlow(
            env_config={
                "dt": dt,
                "action_time": config.action_time,
                "save_time": config.save_time,
                "initial_perturbation_amplitude": config.initial_perturbation_amplitude,
                "max_episode_steps": config.uncontrolled_burn_in_intervals
                + config.controller_warmup_intervals
                + config.scored_intervals
                + 2,
                "observation_mode": "signed_forced_mode",
            },
            flow_config={
                "Re": config.reynolds_number,
                "k": config.forcing_wavenumber,
                "forcing_phase": phase_radians,
                "grid_size": grid_size,
                "obs_size": 4,
            },
        )
        if self.env.observation_metadata()["version"] != config.observation_contract_version:
            raise RuntimeError("HydroGym observation contract does not match Gate 0")
        self.params = self.env.default_params
        self.reset = jax.jit(self.env.reset_env)
        self.step = jax.jit(self.env.step_env)


class _HydroGymEpisode:
    def __init__(
        self,
        config: Gate0Config,
        case: Gate0Case,
        runtime: _HydroGymRuntime,
        *,
        execution_dt: float,
    ):
        self._jax = runtime.jax
        self._jnp = runtime.jnp
        self._config = config
        self._execution_dt = execution_dt
        self._case = case
        # ``reset_env`` advances one uncontrolled action interval.  The generic
        # runner subtracts this from the preregistered total D horizon.
        self.uncontrolled_reset_prelude_intervals = 1
        self._env = runtime.env
        self._params = runtime.params
        self._reset = runtime.reset
        self._step = runtime.step
        jax = self._jax
        self._key = jax.random.PRNGKey(case.seed)
        observation, self._state = self._reset(self._key, self._params)
        self.observation = tuple(float(value) for value in observation)
        self.state_digest = self._state_digest()
        self._tkes: list[float] = []
        self._all_frames_finite = True
        self._maximum_zero_mode_ratio = 0.0
        self._maximum_divergence_ratio = 0.0
        self._maximum_retained_tail_fraction = 0.0
        self._maximum_cfl = 0.0
        self._maximum_reward_identity_relative_error = 0.0
        self._update_saved_frame_diagnostics()

    def _state_digest(self) -> str:
        import numpy as np

        value = np.asarray(self._jax.device_get(self._state.omega_hat))
        hasher = hashlib.sha256()
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(_canonical(list(value.shape)).encode("ascii"))
        hasher.update(value.tobytes(order="C"))
        return hasher.hexdigest()

    def clone_after_development(self, total_uncontrolled_intervals: int) -> "_HydroGymEpisode":
        """Fork an immutable JAX state so every arm starts from byte-identical development."""
        cloned = copy.copy(self)
        cloned._tkes = list(self._tkes)
        cloned.uncontrolled_reset_prelude_intervals = total_uncontrolled_intervals
        return cloned

    def advance(self, action: Action) -> EpisodeStep:
        step_index = len(self._tkes) + 1
        key = self._jax.random.fold_in(self._key, step_index)
        observation, self._state, reward, _, info = self._step(
            key,
            self._state,
            self._jnp.asarray(action),
            self._params,
        )
        self.observation = tuple(float(value) for value in observation)
        mean_tke = float(info["mean_tke"])
        reward_value = float(reward)
        reward_total = float(info["reward_total"])
        reward_tke = float(info["reward_tke"])
        reward_action = float(info["reward_action_l1"])
        reward_residual = max(
            abs(reward_value - reward_total),
            abs(reward_total - reward_tke - reward_action),
            abs(reward_tke + float(self._params.reward_alpha) * mean_tke),
        )
        reward_scale = max(1.0, abs(reward_value), abs(reward_total), abs(mean_tke))
        self._maximum_reward_identity_relative_error = max(
            self._maximum_reward_identity_relative_error,
            reward_residual / reward_scale,
        )
        self._all_frames_finite = self._all_frames_finite and all(
            math.isfinite(value)
            for value in (mean_tke, reward_value, reward_total, reward_tke, reward_action)
        )
        self._tkes.append(mean_tke)
        self.state_digest = self._state_digest()
        self._update_saved_frame_diagnostics()
        return EpisodeStep(self.observation, mean_tke, self.state_digest)

    def _update_saved_frame_diagnostics(self) -> None:
        import numpy as np

        omega_hat = np.asarray(self._jax.device_get(self._state.trajectory))
        nx, rfft_ny = omega_hat.shape[-2:]
        ny = 2 * (rfft_ny - 1)
        self._all_frames_finite = self._all_frames_finite and bool(np.isfinite(omega_hat).all())

        kx_modes = np.fft.fftfreq(nx) * nx
        ky_modes = np.fft.rfftfreq(ny) * ny
        kx, ky = np.meshgrid(kx_modes, ky_modes, indexing="ij")
        laplacian = -(kx**2 + ky**2)
        safe_laplacian = laplacian.copy()
        safe_laplacian[0, 0] = 1.0
        psi_hat = -omega_hat / safe_laplacian
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        divergence_hat = 1j * (kx * u_hat + ky * v_hat)

        spectral_scale = np.sqrt(np.sum(np.abs(omega_hat) ** 2, axis=(-2, -1)))
        zero_ratio = np.divide(
            np.abs(omega_hat[..., 0, 0]),
            spectral_scale,
            out=np.zeros_like(spectral_scale, dtype=np.float64),
            where=spectral_scale > 0.0,
        )
        velocity_gradient_scale = np.sqrt(
            np.sum(np.abs(kx * u_hat) ** 2 + np.abs(ky * v_hat) ** 2, axis=(-2, -1))
        )
        divergence_scale = np.sqrt(np.sum(np.abs(divergence_hat) ** 2, axis=(-2, -1)))
        divergence_ratio = np.divide(
            divergence_scale,
            velocity_gradient_scale,
            out=np.zeros_like(divergence_scale, dtype=np.float64),
            where=velocity_gradient_scale > 0.0,
        )

        retained = (np.abs(kx) <= nx // 3) & (ky <= ny // 3)
        tail_kx = math.ceil(0.8 * (nx // 3))
        tail_ky = math.ceil(0.8 * (ny // 3))
        retained_tail = retained & ((np.abs(kx) >= tail_kx) | (ky >= tail_ky))
        rfft_weights = np.ones(rfft_ny, dtype=np.float64)
        if rfft_ny > 2:
            rfft_weights[1:-1] = 2.0
        weighted_power = np.abs(omega_hat) ** 2 * rfft_weights
        retained_power = np.sum(weighted_power[..., retained], axis=-1)
        tail_power = np.sum(weighted_power[..., retained_tail], axis=-1)
        tail_fraction = np.divide(
            tail_power,
            retained_power,
            out=np.zeros_like(tail_power, dtype=np.float64),
            where=retained_power > 0.0,
        )

        u = np.fft.irfftn(u_hat, s=(nx, ny), axes=(-2, -1))
        v = np.fft.irfftn(v_hat, s=(nx, ny), axes=(-2, -1))
        dx = 2.0 * np.pi / nx
        dy = 2.0 * np.pi / ny
        cfl = self._execution_dt * (
            np.max(np.abs(u), axis=(-2, -1)) / dx + np.max(np.abs(v), axis=(-2, -1)) / dy
        )
        self._maximum_zero_mode_ratio = max(self._maximum_zero_mode_ratio, float(np.max(zero_ratio)))
        self._maximum_divergence_ratio = max(self._maximum_divergence_ratio, float(np.max(divergence_ratio)))
        self._maximum_retained_tail_fraction = max(
            self._maximum_retained_tail_fraction,
            float(np.max(tail_fraction)),
        )
        self._maximum_cfl = max(self._maximum_cfl, float(np.max(cfl)))

    def numerical_gates(self) -> Mapping[str, bool]:
        import numpy as np

        finite = bool(self._all_frames_finite and np.isfinite(self._tkes).all())
        return {
            "finite_state_and_metrics": finite,
            "nonnegative_tke": bool(self._tkes and min(self._tkes) >= -1.0e-7),
            "reward_tke_identity": self._maximum_reward_identity_relative_error
            <= self._config.maximum_reward_identity_relative_error,
            "zero_mean_vorticity": self._maximum_zero_mode_ratio <= self._config.maximum_zero_mode_ratio,
            "incompressible_velocity": self._maximum_divergence_ratio
            <= self._config.maximum_divergence_ratio,
            "spectral_tail_controlled": self._maximum_retained_tail_fraction
            <= self._config.maximum_spectral_tail_fraction,
            "cfl_controlled": self._maximum_cfl <= self._config.maximum_cfl,
        }


class HydroGymEpisodeFactory:
    """Actual signed-get_obs HydroGym backend; no PPO, GPU, MLflow, or model calls."""

    def __init__(
        self,
        config: Gate0Config,
        *,
        execution_grid_size: tuple[int, int] | None = None,
        execution_dt: float | None = None,
    ):
        import jax

        if config.precision == "float64" and not jax.config.x64_enabled:
            raise RuntimeError("Gate 0 precision='float64' requires JAX_ENABLE_X64=1 before process start")
        self.config = config
        self.execution_grid_size = config.grid_size if execution_grid_size is None else execution_grid_size
        self.execution_dt = config.dt if execution_dt is None else _finite(execution_dt, "execution_dt")
        if (
            len(self.execution_grid_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.execution_grid_size)
            or self.execution_grid_size[0] != self.execution_grid_size[1]
        ):
            raise ValueError("execution_grid_size must be a positive square grid")
        if self.execution_dt <= 0.0:
            raise ValueError("execution_dt must be positive")
        _integer_ratio(config.save_time, self.execution_dt, "save_time/execution_dt")
        self._runtimes: dict[float, _HydroGymRuntime] = {}
        self._developed: dict[tuple[str, int], _HydroGymEpisode] = {}

    def __call__(self, case: Gate0Case) -> Gate0Episode:
        runtime = self._runtimes.get(case.phase_turns)
        if runtime is None:
            runtime = _HydroGymRuntime(
                self.config,
                case.phase_radians,
                grid_size=self.execution_grid_size,
                dt=self.execution_dt,
            )
            self._runtimes[case.phase_turns] = runtime
        return _HydroGymEpisode(
            self.config,
            case,
            runtime,
            execution_dt=self.execution_dt,
        )

    def developed_episode(self, case: Gate0Case, total_intervals: int) -> Gate0Episode:
        """Cache the expensive zero-action development once, then branch each arm safely."""
        key = (case.case_id, total_intervals)
        developed = self._developed.get(key)
        if developed is None:
            developed = self(case)
            remaining = total_intervals - developed.uncontrolled_reset_prelude_intervals
            if remaining < 0:
                raise ValueError("development horizon is shorter than the environment reset prelude")
            for _ in range(remaining):
                developed.advance(ZERO_ACTION)
            self._developed[key] = developed
        return developed.clone_after_development(total_intervals)


__all__ = [
    "CLAIM_BOUNDARY",
    "EpisodeStep",
    "FrozenSignedController",
    "Gate0Case",
    "Gate0ConvergenceAttestation",
    "Gate0ConvergenceRun",
    "Gate0Config",
    "Gate0DevelopmentLock",
    "Gate0DevelopmentSearchError",
    "Gate0RefinementProvenance",
    "Gate0RefinementEvidence",
    "Gate0RefinementTraceSummary",
    "Gate0Report",
    "GATE0_CONVERGENCE_SCHEMA_VERSION",
    "HydroGymEpisodeFactory",
    "REQUIRED_CONVERGENCE_GATES",
    "REQUIRED_PRIMARY_GATES",
    "lock_development_controls",
    "oracle_action",
    "run_gate0",
    "run_gate0_convergence",
]
