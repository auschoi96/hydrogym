"""Strict data contracts for the codex_hydrogym learning outer loop.

Language models may propose bounded scalar settings and explain a hypothesis. They
never emit executable reward code, mutate the CFD solver, or bypass physics gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from codex_hydrogym import PROJECT_LABEL, REWARD_FORMULA_VERSION


REWARD_CANDIDATE_SCHEMA_VERSION = "codex_hydrogym.reward_candidate.v1"
_CANDIDATE_ID = re.compile(r"^codex_hydrogym_[a-z0-9][a-z0-9_-]{2,47}$")
_REQUIRED_CANDIDATE_FIELDS = {
    "schema_version",
    "candidate_id",
    "reward_alpha",
    "learning_rate",
    "entropy_coefficient",
    "gamma",
    "gae_lambda",
    "num_updates",
    "hypothesis",
    "rationale",
}

REWARD_CANDIDATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_REQUIRED_CANDIDATE_FIELDS),
    "properties": {
        "schema_version": {"type": "string", "const": REWARD_CANDIDATE_SCHEMA_VERSION},
        "candidate_id": {"type": "string", "pattern": _CANDIDATE_ID.pattern},
        "reward_alpha": {"type": "number", "minimum": 0.1, "maximum": 10.0},
        "learning_rate": {"type": "number", "minimum": 1.0e-6, "maximum": 1.0e-2},
        "entropy_coefficient": {"type": "number", "minimum": 0.0, "maximum": 0.1},
        "gamma": {"type": "number", "minimum": 0.9, "maximum": 1.0},
        "gae_lambda": {"type": "number", "minimum": 0.9, "maximum": 1.0},
        "num_updates": {"type": "integer", "minimum": 1, "maximum": 10_000},
        "hypothesis": {"type": "string", "minLength": 1, "maxLength": 1_000},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 2_000},
    },
}


def _bounded_number(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]") from error
    if not math.isfinite(converted) or not minimum <= converted <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return converted


def _bounded_text(value: Any, *, name: str, maximum_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > maximum_length:
        raise ValueError(f"{name} must contain 1 to {maximum_length} characters")
    return normalized


@dataclass(frozen=True)
class RewardCandidate:
    """A bounded, directly executable PPO/reward hypothesis.

    ``reward_alpha`` is the positive TKE penalty already implemented by the
    environment. The other fields tune the PPO optimizer and rollout budget.
    Arbitrary Python, formulas, model-generated source, and solver settings are
    deliberately absent from this contract.
    """

    candidate_id: str
    reward_alpha: float
    learning_rate: float
    entropy_coefficient: float
    gamma: float
    gae_lambda: float
    num_updates: int
    hypothesis: str
    rationale: str
    schema_version: str = REWARD_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REWARD_CANDIDATE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REWARD_CANDIDATE_SCHEMA_VERSION}")
        if not isinstance(self.candidate_id, str) or _CANDIDATE_ID.fullmatch(self.candidate_id) is None:
            raise ValueError("candidate_id must start with codex_hydrogym_ and contain only lowercase slug characters")
        object.__setattr__(
            self,
            "reward_alpha",
            _bounded_number(self.reward_alpha, name="reward_alpha", minimum=0.1, maximum=10.0),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _bounded_number(self.learning_rate, name="learning_rate", minimum=1.0e-6, maximum=1.0e-2),
        )
        object.__setattr__(
            self,
            "entropy_coefficient",
            _bounded_number(
                self.entropy_coefficient,
                name="entropy_coefficient",
                minimum=0.0,
                maximum=0.1,
            ),
        )
        object.__setattr__(self, "gamma", _bounded_number(self.gamma, name="gamma", minimum=0.9, maximum=1.0))
        object.__setattr__(
            self,
            "gae_lambda",
            _bounded_number(self.gae_lambda, name="gae_lambda", minimum=0.9, maximum=1.0),
        )
        if isinstance(self.num_updates, bool) or not isinstance(self.num_updates, int):
            raise TypeError("num_updates must be an integer")
        if not 1 <= self.num_updates <= 10_000:
            raise ValueError("num_updates must be in [1, 10000]")
        object.__setattr__(
            self,
            "hypothesis",
            _bounded_text(self.hypothesis, name="hypothesis", maximum_length=1_000),
        )
        object.__setattr__(
            self,
            "rationale",
            _bounded_text(self.rationale, name="rationale", maximum_length=2_000),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply(self, baseline_config):
        """Return a validated config; the input object is never mutated."""
        return replace(
            baseline_config,
            run_name=self.candidate_id,
            reward_alpha=self.reward_alpha,
            learning_rate=self.learning_rate,
            entropy_coefficient=self.entropy_coefficient,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            total_timesteps=self.num_updates * baseline_config.total_batch_size,
        )


def parse_reward_candidate(payload: str | Mapping[str, Any]) -> RewardCandidate:
    """Parse strict JSON or a mapping and reject unknown/missing fields."""
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("reward candidate must be a single JSON object without Markdown fences") from error
    elif isinstance(payload, Mapping):
        raw = dict(payload)
    else:
        raise TypeError("reward candidate must be JSON text or a mapping")
    if not isinstance(raw, dict):
        raise ValueError("reward candidate must be a JSON object")

    fields = set(raw)
    if fields != _REQUIRED_CANDIDATE_FIELDS:
        missing = sorted(_REQUIRED_CANDIDATE_FIELDS - fields)
        unknown = sorted(fields - _REQUIRED_CANDIDATE_FIELDS)
        raise ValueError(f"reward candidate fields do not match the contract; missing={missing}, unknown={unknown}")
    return RewardCandidate(**raw)


@dataclass(frozen=True)
class RolloutEvidence:
    """Real held-out CFD/PPO evidence referenced by an MLflow run and artifact."""

    run_id: str
    context_fingerprint: str
    frozen_training_fingerprint: str
    heldout_evidence_digest: str
    mean_tke: float
    control_l1: float
    reward_total: float
    physics_gates_passed: bool
    artifact_uri: str

    def __post_init__(self) -> None:
        for field_name in ("run_id", "artifact_uri"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
        for field_name in (
            "context_fingerprint",
            "frozen_training_fingerprint",
            "heldout_evidence_digest",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        for field_name in ("mean_tke", "control_l1", "reward_total"):
            value = _bounded_number(
                getattr(self, field_name),
                name=field_name,
                minimum=-1.0e12,
                maximum=1.0e12,
            )
            object.__setattr__(self, field_name, value)
        if self.mean_tke < -1.0e-7 or self.control_l1 < -1.0e-7:
            raise ValueError("mean_tke and control_l1 must be nonnegative within numerical tolerance")
        if not isinstance(self.physics_gates_passed, bool):
            raise TypeError("physics_gates_passed must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_candidate_evaluation_record(
    *,
    candidate: RewardCandidate,
    baseline: RolloutEvidence,
    candidate_rollout: RolloutEvidence,
) -> dict[str, dict[str, Any]]:
    """Build one MLflow evaluation record from comparable, real rollouts."""
    if baseline.context_fingerprint != candidate_rollout.context_fingerprint:
        raise ValueError("baseline and candidate rollouts must share a held-out context fingerprint")
    if baseline.frozen_training_fingerprint != candidate_rollout.frozen_training_fingerprint:
        raise ValueError("baseline and candidate rollouts must share a frozen-training fingerprint")

    return {
        "inputs": {
            "project": PROJECT_LABEL,
            "candidate": candidate.as_dict(),
            "baseline": baseline.as_dict(),
        },
        "outputs": {
            "candidate_rollout": candidate_rollout.as_dict(),
            "delta_mean_tke": candidate_rollout.mean_tke - baseline.mean_tke,
            "delta_control_l1": candidate_rollout.control_l1 - baseline.control_l1,
            "delta_reward_total": candidate_rollout.reward_total - baseline.reward_total,
        },
        "expectations": {
            "expected_response": (
                "Rate fluid reward plausibility from 1 to 5. A promotable candidate must use real comparable "
                "rollouts, pass every physics gate, reduce held-out mean TKE, and avoid disproportionate "
                "control effort."
            )
        },
    }


def build_gepa_record(*, scenario: Mapping[str, Any], expected_behavior: str) -> dict[str, dict[str, Any]]:
    """Build the inputs+expectations shape required by MLflow GEPA."""
    if not scenario:
        raise ValueError("scenario must not be empty")
    expected = _bounded_text(expected_behavior, name="expected_behavior", maximum_length=4_000)
    return {
        "inputs": {"scenario": dict(scenario)},
        "expectations": {"expected_response": expected},
    }


# The SDK-harness contract is intentionally separate from RewardCandidate.  The
# older GEPA path can change the training budget through ``num_updates``; the
# paired Codex/Claude proof must keep compute and the held-out context fixed.
RUN_BUNDLE_SCHEMA_VERSION = "codex_hydrogym.run_bundle.v1"
LEGACY_REWARD_SPEC_SCHEMA_VERSION = "codex_hydrogym.reward_spec.v1"
REWARD_SPEC_SCHEMA_VERSION = "codex_hydrogym.reward_spec.v2"
AGENT_FEEDBACK_SCHEMA_VERSION = "codex_hydrogym.agent_feedback.v2"

_HARNESS_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FEEDBACK_DECISIONS = {"stop", "collect_evidence", "run_bounded_trial"}
_FEEDBACK_COSTS = {"none", "cpu_gate", "single_gpu_short"}
_EVIDENCE_KINDS = {"measured", "synthetic_contract", "unverified_diagnostic"}
REQUIRED_PHYSICS_GATES = frozenset(
    {
        "finite_state_and_metrics",
        "reward_decomposition_identity",
        "nonnegative_tke",
        "bounded_control_effort",
        "zero_mean_vorticity",
        "incompressible_velocity",
        "spectral_tail_controlled",
        "cfl_controlled",
        "update_count_valid",
    }
)
MAX_COMPARATOR_ARMS = 12
MAX_RUN_BUNDLE_BYTES = 250_000


def _harness_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HARNESS_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 3-128 character lowercase identifier")
    return value


def _harness_json_mapping(
    value: Any,
    *,
    name: str,
    allow_empty: bool = False,
    maximum_bytes: int = 50_000,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied = dict(value)
    if not allow_empty and not copied:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(key, str) or not key for key in copied):
        raise ValueError(f"{name} keys must be non-empty strings")
    try:
        encoded = json.dumps(copied, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} exceeds the {maximum_bytes}-byte contract limit")
    # Round-tripping severs references to caller-owned containers.  Recursively
    # freezing the result also keeps the evidence digest stable after validation.
    return _freeze_json_value(json.loads(encoded))


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _harness_metric_mapping(value: Any, *, name: str, allow_empty: bool = True) -> Mapping[str, float]:
    raw = _harness_json_mapping(value, name=name, allow_empty=allow_empty, maximum_bytes=20_000)
    metrics: dict[str, float] = {}
    for key, item in raw.items():
        metrics[key] = _bounded_number(item, name=f"{name}.{key}", minimum=-1.0e12, maximum=1.0e12)
    return MappingProxyType(metrics)


def _harness_gate_mapping(value: Any, *, name: str) -> Mapping[str, bool]:
    raw = _harness_json_mapping(value, name=name, maximum_bytes=10_000)
    if any(not isinstance(item, bool) for item in raw.values()):
        raise TypeError(f"{name} values must be boolean")
    return MappingProxyType(dict(raw))


def _harness_text_sequence(
    value: Any,
    *,
    name: str,
    maximum_items: int,
    maximum_length: int,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of strings")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > maximum_items:
        raise ValueError(f"{name} must contain at most {maximum_items} items")
    return tuple(
        _bounded_text(item, name=f"{name}[{index}]", maximum_length=maximum_length)
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class EvidenceArm:
    """One deterministic evaluation arm inside a canonical run bundle."""

    arm_id: str
    run_id: str
    evidence_kind: str
    artifact_ref: str
    artifact_sha256: str
    context_fingerprint: str
    controller_kind: str
    uses_observations: bool
    mean_tke: float
    control_effort: float
    physics_gates: Mapping[str, bool]
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _harness_id(self.arm_id, name="arm_id"))
        object.__setattr__(self, "run_id", _bounded_text(self.run_id, name="run_id", maximum_length=256))
        if self.evidence_kind not in _EVIDENCE_KINDS:
            raise ValueError(f"evidence_kind must be one of {sorted(_EVIDENCE_KINDS)}")
        object.__setattr__(
            self,
            "artifact_ref",
            _bounded_text(self.artifact_ref, name="artifact_ref", maximum_length=1_000),
        )
        if not isinstance(self.artifact_sha256, str) or _SHA256.fullmatch(self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        if self.evidence_kind == "synthetic_contract" and not self.artifact_ref.startswith("synthetic://"):
            raise ValueError("synthetic_contract evidence must use a synthetic:// artifact_ref")
        if self.evidence_kind == "unverified_diagnostic" and not self.artifact_ref.startswith("diagnostic://"):
            raise ValueError("unverified_diagnostic evidence must use a diagnostic:// artifact_ref")
        if self.evidence_kind == "measured" and self.artifact_ref.startswith(("synthetic://", "diagnostic://")):
            raise ValueError("measured evidence cannot use a synthetic or diagnostic artifact_ref")
        object.__setattr__(
            self,
            "context_fingerprint",
            _bounded_text(self.context_fingerprint, name="context_fingerprint", maximum_length=256),
        )
        if _SHA256.fullmatch(self.context_fingerprint) is None:
            raise ValueError("context_fingerprint must be a lowercase SHA-256 digest")
        object.__setattr__(
            self,
            "controller_kind",
            _bounded_text(self.controller_kind, name="controller_kind", maximum_length=128),
        )
        if not isinstance(self.uses_observations, bool):
            raise TypeError("uses_observations must be boolean")
        mean_tke = _bounded_number(self.mean_tke, name="mean_tke", minimum=-1.0e-7, maximum=1.0e12)
        control_effort = _bounded_number(
            self.control_effort,
            name="control_effort",
            minimum=-1.0e-7,
            maximum=1.0e12,
        )
        object.__setattr__(self, "mean_tke", max(0.0, mean_tke))
        object.__setattr__(self, "control_effort", max(0.0, control_effort))
        object.__setattr__(
            self,
            "physics_gates",
            _harness_gate_mapping(self.physics_gates, name="physics_gates"),
        )
        object.__setattr__(self, "metrics", _harness_metric_mapping(self.metrics, name="metrics"))

    @property
    def all_gates_passed(self) -> bool:
        return REQUIRED_PHYSICS_GATES.issubset(self.physics_gates) and all(self.physics_gates.values())

    @property
    def missing_physics_gates(self) -> tuple[str, ...]:
        return tuple(sorted(REQUIRED_PHYSICS_GATES - self.physics_gates.keys()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "run_id": self.run_id,
            "evidence_kind": self.evidence_kind,
            "artifact_ref": self.artifact_ref,
            "artifact_sha256": self.artifact_sha256,
            "context_fingerprint": self.context_fingerprint,
            "controller_kind": self.controller_kind,
            "uses_observations": self.uses_observations,
            "mean_tke": self.mean_tke,
            "control_effort": self.control_effort,
            "physics_gates": dict(self.physics_gates),
            "metrics": dict(self.metrics),
        }


_EVIDENCE_ARM_FIELDS = {
    "arm_id",
    "run_id",
    "evidence_kind",
    "artifact_ref",
    "artifact_sha256",
    "context_fingerprint",
    "controller_kind",
    "uses_observations",
    "mean_tke",
    "control_effort",
    "physics_gates",
    "metrics",
}


def _parse_evidence_arm(value: Any, *, name: str) -> EvidenceArm:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    raw = dict(value)
    if set(raw) != _EVIDENCE_ARM_FIELDS:
        raise ValueError(f"{name} fields do not match the evidence-arm contract")
    return EvidenceArm(**raw)


@dataclass(frozen=True)
class RunBundle:
    """The only evidence exposed to either coding-agent harness.

    ``group_id`` is the leakage boundary: both Codex and Claude outputs for a
    bundle must remain in the same train/test fold.  Comparators with a different
    context remain visible for diagnosis but are never treated as promotable
    evidence by deterministic gates.
    """

    bundle_id: str
    group_id: str
    task_contract_version: str
    task: Mapping[str, Any]
    training: Mapping[str, Any]
    candidate: EvidenceArm
    comparators: tuple[EvidenceArm, ...]
    diagnostics: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    schema_version: str = RUN_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {RUN_BUNDLE_SCHEMA_VERSION}")
        object.__setattr__(self, "bundle_id", _harness_id(self.bundle_id, name="bundle_id"))
        object.__setattr__(self, "group_id", _harness_id(self.group_id, name="group_id"))
        object.__setattr__(
            self,
            "task_contract_version",
            _harness_id(self.task_contract_version, name="task_contract_version"),
        )
        object.__setattr__(self, "task", _harness_json_mapping(self.task, name="task"))
        object.__setattr__(self, "training", _harness_json_mapping(self.training, name="training"))
        if not isinstance(self.candidate, EvidenceArm):
            raise TypeError("candidate must be EvidenceArm")
        if not isinstance(self.comparators, tuple) or any(
            not isinstance(arm, EvidenceArm) for arm in self.comparators
        ):
            raise TypeError("comparators must be a tuple of EvidenceArm values")
        if len(self.comparators) > MAX_COMPARATOR_ARMS:
            raise ValueError(f"comparators must contain at most {MAX_COMPARATOR_ARMS} arms")
        arm_ids = [self.candidate.arm_id, *(arm.arm_id for arm in self.comparators)]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("candidate and comparator arm_id values must be unique")
        object.__setattr__(
            self,
            "diagnostics",
            _harness_text_sequence(
                self.diagnostics,
                name="diagnostics",
                maximum_items=20,
                maximum_length=2_000,
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "artifact_refs",
            _harness_text_sequence(
                self.artifact_refs,
                name="artifact_refs",
                maximum_items=20,
                maximum_length=1_000,
                allow_empty=True,
            ),
        )
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_RUN_BUNDLE_BYTES:
            raise ValueError(f"run bundle exceeds the {MAX_RUN_BUNDLE_BYTES}-byte contract limit")

    def comparison_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.comparators:
            issues.append("no comparator arms are present")
        if not self.candidate.uses_observations:
            issues.append(f"candidate {self.candidate.arm_id} does not use observations")
        if self.candidate.evidence_kind != "measured":
            issues.append(
                f"candidate {self.candidate.arm_id} evidence is {self.candidate.evidence_kind}, not measured"
            )
        if self.candidate.missing_physics_gates:
            issues.append(
                f"candidate {self.candidate.arm_id} is missing physics gates "
                f"{list(self.candidate.missing_physics_gates)}"
            )
        if not self.candidate.all_gates_passed:
            issues.append(f"candidate {self.candidate.arm_id} failed at least one physics gate")
        if not any(not arm.uses_observations for arm in self.comparators):
            issues.append("no observation-free comparator is present")
        for arm in self.comparators:
            if arm.evidence_kind != "measured":
                issues.append(f"comparator {arm.arm_id} evidence is {arm.evidence_kind}, not measured")
            if arm.missing_physics_gates:
                issues.append(f"comparator {arm.arm_id} is missing physics gates {list(arm.missing_physics_gates)}")
            if arm.context_fingerprint != self.candidate.context_fingerprint:
                issues.append(f"comparator {arm.arm_id} has a different context fingerprint")
            if not arm.all_gates_passed:
                issues.append(f"comparator {arm.arm_id} failed at least one physics gate")
            dominates_candidate = (
                not arm.uses_observations
                and arm.context_fingerprint == self.candidate.context_fingerprint
                and arm.mean_tke <= self.candidate.mean_tke
                and arm.control_effort <= self.candidate.control_effort
                and (arm.mean_tke < self.candidate.mean_tke or arm.control_effort < self.candidate.control_effort)
            )
            if dominates_candidate:
                issues.append(
                    f"observation-free comparator {arm.arm_id} Pareto-dominates candidate {self.candidate.arm_id}"
                )
        return tuple(issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "group_id": self.group_id,
            "task_contract_version": self.task_contract_version,
            "task": _thaw_json_value(self.task),
            "training": _thaw_json_value(self.training),
            "candidate": self.candidate.as_dict(),
            "comparators": [arm.as_dict() for arm in self.comparators],
            "diagnostics": list(self.diagnostics),
            "artifact_refs": list(self.artifact_refs),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def evidence_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


_RUN_BUNDLE_FIELDS = {
    "schema_version",
    "bundle_id",
    "group_id",
    "task_contract_version",
    "task",
    "training",
    "candidate",
    "comparators",
    "diagnostics",
    "artifact_refs",
}


def parse_run_bundle(payload: str | Mapping[str, Any]) -> RunBundle:
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("run bundle must be one JSON object") from error
    elif isinstance(payload, Mapping):
        raw = dict(payload)
    else:
        raise TypeError("run bundle must be JSON text or a mapping")
    if not isinstance(raw, dict) or set(raw) != _RUN_BUNDLE_FIELDS:
        raise ValueError("run bundle fields do not match the canonical contract")
    comparators = raw["comparators"]
    if not isinstance(comparators, list):
        raise TypeError("comparators must be a list")
    if not isinstance(raw["diagnostics"], list):
        raise TypeError("diagnostics must be a list")
    if not isinstance(raw["artifact_refs"], list):
        raise TypeError("artifact_refs must be a list")
    return RunBundle(
        schema_version=raw["schema_version"],
        bundle_id=raw["bundle_id"],
        group_id=raw["group_id"],
        task_contract_version=raw["task_contract_version"],
        task=raw["task"],
        training=raw["training"],
        candidate=_parse_evidence_arm(raw["candidate"], name="candidate"),
        comparators=tuple(
            _parse_evidence_arm(arm, name=f"comparators[{index}]") for index, arm in enumerate(comparators)
        ),
        diagnostics=tuple(raw["diagnostics"]),
        artifact_refs=tuple(raw["artifact_refs"]),
    )


@dataclass(frozen=True)
class LegacyRewardSpecV1:
    """Read-only parser target for archived, causally confounded v1 outputs."""

    reward_alpha: float
    learning_rate: float
    entropy_coefficient: float
    gamma: float
    gae_lambda: float
    schema_version: str = LEGACY_REWARD_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEGACY_REWARD_SPEC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {LEGACY_REWARD_SPEC_SCHEMA_VERSION}")
        object.__setattr__(
            self,
            "reward_alpha",
            _bounded_number(self.reward_alpha, name="reward_alpha", minimum=0.1, maximum=10.0),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _bounded_number(self.learning_rate, name="learning_rate", minimum=1.0e-6, maximum=1.0e-2),
        )
        object.__setattr__(
            self,
            "entropy_coefficient",
            _bounded_number(self.entropy_coefficient, name="entropy_coefficient", minimum=0.0, maximum=0.1),
        )
        object.__setattr__(self, "gamma", _bounded_number(self.gamma, name="gamma", minimum=0.9, maximum=1.0))
        object.__setattr__(
            self,
            "gae_lambda",
            _bounded_number(self.gae_lambda, name="gae_lambda", minimum=0.9, maximum=1.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_LEGACY_REWARD_SPEC_V1_FIELDS = {
    "schema_version",
    "reward_alpha",
    "learning_rate",
    "entropy_coefficient",
    "gamma",
    "gae_lambda",
}


def parse_legacy_reward_spec_v1(value: Any) -> LegacyRewardSpecV1:
    if not isinstance(value, Mapping) or set(value) != _LEGACY_REWARD_SPEC_V1_FIELDS:
        raise ValueError("legacy reward_spec fields do not match the v1 contract")
    return LegacyRewardSpecV1(**dict(value))


@dataclass(frozen=True)
class RewardSpec:
    """A reward-only proposal; optimizer settings and compute are deliberately absent."""

    evidence_digest: str
    control_l1_weight: float
    action_delta_l2_weight: float
    formula_version: str = REWARD_FORMULA_VERSION
    schema_version: str = REWARD_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REWARD_SPEC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REWARD_SPEC_SCHEMA_VERSION}")
        if self.formula_version != REWARD_FORMULA_VERSION:
            raise ValueError(f"formula_version must be {REWARD_FORMULA_VERSION}")
        if not isinstance(self.evidence_digest, str) or _SHA256.fullmatch(self.evidence_digest) is None:
            raise ValueError("evidence_digest must be a lowercase SHA-256 digest")
        object.__setattr__(
            self,
            "control_l1_weight",
            _bounded_number(self.control_l1_weight, name="control_l1_weight", minimum=0.05, maximum=1.0),
        )
        object.__setattr__(
            self,
            "action_delta_l2_weight",
            _bounded_number(
                self.action_delta_l2_weight,
                name="action_delta_l2_weight",
                minimum=0.0,
                maximum=0.25,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def canonical_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


_REWARD_SPEC_FIELDS = {
    "schema_version",
    "formula_version",
    "evidence_digest",
    "control_l1_weight",
    "action_delta_l2_weight",
}


def parse_reward_spec(value: Any) -> RewardSpec:
    if not isinstance(value, Mapping) or set(value) != _REWARD_SPEC_FIELDS:
        raise ValueError("reward_spec fields do not match the reward-only v2 contract")
    return RewardSpec(**dict(value))


REWARD_SPEC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_REWARD_SPEC_FIELDS),
    "properties": {
        "schema_version": {"type": "string", "const": REWARD_SPEC_SCHEMA_VERSION},
        "formula_version": {"type": "string", "const": REWARD_FORMULA_VERSION},
        "evidence_digest": {"type": "string", "pattern": _SHA256.pattern},
        "control_l1_weight": {"type": "number", "minimum": 0.05, "maximum": 1.0},
        "action_delta_l2_weight": {"type": "number", "minimum": 0.0, "maximum": 0.25},
    },
}

REWARD_SPEC_TRANSPORT_SCHEMA: dict[str, Any] = {
    **REWARD_SPEC_JSON_SCHEMA,
    "properties": {
        **REWARD_SPEC_JSON_SCHEMA["properties"],
        "evidence_digest": {"type": "string"},
    },
}


@dataclass(frozen=True)
class AgentFeedback:
    """Strict shared output parsed identically for Codex and Claude."""

    feedback_id: str
    decision: str
    diagnosis: str
    evidence: tuple[str, ...]
    falsification_test: str
    claim_boundary: str
    estimated_cost: str
    reward_spec: RewardSpec | None
    schema_version: str = AGENT_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AGENT_FEEDBACK_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {AGENT_FEEDBACK_SCHEMA_VERSION}")
        object.__setattr__(self, "feedback_id", _harness_id(self.feedback_id, name="feedback_id"))
        if self.decision not in _FEEDBACK_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(_FEEDBACK_DECISIONS)}")
        if self.estimated_cost not in _FEEDBACK_COSTS:
            raise ValueError(f"estimated_cost must be one of {sorted(_FEEDBACK_COSTS)}")
        object.__setattr__(
            self,
            "diagnosis",
            _bounded_text(self.diagnosis, name="diagnosis", maximum_length=4_000),
        )
        object.__setattr__(
            self,
            "evidence",
            _harness_text_sequence(
                self.evidence,
                name="evidence",
                maximum_items=8,
                maximum_length=2_000,
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "falsification_test",
            _bounded_text(self.falsification_test, name="falsification_test", maximum_length=4_000),
        )
        object.__setattr__(
            self,
            "claim_boundary",
            _bounded_text(self.claim_boundary, name="claim_boundary", maximum_length=2_000),
        )
        if self.decision == "run_bounded_trial":
            if not isinstance(self.reward_spec, RewardSpec):
                raise ValueError("run_bounded_trial requires one bounded reward_spec")
            if self.estimated_cost == "none":
                raise ValueError("run_bounded_trial must declare a nonzero bounded cost")
        elif self.reward_spec is not None:
            raise ValueError("stop and collect_evidence decisions must set reward_spec to null")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feedback_id": self.feedback_id,
            "decision": self.decision,
            "diagnosis": self.diagnosis,
            "evidence": list(self.evidence),
            "falsification_test": self.falsification_test,
            "claim_boundary": self.claim_boundary,
            "estimated_cost": self.estimated_cost,
            "reward_spec": None if self.reward_spec is None else self.reward_spec.as_dict(),
        }


_AGENT_FEEDBACK_FIELDS = {
    "schema_version",
    "feedback_id",
    "decision",
    "diagnosis",
    "evidence",
    "falsification_test",
    "claim_boundary",
    "estimated_cost",
    "reward_spec",
}


def parse_agent_feedback(payload: str | Mapping[str, Any]) -> AgentFeedback:
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("agent feedback must be a single JSON object without Markdown fences") from error
    elif isinstance(payload, Mapping):
        raw = dict(payload)
    else:
        raise TypeError("agent feedback must be JSON text or a mapping")
    if not isinstance(raw, dict) or set(raw) != _AGENT_FEEDBACK_FIELDS:
        raise ValueError("agent feedback fields do not match the shared contract")
    if not isinstance(raw["evidence"], list):
        raise TypeError("evidence must be a list")
    reward_spec = None if raw["reward_spec"] is None else parse_reward_spec(raw["reward_spec"])
    return AgentFeedback(
        schema_version=raw["schema_version"],
        feedback_id=raw["feedback_id"],
        decision=raw["decision"],
        diagnosis=raw["diagnosis"],
        evidence=tuple(raw["evidence"]),
        falsification_test=raw["falsification_test"],
        claim_boundary=raw["claim_boundary"],
        estimated_cost=raw["estimated_cost"],
        reward_spec=reward_spec,
    )


AGENT_FEEDBACK_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_AGENT_FEEDBACK_FIELDS),
    "properties": {
        "schema_version": {"type": "string", "const": AGENT_FEEDBACK_SCHEMA_VERSION},
        "feedback_id": {"type": "string", "pattern": _HARNESS_ID.pattern},
        "decision": {"type": "string", "enum": sorted(_FEEDBACK_DECISIONS)},
        "diagnosis": {"type": "string", "minLength": 1, "maxLength": 4_000},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
        "falsification_test": {"type": "string", "minLength": 1, "maxLength": 4_000},
        "claim_boundary": {"type": "string", "minLength": 1, "maxLength": 2_000},
        "estimated_cost": {"type": "string", "enum": sorted(_FEEDBACK_COSTS)},
        "reward_spec": {"anyOf": [REWARD_SPEC_JSON_SCHEMA, {"type": "null"}]},
    },
    "allOf": [
        {
            "if": {"properties": {"decision": {"const": "run_bounded_trial"}}},
            "then": {"properties": {"reward_spec": REWARD_SPEC_JSON_SCHEMA}},
            "else": {"properties": {"reward_spec": {"type": "null"}}},
        }
    ],
}

# Databricks Foundation Model APIs deliberately support a smaller JSON Schema
# subset than the local evidence contract: no patterns or composition keywords
# such as anyOf/allOf.  Both SDK transports use this shape, then
# ``parse_agent_feedback`` enforces every strict length, identity, conditional,
# and numeric rule locally before an output can become evidence.
AGENT_FEEDBACK_TRANSPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_AGENT_FEEDBACK_FIELDS),
    "properties": {
        "schema_version": {"type": "string", "const": AGENT_FEEDBACK_SCHEMA_VERSION},
        "feedback_id": {"type": "string"},
        "decision": {"type": "string", "enum": sorted(_FEEDBACK_DECISIONS)},
        "diagnosis": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "falsification_test": {"type": "string"},
        "claim_boundary": {"type": "string"},
        "estimated_cost": {"type": "string", "enum": sorted(_FEEDBACK_COSTS)},
        "reward_spec": {**REWARD_SPEC_TRANSPORT_SCHEMA, "type": ["object", "null"]},
    },
}
