"""Deterministic compilation and execution of human-approved reward specs.

The language model may select only two bounded coefficients.  Calibration,
approval, canonical hashing, and per-step arithmetic remain deterministic code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re
from typing import Any, Mapping

import jax.numpy as jnp

from codex_hydrogym import LEGACY_REWARD_FORMULA_VERSION, REWARD_FORMULA_VERSION
from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.genai.contracts import RewardSpec
from hydrogym.jax.env_core import GymnaxWrapper


COMPILED_REWARD_SCHEMA_VERSION = "codex_hydrogym.compiled_reward.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPILED_FIELDS = {
    "schema_version",
    "formula_version",
    "reference_tke",
    "control_l1_weight",
    "action_delta_l2_weight",
    "evidence_digest",
    "reward_spec_digest",
    "approval_digest",
    "compiled_digest",
}
_CONFIG_REWARD_FIELDS = {
    "reward_alpha",
    "reward_formula_version",
    "reward_reference_tke",
    "reward_control_l1_weight",
    "reward_action_delta_l2_weight",
    "reward_spec_digest",
    "reward_evidence_digest",
    "reward_approval_digest",
    "reward_compiled_digest",
}


def _require_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_finite(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or not minimum <= converted <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return converted


@dataclass(frozen=True)
class CompiledReward:
    """Frozen executable reward manifest, including provenance and approval."""

    reference_tke: float
    control_l1_weight: float
    action_delta_l2_weight: float
    evidence_digest: str
    reward_spec_digest: str
    approval_digest: str
    formula_version: str = REWARD_FORMULA_VERSION
    schema_version: str = COMPILED_REWARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COMPILED_REWARD_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {COMPILED_REWARD_SCHEMA_VERSION}")
        if self.formula_version != REWARD_FORMULA_VERSION:
            raise ValueError(f"formula_version must be {REWARD_FORMULA_VERSION}")
        object.__setattr__(
            self,
            "reference_tke",
            _require_finite(self.reference_tke, name="reference_tke", minimum=1.0e-12, maximum=1.0e12),
        )
        object.__setattr__(
            self,
            "control_l1_weight",
            _require_finite(self.control_l1_weight, name="control_l1_weight", minimum=0.05, maximum=1.0),
        )
        object.__setattr__(
            self,
            "action_delta_l2_weight",
            _require_finite(
                self.action_delta_l2_weight,
                name="action_delta_l2_weight",
                minimum=0.0,
                maximum=0.25,
            ),
        )
        for name in ("evidence_digest", "reward_spec_digest", "approval_digest"):
            object.__setattr__(self, name, _require_digest(getattr(self, name), name=name))

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def canonical_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "compiled_digest": self.canonical_digest()}

    def apply(self, baseline: KolmogorovPPOConfig) -> KolmogorovPPOConfig:
        """Change only reward fields, preserving the full PPO/scientific context."""
        before = frozen_training_fingerprint(baseline)
        compiled = replace(
            baseline,
            reward_alpha=1.0,
            reward_formula_version=self.formula_version,
            reward_reference_tke=self.reference_tke,
            reward_control_l1_weight=self.control_l1_weight,
            reward_action_delta_l2_weight=self.action_delta_l2_weight,
            reward_spec_digest=self.reward_spec_digest,
            reward_evidence_digest=self.evidence_digest,
            reward_approval_digest=self.approval_digest,
            reward_compiled_digest=self.canonical_digest(),
        )
        if frozen_training_fingerprint(compiled) != before:
            raise AssertionError("reward compilation changed frozen training fields")
        return compiled


def compile_reward_spec(
    spec: RewardSpec,
    *,
    reference_tke: float,
    calibration_evidence_digest: str,
    approval_digest: str,
) -> CompiledReward:
    """Bind a model proposal to development calibration and human approval."""
    if not isinstance(spec, RewardSpec):
        raise TypeError("spec must be a reward-only RewardSpec")
    calibration_digest = _require_digest(calibration_evidence_digest, name="calibration_evidence_digest")
    if spec.evidence_digest != calibration_digest:
        raise ValueError("RewardSpec evidence_digest does not match calibration evidence")
    return CompiledReward(
        reference_tke=reference_tke,
        control_l1_weight=spec.control_l1_weight,
        action_delta_l2_weight=spec.action_delta_l2_weight,
        evidence_digest=calibration_digest,
        reward_spec_digest=spec.canonical_digest(),
        approval_digest=_require_digest(approval_digest, name="approval_digest"),
    )


def parse_compiled_reward(value: Mapping[str, Any]) -> CompiledReward:
    """Strictly parse and verify a serialized compiled manifest."""
    if not isinstance(value, Mapping) or set(value) != _COMPILED_FIELDS:
        raise ValueError("compiled reward fields do not match the v1 contract")
    raw = dict(value)
    supplied_digest = _require_digest(raw.pop("compiled_digest"), name="compiled_digest")
    compiled = CompiledReward(**raw)
    if compiled.canonical_digest() != supplied_digest:
        raise ValueError("compiled reward digest does not match its canonical payload")
    return compiled


def frozen_training_fingerprint(config: KolmogorovPPOConfig) -> str:
    """Hash all config fields except run labeling and reward choice."""
    payload = asdict(config)
    payload.pop("run_name", None)
    for name in _CONFIG_REWARD_FIELDS:
        payload.pop(name, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compiled_reward_from_config(config: KolmogorovPPOConfig) -> CompiledReward | None:
    """Recover and verify the manifest embedded in a validated PPO config."""
    if config.reward_formula_version == LEGACY_REWARD_FORMULA_VERSION:
        return None
    if config.reward_formula_version != REWARD_FORMULA_VERSION:
        raise ValueError("unsupported reward_formula_version")
    compiled = CompiledReward(
        reference_tke=config.reward_reference_tke,
        control_l1_weight=config.reward_control_l1_weight,
        action_delta_l2_weight=config.reward_action_delta_l2_weight,
        evidence_digest=config.reward_evidence_digest,
        reward_spec_digest=config.reward_spec_digest,
        approval_digest=config.reward_approval_digest,
    )
    if compiled.canonical_digest() != config.reward_compiled_digest:
        raise ValueError("PPO config reward fields do not match reward_compiled_digest")
    return compiled


def reward_terms(
    *,
    mean_tke,
    control_l1,
    action_delta_l2,
    compiled: CompiledReward,
) -> dict[str, Any]:
    """Compute the approved reward decomposition with JAX-compatible arithmetic."""
    tke = -jnp.asarray(mean_tke) / compiled.reference_tke
    action_l1 = -compiled.control_l1_weight * jnp.asarray(control_l1) / 2.0
    action_delta = -compiled.action_delta_l2_weight * jnp.asarray(action_delta_l2) / 4.0
    return {
        "reward_tke": tke,
        "reward_action_l1": action_l1,
        "reward_action_delta_l2": action_delta,
        "reward_total": tke + action_l1 + action_delta,
    }


class DeterministicRewardWrapper(GymnaxWrapper):
    """Replace the legacy environment reward using only measured info fields."""

    def __init__(self, env, compiled: CompiledReward):
        super().__init__(env)
        if not isinstance(compiled, CompiledReward):
            raise TypeError("compiled must be a CompiledReward")
        self.compiled = compiled

    def step(self, key, state, action, params=None):
        observation, next_state, _legacy_reward, done, info = self._env.step(key, state, action, params)
        required = {"mean_tke", "control_l1", "action_delta_l2"}
        missing = required - info.keys()
        if missing:
            raise ValueError(f"environment info lacks deterministic reward fields: {sorted(missing)}")
        terms = reward_terms(
            mean_tke=info["mean_tke"],
            control_l1=info["control_l1"],
            action_delta_l2=info["action_delta_l2"],
            compiled=self.compiled,
        )
        updated_info = dict(info)
        updated_info.update(terms)
        return observation, next_state, terms["reward_total"], done, updated_info
