"""Validated experiment configuration for codex_hydrogym."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any

from codex_hydrogym import LEGACY_REWARD_FORMULA_VERSION, PROJECT_LABEL, REWARD_FORMULA_VERSION
from hydrogym.jax.kolmogorov_contract import (
    FORCED_MODE_QUADRATURE_BASIS_VERSION,
    LEGACY_SPEED_GRID_OBSERVATION_MODE,
    OBSERVATION_CONTRACT_VERSIONS,
)


def _integer_ratio(duration: float, interval: float, label: str) -> int:
    if duration <= 0.0 or interval <= 0.0:
        raise ValueError(f"{label} values must be positive")
    ratio = duration / interval
    rounded = int(round(ratio))
    if not math.isclose(ratio, rounded, rel_tol=1.0e-10, abs_tol=1.0e-9):
        raise ValueError(f"{duration} must be an integer multiple of {interval} for {label}")
    return rounded


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class KolmogorovPPOConfig:
    """One fully reproducible Kolmogorov PPO workload configuration."""

    project_label: str = field(init=False, default=PROJECT_LABEL)
    run_name: str = "ppo"
    seed: int = 0
    precision: str = "float32"

    reynolds_number: float = 200.0
    forcing_wavenumber: int = 4
    forcing_phase: float = 0.0
    actuation_basis_version: str = FORCED_MODE_QUADRATURE_BASIS_VERSION
    observation_mode: str = LEGACY_SPEED_GRID_OBSERVATION_MODE
    observation_contract_version: str | None = None
    grid_size: tuple[int, int] = (64, 64)
    obs_size: int = 8
    dt: float = 1.0e-3
    action_time: float = 10.0
    save_time: float = 1.0
    reward_alpha: float = 1.0
    reward_formula_version: str = LEGACY_REWARD_FORMULA_VERSION
    reward_reference_tke: float | None = None
    reward_control_l1_weight: float | None = None
    reward_action_delta_l2_weight: float | None = None
    reward_spec_digest: str | None = None
    reward_evidence_digest: str | None = None
    reward_approval_digest: str | None = None
    reward_compiled_digest: str | None = None
    initial_perturbation_amplitude: float = 1.0e-3
    max_episode_steps: int = 1_000

    learning_rate: float = 1.0e-4
    num_envs: int = 8
    num_steps: int = 16
    total_timesteps: int = 4_096
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.985
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.0
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    activation: str = "tanh"
    anneal_learning_rate: bool = False
    normalize_environment: bool = False

    allow_zero_physics_reward: bool = False
    allow_deterministic_resets: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid_size", tuple(self.grid_size))
        expected_observation_version = OBSERVATION_CONTRACT_VERSIONS.get(self.observation_mode)
        if expected_observation_version is None:
            raise ValueError(f"observation_mode must be one of {sorted(OBSERVATION_CONTRACT_VERSIONS)}")
        if self.observation_contract_version is None:
            object.__setattr__(self, "observation_contract_version", expected_observation_version)
        elif self.observation_contract_version != expected_observation_version:
            raise ValueError("observation_contract_version does not match observation_mode")
        if not self.run_name.startswith(PROJECT_LABEL):
            object.__setattr__(self, "run_name", f"{PROJECT_LABEL}_{self.run_name}")

        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be 'float32' or 'float64'")
        if len(self.grid_size) != 2 or any(size <= 0 for size in self.grid_size):
            raise ValueError("grid_size must contain two positive dimensions")
        if any(size % 2 for size in self.grid_size):
            raise ValueError("grid_size dimensions must be even for the current rFFT solver")
        if self.obs_size <= 0 or any(size % self.obs_size for size in self.grid_size):
            raise ValueError("obs_size must divide both grid dimensions")

        nyquist_y = self.grid_size[1] // 2
        highest_actuator_mode = 7
        if highest_actuator_mode >= nyquist_y:
            raise ValueError("the y grid must resolve every actuator mode below Nyquist")
        if self.forcing_wavenumber <= 0 or self.forcing_wavenumber >= nyquist_y:
            raise ValueError("forcing_wavenumber must be positive and below Nyquist")
        if not math.isfinite(self.forcing_phase):
            raise ValueError("forcing_phase must be finite")
        if self.actuation_basis_version != FORCED_MODE_QUADRATURE_BASIS_VERSION:
            raise ValueError(
                "actuation_basis_version must identify the supported forced-mode quadrature action basis"
            )
        if self.reynolds_number <= 0.0:
            raise ValueError("reynolds_number must be positive")

        if self.reward_alpha == 0.0 and not self.allow_zero_physics_reward:
            raise ValueError("reward_alpha=0 requires allow_zero_physics_reward=True")
        if self.reward_alpha < 0.0:
            raise ValueError("reward_alpha must be positive for turbulence suppression")
        compiled_reward_fields = (
            self.reward_reference_tke,
            self.reward_control_l1_weight,
            self.reward_action_delta_l2_weight,
            self.reward_spec_digest,
            self.reward_evidence_digest,
            self.reward_approval_digest,
            self.reward_compiled_digest,
        )
        if self.reward_formula_version == LEGACY_REWARD_FORMULA_VERSION:
            if any(value is not None for value in compiled_reward_fields):
                raise ValueError("legacy reward configuration cannot contain compiled reward fields")
        elif self.reward_formula_version == REWARD_FORMULA_VERSION:
            if any(value is None for value in compiled_reward_fields):
                raise ValueError("normalized reward configuration requires every compiled reward field")
            if self.reward_alpha != 1.0:
                raise ValueError("normalized reward configuration requires inert legacy reward_alpha=1")
            if not math.isfinite(self.reward_reference_tke) or self.reward_reference_tke <= 0.0:
                raise ValueError("reward_reference_tke must be finite and positive")
            if not 0.05 <= self.reward_control_l1_weight <= 1.0:
                raise ValueError("reward_control_l1_weight must be in [0.05, 1]")
            if not 0.0 <= self.reward_action_delta_l2_weight <= 0.25:
                raise ValueError("reward_action_delta_l2_weight must be in [0, 0.25]")
            for name in (
                "reward_spec_digest",
                "reward_evidence_digest",
                "reward_approval_digest",
                "reward_compiled_digest",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        else:
            raise ValueError("reward_formula_version is not supported")
        if self.num_envs > 1 and self.initial_perturbation_amplitude <= 0.0 and not self.allow_deterministic_resets:
            raise ValueError(
                "initial_perturbation_amplitude must be positive for vectorized training; "
                "set allow_deterministic_resets=True only for an explicit deterministic baseline"
            )

        if self.num_envs <= 0 or self.num_steps <= 0 or self.total_timesteps <= 0:
            raise ValueError("num_envs, num_steps, and total_timesteps must be positive")
        if self.num_minibatches <= 0 or self.update_epochs <= 0:
            raise ValueError("num_minibatches and update_epochs must be positive")
        if self.total_batch_size % self.num_minibatches:
            raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
        if self.total_timesteps % self.total_batch_size:
            raise ValueError("total_timesteps must be a multiple of num_envs * num_steps")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.activation not in {"tanh", "relu"}:
            raise ValueError("activation must be 'tanh' or 'relu'")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")

        if self.action_steps % self.save_steps:
            raise ValueError("action_time must contain an integer multiple of save_time")

    @property
    def total_batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.total_batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.total_batch_size

    @property
    def save_steps(self) -> int:
        return _integer_ratio(self.save_time, self.dt, "save_time/dt")

    @property
    def action_steps(self) -> int:
        return _integer_ratio(self.action_time, self.dt, "action_time/dt")

    def environment_config(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "action_time": self.action_time,
            "save_time": self.save_time,
            "reward_alpha": self.reward_alpha,
            "initial_perturbation_amplitude": self.initial_perturbation_amplitude,
            "max_episode_steps": self.max_episode_steps,
            "observation_mode": self.observation_mode,
        }

    def flow_config(self) -> dict[str, Any]:
        return {
            "Re": self.reynolds_number,
            "k": self.forcing_wavenumber,
            "forcing_phase": self.forcing_phase,
            "grid_size": self.grid_size,
            "obs_size": self.obs_size,
        }

    def as_mlflow_params(self) -> dict[str, str | int | float | bool]:
        params = {key: value for key, value in asdict(self).items() if value is not None}
        params["grid_size"] = f"{self.grid_size[0]}x{self.grid_size[1]}"
        params["total_batch_size"] = self.total_batch_size
        params["minibatch_size"] = self.minibatch_size
        params["num_updates"] = self.num_updates
        params["save_steps"] = self.save_steps
        params["action_steps"] = self.action_steps
        return params


def config_fingerprint(config: KolmogorovPPOConfig) -> str:
    """Return a stable SHA-256 fingerprint of every validated config field."""
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
