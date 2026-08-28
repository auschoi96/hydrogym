"""Physics and numerical validity gates for codex_hydrogym training runs."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import jax
import numpy as np

from codex_hydrogym import PROJECT_LABEL, REWARD_FORMULA_VERSION
from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.training.ppo import RunnerState


class PhysicsValidationError(RuntimeError):
    """One or more required training validity gates failed."""


@dataclass(frozen=True)
class ValidationThresholds:
    reward_identity_atol: float = 1.0e-5
    minimum_tke: float = -1.0e-7
    maximum_action_l1_tolerance: float = 1.0e-5
    maximum_mean_vorticity_ratio: float = 1.0e-5
    maximum_divergence_ratio: float = 1.0e-5
    maximum_spectral_tail_fraction: float = 5.0e-2
    maximum_cfl: float = 1.0


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float
    operator: str
    threshold: float
    detail: str


@dataclass(frozen=True)
class PhysicsValidationReport:
    project_label: str
    passed: bool
    gates: tuple[GateResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_label": self.project_label,
            "passed": bool(self.passed),
            "gates": [
                {
                    **asdict(gate),
                    "passed": bool(gate.passed),
                    "value": float(gate.value),
                    "threshold": float(gate.threshold),
                }
                for gate in self.gates
            ],
        }

    def raise_if_failed(self) -> None:
        if not self.passed:
            failures = ", ".join(gate.name for gate in self.gates if not gate.passed)
            raise PhysicsValidationError(f"codex_hydrogym physics validation failed: {failures}")


def _core_flow_state(env_state):
    while hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    if not hasattr(env_state, "omega_hat"):
        raise TypeError("runner state does not contain a Kolmogorov omega_hat field")
    return env_state


def _nonfinite_count(tree: Any) -> int:
    count = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        value = np.asarray(leaf)
        if np.issubdtype(value.dtype, np.number):
            count += int(np.size(value) - np.count_nonzero(np.isfinite(value)))
    return count


def _spectral_diagnostics(omega_hat: np.ndarray, config: KolmogorovPPOConfig) -> dict[str, float]:
    nx, rfft_ny = omega_hat.shape[-2:]
    ny = 2 * (rfft_ny - 1)
    kx_cycles = np.fft.fftfreq(nx, d=(2.0 * np.pi) / nx)
    ky_cycles = np.fft.rfftfreq(ny, d=(2.0 * np.pi) / ny)
    kx, ky = np.meshgrid(kx_cycles, ky_cycles, indexing="ij")

    laplacian = (2.0j * np.pi) ** 2 * (kx**2 + ky**2)
    laplacian[0, 0] = 1.0
    psi_hat = -omega_hat / laplacian
    u_hat = (2.0j * np.pi) * ky * psi_hat
    v_hat = (-2.0j * np.pi) * kx * psi_hat

    divergence_hat = (2.0j * np.pi) * (kx * u_hat + ky * v_hat)
    gradient_scale = np.sqrt(
        np.sum(np.abs((2.0 * np.pi) * kx * u_hat) ** 2 + np.abs((2.0 * np.pi) * ky * v_hat) ** 2, axis=(-2, -1))
    )
    divergence_scale = np.sqrt(np.sum(np.abs(divergence_hat) ** 2, axis=(-2, -1)))
    divergence_ratio = np.divide(
        divergence_scale,
        gradient_scale,
        out=np.zeros_like(divergence_scale, dtype=np.float64),
        where=gradient_scale > 0.0,
    )

    field_scale = np.sqrt(np.mean(np.abs(omega_hat) ** 2, axis=(-2, -1)))
    zero_mode_ratio = np.divide(
        np.abs(omega_hat[..., 0, 0]),
        field_scale,
        out=np.zeros_like(field_scale, dtype=np.float64),
        where=field_scale > 0.0,
    )

    kx_modes = np.fft.fftfreq(nx) * nx
    ky_modes = np.fft.rfftfreq(ny) * ny
    resolved = (np.abs(kx_modes) <= nx // 3)[:, None] & (np.abs(ky_modes) <= ny // 3)[None, :]
    total_power = np.sum(np.abs(omega_hat) ** 2, axis=(-2, -1))
    tail_power = np.sum(np.abs(omega_hat[..., ~resolved]) ** 2, axis=-1)
    tail_fraction = np.divide(
        tail_power,
        total_power,
        out=np.zeros_like(total_power, dtype=np.float64),
        where=total_power > 0.0,
    )

    u = np.fft.irfftn(u_hat, s=(nx, ny), axes=(-2, -1))
    v = np.fft.irfftn(v_hat, s=(nx, ny), axes=(-2, -1))
    dx = (2.0 * np.pi) / nx
    dy = (2.0 * np.pi) / ny
    cfl = config.dt * (np.max(np.abs(u), axis=(-2, -1)) / dx + np.max(np.abs(v), axis=(-2, -1)) / dy)
    return {
        "maximum_mean_vorticity_ratio": float(np.max(zero_mode_ratio)),
        "maximum_divergence_ratio": float(np.max(divergence_ratio)),
        "maximum_spectral_tail_fraction": float(np.max(tail_fraction)),
        "maximum_cfl": float(np.max(cfl)),
    }


def validate_training_result(
    config: KolmogorovPPOConfig,
    runner_state: RunnerState,
    metrics: Mapping[str, Any],
    *,
    thresholds: ValidationThresholds | None = None,
) -> PhysicsValidationReport:
    """Evaluate hard numerical and physics invariants for a completed chunk."""
    thresholds = thresholds or ValidationThresholds()
    required_metrics = {"mean_tke", "control_l1", "reward_tke", "reward_action_l1", "reward_total"}
    if config.reward_formula_version == REWARD_FORMULA_VERSION:
        required_metrics.add("reward_action_delta_l2")
    missing = sorted(required_metrics - metrics.keys())
    if missing:
        raise ValueError(f"missing metrics required for physics validation: {', '.join(missing)}")

    nonfinite = _nonfinite_count((runner_state, metrics))
    reward_action_delta = np.asarray(metrics.get("reward_action_delta_l2", 0.0))
    reward_residual = np.max(
        np.abs(
            np.asarray(metrics["reward_total"])
            - np.asarray(metrics["reward_tke"])
            - np.asarray(metrics["reward_action_l1"])
            - reward_action_delta
        )
    )
    minimum_tke = float(np.min(np.asarray(metrics["mean_tke"])))
    maximum_action_l1 = float(np.max(np.asarray(metrics["control_l1"])))
    core_state = _core_flow_state(runner_state.env_state)
    action_dim = int(core_state.last_action.shape[-1])
    maximum_allowed_action_l1 = 0.5 * action_dim + thresholds.maximum_action_l1_tolerance
    diagnostics = _spectral_diagnostics(np.asarray(core_state.omega_hat), config)
    completed_updates = int(np.asarray(runner_state.completed_updates))

    gates = (
        GateResult("finite_state_and_metrics", nonfinite == 0, float(nonfinite), "<=", 0.0, "non-finite scalar count"),
        GateResult(
            "reward_decomposition_identity",
            reward_residual <= thresholds.reward_identity_atol,
            float(reward_residual),
            "<=",
            thresholds.reward_identity_atol,
            "max |total - normalized physics - effort - smoothness|",
        ),
        GateResult(
            "nonnegative_tke",
            minimum_tke >= thresholds.minimum_tke,
            minimum_tke,
            ">=",
            thresholds.minimum_tke,
            "minimum trajectory mean TKE",
        ),
        GateResult(
            "bounded_control_effort",
            maximum_action_l1 <= maximum_allowed_action_l1,
            maximum_action_l1,
            "<=",
            maximum_allowed_action_l1,
            "maximum L1 effort after environment clipping",
        ),
        GateResult(
            "zero_mean_vorticity",
            diagnostics["maximum_mean_vorticity_ratio"] <= thresholds.maximum_mean_vorticity_ratio,
            diagnostics["maximum_mean_vorticity_ratio"],
            "<=",
            thresholds.maximum_mean_vorticity_ratio,
            "maximum normalized zero Fourier mode",
        ),
        GateResult(
            "incompressible_velocity",
            diagnostics["maximum_divergence_ratio"] <= thresholds.maximum_divergence_ratio,
            diagnostics["maximum_divergence_ratio"],
            "<=",
            thresholds.maximum_divergence_ratio,
            "maximum spectral divergence ratio",
        ),
        GateResult(
            "spectral_tail_controlled",
            diagnostics["maximum_spectral_tail_fraction"] <= thresholds.maximum_spectral_tail_fraction,
            diagnostics["maximum_spectral_tail_fraction"],
            "<=",
            thresholds.maximum_spectral_tail_fraction,
            "maximum vorticity power outside the two-thirds band",
        ),
        GateResult(
            "cfl_controlled",
            diagnostics["maximum_cfl"] <= thresholds.maximum_cfl,
            diagnostics["maximum_cfl"],
            "<=",
            thresholds.maximum_cfl,
            "maximum final-state advective CFL estimate",
        ),
        GateResult(
            "update_count_valid",
            0 <= completed_updates <= config.num_updates,
            float(completed_updates),
            "<=",
            float(config.num_updates),
            "completed PPO updates within configured budget",
        ),
    )
    return PhysicsValidationReport(PROJECT_LABEL, all(gate.passed for gate in gates), gates)
