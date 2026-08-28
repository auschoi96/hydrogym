"""Headless, reviewable training artifacts for codex_hydrogym runs."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.training.ppo import RunnerState
from codex_hydrogym.training.validation import PhysicsValidationReport


@dataclass(frozen=True)
class ArtifactPaths:
    directory: Path
    config: Path
    metrics: Path
    validation: Path
    training_curves: Path
    final_vorticity: Path
    manifest: Path

    def files(self) -> tuple[Path, ...]:
        return (
            self.config,
            self.metrics,
            self.validation,
            self.training_curves,
            self.final_vorticity,
            self.manifest,
        )


def summarize_metrics(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Reduce arbitrary metric tensors to JSON-safe scalar summaries."""
    summaries: dict[str, dict[str, Any]] = {}
    for name, raw_value in sorted(metrics.items()):
        value = np.asarray(jax.device_get(raw_value))
        numeric = value.astype(np.float64, copy=False)
        summaries[name] = {
            "shape": list(value.shape),
            "count": int(value.size),
            "nonfinite_count": int(value.size - np.count_nonzero(np.isfinite(numeric))),
            "minimum": float(np.min(numeric)),
            "maximum": float(np.max(numeric)),
            "mean": float(np.mean(numeric)),
            "last": float(numeric.reshape(-1)[-1]),
        }
    return summaries


def _atomic_json(path: Path, value: Any) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_figure(path: Path, figure: plt.Figure) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, dpi=160, bbox_inches="tight")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
        plt.close(figure)


def _mean_by_update(value: Any) -> np.ndarray:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    if array.ndim <= 1:
        return array.reshape(-1)
    return np.mean(array, axis=tuple(range(1, array.ndim)))


def _training_curve_figure(metrics: Mapping[str, Any]) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = (
        (axes[0, 0], ("mean_tke",), "Mean TKE"),
        (
            axes[0, 1],
            ("reward_total", "reward_tke", "reward_action_l1", "reward_action_delta_l2"),
            "Reward decomposition",
        ),
        (axes[1, 0], ("control_l1", "action_delta_l2"), "Control effort"),
        (axes[1, 1], ("loss_total", "loss_actor", "loss_value"), "PPO losses"),
    )
    for axis, names, title in panels:
        for name in names:
            if name in metrics:
                values = _mean_by_update(metrics[name])
                axis.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1.5, label=name)
        axis.set_title(title)
        axis.set_xlabel("PPO update")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    figure.suptitle("codex_hydrogym training diagnostics")
    return figure


def _flow_state(env_state):
    while hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    return env_state


def _vorticity_figure(runner_state: RunnerState) -> plt.Figure:
    omega_hat = np.asarray(jax.device_get(_flow_state(runner_state.env_state).omega_hat))
    if omega_hat.ndim == 2:
        omega_hat = omega_hat[np.newaxis, ...]
    omega = np.fft.irfftn(omega_hat[0], axes=(-2, -1))
    spectrum = np.log10(np.abs(omega_hat[0]) ** 2 + 1.0e-16)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    field = axes[0].imshow(omega.T, origin="lower", cmap="RdBu_r", aspect="equal")
    axes[0].set_title("Final vorticity · environment 0")
    axes[0].set_xlabel("x index")
    axes[0].set_ylabel("y index")
    figure.colorbar(field, ax=axes[0], shrink=0.85)

    power = axes[1].imshow(spectrum.T, origin="lower", cmap="magma", aspect="auto")
    axes[1].set_title("Log₁₀ vorticity spectral power")
    axes[1].set_xlabel("kx index")
    axes[1].set_ylabel("ky index")
    figure.colorbar(power, ax=axes[1], shrink=0.85)
    figure.suptitle("codex_hydrogym final flow state")
    return figure


def write_training_artifacts(
    output_directory: str | Path,
    config: KolmogorovPPOConfig,
    runner_state: RunnerState,
    metrics: Mapping[str, Any],
    validation: PhysicsValidationReport,
    *,
    overwrite: bool = False,
) -> ArtifactPaths:
    """Write labeled JSON evidence and headless PNG diagnostics."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = PROJECT_LABEL
    paths = ArtifactPaths(
        directory=directory,
        config=directory / f"{prefix}_config.json",
        metrics=directory / f"{prefix}_metrics.json",
        validation=directory / f"{prefix}_physics_validation.json",
        training_curves=directory / f"{prefix}_training_curves.png",
        final_vorticity=directory / f"{prefix}_final_vorticity.png",
        manifest=directory / f"{prefix}_artifact_manifest.json",
    )
    existing = [path for path in paths.files() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {existing[0]}")

    _atomic_json(paths.config, {"project_label": PROJECT_LABEL, "config": asdict(config)})
    _atomic_json(paths.metrics, {"project_label": PROJECT_LABEL, "metrics": summarize_metrics(metrics)})
    _atomic_json(paths.validation, validation.as_dict())
    _atomic_figure(paths.training_curves, _training_curve_figure(metrics))
    _atomic_figure(paths.final_vorticity, _vorticity_figure(runner_state))
    _atomic_json(
        paths.manifest,
        {
            "project_label": PROJECT_LABEL,
            "run_name": config.run_name,
            "completed_updates": int(np.asarray(jax.device_get(runner_state.completed_updates))),
            "physics_validation_passed": validation.passed,
            "artifacts": [path.name for path in paths.files() if path != paths.manifest],
        },
    )
    return paths
